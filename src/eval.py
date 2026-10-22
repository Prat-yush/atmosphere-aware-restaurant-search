"""Evaluation: pooling for hand grading, metrics, popularity bias, NLI accuracy, dev tuning.

Workflow:
    python -m src.eval pool                 # -> data/eval/pool.csv: union of top-10 from all systems, blind
    (grade data/eval/pool.csv: fill `grade` with 0, 1 or 2)
    python -m src.eval import-grades        # data/eval/pool.csv -> queries/qrels.tsv
    python -m src.eval metrics              # nDCG@10, P@5 overall and per category; popularity check
    python -m src.eval polarity-sample      # -> data/eval/polarity_sample.csv (100 sentences to label)
    (label it: `label` = pos, neg or none)
    python -m src.eval polarity-accuracy    # NLI accuracy against your labels
    python -m src.eval tune                 # grid over k, m, mu, top_n on dev queries only
                                            # (also judged@10 and judged-only nDCG: settings far
                                            #  from the pooled ones surface unjudged restaurants)

Queries (written by hand): queries/{test,dev}.jsonl with fields qid, text,
category in {single, mixed, negation, contrast}. Qrels: queries/qrels.tsv with header
qid, business_id, grade (0, 1, 2). Unjudged restaurants count as grade 0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

CATEGORIES = ("single", "mixed", "negation", "contrast")
SYSTEMS = ("bm25_baseline", "no_polarity", "full")
POOL_DEPTH = 10
POLARITY_SAMPLE_SIZE = 100
TUNE_GRID = {"k": (5, 10, 20), "m": (0, 2, 4, 8), "mu": (None, 0.0), "top_n": (1000, 3000)}  # mu None = per-target mean raw


# --- Metrics (pure) -------------------------------------------------------------

def dcg(gains: Sequence[float]) -> float:
    """Linear gain, log2(rank + 1) discount, rank from 1 (trec_eval convention)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], grades: dict[str, int], k: int = 10) -> float:
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([grades.get(b, 0) for b in ranked[:k]]) / ideal


def judged_only(ranked: Sequence[str], grades: dict[str, int]) -> list[str]:
    """Condensed list (Sakai 2007): drop unjudged results so they are not silently counted as grade 0."""
    return [b for b in ranked if b in grades]


def judged_fraction(ranked: Sequence[str], grades: dict[str, int], k: int = 10) -> float:
    top = list(ranked[:k])
    return sum(b in grades for b in top) / len(top) if top else 0.0


def precision_at_k(ranked: Sequence[str], grades: dict[str, int], k: int = 5, min_grade: int = 1) -> float:
    """Relevant (grade >= min_grade) in the top k, divided by k; missing ranks count as non-relevant."""
    return sum(grades.get(b, 0) >= min_grade for b in ranked[:k]) / k


def spearman_rank_vs_popularity(ranked: Sequence[str], review_count: dict[str, int]) -> float:
    """Spearman rho between rank position (1 = best) and review_count. Negative = popular ranked higher."""
    from scipy.stats import spearmanr

    if len(ranked) < 3:
        return float("nan")
    return float(spearmanr(np.arange(1, len(ranked) + 1), [review_count[b] for b in ranked]).statistic)


# --- Query and qrels I/O -----------------------------------------------------------

def load_queries(path: Path) -> list[dict]:
    queries = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    for q in queries:
        missing = {"qid", "text", "category"} - set(q)
        if missing:
            raise ValueError(f"{path}: query {q} missing {missing}")
        if q["category"] not in CATEGORIES:
            raise ValueError(f"{path}: qid {q['qid']} category {q['category']!r} not in {CATEGORIES}")
    if len({q["qid"] for q in queries}) != len(queries):
        raise ValueError(f"{path}: duplicate qid")
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    df = pd.read_csv(path, sep="\t", dtype={"qid": str, "business_id": str, "grade": int})
    if not df["grade"].isin([0, 1, 2]).all():
        raise ValueError(f"{path}: grades must be 0, 1 or 2")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in df.itertuples():
        qrels[row.qid][row.business_id] = row.grade
    return dict(qrels)


# --- Running systems ------------------------------------------------------------------

def run_system(ranker, queries: list[dict], system: str, depth: int) -> dict[str, list[str]]:
    return {q["qid"]: [r.business_id for r in ranker.rank(q["text"], system, depth)[1]] for q in queries}


def run_all(ranker, queries: list[dict], depth: int) -> dict[str, dict[str, list[str]]]:
    return {s: run_system(ranker, queries, s, depth) for s in SYSTEMS}


# --- Pooling ---------------------------------------------------------------------------

def build_pool(runs: dict[str, dict[str, list[str]]], queries: list[dict], businesses: pd.DataFrame, depth: int = POOL_DEPTH) -> pd.DataFrame:
    """Blind pool: no system names or ranks; rows sorted by query then restaurant name."""
    rows = []
    for q in queries:
        pooled = sorted({b for run in runs.values() for b in run[q["qid"]][:depth]})
        for bid in pooled:
            b = businesses.loc[bid]
            rows.append({
                "qid": q["qid"], "query": q["text"], "business_id": bid, "name": b["name"],
                "address": b["address"], "postal_code": b["postal_code"],
                "price": "" if pd.isna(b["price"]) else "$" * int(b["price"]),
                "categories": b["categories"], "grade": "",
            })
    return pd.DataFrame(rows).sort_values(["qid", "name", "business_id"], kind="stable").reset_index(drop=True)


def grades_to_qrels(pool: pd.DataFrame) -> pd.DataFrame:
    graded = pool[pool["grade"].astype(str).str.strip() != ""].copy()
    graded["grade"] = graded["grade"].astype(float).astype(int)
    if not graded["grade"].isin([0, 1, 2]).all():
        raise ValueError("grades must be 0, 1 or 2")
    missing = len(pool) - len(graded)
    if missing:
        print(f"warning: {missing} pooled rows have no grade and count as 0")
    return graded[["qid", "business_id", "grade"]]


# --- Metric report ------------------------------------------------------------------------

def evaluate(runs: dict[str, dict[str, list[str]]], queries: list[dict], qrels: dict[str, dict[str, int]]) -> pd.DataFrame:
    """One row per (system, qid) with ndcg@10 and p@5."""
    rows = []
    for system, run in runs.items():
        for q in queries:
            grades = qrels.get(q["qid"], {})
            rows.append({
                "system": system, "qid": q["qid"], "category": q["category"],
                "ndcg@10": ndcg_at_k(run[q["qid"]], grades, 10),
                "p@5": precision_at_k(run[q["qid"]], grades, 5),
            })
    return pd.DataFrame(rows)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per system, overall ("all") and per category, with query counts."""
    agg = {"ndcg@10": ("ndcg@10", "mean"), "p@5": ("p@5", "mean"), "n": ("qid", "size")}
    overall = per_query.groupby("system").agg(**agg).reset_index().assign(category="all")
    by_cat = per_query.groupby(["system", "category"]).agg(**agg).reset_index()
    out = pd.concat([overall, by_cat], ignore_index=True)
    out["_sys"] = out["system"].map({s: i for i, s in enumerate(SYSTEMS)})
    out["_cat"] = out["category"].map({c: i for i, c in enumerate(("all",) + CATEGORIES)})
    return out.sort_values(["_cat", "_sys"]).drop(columns=["_sys", "_cat"])[["category", "system", "ndcg@10", "p@5", "n"]].reset_index(drop=True)


def popularity_report(runs: dict[str, dict[str, list[str]]], review_count: dict[str, int], depth: int) -> pd.DataFrame:
    rows = []
    for system, run in runs.items():
        rhos = [spearman_rank_vs_popularity(r[:depth], review_count) for r in run.values()]
        rhos = [x for x in rhos if not math.isnan(x)]
        rows.append({"system": system, "mean_spearman": float(np.mean(rhos)) if rhos else float("nan"), "queries": len(rhos)})
    return pd.DataFrame(rows)


# --- Polarity component check -----------------------------------------------------------

def sample_polarity(sentences: pd.DataFrame, lexicon: dict, n: int = POLARITY_SAMPLE_SIZE, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified by aspect (as equal as possible); one row per (sentence, aspect) to label."""
    aspects = list(lexicon["aspects"])
    per = [n // len(aspects) + (i < n % len(aspects)) for i in range(len(aspects))]
    rng = np.random.default_rng(seed)
    rows = []
    for aspect, k in zip(aspects, per):
        pool = sentences[sentences["matched_aspects"].map(lambda a, asp=aspect: asp in a)]
        pick = rng.choice(len(pool), size=k, replace=False)
        for i in sorted(pick):
            s = pool.iloc[i]
            poles = lexicon["aspects"][aspect]
            rows.append({
                "sample_id": len(rows) + 1, "sent_id": s["sent_id"], "aspect": aspect,
                "pos_pole": poles["positive"]["id"], "neg_pole": poles["negative"]["id"],
                "text": s["text"], "label": "",
            })
    return pd.DataFrame(rows)


def predict_label(p_pos: float, p_neg: float) -> str:
    """NLI's call for an aspect: the pole with the higher positive polarity, else none."""
    if max(p_pos, p_neg) <= 0:
        return "none"
    return "pos" if p_pos >= p_neg else "neg"


def polarity_accuracy(labeled: pd.DataFrame, polarity: pd.DataFrame) -> dict:
    lab = labeled[labeled["label"].astype(str).str.strip() != ""].copy()
    lab["label"] = lab["label"].str.strip().str.lower()
    if not lab["label"].isin(["pos", "neg", "none"]).all():
        raise ValueError("labels must be pos, neg or none")
    pol = polarity.set_index("sent_id")
    lab["pred"] = [
        predict_label(pol.at[r.sent_id, f"{r.aspect}.{r.pos_pole}"], pol.at[r.sent_id, f"{r.aspect}.{r.neg_pole}"])
        for r in lab.itertuples()
    ]
    polar = lab[lab["label"] != "none"]
    return {
        "n_labeled": len(lab),
        "accuracy_3way": float((lab["pred"] == lab["label"]).mean()),
        "n_polar": len(polar),
        "accuracy_on_pos_neg": float((polar["pred"] == polar["label"]).mean()) if len(polar) else float("nan"),
        "confusion": pd.crosstab(lab["label"], lab["pred"], rownames=["human"], colnames=["nli"]),
        "per_aspect": lab.assign(ok=lab["pred"] == lab["label"]).groupby("aspect")["ok"].mean(),
        "rows": lab,
    }


# --- Commands ----------------------------------------------------------------------------

def cmd_pool(args) -> None:
    from src.rank import Ranker

    queries = load_queries(config.TEST_QUERIES_PATH) + load_queries(config.DEV_QUERIES_PATH)
    ranker = Ranker()
    pool = build_pool(run_all(ranker, queries, POOL_DEPTH), queries, ranker.businesses)
    out = config.EVAL_DIR / "pool.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; it may hold your grades. Use --force to overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False)
    print(f"wrote {out}: {len(pool)} rows to grade, {len(queries)} queries "
          f"({len(pool) / len(queries):.1f} restaurants per query)")


def cmd_import_grades(args) -> None:
    qrels = grades_to_qrels(pd.read_csv(config.EVAL_DIR / "pool.csv", dtype=str, keep_default_na=False))
    qrels.to_csv(config.QRELS_PATH, sep="\t", index=False)
    print(f"wrote {config.QRELS_PATH}: {len(qrels)} judgments, grade counts {qrels['grade'].value_counts().sort_index().to_dict()}")


def cmd_metrics(args) -> None:
    from src.rank import Ranker

    queries = load_queries(config.TEST_QUERIES_PATH)
    qrels = load_qrels(config.QRELS_PATH)
    ranker = Ranker(k=args.k, m=args.m, mu=args.mu)
    runs = run_all(ranker, queries, POOL_DEPTH)
    per_query = evaluate(runs, queries, qrels)
    summary = summarize(per_query)
    review_count = ranker.businesses["review_count"].to_dict()
    pop = popularity_report(runs, review_count, POOL_DEPTH)
    print(summary.to_string(index=False, float_format="%.3f"))
    print("\npopularity bias (Spearman of rank vs review_count in top 10; negative = popular ranked higher)")
    print(pop.to_string(index=False, float_format="%.3f"))
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(config.EVAL_DIR / "metrics_per_query.csv", index=False)
    summary.to_csv(config.EVAL_DIR / "metrics_summary.csv", index=False)
    pop.to_csv(config.EVAL_DIR / "popularity.csv", index=False)


def cmd_polarity_sample(args) -> None:
    from src.prep import load_lexicon

    out = config.EVAL_DIR / "polarity_sample.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; it may hold your labels. Use --force to overwrite.")
    sample = sample_polarity(pd.read_parquet(config.SENTENCES_PATH), load_lexicon())
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"wrote {out}: {len(sample)} sentences; fill `label` with pos, neg or none")


def cmd_polarity_accuracy(args) -> None:
    res = polarity_accuracy(
        pd.read_csv(config.EVAL_DIR / "polarity_sample.csv", dtype=str, keep_default_na=False),
        pd.read_parquet(config.POLARITY_PATH),
    )
    print(f"labeled {res['n_labeled']}: 3-way accuracy {res['accuracy_3way']:.3f}; "
          f"on pos/neg only ({res['n_polar']}): {res['accuracy_on_pos_neg']:.3f}")
    print("\n", res["confusion"], "\n\nper aspect:\n", res["per_aspect"].round(3).to_string())
    res["rows"].to_csv(config.EVAL_DIR / "polarity_predictions.csv", index=False)


def cmd_tune(args) -> None:
    from src.rank import Ranker

    queries = load_queries(config.DEV_QUERIES_PATH)
    qrels = load_qrels(config.QRELS_PATH)
    ranker = Ranker(load_baseline=False)
    rows = []
    for k, m, mu, top_n in itertools.product(*TUNE_GRID.values()):
        ranker.k, ranker.m, ranker.mu, ranker.top_n = k, m, mu, top_n
        run = run_system(ranker, queries, "full", 50)  # deeper than 10 so the condensed list can refill
        res = evaluate({"full": run}, queries, qrels)
        g = [qrels.get(q["qid"], {}) for q in queries]
        rows.append({
            "k": k, "m": m, "mu": "auto" if mu is None else f"{mu:g}", "top_n": top_n,
            "ndcg@10": res["ndcg@10"].mean(), "p@5": res["p@5"].mean(),
            "judged@10": np.mean([judged_fraction(run[q["qid"]], gr) for q, gr in zip(queries, g)]),
            "ndcg@10_judged": np.mean([ndcg_at_k(judged_only(run[q["qid"]], gr), gr) for q, gr in zip(queries, g)]),
        })
    table = pd.DataFrame(rows).sort_values(["ndcg@10", "p@5"], ascending=False)
    print(table.to_string(index=False, float_format="%.3f"))
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.EVAL_DIR / "tuning_dev.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "polarity-sample"):
        sub.add_parser(name).add_argument("--force", action="store_true", help="overwrite an existing CSV")
    sub.add_parser("import-grades")
    sub.add_parser("polarity-accuracy")
    sub.add_parser("tune")
    p = sub.add_parser("metrics")
    p.add_argument("--k", type=int, default=config.K_PER_RESTAURANT)
    p.add_argument("--m", type=float, default=config.M_SHRINK)
    p.add_argument("--mu", type=float, default=None, help="fixed mu; default = per-target mean raw")
    args = parser.parse_args()
    {
        "pool": cmd_pool, "import-grades": cmd_import_grades, "metrics": cmd_metrics,
        "polarity-sample": cmd_polarity_sample, "polarity-accuracy": cmd_polarity_accuracy, "tune": cmd_tune,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
