"""Support for LLM-assisted relevance grading, validated against human grades.

Grading evidence per pooled restaurant comes from sources the ranking systems do not rank by:
  - Yelp business attributes voted by users (NoiseLevel, Ambience, Music, ...)
  - a seeded RANDOM sample of review sentences per atmosphere topic, matched with a broad
    keyword list that is independent of lexicon.yaml (not the systems' retrieved evidence)
The grader follows queries/grading_rubric.md. A human grades a held-out subset blind
(data/eval/human_check.csv); agreement is reported as exact agreement and Cohen's kappa.

Usage:
    python -m src.judge evidence     # -> data/eval/grading_evidence.parquet, data/eval/attributes.json
    python -m src.judge human-check  # -> data/eval/human_check.csv (blind; fill `grade`)
    python -m src.judge agreement    # human_check.csv vs grades in pool.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config
from src.prep import iter_target_reviews

SENTENCES_PER_TOPIC = 4
HUMAN_CHECK_QIDS = ("t09", "t17", "t24")  # one mixed, one negation, one contrast

# Broad topic keywords for sampling grading evidence; deliberately not lexicon.yaml.
TOPICS = {
    "noise": r"quiet|loud|noise|noisy|hear|conversation|talk|shout|yell|peaceful|calm",
    "energy": r"lively|energy|energetic|vibe|buzz\w*|busy|packed|dead|empty|boring|fun|atmosphere|crowd|scene|upbeat|vibrant",
    "space": r"space|spacious|cozy|cosy|cramped|tight|small|roomy|seating|crowded|tables|airy",
    "romance": r"romantic|date|intimate|anniversary|couple|candle\w*|special occasion|sports bar|tvs?",
    "lighting": r"light\w*|lit|dim\w*|dark|bright|candle\w*|windows?|sun\w*",
    "music": r"music|playlist|dj|band|jazz|live|songs?|speakers?|volume",
    "formality": r"stuffy|pretentious|formal|upscale|casual|laid.back|relaxed|dress\w*|snooty",
}
TOPIC_RE = {t: re.compile(rf"\b(?:{p})\b", re.IGNORECASE) for t, p in TOPICS.items()}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
ATTRIBUTE_KEYS = ("NoiseLevel", "Ambience", "Music", "RestaurantsGoodForGroups", "GoodForKids", "RestaurantsAttire", "HasTV")


# --- Evidence ------------------------------------------------------------------------

def parse_attribute(value):
    """Yelp stores attributes as Python-literal strings ("u'quiet'", "{'romantic': True}")."""
    if not isinstance(value, str):
        return value
    try:
        out = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return {k: v for k, v in out.items() if v is not None} if isinstance(out, dict) else out


def load_attributes(business_ids: set[str]) -> dict[str, dict]:
    attrs = {}
    with open(config.RAW_BUSINESS_PATH, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b["business_id"] in business_ids:
                raw = b.get("attributes") or {}
                attrs[b["business_id"]] = {k: parse_attribute(raw[k]) for k in ATTRIBUTE_KEYS if raw.get(k) not in (None, "None")}
    return attrs


def topic_sentences(text: str) -> dict[str, list[str]]:
    out = defaultdict(list)
    for s in SENT_SPLIT.split(text):
        s = " ".join(s.split())
        if 20 <= len(s) <= 300:
            for topic, rx in TOPIC_RE.items():
                if rx.search(s):
                    out[topic].append(s)
    return out


def sample_evidence(business_ids: set[str], n: int = SENTENCES_PER_TOPIC, seed: int = config.SEED) -> pd.DataFrame:
    """Seeded random sample of up to n sentences per (restaurant, topic) from all its reviews."""
    found: dict[tuple[str, str], list[str]] = defaultdict(list)
    for review in iter_target_reviews(config.RAW_REVIEW_PATH, business_ids):
        for topic, sents in topic_sentences(review.get("text") or "").items():
            found[(review["business_id"], topic)].extend(sents)
    rng = np.random.default_rng(seed)
    rows = []
    for (bid, topic), sents in sorted(found.items()):
        pick = rng.choice(len(sents), size=min(n, len(sents)), replace=False)
        rows += [{"business_id": bid, "topic": topic, "text": sents[i], "n_available": len(sents)} for i in sorted(pick)]
    return pd.DataFrame(rows)


# --- Human check and agreement --------------------------------------------------------------

def lookup_url(name: str, address: str) -> str:
    """Google Maps search for the restaurant, so the grader can read reviews and see photos."""
    from urllib.parse import quote_plus

    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{name} {address} Philadelphia")


def human_check_rows(pool: pd.DataFrame, qids=HUMAN_CHECK_QIDS) -> pd.DataFrame:
    rows = pool[pool["qid"].isin(qids)].drop(columns=[c for c in pool.columns if c not in {
        "qid", "query", "business_id", "name", "address", "postal_code", "price", "categories"}])
    rows = rows.assign(lookup=[lookup_url(n, a) for n, a in zip(rows["name"], rows["address"])])
    return rows.assign(grade="").reset_index(drop=True)


def cohen_kappa(a: list[int], b: list[int], labels=(0, 1, 2), weights: str | None = None) -> float:
    """Cohen's kappa; weights=None (unweighted) or "quadratic" for ordinal grades."""
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = ((i - j) ** 2 / (k - 1) ** 2) if weights == "quadratic" else (i != j).astype(float)
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def agreement(human: pd.DataFrame, judged: pd.DataFrame) -> dict:
    h = human[human["grade"].astype(str).str.strip() != ""]
    merged = h.merge(judged, on=["qid", "business_id"], suffixes=("_human", "_llm"))
    a = merged["grade_human"].astype(float).astype(int).tolist()
    b = merged["grade_llm"].astype(float).astype(int).tolist()
    return {
        "n": len(merged),
        "exact": float(np.mean(np.array(a) == np.array(b))) if a else float("nan"),
        "kappa": cohen_kappa(a, b) if a else float("nan"),
        "kappa_quadratic": cohen_kappa(a, b, weights="quadratic") if a else float("nan"),
        "confusion": pd.crosstab(pd.Series(a, name="human"), pd.Series(b, name="llm")),
        "rows": merged,
    }


# --- Commands ------------------------------------------------------------------------------------

def cmd_evidence(_args) -> None:
    pool = pd.read_csv(config.EVAL_DIR / "pool.csv", dtype=str, keep_default_na=False)
    ids = set(pool["business_id"])
    attrs = load_attributes(ids)
    (config.EVAL_DIR / "attributes.json").write_text(json.dumps(attrs, indent=1, sort_keys=True, default=str))
    ev = sample_evidence(ids)
    ev.to_parquet(config.EVAL_DIR / "grading_evidence.parquet", index=False)
    print(f"{len(ids)} restaurants: attributes for {len(attrs)}, {len(ev):,} sampled sentences")


def cmd_human_check(args) -> None:
    out = config.EVAL_DIR / "human_check.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; it may hold your grades. Use --force to overwrite.")
    rows = human_check_rows(pd.read_csv(config.EVAL_DIR / "pool.csv", dtype=str, keep_default_na=False))
    rows.to_csv(out, index=False)
    print(f"wrote {out}: {len(rows)} rows for queries {', '.join(HUMAN_CHECK_QIDS)}")


def cmd_agreement(_args) -> None:
    res = agreement(
        pd.read_csv(config.EVAL_DIR / "human_check.csv", dtype=str, keep_default_na=False),
        pd.read_csv(config.EVAL_DIR / "pool.csv", dtype=str, keep_default_na=False)[["qid", "business_id", "grade"]],
    )
    print(f"{res['n']} pairs: exact agreement {res['exact']:.3f}, kappa {res['kappa']:.3f}, "
          f"quadratic-weighted kappa {res['kappa_quadratic']:.3f}\n\n{res['confusion']}")
    res["rows"].to_csv(config.EVAL_DIR / "agreement_rows.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evidence")
    sub.add_parser("human-check").add_argument("--force", action="store_true")
    sub.add_parser("agreement")
    args = parser.parse_args()
    {"evidence": cmd_evidence, "human-check": cmd_human_check, "agreement": cmd_agreement}[args.cmd](args)


if __name__ == "__main__":
    main()
