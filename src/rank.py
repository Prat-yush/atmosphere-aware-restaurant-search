"""Online ranking: query -> restaurants scored from retrieved review-sentence evidence.

Per target (aspect, pole) from the parser:
  1. Retrieve sentences whose lexicon match includes the target's aspect, with BM25 and
     dense (FAISS) separately, using the text "<target hypothesis> <matched query term>".
     Fuse with reciprocal rank fusion (k_rrf) and keep the top_n fused sentences.
  2. Group by restaurant; keep each restaurant's top-k sentences by fused score: E_r.
  3. Aggregate over retrieved evidence only (the denominator is E_r, never all reviews):
        w_i     = sim_i / sum_{j in E_r} sim_j        (sim = fused RRF score; sum_i w_i = 1)
        raw_r   = sum_i w_i * p_i                     (p_i = NLI polarity for the target pole)
        n_r     = |E_r| <= k
        score_r = (n_r * raw_r + m * mu) / (n_r + m)  (shrinks low-evidence restaurants to mu)
     mu = mean raw_r of the full system over restaurants with evidence for that target,
     unless a fixed mu is given.
  4. Final score = mean of per-target scores; a restaurant with no evidence for a target
     gets mu for it. Only restaurants with evidence for at least one target are ranked.

Systems: "full"; "no_polarity" (every p_i = +1, same mu as full); "bm25_baseline"
(BM25 over each restaurant's concatenated reviews, no aspects).

Usage:
    python -m src.rank "quiet but not boring, good for a first date" [--system full] [--top 10]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, index
from src.nli import load_hypotheses
from src.parse import QueryParser, Target
from src.prep import load_lexicon

SYSTEMS = ("full", "no_polarity", "bm25_baseline")


# --- Pure scoring functions ----------------------------------------------------

def rrf_fuse(rankings: list[np.ndarray], k_rrf: int = config.K_RRF) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k_rrf + rank), rank from 1."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            fused[int(row)] += 1.0 / (k_rrf + rank)
    return dict(fused)


def top_fused(fused: dict[int, float], top_n: int) -> list[tuple[int, float]]:
    """Best top_n (row, score), ties broken by row id for determinism."""
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


def group_top_k(rows_sims: list[tuple[int, float]], business_of_row: np.ndarray, k: int) -> dict[str, list[tuple[int, float]]]:
    """Per restaurant, its k best (row, sim) pairs. Input must already be sorted best first."""
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, sim in rows_sims:
        bid = business_of_row[row]
        if len(groups[bid]) < k:
            groups[bid].append((row, sim))
    return dict(groups)


def evidence_weights(sims: np.ndarray) -> np.ndarray:
    """Normalize within the restaurant so weights sum to 1."""
    sims = np.asarray(sims, dtype=np.float64)
    total = sims.sum()
    if total <= 0:
        raise ValueError("evidence similarities must be positive")
    return sims / total


def raw_score(sims: np.ndarray, polarities: np.ndarray) -> float:
    return float(np.dot(evidence_weights(sims), polarities))


def shrink(raw: float, n: int, m: float, mu: float) -> float:
    return (n * raw + m * mu) / (n + m)


# --- Result types ----------------------------------------------------------------

@dataclass
class Evidence:
    sent_id: str
    text: str
    weight: float
    polarity: float
    label: str  # "supports" if polarity > 0 else "contradicts"


@dataclass
class TargetScore:
    target: Target
    score: float
    raw: float | None  # None when the restaurant has no evidence for this target
    n: int
    m: float
    mu: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Result:
    business_id: str
    score: float
    targets: list[TargetScore]

    @property
    def n_evidence(self) -> int:
        return sum(t.n for t in self.targets)


# --- Ranker -------------------------------------------------------------------------

class Ranker:
    """Loads all precomputed artifacts once; rank() does only query encoding + lookups."""

    def __init__(
        self,
        k: int = config.K_PER_RESTAURANT,
        m: float = config.M_SHRINK,
        mu: float | None = None,
        k_rrf: int = config.K_RRF,
        top_n: int = config.TOP_N_SENTENCES,
        lexicon_path: Path = config.LEXICON_PATH,
        load_baseline: bool = True,
    ):
        self.k, self.m, self.mu, self.k_rrf, self.top_n = k, m, mu, k_rrf, top_n
        lexicon = load_lexicon(lexicon_path)
        self.parser = QueryParser(lexicon)
        self.hypothesis = dict(load_hypotheses(lexicon))  # column -> hypothesis

        self.bm25, self.dense, ids = index.load_indexes()
        sents = pd.read_parquet(config.SENTENCES_PATH)
        pol = pd.read_parquet(config.POLARITY_PATH)
        if not (ids["sent_id"].equals(sents["sent_id"]) and pol["sent_id"].equals(sents["sent_id"])):
            raise ValueError("sentences, indexes and polarity are not row-aligned; rebuild them")
        self.sent_ids = sents["sent_id"].to_numpy()
        self.texts = sents["text"].to_numpy()
        self.business_of_row = sents["business_id"].to_numpy()
        self.polarity = {c: pol[c].to_numpy() for c in pol.columns if c != "sent_id"}
        self.aspect_rows = {
            aspect: np.flatnonzero(sents["matched_aspects"].map(lambda a, asp=aspect: asp in a).to_numpy())
            for aspect in lexicon["aspects"]
        }
        self.aspect_mask = {a: np.isin(np.arange(len(sents)), rows) for a, rows in self.aspect_rows.items()}
        self.businesses = pd.read_parquet(config.BUSINESSES_PATH).set_index("business_id")
        self.encode = index.make_encoder(index.load_encoder(device="cpu"))
        self.baseline = index.load_restaurant_bm25() if load_baseline else None

    # -- retrieval
    def retrieval_text(self, target: Target) -> str:
        return f"{self.hypothesis[target.column]} {target.term}"

    def retrieve(self, target: Target, query_emb: np.ndarray) -> list[tuple[int, float]]:
        text = self.retrieval_text(target)
        mask = self.aspect_mask[target.aspect]

        bm25 = index.bm25_scores(self.bm25, text)
        bm25 = np.where(mask & (bm25 > 0), bm25, -np.inf)
        n_pos = int(np.isfinite(bm25).sum())
        depth = min(self.top_n, n_pos)
        bm25_rank = np.argpartition(-bm25, depth - 1)[:depth] if depth else np.array([], dtype=int)
        bm25_rank = bm25_rank[np.lexsort((bm25_rank, -bm25[bm25_rank]))]

        dense_rank = dense_search_subset(self.dense, query_emb, self.aspect_rows[target.aspect], self.top_n)
        return top_fused(rrf_fuse([bm25_rank, dense_rank], self.k_rrf), self.top_n)

    # -- scoring
    def score_target(self, target: Target, query_emb: np.ndarray, use_polarity: bool) -> tuple[dict[str, TargetScore], float]:
        groups = group_top_k(self.retrieve(target, query_emb), self.business_of_row, self.k)
        pol = self.polarity[target.column]
        raws = {}
        for bid, ev in groups.items():
            rows = np.array([r for r, _ in ev])
            sims = np.array([s for _, s in ev])
            raws[bid] = (rows, sims, raw_score(sims, pol[rows]))
        mu = self.mu if self.mu is not None else (float(np.mean([r for *_, r in raws.values()])) if raws else 0.0)

        scores = {}
        for bid, (rows, sims, raw) in raws.items():
            p = pol[rows] if use_polarity else np.ones(len(rows))
            r = raw if use_polarity else raw_score(sims, p)
            w = evidence_weights(sims)
            evidence = [
                Evidence(self.sent_ids[i], self.texts[i], float(wi), float(pi), "supports" if pi > 0 else "contradicts")
                for i, wi, pi in zip(rows, w, p)
            ]
            scores[bid] = TargetScore(target, shrink(r, len(rows), self.m, mu), r, len(rows), self.m, mu, evidence)
        return scores, mu

    def rank(self, query: str, system: str = "full", top: int | None = None) -> tuple[list[Target], list[Result]]:
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
        targets = self.parser.parse(query)
        if system == "bm25_baseline":
            return targets, self.rank_baseline(query, top)
        if not targets:
            return targets, []

        embs = self.encode([self.retrieval_text(t) for t in targets])
        per_target = [self.score_target(t, e, use_polarity=(system == "full")) for t, e in zip(targets, embs)]
        candidates = sorted({bid for scores, _ in per_target for bid in scores})
        results = []
        for bid in candidates:
            ts = [
                scores.get(bid) or TargetScore(t, mu, None, 0, self.m, mu)
                for t, (scores, mu) in zip(targets, per_target)
            ]
            results.append(Result(bid, float(np.mean([x.score for x in ts])), ts))
        results.sort(key=lambda r: (-r.score, -r.n_evidence, r.business_id))
        return targets, results[:top] if top else results

    def rank_baseline(self, query: str, top: int | None) -> list[Result]:
        if self.baseline is None:
            raise RuntimeError("restaurant BM25 not loaded; run `python -m src.index` first")
        bm25, bids = self.baseline
        scores = index.bm25_scores(bm25, query)
        order = np.lexsort((bids, -scores))
        results = [Result(bids[i], float(scores[i]), []) for i in order if scores[i] > 0]
        return results[:top] if top else results


def dense_search_subset(dense, query_emb: np.ndarray, rows: np.ndarray, top_n: int) -> np.ndarray:
    """FAISS search restricted to `rows` (exact, via an ID selector); row ids best first."""
    import faiss

    params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(rows.astype(np.int64)))
    _, ids = dense.search(query_emb.reshape(1, -1).astype(np.float32), min(top_n, len(rows)), params=params)
    return ids[0][ids[0] >= 0]


# --- CLI ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--system", choices=SYSTEMS, default="full")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    ranker = Ranker()
    t0 = time.time()
    targets, results = ranker.rank(args.query, args.system, args.top)
    print(f"targets: {[t.column + (' (negated)' if t.negated else '') for t in targets]}  ({time.time() - t0:.2f}s)")
    for i, r in enumerate(results, 1):
        name = ranker.businesses.loc[r.business_id, "name"]
        print(f"\n{i:2d}. {name}  score {r.score:+.3f}  evidence {r.n_evidence}")
        for ts in r.targets:
            raw = "  -  " if ts.raw is None else f"{ts.raw:+.3f}"
            print(f"    {ts.target.column:18s} score {ts.score:+.3f}  raw {raw}  n {ts.n}  mu {ts.mu:+.3f}")
            for e in ts.evidence[:2]:
                print(f"      [{e.label:11s} p={e.polarity:+.2f} w={e.weight:.2f}] {e.text[:110]}")


if __name__ == "__main__":
    main()
