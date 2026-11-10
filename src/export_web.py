"""Export everything the static web UI (docs/) needs to rank queries in the browser.

The ranker's retrieval text is always "<pole hypothesis> <matched lexicon term>", so the set of
possible retrievals is finite: every lexicon term, for each of its aspect's two poles (a negation
cue flips a term to the other pole). We run each one here with the real BM25 + dense + RRF
retrieval and keep only the sentences the ranker can use (the first k per restaurant in fused
order). The browser then only parses the query and does the scoring arithmetic, which
reproduces src/rank.py's "full" system.

Output (docs/idx/):
    lexicon.json        parser rules, pole ids and hypotheses, retrieval file per (column, term)
    restaurants.json    display fields per restaurant
    r/<n>.json          one retrieval: [[restaurant index, fused sim, polarity, text], ...]

Usage:
    python -m src.export_web [--out docs/idx]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src import config
from src.parse import NEGATION_CUES, CLAUSE_BOUNDARIES, NEGATION_WINDOW, Target, term_key
from src.prep import load_lexicon, normalize_text
from src.rank import Ranker, group_top_k

DEFAULT_OUT = config.ROOT / "docs" / "idx"


def retrieval_keys(lexicon: dict) -> list[tuple[str, str, str, str]]:
    """(aspect, side, pole id, canonical term) for every term x both poles of its aspect."""
    seen: dict[str, tuple[str, str]] = {}
    keys = []
    for aspect, poles in lexicon["aspects"].items():
        for side in ("positive", "negative"):
            for term in poles[side]["terms"]:
                key = term_key(normalize_text(term))
                if key in seen:
                    continue
                seen[key] = (aspect, side)
                for target_side in ("positive", "negative"):
                    keys.append((aspect, target_side, poles[target_side]["id"], normalize_text(term)))
    return keys


def export(out: Path) -> None:
    ranker = Ranker(load_baseline=False)
    lex = load_lexicon()
    keys = retrieval_keys(lex)

    bids = ranker.businesses.index.to_list()
    bidx = {b: i for i, b in enumerate(bids)}

    if out.exists():
        shutil.rmtree(out)
    (out / "r").mkdir(parents=True)

    texts = [f"{ranker.hypothesis[f'{a}.{pid}']} {term}" for a, _, pid, term in keys]
    embs = ranker.encode(texts)
    files: dict[str, int] = {}
    kept_total = 0
    for n, ((aspect, side, pid, term), emb) in enumerate(zip(keys, embs)):
        target = Target(aspect, pid, side, term, False)
        fused = ranker.retrieve(target, emb)
        groups = group_top_k(fused, ranker.business_of_row, ranker.k)
        keep = {row for ev in groups.values() for row, _ in ev}
        pol = ranker.polarity[target.column]
        rows = [
            [bidx[ranker.business_of_row[row]], sim, round(float(pol[row]), 6), str(ranker.texts[row])]
            for row, sim in fused
            if row in keep
        ]
        kept_total += len(rows)
        (out / "r" / f"{n}.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        files[f"{target.column}|{term_key(term)}"] = n

    lexicon_out = {
        "exclude": [normalize_text(p) for p in lex.get("exclude") or []],
        "negation_cues": sorted(NEGATION_CUES),
        "clause_boundaries": sorted(CLAUSE_BOUNDARIES),
        "negation_window": NEGATION_WINDOW,
        "aspects": {
            aspect: {
                side: {
                    "id": p["id"],
                    "hypothesis": p["hypothesis"],
                    "terms": [normalize_text(t) for t in p["terms"]],
                }
                for side, p in poles.items()
            }
            for aspect, poles in lex["aspects"].items()
        },
        "params": {"k": ranker.k, "m": ranker.m, "top_n": ranker.top_n, "k_rrf": ranker.k_rrf},
        "files": files,
    }
    (out / "lexicon.json").write_text(json.dumps(lexicon_out, ensure_ascii=False, indent=1))

    b = ranker.businesses
    restaurants = {
        "fields": ["business_id", "name", "address", "postal_code", "stars", "review_count", "price", "categories"],
        "rows": [
            [
                bid,
                b.at[bid, "name"],
                b.at[bid, "address"],
                b.at[bid, "postal_code"],
                float(b.at[bid, "stars"]),
                int(b.at[bid, "review_count"]),
                None if pd.isna(b.at[bid, "price"]) else int(b.at[bid, "price"]),
                b.at[bid, "categories"],
            ]
            for bid in bids
        ],
    }
    (out / "restaurants.json").write_text(json.dumps(restaurants, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out}: {len(keys)} retrievals, {kept_total:,} evidence rows, {len(bids)} restaurants")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
