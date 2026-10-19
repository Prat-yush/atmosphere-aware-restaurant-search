"""Step 1: filter Philadelphia restaurants, stream reviews, keep lexicon-matching sentences.

Usage:
    python -m src.prep                      # full run -> data/proc/sentences.parquet
    python -m src.prep --max-reviews 50000  # sample  -> data/proc/sentences.sample50000.parquet
    python -m src.prep --force              # recompute even if outputs exist
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import yaml

from src import config

SENTENCE_COLUMNS = ["sent_id", "business_id", "review_id", "text", "matched_aspects"]
BUSINESS_COLUMNS = [
    "business_id", "name", "address", "postal_code", "latitude", "longitude",
    "stars", "review_count", "price", "categories",
]


# --- Lexicon matching ------------------------------------------------------

def load_lexicon(path: Path = config.LEXICON_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Lowercase and normalize curly quotes so lexicon terms match consistently."""
    return text.replace("’", "'").replace("‘", "'").lower()


def term_to_pattern(term: str) -> str:
    """Regex for one term: spaces/hyphens interchangeable, whole-word boundaries."""
    words = [re.escape(w) for w in re.split(r"[\s-]+", normalize_text(term).strip())]
    return r"[\s-]+".join(words)


def compile_terms(terms: Iterable[str]) -> re.Pattern[str]:
    # Longest first so alternation prefers "dimly lit" over "dim".
    ordered = sorted(set(terms), key=len, reverse=True)
    alternation = "|".join(term_to_pattern(t) for t in ordered)
    return re.compile(rf"(?<![\w'])(?:{alternation})(?![\w'])")


def build_matchers(lexicon: dict) -> dict[str, re.Pattern[str]]:
    """One compiled pattern per aspect, covering the terms of both poles."""
    return {
        aspect: compile_terms(poles["positive"]["terms"] + poles["negative"]["terms"])
        for aspect, poles in lexicon["aspects"].items()
    }


def build_exclusion(lexicon: dict) -> re.Pattern[str] | None:
    phrases = lexicon.get("exclude") or []
    return compile_terms(phrases) if phrases else None


def match_aspects(
    text: str,
    matchers: dict[str, re.Pattern[str]],
    exclusion: re.Pattern[str] | None = None,
) -> list[str]:
    """Aspects (in lexicon order) with at least one term in `text`."""
    norm = normalize_text(text)
    if exclusion is not None:
        norm = exclusion.sub(" ", norm)
    return [aspect for aspect, pat in matchers.items() if pat.search(norm)]


# --- Business filtering ----------------------------------------------------

def normalize_city(city: str | None) -> str:
    return " ".join((city or "").split()).casefold()


def parse_categories(categories: str | None) -> list[str]:
    return [c.strip() for c in (categories or "").split(",") if c.strip()]


def is_target_business(record: dict) -> bool:
    return (
        normalize_city(record.get("city")) == config.TARGET_CITY
        and config.TARGET_CATEGORY in parse_categories(record.get("categories"))
        and (record.get("review_count") or 0) >= config.MIN_REVIEW_COUNT
    )


def parse_price(record: dict) -> int | None:
    raw = (record.get("attributes") or {}).get("RestaurantsPriceRange2")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_target_businesses(path: Path) -> pd.DataFrame:
    rows = [
        {
            "business_id": r["business_id"],
            "name": r.get("name"),
            "address": r.get("address"),
            "postal_code": r.get("postal_code"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "stars": r.get("stars"),
            "review_count": r.get("review_count"),
            "price": parse_price(r),
            "categories": r.get("categories"),
        }
        for r in iter_jsonl(path)
        if is_target_business(r)
    ]
    df = pd.DataFrame(rows, columns=BUSINESS_COLUMNS)
    df["price"] = df["price"].astype("Int64")
    return df.sort_values("business_id").reset_index(drop=True)


# --- Review streaming and sentence splitting -------------------------------

def iter_target_reviews(
    path: Path, business_ids: set[str], max_reviews: int | None = None
) -> Iterator[dict]:
    """Stream reviews for target businesses in file order, never loading the file."""
    n = 0
    for review in iter_jsonl(path):
        if review["business_id"] not in business_ids:
            continue
        yield review
        n += 1
        if max_reviews is not None and n >= max_reviews:
            return


def load_sentencizer(model: str = config.SPACY_MODEL):
    """en_core_web_sm tokenizer + rule-based sentencizer; statistical components excluded."""
    import spacy

    nlp = spacy.load(
        model, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"]
    )
    nlp.add_pipe("sentencizer")
    return nlp


def split_paragraphs(text: str) -> list[str]:
    """Reviews often use line breaks without punctuation; treat each line as a hard boundary."""
    return [p.strip() for p in text.splitlines() if p.strip()]


def clean_sentence(text: str) -> str:
    return " ".join(text.split())


def iter_review_sentences(
    reviews: Iterable[dict], nlp, batch_size: int = config.SPACY_BATCH_SIZE
) -> Iterator[tuple[dict, int, str]]:
    """Yield (review, sentence_index_within_review, sentence_text)."""

    def paragraphs() -> Iterator[tuple[str, tuple[dict, list[int]]]]:
        for review in reviews:
            counter = [0]  # shared across this review's paragraphs
            for para in split_paragraphs(review.get("text") or ""):
                yield para, (review, counter)

    for doc, (review, counter) in nlp.pipe(paragraphs(), as_tuples=True, batch_size=batch_size):
        for sent in doc.sents:
            text = clean_sentence(sent.text)
            if text:
                yield review, counter[0], text
                counter[0] += 1


def extract_sentences(
    reviews: Iterable[dict],
    nlp,
    matchers: dict[str, re.Pattern[str]],
    exclusion: re.Pattern[str] | None,
    stats: Counter,
) -> pd.DataFrame:
    rows = []
    seen_reviews: set[str] = set()
    for review, idx, text in iter_review_sentences(reviews, nlp):
        seen_reviews.add(review["review_id"])
        stats["sentences_total"] += 1
        aspects = match_aspects(text, matchers, exclusion)
        if not aspects:
            continue
        rows.append(
            {
                "sent_id": f"{review['review_id']}_{idx}",
                "business_id": review["business_id"],
                "review_id": review["review_id"],
                "text": text,
                "matched_aspects": aspects,
            }
        )
    stats["reviews"] = len(seen_reviews)
    return pd.DataFrame(rows, columns=SENTENCE_COLUMNS)


# --- Reporting -------------------------------------------------------------

def aspect_counts(sentences: pd.DataFrame, aspects: Iterable[str]) -> dict[str, int]:
    counts = Counter(a for lst in sentences["matched_aspects"] for a in lst)
    return {a: counts.get(a, 0) for a in aspects}


def report(businesses: pd.DataFrame, sentences: pd.DataFrame, stats: Counter, aspects: list[str]) -> None:
    print(f"restaurants (filtered):          {len(businesses):>10,}")
    print(f"restaurants with kept sentences: {sentences['business_id'].nunique():>10,}")
    print(f"reviews processed:               {stats['reviews']:>10,}")
    print(f"sentences (all):                 {stats['sentences_total']:>10,}")
    print(f"sentences kept:                  {len(sentences):>10,}")
    print("kept sentences per aspect (a sentence can match several):")
    for aspect, n in aspect_counts(sentences, aspects).items():
        print(f"  {aspect:10s} {n:>10,}")


# --- CLI -------------------------------------------------------------------

def sentences_output_path(max_reviews: int | None) -> Path:
    if max_reviews is None:
        return config.SENTENCES_PATH
    return config.PROC_DIR / f"sentences.sample{max_reviews}.parquet"


def run(
    business_path: Path,
    review_path: Path,
    businesses_out: Path,
    sentences_out: Path,
    max_reviews: int | None = None,
    force: bool = False,
    lexicon_path: Path = config.LEXICON_PATH,
) -> pd.DataFrame:
    lexicon = load_lexicon(lexicon_path)
    aspects = list(lexicon["aspects"])

    if businesses_out.exists() and not force:
        print(f"skip: {businesses_out} exists")
        businesses = pd.read_parquet(businesses_out)
    else:
        businesses = load_target_businesses(business_path)
        businesses_out.parent.mkdir(parents=True, exist_ok=True)
        businesses.to_parquet(businesses_out, index=False)

    if sentences_out.exists() and not force:
        print(f"skip: {sentences_out} exists (use --force to recompute)")
        sentences = pd.read_parquet(sentences_out)
        print(f"sentences kept: {len(sentences):,}")
        return sentences

    stats: Counter = Counter()
    reviews = iter_target_reviews(review_path, set(businesses["business_id"]), max_reviews)
    sentences = extract_sentences(
        reviews, load_sentencizer(), build_matchers(lexicon), build_exclusion(lexicon), stats
    )
    sentences_out.parent.mkdir(parents=True, exist_ok=True)
    sentences.to_parquet(sentences_out, index=False)
    report(businesses, sentences, stats, aspects)
    print(f"wrote {sentences_out}")
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-reviews", type=int, default=None, help="stop after N target reviews (sample run)")
    parser.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    args = parser.parse_args()
    run(
        config.RAW_BUSINESS_PATH,
        config.RAW_REVIEW_PATH,
        config.BUSINESSES_PATH,
        sentences_output_path(args.max_reviews),
        max_reviews=args.max_reviews,
        force=args.force,
    )


if __name__ == "__main__":
    main()
