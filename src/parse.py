"""Rule-based query parser: atmosphere query -> list of (aspect, target pole) targets.

Rules:
  - Lexicon terms are matched longest-first and without overlap, so "loud music" maps to
    music and not to noise, and "not romantic" is a negative-pole term in its own right.
  - Negation: a cue (not, no, never, without, n't, isn't, wasn't, ...) within 3 tokens before
    a term flips its pole. A cue binds only to the nearest following term, and it does not
    reach past a clause boundary (but, yet, though, punctuation) or an earlier term.
    Known limitation: "not loud or crowded" negates only "loud".
  - Mixed queries ("X but Y", "X and Y") give one target per term. Repeats of the same
    (aspect, pole) are dropped.
  - A query with no lexicon term gives no targets.

Usage:
    python -m src.parse "quiet but not boring, good for a first date"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from functools import lru_cache

from src.prep import build_exclusion, compile_terms, load_lexicon, normalize_text

NEGATION_CUES = frozenset({"not", "no", "never", "without", "nothing", "isnt", "wasnt", "dont", "doesnt", "arent", "werent"})
CLAUSE_BOUNDARIES = frozenset({"but", "yet", "though", "although", "however", "while"})
NEGATION_WINDOW = 3  # tokens
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
PUNCT_RE = re.compile(r"[,.;:!?()]")


@dataclass(frozen=True)
class Target:
    aspect: str
    target_pole: str  # pole id, e.g. "quiet"; polarity column is f"{aspect}.{target_pole}"
    side: str  # "positive" | "negative"
    term: str  # matched query text
    negated: bool  # True when a negation cue flipped the term's pole

    @property
    def column(self) -> str:
        return f"{self.aspect}.{self.target_pole}"


def term_key(text: str) -> str:
    return re.sub(r"[\s-]+", " ", text.strip())


class QueryParser:
    def __init__(self, lexicon: dict):
        self.poles = {aspect: {side: p["id"] for side, p in poles.items()} for aspect, poles in lexicon["aspects"].items()}
        self.term_pole: dict[str, tuple[str, str]] = {}
        for aspect, poles in lexicon["aspects"].items():
            for side, pole in poles.items():
                for term in pole["terms"]:
                    self.term_pole[term_key(normalize_text(term))] = (aspect, side)
        self.matcher = compile_terms(self.term_pole)
        self.exclusion = build_exclusion(lexicon)

    def parse(self, query: str) -> list[Target]:
        text = normalize_text(query)
        if self.exclusion is not None:
            text = self.exclusion.sub(lambda m: " " * len(m.group(0)), text)  # keep offsets
        tokens = list(TOKEN_RE.finditer(text))
        matches = list(self.matcher.finditer(text))

        targets: list[Target] = []
        used_cues: set[int] = set()
        prev_end = 0
        for m in matches:
            aspect, side = self.term_pole[term_key(m.group(0))]
            cue = find_negation_cue(text, tokens, m.start(), prev_end, used_cues)
            if cue is not None:
                used_cues.add(cue)
                side = "negative" if side == "positive" else "positive"
            prev_end = m.end()
            target = Target(aspect, self.poles[aspect][side], side, m.group(0), cue is not None)
            if all((t.aspect, t.side) != (target.aspect, target.side) for t in targets):
                targets.append(target)
        return targets


def is_negation_cue(token: str) -> bool:
    return token in NEGATION_CUES or token.endswith("n't")


def find_negation_cue(
    text: str, tokens: list[re.Match], term_start: int, prev_end: int, used: set[int]
) -> int | None:
    """Index of an unused negation cue within the window before term_start, else None."""
    before = [i for i, t in enumerate(tokens) if prev_end <= t.start() and t.end() <= term_start]
    for i in reversed(before[-NEGATION_WINDOW:]):
        tok = tokens[i]
        gap = text[tok.end() : term_start if i == before[-1] else tokens[i + 1].start()]
        if PUNCT_RE.search(gap) or tok.group(0) in CLAUSE_BOUNDARIES:
            return None
        if is_negation_cue(tok.group(0)) and i not in used:
            return i
    return None


@lru_cache(maxsize=1)
def default_parser() -> QueryParser:
    return QueryParser(load_lexicon())


def parse_query(query: str) -> list[Target]:
    return default_parser().parse(query)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    args = parser.parse_args()
    targets = parse_query(args.query)
    if not targets:
        print("no atmosphere aspects found")
    for t in targets:
        flag = "  (negated)" if t.negated else ""
        print(f"{t.aspect:9s} -> {t.target_pole:11s} from {t.term!r}{flag}")


if __name__ == "__main__":
    main()
