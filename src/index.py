"""Step 2: retrieval indexes.

Sentence level (AmbiSense): BM25 and dense (MiniLM + FAISS inner product) over kept
sentences. Row i of both is row i of data/proc/index_ids.parquet (sent_id, business_id).

Restaurant level (bm25_baseline): BM25 over each restaurant's concatenated reviews,
built by streaming the raw review file again. Stored as a sparse term-count matrix
(rank_bm25 keeps a Python dict per document, too large for ~73M tokens) and scored
with the same BM25Okapi formula.

Usage:
    python -m src.index          # -> bm25.pkl, dense.faiss, index_ids.parquet, bm25_restaurants.pkl
    python -m src.index --force  # rebuild even if outputs exist
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.prep import normalize_text

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

Encoder = Callable[[Sequence[str]], np.ndarray]


# --- BM25 ------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps contractions whole ("wasn't") so negation cues survive."""
    return TOKEN_RE.findall(normalize_text(text))


def build_bm25(texts: Sequence[str]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(t) for t in texts])


def bm25_scores(bm25, query: str) -> np.ndarray:
    return bm25.get_scores(tokenize(query))


class SparseBM25:
    """BM25Okapi (rank_bm25's exact formula and idf floor) over a CSR doc x term count matrix."""

    def __init__(self, counts, vocab: dict[str, int], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        import scipy.sparse as sp

        self.counts = sp.csc_matrix(counts, dtype=np.float32)  # column access per query term
        self.vocab = vocab
        self.k1, self.b = k1, b
        self.corpus_size = self.counts.shape[0]
        self.doc_len = np.asarray(self.counts.sum(axis=1)).ravel()
        self.avgdl = float(self.doc_len.mean())
        df = np.diff(self.counts.indptr)  # docs containing each term
        idf = np.log(self.corpus_size - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * idf.mean()
        self.idf = idf

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for q in query:  # a repeated query token counts again, as in rank_bm25
            j = self.vocab.get(q)
            if j is None:
                continue
            lo, hi = self.counts.indptr[j], self.counts.indptr[j + 1]
            docs, tf = self.counts.indices[lo:hi], self.counts.data[lo:hi]
            score[docs] += self.idf[j] * tf * (self.k1 + 1) / (tf + norm[docs])
        return score


def build_restaurant_counts(reviews: Iterable[dict], business_ids: Sequence[str], flush_every: int = 50_000):
    """CSR (restaurant x term) counts over all review text, plus the vocabulary."""
    import scipy.sparse as sp

    row_of = {bid: i for i, bid in enumerate(business_ids)}
    vocab: dict[str, int] = {}
    total = None
    rows: list[int] = []
    cols: list[int] = []

    def flush():
        nonlocal total, rows, cols
        if rows:
            part = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(row_of), max(len(vocab), 1)))
            total = part if total is None else _pad_cols(total, part.shape[1]) + part
        rows, cols = [], []

    for n, review in enumerate(reviews, 1):
        r = row_of[review["business_id"]]
        for tok in tokenize(review.get("text") or ""):
            rows.append(r)
            cols.append(vocab.setdefault(tok, len(vocab)))
        if n % flush_every == 0:
            flush()
    flush()
    total = _pad_cols(total, len(vocab)) if total is not None else sp.csr_matrix((len(row_of), 0))
    return total.tocsr(), vocab


def _pad_cols(m, n_cols: int):
    import scipy.sparse as sp

    if m.shape[1] == n_cols:
        return m
    return sp.csr_matrix((m.data, m.indices, m.indptr), shape=(m.shape[0], n_cols))


# --- Dense -----------------------------------------------------------------

def load_encoder(model_name: str = config.DENSE_MODEL, device: str | None = config.EMBED_DEVICE):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def make_encoder(model, batch_size: int = config.EMBED_BATCH_SIZE, show_progress: bool = False) -> Encoder:
    def encode(texts: Sequence[str]) -> np.ndarray:
        emb = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    return encode


def build_dense(embeddings: np.ndarray):
    """Inner-product index; embeddings must be L2-normalized so IP == cosine."""
    import faiss

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("embeddings must be L2-normalized")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


def dense_search(index, query_emb: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    """(scores, row_ids) for one normalized query embedding, best first."""
    scores, ids = index.search(query_emb.reshape(1, -1).astype(np.float32), top_n)
    keep = ids[0] >= 0
    return scores[0][keep], ids[0][keep]


# --- Persistence -----------------------------------------------------------

def save_indexes(bm25, dense, ids: pd.DataFrame, bm25_path: Path, dense_path: Path, ids_path: Path) -> None:
    import faiss

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    faiss.write_index(dense, str(dense_path))
    ids.to_parquet(ids_path, index=False)


def load_indexes(
    bm25_path: Path = config.BM25_PATH,
    dense_path: Path = config.DENSE_INDEX_PATH,
    ids_path: Path = config.INDEX_IDS_PATH,
):
    """Load (bm25, dense, ids) and check that all three describe the same rows."""
    import faiss

    with open(bm25_path, "rb") as f:  # our own artifact, written by save_indexes
        bm25 = pickle.load(f)
    dense = faiss.read_index(str(dense_path))
    ids = pd.read_parquet(ids_path)
    if not (bm25.corpus_size == dense.ntotal == len(ids)):
        raise ValueError(
            f"index size mismatch: bm25={bm25.corpus_size} dense={dense.ntotal} ids={len(ids)}; "
            "rebuild with `python -m src.index --force`"
        )
    return bm25, dense, ids


# --- CLI -------------------------------------------------------------------

def run(
    sentences_path: Path,
    bm25_path: Path,
    dense_path: Path,
    ids_path: Path,
    encoder: Encoder | None = None,
    force: bool = False,
) -> None:
    outputs = (bm25_path, dense_path, ids_path)
    if all(p.exists() for p in outputs) and not force:
        print(f"skip: indexes exist in {bm25_path.parent} (use --force to rebuild)")
        return

    sentences = pd.read_parquet(sentences_path, columns=["sent_id", "business_id", "text"])
    texts = sentences["text"].tolist()
    print(f"indexing {len(texts):,} sentences from {sentences_path}")

    bm25 = build_bm25(texts)
    print("bm25: built")

    if encoder is None:
        encoder = make_encoder(load_encoder(), show_progress=True)
    dense = build_dense(encoder(texts))
    print(f"dense: {dense.ntotal:,} vectors, dim {dense.d}")

    save_indexes(bm25, dense, sentences[["sent_id", "business_id"]], *outputs)
    for p in outputs:
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


def run_restaurant_bm25(review_path: Path, businesses_path: Path, out_path: Path, force: bool = False) -> None:
    from src.prep import iter_target_reviews

    if out_path.exists() and not force:
        print(f"skip: {out_path} exists (use --force to rebuild)")
        return
    bids = pd.read_parquet(businesses_path, columns=["business_id"])["business_id"].to_numpy()
    print(f"restaurant BM25: streaming reviews for {len(bids):,} restaurants")
    counts, vocab = build_restaurant_counts(iter_target_reviews(review_path, set(bids)), bids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_restaurant_bm25(counts, vocab, bids, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB; {len(vocab):,} terms, "
          f"{int(counts.sum()):,} tokens)")


def save_restaurant_bm25(counts, vocab: dict[str, int], bids: np.ndarray, path: Path) -> None:
    """Plain data only (no class objects), so loading never depends on how this module was run."""
    with open(path, "wb") as f:
        pickle.dump({"counts": counts.tocsr(), "vocab": vocab, "business_ids": np.asarray(bids)}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_restaurant_bm25(path: Path = config.RESTAURANT_BM25_PATH) -> tuple[SparseBM25, np.ndarray] | None:
    """(BM25 over each restaurant's concatenated reviews, business ids in row order), or None if not built."""
    if not path.exists():
        return None
    with open(path, "rb") as f:  # our own artifact, written by save_restaurant_bm25
        state = pickle.load(f)
    return SparseBM25(state["counts"], state["vocab"]), state["business_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = parser.parse_args()
    run(config.SENTENCES_PATH, config.BM25_PATH, config.DENSE_INDEX_PATH, config.INDEX_IDS_PATH, force=args.force)
    run_restaurant_bm25(config.RAW_REVIEW_PATH, config.BUSINESSES_PATH, config.RESTAURANT_BM25_PATH, force=args.force)


if __name__ == "__main__":
    main()
