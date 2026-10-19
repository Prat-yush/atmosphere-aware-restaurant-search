"""Step 2: BM25 and dense (MiniLM + FAISS inner product) indexes over kept sentences.

Row i of both indexes is row i of data/proc/index_ids.parquet (sent_id, business_id).

Usage:
    python -m src.index                  # -> bm25.pkl, dense.faiss, index_ids.parquet
    python -m src.index --force          # rebuild even if outputs exist
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections.abc import Callable, Sequence
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = parser.parse_args()
    run(config.SENTENCES_PATH, config.BM25_PATH, config.DENSE_INDEX_PATH, config.INDEX_IDS_PATH, force=args.force)


if __name__ == "__main__":
    main()
