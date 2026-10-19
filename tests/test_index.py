import numpy as np
import pandas as pd
import pytest

from src import config, index

TEXTS = [
    "It was so quiet we could hear each other.",
    "The music was way too loud.",
    "Dimly lit and very romantic.",
    "Packed on a Friday night, great energy.",
]
VOCAB = ["quiet", "loud", "romantic", "energy", "music", "dimly"]


def fake_encoder(texts):
    """Normalized bag-of-vocab vectors: deterministic, no model download."""
    rows = []
    for t in texts:
        toks = index.tokenize(t)
        v = np.array([toks.count(w) for w in VOCAB], dtype=np.float32) + 1e-3
        rows.append(v / np.linalg.norm(v))
    return np.stack(rows)


# --- Tokenizer -------------------------------------------------------------------

def test_tokenize_lowercases_and_keeps_contractions():
    assert index.tokenize("It WASN'T loud, isn’t it?") == ["it", "wasn't", "loud", "isn't", "it"]


def test_tokenize_splits_hyphens_and_drops_punctuation():
    assert index.tokenize("dimly-lit... 5-star!") == ["dimly", "lit", "5", "star"]


# --- BM25 --------------------------------------------------------------------------

def test_bm25_ranks_matching_sentence_first():
    bm25 = index.build_bm25(TEXTS)
    scores = index.bm25_scores(bm25, "romantic")
    assert scores.shape == (len(TEXTS),)
    assert int(np.argmax(scores)) == 2


def test_bm25_unknown_query_scores_zero():
    bm25 = index.build_bm25(TEXTS)
    assert np.all(index.bm25_scores(bm25, "zzzz") == 0)


# --- Dense -------------------------------------------------------------------------

def test_dense_search_returns_cosine_order():
    dense = index.build_dense(fake_encoder(TEXTS))
    scores, ids = index.dense_search(dense, fake_encoder(["loud music"])[0], top_n=4)
    assert ids[0] == 1
    assert np.all(np.diff(scores) <= 0)
    assert scores[0] == pytest.approx(1.0, abs=1e-3)


def test_dense_search_caps_at_corpus_size():
    dense = index.build_dense(fake_encoder(TEXTS))
    _, ids = index.dense_search(dense, fake_encoder(["quiet"])[0], top_n=1000)
    assert sorted(ids.tolist()) == [0, 1, 2, 3]


def test_dense_search_matches_numpy_brute_force():
    rng = np.random.default_rng(config.SEED)
    emb = rng.normal(size=(500, 32)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[7] + 0.1 * rng.normal(size=32).astype(np.float32)
    q /= np.linalg.norm(q)
    scores, ids = index.dense_search(index.build_dense(emb), q, top_n=20)
    expected = np.argsort(-(emb @ q))[:20]
    assert ids.tolist() == expected.tolist()
    assert np.allclose(scores, (emb @ q)[expected], atol=1e-5)


def test_build_dense_rejects_unnormalized():
    with pytest.raises(ValueError):
        index.build_dense(np.ones((2, 4), dtype=np.float32))


# --- run / persistence -------------------------------------------------------------

@pytest.fixture
def paths(tmp_path):
    sents = pd.DataFrame(
        {
            "sent_id": [f"r{i}_0" for i in range(len(TEXTS))],
            "business_id": ["a", "a", "b", "c"],
            "review_id": [f"r{i}" for i in range(len(TEXTS))],
            "text": TEXTS,
            "matched_aspects": [["noise"], ["noise", "music"], ["romance", "lighting"], ["energy"]],
        }
    )
    sp = tmp_path / "sentences.parquet"
    sents.to_parquet(sp, index=False)
    return sp, tmp_path / "bm25.pkl", tmp_path / "dense.faiss", tmp_path / "ids.parquet"


def test_run_roundtrip_keeps_rows_aligned(paths):
    sp, *out = paths
    index.run(sp, *out, encoder=fake_encoder)
    bm25, dense, ids = index.load_indexes(*out)
    assert ids["sent_id"].tolist() == ["r0_0", "r1_0", "r2_0", "r3_0"]
    assert ids["business_id"].tolist() == ["a", "a", "b", "c"]
    assert int(np.argmax(index.bm25_scores(bm25, "romantic"))) == 2
    _, top = index.dense_search(dense, fake_encoder(["romantic"])[0], 1)
    assert top.tolist() == [2]


def test_run_is_idempotent_and_force_rebuilds(paths, capsys):
    sp, *out = paths
    index.run(sp, *out, encoder=fake_encoder)
    mtimes = [p.stat().st_mtime_ns for p in out]
    index.run(sp, *out, encoder=fake_encoder)
    assert "skip:" in capsys.readouterr().out
    assert [p.stat().st_mtime_ns for p in out] == mtimes
    index.run(sp, *out, encoder=fake_encoder, force=True)
    assert all(p.stat().st_mtime_ns != m for p, m in zip(out, mtimes))


def test_load_indexes_detects_misalignment(paths):
    sp, bm25_p, dense_p, ids_p = paths
    index.run(sp, bm25_p, dense_p, ids_p, encoder=fake_encoder)
    pd.read_parquet(ids_p).head(2).to_parquet(ids_p, index=False)
    with pytest.raises(ValueError, match="mismatch"):
        index.load_indexes(bm25_p, dense_p, ids_p)


# --- Real model (downloads all-MiniLM-L6-v2 once, ~90 MB) ----------------------------

def test_real_encoder_is_normalized_and_semantic():
    enc = index.make_encoder(index.load_encoder(config.DENSE_MODEL, device="cpu"))
    emb = enc(TEXTS)
    assert emb.shape == (4, 384) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    _, ids = index.dense_search(index.build_dense(emb), enc(["a peaceful place to talk"])[0], 1)
    assert ids.tolist() == [0]


def test_faiss_stays_exact_across_torch_calls():
    """Guards the macOS libomp workaround in src/__init__.py: faiss, then torch, then faiss."""
    rng = np.random.default_rng(config.SEED)
    emb = rng.normal(size=(2000, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    dense = index.build_dense(emb)
    before = index.dense_search(dense, emb[3], 50)
    enc = index.make_encoder(index.load_encoder(config.DENSE_MODEL, device="cpu"))
    q = enc(["quiet romantic spot"])[0]
    after = index.dense_search(dense, q, 50)
    for (scores, ids), query in ((before, emb[3]), (after, q)):
        exact = emb @ query
        assert ids.tolist() == np.argsort(-exact)[:50].tolist()
        assert np.allclose(scores, exact[ids], atol=1e-5)


# --- Restaurant-level sparse BM25 (bm25_baseline) -----------------------------------

DOCS = [
    "quiet romantic place with dim lighting",
    "loud sports bar loud music",
    "the the the place place",
    "cozy quiet cafe the best",
    "romantic dinner the place",
]


def test_sparse_bm25_matches_rank_bm25():
    from rank_bm25 import BM25Okapi

    reviews = [{"business_id": f"b{i}", "text": t} for i, t in enumerate(DOCS)]
    counts, vocab = index.build_restaurant_counts(reviews, [f"b{i}" for i in range(len(DOCS))], flush_every=2)
    ours = index.SparseBM25(counts, vocab)
    ref = BM25Okapi([index.tokenize(t) for t in DOCS])
    # "the"/"place" appear in >half the docs (negative idf -> epsilon floor); "loud loud" repeats; "zzz" unknown
    for q in ["quiet", "romantic place", "the place", "loud loud", "zzz", "quiet zzz the", ""]:
        assert np.allclose(ours.get_scores(index.tokenize(q)), ref.get_scores(index.tokenize(q))), q


def test_restaurant_counts_concatenate_reviews_per_business():
    reviews = [
        {"business_id": "b", "text": "quiet place"},
        {"business_id": "a", "text": "loud"},
        {"business_id": "b", "text": "quiet again"},
    ]
    counts, vocab = index.build_restaurant_counts(reviews, ["a", "b"], flush_every=1)
    assert counts.shape == (2, 4)
    assert counts[1, vocab["quiet"]] == 2 and counts[0, vocab["loud"]] == 1
    assert counts[0, vocab["quiet"]] == 0


def test_restaurant_bm25_save_load_roundtrip(tmp_path):
    """Stored as plain data, so it loads no matter which module wrote it (was a __main__ pickle bug)."""
    reviews = [{"business_id": f"b{i}", "text": t} for i, t in enumerate(DOCS)]
    bids = [f"b{i}" for i in range(len(DOCS))]
    counts, vocab = index.build_restaurant_counts(reviews, bids)
    path = tmp_path / "bm25_restaurants.pkl"
    index.save_restaurant_bm25(counts, vocab, np.array(bids), path)
    bm25, loaded_ids = index.load_restaurant_bm25(path)
    assert loaded_ids.tolist() == bids
    q = index.tokenize("quiet romantic place")
    assert np.allclose(bm25.get_scores(q), index.SparseBM25(counts, vocab).get_scores(q))
    assert index.load_restaurant_bm25(tmp_path / "missing.pkl") is None
