import numpy as np
import pandas as pd
import pytest

from src import config, nli
from src.prep import load_lexicon

TEXTS = [f"sentence number {i}" for i in range(10)]


def test_hypotheses_cover_every_pole_in_lexicon_order():
    hyps = nli.load_hypotheses(load_lexicon())
    assert len(hyps) == 12
    assert hyps[0] == ("noise.quiet", "The atmosphere was quiet.")
    assert hyps[1][0] == "noise.loud"
    assert len({c for c, _ in hyps}) == 12


def test_label_indices_read_from_config():
    assert nli.label_indices({0: "contradiction", 1: "entailment", 2: "neutral"}) == (1, 0)
    assert nli.label_indices({0: "ENTAILMENT", 1: "neutral", 2: "CONTRADICTION"}) == (0, 2)


# --- Checkpointed run with a fake scorer ---------------------------------------------

class FakeScorer:
    """Deterministic scores in [-1, 1]; records which texts it was asked to score."""

    def __init__(self, n_hyps):
        self.n_hyps = n_hyps
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen += list(texts)
        ids = np.array([int(t.split()[-1]) for t in texts], dtype=np.float32)
        return np.tanh(ids[:, None] / 10 - np.arange(self.n_hyps)[None, :] / 12)


@pytest.fixture
def paths(tmp_path):
    sp = tmp_path / "sentences.parquet"
    pd.DataFrame({"sent_id": [f"s{i}" for i in range(len(TEXTS))], "text": TEXTS}).to_parquet(sp, index=False)
    return sp, tmp_path / "polarity.parquet", tmp_path / "chunks"


def run(paths, scorers, **kw):
    def factory(hyps):
        s = FakeScorer(len(hyps))
        scorers.append(s)
        return s

    nli.run(*paths, scorer_factory=factory, chunk_size=3, **kw)


def test_run_writes_aligned_scores(paths):
    scorers = []
    run(paths, scorers)
    out = pd.read_parquet(paths[1])
    assert out["sent_id"].tolist() == [f"s{i}" for i in range(10)]
    assert list(out.columns[1:]) == [c for c, _ in nli.load_hypotheses(load_lexicon())]
    assert out.iloc[:, 1:].dtypes.eq(np.float32).all()
    expected = FakeScorer(12)(TEXTS)
    assert np.allclose(out.iloc[:, 1:].to_numpy(), expected)
    assert len(list(paths[2].glob("chunk_*.parquet"))) == 4  # ceil(10 / 3)


def test_resume_scores_only_missing_chunks(paths):
    scorers = []
    run(paths, scorers)
    paths[1].unlink()
    (paths[2] / "chunk_00001.parquet").unlink()  # simulate a run killed mid-way
    run(paths, scorers)
    assert scorers[-1].seen == TEXTS[3:6]
    assert pd.read_parquet(paths[1])["sent_id"].tolist() == [f"s{i}" for i in range(10)]


def test_existing_output_is_skipped_and_force_recomputes(paths, capsys):
    scorers = []
    run(paths, scorers)
    run(paths, scorers)
    assert "skip:" in capsys.readouterr().out and len(scorers) == 1
    run(paths, scorers, force=True)
    assert len(scorers) == 2 and scorers[-1].seen == TEXTS


def test_stale_chunks_are_rejected(paths):
    scorers = []
    run(paths, scorers)
    paths[1].unlink()
    pd.DataFrame({"sent_id": ["other", "ids", "here"], "text": ["a", "b", "c"]}).to_parquet(paths[0], index=False)
    with pytest.raises(ValueError, match="do not match"):
        run(paths, scorers)


# --- Real model (CPU; downloads nli-deberta-v3-small once) ---------------------------

def test_real_model_signs():
    """Polarity has the right sign for clear support, clear contradiction and negation."""
    hyps = dict(nli.load_hypotheses(load_lexicon()))
    scorer = nli.NLIScorer([hyps["noise.quiet"], hyps["lighting.dim"]], device="cpu")
    p = scorer([
        "It was so quiet we could talk.",
        "We had to shout to hear each other.",
        "It was not loud at all.",
        "Dimly lit with candles on every table.",
        "Bright fluorescent lights everywhere.",
    ])
    assert p.shape == (5, 2) and p.dtype == np.float32
    assert p[0, 0] > 0.5 and p[1, 0] < -0.5 and p[2, 0] > 0
    assert p[3, 1] > 0.5 and p[4, 1] < -0.5
    assert np.all(np.abs(p) <= 1)
