import numpy as np
import pytest

from src import config, rank


# --- Reciprocal rank fusion --------------------------------------------------------------

def test_rrf_hand_computed():
    fused = rank.rrf_fuse([np.array([7, 3]), np.array([3, 9])], k_rrf=60)
    assert fused[3] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[7] == pytest.approx(1 / 61)
    assert fused[9] == pytest.approx(1 / 62)


def test_top_fused_orders_by_score_then_row():
    fused = {5: 0.1, 2: 0.3, 9: 0.1, 1: 0.2}
    assert rank.top_fused(fused, 3) == [(2, 0.3), (1, 0.2), (5, 0.1)]


def test_group_top_k_keeps_k_best_per_restaurant():
    business = np.array(["a", "a", "b", "a", "b"])
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
    groups = rank.group_top_k(ranked, business, k=2)
    assert groups == {"a": [(0, 0.9), (1, 0.8)], "b": [(2, 0.7), (4, 0.5)]}


# --- Aggregation math -------------------------------------------------------------------

@pytest.mark.parametrize("sims", [[1.0], [0.2, 0.5, 0.3], [1 / 61, 1 / 62, 1 / 90], np.random.default_rng(0).random(10) + 0.01])
def test_weights_sum_to_one(sims):
    w = rank.evidence_weights(np.array(sims))
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0)


def test_weights_reject_nonpositive_total():
    with pytest.raises(ValueError):
        rank.evidence_weights(np.array([0.0, 0.0]))


def test_raw_score_is_weighted_mean_in_range():
    assert rank.raw_score(np.array([1.0, 3.0]), np.array([1.0, -1.0])) == pytest.approx(-0.5)
    assert rank.raw_score(np.array([2.0, 2.0]), np.array([0.4, 0.4])) == pytest.approx(0.4)
    rng = np.random.default_rng(1)
    for _ in range(100):
        r = rank.raw_score(rng.random(8) + 1e-3, rng.uniform(-1, 1, 8))
        assert -1 <= r <= 1


def test_shrink_hand_computed():
    # (n*raw + m*mu) / (n + m) = (2*0.9 + 4*0.1) / 6
    assert rank.shrink(0.9, n=2, m=4, mu=0.1) == pytest.approx(2.2 / 6)


def test_shrinkage_pulls_low_n_toward_mu():
    mu, raw, m = 0.1, 0.9, 4
    scores = [rank.shrink(raw, n, m, mu) for n in range(1, 11)]
    assert all(mu < s < raw for s in scores)
    assert scores == sorted(scores)  # more evidence -> closer to raw
    assert abs(scores[0] - mu) < abs(scores[-1] - mu)
    assert rank.shrink(raw, 0, m, mu) == pytest.approx(mu)


def test_many_matching_sentences_do_not_beat_few_with_equal_mean_polarity():
    """1000 retrieved sentences vs 10, same polarity: the k cap makes n equal, so scores tie."""
    k, m, mu = config.K_PER_RESTAURANT, 4, 0.0
    business = np.array(["big"] * 1000 + ["small"] * 10)
    sims = np.linspace(1.0, 0.01, len(business))
    order = np.argsort(-sims, kind="stable")
    ranked = [(int(i), float(sims[i])) for i in order]
    polarity = np.full(len(business), 0.6)

    groups = rank.group_top_k(ranked, business, k)
    assert len(groups["big"]) == len(groups["small"]) == 10
    scores = {}
    for bid, ev in groups.items():
        rows = np.array([r for r, _ in ev])
        s = np.array([x for _, x in ev])
        scores[bid] = rank.shrink(rank.raw_score(s, polarity[rows]), len(rows), m, mu)
    assert scores["big"] == pytest.approx(scores["small"])


def test_restaurant_size_does_not_change_raw_score():
    """Normalized weights: duplicating a restaurant's evidence leaves raw unchanged."""
    sims, pol = np.array([0.5, 0.3, 0.2]), np.array([0.8, -0.2, 0.5])
    assert rank.raw_score(np.tile(sims, 3), np.tile(pol, 3)) == pytest.approx(rank.raw_score(sims, pol))
