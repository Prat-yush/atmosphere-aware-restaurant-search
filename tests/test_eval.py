import json
import math

import numpy as np
import pandas as pd
import pytest

from src import eval as ev


# --- nDCG@10, hand-computed -------------------------------------------------------------

def test_ndcg_perfect_ranking_is_one():
    grades = {"a": 2, "b": 1, "c": 0}
    assert ev.ndcg_at_k(["a", "b", "c"], grades) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # ranked gains [0, 2, 1]: DCG = 0/log2(2) + 2/log2(3) + 1/log2(4) = 1.26186 + 0.5
    # ideal [2, 1, 0]:        IDCG = 2/1 + 1/log2(3) = 2.63093
    grades = {"a": 2, "b": 1, "x": 0}
    expected = (2 / math.log2(3) + 1 / 2) / (2 + 1 / math.log2(3))
    assert ev.ndcg_at_k(["x", "a", "b"], grades) == pytest.approx(expected)
    assert expected == pytest.approx(0.6696, abs=1e-4)


def test_ndcg_unjudged_counts_zero_and_cutoff_applies():
    grades = {"rel": 1}
    ranked = [f"u{i}" for i in range(10)] + ["rel"]  # relevant item at rank 11
    assert ev.ndcg_at_k(ranked, grades, k=10) == 0.0
    assert ev.ndcg_at_k(["rel"], grades, k=10) == pytest.approx(1.0)


def test_ndcg_ideal_uses_all_judged_not_just_retrieved():
    grades = {"a": 1, "b": 2}  # b is never retrieved
    assert ev.ndcg_at_k(["a"], grades) == pytest.approx(1 / (2 + 1 / math.log2(3)))


def test_ndcg_no_relevant_is_zero():
    assert ev.ndcg_at_k(["a", "b"], {"a": 0}) == 0.0
    assert ev.ndcg_at_k([], {}) == 0.0


# --- P@5 -----------------------------------------------------------------------------------

def test_precision_at_5_hand_computed():
    grades = {"a": 2, "b": 0, "c": 1, "d": 0, "e": 1, "f": 2}
    assert ev.precision_at_k(["a", "b", "c", "d", "e", "f"], grades) == pytest.approx(3 / 5)


def test_precision_short_list_divides_by_k():
    assert ev.precision_at_k(["a"], {"a": 1}) == pytest.approx(1 / 5)
    assert ev.precision_at_k([], {"a": 1}) == 0.0


def test_precision_threshold():
    assert ev.precision_at_k(["a", "b"], {"a": 1, "b": 2}, k=2, min_grade=2) == pytest.approx(0.5)


# --- Popularity --------------------------------------------------------------------------------

def test_spearman_sign():
    counts = {"a": 500, "b": 300, "c": 100, "d": 50}
    assert ev.spearman_rank_vs_popularity(["a", "b", "c", "d"], counts) == pytest.approx(-1.0)
    assert ev.spearman_rank_vs_popularity(["d", "c", "b", "a"], counts) == pytest.approx(1.0)
    assert math.isnan(ev.spearman_rank_vs_popularity(["a", "b"], counts))


# --- Queries, qrels, pooling ------------------------------------------------------------------------

def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_queries_validates(tmp_path):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [{"qid": "t1", "text": "quiet", "category": "single"}])
    assert ev.load_queries(p)[0]["qid"] == "t1"
    write_jsonl(p, [{"qid": "t1", "text": "quiet", "category": "weird"}])
    with pytest.raises(ValueError, match="category"):
        ev.load_queries(p)
    write_jsonl(p, [{"qid": "t1", "text": "a", "category": "single"}, {"qid": "t1", "text": "b", "category": "mixed"}])
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_queries(p)


def test_pool_is_union_of_top_depth_and_blind():
    queries = [{"qid": "q1", "text": "quiet", "category": "single"}]
    runs = {"full": {"q1": ["a", "b", "z"]}, "bm25_baseline": {"q1": ["b", "c"]}, "no_polarity": {"q1": []}}
    biz = pd.DataFrame({
        "name": ["Zed", "Alpha", "Bravo", "Charlie"], "address": "", "postal_code": "", "price": [2, np.nan, 1, 3],
        "categories": "",
    }, index=pd.Index(["z", "a", "b", "c"], name="business_id"))
    pool = ev.build_pool(runs, queries, biz, depth=2)
    assert pool["business_id"].tolist() == ["a", "b", "c"]  # z is rank 3 in full -> outside depth 2
    assert pool["name"].tolist() == ["Alpha", "Bravo", "Charlie"]  # sorted by name, not by system rank
    assert not {"system", "rank", "score"} & set(pool.columns)
    assert pool["price"].tolist() == ["", "$", "$$$"]


def test_grades_roundtrip_to_qrels(tmp_path, capsys):
    pool = pd.DataFrame({"qid": ["q1", "q1", "q2"], "business_id": ["a", "b", "c"], "grade": ["2", "", "0"]})
    qrels = ev.grades_to_qrels(pool)
    assert qrels.values.tolist() == [["q1", "a", 2], ["q2", "c", 0]]
    assert "1 pooled rows have no grade" in capsys.readouterr().out
    path = tmp_path / "qrels.tsv"
    qrels.to_csv(path, sep="\t", index=False)
    assert ev.load_qrels(path) == {"q1": {"a": 2}, "q2": {"c": 0}}
    with pytest.raises(ValueError):
        ev.grades_to_qrels(pd.DataFrame({"qid": ["q"], "business_id": ["a"], "grade": ["3"]}))


def test_evaluate_and_summarize():
    queries = [{"qid": "q1", "text": "", "category": "single"}, {"qid": "q2", "text": "", "category": "negation"}]
    qrels = {"q1": {"a": 1}, "q2": {"b": 2}}
    runs = {"full": {"q1": ["a"], "q2": ["x", "b"]}, "bm25_baseline": {"q1": ["x"], "q2": []}}
    per_query = ev.evaluate(runs, queries, qrels)
    full = per_query[per_query.system == "full"].set_index("qid")
    assert full.loc["q1", "ndcg@10"] == pytest.approx(1.0)
    assert full.loc["q2", "ndcg@10"] == pytest.approx(1 / math.log2(3))
    summary = ev.summarize(per_query)
    row = summary[(summary.system == "full") & (summary.category == "all")].iloc[0]
    assert row["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 2) and row["n"] == 2
    assert summary["category"].tolist()[:2] == ["all", "all"]


# --- Polarity sample and accuracy ---------------------------------------------------------------------

LEX = {"aspects": {
    "noise": {"positive": {"id": "quiet"}, "negative": {"id": "loud"}},
    "music": {"positive": {"id": "good_music"}, "negative": {"id": "loud_music"}},
}}


def test_sample_is_stratified_and_deterministic():
    sents = pd.DataFrame({
        "sent_id": [f"s{i}" for i in range(40)],
        "text": [f"t{i}" for i in range(40)],
        "matched_aspects": [["noise"] if i < 30 else ["music"] for i in range(40)],
    })
    a = ev.sample_polarity(sents, LEX, n=7)
    b = ev.sample_polarity(sents, LEX, n=7)
    assert a.equals(b)
    assert a["aspect"].value_counts().to_dict() == {"noise": 4, "music": 3}
    assert a["sample_id"].tolist() == list(range(1, 8))
    assert (a["label"] == "").all()


@pytest.mark.parametrize("p_pos, p_neg, expected", [(0.8, -0.9, "pos"), (-0.7, 0.6, "neg"), (-0.2, -0.5, "none"), (0.3, 0.3, "pos"), (0.0, 0.0, "none")])
def test_predict_label(p_pos, p_neg, expected):
    assert ev.predict_label(p_pos, p_neg) == expected


def test_polarity_accuracy():
    labeled = pd.DataFrame({
        "sent_id": ["s1", "s2", "s3", "s4"], "aspect": "noise", "pos_pole": "quiet", "neg_pole": "loud",
        "label": ["pos", "neg", "none", ""],
    })
    pol = pd.DataFrame({"sent_id": ["s1", "s2", "s3", "s4"], "noise.quiet": [0.9, 0.5, -0.1, 0.0], "noise.loud": [-0.9, -0.2, -0.3, 0.0]})
    res = ev.polarity_accuracy(labeled, pol)
    # s1 pos->pos (ok), s2 neg->pos (wrong), s3 none->none (ok), s4 unlabeled (skipped)
    assert res["n_labeled"] == 3
    assert res["accuracy_3way"] == pytest.approx(2 / 3)
    assert res["accuracy_on_pos_neg"] == pytest.approx(1 / 2)


# --- Unjudged results ------------------------------------------------------------------------------

def test_judged_only_and_fraction():
    grades = {"a": 2, "b": 0}
    ranked = ["x", "a", "y", "b"]
    assert ev.judged_only(ranked, grades) == ["a", "b"]
    assert ev.judged_fraction(ranked, grades, k=4) == pytest.approx(0.5)
    assert ev.judged_fraction([], grades) == 0.0
    # an unjudged result above a relevant one costs nDCG, but not after condensing
    assert ev.ndcg_at_k(ranked, grades) < 1.0
    assert ev.ndcg_at_k(ev.judged_only(ranked, grades), grades) == pytest.approx(1.0)
