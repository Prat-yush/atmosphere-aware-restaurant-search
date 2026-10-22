import pandas as pd
import pytest

from src import judge


# --- Yelp attribute parsing -------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("u'quiet'", "quiet"),
        ("'average'", "average"),
        ("True", True),
        ("{'romantic': True, 'intimate': False, 'classy': None}", {"romantic": True, "intimate": False}),
        ("not a literal", "not a literal"),
        (None, None),
    ],
)
def test_parse_attribute(raw, expected):
    assert judge.parse_attribute(raw) == expected


# --- Topic sentence matching ---------------------------------------------------------------

def test_topic_sentences_splits_and_tags():
    text = "It was so quiet we could talk. The live band was great!\nTiny."
    out = judge.topic_sentences(text)
    assert out["noise"] == ["It was so quiet we could talk."]
    assert out["music"] == ["The live band was great!"]
    assert "Tiny." not in sum(out.values(), [])  # under 20 characters


def test_topic_keywords_are_whole_words():
    assert judge.topic_sentences("The cloudy weather was depressing today.") == {}


# --- Cohen's kappa, hand-computed ---------------------------------------------------------------

def test_kappa_hand_computed():
    # observed agreement 3/4; expected 0.5*0.25 + 0.25*0.5 + 0.25*0.25 = 0.3125
    a, b = [0, 0, 1, 2], [0, 1, 1, 2]
    assert judge.cohen_kappa(a, b) == pytest.approx((0.75 - 0.3125) / (1 - 0.3125))


def test_quadratic_kappa_hand_computed():
    # weighted observed disagreement 0.0625; weighted expected 0.3125
    a, b = [0, 0, 1, 2], [0, 1, 1, 2]
    assert judge.cohen_kappa(a, b, weights="quadratic") == pytest.approx(1 - 0.0625 / 0.3125)


def test_kappa_perfect_and_constant():
    assert judge.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert judge.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # no expected disagreement


def test_quadratic_kappa_penalizes_far_disagreement_more():
    near = judge.cohen_kappa([0, 1, 2, 2, 0], [1, 1, 2, 2, 0], weights="quadratic")
    far = judge.cohen_kappa([0, 1, 2, 2, 0], [2, 1, 2, 2, 0], weights="quadratic")
    assert near > far


# --- Human check and agreement -----------------------------------------------------------------

POOL = pd.DataFrame({
    "qid": ["t09", "t09", "t17", "t01"],
    "query": ["q9", "q9", "q17", "q1"],
    "business_id": ["a", "b", "a", "c"],
    "name": ["A", "B", "A", "C"],
    "address": "", "postal_code": "", "price": "", "categories": "",
    "grade": ["2", "0", "1", "2"],
    "judge_reason": ["r1", "r2", "r3", "r4"],
})


def test_human_check_is_blind_and_limited_to_check_queries():
    rows = judge.human_check_rows(POOL, qids=("t09", "t17"))
    assert rows["qid"].tolist() == ["t09", "t09", "t17"]
    assert "judge_reason" not in rows.columns
    assert (rows["grade"] == "").all()  # LLM grades never shown to the human


def test_agreement_joins_on_query_and_business():
    human = pd.DataFrame({"qid": ["t09", "t09", "t17"], "business_id": ["a", "b", "a"], "grade": ["2", "1", ""]})
    res = judge.agreement(human, POOL[["qid", "business_id", "grade"]])
    assert res["n"] == 2  # the ungraded row is skipped
    assert res["exact"] == pytest.approx(0.5)
    assert res["rows"]["qid"].tolist() == ["t09", "t09"]
