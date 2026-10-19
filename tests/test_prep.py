import json

import pandas as pd
import pytest

from src import prep


@pytest.fixture(scope="module")
def lexicon():
    return prep.load_lexicon()


@pytest.fixture(scope="module")
def matchers(lexicon):
    return prep.build_matchers(lexicon)


@pytest.fixture(scope="module")
def exclusion(lexicon):
    return prep.build_exclusion(lexicon)


@pytest.fixture(scope="module")
def nlp():
    return prep.load_sentencizer()


# --- Lexicon -----------------------------------------------------------------

def test_lexicon_shape(lexicon):
    assert list(lexicon["aspects"]) == ["noise", "energy", "space", "romance", "lighting", "music"]
    ids = []
    for poles in lexicon["aspects"].values():
        assert set(poles) == {"positive", "negative"}
        for pole in poles.values():
            assert pole["hypothesis"].strip()
            assert 8 <= len(pole["terms"]) <= 15
            assert len(set(pole["terms"])) == len(pole["terms"])
            ids.append(pole["id"])
    assert len(set(ids)) == 12


# --- Matching ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("It was so LOUD in there.", ["noise"]),
        ("Nice and quiet, we could hear each other.", ["noise"]),
        ("We had to   shout to be heard.", ["noise"]),
        ("A low lighting, romantic spot.", ["romance", "lighting"]),
        ("Dimly-lit and cozy.", ["space", "lighting"]),
        ("The music was too loud.", ["noise", "music"]),
        ("Couldn’t hear a thing.", ["noise"]),  # curly apostrophe
    ],
)
def test_match_aspects_hits(text, expected, matchers, exclusion):
    assert prep.match_aspects(text, matchers, exclusion) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Cloudy day, great pho.",          # 'loud' inside a word
        "The dimension of flavors.",       # 'dim' inside a word
        "Best dim sum in Chinatown.",      # excluded phrase
        "The sea bass was perfect.",       # excluded phrase
        "Burrito packed with flavor.",     # excluded phrase
        "He packed up our leftovers.",     # excluded phrase
        "I was dead wrong about this place.",
        "Fresh squeezed orange juice.",
        "Two eggs sunny side up.",
        "We drove to Mt. Airy for dinner.",
        "The waffle was so airy.",
        "The crepe was mind-blasting.",
        "Quick lunch before hopping on a flight.",
        "They sell birthday candles too.",
        "Sauce to jazz them up.",
        "The food was delicious.",
    ],
)
def test_match_aspects_misses(text, matchers, exclusion):
    assert prep.match_aspects(text, matchers, exclusion) == []


def test_exclusion_only_removes_the_phrase(matchers, exclusion):
    assert prep.match_aspects("Dim sum was great but it was loud.", matchers, exclusion) == ["noise"]


# --- Business filter ------------------------------------------------------------

def biz(**kw):
    base = {"business_id": "b", "city": "Philadelphia", "categories": "Restaurants, Bars", "review_count": 20}
    return base | kw


@pytest.mark.parametrize(
    "record, keep",
    [
        (biz(), True),
        (biz(city="  philadelphia "), True),
        (biz(city="PHILADELPHIA"), True),
        (biz(city="Phila delphia"), False),
        (biz(city="Philadelphia PA"), False),
        (biz(review_count=19), False),
        (biz(categories="Bars, Nightlife"), False),
        (biz(categories="Restaurants Supply"), False),
        (biz(categories=None), False),
        (biz(city=None), False),
    ],
)
def test_is_target_business(record, keep):
    assert prep.is_target_business(record) is keep


def test_parse_price():
    assert prep.parse_price({"attributes": {"RestaurantsPriceRange2": "2"}}) == 2
    assert prep.parse_price({"attributes": {"RestaurantsPriceRange2": "None"}}) is None
    assert prep.parse_price({"attributes": None}) is None


# --- Sentence splitting -----------------------------------------------------------

def test_sentences_split_on_newlines_and_punctuation(nlp):
    review = {"review_id": "r1", "text": "Great place\nSo quiet. Loved it!\n\n  Will return  "}
    out = [(i, t) for _, i, t in prep.iter_review_sentences([review], nlp)]
    assert out == [(0, "Great place"), (1, "So quiet."), (2, "Loved it!"), (3, "Will return")]


def test_sentence_index_restarts_per_review(nlp):
    reviews = [{"review_id": "a", "text": "One. Two."}, {"review_id": "b", "text": "Three."}]
    out = [(r["review_id"], i) for r, i, _ in prep.iter_review_sentences(reviews, nlp)]
    assert out == [("a", 0), ("a", 1), ("b", 0)]


# --- End to end -----------------------------------------------------------------

def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture
def raw_files(tmp_path):
    businesses = [
        biz(business_id="keep1"),
        biz(business_id="keep2", attributes={"RestaurantsPriceRange2": "3"}),
        biz(business_id="tampa", city="Tampa"),
    ]
    reviews = [
        {"review_id": "r1", "business_id": "keep1", "text": "Very loud. Food fine."},
        {"review_id": "r2", "business_id": "tampa", "text": "So quiet and romantic."},
        {"review_id": "r3", "business_id": "keep2", "text": "Tasty.\nCozy and dimly lit room."},
        {"review_id": "r4", "business_id": "keep1", "text": "Quiet night."},
    ]
    write_jsonl(tmp_path / "business.json", businesses)
    write_jsonl(tmp_path / "review.json", reviews)
    return tmp_path


def run_prep(d, **kw):
    return prep.run(
        d / "business.json", d / "review.json", d / "proc" / "biz.parquet", d / "proc" / "sent.parquet", **kw
    )


def test_run_end_to_end(raw_files):
    out = run_prep(raw_files)
    assert list(out.columns) == prep.SENTENCE_COLUMNS
    assert out["sent_id"].tolist() == ["r1_0", "r3_1", "r4_0"]
    assert out["matched_aspects"].map(list).tolist() == [["noise"], ["space", "lighting"], ["noise"]]
    assert set(out["business_id"]) == {"keep1", "keep2"}

    on_disk = pd.read_parquet(raw_files / "proc" / "sent.parquet")
    assert on_disk["sent_id"].tolist() == out["sent_id"].tolist()
    biz_df = pd.read_parquet(raw_files / "proc" / "biz.parquet")
    assert biz_df["business_id"].tolist() == ["keep1", "keep2"]
    assert biz_df["price"].tolist()[1] == 3


def test_run_max_reviews_counts_only_target_reviews(raw_files):
    out = run_prep(raw_files, max_reviews=2)
    assert set(out["review_id"]) == {"r1", "r3"}


def test_run_is_idempotent_and_force_recomputes(raw_files, capsys):
    run_prep(raw_files)
    sent_path = raw_files / "proc" / "sent.parquet"
    mtime = sent_path.stat().st_mtime_ns
    run_prep(raw_files)
    assert "skip:" in capsys.readouterr().out
    assert sent_path.stat().st_mtime_ns == mtime
    run_prep(raw_files, force=True)
    assert sent_path.stat().st_mtime_ns != mtime
