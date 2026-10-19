import pytest

from src.parse import parse_query


def poles(query):
    return [(t.aspect, t.target_pole) for t in parse_query(query)]


# --- Single aspect -------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("quiet", [("noise", "quiet")]),
        ("LOUD", [("noise", "loud")]),
        ("a romantic spot", [("romance", "romantic")]),
        ("dimly-lit bar", [("lighting", "dim")]),
        ("somewhere cozy", [("space", "cozy")]),
        ("place with live music", [("music", "good_music")]),
    ],
)
def test_single_aspect(query, expected):
    assert poles(query) == expected


def test_longest_match_wins():
    # "loud music" is a music term; it must not also produce a noise target
    assert poles("loud music") == [("music", "loud_music")]
    assert poles("dimly lit") == [("lighting", "dim")]


# --- Negation --------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("not loud", [("noise", "quiet")]),
        ("isn't loud", [("noise", "quiet")]),
        ("wasn't too noisy", [("noise", "quiet")]),
        ("never boring", [("energy", "lively")]),
        ("no loud music", [("music", "good_music")]),
        ("without bright lights", [("lighting", "dim")]),
        ("not cramped", [("space", "cozy")]),
        ("isnt romantic", [("romance", "unromantic")]),  # apostrophe-less cue
        ("not quiet", [("noise", "loud")]),  # flips positive to negative too
    ],
)
def test_negation_flips_pole(query, expected):
    assert poles(query) == expected


def test_negation_marks_target():
    (t,) = parse_query("not boring")
    assert t.negated and t.term == "boring" and t.side == "positive"


def test_negated_lexicon_term_is_not_flipped_again():
    # "not romantic" is itself a negative-pole term; its own "not" is not a cue
    (t,) = parse_query("not romantic")
    assert (t.target_pole, t.negated) == ("unromantic", False)
    assert poles("not too loud") == [("noise", "quiet")]


def test_cue_outside_window_does_not_flip():
    # cue is 5 tokens before the term: "not a place that is loud"
    assert poles("not a place that is loud") == [("noise", "loud")]


def test_cue_does_not_cross_clause_boundary():
    assert poles("not loud but lively") == [("noise", "quiet"), ("energy", "lively")]
    assert poles("not fancy, just cozy") == [("space", "cozy")]


def test_cue_binds_to_nearest_term_only():
    # documented limitation: the second term in a negated list is not flipped
    assert poles("not loud or crowded") == [("noise", "quiet"), ("space", "cramped")]


def test_one_cue_per_term():
    assert poles("not loud, not boring") == [("noise", "quiet"), ("energy", "lively")]


# --- Mixed queries -------------------------------------------------------------------

def test_mixed_query_from_spec():
    assert poles("quiet but not boring, good for a first date") == [
        ("noise", "quiet"),
        ("energy", "lively"),
        ("romance", "romantic"),
    ]


def test_and_query_gives_multiple_targets():
    assert poles("lively and romantic with dim lighting") == [
        ("energy", "lively"),
        ("romance", "romantic"),
        ("lighting", "dim"),
    ]


def test_duplicate_targets_are_dropped():
    assert poles("quiet and peaceful") == [("noise", "quiet")]


def test_contrasting_poles_of_one_aspect_are_kept():
    assert poles("lively but not loud") == [("energy", "lively"), ("noise", "quiet")]
    assert poles("quiet, not loud") == [("noise", "quiet")]  # same target twice -> once


# --- No aspect ---------------------------------------------------------------------

@pytest.mark.parametrize("query", ["best cheesesteak in town", "", "   ", "cheap tacos open late"])
def test_no_aspect_query(query):
    assert parse_query(query) == []


def test_excluded_phrase_is_not_an_aspect():
    assert parse_query("good dim sum") == []
    assert poles("dim sum place that is quiet") == [("noise", "quiet")]


def test_column_name():
    (t,) = parse_query("cozy")
    assert t.column == "space.cozy"
