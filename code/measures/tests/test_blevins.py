import math

import pytest

import blevins as bl


def T(words, pos=None):
    pos = pos or ["X"] * len(words)
    return [(w, p, 0, 0) for w, p in zip(words, pos)]


def test_len_and_novelty_hand():
    f = bl.features(T(["the", "cat", "sat"]), T(["the", "dog"]))
    assert math.isclose(f["blevins_len"], 1 - 1 / (5 + 1e-5))
    assert math.isclose(f["blevins_novtok"], 2 / 3)          # cat, sat are novel vs prompt
    assert math.isclose(f["blevins_tokoverlap"], 1 / 3)
    assert math.isclose(f["blevins_conv"], (f["blevins_len"] + 1 / 3) / 2)


def test_novelty_asymmetric():
    a, b = T(["a", "b", "c", "d"]), T(["a", "b"])
    assert bl.features(a, b)["blevins_novtok"] == 0.5
    assert bl.features(b, a)["blevins_novtok"] == 0.0


def test_propn():
    trg = T(["Alice", "met", "Bob"], ["PROPN", "VERB", "PROPN"])
    ref = T(["bob", "and", "Carl"], ["PROPN", "CCONJ", "PROPN"])
    f = bl.features(trg, ref)
    assert f["blevins_propn"] == 1 and f["blevins_propn_frac"] == 0.5


def test_liwc_negation_contraction():
    f = bl.features(T(["do", "n't"]), T(["do", "n't"]))
    assert f["blevins_liwc"] > 0.999 and f["blevins_liwc_negate"] > 0.999


def test_empty_nan():
    assert math.isnan(bl.features([], T(["a"]))["blevins_conv"])


def test_spacy_end_to_end():
    pytest.importorskip("spacy")
    try:
        nlp = bl.load_nlp()
    except OSError:
        pytest.skip("en_core_web_sm missing")
    s = bl.score("Tell Sam how to fix the heater.", "Sam should call the landlord.", nlp=nlp)
    assert s["blevins_propn"] == 1
    s2 = bl.score("please tell me how to reset my password today", "tell me how to reset my password", nlp=nlp, novel=True)
    assert math.isnan(s2["blevins_conv"])          # all reply tokens copied
