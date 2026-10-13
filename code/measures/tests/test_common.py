import os
import sys

import pytest

from common import CATS, copied_ranges, novel_tokens, tokens, word_cats


def test_tokens_apostrophes():
    assert [t[0] for t in tokens("Don’t go, it's 5pm")] == ["don't", "go", "it's", "5pm"]


def test_word_cats():
    assert "article" in word_cats("the")
    assert "ppron" in word_cats("their")            # prefix pattern \btheir
    assert "negate" in word_cats("don't") and "auxverb" in word_cats("don't")
    assert "negate" in word_cats("nobody")
    assert word_cats("dog") == []
    assert set(CATS) == {"ppron", "ipron", "article", "conj", "preps", "auxverb", "adverb", "negate", "quant"}


def test_novel_tokens_drops_copied_4grams():
    prompt = "please tell me how to reset my password today"
    reply = "sure you can tell me how to reset my password by clicking here"
    kept = [t[0] for t in novel_tokens(reply, prompt)]
    assert kept == ["sure", "you", "can", "by", "clicking", "here"]
    assert novel_tokens("tell me how to", prompt) == []


def test_quotes_dropped():
    assert [t[0] for t in novel_tokens('say "the big dog" now', "hello")] == ["say", "now"]


def test_parity_with_detectors_base():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root)
    try:
        from detectors.base import copied_ranges as ref
    except Exception:
        pytest.skip("detectors.base not importable")
    p = "My landlord did not fix the heater last week, and now the apartment is freezing."
    r = 'The landlord did not fix the heater last week and "now the apartment" is bad. Fix it.'
    assert copied_ranges(r, p) == ref(r, p)
