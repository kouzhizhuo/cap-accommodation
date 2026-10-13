import math

from lsm import lsm_c, score


def test_identical_is_one():
    s = score("I do not like the cat", "I do not like the cat")
    assert abs(s["lsm"] - 1) < 1e-3


def test_hand_example():
    # prompt: 3 words, all articles (100%); reply: 2 words, 1 article (50%), 'dog' in no category
    s = score("the the the", "a dog")
    assert s["p_article"] == 100 and s["r_article"] == 50
    assert math.isclose(s["lsm_article"], 1 - 50 / (150 + 1e-4))
    # every other category has p = r = 0 -> LSM_c = 1 - 0/(0+eps) = 1
    assert all(s[f"lsm_{c}"] == 1 for c in ["ppron", "ipron", "conj", "preps", "auxverb", "adverb", "negate", "quant"])
    assert math.isclose(s["lsm"], (8 + s["lsm_article"]) / 9)


def test_formula():
    assert lsm_c(10, 10) > 0.999 and lsm_c(0, 10) < 1e-3


def test_novel_removes_copies():
    p = "please tell me how to reset my password today"
    r = "tell me how to reset my password"
    assert math.isnan(score(p, r, novel=True)["lsm"])      # everything copied
    assert not math.isnan(score(p, r)["lsm"])
    r2 = "you can tell me how to reset my password if you are not sure"
    s = score(p, r2, novel=True)
    assert s["n_r"] == 7 and s["n_p"] == 9 and s["r_negate"] > 0


def test_empty():
    assert math.isnan(score("hi", "")["lsm"])
