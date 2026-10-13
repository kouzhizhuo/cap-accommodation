import numpy as np

import coordination as co
from common import CATS


def test_markers():
    s = co.score("The cat", "a dog and a bird")
    assert s["x_article"] == 1 and s["y_article"] == 1 and s["y_conj"] == 1 and s["x_conj"] == 0 and s["x_negate"] == 0
    assert co.score("x", "the dog", novel=False)["y_article"] == 1


def test_hand_coordination():
    i = CATS.index("article")
    x = np.zeros((4, len(CATS))); y = np.zeros((4, len(CATS)))
    x[:2, i] = 1
    y[0, i] = 1
    r = co.coordination(x, y, clusters=None)
    # P(y|x=1)=1/2, P(y)=1/4 -> 0.25 ; other categories: no x=1 pair -> NaN
    assert abs(r["article"]["coord"] - 0.25) < 1e-12
    assert np.isnan(r["ppron"]["coord"])
    assert abs(r["mean"]["coord"] - 0.25) < 1e-12


def test_perfect_and_null():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=(4000, len(CATS)))
    assert co.coordination(x, x.copy())["mean"]["coord"] > 0.45           # y = x: 1 - 0.5
    y = rng.integers(0, 2, size=x.shape)
    assert abs(co.coordination(x, y)["mean"]["coord"]) < 0.05             # independent


def test_bootstrap_ci_and_contrast():
    rng = np.random.default_rng(2)
    n = 600
    cl = np.repeat(np.arange(50).astype(str), n // 50)
    x = rng.integers(0, 2, size=(n, len(CATS)))
    y = np.where(rng.random(x.shape) < 0.7, x, rng.integers(0, 2, size=x.shape))
    r = co.coordination(x, y, cl, B=200)
    lo, hi = r["mean"]["ci"]
    assert lo < r["mean"]["coord"] < hi
    y0 = rng.integers(0, 2, size=x.shape)
    d = co.coordination_contrast(x, y, cl, x, y0, cl, B=200)["mean"]
    assert d["ci"][0] > 0 and d["delta"] > 0
