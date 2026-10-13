"""Numerical parity with the authors' `analyze()` (third_party/llm-convergence/postprocess_stylometrics.py) when both are
given the SAME lexicon (our ConvoKit-based lists exported as regexes, as their fullmatch-based matcher expects) and the
SAME spaCy parser.  Texts are single-line (no whitespace tokens: our port drops them)."""
import csv
import importlib.util
import math
import os

import pytest

import blevins as bl
from common import CATS, LEX_EXACT, LEX_PREFIX

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "llm-convergence"))
TEXTS = [
    ("My landlord did not fix the heater last week, and now the apartment is freezing.",
     "You should call your landlord in writing and keep a copy of every message. Don't wait too long."),
    ("Alice and Bob went to Paris because they wanted to see the Louvre.",
     "That sounds great! Paris is lovely, and Bob will love the Louvre. Nobody is ever bored there."),
    ("I ain't never seen nothing like it.", "It is not something we have all seen, but many of us can't say."),
]


def test_parity_with_authors_code(tmp_path, monkeypatch):
    pytest.importorskip("spacy")
    if not os.path.exists(os.path.join(REPO, "postprocess_stylometrics.py")):
        pytest.skip("third_party/llm-convergence not cloned")
    import re
    (tmp_path / "data" / "liwc").mkdir(parents=True)
    with open(tmp_path / "data" / "liwc" / "liwc2007_subset.csv", "w", newline="") as f:
        w = csv.writer(f)
        for c in CATS:
            pats = [re.escape(x) for x in sorted(LEX_EXACT[c])] + [re.escape(p) + ".*" for p in LEX_PREFIX[c]]
            if c == "negate":
                pats.append(re.escape("n't"))
            w.writerow([c, "", *pats])
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("theirs", os.path.join(REPO, "postprocess_stylometrics.py"))
    theirs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theirs)
    nlp = bl.load_nlp()
    for ref, trg in TEXTS:
        t = theirs.analyze(trg, ref, nlp)
        o = bl.score(ref, trg, nlp=nlp)
        assert math.isclose(t["utter_diff"], o["blevins_len"], rel_tol=1e-9)
        assert math.isclose(t["novel_tok"], o["blevins_novtok"], rel_tol=1e-9)
        assert t["propn"] == o["blevins_propn"]
        for c in CATS:
            assert math.isclose(t[f"liwc_{c}"], o[f"blevins_liwc_{c}"], rel_tol=1e-9), c
        pos_mean = sum(v for k, v in t.items() if k.startswith("pos_")) / 15
        assert math.isclose(pos_mean, o["blevins_pos"], rel_tol=1e-9)
