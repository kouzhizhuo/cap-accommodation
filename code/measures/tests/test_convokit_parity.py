"""Our marker matcher vs ConvoKit's own annotator (Coordination._annot_liwc_cats) on real replies (skipped without convokit)."""
import glob
import json
import os
import random

import pytest

from common import CATS, tokens, word_cats_cached
from coordination import markers

GEN = os.environ.get("P3_GEN", "<HOME>/arr_oct/runs/P3/P0/gen")


class _Utt:
    def __init__(self, t):
        self.text, self.meta = t, {}


class _Corpus:
    def __init__(self, texts):
        self.u = [_Utt(t) for t in texts]

    def iter_utterances(self):
        return iter(self.u)


def test_agrees_with_convokit_annotator():
    ck = pytest.importorskip("convokit")
    from convokit.coordination.coordination import Coordination
    files = sorted(glob.glob(os.path.join(GEN, "*.jsonl")))
    if not files:
        texts = ["The cat sat on the mat, and I don't think they'll mind.", "Something's up; nobody knows anything."] * 50
    else:
        random.seed(0)
        texts = []
        for f in files[:6]:
            rows = [json.loads(l) for l in open(f)]
            texts += [r["response"] for r in random.sample(rows, 100)] + [r["prompt"] for r in random.sample(rows, 50)]
    co = object.__new__(Coordination)
    co._compute_liwc_reverse_dict()
    corpus = _Corpus(texts)
    co._annot_liwc_cats(corpus)
    same, per_cat_diff = 0, {c: 0 for c in CATS}
    for u, t in zip(corpus.u, texts):
        mine = markers(w[0] for w in tokens(t)) - {"negate"}
        theirs = set(u.meta["liwc-categories"])
        same += mine == theirs
        for c in mine ^ theirs:
            per_cat_diff[c] += 1
    print("exact agreement", same / len(texts), per_cat_diff)
    assert same / len(texts) > 0.9
