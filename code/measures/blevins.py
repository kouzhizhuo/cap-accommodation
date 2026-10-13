"""Blevins, Schmalwieser & Roth (EACL 2026, arXiv:2508.03276) stylometric convergence, adapted to single-turn pairs.

Transcribed from the authors' code (third_party/llm-convergence/postprocess_stylometrics.py, function `analyze`) and the
paper (Sec. 3.1).  Their four convergence metrics of reply r_t w.r.t. the turn before it (here: the prompt):
  utter_diff  : LSM on token counts            1-|a-b|/(a+b+1e-5)
  LIWC agree. : mean over 9 LIWC2007 classes of LSM on proportions (#matching spaCy tokens / #tokens), eps 1e-5
  PROPN       : #shared lower-cased PROPN types (code) -- the paper text says "percentage"; we report both
  novel_tok   : |unique reply tokens not in prompt| / |reply tokens|   (code; asymmetric: normalised by the reply only)
  (+ pos_mean : mean over 15 UPOS tags of LSM on tag proportions; computed by the code, not one of the paper's four)
Reply = "target" (first arg of their `analyze`), prompt = "reference".  See README for differences from their setting
(single turn, no human/random baseline, ConvoKit lists instead of the licensed LIWC2007, spaCy en_core_web_sm instead of
en_core_web_trf, whitespace tokens dropped).
`blevins_conv` is OUR composite (not in the paper): mean(utter_diff LSM, token overlap = 1 - novel_tok).
"""
import math

import numpy as np

from common import CATS, char_mask, copied_ranges, word_cats_cached

EPS = 1e-5
NAN = float("nan")
UPOS = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
        'PUNCT']  # the code's tagset (SYM, X dropped)

_NLP = None


def load_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    return _NLP


def lsm(a, b, eps=EPS):
    return 1 - abs(a - b) / (a + b + eps)


def _cats_of(tok):
    t = tok.lower().replace("’", "'")
    c = list(word_cats_cached(t))
    if t == "n't" and "negate" not in c:      # spaCy splits "don't" -> do + n't
        c.append("negate")
    return c


def toks_from_doc(doc):
    """[(text, pos, start_char, end_char)] without whitespace-only tokens."""
    return [(t.text, t.pos_, t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]


def drop_copied(toks, reply, prompt):
    rng, _ = copied_ranges(reply, prompt)
    if not rng:
        return toks
    m = char_mask(reply, rng)
    return [t for t in toks if not m[t[2]:t[3]].any()]


def features(trg, ref):
    """trg, ref: lists of (text, pos, s, e). Returns the metric dict (code-faithful). NaN if either is empty."""
    keys = ["blevins_len", "blevins_liwc", "blevins_pos", "blevins_novtok", "blevins_tokoverlap", "blevins_propn",
            "blevins_propn_frac", "blevins_conv"]
    out = {"n_tok_p": len(ref), "n_tok_r": len(trg)}
    if not trg or not ref:
        out.update({k: NAN for k in keys})
        return out
    out["blevins_len"] = lsm(len(trg), len(ref))
    # POS
    tp, rp = [t[1] for t in trg], [t[1] for t in ref]
    out["blevins_pos"] = float(np.mean([lsm(tp.count(g) / len(tp), rp.count(g) / len(rp)) for g in UPOS]))
    # LIWC agreement
    def cc(ts):
        d = {c: 0 for c in CATS}
        for t in ts:
            for c in _cats_of(t[0]):
                d[c] += 1
        return d
    tc, rc = cc(trg), cc(ref)
    liwc = {c: lsm(tc[c] / len(trg), rc[c] / len(ref)) for c in CATS}
    out["blevins_liwc"] = float(np.mean(list(liwc.values())))
    for c in CATS:
        out[f"blevins_liwc_{c}"] = liwc[c]
    # PROPN overlap
    tpn = {t[0].lower() for t in trg if t[1] == "PROPN"}
    rpn = {t[0].lower() for t in ref if t[1] == "PROPN"}
    out["blevins_propn"] = len(tpn & rpn)
    out["blevins_propn_frac"] = len(tpn & rpn) / len(tpn) if tpn else NAN
    # token novelty (code: unique novel types / all target tokens, case-sensitive, punctuation included)
    rset = {t[0] for t in ref}
    novel = {t[0] for t in trg if t[0] not in rset}
    out["blevins_novtok"] = len(novel) / len(trg)
    out["blevins_tokoverlap"] = 1 - out["blevins_novtok"]
    out["blevins_conv"] = (out["blevins_len"] + out["blevins_tokoverlap"]) / 2
    return out


def score_docs(prompt, reply, prompt_doc, reply_doc, novel=False):
    trg = toks_from_doc(reply_doc)
    if novel:
        trg = drop_copied(trg, reply, prompt)
    return features(trg, toks_from_doc(prompt_doc))


def score(prompt, reply, nlp=None, novel=False):
    nlp = nlp or load_nlp()
    return score_docs(prompt, reply, nlp(prompt), nlp(reply), novel=novel)
