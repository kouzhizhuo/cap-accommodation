"""Language Style Matching (Ireland & Pennebaker 2010; Durandard et al. 2025).

LSM_c = 1 - |p_c - r_c| / (p_c + r_c + 0.0001),  p_c / r_c = % of words of prompt / reply in category c;
LSM = mean_c LSM_c over the nine categories (common.CATS). Word lists: ConvoKit coord-liwc-patterns + own negations
(see common.py / README) -- NOT LIWC.  novel=True drops reply words in copied 4-grams / quotes first
(prompt untouched).
"""
import math

from common import CATS, cat_counts, novel_tokens, tokens

EPS = 1e-4
NAN = float("nan")


def lsm_c(p, r, eps=EPS):
    return 1.0 - abs(p - r) / (p + r + eps)


def score(prompt, reply, novel=False):
    """-> dict: lsm, lsm_<cat>, p_<cat>, r_<cat> (percentages), n_p, n_r. NaN if either side has no words."""
    pw = [t[0] for t in tokens(prompt)]
    rw = [t[0] for t in (novel_tokens(reply, prompt) if novel else tokens(reply))]
    out = {"n_p": len(pw), "n_r": len(rw)}
    if not pw or not rw:
        out.update({"lsm": NAN, **{f"lsm_{c}": NAN for c in CATS}, **{f"p_{c}": NAN for c in CATS},
                    **{f"r_{c}": NAN for c in CATS}})
        return out
    pc, rc = cat_counts(pw), cat_counts(rw)
    vals = []
    for c in CATS:
        p, r = 100.0 * pc[c] / len(pw), 100.0 * rc[c] / len(rw)
        out[f"p_{c}"], out[f"r_{c}"] = p, r
        out[f"lsm_{c}"] = lsm_c(p, r)
        vals.append(out[f"lsm_{c}"])
    out["lsm"] = sum(vals) / len(vals)
    return out
