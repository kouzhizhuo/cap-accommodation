"""Linguistic style coordination (Danescu-Niculescu-Mizil, Gamon & Dumais 2011; DNM, Lee & Kleinberg 2012).

Per marker category m and (prompt a, reply b) pair: x = 1 if the prompt contains a marker of m, y = 1 if the reply does.
Coordination_m(group) = P(y=1 | x=1) - P(y=1), estimated by pooling all pairs of a group (model x condition).
Mean over categories = overall coordination.  CI: cluster bootstrap over seed_id (prompt clusters).
Native implementation on (prompt, reply) pairs (ConvoKit's Coordination class needs a Corpus of speakers; not used, only
its marker word list, see common.py).  Differences from DNM: pooled over the group instead of averaged over
speaker pairs; single-turn; replies are long so P(y) is near 1 for common categories (ceiling -> coordination ~ 0).
"""
import numpy as np

from common import CATS, novel_tokens, tokens, word_cats_cached


def markers(words):
    s = set()
    for w in words:
        s.update(word_cats_cached(w))
    return s


def score(prompt, reply, novel=False):
    """Pair-level marker indicators: x_<cat> (prompt), y_<cat> (reply) in {0,1}."""
    pm = markers(t[0] for t in tokens(prompt))
    rm = markers(t[0] for t in (novel_tokens(reply, prompt) if novel else tokens(reply)))
    out = {}
    for c in CATS:
        out[f"x_{c}"] = int(c in pm)
        out[f"y_{c}"] = int(c in rm)
    return out


def _stats(x, y):
    """per-category sufficient stats for an array of pairs: n, sx, sy, sxy  (shape (4, len(CATS)))."""
    return np.array([[len(x)] * x.shape[1], x.sum(0), y.sum(0), (x * y).sum(0)], dtype=float)


def _coord_from_stats(S):
    n, sx, sy, sxy = S
    with np.errstate(divide="ignore", invalid="ignore"):
        py1 = np.where(sx > 0, sxy / sx, np.nan)
        py = sy / n
        return py1 - py, py1, py, sx


def coordination(x, y, clusters=None, B=2000, seed=0, alpha=0.05):
    """x, y: (n_pairs, n_cats) 0/1 arrays in common.CATS order; clusters: array of seed_id per pair.
    Returns {cat: {coord, p_y_given_x, p_y, n_x1, ci:[lo,hi]}, 'mean': {...}} (NaN when no pair has x=1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    S = _stats(x, y)
    coord, py1, py, nx1 = _coord_from_stats(S)
    res = {c: {"coord": float(coord[i]), "p_y_given_x": float(py1[i]), "p_y": float(py[i]), "n_x1": int(nx1[i])}
           for i, c in enumerate(CATS)}
    mean = float(np.nanmean(coord)) if np.isfinite(coord).any() else float("nan")
    res["mean"] = {"coord": mean}
    if clusters is not None and B:
        cl = np.asarray(clusters)
        ids = np.unique(cl)
        per = np.stack([_stats(x[cl == k], y[cl == k]) for k in ids])       # (K, 4, C)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(ids), size=(B, len(ids)))
        boot = per[idx].sum(1)                                              # (B, 4, C)
        bc = np.stack([_coord_from_stats(boot[b])[0] for b in range(B)])   # (B, C)
        for i, c in enumerate(CATS):
            v = bc[:, i][np.isfinite(bc[:, i])]
            res[c]["ci"] = [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))] if len(v) else [float("nan")] * 2
        with np.errstate(all="ignore"):
            bm = np.nanmean(bc, axis=1)
        bm = bm[np.isfinite(bm)]
        res["mean"]["ci"] = [float(np.percentile(bm, 100 * alpha / 2)), float(np.percentile(bm, 100 * (1 - alpha / 2)))] if len(bm) else [float("nan")] * 2
    return res


def coordination_contrast(x1, y1, cl1, x0, y0, cl0, B=2000, seed=0, alpha=0.05):
    """Paired cluster bootstrap of mean-over-categories coordination(cond) - coordination(reference), resampling the
    same seed_ids in both arms (seed_ids present in both). Returns {cat|'mean': {delta, ci}}."""
    ids = np.array(sorted(set(np.unique(cl1)) & set(np.unique(cl0))))
    m1, m0 = np.isin(cl1, ids), np.isin(cl0, ids)
    x1, y1, cl1, x0, y0, cl0 = (np.asarray(a)[m] for a, m in ((x1, m1), (y1, m1), (cl1, m1), (x0, m0), (y0, m0), (cl0, m0)))
    x1, y1, x0, y0 = (np.asarray(a, float) for a in (x1, y1, x0, y0))
    def per(x, y, cl):
        return np.stack([_stats(x[cl == k], y[cl == k]) for k in ids])
    P1, P0 = per(x1, y1, cl1), per(x0, y0, cl0)
    c1 = _coord_from_stats(P1.sum(0))[0]
    c0 = _coord_from_stats(P0.sum(0))[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    b1, b0 = P1[idx].sum(1), P0[idx].sum(1)
    d = np.stack([_coord_from_stats(b1[b])[0] - _coord_from_stats(b0[b])[0] for b in range(B)])  # (B, C)
    res = {}
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for i, c in enumerate(CATS):
        v = d[:, i][np.isfinite(d[:, i])]
        res[c] = {"delta": float(c1[i] - c0[i]),
                  "ci": [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [float("nan")] * 2}
    with np.errstate(all="ignore"):
        dm = np.nanmean(d, axis=1)
        pm = float(np.nanmean(c1) - np.nanmean(c0))
    dm = dm[np.isfinite(dm)]
    res["mean"] = {"delta": pm, "ci": [float(np.percentile(dm, lo)), float(np.percentile(dm, hi))] if len(dm) else [float("nan")] * 2}
    return res
