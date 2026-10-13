#!/usr/bin/env python
"""E5 tests: treatment x dose interaction vs the matched control, on per-seed-prompt rate contrasts (units = CR, i.e. AI / mean prompt contrast).
Metrics: crheld (F_heldout items of AAE/SGE, pre-registered FP rule applied) and crg (NGP/APP all-feature rate). Arms: dose 0 = d0_s*, treatment T5k/T20k, control C5k/C20k;
each arm-dose vector = mean over its training seeds, per seed prompt (287 prompts).
  (1) effect at 20k: mean_i [y_T20k,i - y_C20k,i], bootstrap 95% CI over prompts, one-sided paired sign-flip p (10,000);   same at 5k
  (2) interaction: mean_i [slope_T,i - slope_C,i], slope over dose {0,5k,20k} (units: per 10k pairs) -- bootstrap CI + sign-flip p.
Added (lead 2026-10-03, GATE_MEMO §8): `split` = pre-registered S contrast (mean per-prompt CR over {INF,BRE,AUS} minus mean over {AAE,SGE,NGP,APP}; tests T-C at 5k/20k, T-d0, and the exploratory C-d0)
and `ftrain` = uptake check (CR on the F_TRAIN items of AAE/SGE: T vs C and T vs d0). Keys: e5t.split.*, e5t.ftrain.*. All E5@<arm>_s<k> runs present in the scores file are used.
Arm N (authentic prompts) is NOT built yet: this is Arm S only.
  MV_COREF=0 python analysis/e5_tests.py --scores E5/scores.parquet --out results/E5/tests.json"""
import argparse, json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors.pilot_summary import rate_seed_means
from detectors import split_items
ap = argparse.ArgumentParser(); ap.add_argument("--scores"); ap.add_argument("--out"); a = ap.parse_args()
df = pd.read_parquet(a.scores); resp, prm = df[df.kind == "response"], df[df.kind == "prompt"]
rng = np.random.default_rng(0)
HELD = {"AAE": ["AAE.morph.held", "AAE.lex.held"], "SGE": ["SGE.part.held", "SGE.lex.held"]}
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
sae = resp[resp.condition == "SAE"]
excluded = {c for c in df.columns if c.startswith("ITEM.") and (".held." in c or ".train." in c) and c.endswith(".real_n") and (sae[c] > 0).mean() > 0.02}


def held_vec(d, v, novel, clean=True, groups=None):
    cols = [f"ITEM.{g}.{n}.{'real_n' if novel else 'real'}" for g in (groups or HELD)[v] for n in split_items.ITEMS[g]]
    if clean: cols = [c for c in cols if c.replace(".real", ".real_n") not in excluded]
    r = (100.0 * d[cols].sum(axis=1) / d.n_words.clip(lower=1)).groupby([d.condition, d.seed_id]).mean().unstack(0)
    return r


def vecs(model, metric):
    d = resp[resp.model == model]; out = []
    if metric == "split":
        cr = {}
        for v in ("INF", "BRE", "AUS", "AAE", "SGE", "NGP", "APP"):
            if v not in set(d.condition): continue
            R, P = rate_seed_means(d, v, True), rate_seed_means(prm, v, False)
            pc = (P[v] - P["SAE"]).mean()
            if abs(pc) > 1e-9: cr[v] = (R[v] - R["SAE"]) / pc
        pres = [cr[v] for v in ("INF", "BRE", "AUS") if v in cr]; stig = [cr[v] for v in ("AAE", "SGE", "NGP", "APP") if v in cr]
        return (pd.concat(pres, axis=1).mean(axis=1) - pd.concat(stig, axis=1).mean(axis=1)) if pres and stig else None
    for v in (("AAE", "SGE") if metric in ("crheld", "ftrain") else ("NGP", "APP")):
        if v not in set(d.condition): continue
        if metric in ("crheld", "ftrain"):
            G = HELD if metric == "crheld" else TRAIN
            R, P = held_vec(d, v, True, groups=G), held_vec(prm, v, False, groups=G)
        else: R, P = rate_seed_means(d, v, True), rate_seed_means(prm, v, False)
        pc = (P[v] - P["SAE"]).mean()
        if abs(pc) > 1e-9: out.append(((R[v] - R["SAE"]) / pc).rename(v))
    return pd.concat(out, axis=1).mean(axis=1) if out else None


def arm(prefix, metric):
    ms = [m for m in resp.model.unique() if m.startswith("E5@") and re.fullmatch(prefix + r"_s\d+", m[3:])]
    vs = [v for v in (vecs(m, metric) for m in ms) if v is not None]
    return (pd.concat(vs, axis=1).mean(axis=1), len(vs)) if vs else (None, 0)


def test(x, label, sign=1):
    """sign = expected direction of the effect (+1 accommodation raises the metric; -1 for the split S, which should NARROW under treatment). p_one_sided is in the expected direction,
    p_two_sided is the paired sign-flip two-sided p (10,000 flips)."""
    x = x.dropna().values; n = len(x); obs = x.mean()
    flips = (rng.choice([-1, 1], size=(10000, n)) * x).mean(axis=1)
    p1 = (np.sum(sign * flips >= sign * obs) + 1) / 10001; p2 = (np.sum(np.abs(flips) >= abs(obs)) + 1) / 10001
    bs = [x[rng.integers(0, n, n)].mean() for _ in range(2000)]
    return {"mean": float(obs), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "p_one_sided": float(p1), "p_two_sided": float(p2),
            "expected_sign": sign, "n_prompts": int(n)}


out = {"arm": "S (synthetic prompts); Arm N not built", "excluded_heldout_items": sorted(excluded)}
for metric in ("crheld", "crg", "split", "ftrain"):
    A = {k: arm(k, metric) for k in ("d0", "T1k", "T5k", "T20k", "T60k", "C5k", "C20k", "C60k", "A1k", "A5k")}
    res = {"n_runs": {k: v[1] for k, v in A.items()}}
    sg = -1 if metric == "split" else 1      # pre-registered: treatment narrows S (T - C < 0); control-vs-d0 widening is exploratory (+1)
    for dose in ("5k", "20k", "60k"):
        t, c = A[f"T{dose}"][0], A[f"C{dose}"][0]
        if t is None or c is None and dose == "60k": pass
        if t is not None and c is not None: res[f"effect_{dose}_T_minus_C"] = test(t - c, dose, sg)
        if t is not None and A["d0"][0] is not None: res[f"effect_{dose}_T_minus_d0"] = test(t - A["d0"][0].reindex(t.index), dose, sg)
        if c is not None and A["d0"][0] is not None: res[f"exploratory_{dose}_C_minus_d0"] = test(c - A["d0"][0].reindex(c.index), dose, 1)
    if A["T5k"][0] is not None and A["A5k"][0] is not None: res["effect_5k_T_minus_anti"] = test(A["T5k"][0] - A["A5k"][0].reindex(A["T5k"][0].index), "5k", sg)
    if all(A[k][0] is not None for k in ("d0", "T5k", "T20k", "T60k", "C5k", "C20k", "C60k")):
        xs = np.array([0.0, 5.0, 20.0, 60.0]); xc = xs - xs.mean()
        def slope4(ys):
            Y = np.vstack([y.values for y in ys]); return (xc[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum() * 10
        idx4 = A["d0"][0].index
        s4T = slope4([A["d0"][0], *(A[k][0].reindex(idx4) for k in ("T5k", "T20k", "T60k"))]); s4C = slope4([A["d0"][0], *(A[k][0].reindex(idx4) for k in ("C5k", "C20k", "C60k"))])
        res["interaction_slope4_T_minus_C_per10k"] = test(pd.Series(s4T - s4C, index=idx4), "interaction4", sg)
    if all(A[k][0] is not None for k in ("d0", "T5k", "T20k", "C5k", "C20k")):
        x = np.array([0.0, 5.0, 20.0])
        def slope(y0, y1, y2):
            Y = np.vstack([y0.values, y1.values, y2.values]); xc = x - x.mean(); return (xc[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum() * 10
        idx = A["d0"][0].index
        sT = slope(A["d0"][0], A["T5k"][0].reindex(idx), A["T20k"][0].reindex(idx)); sC = slope(A["d0"][0], A["C5k"][0].reindex(idx), A["C20k"][0].reindex(idx))
        res["interaction_slope_T_minus_C_per10k"] = test(pd.Series(sT - sC, index=idx), "interaction", sg)
        res["slope_T_per10k"] = float(np.nanmean(sT)); res["slope_C_per10k"] = float(np.nanmean(sC))
    out[metric] = res
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k in ("crheld", "crg", "split", "ftrain")}, indent=1)[:6000])
