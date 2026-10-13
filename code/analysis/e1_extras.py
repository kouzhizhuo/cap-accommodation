#!/usr/bin/env python
"""E1 extras merged into results/E1/summary.json (lead <agent>, 2026-10-02): 
 (1) per-cell std over the 3 sampling seeds (statistic recomputed on each seed's subset, ddof=1): cr_std, cr_rate_std, df_minus_inf_std, and <model>.split_std
 (2) per model-stage rho = Spearman(CR_v, frozen tier) over the varieties present (tiers: PREREG §2.2: BRE = P(3); SCO, IRE, AUS, IND = M(2); SGE, AAE, APP, NGP = G(1); INF/FOR excluded),
     bootstrap rho_lo/rho_hi (shared prompt resampling), for both CR forms (rho = rate-based, rho_gated); stig_mean.cr / stig_mean.cr_rate = mean over AAE, SGE, NGP, APP
 (3) per lineage dsplit_base_sft, dsplit_sft_dpo, dsplit_dpo_rlvr, locus = dsplit_sft_dpo - dsplit_base_sft with bootstrap lo/hi (ONE prompt resampling shared by all stages)
 S (PREREG §2.2 line 76) = mean CR over P = {BRE, INF, FOR} minus mean CR over G = {AAE, SGE, NGP, APP}; `split` (existing key, P = {INF, BRE, AUS}) is kept as `split_v14`.
 Two CR forms everywhere: gated/opportunity (`*`) and rate (`*_rate`). Missing varieties (SCO, IND, IRE) are simply absent from rho.
  MV_COREF=0 python analysis/e1_extras.py --scores E1/scores.parquet --summary results/E1/summary.json --mbv results/E4/mbv.json"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors import pilot_summary as ps_mod
from analysis.emit_results import LINEAGE, stage_key
ap = argparse.ArgumentParser(); ap.add_argument("--scores"); ap.add_argument("--summary"); ap.add_argument("--mbv", default=None); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args()
df = pd.read_parquet(a.scores); resp, prm = df[df.kind == "response"], df[df.kind == "prompt"]
IDS = sorted(prm.seed_id.unique()); N = len(IDS); rng = np.random.default_rng(0)
P_SET, G_SET = ["BRE", "INF", "FOR"], ["AAE", "SGE", "NGP", "APP"]
TIER = {"BRE": 3, "SCO": 2, "IRE": 2, "AUS": 2, "IND": 2, "SGE": 1, "AAE": 1, "APP": 1, "NGP": 1}
RL = {"L1": "rlvr1", "L2": "instruct", "L3": "instruct", "L4": "final", "L2B": "instruct", "L5": "instruct"}
VARS = [v for v in ps_mod.VARIETIES if v in set(df.condition) and v != "SAE"]


def cell(dm, v):
    """-> (cr_gated(sel), cr_rate(sel), dF_diff per global seed) as functions/arrays on the global seed index"""
    if v not in set(dm.condition): return None
    ps = ps_mod.per_seed(dm, v, novel=True); pD = ps.pivot(index="seed_id", columns="condition", values="D").reindex(IDS)
    if "SAE" not in pD: return None
    pp = ps_mod.per_seed(prm, v, novel=False, gate=False)
    pr, po = (pp.pivot(index="seed_id", columns="condition", values=c).reindex(IDS) for c in ("r", "o"))
    ai = (pD[v] - pD["SAE"]).values; rc, oc, rs, os_ = (x.values for x in (pr[v], po[v], pr["SAE"], po["SAE"]))
    R = ps_mod.rate_seed_means(dm, v, True).reindex(IDS); Pm = ps_mod.rate_seed_means(prm, v, False).reindex(IDS)
    rai = (R[v] - R["SAE"]).values; rpc = (Pm[v] - Pm["SAE"]).values

    def gated(sel):
        den = rc[sel].sum() / max(1, (rc[sel] + oc[sel]).sum()) - rs[sel].sum() / max(1, (rs[sel] + os_[sel]).sum())
        return np.nanmean(ai[sel]) / den if abs(den) > 1e-9 else np.nan

    def rate(sel):
        den = np.nanmean(rpc[sel]); return np.nanmean(rai[sel]) / den if abs(den) > 1e-9 else np.nan
    F = dm.groupby(["condition", "seed_id"]).formality.mean().unstack(0).reindex(IDS)
    ddf = ((F[v] - F["SAE"]) - (F["INF"] - F["SAE"])).values if "INF" in F else np.full(N, np.nan)
    return gated, rate, ddf


def S_of(cells, sel, mode):
    f = (lambda c: c[0](sel)) if mode == "g" else (lambda c: c[1](sel))
    P = [f(cells[v]) for v in P_SET if v in cells]; G = [f(cells[v]) for v in G_SET if v in cells]
    return (np.nanmean(P) - np.nanmean(G)) if P and G else np.nan


def rho_of(cells, sel, mode):
    xs, ys = [], []
    for v, t in TIER.items():
        if v in cells:
            c = (cells[v][0] if mode == "g" else cells[v][1])(sel)
            if not np.isnan(c): xs.append(c); ys.append(t)
    return spearmanr(xs, ys)[0] if len(xs) >= 4 and len(set(ys)) > 1 and len(set(xs)) > 1 else np.nan


ALL = np.arange(N); BS = [rng.integers(0, N, N) for _ in range(a.B)]
S = json.load(open(a.summary)); models = sorted(resp.model.unique()); cells_by = {}; out = {}
def pct(x): x = np.asarray(x, float); return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))
for m in models:
    dm = resp[resp.model == m]; lin, st = m.split("@"); node = out.setdefault(LINEAGE.get(lin, lin.lower()), {}).setdefault(stage_key(st), {})
    cells = {v: c for v in VARS if (c := cell(dm, v)) is not None}; cells_by[m] = cells
    # (1) seed std
    per_s = {s: {v: c for v in VARS if (c := cell(dm[dm.sampling_seed == s], v)) is not None} for s in sorted(dm.sampling_seed.unique())}
    for v in cells:
        vn = node.setdefault(v.lower(), {})
        g = [per_s[s][v][0](ALL) for s in per_s if v in per_s[s]]; r = [per_s[s][v][1](ALL) for s in per_s if v in per_s[s]]
        d = [np.nanmean(per_s[s][v][2]) for s in per_s if v in per_s[s]]
        vn["cr_std"] = float(np.std(g, ddof=1)) if len(g) > 1 else None; vn["cr_rate_std"] = float(np.std(r, ddof=1)) if len(r) > 1 else None
        vn["df_minus_inf_std"] = float(np.std(d, ddof=1)) if len(d) > 1 else None
    sv = [S_of(per_s[s], ALL, "g") for s in per_s]; node["split_std"] = float(np.nanstd(sv, ddof=1)) if len(sv) > 1 else None
    node["split_prereg"] = float(S_of(cells, ALL, "g")); node["split_rate"] = float(S_of(cells, ALL, "r"))
    stig = [(cells[v][0](ALL), cells[v][1](ALL)) for v in G_SET if v in cells]
    if stig: node["stig_mean"] = {"cr": float(np.nanmean([x[0] for x in stig])), "cr_rate": float(np.nanmean([x[1] for x in stig]))}
    # (2) rho (+ bootstrap)
    for mode, key in (("r", "rho"), ("g", "rho_gated")):
        node[key] = float(rho_of(cells, ALL, mode)); lo, hi = pct([rho_of(cells, sel, mode) for sel in BS[:500]]); node[key + "_lo"], node[key + "_hi"] = lo, hi
# (3) dsplit / locus per lineage (shared resampling)
for lin in sorted({m.split("@")[0] for m in models}):
    ms = {"base": f"{lin}@base", "sft": f"{lin}@sft", "dpo": f"{lin}@dpo", "rl": f"{lin}@{RL.get(lin, 'instruct')}"}
    if not all(ms[k] in cells_by for k in ("base", "sft", "dpo")): continue
    ln = out.setdefault(LINEAGE.get(lin, lin.lower()), {})
    for mode, suf in (("g", ""), ("r", "_rate")):
        Sb = {k: np.array([S_of(cells_by[mm], sel, mode) for sel in BS]) for k, mm in ms.items() if mm in cells_by}
        S0 = {k: S_of(cells_by[mm], ALL, mode) for k, mm in ms.items() if mm in cells_by}
        for name, (x, y) in {"dsplit_base_sft": ("base", "sft"), "dsplit_sft_dpo": ("sft", "dpo"), "dsplit_dpo_rlvr": ("dpo", "rl")}.items():
            if x in S0 and y in S0:
                d = Sb[y] - Sb[x]; ln[name + suf] = float(S0[y] - S0[x]); ln[name + suf + "_lo"], ln[name + suf + "_hi"] = pct(d)
        d = (Sb["dpo"] - Sb["sft"]) - (Sb["sft"] - Sb["base"]); ln["locus" + suf] = float((S0["dpo"] - S0["sft"]) - (S0["sft"] - S0["base"])); ln["locus" + suf + "_lo"], ln["locus" + suf + "_hi"] = pct(d)
# merge into summary.json (deep update; existing keys kept, old `split` preserved as split_v14)
def deep(dst, src):
    for k, v in src.items():
        if isinstance(v, dict): deep(dst.setdefault(k, {}), v)
        else: dst[k] = v
for lin, sts in S.items():
    if isinstance(sts, dict) and lin in out:
        for st, nd in sts.items():
            if isinstance(nd, dict) and "split" in nd and "split_v14" not in nd: nd["split_v14"] = nd["split"]
deep(S, out)
for lin, sts in S.items():      # `split` := PREREG S (P = {BRE, INF, FOR}, G = {AAE, SGE, NGP, APP}) wherever computed
    if isinstance(sts, dict):
        for st, nd in sts.items():
            if isinstance(nd, dict) and nd.get("split_prereg") is not None: nd["split"] = nd["split_prereg"]
S.setdefault("meta", {})["extras"] = "e1_extras.py: seed std (ddof=1), rho vs frozen tiers, dsplit/locus (shared resampling, B=%d), split := PREREG S" % a.B
json.dump(S, open(a.summary, "w"), indent=1)
if a.mbv:   # (4) mb_std = SD over seed prompts of the per-prompt MB_v (the SFT reply is a single sample, so there are no sampling seeds)
    print("mbv: mb_std is added by analysis/mbv_analyze.py (SD over prompts)")
print("done", len(models), "models")
