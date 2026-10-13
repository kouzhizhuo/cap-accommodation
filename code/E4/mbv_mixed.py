#!/usr/bin/env python
"""PREREG P4 mixed model of the update diagnostic: MB ~ tier + (1|seed) + (1|variety), tier = frozen prestige tier (AAE=SGE=1, AUS=2, BRE=3; INF excluded: register control), one row per
(lineage, prompt, variety) of the length-normalised MB (as E4/mbv_analyze.py). Pooled over lineages with lineage fixed effects (+ per-lineage fits). Crossed random intercepts via a single-group variance-components MixedLM (statsmodels, REML).
  python E4/mbv_mixed.py --lp _incoming/<host>/E4/mbv_lp --out results/E4/mixed.json      Keys: e4.mixed.<pair>.<lineage|pooled>.{tier,tier_lo,tier_hi,p,var_seed,var_variety,n}"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
LIN = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L5": "olmo2_32b", "L2B": "olmo2_13b", "Z7": "zephyr_7b", "T27": "tulu2_7b", "T213": "tulu2_13b", "O31": "olmo31_32b", "T370": "tulu3_70b", "T270": "tulu2_70b"}
TIER = {"AAE": 1, "SGE": 1, "AUS": 2, "BRE": 3}
ap = argparse.ArgumentParser(); ap.add_argument("--lp", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pairs", nargs="+", default=["dpo:sft", "sft:base"]); a = ap.parse_args()


def load(name):
    for d in a.lp:
        p = Path(d) / f"{name}.jsonl"
        if p.exists() and p.stat().st_size: return {json.loads(l)["item"]: json.loads(l) for l in open(p)}


def rows(lin, pol, ref):
    P, R = load(f"{lin}@{pol}"), load(f"{lin}@{ref}")
    if P is None or R is None: return None
    cell = {}
    for k in P:
        if k in R:
            _, sid, v, pk, rk = k.split("|"); cell.setdefault((sid, v), {})[(pk, rk)] = ((P[k]["lp"] - R[k]["lp"]) / P[k]["n"])
    return [{"lineage": LIN[lin], "seed": sid, "variety": v, "tier": TIER[v], "mb": (d[("v", "v")] - d[("v", "std")]) - (d[("SAE", "v")] - d[("SAE", "std")])}
            for (sid, v), d in cell.items() if v in TIER and len(d) >= 4]


def fit(df, fe, variety_re=True):
    df = df.assign(one=1)
    if variety_re:
        m = smf.mixedlm(f"mb ~ tier{fe}", df, groups="one", re_formula="0", vc_formula={"seed": "0 + C(seed)", "variety": "0 + C(variety)"}).fit(reml=True, method="lbfgs")
    else:
        m = smf.mixedlm(f"mb ~ tier{fe}", df, groups="seed").fit(reml=True, method="lbfgs")
    ci = m.conf_int().loc["tier"]; vc = dict(zip(m.model.exog_vc.names, m.vcomp)) if variety_re else {"seed": float(np.asarray(m.cov_re)[0, 0])}
    return {"tier": float(m.params["tier"]), "tier_lo": float(ci[0]), "tier_hi": float(ci[1]), "p": float(m.pvalues["tier"]), "var_seed": float(vc.get("seed", np.nan)), "var_variety": float(vc.get("variety", np.nan)),
            "var_resid": float(m.scale), "n": int(len(df)), "n_seeds": int(df.seed.nunique()), "converged": bool(m.converged)}


out = {"model": "MB ~ tier + (1|seed) + (1|variety) [+ lineage FE when pooled]; tier 1/1/2/3 = AAE,SGE/AUS/BRE; REML; INF excluded", "caveat": "only 4 varieties: the variety variance component is weakly identified; tier is a between-variety predictor"}
for pair in a.pairs:
    pol, ref = pair.split(":"); allr, node = [], {}
    for lin in LIN:
        r = rows(lin, pol, ref)
        if r: allr += r
    if not allr: continue
    df = pd.DataFrame(allr)
    for l in sorted(df.lineage.unique()): node[l] = fit(df[df.lineage == l], "")
    if df.lineage.nunique() > 1: node["pooled"] = fit(df, " + C(lineage)")
    node["seed_only"] = {l: fit(df[df.lineage == l], "", False) for l in sorted(df.lineage.unique())}      # sensitivity: no variety RE (tier then identified by within-seed contrasts)
    if df.lineage.nunique() > 1: node["seed_only"]["pooled"] = fit(df, " + C(lineage)", False)
    out[f"{pol}_vs_{ref}"] = node
    for l, v in list(node.items()) + [(f"seed_only/{k}", x) for k, x in node.get("seed_only", {}).items()]:
        if "tier" in v: print(pair, l, "tier %.4f [%.4f, %.4f] p=%.3g n=%d" % (v["tier"], v["tier_lo"], v["tier_hi"], v["p"], v["n"]))
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
