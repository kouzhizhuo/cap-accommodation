#!/usr/bin/env python
"""Paired between-cue interaction of the update diagnostic (GATE_MEMO §8; lead 2026-10-03).  Keys: e4.interaction.<pair>.*
Per (lineage, policy-vs-reference pair, prompt, cue v): MB_{i,v} = [r(x_v,y_v)-r(x_v,y_std)] - [r(x_SAE,y_v)-r(x_SAE,y_std)] (length-normalised, as E4/mbv_analyze.py),
  r = logpi_policy - logpi_ref.  Cues: stigmatized = {AAE, SGE}; prestige = {BRE, AUS} (frozen tiers 1 vs 2-3; INF is the register control and reported separately).
Quantities (per pair, over the lineages that have it):
  raw      = mean_i [ mean_{prestige} MB - mean_{stigmatized} MB ]   (prompt bootstrap per lineage; lineage-level mean)
  adjusted = coefficient on the prestige indicator in OLS  MB ~ prestige + z(baseline ll) + z(log tokens) + lineage FE,
             baseline ll = reference-model log-likelihood per token of y_v given x_v; tokens = |y_v| (matched cell); one row per (prompt, cue)
  CIs: two-level bootstrap over lineages (resampled with replacement) and prompts within lineage (B=2000); leave-one-lineage-out adjusted coefficients.
  Also reported: stigmatized-only and prestige-only mean MB, and INF vs stigmatized.   Breadth: family-level instruct_vs_base from E4/mbv_lp_breadth (descriptive).
  python E4/mbv_interaction.py --lp E1_incoming/mbv_lp [--lp-big ...] --out results/E4/interaction.json"""
import argparse, json
from pathlib import Path
import numpy as np

STIG, PRES = ("AAE", "SGE"), ("BRE", "AUS")
LIN = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L5": "olmo2_32b", "Q72": "qwen25_72b", "L2B": "olmo2_13b", "Z7": "zephyr_7b", "T27": "tulu2_7b", "T213": "tulu2_13b", "O31": "olmo31_32b", "T370": "tulu3_70b", "T270": "tulu2_70b"}
ap = argparse.ArgumentParser(); ap.add_argument("--lp", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pairs", nargs="+", default=["dpo:sft", "sft:base", "instruct:base"]); ap.add_argument("--breadth", nargs="*", default=[]); a = ap.parse_args()
rng = np.random.default_rng(0); B = 2000


def load(name):
    for d in a.lp:
        p = Path(d) / f"{name}.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return {json.loads(l)["item"]: json.loads(l) for l in open(p)}
    return None


def rows_for(lin, pol, ref):
    """-> list of dicts (sid, cue, mb, ll, ntok) for one lineage and pair"""
    P, R = load(f"{lin}@{pol}"), load(f"{lin}@{ref}")
    if P is None or R is None: return None
    cell = {}
    for k in P:
        if k in R:
            _, sid, v, pk, rk = k.split("|"); cell.setdefault((sid, v), {})[(pk, rk)] = (P[k]["lp"] - R[k]["lp"], P[k]["n"], R[k]["lp"])
    out = []
    for (sid, v), d in cell.items():
        if v not in STIG + PRES + ("INF",) or len(d) < 4: continue
        n = lambda pk, rk: d[(pk, rk)][0] / d[(pk, rk)][1]
        mb = (n("v", "v") - n("v", "std")) - (n("SAE", "v") - n("SAE", "std"))
        out.append({"sid": sid, "cue": v, "mb": mb, "ll": d[("v", "v")][2] / d[("v", "v")][1], "ntok": d[("v", "v")][1]})
    return out


def design(rows_by_lin, lins):
    """stack rows of the selected lineages -> y, X (prestige, z-ll, z-logn, lineage FE), prompt ids"""
    Y, G, LL, NT, L, S = [], [], [], [], [], []
    for li, lin in enumerate(lins):
        for r in rows_by_lin[lin]:
            if r["cue"] == "INF": continue
            Y.append(r["mb"]); G.append(1.0 if r["cue"] in PRES else 0.0); LL.append(r["ll"]); NT.append(np.log(r["ntok"])); L.append(li); S.append(r["sid"])
    Y, G, LL, NT, L = map(np.array, (Y, G, LL, NT, L))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    X = np.column_stack([G, z(LL), z(NT)] + [(L == i).astype(float) for i in range(len(lins))])
    return Y, X, np.array(S), L


def ols_coef(Y, X): return np.linalg.lstsq(X, Y, rcond=None)[0][0]


def raw_diff(rows):
    by = {}
    for r in rows: by.setdefault(r["sid"], {}).setdefault(r["cue"], []).append(r["mb"])
    d = [np.mean([np.mean(c[v]) for v in PRES if v in c]) - np.mean([np.mean(c[v]) for v in STIG if v in c]) for c in by.values() if any(v in c for v in PRES) and any(v in c for v in STIG)]
    return np.array(d)


def boot_two_level(rows_by_lin, lins, fn):
    """resample lineages with replacement, then prompts within each sampled lineage; fn(rows_by_lin_resampled, lins_resampled) -> scalar"""
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(lins), len(lins)); rb = {}; names = []
        for j, p in enumerate(pick):
            lin = lins[p]; rs = rows_by_lin[lin]; sids = sorted({r["sid"] for r in rs}); ch = rng.choice(sids, len(sids)); idx = {}
            for r in rs: idx.setdefault(r["sid"], []).append(r)
            nm = f"{lin}#{j}"; rb[nm] = [dict(r) for s in ch for r in idx[s]]; names.append(nm)
        vals.append(fn(rb, names))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


out = {"cues": {"stigmatized": list(STIG), "prestige": list(PRES), "register_control": "INF"},
       "method": "paired within-prompt cue contrast of the length-normalised update diagnostic; adjusted = OLS with baseline ll/token-length controls and lineage FE; two-level bootstrap (lineages, prompts)"}
for pair in a.pairs:
    pol, ref = pair.split(":"); rb = {}
    for lin in LIN:
        r = rows_for(lin, pol, ref)
        if r: rb[lin] = r
    if len(rb) == 0: continue
    lins = sorted(rb); node = {"lineages": [LIN[l] for l in lins], "n_lineages": len(lins)}
    per = {}
    for lin in lins:
        d = raw_diff(rb[lin]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
        per[LIN[lin]] = {"raw": float(d.mean()), "raw_lo": float(np.percentile(bs, 2.5)), "raw_hi": float(np.percentile(bs, 97.5)), "n_prompts": int(len(d))}
        for grp, cues in (("stig", STIG), ("pres", PRES), ("inf", ("INF",))):
            v = [r["mb"] for r in rb[lin] if r["cue"] in cues]; per[LIN[lin]][f"mb_{grp}"] = float(np.mean(v)) if v else None
    node["per_lineage"] = per
    raws = [per[LIN[l]]["raw"] for l in lins]; node["raw_mean_over_lineages"] = float(np.mean(raws))
    lo, hi = boot_two_level(rb, lins, lambda r, n: float(np.mean([raw_diff(r[k]).mean() for k in n])))
    node["raw_lo"], node["raw_hi"] = lo, hi
    Y, X, S, L = design(rb, lins); coef = ols_coef(Y, X)
    node["adjusted"] = {"coef": float(coef), "n_rows": int(len(Y))}
    lo, hi = boot_two_level(rb, lins, lambda r, n: ols_coef(*design(r, n)[:2]))
    node["adjusted"]["lo"], node["adjusted"]["hi"] = lo, hi
    node["adjusted"]["leave_one_lineage_out"] = {LIN[l]: float(ols_coef(*design({k: rb[k] for k in lins if k != l}, [k for k in lins if k != l])[:2])) for l in lins} if len(lins) > 1 else {}
    allv = np.array([r["mb"] for l in lins for r in rb[l] if r["cue"] in STIG]); pv = np.array([r["mb"] for l in lins for r in rb[l] if r["cue"] in PRES])
    node["mb_stigmatized_mean"], node["mb_prestige_mean"] = float(allv.mean()), float(pv.mean())
    iv = np.array([r["mb"] for l in lins for r in rb[l] if r["cue"] == "INF"]); node["mb_inf_mean"] = float(iv.mean()) if len(iv) else None
    out[pair.replace(":", "_vs_")] = node
# breadth families (descriptive; single pair per family)
bf = {}
for fam in a.breadth:
    for pol in ("instruct", "chat"):
        r = rows_for(fam, pol, "base")
        if r:
            d = raw_diff(r); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
            bf[fam] = {"pair": f"{pol}_vs_base", "raw": float(d.mean()), "raw_lo": float(np.percentile(bs, 2.5)), "raw_hi": float(np.percentile(bs, 97.5)), "n_prompts": int(len(d))}
if bf: out["breadth_instruct_vs_base"] = bf
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and "adjusted" in v:
        print(k, v["n_lineages"], "raw", round(v["raw_mean_over_lineages"], 4), [round(v["raw_lo"], 4), round(v["raw_hi"], 4)], "adj", round(v["adjusted"]["coef"], 4), [round(v["adjusted"]["lo"], 4), round(v["adjusted"]["hi"], 4)],
              "stig", round(v["mb_stigmatized_mean"], 4), "pres", round(v["mb_prestige_mean"], 4))
