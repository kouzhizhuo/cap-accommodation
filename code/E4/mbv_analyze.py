#!/usr/bin/env python
"""MB_v = [r(x_v,y_v) - r(x_v,y_std)] - [r(x_SAE,y_v) - r(x_SAE,y_std)],  r = log pi_policy(y|x) - log pi_ref(y|x).
Primary: length-normalised (r / |y|, matches open-instruct dpo_norm); `mb_raw`: unnormalised sums (robustness). Mean over seeds with bootstrap 95% CI (B=2000).
  python E4/mbv_analyze.py --lp E4/mbv_lp --pairs L2:dpo:sft L2:sft:base L3:dpo:sft L4:dpo:sft L2B:dpo:sft L1:dpo:sft L1:sft:base --out results/E4/mbv.json"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
LIN = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L5": "olmo2_32b", "Q72": "qwen25_72b", "L2B": "olmo2_13b", "Z7": "zephyr_7b", "T27": "tulu2_7b", "T213": "tulu2_13b", "O31": "olmo31_32b", "T370": "tulu3_70b", "T270": "tulu2_70b"}
ap = argparse.ArgumentParser(); ap.add_argument("--lp"); ap.add_argument("--pairs", nargs="+"); ap.add_argument("--out"); a = ap.parse_args()
rng = np.random.default_rng(0); out = {}


def load(lin, st):
    p = Path(a.lp) / f"{lin}@{st}.jsonl"
    return {json.loads(l)["item"]: json.loads(l) for l in open(p)} if p.exists() else None


for spec in a.pairs:
    lin, pol, ref = spec.split(":"); P, R = load(lin, pol), load(lin, ref)
    if P is None or R is None: print("missing", spec); continue
    cell = defaultdict(dict)
    for k in P:
        if k in R:
            _, sid, v, pk, rk = k.split("|"); cell[(sid, v)][(pk, rk)] = (P[k]["lp"] - R[k]["lp"], P[k]["n"])
    node = out.setdefault(LIN.get(lin, lin.lower()), {}).setdefault(f"{pol}_vs_{ref}", {})
    for v in sorted({v for _, v in cell}):
        norm, raw = [], []
        for (sid, vv), d in cell.items():
            if vv != v or len(d) < 4: continue
            n = lambda pk, rk: d[(pk, rk)][0] / d[(pk, rk)][1]; r = lambda pk, rk: d[(pk, rk)][0]
            norm.append((n("v", "v") - n("v", "std")) - (n("SAE", "v") - n("SAE", "std"))); raw.append((r("v", "v") - r("v", "std")) - (r("SAE", "v") - r("SAE", "std")))
        norm, raw = np.array(norm), np.array(raw)
        bs = [np.mean(norm[rng.integers(0, len(norm), len(norm))]) for _ in range(2000)]
        node[v.lower()] = {"mb": float(norm.mean()), "mb_lo": float(np.percentile(bs, 2.5)), "mb_hi": float(np.percentile(bs, 97.5)), "mb_raw": float(raw.mean()), "mb_std": float(norm.std(ddof=1)), "mb_raw_std": float(raw.std(ddof=1)), "n_seeds": int(len(norm))}
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
for l, ps in out.items():
    for p, vs in ps.items(): print(l, p, {v: round(x["mb"], 4) for v, x in vs.items()})
