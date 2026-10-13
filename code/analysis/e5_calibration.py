#!/usr/bin/env python3
"""How calibrated is the one-sided E5 recipe? Descriptive check on the existing E5 runs (v1 stimuli, novel spans), motivating
the METHOD design (notes/METHOD_DESIGN.md, D24). Lead <agent>, 2026-10-05.
Per run, F_train features only (the features the E5 pairs train on):
  unsolicited  = F_train hits per 100 words in replies to Standard-English (SAE) requests (AAE + SGE items);
  reply / user = mean reply F_train rate vs mean user (prompt) F_train rate, per variety, over seed prompts (ratio of means);
  over2x       = share of seed prompts whose mean reply rate exceeds twice the user's rate;
  corr         = Spearman correlation of reply rate with user rate across seed prompts (content-confounded; the randomized
                 two-level density test of the method design replaces it).
Output: results/derived/e5_calibration.json (keys e5c.*)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP))
from detectors import split_items

TR = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
d = pd.read_parquet(EXP / "_incoming/<host>/E5/scores.parquet")


def cols(v, novel):
    return [f"ITEM.{g}.{n}.{'real_n' if novel else 'real'}" for g in TR[v] for n in split_items.ITEMS[g] if f"ITEM.{g}.{n}.real" in d.columns]


rate = lambda x, cc: 100 * x[cc].sum(axis=1) / x.n_words.clip(lower=1)
resp, prm = d[d.kind == "response"], d[d.kind == "prompt"]
user = {v: rate(prm[prm.condition == v], cols(v, False)).groupby(prm[prm.condition == v].seed_id).mean() for v in TR}
runs = {}
for m in sorted(resp.model.unique()):
    arm = m.split("@")[-1].split("_")[0]
    if not m.startswith("E5@") or arm not in ("T60k", "C60k", "d0"): continue
    r = resp[resp.model == m]; s = r[r.condition == "SAE"]
    o = {"arm": arm, "unsolicited": float((rate(s, cols("AAE", True)) + rate(s, cols("SGE", True))).mean())}
    for v in TR:
        rv = r[r.condition == v]; y = rate(rv, cols(v, True)).groupby(rv.seed_id).mean()
        j = pd.concat([y.rename("y"), user[v].rename("x")], axis=1).dropna()
        o[v.lower()] = {"reply": float(j.y.mean()), "user": float(j.x.mean()), "ratio": float(j.y.mean() / j.x.mean()),
                        "over2x": float((j.y > 2 * j.x).mean()), "corr": None if j.y.std() == 0 else float(j[["x", "y"]].corr(method="spearman").iloc[0, 1]),
                        "ratio__digits": 2, "over2x__digits": 2, "corr__digits": 2}
    runs[m.split("@")[-1].lower()] = o
T = [v for v in runs.values() if v["arm"] == "T60k"]; C = [v for v in runs.values() if v["arm"] in ("C60k", "d0")]
out = {"runs": runs,
       "t60k": {"unsolicited_min": min(v["unsolicited"] for v in T), "unsolicited_max": max(v["unsolicited"] for v in T),
                "aae_ratio_min": min(v["aae"]["ratio"] for v in T), "aae_ratio_max": max(v["aae"]["ratio"] for v in T),
                "sge_ratio_min": min(v["sge"]["ratio"] for v in T), "sge_ratio_max": max(v["sge"]["ratio"] for v in T),
                "sge_over2x_min": min(v["sge"]["over2x"] for v in T), "sge_over2x_max": max(v["sge"]["over2x"] for v in T), "n_runs": len(T),
                **{k + "__digits": 2 for k in ("unsolicited_min", "unsolicited_max", "aae_ratio_min", "aae_ratio_max", "sge_ratio_min", "sge_ratio_max", "sge_over2x_min", "sge_over2x_max")}},
       "controls_unsolicited_max": max(v["unsolicited"] for v in C), "controls_unsolicited_max__digits": 2,
       "user_density_per100w": {v.lower(): float(user[v].mean()) for v in TR},
       "note": "F_train features only, novel spans; v1 stimuli; descriptive (existing E5 runs)"}
(EXP / "results/derived/e5_calibration.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["t60k"], indent=1)); print("controls unsolicited max", out["controls_unsolicited_max"])
