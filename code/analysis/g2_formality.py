#!/usr/bin/env python
"""G2 formality test (PREREG): is the model MORE formal to a stigmatized-variety speaker than to an equally informal Standard-English speaker?
  d_i = dF_v,i - dF_INF,i  per seed prompt i, with dF_x,i = mean F(reply | condition x) - mean F(reply | SAE)   (F = Heylighen-Dewaele formality, higher = more formal)
  H1: mean(d) > 0.  Paired sign-flip permutation test over the seed prompts (10,000 flips), Holm within model over the stigmatized varieties; bootstrap 95% CI of mean(d).
  MV_COREF=0 python analysis/g2_formality.py --scores E1/scores.parquet --out results/E1/g2.json [--models-regex '@(dpo|instruct|final)']"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
ap = argparse.ArgumentParser(); ap.add_argument("--scores"); ap.add_argument("--out"); ap.add_argument("--models-regex", default=".*"); ap.add_argument("--col", default="formality"); ap.add_argument("--ranker", default=None, help="parquet with columns model,condition,seed_id,sampling_seed,sample,formality_ranker (harness/ranker.py score)"); a = ap.parse_args()
df = pd.read_parquet(a.scores)
if a.ranker:
    rk = pd.read_parquet(a.ranker).drop_duplicates(["model", "condition", "seed_id", "sampling_seed", "sample"]); df = df.merge(rk, on=["model", "condition", "seed_id", "sampling_seed", "sample"], how="left")
df = df[(df.kind == "response") & df[a.col].notna()]
STIG = ["AAE", "SGE", "NGP", "APP"]; rng = np.random.default_rng(0); out = {}
LIN = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L2B": "olmo2_13b", "L5": "olmo2_32b", "Q72": "qwen25_72b"}
for m, d in df.groupby("model"):
    if not re.search(a.models_regex, m): continue
    F = d.groupby(["condition", "seed_id"])[a.col].mean().unstack(0)
    if "INF" not in F or "SAE" not in F: continue
    dinf = F["INF"] - F["SAE"]; res = {}
    for v in STIG:
        if v not in F: continue
        x = ((F[v] - F["SAE"]) - dinf).dropna().values; n = len(x)
        obs = x.mean(); flips = rng.choice([-1, 1], size=(10000, n)); perm = (flips * x).mean(axis=1)
        p = (np.sum(perm >= obs) + 1) / (10001)
        bs = [x[rng.integers(0, n, n)].mean() for _ in range(2000)]
        res[v] = {"dF_v_minus_dF_inf": float(obs), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "p_one_sided": float(p), "n_seeds": int(n),
                  "dF_v": float((F[v] - F["SAE"]).mean()), "dF_inf": float(dinf.mean())}
    order = sorted(res, key=lambda k: res[k]["p_one_sided"]); k = len(order)         # Holm step-down
    run = 0.0
    for i, v in enumerate(order):
        run = max(run, min(1.0, (k - i) * res[v]["p_one_sided"])); res[v]["p_holm"] = run
    lin, st = m.split("@"); out.setdefault(LIN.get(lin, lin.lower()), {})[st.lower().replace("-", "_")] = {v.lower(): r for v, r in res.items()}
json.dump(out, open(a.out, "w"), indent=1)
for l, sts in out.items():
    for st, vs in sts.items(): print(l, st, {v: (round(x["dF_v_minus_dF_inf"], 2), round(x["p_holm"], 3)) for v, x in vs.items()})
