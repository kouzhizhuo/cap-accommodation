#!/usr/bin/env python
"""1B pilot GATE on V300 (D24 'Gate'): CAP-60k s0 vs the control (C60k, default s0).  V300 = 300 validation requests (SAE twin + AAE/SGE F_train renders), 3 sampling seeds x 2 samples.
(1) end to end: the run has V300 generations;  (2) trained-feature CR (CR on F_train items, AAE+SGE mean) CAP - control, request bootstrap CI-lo > 0;
(3) calibration = mean reply F_train rate / mean user F_train rate (ratio of means) in [0.5, 1.5] for BOTH varieties;  (4) insertion (F_train + F_heldout hits per 100 words in replies to the SAE twins) CAP - control CI-hi <= 0.05.
  MV_COREF=0 python analysis/gate_pilot.py --scores E5/score_v300.parquet --treat 'E5@CAP60k_s0$' --control 'E5@C60k_s0$' --out results/METHOD/gate_pilot.json   (--treat/--control are regexes on the normalised model name)"""
import argparse, json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors import split_items
ap = argparse.ArgumentParser(); ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--treat", required=True); ap.add_argument("--control", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args()
df = pd.concat([pd.read_parquet(p) for p in a.scores], ignore_index=True); df["model"] = df.model.str.replace(r"^(\w+)@\1@", r"\1@", regex=True)
resp, prm = df[df.kind == "response"], df[df.kind == "prompt"].drop_duplicates(["condition", "seed_id"])
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}; HELD = {"AAE": ["AAE.morph.held", "AAE.lex.held"], "SGE": ["SGE.part.held", "SGE.lex.held"]}
ITEMCOLS = [c for c in df.columns if c.startswith("ITEM.") and c.endswith(".real_n")]
sae_all = resp[resp.condition == "SAE"]; EXCL = {c for c in ITEMCOLS if (".held." in c or ".train." in c) and float((sae_all[c] > 0).mean()) > 0.02}
SEEDS = sorted(prm.seed_id.unique())


def cols(groups, v, novel):
    cs = [f"ITEM.{g}.{n}.{'real_n' if novel else 'real'}" for g in groups[v] for n in split_items.ITEMS[g]]; return [c for c in cs if c in df.columns and c.replace(".real", ".real_n") not in EXCL]


def ser(d, cs):
    r = 100.0 * d[cs].sum(axis=1) / d.n_words.clip(lower=1); return r.groupby([d.condition, d.seed_id]).mean().unstack(0).reindex(SEEDS)


def per_run(model):
    d = resp[resp.model == model]; out = {}
    for v in ("AAE", "SGE"):
        R, P = ser(d, cols(TRAIN, v, True)), ser(prm, cols(TRAIN, v, False))
        out[v] = (R[v].values, R["SAE"].values, P[v].values, P["SAE"].values)
    sae = d[d.condition == "SAE"]; both = cols(TRAIN, "AAE", True) + cols(HELD, "AAE", True) + cols(TRAIN, "SGE", True) + cols(HELD, "SGE", True)
    out["ins"] = ser(sae, both)["SAE"].values
    return out


def crv(x, idx): Rv, Rs, Pv, Ps = x; pc = np.nanmean((Pv - Ps)[idx]); return float(np.nanmean((Rv - Rs)[idx]) / pc) if abs(pc) > 1e-9 else np.nan


def stats(R, idx):
    cr = float(np.nanmean([crv(R[v], idx) for v in ("AAE", "SGE")])); cal = {v: float(np.nanmean(R[v][0][idx]) / np.nanmean(R[v][2][idx])) for v in ("AAE", "SGE")}
    return cr, cal, float(np.nanmean(R["ins"][idx]))


models = sorted(resp.model.unique()); T = [m for m in models if re.search(a.treat, m)]; C = [m for m in models if re.search(a.control, m)]
out = {"treat_runs": T, "control_runs": C, "n_requests": len(SEEDS)}
if not T or not C:
    out["gate"] = {"end_to_end": bool(T), "note": "control or treatment V300 generations missing"}; json.dump(out, open(a.out, "w"), indent=1); print(out); sys.exit(0)
RT = {m: per_run(m) for m in T}; RC = {m: per_run(m) for m in C}; rng = np.random.default_rng(20261012); N = len(SEEDS); FULL = np.arange(N); B = a.B
def agg(RS, idx):
    s = [stats(R, idx) for R in RS.values()]; return np.mean([x[0] for x in s]), {v: np.mean([x[1][v] for x in s]) for v in ("AAE", "SGE")}, np.mean([x[2] for x in s])
cT, calT, insT = agg(RT, FULL); cC, calC, insC = agg(RC, FULL)
bs = []
for _ in range(B):
    idx = rng.integers(0, N, N); t, ct, it = agg(RT, idx); c, cc, ic = agg(RC, idx); bs.append((t - c, it - ic, ct["AAE"], ct["SGE"], t, c))
bs = np.array(bs); q = lambda j: [float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]
out["trained_feature_cr"] = {"treat": float(cT), "control": float(cC), "diff": float(cT - cC), "diff_ci": q(0)}
out["calibration_ratio_of_means"] = {"treat": calT, "control": calC, "treat_ci_AAE": q(2), "treat_ci_SGE": q(3)}
out["insertion_per100w"] = {"treat": float(insT), "control": float(insC), "diff": float(insT - insC), "diff_ci": q(1)}
out["gate"] = {"end_to_end": True, "trained_cr_ci_lo_gt0": bool(q(0)[0] > 0), "calibration_in_0.5_1.5_both": bool(all(0.5 <= calT[v] <= 1.5 for v in ("AAE", "SGE"))), "insertion_ci_hi_le_0.05": bool(q(1)[1] <= 0.05)}
out["gate"]["PASS"] = all(out["gate"][k] for k in ("end_to_end", "trained_cr_ci_lo_gt0", "calibration_in_0.5_1.5_both", "insertion_ci_hi_le_0.05"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))
