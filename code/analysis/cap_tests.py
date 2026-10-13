#!/usr/bin/env python3
"""Pre-declared CAP analysis (D24; notes/METHOD_DESIGN.md §5-6). Lead <agent>, 2026-10-05, written BEFORE any CAP result and
validated on the existing E5 runs (T60k vs C60k reproduces the e5_tests pooled 60k effect).

Input: a scores parquet in the E5 schema (kind, model, condition, seed_id, n_words, ITEM.* columns); arms are regexes over model.
Per training run r and request i (as analysis/e5_tests.py): CR contrasts against the SAE twin, divided by the mean prompt contrast,
  crheld (held-out features of AAE+SGE, pre-registered 2% FP exclusion), ftrain (trained features), crg (NGP+APP, all features);
per run and variety: calibration = mean reply F_train rate / mean user F_train rate (ratio of means, per 100 words), tail = share of
  requests (with a nonzero user rate) whose reply rate exceeds twice the user's, insertion = F_train + F_heldout hits per 100 words in replies to SAE twins;
density intervention (conditions <v>_lo / <v>_hi): tracking slope beta from y_ij = a_i + beta x_ij (two levels -> within-request
  slope), requests with an achieved user contrast < 0.5 per 100 words excluded;
overrides (conditions O_*): counted-feature rate per 100 words in replies.
Arm contrasts: Delta = mean over the treatment arm's runs - mean over the comparison arm's runs; 95% CI from a two-level bootstrap
(runs within arm with replacement, then requests, paired across arms), B 2000, seed 20261012, one stream per estimate.
Criteria (D24): M1 crheld Delta CI-lo > 0 and Delta >= MIN_GAIN; M2 |calibration - 1| below the reference arm's (CI of the difference
< 0) per variety, tail below the reference's, insertion Delta CI-hi <= 0.05; M3 beta CI-lo > 0 and beta - beta_ref CI-lo > 0;
M5 override rate Delta CI-hi <= 0.05. M4 (IFEval non-inferiority) is computed from per-prompt IFEval files elsewhere.
  python analysis/cap_tests.py --scores E5/scores.parquet --treat 'E5@T60k_s\\d+' --control 'E5@C60k_s\\d+' --ref 'E5@T60k_s\\d+' --out /tmp/x.json
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP))
from detectors import split_items
from detectors.pilot_summary import rate_seed_means
import detectors.pilot_summary as PS
# pilot_summary resolves its validation / FP-audit files relative to the cwd; pin them so the script runs from any host layout
_VAL = next((str(c) for c in (EXP / 'detectors/validation_synthetic.json', EXP / 'results/detectors/validation_synthetic.json',
                              EXP / '_incoming/<host>/detectors/validation_synthetic.json') if c.exists()), None)
_FPS = [c for c in (EXP / 'detectors/fp_audit_SAE.json', EXP / 'E1/fp_audit_SAE.json', EXP / '_incoming/<host>/detectors/fp_audit_SAE.json') if c.exists()]
PS.retained_set(val=_VAL, fp=str(max(_FPS, key=lambda q: q.stat().st_mtime)) if _FPS else None)

MIN_GAIN, INS_MARGIN, TRACK_MIN, B, SEED = 0.02, 0.05, 0.5, 2000, 20261012
HELD = {"AAE": ["AAE.morph.held", "AAE.lex.held"], "SGE": ["SGE.part.held", "SGE.lex.held"]}
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--treat", required=True); ap.add_argument("--control", required=True)
ap.add_argument("--ref", help="reference arm for M2/M3 (B5: one-sided recipe)"); ap.add_argument("--out", required=True)
ap.add_argument("--no-excl", action="store_true", help="sensitivity only: skip the pre-registered 2% false-positive exclusion of held-out/trained items")
a = ap.parse_args()
df = pd.concat([pd.read_parquet(p) for p in a.scores], ignore_index=True)
# post_extra.sh tags the density-intervention and override generations <TAG>d@<run> / <TAG>o@<run>: merge them into the run <TAG>@<run>.
# Validation generations (<TAG>v@<run>) are NOT merged: the gate reads them separately and they never enter the test estimands.
df["model"] = df["model"].str.replace(r"^([A-Za-z0-9]+)@\1[do]?@", r"\1@", regex=True)   # harvested names repeat the tag (E5@E5d@C60k_s0, Q3@Q3@INSTRUCT)
df["model"] = df["model"].str.replace(r"^([A-Za-z0-9]+?)[do]@", r"\1@", regex=True)
resp, prm = df[df.kind == "response"], df[df.kind == "prompt"]
sae_all = resp[resp.condition == "SAE"]
EXCL = {c for c in df.columns if c.startswith("ITEM.") and (".held." in c or ".train." in c) and c.endswith(".real_n") and (sae_all[c] > 0).mean() > 0.02}
if a.no_excl: EXCL = set()   # sensitivity analysis (2026-10-07), never the registered estimand


def cols(groups, v, novel):
    cs = [f"ITEM.{g}.{n}.{'real_n' if novel else 'real'}" for g in groups[v] for n in split_items.ITEMS[g]]
    return [c for c in cs if c in df.columns and c.replace(".real_n", ".real").replace(".real", ".real_n") not in EXCL]


def rate(d, cs):   # per (condition, seed_id): mean over rows of hits per 100 words
    return (100.0 * d[cs].sum(axis=1) / d.n_words.clip(lower=1)).groupby([d.condition, d.seed_id]).mean().unstack(0)


def runs(rx): return sorted(m for m in resp.model.unique() if re.fullmatch(rx, m))


def per_run(m):
    """-> dict of per-request Series (index seed_id) and per-run scalars for one training run."""
    d = resp[resp.model == m]; o = {}
    for name, G in (("crheld", HELD), ("ftrain", TRAIN)):
        parts = []
        for v in ("AAE", "SGE"):
            R, P = rate(d[d.condition.isin([v, "SAE"])], cols(G, v, True)), rate(prm[prm.condition.isin([v, "SAE"])], cols(G, v, False))
            if v in R and "SAE" in R and v in P:
                pc = (P[v] - P.get("SAE", 0)).mean()
                if abs(pc) > 1e-9: parts.append(((R[v] - R["SAE"]) / pc).rename(v))
        o[name] = pd.concat(parts, axis=1).mean(axis=1) if parts else None
    parts = []
    for v in ("NGP", "APP"):
        if v not in set(d.condition): continue
        R, P = rate_seed_means(d, v, True), rate_seed_means(prm, v, False)
        pc = (P[v] - P["SAE"]).mean()
        if abs(pc) > 1e-9: parts.append(((R[v] - R["SAE"]) / pc).rename(v))
    o["crg"] = pd.concat(parts, axis=1).mean(axis=1) if parts else None
    for v in ("AAE", "SGE"):   # calibration on trained features (reply vs user, per 100 words)
        R, P = rate(d[d.condition == v], cols(TRAIN, v, True)), rate(prm[prm.condition == v], cols(TRAIN, v, False))
        o[f"reply_{v}"], o[f"user_{v}"] = (R[v] if v in R else None), (P[v] if v in P else None)
    s = d[d.condition == "SAE"]
    ins_cols = sorted(set(cols(TRAIN, "AAE", True) + cols(TRAIN, "SGE", True) + cols(HELD, "AAE", True) + cols(HELD, "SGE", True)))
    o["insertion"] = (100.0 * s[ins_cols].sum(axis=1) / s.n_words.clip(lower=1)).groupby(s.seed_id).mean()
    for v in ("AAE", "SGE"):   # density intervention (harness/e5/make_eval_sets.py writes <v>_lo / <v>_hi)
        lo, hi = next(((f"{v}{s}lo", f"{v}{s}hi") for s in ("_", "@") if {f"{v}{s}lo", f"{v}{s}hi"} <= set(d.condition)), (None, None))
        if lo:
            Y = rate(d[d.condition.isin([lo, hi])], cols(TRAIN, v, True)); X = rate(prm[prm.condition.isin([lo, hi])], cols(TRAIN, v, False))
            j = pd.DataFrame({"dy": Y[hi] - Y[lo], "dx": X[hi] - X[lo]}).dropna(); o[f"track_{v}"] = j[j.dx.abs() >= TRACK_MIN]
    for c in sorted(set(d.condition)):   # overrides (make_eval_sets.py: O_STD / O_FORMAL / O_NOMIMIC on dialect prompts; O_AAE / O_SGE on SAE prompts)
        if not c.startswith("O_"): continue
        vs = [c[2:]] if c[2:] in TRAIN else ["AAE", "SGE"]   # explicit-variety requests count that variety; restraint overrides count both
        cc = sorted({x for v in vs for x in cols(TRAIN, v, True) + cols(HELD, v, True)})
        if cc: dd = d[d.condition == c]; o[f"ov:{c}"] = (100.0 * dd[cc].sum(axis=1) / dd.n_words.clip(lower=1)).groupby(dd.seed_id).mean()
    return o


def mat(rs, key):
    ser = [r[key] for r in rs if r.get(key) is not None]
    return pd.concat([x.rename(f'run{k}') for k, x in enumerate(ser)], axis=1) if ser else None   # requests x runs (unique column names)


def two_level(fT, fC, stat, seed=SEED):
    """stat(TT, CC, idx) on request x run DataFrames (aligned rows); returns point, lo, hi over the two-level bootstrap."""
    rng = np.random.default_rng(seed)
    idx0 = np.arange(len(fT)); point = stat(fT, fC, idx0); bs = []
    for _ in range(B):
        tt = fT.iloc[:, rng.integers(0, fT.shape[1], fT.shape[1])]; cc = fC.iloc[:, rng.integers(0, fC.shape[1], fC.shape[1])]
        bs.append(stat(tt, cc, rng.integers(0, len(fT), len(fT))))
    return {"delta": float(point), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}


diff_mean = lambda T, C, i: float(np.nanmean(T.mean(axis=1).values[i] - C.mean(axis=1).values[i]))
out = {"spec": "D24 pre-declared CAP analysis (analysis/cap_tests.py)", "excluded_items": sorted(EXCL), "arms": {}}
R = {k: [per_run(m) for m in runs(rx)] for k, rx in (("treat", a.treat), ("control", a.control), ("ref", a.ref)) if rx}
for k, rx in (("treat", a.treat), ("control", a.control), ("ref", a.ref)):
    if rx: out["arms"][k] = runs(rx)
for metric in ("crheld", "ftrain", "crg", "insertion"):
    T, C = mat(R["treat"], metric), mat(R["control"], metric)
    if T is None or C is None: continue
    j = T.join(C, how="inner", lsuffix="_t", rsuffix="_c"); T, C = j.iloc[:, :T.shape[1]], j.iloc[:, T.shape[1]:]
    out[metric] = {**two_level(T, C, diff_mean), "treat_runs": [float(T[c].mean()) for c in T], "control_runs": [float(C[c].mean()) for c in C], "n_requests": len(j)}
    _rng = np.random.default_rng(SEED); _dv = (T.mean(axis=1) - C.mean(axis=1)).values; _bs = [np.nanmean(_dv[_rng.integers(0, len(_dv), len(_dv))]) for _ in range(B)]
    out[metric]["reqonly_lo"], out[metric]["reqonly_hi"] = float(np.nanpercentile(_bs, 2.5)), float(np.nanpercentile(_bs, 97.5))   # sensitivity: request-only bootstrap
calib = {}
for v in ("AAE", "SGE"):
    def arm_ratio(rs):
        Rm, Um = mat(rs, f"reply_{v}"), mat(rs, f"user_{v}")
        return (None, None) if Rm is None else (Rm, Um.iloc[:, 0])
    Rt, U = arm_ratio(R["treat"])
    if Rt is None: continue
    rat = lambda M, i: float(np.nanmean(M.mean(axis=1).values[i]) / np.nanmean(U.values[i]))
    def tail(M, i):   # over requests whose user wrote at least one trained feature (else any hit would count as 'over')
        u, r = U.values[i], M.mean(axis=1).values[i]; k = u > 0
        return float(np.mean(r[k] > 2 * u[k])) if k.any() else float('nan')
    calib[v] = {"treat_ratio": rat(Rt, np.arange(len(U))), "treat_tail": tail(Rt, np.arange(len(U))), "treat_runs_ratio": [float(Rt[c].mean() / U.mean()) for c in Rt]}
    if a.ref and R.get("ref"):
        Rr, _ = arm_ratio(R["ref"])
        calib[v]["ref_ratio"] = rat(Rr, np.arange(len(U))); calib[v]["ref_tail"] = tail(Rr, np.arange(len(U)))
        calib[v]["abs_dev_diff"] = two_level(Rt, Rr, lambda T, C, i: abs(rat(T, i) - 1) - abs(rat(C, i) - 1))
        calib[v]["tail_diff"] = two_level(Rt, Rr, lambda T, C, i: tail(T, i) - tail(C, i))
out["calibration"] = calib
track = {}
for v in ("AAE", "SGE"):
    def betas(rs):
        js = [r.get(f"track_{v}") for r in rs if r.get(f"track_{v}") is not None]
        return js
    jt = betas(R["treat"])
    if not jt: continue
    def beta_arm(js, rng=None):
        tot_xy = tot_xx = 0.0
        for j in js:
            ii = np.arange(len(j)) if rng is None else rng.integers(0, len(j), len(j))
            tot_xy += float((j.dy.values[ii] * j.dx.values[ii]).sum()); tot_xx += float((j.dx.values[ii] ** 2).sum())
        return tot_xy / tot_xx if tot_xx > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    bs = [beta_arm([jt[k] for k in rng.integers(0, len(jt), len(jt))], rng) for _ in range(B)]
    track[v] = {"beta": beta_arm(jt), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)), "per_run": [beta_arm([j]) for j in jt]}
    if R.get("ref"):
        jr = betas(R["ref"])
        if jr:
            rng = np.random.default_rng(SEED)
            bd = [beta_arm([jt[k] for k in rng.integers(0, len(jt), len(jt))], rng) - beta_arm([jr[k] for k in rng.integers(0, len(jr), len(jr))], rng) for _ in range(B)]
            track[v]["minus_ref"] = {"delta": beta_arm(jt) - beta_arm(jr), "lo": float(np.nanpercentile(bd, 2.5)), "hi": float(np.nanpercentile(bd, 97.5))}
out["tracking"] = track
ov = {}
for key in sorted({k for r in R["treat"] for k in r if k.startswith("ov:")}):
    T, C = mat(R["treat"], key), mat(R["control"], key)
    if T is not None and C is not None:
        j = T.join(C, how="inner", lsuffix="_t", rsuffix="_c"); ov[key[3:]] = two_level(j.iloc[:, :T.shape[1]], j.iloc[:, T.shape[1]:], diff_mean)
        ov[key[3:]].update({"treat": float(j.iloc[:, :T.shape[1]].mean(axis=1).mean()), "control": float(j.iloc[:, T.shape[1]:].mean(axis=1).mean())})   # arm levels (table), 2026-10-06
out["overrides"] = ov
crit = {}
if "crheld" in out: crit["M1"] = bool(out["crheld"]["lo"] > 0 and out["crheld"]["delta"] >= MIN_GAIN)
if calib and all("abs_dev_diff" in c for c in calib.values()) and "insertion" in out:
    crit["M2"] = bool(all(c["abs_dev_diff"]["hi"] < 0 and c["tail_diff"]["hi"] < 0 for c in calib.values()) and out["insertion"]["hi"] <= INS_MARGIN)
if track and all("minus_ref" in t for t in track.values()): crit["M3"] = bool(all(t["lo"] > 0 and t["minus_ref"]["lo"] > 0 for t in track.values()))
if "O_STD" in ov: crit["M5"] = bool(ov["O_STD"]["hi"] <= INS_MARGIN)
out["criteria"] = crit
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in out if k not in ("excluded_items",)}, indent=1)[:4000])
