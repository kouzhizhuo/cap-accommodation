#!/usr/bin/env python
"""METHOD evaluation (CAP; METHOD_DESIGN v3/v4) on the F300 fresh test.  Reads detector score parquets of F300 generations (kind prompt/response; models E5@<run>[-B3|-B4|-B34] and Q3@<run>[...]).
Per run (request-level, bootstrap over the 300 requests): crheld (primary; CR on the held-out features of AAE/SGE), ftrain (CR on trained features = calibration, target <= 1), crg (NGP/APP all-feature rate CR),
split S, calibration tail (share of requests with prompt density > 0 whose mean reply density exceeds 2x the prompt's, by variety), insertion on the SAE twins (F_train and all-feature hits /100 words, share of replies with >= 1 hit).
Arms: run id minus _s<k>; seeds are averaged.  Tests: arm A - arm B (mean over runs of A minus mean over runs of B on the SAME bootstrap request sample), 95% percentile CI, B=2000.
  MV_COREF=0 python analysis/method_eval.py --scores E5/score_f300.parquet [more.parquet] --out results/METHOD/f300.json [--tests CAP60k:C60k CAP60k:d0 CAP60k:T60k]"""
import argparse, json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors.pilot_summary import rate_seed_means
from detectors import split_items

ap = argparse.ArgumentParser(); ap.add_argument("--scores", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tests", nargs="*", default=[]); ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args()
df = pd.concat([pd.read_parquet(p) for p in a.scores], ignore_index=True)
df["model"] = df.model.str.replace(r"^(\w+)@\1@", r"\1@", regex=True)
resp, prm = df[df.kind == "response"], df[df.kind == "prompt"].drop_duplicates(["condition", "seed_id"])
HELD = {"AAE": ["AAE.morph.held", "AAE.lex.held"], "SGE": ["SGE.part.held", "SGE.lex.held"]}
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
ITEMCOLS = [c for c in df.columns if c.startswith("ITEM.") and c.endswith(".real_n")]
sae_all = resp[resp.condition == "SAE"]                      # FP audit pooled over every run's SAE replies (pre-registered 2% rule, as E5)
ITEM_FP = {c: float((sae_all[c] > 0).mean()) for c in ITEMCOLS}
EXCL = {c for c in ITEMCOLS if (".held." in c or ".train." in c) and ITEM_FP[c] > 0.02}
SEEDS = sorted(prm.seed_id.unique()); SI = {s: i for i, s in enumerate(SEEDS)}


def item_cols(groups, v, novel, clean=True):
    cols = [f"ITEM.{g}.{n}.{'real_n' if novel else 'real'}" for g in groups[v] for n in split_items.ITEMS[g]]
    return [c for c in cols if c in df.columns and (not clean or c.replace(".real", ".real_n") not in EXCL)]


def series(d, cols):
    r = 100.0 * d[cols].sum(axis=1) / d.n_words.clip(lower=1)
    s = r.groupby([d.condition, d.seed_id]).mean().unstack(0)
    return s.reindex(SEEDS)


def vec(s, c): return s[c].values if c in s.columns else np.full(len(SEEDS), np.nan)


def crv(Rv, Rs, Pv, Ps, idx):
    pc = np.nanmean((Pv - Ps)[idx]); return float(np.nanmean((Rv - Rs)[idx]) / pc) if abs(pc) > 1e-9 else np.nan


def run_vectors(model):
    d = resp[resp.model == model]; V = {}
    for v in ("AAE", "SGE"):
        for tag, groups in (("held", HELD), ("train", TRAIN)):
            cols = item_cols(groups, v, True); pcols = item_cols(groups, v, False)
            R, P = series(d, cols), series(prm, pcols)
            V[(tag, v)] = (vec(R, v), vec(R, "SAE"), vec(P, v), vec(P, "SAE"))
    for v in ("NGP", "APP", "INF", "BRE", "AUS", "AAE", "SGE"):
        if v in set(d.condition):
            R, P = rate_seed_means(d, v, True).reindex(SEEDS), rate_seed_means(prm, v, False).reindex(SEEDS)
            V[("all", v)] = (vec(R, v), vec(R, "SAE"), vec(P, v), vec(P, "SAE"))
    return V


def metric(V, name, idx):
    if name in ("crheld", "ftrain"):
        tag = "held" if name == "crheld" else "train"; xs = [crv(*V[(tag, v)], idx) for v in ("AAE", "SGE") if (tag, v) in V]; return float(np.nanmean(xs)) if xs else np.nan
    if name == "crg":
        xs = [crv(*V[("all", v)], idx) for v in ("NGP", "APP") if ("all", v) in V]; return float(np.nanmean(xs)) if xs else np.nan
    if name == "split":
        p = [crv(*V[("all", v)], idx) for v in ("INF", "BRE", "AUS") if ("all", v) in V]; s = [crv(*V[("all", v)], idx) for v in ("AAE", "SGE", "NGP", "APP") if ("all", v) in V]
        return float(np.nanmean(p) - np.nanmean(s)) if p and s else np.nan
    if name.startswith("tail2x_"):
        v = name.split("_")[1]; Rv, Rs, Pv, Ps = V[("train", v)]; ok = (Pv > 0)[idx]
        return float(np.mean((Rv[idx][ok]) > 2 * Pv[idx][ok])) if ok.sum() else np.nan
    raise KeyError(name)


METRICS = ["crheld", "ftrain", "crg", "split", "tail2x_AAE", "tail2x_SGE"]
rng = np.random.default_rng(0); B = a.B; N = len(SEEDS); BOOT = [rng.integers(0, N, N) for _ in range(B)]; FULL = np.arange(N)
models = sorted(resp.model.unique()); VEC = {m: run_vectors(m) for m in models}
out = {"definitions": __doc__.split("\n")[1:5], "n_requests": N, "heldout_items_excluded_fp_gt_2pct": sorted(EXCL), "runs": {}}
for m in models:
    d = resp[resp.model == m]; r = {k: metric(VEC[m], k, FULL) for k in METRICS}
    r["bootstrap_ci"] = {k: [float(np.nanpercentile([metric(VEC[m], k, b) for b in BOOT[:400]], q)) for q in (2.5, 97.5)] for k in ("crheld", "ftrain")}
    sae = d[d.condition == "SAE"]
    for v in ("AAE", "SGE"):
        tc = item_cols(TRAIN, v, True); ac = [c for c in df.columns if c.startswith(f"{v}.") and c.endswith(".real_n")]
        r[f"insertion_sae_train_{v}"] = float((100.0 * sae[tc].sum(axis=1) / sae.n_words.clip(lower=1)).mean()) if tc else None
        r[f"insertion_sae_any_{v}"] = float((sae[tc].sum(axis=1) > 0).mean()) if tc else None
        r[f"insertion_sae_all_{v}"] = float((100.0 * sae[ac].sum(axis=1) / sae.n_words.clip(lower=1)).mean()) if ac else None
    r["mean_reply_words"] = float(d.n_words.mean()); out["runs"][m] = r


def arm_of(m):
    tag, rid = m.split("@", 1); sfx = re.search(r"(-B3|-B34|-B4)$", rid); rid2 = rid[:sfx.start()] if sfx else rid
    return tag, re.sub(r"_s\d+$", "", rid2) + (sfx.group(1) if sfx else "")


arms = {}
for m in models: arms.setdefault(arm_of(m)[1], []).append(m)
out["arms"] = {}
for arm, ms in arms.items():
    out["arms"][arm] = {"runs": ms, "n_runs": len(ms)}
    for k in METRICS:
        vals = [out["runs"][m][k] for m in ms if out["runs"][m][k] == out["runs"][m][k]]
        if vals: out["arms"][arm][k] = float(np.mean(vals)); out["arms"][arm][k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    for k in ("insertion_sae_train_AAE", "insertion_sae_train_SGE", "insertion_sae_any_AAE", "insertion_sae_any_SGE", "insertion_sae_all_AAE", "insertion_sae_all_SGE", "mean_reply_words"):
        vals = [out["runs"][m][k] for m in ms if out["runs"][m][k] is not None]
        if vals: out["arms"][arm][k] = float(np.mean(vals))
out["tests"] = {}
for t in a.tests:
    A, Bn = t.split(":")
    if A not in arms or Bn not in arms: continue
    res = {}
    for k in METRICS:
        f = lambda idx: np.nanmean([metric(VEC[m], k, idx) for m in arms[A]]) - np.nanmean([metric(VEC[m], k, idx) for m in arms[Bn]])
        est = f(FULL); bs = [f(b) for b in BOOT]; res[k] = {"diff": float(est), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
    # per-run M1: every run of A minus the mean of B on the same request sample (CI > 0 in every CAP run)
    res["per_run_crheld"] = {}
    for m in arms[A]:
        f = lambda idx: metric(VEC[m], "crheld", idx) - np.nanmean([metric(VEC[x], "crheld", idx) for x in arms[Bn]])
        bs = [f(b) for b in BOOT]; res["per_run_crheld"][m] = {"diff": float(f(FULL)), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
    out["tests"][t] = res
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
for arm, v in out["arms"].items(): print(arm, {k: round(x, 4) for k, x in v.items() if isinstance(x, float)})
for t, v in out["tests"].items(): print(t, {k: (round(x["diff"], 4), round(x["lo"], 4), round(x["hi"], 4)) for k, x in v.items() if k != "per_run_crheld"})
