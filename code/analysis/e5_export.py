#!/usr/bin/env python
"""E5 export: per run -> crheld, crg, split (rate-based CR, productive spans, per-seed bootstrap not needed here), then rows with std over training seeds, keys e5.<row>.<col>
(notes/RES_KEYS.md).  Definitions (all RATE-based per 100 words, no opportunity gate; features flagged by the SAE-reply FP audit are excluded):
  crheld = mean over {AAE, SGE} of CR on the F_HELDOUT items of that variety (detectors/split_items.py, split v1.1)
  crg    = mean over the generalisation varieties {NGP, APP} of the all-feature rate CR   (SCO/IND join when their stimuli exist; APP shares Multi-VALUE rules with AAE: not independent)
  split  = mean rate-CR over prestige cues {INF, BRE, AUS} minus mean over {AAE, SGE, NGP, APP}
  ftrain = mean over {AAE, SGE} of CR on the F_TRAIN items (trained-feature uptake; same FP exclusion rule as crheld) -- per run in `runs`, arm mean/std in the row, key e5.<row>.ftrain
  ifeval = prompt-level strict accuracy from results/E5/ifeval.json when present (else null).  Rows: d0 (d0_s*), aa1k/aa5k/aa20k (T*), ctl5k/ctl20k (C*), anti1k/anti5k (A*), sft/dpo (released OLMo-2-0425-1B stages, L1@sft/dpo).
  MV_COREF=0 python analysis/e5_export.py --scores E5/scores.parquet --out results/E5/summary.json"""
import argparse, json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors.pilot_summary import rate_seed_means
ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
df = pd.read_parquet(a.scores); resp, prm = df[df.kind == "response"], df[df.kind == "prompt"]
HELD = {"AAE": ["AAE.morph.held", "AAE.lex.held"], "SGE": ["SGE.part.held", "SGE.lex.held"]}
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
ROW = {"d0": "d0", "T1k": "aa1k", "T5k": "aa5k", "T20k": "aa20k", "T60k": "aa60k", "C5k": "ctl5k", "C20k": "ctl20k", "C60k": "ctl60k", "A1k": "anti1k", "A5k": "anti5k"}


def cr(Rc, Rs, Pc, Ps):
    idx = Rc.index.intersection(Pc.index); ai = (Rc.loc[idx] - Rs.loc[idx]).values; pc = (Pc.loc[idx] - Ps.loc[idx]).values
    return float(np.nanmean(ai) / np.nanmean(pc)) if abs(np.nanmean(pc)) > 1e-9 else np.nan


# pre-registered rule: a held-out item whose share of SAE-condition replies with >= 1 hit exceeds 2% (any E5 model) is excluded from `crheld`; `crheld_all` keeps every item
ITEMCOLS = [c for c in df.columns if c.startswith("ITEM.") and c.endswith(".real_n")]
sae = resp[resp.condition == "SAE"]
ITEM_FP = {c: float((sae[c] > 0).mean()) for c in ITEMCOLS}
EXCLUDED = sorted(c for c in ITEMCOLS if (".held." in c or ".train." in c) and ITEM_FP[c] > 0.02)


def held_series(d, v, novel, clean=False, groups=None):
    cols = [f"ITEM.{g}.{n}.{'real_n' if novel else 'real'}" for g in (groups or HELD)[v] for n in __import__("detectors.split_items", fromlist=["ITEMS"]).ITEMS[g]]
    if clean: cols = [c for c in cols if c.replace(".real", ".real_n") not in EXCLUDED]
    return (100.0 * d[cols].sum(axis=1) / d.n_words.clip(lower=1)).groupby([d.condition, d.seed_id]).mean().unstack(0)


def run_metrics(model):
    d = resp[resp.model == model]; cond = set(d.condition); m = {}
    allv = {}
    for v in ("AAE", "SGE", "INF", "BRE", "AUS", "NGP", "APP"):
        if v not in cond: continue
        R, P = rate_seed_means(d, v, True), rate_seed_means(prm, v, False)
        allv[v] = cr(R[v], R["SAE"], P[v], P["SAE"])
    for key, clean in (("crheld", True), ("crheld_all", False)):
        held = []
        for v in ("AAE", "SGE"):
            if v in cond:
                R, P = held_series(d, v, True, clean), held_series(prm, v, False, clean); held.append(cr(R[v], R["SAE"], P[v], P["SAE"]))
        m[key] = float(np.nanmean(held)) if held else None
    ft = []
    for v in ("AAE", "SGE"):
        if v in cond:
            R, P = held_series(d, v, True, True, TRAIN), held_series(prm, v, False, True, TRAIN); ft.append(cr(R[v], R["SAE"], P[v], P["SAE"]))
    m["ftrain"] = float(np.nanmean(ft)) if ft else None
    g = [allv[v] for v in ("NGP", "APP") if v in allv]; m["crg"] = float(np.nanmean(g)) if g else None
    pres = [allv[v] for v in ("INF", "BRE", "AUS") if v in allv]; stig = [allv[v] for v in ("AAE", "SGE", "NGP", "APP") if v in allv]
    m["split"] = float(np.nanmean(pres) - np.nanmean(stig)) if pres and stig else None
    m["cr_rate_by_variety"] = {k.lower(): round(x, 4) for k, x in allv.items()}
    return m


runs = {}
for model in sorted(resp.model.unique()):
    lin, st = model.split("@")
    if lin == "E5":
        rid = st; base = re.sub(r"_s\d+$", "", rid); row = ROW.get(base.split("_")[0] if base.split("_")[0] in ROW else base)
        if row is not None: runs.setdefault(row, {})[rid] = run_metrics(model)
    elif lin == "L1" and st in ("sft", "dpo"):
        runs.setdefault(st, {})[model] = run_metrics(model)
out = {"definitions": __doc__.split("\n")[1:7], "heldout_item_fp_sae_replies": {k: round(v, 4) for k, v in ITEM_FP.items() if ".held." in k}, "heldout_items_excluded_fp_gt_2pct": EXCLUDED, "runs": {}}
for row, rs in runs.items():
    out["runs"][row] = {rid: {k: v for k, v in m.items()} for rid, m in rs.items()}
    for col in ("crg", "crheld", "crheld_all", "split", "ftrain"):
        vals = [m[col] for m in rs.values() if m[col] is not None and not np.isnan(m[col])]
        if vals:
            out.setdefault(row, {})[col] = round(float(np.mean(vals)), 5)
            if len(vals) > 1: out[row][col + "_std"] = round(float(np.std(vals, ddof=1)), 5)
    out.setdefault(row, {})["n_seeds"] = len(rs); out[row]["ifeval"] = None
IF = Path(a.out).parent / "ifeval.json"
if IF.exists():      # IFEval (greedy, prompt-level strict/loose) per run id -> row mean
    ifj = json.load(open(IF))
    for row, rs in runs.items():
        vals = [ifj[f"E5@ifeval-{rid}"]["prompt_strict"] for rid in rs if f"E5@ifeval-{rid}" in ifj]
        if vals: out[row]["ifeval"] = round(float(np.mean(vals)), 4); out[row]["ifeval_n_runs"] = len(vals)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
for row in runs: print(row, {k: v for k, v in out[row].items()})
