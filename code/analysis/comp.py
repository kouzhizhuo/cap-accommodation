#!/usr/bin/env python
"""COMP / CAR summary (H3 'can, but won't'): realisation of the REQUESTED variety's rate features in on-request replies vs authentic-corpus rates.
COMP_v = min(1, rate_request / rate_authentic), CAR_v = rate_request / rate_authentic (unclipped; > 1.3 = caricature), per 100 words, novel spans;
also reports the default-reply rate (same model, no instruction, variety prompt) and the gap COMP - default-rate ratio. Authentic: AAE = Groenwold AAVE tweets,
SGE = NUS SMS (Singapore senders). Bootstrap over seed prompts (B=2000). Env p3mv.
  MV_COREF=0 python analysis/comp.py --request E1/request_scores.parquet --scores E1/scores.parquet --raw data_raw_small --out results/E1/comp.json"""
import argparse, json, re, sys, warnings, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detectors  # noqa: F401
import detectors.aae as A, detectors.sge as S
ap = argparse.ArgumentParser(); [ap.add_argument(f"--{k}", required=True) for k in ("request", "scores", "raw", "out")]; a = ap.parse_args()
RATE = {"AAE": ["ain_t", "completive_done", "finna_gon", "negative_concord", "yall", "lexicon", "g_drop", "habitual_be"], "SGE": ["particles", "lexicon"]}
REC = {"AAE": {"ain_t": A.ain_t, "completive_done": A.completive_done, "finna_gon": A.finna_gon, "negative_concord": A.negative_concord, "yall": A.yall,
               "lexicon": A.aae_lexicon, "g_drop": A.g_drop, "habitual_be": A.habitual_be}, "SGE": {"particles": S.particles, "lexicon": S.sge_lexicon}}
raw = Path(a.raw)
auth = {"AAE": [l.strip() for l in open(raw / "aave_samples.txt", errors="ignore") if l.strip()], "SGE": []}
for m in ET.parse(raw / "smsCorpus_en_2015.03.09_all.xml").getroot().iter("message"):
    if (m.findtext(".//country") or "").strip() in ("Singapore", "SG"): auth["SGE"].append(m.findtext("text") or "")
auth_rate = {v: {f: 100.0 * sum(len(r(t)) for t in ts) / max(1, sum(len(t.split()) for t in ts)) for f, r in REC[v].items()} for v, ts in auth.items()}
req, sc = pd.read_parquet(a.request), pd.read_parquet(a.scores); req = req[req.kind == "response"]; sc = sc[sc.kind == "response"]
rng = np.random.default_rng(0); out = {"authentic_rate_per100w": auth_rate, "n_authentic_texts": {v: len(t) for v, t in auth.items()}}


def per_seed(d, v):   # per-seed mean of the summed-feature rate (per 100 words, novel spans)
    cols = {f: f"{v}.{f}.real_n" for f in RATE[v]}
    return {f: (100.0 * d[c] / d.n_words.clip(lower=1)).groupby(d.seed_id).mean() for f, c in cols.items()}


for m in sorted(req.model.unique()):
    lin, st = m.split("@"); base = st.replace("-request", ""); node = out.setdefault(lin.lower() if lin not in ("L1", "L2", "L3", "L4") else lin.lower(), {}).setdefault(base, {})
    for v in ("AAE", "SGE"):
        d = req[(req.model == m) & (req.condition == v)]; dd = sc[(sc.model == f"{lin}@{base}") & (sc.condition == v)]
        if d.empty: continue
        comp, car, dflt = [], [], []
        ps, pd_ = per_seed(d, v), per_seed(dd, v) if not dd.empty else None
        feats = [f for f in RATE[v] if auth_rate[v][f] > 0.01]        # only features with a measurable authentic rate
        rr = {f: float(ps[f].mean()) for f in feats}
        ratios = {f: rr[f] / auth_rate[v][f] for f in feats}
        dr = {f: float(pd_[f].mean()) / auth_rate[v][f] for f in feats} if pd_ is not None else {}
        node[v.lower()] = {"comp": float(np.mean([min(1.0, r) for r in ratios.values()])), "car": float(np.mean(list(ratios.values()))),
                           "default_ratio": float(np.mean(list(dr.values()))) if dr else None, "n_features": len(feats),
                           "share_features_ge_50pct_of_authentic": float(np.mean([r >= 0.5 for r in ratios.values()])),
                           "by_feature_ratio": {f: round(r, 3) for f, r in ratios.items()}}
json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
for l, sts in out.items():
    if isinstance(sts, dict) and l not in ("authentic_rate_per100w", "n_authentic_texts"):
        for st, vs in sts.items(): print(l, st, {v: (round(x["comp"], 2), round(x["car"], 2), x["default_ratio"] and round(x["default_ratio"], 2)) for v, x in vs.items()})
