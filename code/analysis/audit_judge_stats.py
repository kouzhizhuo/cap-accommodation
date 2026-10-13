#!/usr/bin/env python
"""Judge-vs-human and judge-vs-judge agreement on the V4 audit items (coordinator 2026-10-03). Cohen's kappa and Gwet's AC1 with bootstrap 95% CIs (items resampled, B=5000),
per pre-declared rule group: A = pairwise formality (40), B = feature presence on on-request replies (30), C = feature presence on main (default) replies (14).  Keys: audit.v4judge.<group>.<pair>.{kappa,ac1,agree,*_lo,*_hi}.
  python analysis/audit_judge_stats.py --out results/audit/audit_v4_judges.json"""
import argparse, json
from pathlib import Path
import numpy as np
G = Path(__file__).resolve().parents[1] / "gold"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = np.random.default_rng(0); B = 5000
meta = json.load(open(G / "HUMAN_AUDIT_V4_meta.json"))["item_provenance"]
todo = {json.loads(l)["id"]: json.loads(l) for l in open(G / "HUMAN_AUDIT_V4_TODO.jsonl")}
raters = {"human": {json.loads(l)["id"]: json.loads(l)["label"] for l in open(G / "HUMAN_AUDIT_V4_LABELS.jsonl")}}
for tag in ("qwen3_8b", "gptoss20b"):
    p = G / f"judge_V4_{tag}.jsonl"
    if p.exists(): raters[tag] = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(p)}
GROUP = {"formality_pair": "A_formality", "feature_on_request": "B_feature_on_request", "feature_main_reply": "C_feature_main_reply"}
kind = {i: ("formality_pair" if "tercile" in p else "feature_on_request" if p.get("stage") == "dpo-request" else "feature_main_reply") for i, p in meta.items()}


def stats(x, y):
    x, y = np.array(x), np.array(y); n = len(x)
    if n == 0: return {"n": 0}
    cats = sorted(set(x) | set(y)); q = max(len(cats), 2)
    po = float((x == y).mean())
    px = np.array([(x == c).mean() for c in cats]); py = np.array([(y == c).mean() for c in cats])
    pe_k = float((px * py).sum()); k = (po - pe_k) / (1 - pe_k) if pe_k < 1 else np.nan
    pi = (px + py) / 2; pe_g = float((pi * (1 - pi)).sum() / (q - 1)); ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else np.nan
    return {"n": n, "agree": po, "kappa": float(k), "ac1": float(ac1)}


def boot(x, y):
    base = stats(x, y); x, y = np.array(x), np.array(y); n = len(x)
    ks, as_, ag = [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n); s = stats(x[i], y[i]); ks.append(s["kappa"]); as_.append(s["ac1"]); ag.append(s["agree"])
    f = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    base["kappa_ci"], base["ac1_ci"], base["agree_ci"] = f(ks), f(as_), f(ag)
    return base


out = {"note": "human = author labels (attested); judges blind, same instruction/note/options; kappa undefined (null) when a rater is constant", "groups": {}}
pairs = [("human", t) for t in raters if t != "human"] + [(t1, t2) for i, t1 in enumerate([t for t in raters if t != "human"]) for t2 in [t for t in raters if t != "human"][i + 1:]]
for kd, gname in GROUP.items():
    ids = [i for i in todo if kind[i] == kd]; node = {"n_items": len(ids)}
    for r1, r2 in pairs:
        both = [i for i in ids if raters[r1].get(i) is not None and raters[r2].get(i) is not None]
        s = boot([raters[r1][i] for i in both], [raters[r2][i] for i in both])
        node[f"{r1}_vs_{r2}"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in s.items()}
    node["label_distribution"] = {r: {c: sum(1 for i in ids if raters[r].get(i) == c) for c in sorted({raters[r].get(i) for i in ids if raters[r].get(i)})} for r in raters}
    out["groups"][gname] = node
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
for g, nd in out["groups"].items():
    for k, v in nd.items():
        if isinstance(v, dict) and "kappa" in v: print(g, k, "n", v["n"], "agree %.2f" % v["agree"], "kappa", None if v["kappa"] is None else round(v["kappa"], 2), "ac1", None if v["ac1"] is None else round(v["ac1"], 2))
