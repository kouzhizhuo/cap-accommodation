#!/usr/bin/env python
"""Authentic twin stimuli (lead 2026-10-04, expansion A): real dialect text vs its SAE twin, written in gen.py format (one stimuli dir per set, dialect side = <VAR>.jsonl, SAE twin = SAE.jsonl, same seed_ids).
  auth_gro        Groenwold et al. 2020 AAVE tweets (real) <-> MTurk SAE rewrites; 600 pairs whose AAVE side has >=1 RETAINED AAE morphosyntactic feature (E4 prefdata recognisers); condition AAE
  auth_endive_aae EnDive (Gupta et al. 2025) AAVE translation vs the SAE original, 500 pairs stratified by task (condition AAE)
  auth_endive_sge EnDive CollSgE translation vs the SAE original, 500 pairs stratified by task (condition SGE)
User turn = the text itself (no instruction added).  Seed 20261004.  Writes stimuli/auth_*/{AAE|SGE,SAE}.jsonl + READY + stimuli/auth_stats.json.
  python stimuli/make_authentic.py   (CPU; regex recognisers only)"""
import csv, json, random, sys, os
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP)); sys.path.insert(0, str(EXP / "E4"))
os.environ.setdefault("MV_COREF", "0")
from prefdata_bulk import FEATS
rng = random.Random(20261004); stats = {}
MORPH = ["AAE.ain_t", "AAE.completive_done", "AAE.finna_gon", "AAE.negative_concord", "AAE.yall"]
nw = lambda t: len(t.split())


def write(name, cond, pairs, meta):
    d = EXP / "stimuli" / name; d.mkdir(parents=True, exist_ok=True)
    for c, side in ((cond, 0), ("SAE", 1)):
        with open(d / f"{c}.jsonl", "w") as f:
            for i, p in enumerate(pairs): f.write(json.dumps({"seed_id": f"{name[5:8]}{i:04d}", "condition": c, "arm": "A", "text": p[side], "task_type": p[2], "context": name, "source": p[3]}, ensure_ascii=False) + "\n")
        (d / f"{c}.READY").write_text("ok")
    stats[name] = {"n_pairs": len(pairs), **meta, "mean_words": {cond: round(sum(nw(p[0]) for p in pairs) / len(pairs), 1), "SAE": round(sum(nw(p[1]) for p in pairs) / len(pairs), 1)}}


# Groenwold (line-aligned aave_samples.txt / sae_samples.txt)
G = EXP / "data_raw" / "groenwold2020" / "EMNLP-AAVE-files"
A = [l.rstrip("\n") for l in open(G / "aave_samples.txt", errors="ignore")]; S = [l.rstrip("\n") for l in open(G / "sae_samples.txt", errors="ignore")]
assert len(A) == len(S)
cand = [(a, s, "tweet", f"groenwold2020:{i}") for i, (a, s) in enumerate(zip(A, S)) if 8 <= nw(a) <= 80 and 8 <= nw(s) <= 80 and any(len(FEATS[m](a)) > 0 for m in MORPH) and a.strip() != s.strip()]
rng.shuffle(cand); write("auth_gro", "AAE", cand[:600], {"eligible_with_retained_AAE_hit": len(cand), "source": "Groenwold et al. 2020 (EMNLP-AAVE-files); AAVE side real tweets"})
# EnDive
E = EXP / "data_raw" / "endive"
TASKS = {"SST-2": ("Original Sentence", "{v} (Original Sentence)"), "COPA": ("Premise", "{v} (Premise)"), "GSM8K": ("Original", "{v} (Original)"), "SVAMP": ("Original", "{v} (Original)"),
         "WSC": ("Original Paragraph", "{v} (Original Paragraph)"), "MBPP": ("Original", "{v} (Original)"), "Logic Bench YN": ("Context", "{v} (context)"), "FOLIO": ("Premises", "{v} (Premises)")}
for name, v, cond in (("auth_endive_aae", "AAVE", "AAE"), ("auth_endive_sge", "CollSgE", "SGE")):
    per = {}
    for t, (sc, dc) in TASKS.items():
        rows = list(csv.DictReader(open(E / t / v / "filtered_bleu_scores.csv", newline="", errors="ignore")))
        ok = []
        for i, r in enumerate(rows):
            s, d = (r.get(sc) or "").strip(), (r.get(dc.format(v=v)) or "").strip()
            if 8 <= nw(s) <= 120 and 8 <= nw(d) <= 140 and s != d: ok.append((d, s, t, f"endive:{t}:{v}:{i}"))
        rng.shuffle(ok); per[t] = ok
    n = 500; take = {t: 0 for t in TASKS}; pairs = []
    while len(pairs) < n and any(take[t] < len(per[t]) for t in TASKS):          # round-robin: stratified by task
        for t in TASKS:
            if take[t] < len(per[t]) and len(pairs) < n: pairs.append(per[t][take[t]]); take[t] += 1
    write(name, cond, pairs, {"per_task": {t: take[t] for t in TASKS}, "source": f"EnDive {v} (LLM-translated dialect vs SAE original; Gupta et al. 2025)"})
json.dump(stats, open(EXP / "stimuli" / "auth_stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))
