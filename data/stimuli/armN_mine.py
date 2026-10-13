#!/usr/bin/env python
"""Arm N (authentic prompts): mine real user prompts from WildChat-1M (allenai/WildChat-1M, ODC-BY; first user turn of English, non-toxic, non-redacted conversations).
Dialect-bearing = >=1 hit of a RETAINED morphosyntactic feature of the variety (same recognisers/rule as E4 prefdata: AAE: ain't, completive done, finna/gon, negative concord, y'all; SGE: can/cannot, finish-completive)
or a lexical/rate feature (E4 LEXRATE); BRE = British spelling / lexis / have got. CTRL = prompts with ZERO hits of every feature (AAE, SGE, BRE, AUS), matched to the dialect-bearing sets on word-count decile.
Writes stimuli/armN/{AAEN,SGEN,BREN,CTRL}.jsonl (gen.py format: seed_id, condition, text, arm "N", source ids) + mining_stats.json + audit_candidates.jsonl (60 random flagged prompts for a blind human spot-check; not used here).
  MV_COREF=0 python stimuli/armN_mine.py --data DIR_WITH_PARQUET --out stimuli/armN --n-aae 400 --n-sge 200 --n-bre 300 --n-ctrl 900   (seed 20261004)"""
import argparse, glob, json, random, re, sys, os
from pathlib import Path
import pyarrow.parquet as pq
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP)); sys.path.insert(0, str(EXP / "E4"))
os.environ.setdefault("MV_COREF", "0")
from prefdata_bulk import FEATS, FNAMES            # regex recognisers only
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n-aae", type=int, default=500); ap.add_argument("--n-sge", type=int, default=500); ap.add_argument("--n-bre", type=int, default=0); ap.add_argument("--n-ctrl", type=int, default=500); ap.add_argument("--sge-min", type=int, default=2); ap.add_argument("--sge-morph", action="store_true"); a = ap.parse_args()
rng = random.Random(20261004)
MORPH = {"AAE": ["AAE.ain_t", "AAE.completive_done", "AAE.finna_gon", "AAE.negative_concord", "AAE.yall"], "SGE": ["SGE.can_cannot", "SGE.finish_completive"]}
LEX = {"AAE": ["AAE.lexicon", "AAE.habitual_be", "AAE.g_drop"], "SGE": ["SGE.particles", "SGE.lexicon"]}
BREF = ["BRE.spelling", "BRE.lexis", "BRE.have_got"]
CODE = re.compile(r"```|\bdef \w+\(|\bSELECT\b|<\w+>|\{\s*\"|;\s*$|http\S+|\\n")
NONASCII = 0.03
rows = []
for f in sorted(glob.glob(f"{a.data}/*.parquet")):
    t = pq.ParquetFile(f)
    for rg in range(t.num_row_groups):
        for r in t.read_row_group(rg, columns=["conversation_hash", "conversation", "language", "toxic", "redacted"]).to_pylist():
            if r["language"] != "English" or r["toxic"] or r["redacted"]: continue
            c = r["conversation"]
            if not c or c[0]["role"] != "user": continue
            txt = (c[0]["content"] or "").strip(); w = txt.split()
            if not (20 <= len(w) <= 300) or CODE.search(txt) or sum(ord(ch) > 127 for ch in txt) / max(1, len(txt)) > NONASCII: continue
            hits = {n: len(FEATS[n](txt)) for n in FEATS if n.split(".")[0] in ("AAE", "SGE", "BRE", "AUS")}
            rows.append({"hash": r["conversation_hash"], "text": txt, "n": len(w), "hits": hits})
print("eligible first-turn English prompts:", len(rows), flush=True)


def has(r, names): return any(r["hits"].get(n, 0) > 0 for n in names)


nh = lambda r, names: sum(r["hits"].get(n, 0) for n in names)
pool = {"AAEN": [r for r in rows if has(r, MORPH["AAE"]) and not has(r, BREF)], "SGEN": [r for r in rows if nh(r, LEX["SGE"] + (MORPH["SGE"] if a.sge_morph else [])) >= a.sge_min and not has(r, MORPH["AAE"] + LEX["AAE"])],
        "BREN": [r for r in rows if has(r, BREF) and not has(r, MORPH["AAE"] + MORPH["SGE"] + LEX["AAE"] + LEX["SGE"])],
        "CTRL": [r for r in rows if not any(v > 0 for v in r["hits"].values())]}
want = {"AAEN": a.n_aae, "SGEN": a.n_sge, "BREN": a.n_bre}
sel = {}
for k, n in want.items():
    p = pool[k]; rng.shuffle(p); sel[k] = p[:n]
# controls: match the pooled word-count distribution of the dialect-bearing selection (deciles)
allsel = [r for k in sel for r in sel[k]]; lens = sorted(r["n"] for r in allsel); cuts = [lens[int(len(lens) * q / 10)] for q in range(1, 10)]
bin_of = lambda n: sum(n > c for c in cuts)
byb = {}
for r in pool["CTRL"]: byb.setdefault(bin_of(r["n"]), []).append(r)
ctrl = []
for b in range(10):
    k = int(round(a.n_ctrl * sum(1 for r in allsel if bin_of(r["n"]) == b) / len(allsel))); pb = byb.get(b, []); rng.shuffle(pb); ctrl += pb[:k]
sel["CTRL"] = ctrl
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
prev = {"AAE_bearing(>=1 retained AAE morph hit)": sum(has(r, MORPH["AAE"]) for r in rows), "SGE_bearing(>=2 particle/lexis hits)": sum(nh(r, LEX["SGE"]) >= 2 for r in rows), "SGE_bearing(>=1 particle/lexis hit)": sum(nh(r, LEX["SGE"]) >= 1 for r in rows), "SGE_any_hit": sum(has(r, MORPH["SGE"] + LEX["SGE"]) for r in rows), "BRE_any": sum(has(r, BREF) for r in rows)}
stats = {"eligible": len(rows), "prevalence_n": prev, "prevalence_share": {k: round(v / len(rows), 5) for k, v in prev.items()}, "pool_sizes": {k: len(v) for k, v in pool.items()}, "selected": {k: len(v) for k, v in sel.items()}, "word_decile_cuts": cuts, "seed": 20261004,
         "mean_words": {k: round(sum(r["n"] for r in v) / max(1, len(v)), 1) for k, v in sel.items()}, "source": "allenai/WildChat-1M first user turns (English, non-toxic, non-redacted, 20-300 words, no code/URL/non-ASCII>3%)"}
for k, v in sel.items():
    with open(out / f"{k}.jsonl", "w") as f:
        for i, r in enumerate(v): f.write(json.dumps({"seed_id": f"n{k[:3]}{i:04d}", "condition": k, "arm": "N", "text": r["text"], "task_type": "authentic", "context": "wildchat", "source_hash": r["hash"]}) + "\n")
    (out / f"{k}.READY").write_text("ok")
cand = [r for k in ("AAEN", "SGEN", "BREN") for r in rng.sample(sel[k], min(20, len(sel[k])))]
with open(out / "audit_candidates.jsonl", "w") as f:
    for r in cand: f.write(json.dumps({"text": r["text"][:600], "flagged": {n: c for n, c in r["hits"].items() if c}}) + "\n")
json.dump(stats, open(out / "mining_stats.json", "w"), indent=1); print(json.dumps(stats))
