#!/usr/bin/env python
"""B7 self-rewrite baseline (D24; 2 passes) for Qwen3-8B INSTRUCT on F300.  Pass 1: for every INSTRUCT draft (cond x request x sampling seed x sample) ask the SAME model to rewrite its own draft given the user message and the B3 instruction;
pass 2: rewrite the pass-1 output again.  `stim` builds the pass-<k> stimuli (one row per draft, unique seed_id); `collect` maps pass-2 outputs back to the F300 gen format (model Q3@INSTRUCT-B7).
  python b7_make.py stim 1 --draft <email> --f300 F300dir --out stim_p1 ;  stim 2 --draft stim_p1_out.jsonl ... ; collect --p2 out2.jsonl --draft ... --out <email>"""
import argparse, json
from pathlib import Path
B3 = "Reply in the same variety and register the user writes in, matching how much they use it; never exaggerate it."
ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("k", nargs="?", type=int); ap.add_argument("--draft"); ap.add_argument("--f300"); ap.add_argument("--out"); ap.add_argument("--p2"); a = ap.parse_args()
CONDS = ("SAE", "AAE", "SGE", "NGP", "APP")
def rows(p): return [json.loads(l) for l in open(p)]
if a.mode == "stim":
    user = {}
    for c in CONDS:
        for r in rows(Path(a.f300) / f"{c}.jsonl"): user[(c, r["seed_id"])] = r
    out = {c: [] for c in CONDS}
    for g in rows(a.draft):
        if a.k == 1:
            c, sid, ss, k = g["condition"], g["seed_id"], g["sampling_seed"], g["sample"]; key = f"{sid}__{ss}_{k}"; draft = g["response"]
        else:   # pass-2 input = pass-1 output rows (seed_id already the unique key; condition kept)
            c, key = g["condition"], g["seed_id"]; sid = key.split("__")[0]; draft = g["response"]
        u = user[(c, sid)]
        text = f"{u['text']}\n\nHere is a draft reply to the message above:\n\n{draft}\n\nRewrite the draft. {B3} Output only the rewritten reply, nothing else."
        out[c].append({"seed_id": key, "condition": c, "arm": "B7", "text": text, "task_type": u.get("task_type"), "context": u.get("context")})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c, rs in out.items():
        with open(Path(a.out) / f"{c}.jsonl", "w") as f:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({c: len(r) for c, r in out.items()})
else:
    n = 0
    with open(a.out, "w") as f:
        for g in rows(a.p2):
            sid, rest = g["seed_id"].split("__"); ss, k = rest.split("_")
            g2 = dict(g); g2.update({"seed_id": sid, "sampling_seed": int(ss), "sample": int(k), "model": "Q3@Q3@INSTRUCT-B7", "stage": "INSTRUCT-B7"}); f.write(json.dumps(g2, ensure_ascii=False) + "\n"); n += 1
    print("collected", n)
