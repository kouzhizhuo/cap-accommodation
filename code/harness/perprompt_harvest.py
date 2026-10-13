#!/usr/bin/env python
"""Per-PROMPT helpfulness files for the M4 non-inferiority CI (lead's harvest contract).  Output jsonl rows {run, id, correct, [strict, loose]}:
  IFEval: correct = strict prompt-level follow-all; also `loose`.   GSM8K: correct = final number equals gold.   run = model name as in the scores parquet (E5@<run> / Q3@<run>[-B3|...]).
  python harness/perprompt_harvest.py --root _incoming/<host> --out results/METHOD/perprompt"""
import argparse, glob, json, re, sys
from pathlib import Path
here = Path(__file__).resolve().parent; sys.path.insert(0, str(here / "ifeval" / "pylibs"))
from lm_eval.tasks.ifeval import utils as U
from langdetect import DetectorFactory; DetectorFactory.seed = 0   # langdetect is random unless seeded; fixed 2026-10-06 so per-prompt IFEval (M4) is reproducible
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
data = {f"if{r['key']}": r for r in (json.loads(l) for l in open(here / "ifeval/ifeval_input_data.jsonl") if l.strip())}
gold = {json.loads(l)["seed_id"]: json.loads(l)["gold"] for l in open(here.parent / "stimuli/gsm8k/GSM8K.jsonl")}
num = lambda s: re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
def norm(x):
    try: return float(x.replace(",", "").rstrip("."))
    except Exception: return None
def runname(stem, kind):   # E5@ifeval-CAP60k_s0 -> E5@CAP60k_s0 ; Q3@gsm-INSTRUCT-B3 -> Q3@INSTRUCT-B3 ; E5@ifeval-B3-d0_s0 -> E5@d0_s0-B3
    m = re.match(r"^(E5|Q3)@(?:ifeval|gsm)-(.*)$", stem); tag, rid = m.group(1), m.group(2)
    m2 = re.match(r"^(B3|B4|B34)-(.*)$", rid)
    return f"{tag}@{m2.group(2)}-{m2.group(1)}" if m2 else f"{tag}@{rid}"
for kind, pat in (("ifeval", "*/ifeval/*@ifeval-*.jsonl"), ("gsm8k", "*/gsm8k/*@gsm-*.jsonl")):
    rows = []
    for fn in sorted(glob.glob(str(Path(a.root) / pat)) + glob.glob(str(Path(a.root) / pat.replace("*/", "*/*/", 1)))):
        run = runname(Path(fn).stem, kind)
        for l in open(fn):
            g = json.loads(l); sid = g["seed_id"]
            if kind == "ifeval":
                d = data[sid]; inp = U.InputExample(key=d["key"], instruction_id_list=d["instruction_id_list"], prompt=d["prompt"], kwargs=d["kwargs"])
                s, lo = U.test_instruction_following_strict(inp, g["response"]), U.test_instruction_following_loose(inp, g["response"])
                rows.append({"run": run, "id": sid, "correct": bool(s.follow_all_instructions), "strict": bool(s.follow_all_instructions), "loose": bool(lo.follow_all_instructions)})
            else:
                resp = g["response"]; m = re.search(r"answer is[:\s]*\$?\s*(-?\d[\d,]*\.?\d*)", resp, re.I); cand = m.group(1) if m else (num(resp)[-1] if num(resp) else None)
                x, y = (norm(cand) if cand else None), norm(gold[sid]); rows.append({"run": run, "id": sid, "correct": bool(x is not None and y is not None and abs(x - y) < 1e-6)})
    with open(Path(a.out) / f"{kind}_perprompt.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    runs = sorted({r["run"] for r in rows}); print(kind, len(rows), "rows", len(runs), "runs")
