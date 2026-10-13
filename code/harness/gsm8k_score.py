#!/usr/bin/env python
"""GSM8K scorer: accuracy = share of replies whose final answer (after 'The answer is', else the last number) equals the gold number.  python gsm8k_score.py <gen.jsonl> [more ...] -> prints/dumps {name: {acc, n, truncated_share}}"""
import json, re, sys
from pathlib import Path
gold = {json.loads(l)["seed_id"]: json.loads(l)["gold"] for l in open(Path(__file__).resolve().parents[1] / "stimuli/gsm8k/GSM8K.jsonl")}
num = lambda s: re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
def norm(x):
    x = x.replace(",", "").rstrip("."); 
    try: return float(x)
    except ValueError: return None
out = {}
for p in sys.argv[1:]:
    ok = n = tr = 0
    for l in open(p):
        r = json.loads(l); n += 1; tr += r.get("finish") == "length"; resp = r["response"]
        m = re.search(r"answer is[:\s]*\$?\s*(-?\d[\d,]*\.?\d*)", resp, re.I); cand = m.group(1) if m else (num(resp)[-1] if num(resp) else None)
        a, g = norm(cand) if cand else None, norm(gold[r["seed_id"]])
        ok += a is not None and g is not None and abs(a - g) < 1e-6
    out[Path(p).stem] = {"acc": ok / max(1, n), "n": n, "truncated_share": tr / max(1, n)}
print(json.dumps(out, indent=1))
