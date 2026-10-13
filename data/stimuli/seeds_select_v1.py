#!/usr/bin/env python
"""Stimuli v1 step 2: filter candidates and select 300 seeds (25 per task_type x context). Env p3mv (CPU).
Checks: AAE/SGE opportunities (seeds_check.passes: >=2 aux/copula, >=1 3sg VBZ, >=1 negation, >=1 question), length 25-60 words, no contractions,
no chatbot boilerplate, near-duplicate removal, and BRE+AUS opportunity (>=1 transformable US item by build.uk_transform for BOTH conditions).
  MV_COREF=0 python seeds_select_v1.py --cands seeds_v1_candidates.jsonl --out seeds_v1.jsonl"""
import argparse, json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mv_env  # noqa: F401
from seeds_check import opportunities, passes
from build import uk_transform

ap = argparse.ArgumentParser(); ap.add_argument("--cands"); ap.add_argument("--out"); ap.add_argument("--per-cell", type=int, default=25); ap.add_argument("--no-ukcheck", action="store_true"); ap.add_argument("--exclude", default=None)
ap.add_argument("--id-prefix", default="v1s"); a = ap.parse_args()
rows = [json.loads(l) for l in open(a.cands)]
seen, keep, why = set(), [], {}
if a.exclude:
    seen |= {re.sub(r"\W+", "", json.loads(l)["text"].lower())[:45] for l in open(a.exclude)}
def rej(k): why[k] = why.get(k, 0) + 1
for i, r in enumerate(rows):
    t = re.sub(r"\s+", " ", r["text"]).strip(); w = len(t.split())
    if not (25 <= w <= 60): rej("length"); continue
    if re.search(r"\b\w+n't\b|\b(?:I'm|it's|I've|we're|they're|he's|she's|that's|can't)\b", t) or "'" in t.replace("'s ", " ").replace("s' ", " ") and re.search(r"\b(I|we|they|you)'(m|re|ve|ll|d)\b", t): rej("contraction"); continue
    if re.search(r"(?i)as an ai|i am an ai|sure[,!]|here is|here's|\n", t) or not t.endswith("?"): rej("format"); continue
    key = re.sub(r"\W+", "", t.lower())[:45]
    if key in seen: rej("duplicate"); continue
    if not passes(opportunities(t)): rej("aae_sge_opportunity"); continue
    if not a.no_ukcheck and (uk_transform(t, "BRE", "x")[0] == t or uk_transform(t, "AUS", "x")[0] == t): rej("bre_aus_opportunity"); continue
    seen.add(key); keep.append(dict(r, text=t))
print("kept", len(keep), "of", len(rows), "rejections:", why)
rng = random.Random(0); out = []
for t in sorted({r["task_type"] for r in keep}):
    for c in ("casual", "formal"):
        cell = [r for r in keep if r["task_type"] == t and r["context"] == c]; rng.shuffle(cell)
        print(t, c, "available", len(cell)); out += cell[:a.per_cell]
out.sort(key=lambda r: (r["task_type"], r["context"]))
with open(a.out, "w") as f:
    for i, r in enumerate(out):
        f.write(json.dumps({"seed_id": f"{a.id_prefix}{i:05d}", "task_type": r["task_type"], "context": r["context"], "text": r["text"],
                            "source": "Qwen2.5-7B-Instruct seed writer (make_seeds_v1.py), filtered; author review pending"}, ensure_ascii=False) + "\n")
print("selected", len(out))
