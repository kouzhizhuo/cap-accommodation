#!/usr/bin/env python
"""RE-PAIRED density stimuli (astra's fix; METHOD_DESIGN §6).  Within each variety the realised user densities of the v1 requests (feature hits per 100 words of the rendered request) are PERMUTED
(marginal kept, density independent of content; fixed cyclic-shift of a seeded shuffle, no request keeps its own target) and every request is re-rendered at its permuted density at SENTENCE granularity:
each sentence is either the v1 rendering or the SAE source; the subset whose realised density is closest to the target is used (the achieved density is recorded; the tracking slope uses achieved values).
  python make_repaired.py --v1 stimuli/v1 --out <dir>   -> <dir>/AAE_rp.jsonl, SGE_rp.jsonl (+ repaired_meta.json); the original-density replies are the plain AAE/SGE v1 replies."""
import argparse, itertools, json, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import re
def split_sents(text): return [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]   # = stimuli/build.py split_sents
ap = argparse.ArgumentParser(); ap.add_argument("--v1", default=str(Path(__file__).resolve().parent / "v1")); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True); meta = {}
for v in ("AAE", "SGE"):
    rows = [json.loads(l) for l in open(Path(a.v1) / f"{v}.jsonl")]; items = []
    for r in rows:
        S, O = split_sents(r["source_text"]), split_sents(r["text"])
        if len(S) != len(O) or " ".join(O) != r["text"] or " ".join(S) != r["source_text"] or len(O) > 10: continue
        nf = [sum(1 for f in r["features_applied"] if f.get("sent") == k) for k in range(len(O))]
        d0 = 100.0 * sum(nf) / max(1, len(r["text"].split()))
        items.append({"r": r, "S": S, "O": O, "nf": nf, "d0": d0})
    rng = random.Random(f"{a.seed}:{v}"); order = list(range(len(items))); rng.shuffle(order)
    # target of item order[i] = density of item order[i+1] (cyclic) -> a permutation without fixed points
    tgt = {order[i]: items[order[(i + 1) % len(order)]]["d0"] for i in range(len(order))}
    out, ach = [], []
    for i, it in enumerate(items):
        k = len(it["O"]); best = None
        for mask in itertools.product((0, 1), repeat=k):
            text = " ".join(o if m else s for s, o, m in zip(it["S"], it["O"], mask)); w = max(1, len(text.split()))
            d = 100.0 * sum(n for n, m in zip(it["nf"], mask) if m) / w; key = (abs(d - tgt[i]), rng.random())
            if best is None or key < best[0]: best = (key, mask, text, d)
        _, mask, text, d = best
        out.append({"seed_id": it["r"]["seed_id"], "condition": f"{v}_rp", "arm": "RP", "text": text, "source_text": it["r"]["source_text"], "task_type": it["r"]["task_type"], "context": it["r"]["context"],
                    "d_orig": it["d0"], "d_target": tgt[i], "d_achieved": d, "mask": list(mask), "n_sent": k})
        ach.append((it["d0"], tgt[i], d))
    with open(Path(a.out) / f"{v}_rp.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    import statistics as st, math
    dd = [x[2] - x[0] for x in ach]; dt = [x[1] - x[0] for x in ach]
    import numpy as np; cor = float(np.corrcoef(dt, dd)[0, 1])
    meta[v] = {"n_in": len(rows), "n_ok": len(items), "mean_d_orig": st.mean(x[0] for x in ach), "mean_d_achieved": st.mean(x[2] for x in ach), "sd_delta_achieved": st.pstdev(dd), "corr_target_vs_achieved_delta": cor,
               "share_changed": sum(1 for x in dd if abs(x) > 1e-9) / len(dd)}
json.dump(meta, open(Path(a.out) / "repaired_meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
