#!/usr/bin/env python
"""METHOD v3/v4 evaluation stimuli on the F300 fresh test requests (env p3mv, CPU).  Inputs: <root>/F300_seeds.jsonl (pool requests in no training pair, no V300, not in v1), records_U1.jsonl (CAP prompt densities).
  F300 (primary fresh test): rendered with the v1 renderer (stimuli/build.py: SAE AAE SGE NGP APP) -> <out>/F300/  (run build.py separately; this script only checks it exists)
  D (density intervention): each F300 request rendered in AAE and SGE at two F_train TARGET densities = 25th / 75th percentile of the CAP prompt densities (per variety); sentence-level subset closest to the target
     (overshoot cap 1.25x); matched decoding seeds; requests whose ACHIEVED contrast (hi - lo) < 0.5 hits/100 words are dropped -> <out>/D/{AAE_lo,AAE_hi,SGE_lo,SGE_hi}.jsonl + D_meta.json
  O (overrides): 100 dialect prompts (F300 v1-render AAE/SGE, alternating) x {STD "Please answer in Standard English." / FORMAL "Please write formally." / NOMIMIC "Please don't imitate my wording."};
     100 SAE prompts x {AAE "Please reply in African American English." / SGE "Please reply in Singlish."} -> <out>/O/O_<name>.jsonl
  python make_eval_sets.py --root pairs_v3 --f300 <dir with AAE.jsonl SGE.jsonl SAE.jsonl> --out E5/evalsets"""
import argparse, json, random, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import pairs_build_v2 as B
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--f300", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pct", type=float, nargs=2, default=[0.25, 0.75]); ap.add_argument("--dname", default="D"); a = ap.parse_args()
R, F, O = Path(a.root), Path(a.f300), Path(a.out); (O / a.dname).mkdir(parents=True, exist_ok=True); (O / "O").mkdir(parents=True, exist_ok=True)
seeds = [json.loads(l) for l in open(R / "F300_seeds.jsonl")]; U1 = [json.loads(l) for l in open(R / "records_U1.jsonl")]
q = {}
for v in ("AAE", "SGE"):
    d = sorted(r["d_prompt"] for r in U1 if r["variety"] == v); q[v] = (d[int(a.pct[0] * len(d))], d[int(a.pct[1] * len(d))])
meta = {"targets_per100w": q, "n_in": len(seeds)}
for v in ("AAE", "SGE"):
    lo, hi, dropped = [], [], 0
    for sd in seeds:
        text = sd["text"]; lines = B.sent_pairs(text, v, f"D:{sd['seed_id']}"); words = max(1, len(text.split()))
        nf = {(li, si): B.n_train(n, v) - B.n_train(o, v) for li, line in enumerate(lines) for si, (o, n) in enumerate(line)}
        outs = []
        for tag, tgt in (("lo", q[v][0]), ("hi", q[v][1])):
            pm = B.pick_mirror(nf, tgt * words / 100.0, random.Random(f"D{tag}:{sd['seed_id']}"))
            if pm is None: outs.append(None); continue
            outs.append((B.join(lines, lambda li, si: (li, si) in pm[0]), 100.0 * pm[1] / words))
        if None in outs or outs[1][1] - outs[0][1] < 0.5: dropped += 1; continue
        base = {"seed_id": sd["seed_id"], "task_type": sd.get("task_type"), "context": sd.get("context"), "source_text": text}
        lo.append({**base, "condition": f"{v}_lo", "text": outs[0][0], "d_achieved": outs[0][1]}); hi.append({**base, "condition": f"{v}_hi", "text": outs[1][0], "d_achieved": outs[1][1]})
    for nm, rows in ((f"{v}_lo", lo), (f"{v}_hi", hi)):
        with open(O / a.dname / f"{nm}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta[v] = {"kept": len(lo), "dropped_contrast_lt_0.5": dropped, "mean_d_lo": st.mean(r["d_achieved"] for r in lo), "mean_d_hi": st.mean(r["d_achieved"] for r in hi)}
json.dump(meta, open(O / f"{a.dname}_meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
# O overrides (needs the v1-rendered F300)
if a.dname == "D" and (F / "AAE.jsonl").exists():
    aae = [json.loads(l) for l in open(F / "AAE.jsonl")]; sge = [json.loads(l) for l in open(F / "SGE.jsonl")]; sae = [json.loads(l) for l in open(F / "SAE.jsonl")]
    dia = [x for p in zip(aae, sge) for x in p][:100]; saes = sae[:100]
    spec = {"STD": (dia, "Please answer in Standard English."), "FORMAL": (dia, "Please write formally."), "NOMIMIC": (dia, "Please don't imitate my wording."),
            "AAE": (saes, "Please reply in African American English."), "SGE": (saes, "Please reply in Singlish.")}
    for nm, (rows, ins) in spec.items():
        with open(O / "O" / f"O_{nm}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps({"seed_id": r["seed_id"], "condition": f"O_{nm}", "text": r["text"] + " " + ins, "task_type": r.get("task_type"), "context": r.get("context"), "base_condition": r["condition"]}, ensure_ascii=False) + "\n")
    print("O sets written")
else: print("F300 v1 render missing -> O sets not written")
