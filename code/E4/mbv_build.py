#!/usr/bin/env python
"""MB_v inputs (plan §3.9): per lineage and seed, y_std = the SFT model's SAE-condition reply (first sample, finish=stop), y_v = its transform into variety v
(density-calibrated Arm-S transforms: Multi-VALUE whitelists + lexical layer for AAE/SGE, rules for INF/BRE/AUS); x_v = the v1 prompt in v, x_SAE = the SAE prompt.
Items (4 per seed x variety): (x_v,y_v) (x_v,y_std) (x_SAE,y_v) (x_SAE,y_std).  Env p3mv, CPU.
  MV_COREF=0 python E4/mbv_build.py --gen E1/gen --lineages L2 L3 L4 L2B L1 --stimuli stimuli/v1 --out E4/mbv_items"""
import argparse, json, random, re, sys
from multiprocessing import Pool
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "stimuli")); sys.path.insert(0, str(EXP))
import mv_env  # noqa: F401
import build
from inf_rules import inf_transform
VARS = ["AAE", "SGE", "INF", "BRE", "AUS"]
_D = {}


def dial(c):
    if c not in _D:
        d = mv_env.get_dialect(build.DIALECT[c])
        d.morphosyntax_transforms = {k: min(1.0, float(v) * build.RATE_SCALE) for k, v in d.morphosyntax_transforms.items() if k in build.WHITELIST[c]}
        _D[c] = d
    return _D[c]


def to_variety(text, v, key):
    if v in ("AAE", "SGE"):
        rng = random.Random(key); lines = []
        for li, ln in enumerate(text.split("\n")):
            sents = [s for s in re.split(r"(?<=[.?!])\s+", ln) if s]
            outs = [build.transform_sentence(dial(v), v, s, f"{key}:{li}:{i}")[0] if len(s.split()) >= 3 else s for i, s in enumerate(sents)]
            outs, _ = build.lex_layer(v, outs, rng); lines.append(" ".join(outs))
        return "\n".join(lines)
    if v == "INF":
        return inf_transform(text, seed=key)[0]
    return build.uk_transform(text, v, key)[0]


def work(a):
    lin, sid, v, ystd = a
    return lin, sid, v, to_variety(ystd, v, f"mbv:{lin}:{sid}:{v}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen"); ap.add_argument("--lineages", nargs="+"); ap.add_argument("--stimuli"); ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--ystd-stage", default="sft", help="stage whose SAE-condition reply is y_std; per-lineage override with LIN:stage, e.g. Q72:instruct")
    a = ap.parse_args(); prm = {}
    for c in ["SAE"] + VARS:
        prm[c] = {json.loads(l)["seed_id"]: json.loads(l)["text"] for l in open(Path(a.stimuli) / f"{c}.jsonl")}
    jobs, ystd = [], {}
    for lin in a.lineages:
        lin, _, stg = lin.partition(":"); fn = Path(a.gen) / f"{lin}@{stg or a.ystd_stage}.jsonl"
        if not fn.exists(): print("missing", fn); continue
        for l in open(fn):
            r = json.loads(l)
            if r["condition"] == "SAE" and r["sampling_seed"] == 0 and r["sample"] == 0 and r["finish"] == "stop" and len(r["response"].split()) >= 15:
                ystd[(lin, r["seed_id"])] = r["response"]
    for (lin, sid), y in ystd.items():
        for v in VARS: jobs.append((lin, sid, v, y))
    with Pool(a.workers) as p: res = p.map(work, jobs, chunksize=16)
    yv = {(lin, sid, v): t for lin, sid, v, t in res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); n = 0
    with open(a.out + ".jsonl", "w") as f:
        for (lin, sid, v), t in yv.items():
            std = ystd[(lin, sid)]
            if t == std: continue                      # transform changed nothing -> no contrast
            for pk, x in (("v", prm[v][sid]), ("SAE", prm["SAE"][sid])):
                for rk, y in (("v", t), ("std", std)):
                    f.write(json.dumps({"item": f"{lin}|{sid}|{v}|{pk}|{rk}", "lineage": lin, "seed_id": sid, "variety": v, "prompt_kind": pk, "resp_kind": rk, "x": x, "y": y}, ensure_ascii=False) + "\n"); n += 1
    print("items", n, "lineages", a.lineages, "unchanged-skipped", sum(1 for k, t in yv.items() if t == ystd[k[:2]]))


if __name__ == "__main__":
    main()
