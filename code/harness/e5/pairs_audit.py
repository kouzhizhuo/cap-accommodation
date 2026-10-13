#!/usr/bin/env python
"""CPU AUDIT of the CAP pair sets BEFORE any training (METHOD_DESIGN v3 §audit).  Env p3mv.  Reads <root>/records_{U1,X,A}.jsonl (+V300/F300) written by pairs_build_v2.py.
Reports per pair type {U, O, A, Uu (unmatched, E5 recipe)}, chosen vs rejected:  held-out-feature hits /100 words (must NOT differ), words, sentences, punctuation marks /100 words, newline/bullet formatting,
MIRROR target-vs-achieved error, source disjointness (train / V300 / F300 / v1, by seed id and normalised text).  Also writes <root>/audit_lp_items.jsonl for the SFT log-prob/token of MIRROR vs FULL vs ORIG (GPU step: E4/mbv_score_vllm.py on the 1B SFT; merged with `--merge-lp`).
  python pairs_audit.py --root pairs_v3 --out results/E5/pairs_audit.json [--merge-lp lp.jsonl]"""
import argparse, json, re, sys, statistics as st
from multiprocessing import Pool
from pathlib import Path
EXP = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(EXP)); sys.path.insert(0, str(EXP / "stimuli"))
from detectors import split_items
HELD = {"AAE": ["AAE.morph.held", "AAE.lex.held"], "SGE": ["SGE.part.held", "SGE.lex.held"]}
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--merge-lp", default=None)
ap.add_argument("--workers", type=int, default=32); ap.add_argument("--v1-seeds", default=str(EXP / "stimuli/seeds_v1.jsonl")); a = ap.parse_args(); R = Path(a.root)


def feats(t, v, groups): return sum(len(split_items.spans_of(g, t)) for g in groups[v])


def stats(args):
    t, v = args; w = max(1, len(t.split()))
    return {"held": 100.0 * feats(t, v, HELD) / w, "train": 100.0 * feats(t, v, TRAIN) / w, "words": w, "sents": len([s for s in re.split(r"(?<=[.?!])\s+", t.strip()) if s]),
            "punct": 100.0 * len(re.findall(r"[^\w\s]", t)) / w, "newlines": t.count("\n"), "bullets": len(re.findall(r"(?m)^\s*(?:[-*•]|\d+\\.)\s", t))}


def load(n): return [json.loads(l) for l in open(R / f"records_{n}.jsonl")]


def main():
    U1, X, A, V, F = (load(n) for n in ("U1", "X", "A", "V300", "F300"))
    types = {"U": (U1, "matched", "orig"), "O": (X, "matched", "full"), "A": (A, "orig", "full"), "Uu": (U1, "full", "orig")}
    out = {"types": {}}; jobs = []; idx = []
    for tp, (rows, ck, rk) in types.items():
        sel = rows[:: max(1, len(rows) // a.n)][: a.n]
        for r in sel: jobs += [(r[ck], r["variety"]), (r[rk], r["variety"])]; idx.append(tp)
    with Pool(a.workers) as p: res = p.map(stats, jobs, chunksize=64)
    agg = {}
    for i, tp in enumerate(idx):
        c, j = res[2 * i], res[2 * i + 1]
        for k in c: agg.setdefault(tp, {}).setdefault(k, {"chosen": [], "rejected": []}); agg[tp][k]["chosen"].append(c[k]); agg[tp][k]["rejected"].append(j[k])
    for tp, d in agg.items():
        out["types"][tp] = {k: {"chosen_mean": st.mean(v["chosen"]), "rejected_mean": st.mean(v["rejected"]), "diff": st.mean(c - r for c, r in zip(v["chosen"], v["rejected"])),
                                "diff_se": st.pstdev([c - r for c, r in zip(v["chosen"], v["rejected"])]) / max(1, len(v["chosen"])) ** 0.5} for k, v in d.items()}
        out["types"][tp]["n_pairs"] = len(d["held"]["chosen"])
    # MIRROR target vs achieved
    err = [(r["n_matched"] - r["target_hits"], abs(r["n_matched"] - r["target_hits"]) / r["target_hits"]) for r in U1 + X]
    out["mirror_target_vs_achieved"] = {"n": len(err), "mean_signed_hits": st.mean(e[0] for e in err), "mean_abs_rel": st.mean(e[1] for e in err), "share_within_25pct": sum(e[1] <= 0.25 for e in err) / len(err), "share_overshoot": sum(e[0] > 1e-9 for e in err) / len(err)}
    # disjointness
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    tr = {r["seed_id"]: norm(r["prompt_sae"]) for r in U1 + X + A}; v1 = [json.loads(l) for l in open(a.v1_seeds)]
    sets = {"train": (set(tr), set(tr.values())), "V300": ({r["seed_id"] for r in V}, {norm(r["prompt_sae"]) for r in V}), "F300": ({r["seed_id"] for r in F}, {norm(r["prompt_sae"]) for r in F}),
            "v1": ({r["seed_id"] for r in v1}, {norm(r["text"]) for r in v1})}
    out["disjointness"] = {f"{x}_vs_{y}": {"seed_id_overlap": len(sets[x][0] & sets[y][0]), "text_overlap": len(sets[x][1] & sets[y][1])} for i, x in enumerate(sets) for y in list(sets)[i + 1:]}
    out["train_seed_reuse"] = {"n_rows": len(U1) + len(X) + len(A), "n_unique_seeds": len(tr)}
    # SFT log-prob items (GPU step)
    import random
    rng = random.Random(0); items = []
    for tp, rows in (("U", U1), ("O", X)):
        for r in rng.sample(rows, min(1500, len(rows))):
            for nm in ("matched", "full", "orig"): items.append({"item": f"{tp}|{r['seed_id']}|{nm}", "lineage": "E5", "x": r["prompt_v"], "y": r[nm]})
    with open(R / "audit_lp_items.jsonl", "w") as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    if a.merge_lp:
        lp = {json.loads(l)["item"]: json.loads(l) for l in open(a.merge_lp)}; agg2 = {}
        for k, v in lp.items():
            tp, sid, nm = k.split("|"); agg2.setdefault(tp, {}).setdefault(nm, {})[sid] = v["lp"] / v["n"]
        out["sft_logprob_per_token"] = {tp: {nm: st.mean(d.values()) for nm, d in dd.items()} | {"mirror_minus_full": st.mean(dd["matched"][s] - dd["full"][s] for s in dd["matched"] if s in dd["full"])} for tp, dd in agg2.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
