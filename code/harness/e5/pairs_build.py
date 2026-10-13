#!/usr/bin/env python
"""E5 addition-design pair files (notes/E5_MANIFEST.md). Env p3mv (CPU). F_train features ONLY (detectors/feature_split_v1.1.json):
  AAE morph F_train {finna_gon, completive_done, ain_t} -> Multi-VALUE rules {finna_future, future_sub_gon, completive_done, aint_have, aint_before_main};
  AAE lex F_train {talmbout, bruh, tryna, yo}; SGE lex/particles F_train {sia, ah, sian, walao}  (siao has no natural slot and is not used).
Per seed i (55% AAE / 45% SGE by position: AAE loses ~35% of pairs to 'no F_train feature', so it needs more seeds; rows are re-interleaved A,S after the drops, so every prefix is balanced):  prompt_v = F_train transform of the SAE seed;
  TREATMENT  chosen = SFT reply r0 transformed into v, rejected = r0 (untransformed)
  CONTROL    chosen = r0, rejected = r1 (second SFT sample), same prompt_v       ANTI  = treatment with chosen/rejected swapped.
Nested doses: rows [:1000] ⊂ [:5000] ⊂ [:20000]. Pairs whose replies were truncated (finish=length) or whose chosen reply received no feature are dropped.
  MV_COREF=0 python pairs_build.py --seeds pool.jsonl --replies <email> --base subset120k/data/train-00000-of-00001.parquet --out-root pairs/"""
import argparse, json, random, re, sys
from multiprocessing import Pool
from pathlib import Path
EXP = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(EXP / "stimuli")); sys.path.insert(0, str(EXP))
import mv_env  # noqa: F401
import build

AAE_RULES = {"finna_future", "future_sub_gon", "completive_done", "aint_have", "aint_before_main"}
_D = {}


def dialect():
    if "d" not in _D:
        d = mv_env.get_dialect(build.DIALECT["AAE"])
        d.morphosyntax_transforms = {k: min(1.0, float(v) * build.RATE_SCALE) for k, v in d.morphosyntax_transforms.items() if k in AAE_RULES}
        _D["d"] = d
    return _D["d"]


def lex_aae(sents, rng):
    feats, out = [], []
    for i, s in enumerate(sents):
        if re.search(r"\btrying to\b", s): s = re.sub(r"\btrying to\b", "tryna", s, count=1); feats.append("tryna")
        if re.search(r"\btalking about\b", s): s = re.sub(r"\btalking about\b", "talmbout", s, count=1); feats.append("talmbout")
        if i == 0 and rng.random() < 0.12: s = "Yo, " + (s if s.startswith("I ") or s.startswith("I'") else s[0].lower() + s[1:]); feats.append("yo")
        elif rng.random() < 0.06 and s.endswith((".", "?", "!")) and len(s.split()) > 4: s = s[:-1] + ", bruh" + s[-1]; feats.append("bruh")
        out.append(s)
    return out, feats


def lex_sge(sents, rng):
    feats, out = [], []
    for i, s in enumerate(sents):
        if s.endswith("?") and rng.random() < 0.30 and len(s.split()) > 3: s = s[:-1] + " ah?"; feats.append("ah")
        elif s.endswith(".") and rng.random() < 0.30 and len(s.split()) > 4: s = s[:-1] + " sia."; feats.append("sia")
        if i == 0 and rng.random() < 0.10: s = "Walao, " + (s if s.startswith("I ") or s.startswith("I'") else s[0].lower() + s[1:]); feats.append("walao")
        elif i == 0 and rng.random() < 0.06: s = "Sian, " + (s if s.startswith("I ") or s.startswith("I'") else s[0].lower() + s[1:]); feats.append("sian")
        out.append(s)
    return out, feats


def split_sents(text):   # keep line structure: transform sentence by sentence inside each line
    return [[s for s in re.split(r"(?<=[.?!])\s+", ln) if s] if ln.strip() else [] for ln in text.split("\n")]


def transform(text, v, key):
    rng = random.Random(key); feats = []; lines = split_sents(text); out_lines = []
    for li, sents in enumerate(lines):
        if v == "AAE":
            new = []
            for si, s in enumerate(sents):
                if len(s.split()) < 3: new.append(s); continue
                o, r, att, _ = build.transform_sentence(dialect(), "AAE", s, f"{key}:{li}:{si}")
                new.append(o); feats += [x["feature"] for x in r]
            new, lf = lex_aae(new, rng)
        else:
            new, lf = lex_sge(sents, rng)
        feats += lf; out_lines.append(" ".join(new))
    return "\n".join(out_lines), feats


def work(args):
    i, sid, v, prompt, r0, r1 = args
    pv, pf = transform(prompt, v, f"e5p:{sid}")
    c, cf = transform(r0, v, f"e5r:{sid}")
    return i, sid, v, pv, c, len(cf), sorted(set(cf)), len(pf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds"); ap.add_argument("--replies"); ap.add_argument("--base"); ap.add_argument("--out-root")
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--doses", type=int, nargs="+", default=[1000, 5000, 20000]); a = ap.parse_args()
    seeds = [json.loads(l) for l in open(a.seeds)]
    rep = {}
    for l in open(a.replies):
        r = json.loads(l); rep.setdefault(r["seed_id"], {})[r["sampling_seed"]] = r
    jobs, dropped = [], {"truncated_or_short": 0, "missing": 0}
    for i, sd in enumerate(sorted(seeds, key=lambda s: s["seed_id"])):
        r = rep.get(sd["seed_id"], {})
        if 0 not in r or 1 not in r: dropped["missing"] += 1; continue
        if r[0]["finish"] == "length" or r[1]["finish"] == "length" or len(r[0]["response"].split()) < 20 or len(r[1]["response"].split()) < 20:
            dropped["truncated_or_short"] += 1; continue
        jobs.append((len(jobs), sd["seed_id"], "AAE" if len(jobs) % 20 < 11 else "SGE", sd["text"], r[0]["response"], r[1]["response"]))
    with Pool(a.workers) as pool:
        res = pool.map(work, jobs, chunksize=16)
    byid = {j[1]: j for j in jobs}; rows, dropped["no_feature"] = [], 0
    for i, sid, v, pv, c, nf, feats, npf in res:
        if nf == 0: dropped["no_feature"] += 1; continue
        j = byid[sid]; rows.append({"seed_id": sid, "variety": v, "prompt": pv, "chosen_T": c, "reject_T": j[4], "reject_C": j[5], "n_feats": nf, "feats": feats, "prompt_feats": npf})
    # re-interleave AAE/SGE so that every prefix is balanced after the drops
    A = [r for r in rows if r["variety"] == "AAE"]; S = [r for r in rows if r["variety"] == "SGE"]; rows = [x for p in zip(A, S) for x in p]
    import pyarrow as pa, pyarrow.parquet as pq
    schema = pq.read_schema(a.base)
    def msg(p, r): return [{"content": p, "role": "user"}, {"content": r, "role": "assistant"}]
    def write(name, sel, kind):
        recs = []
        for r in sel:
            ch, rj = {"T": (r["chosen_T"], r["reject_T"]), "C": (r["reject_T"], r["reject_C"]), "A": (r["reject_T"], r["chosen_T"])}[kind]
            recs.append({"chosen": msg(r["prompt"], ch), "rejected": msg(r["prompt"], rj), "chosen_model": "OLMo-2-0425-1B-SFT" + ("+F_train" if kind == "T" else ""),
                         "rejected_model": "OLMo-2-0425-1B-SFT" + ("+F_train" if kind == "A" else ""), "id": f"e5_{kind}_{r['seed_id']}", "source": f"e5_{kind}_{r['variety']}"})
        d = Path(a.out_root) / name / "data"; d.mkdir(parents=True, exist_ok=True)
        tbl = pa.Table.from_pylist(recs, schema=schema).replace_schema_metadata(None)
        pq.write_table(tbl, d / "train-00000-of-00001.parquet"); return len(recs)
    n = len(rows); sizes = {}
    for N in a.doses:
        if N > len(rows): continue
        sel = rows[:N]
        for kind, pre in (("T", "T"), ("C", "C"), ("A", "A")):
            if kind == "A" and N >= 20000: continue
            sizes[f"{pre}{N // 1000}k"] = write(f"{pre}{N // 1000}k", sel, kind)
    stats = {"seeds_in": len(seeds), "pairs_valid": n, "dropped": dropped, "sizes": sizes, "mean_feats_per_chosen": sum(r["n_feats"] for r in rows) / max(1, n),
             "feature_counts": {}, "note": "F_train only; siao not used"}
    for r in rows:
        for f in r["feats"]: stats["feature_counts"][f"{r['variety']}.{f}"] = stats["feature_counts"].get(f"{r['variety']}.{f}", 0) + 1
    json.dump(stats, open(Path(a.out_root) / "PAIRS_STATS.json", "w"), indent=1)
    with open(Path(a.out_root) / "pairs_preview.jsonl", "w") as f:
        for r in rows[:30]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
