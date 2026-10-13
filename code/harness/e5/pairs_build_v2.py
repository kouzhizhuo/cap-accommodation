"""METHOD data builder v3 (P3, 2026-10-05; notes/METHOD_DESIGN.md): CAP = Calibrated Accommodation Pairs.  Env p3mv (CPU).  Reuses the E5 transforms (F_train features ONLY).
Per seed (variety v = AAE 55% / SGE 45% by position, as in E5): prompt_v = F_train transform of the SAE prompt; d_p = F_train hits per 100 words of prompt_v (the USER's density);
sentence-level variants of the SFT reply r0; MIRROR = random subset of transformed sentences until hits >= max(1, round(d_p*words/100)); FULL = every sentence transformed; ORIG = r0 (whitespace-matched).
Pair types:  U dial_matched (prompt_v; chosen MIRROR, rejected ORIG) | Uu dial_unmatched (prompt_v; chosen FULL, rejected ORIG; the E5 recipe) | O dial_over (prompt_v; chosen MIRROR, rejected FULL; kept only if
FULL hits >= 2x MIRROR hits) | A sae_cond (SAE prompt; chosen ORIG, rejected FULL).
Seeds are partitioned into ROLE POOLS (each seed is used once per set; sets share seeds): X (O role; only seeds whose FULL has >=2x MIRROR hits), U1, A, plus a VALIDATION tail (300 seeds, never in any pair).
With N added pairs:  CAP = U1[N/2] + O(X)[N/4] + A[N/4];  MC (-O) = U(U1+X)[3N/4] + A;  MU (-matching) = Uu(U1+X) + A;  MUO (-A) = U(U1+A) + O(X);  T (E5 recipe) = Uu on U1+X+A... [N].
Also CAP at smaller doses (--small, prefixes of every pool).  Mixed datasets (--base-sub rows of --base + added, permuted seed 0) are written to <out-root>/mix_<NAME>/data/.
Validation stimuli (SAE/AAE/SGE, F_train render, 300 requests) -> <out-root>/val/.  PAIRS_V3_STATS.json reports density ratios, O-pair hit ratio and seed reuse.
  python pairs_build_v2.py --seeds pool_all.jsonl --replies replies_all.jsonl --base <base parquet> --out-root DIR --n 60000 --small 20000 --ablate --workers 64"""
import argparse, json, random, sys
from multiprocessing import Pool
from pathlib import Path
EXP = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(EXP / "stimuli")); sys.path.insert(0, str(EXP)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairs_build as PB
from detectors import split_items
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}


def n_train(text, v): return sum(len(split_items.spans_of(g, text)) for g in TRAIN[v])


def sent_pairs(text, v, key):
    """list of lines; each line a list of (orig_sentence, transformed_sentence) -- same per-sentence transformation procedure as pairs_build.transform"""
    rng = random.Random(key); out = []
    for li, sents in enumerate(PB.split_sents(text)):
        if v == "AAE":
            new = []
            for si, s in enumerate(sents):
                if len(s.split()) < 3: new.append(s); continue
                o, r, att, _ = PB.build.transform_sentence(PB.dialect(), "AAE", s, f"{key}:{li}:{si}"); new.append(o)
            new, _ = PB.lex_aae(new, rng)
        else: new, _ = PB.lex_sge(sents, rng)
        out.append(list(zip(sents, new)))
    return out


def join(lines, pick):   # pick(line_idx, sent_idx) -> use transformed?
    return "\n".join(" ".join(n if pick(li, si) else o for si, (o, n) in enumerate(line)) for li, line in enumerate(lines))


CAP_OVER = 1.25   # MIRROR overshoot cap (v3): achieved hits <= 1.25 x target


def pick_mirror(nf, T, rng):
    """subset of transformable sentences whose F_train hit sum is CLOSEST to the target T (<= CAP_OVER*T); returns (set_of_sites, achieved) or None (zero target / no feasible subset)"""
    sites = [k for k, c in nf.items() if c > 0]; rng.shuffle(sites)
    if T < 0.5 or not sites: return None
    cap = CAP_OVER * T; reach = {0: ()}
    for k in sites:
        c = nf[k]
        for sm, ch in list(reach.items()):
            if sm + c <= cap + 1e-9 and sm + c not in reach: reach[sm + c] = ch + (k,)
    cand = [sm for sm in reach if sm > 0]
    if not cand: return None
    best = min(cand, key=lambda sm: (abs(sm - T), rng.random()))
    return set(reach[best]), best


def work(args):
    sid, v, prompt, r0, ratio, salt, r1 = args
    pv, _ = PB.transform(prompt, v, f"e5p{salt}:{sid}"); d_p = 100.0 * n_train(pv, v) / max(1, len(pv.split()))
    lines = sent_pairs(r0, v, f"e5r{salt}:{sid}"); words = max(1, len(r0.split()))
    nf = {(li, si): n_train(n, v) - n_train(o, v) for li, line in enumerate(lines) for si, (o, n) in enumerate(line)}
    orig = join(lines, lambda li, si: False); full = join(lines, lambda li, si: True)
    n_full = sum(c for c in nf.values() if c > 0)
    if n_full == 0: return {"seed_id": sid, "drop": "no_feature_in_full", "salt": salt}
    T = ratio * d_p * words / 100.0; pm = pick_mirror(nf, T, random.Random(f"sel{salt}:{sid}"))
    if pm is None: return {"seed_id": sid, "drop": "zero_or_infeasible_target", "target": T, "a_only": True, "variety": v, "prompt_sae": prompt, "prompt_v": pv, "orig": orig, "full": full, "r1": r1, "n_full": n_full, "words": words}
    chosen_set, cum = pm
    matched = join(lines, lambda li, si: (li, si) in chosen_set)
    if matched == orig: return {"seed_id": sid, "drop": "identical"}
    return {"seed_id": sid, "variety": v, "prompt_sae": prompt, "prompt_v": pv, "r1": r1, "orig": orig, "full": full, "matched": matched, "d_prompt": d_p, "words": words, "target_hits": T, "salt": salt,
            "n_full": n_full, "n_matched": cum, "d_full": 100.0 * n_full / words, "d_matched": 100.0 * cum / words}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds"); ap.add_argument("--replies"); ap.add_argument("--base"); ap.add_argument("--out-root")
    ap.add_argument("--n", type=int, default=60000); ap.add_argument("--small", type=int, nargs="*", default=[20000]); ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--base-sub", type=int, default=0, help="rows of --base in the mixes (0 = all)"); ap.add_argument("--n-val", type=int, default=300); ap.add_argument("--n-f300", type=int, default=300)
    ap.add_argument("--salt2", default="b", help="salt of the second rendering pass used when the pool of U-capable seeds is too small ('' = off)"); ap.add_argument("--ctl", action="store_true", help="also write CTL (r0 > r1, both standard; same prompts) and UNM (FULL > r0 on the dialect prompts; E5 recipe) over the SAME seeds as the primary CAP set (METHOD v4 Qwen3-8B arms)"); ap.add_argument("--exclude-ids", default=None, help="jsonl/records files whose seed_id rows are removed from the pool (holdouts of another build)"); ap.add_argument("--o-ratio", type=float, default=2.0); ap.add_argument("--ratio", type=float, default=1.0); ap.add_argument("--workers", type=int, default=48); ap.add_argument("--reply-key", default="response"); ap.add_argument("--no-mix", action="store_true")
    a = ap.parse_args(); Path(a.out_root).mkdir(parents=True, exist_ok=True)
    seeds = [json.loads(l) for l in open(a.seeds)]; rep = {}
    for l in open(a.replies):
        r = json.loads(l); rep.setdefault(r["seed_id"], {})[r["sampling_seed"]] = r
    excl = set()
    for fn in (a.exclude_ids.split(",") if a.exclude_ids else []): excl |= {json.loads(l)["seed_id"] for l in open(fn)}
    seeds = [sd for sd in seeds if sd["seed_id"] not in excl]
    jobs = []; val_src = []
    for sd in sorted(seeds, key=lambda s: s["seed_id"]):
        r = rep.get(sd["seed_id"], {})
        if 0 not in r or r[0]["finish"] == "length" or len(r[0]["response"].split()) < 20: continue
        if a.ctl and (1 not in r or r[1]["finish"] == "length" or len(r[1]["response"].split()) < 20): continue
        jobs.append((sd["seed_id"], "AAE" if len(jobs) % 20 < 11 else "SGE", sd["text"], r[0]["response"], a.ratio, "", r[1]["response"] if 1 in r else "")); val_src.append(sd)
    with Pool(a.workers) as pool:
        allres = [x for x in pool.map(work, jobs, chunksize=16) if x]
        res2 = [x for x in pool.map(work, [j[:5] + (a.salt2, j[6]) for j in jobs], chunksize=16) if x and "drop" not in x] if a.salt2 else []   # second RENDERING pass (same replies, different random F_train render / MIRROR subset)
    import collections
    drops = collections.Counter(x["drop"] for x in allres if "drop" in x); res = [x for x in allres if "drop" not in x]; aonly = [x for x in allres if x.get("a_only")]   # A pairs need only features in FULL
    byid = {sd["seed_id"]: sd for sd in seeds}
    # validation (V300) and fresh test (F300): the last usable seeds by seed_id order; disjoint from every training pair
    nh = a.n_val + a.n_f300; val = res[len(res) - a.n_val:]; f300 = res[len(res) - nh:len(res) - a.n_val]; res = res[:len(res) - nh]
    with open(Path(a.out_root) / "F300_seeds.jsonl", "w") as f:
        for r in f300: sd = byid[r["seed_id"]]; f.write(json.dumps({"seed_id": sd["seed_id"], "task_type": sd.get("task_type"), "context": sd.get("context"), "text": sd["text"]}, ensure_ascii=False) + "\n")
    with open(Path(a.out_root) / "V300_seeds.jsonl", "w") as f:
        for r in val: sd = byid[r["seed_id"]]; f.write(json.dumps({"seed_id": sd["seed_id"], "task_type": sd.get("task_type"), "context": sd.get("context"), "text": sd["text"]}, ensure_ascii=False) + "\n")
    def inter(rows):
        A = [r for r in rows if r["variety"] == "AAE"]; S = [r for r in rows if r["variety"] == "SGE"]; return [x for p in zip(A, S) for x in p]
    hid = {r["seed_id"] for r in val + f300}; res2 = [r for r in res2 if r["seed_id"] not in hid]
    Q = [r for r in res if r["n_full"] >= a.o_ratio * r["n_matched"]]; Q2 = [r for r in res2 if r["n_full"] >= a.o_ratio * r["n_matched"]]; qid = {r["seed_id"] for r in Q}
    N = a.n; nU, nO, nA = N // 2, N // 4, N // 4
    X = inter(Q)[:nO]; xid = {r["seed_id"] for r in X}
    if len(X) < nO: X += [r for r in inter(Q2) if r["seed_id"] not in xid][: nO - len(X)]; xid = {r["seed_id"] for r in X}      # O from the second rendering (new O-qualifying renders; seed reuse reported)
    rest = inter([r for r in res if r["seed_id"] not in xid]); U1 = rest[:nU]; seen = {r["seed_id"] for r in rest}
    new2 = inter([r for r in res2 if r["seed_id"] not in seen and r["seed_id"] not in xid]); reuse2 = inter([r for r in res2 if r["seed_id"] in seen and r["seed_id"] not in xid])
    extraU = rest[nU:] + new2 + reuse2
    Ufill = extraU[:max(0, nO - len(X))]; leftover = extraU[len(Ufill):]     # U pairs fill the O slots (and the shortfall of U-capable seeds); 2nd-pass renders of already-used seeds come last
    Apool = (inter(aonly) + leftover)[:nA]; leftover = leftover[max(0, nA - len(aonly)):] if len(aonly) < nA else leftover     # A role: seeds with no feasible MIRROR first (not usable for U/O), then leftovers; each seed once
    if len(X) < nO or len(Apool) < nA or len(U1) < nU: print("WARNING pool: O", len(X), "need", nO, "A", len(Apool), "need", nA, "U1", len(U1), "need", nU, "U-fill", len(Ufill), file=sys.stderr)
    import pyarrow as pa, pyarrow.parquet as pq
    schema = pq.read_schema(a.base)
    def msg(p, r): return [{"content": p, "role": "user"}, {"content": r, "role": "assistant"}]
    def rec(kind, r):
        if kind == "U": p, c, j = r["prompt_v"], r["matched"], r["orig"]
        elif kind == "Uu": p, c, j = r["prompt_v"], r["full"], r["orig"]
        elif kind == "O": p, c, j = r["prompt_v"], r["matched"], r["full"]
        else: p, c, j = r["prompt_sae"], r["orig"], r["full"]
        return {"chosen": msg(p, c), "rejected": msg(p, j), "chosen_model": "OLMo-2-SFT" + ("" if kind == "A" else "+F_train"), "rejected_model": "OLMo-2-SFT" + ("+F_train" if kind in ("A", "O") else ""),
                "id": f"v3_{kind}_{r['seed_id']}", "source": f"v3_{kind}_{r['variety']}"}
    def mk(parts): return [rec(k, r) for k, rows in parts for r in rows]
    sets = {}
    UU = U1 + Ufill
    def cap(n, tag):
        a_, o_ = n // 4, min(len(X), n // 4); u_ = n - a_ - o_
        sets[f"CAP{tag}"] = mk([("U", UU[:u_]), ("O", X[:o_]), ("A", Apool[:a_])])
    cap(N, f"{N // 1000}k")
    if a.ctl:
        a_, o_ = N // 4, min(len(X), N // 4); u_ = N - a_ - o_; tag = f"{N // 1000}k"
        allp = [("U", r) for r in UU[:u_]] + [("O", r) for r in X[:o_]] + [("A", r) for r in Apool[:a_]]
        def recC(kind, r):
            p = r["prompt_sae"] if kind == "A" else r["prompt_v"]
            return {"chosen": msg(p, r["orig"]), "rejected": msg(p, r["r1"]), "chosen_model": "SFT-r0", "rejected_model": "SFT-r1", "id": f"v3_C_{kind}_{r['seed_id']}", "source": f"v3_C_{kind}"}
        def recU(kind, r): return {"chosen": msg(r["prompt_v"], r["full"]), "rejected": msg(r["prompt_v"], r["orig"]), "chosen_model": "SFT+F_train", "rejected_model": "SFT", "id": f"v3_UNM_{kind}_{r['seed_id']}", "source": f"v3_UNM_{kind}"}
        sets[f"CTL{tag}"] = [recC(k, r) for k, r in allp]; sets[f"UNM{tag}"] = [recU(k, r) for k, r in allp]
    for n in a.small: cap(n, f"{n // 1000}k")
    if a.ablate:
        sets[f"MC{N // 1000}k"] = mk([("U", (UU + X)[:N - nA]), ("A", Apool)])                                        # -O: U fills the O slots (same # of mirrored chosen)
        sets[f"MUO{N // 1000}k"] = mk([("U", (UU + leftover)[:N - len(X)]), ("O", X)])                                # -A: no SAE-conditional pairs
        sets[f"T{N // 1000}k"] = mk([("Uu", (UU + X + leftover)[:N])])                                                # E5 recipe: unmatched one-sided
    sizes = {}; base_tbl = None
    if not a.no_mix:
        import numpy as np
        base_tbl = pq.read_table(a.base)
        if a.base_sub and a.base_sub < base_tbl.num_rows: base_tbl = base_tbl.take(pa.array(np.sort(np.random.default_rng(0).choice(base_tbl.num_rows, a.base_sub, replace=False))))
    for name, recs in sets.items():
        tbl = pa.Table.from_pylist(recs, schema=schema).replace_schema_metadata(None)
        d = Path(a.out_root) / name / "data"; d.mkdir(parents=True, exist_ok=True); pq.write_table(tbl, d / "train-00000-of-00001.parquet"); sizes[name] = len(recs)
        if base_tbl is not None:
            import numpy as np
            t = pa.concat_tables([base_tbl.cast(schema), tbl]); t = t.take(pa.array(np.random.default_rng(0).permutation(t.num_rows)))
            d = Path(a.out_root) / f"mix_{name}" / "data"; d.mkdir(parents=True, exist_ok=True); pq.write_table(t.replace_schema_metadata(None), d / "train-00000-of-00001.parquet"); sizes[f"mix_{name}"] = t.num_rows
    # validation stimuli: every request rendered in AAE and SGE with F_train (+ SAE); same record format as the E5 stimuli dirs
    vd = Path(a.out_root) / "val"; vd.mkdir(parents=True, exist_ok=True); fo = {c: open(vd / f"{c}.jsonl", "w") for c in ("SAE", "AAE", "SGE")}
    for r in val:
        sd = byid[r["seed_id"]]; base = {"seed_id": r["seed_id"], "task_type": sd.get("task_type"), "context": sd.get("context")}
        fo["SAE"].write(json.dumps({**base, "condition": "SAE", "text": sd["text"]}, ensure_ascii=False) + "\n")
        for v in ("AAE", "SGE"):
            pv, _ = PB.transform(sd["text"], v, f"e5p:{r['seed_id']}"); fo[v].write(json.dumps({**base, "condition": v, "text": pv, "source_text": sd["text"], "d_train": 100.0 * n_train(pv, v) / max(1, len(pv.split()))}, ensure_ascii=False) + "\n")
    for f in fo.values(): f.close()
    import statistics as st
    def rat(rows, k): return [r[k] / r["d_prompt"] for r in rows if r["d_prompt"] > 0]
    Uall = U1 + Ufill + X; rm, rf = rat(Uall, "d_matched"), rat(Uall, "d_full")
    Orat = [r["n_full"] / max(1, r["n_matched"]) for r in X]
    stats = {"seeds_in": len(seeds), "usable": len(res) + len(val), "validation": len(val), "o_qualifying_seeds": len(Q), "role_sizes": {"U1": len(U1), "U_fill_for_O_slots": len(Ufill), "X(O)": len(X), "A": len(Apool), "leftover_U_capable": len(leftover), "second_pass_new_seeds": len(new2), "second_pass_reused_seeds": len(reuse2), "n_pass2_usable": len(res2)}, "o_ratio": a.o_ratio, "o_qualifying_at": {str(t): sum(1 for r in res if r["n_full"] >= t * r["n_matched"]) for t in (1.25, 1.5, 2.0, 3.0)}, "N": N, "sizes": sizes,
             "seed_reuse": "within a set every seed is used once; all sets share the same role pools (the T/MC/MU/MUO ablations re-assign roles)",
             "density_ratio_chosen_over_prompt": {"matched_median": st.median(rm), "matched_mean": st.mean(rm), "matched_share_gt_1.5x": sum(x > 1.5 for x in rm) / len(rm),
                                                   "unmatched_median": st.median(rf), "unmatched_mean": st.mean(rf), "unmatched_share_gt_1.5x": sum(x > 1.5 for x in rf) / len(rf)},
             "o_share_of_usable": len(Q) / max(1, len(res)), "o_qualifying_at_1.5x": sum(1 for r in res if r["n_full"] >= 1.5 * r["n_matched"]), "a_only_seeds": len(aonly), "O_pair_hit_ratio_full_over_matched": {"mean": st.mean(Orat) if Orat else None, "median": st.median(Orat) if Orat else None},
             "mean_prompt_density_per100w": st.mean(r["d_prompt"] for r in Uall), "mean_matched_density_per100w": st.mean(r["d_matched"] for r in Uall), "mean_full_density_per100w": st.mean(r["d_full"] for r in Uall)}
    err = [r["n_matched"] - r["target_hits"] for r in Uall]; rel = [abs(r["n_matched"] - r["target_hits"]) / r["target_hits"] for r in Uall]
    stats["drops"] = dict(drops); stats["seeds_after_drops"] = len(res) + len(val) + len(f300); stats["mirror_target_minus_achieved_hits"] = {"mean": st.mean(err), "median": st.median(err), "mean_abs_rel": st.mean(rel), "share_within_25pct": sum(x <= 0.25 for x in rel) / len(rel), "cap_over": CAP_OVER}
    stats["n_f300"], stats["n_v300"] = len(f300), len(val)
    for nm, rows in (("U1", U1 + Ufill), ("X", X), ("A", Apool), ("V300", val), ("F300", f300)):
        with open(Path(a.out_root) / f"records_{nm}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # mirrored-target SFT data (messages format): every CAP60k source -> (prompt_v -> MIRROR) for U/O, (prompt_sae -> r0) for A
    sft = [{"id": f"sft_U_{r['seed_id']}", "messages": msg(r["prompt_v"], r["matched"])} for r in UU[:N - nA - min(len(X), nO)]] + [{"id": f"sft_O_{r['seed_id']}", "messages": msg(r["prompt_v"], r["matched"])} for r in X[:nO]] \
          + [{"id": f"sft_A_{r['seed_id']}", "messages": msg(r["prompt_sae"], r["orig"])} for r in Apool[:N // 4]]
    d = Path(a.out_root) / f"SFTCAP{N // 1000}k"; d.mkdir(exist_ok=True); import random as _r; _r.Random(0).shuffle(sft)
    with open(d / "train.jsonl", "w") as f:
        for r in sft: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(stats, open(Path(a.out_root) / "PAIRS_V3_STATS.json", "w"), indent=1)
    with open(Path(a.out_root) / "pairs_v3_preview.jsonl", "w") as f:
        for r in (U1[:8] + X[:6] + Apool[:6]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
