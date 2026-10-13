"""Compute LSM / marker coordination / Blevins convergence on every (prompt, reply) pair of the P0 generation files.

  python run_measures.py --gen DIR --out P0/style_scores.parquet --summary P0/style_summary.json [--cache DIR] [--force]

Per gen file a cache parquet is written (interruption-safe / incremental); then all caches are concatenated and
summarised.  Every measure is computed on raw replies and on `_novel` replies (words in copied 4-grams / quotes removed).
CPU only.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blevins  # noqa: E402
import coordination  # noqa: E402
import lsm as lsm_mod  # noqa: E402
from common import CATS, copied_ranges  # noqa: E402

KEYS = ["model", "condition", "seed_id", "sampling_seed", "sample"]
COND_ALT = ["INF", "AAE", "SGE"]


def score_file(df, nproc, log=print):
    """df: rows of ONE gen file -> DataFrame of per-response scores."""
    nlp = blevins.load_nlp()
    prompts = {}
    for (c, s), p in df.groupby(["condition", "seed_id"]).prompt:
        assert p.nunique() == 1, f"prompt not unique for {c},{s}"
        prompts[(c, s)] = p.iloc[0]
    pkeys = list(prompts)
    ptoks = {k: blevins.toks_from_doc(d) for k, d in zip(pkeys, nlp.pipe([prompts[k] for k in pkeys], batch_size=64))}
    replies = df.response.fillna("").tolist()
    t0 = time.time()
    rtoks = [blevins.toks_from_doc(d) for d in nlp.pipe(replies, batch_size=32, n_process=nproc)]
    log(f"  spaCy {len(replies)} replies {time.time() - t0:.0f}s")
    # shuffled-prompt baseline: reply of seed i paired with the prompt of another seed of the same condition
    rng = np.random.default_rng(0)
    shift = {}
    for c in df.condition.unique():
        ids = sorted(df.seed_id[df.condition == c].unique())
        k = int(rng.integers(1, len(ids)))
        shift[c] = {ids[i]: ids[(i + k) % len(ids)] for i in range(len(ids))}
    rows = []
    for (_, r), rep, rt in zip(df.iterrows(), replies, rtoks):
        pr, k = r.prompt, (r.condition, r.seed_id)
        row = {c: r[c] for c in KEYS + ["lineage", "stage", "scaffold", "finish", "task_type", "context"] if c in r}
        _, row["copy_rate"] = copied_ranges(rep, pr)
        for novel, sfx in ((False, ""), (True, "_novel")):
            l = lsm_mod.score(pr, rep, novel=novel)
            for kk, v in l.items():
                if kk.startswith("p_") and novel:
                    continue                      # prompt side identical
                if kk == "n_p" and novel:
                    continue
                row[kk + sfx] = v
            for kk, v in coordination.score(pr, rep, novel=novel).items():
                if kk.startswith("x_") and novel:
                    continue
                row[kk + sfx] = v
            trg = blevins.drop_copied(rt, rep, pr) if novel else rt
            for kk, v in blevins.features(trg, ptoks[k]).items():
                if kk == "n_tok_p" and novel:
                    continue
                row[kk + sfx] = v
        # shuffled baseline (raw only)
        sk = (r.condition, shift[r.condition][r.seed_id])
        row["lsm_shuf"] = lsm_mod.score(prompts[sk], rep)["lsm"]
        row["blevins_conv_shuf"] = blevins.features(rt, ptoks[sk])["blevins_conv"]
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summary
def _ci(v, alpha=0.05):
    return [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]


def cluster_sums(g, cols, ids):
    """per-seed_id (sum, count) matrices over cols (NaN ignored)."""
    a = g[cols].to_numpy(float)
    ok = np.isfinite(a)
    a0 = np.where(ok, a, 0.0)
    sid = g.seed_id.to_numpy()
    S = np.stack([a0[sid == k].sum(0) for k in ids])
    N = np.stack([ok[sid == k].sum(0) for k in ids])
    return S, N


def summarise(sc, B=2000, seed=0, log=print):
    meas = [c for c in sc.columns if (c.startswith(("lsm", "blevins_", "copy_rate")) and sc[c].dtype != object)]
    meas = [c for c in meas if c not in ("n_p",)]
    out = {"definition": "per model x condition mean over responses, 95% percentile bootstrap CI over seed_id clusters "
                         f"(B={B}); style_ai = mean(cond) - mean(SAE) per model, paired bootstrap over shared seed_ids; "
                         "coordination = P(reply has marker | prompt has marker) - P(reply has marker), pooled per group",
           "B": B, "measures": meas, "groups": {}, "style_ai": {}, "coordination": {}, "coordination_ai": {}}
    rng = np.random.default_rng(seed)
    for model, gm in sc.groupby("model"):
        ids_all = sorted(gm.seed_id.unique())
        idx = rng.integers(0, len(ids_all), size=(B, len(ids_all)))
        pre = {}
        for cond, g in gm.groupby("condition"):
            ids = sorted(g.seed_id.unique())
            if ids != ids_all:            # align cluster set; missing clusters contribute zero counts
                gg = g
                S = np.zeros((len(ids_all), len(meas))); N = np.zeros_like(S)
                s2, n2 = cluster_sums(gg, meas, ids)
                for j, k in enumerate(ids):
                    S[ids_all.index(k)], N[ids_all.index(k)] = s2[j], n2[j]
            else:
                S, N = cluster_sums(g, meas, ids)
            pre[cond] = (S, N)
            with np.errstate(all="ignore"):
                mean = S.sum(0) / N.sum(0)
                bm = S[idx].sum(1) / N[idx].sum(1)
            out["groups"].setdefault(model, {})[cond] = {
                m: {"mean": float(mean[i]), "ci": _ci(bm[:, i][np.isfinite(bm[:, i])]) if np.isfinite(bm[:, i]).any() else [None, None],
                    "n": int(N[:, i].sum())} for i, m in enumerate(meas)}
            # coordination
            for kind, sfx in (("raw", ""), ("novel", "_novel")):
                x = g[[f"x_{c}" for c in CATS]].to_numpy(); y = g[[f"y_{c}{sfx}" for c in CATS]].to_numpy()
                out["coordination"].setdefault(model, {}).setdefault(cond, {})[kind] = coordination.coordination(
                    x, y, g.seed_id.to_numpy(), B=B, seed=seed)
        if "SAE" in pre:
            S0, N0 = pre["SAE"]
            for cond in COND_ALT:
                if cond not in pre:
                    continue
                S1, N1 = pre[cond]
                with np.errstate(all="ignore"):
                    d = S1.sum(0) / N1.sum(0) - S0.sum(0) / N0.sum(0)
                    bd = S1[idx].sum(1) / N1[idx].sum(1) - S0[idx].sum(1) / N0[idx].sum(1)
                out["style_ai"].setdefault(model, {})[cond] = {
                    m: {"delta": float(d[i]), "ci": _ci(bd[:, i][np.isfinite(bd[:, i])]) if np.isfinite(bd[:, i]).any() else [None, None]}
                    for i, m in enumerate(meas)}
                g1, g0 = gm[gm.condition == cond], gm[gm.condition == "SAE"]
                for kind, sfx in (("raw", ""), ("novel", "_novel")):
                    cx = [f"x_{c}" for c in CATS]; cy = [f"y_{c}{sfx}" for c in CATS]
                    out["coordination_ai"].setdefault(model, {}).setdefault(cond, {})[kind] = coordination.coordination_contrast(
                        g1[cx].to_numpy(), g1[cy].to_numpy(), g1.seed_id.to_numpy(),
                        g0[cx].to_numpy(), g0[cy].to_numpy(), g0.seed_id.to_numpy(), B=B, seed=seed)
        log(f"  summarised {model}")
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows per file")
    a = ap.parse_args()
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "style_cache")
    os.makedirs(cache, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.gen, "*.jsonl")))
    parts = []
    for f in files:
        name = os.path.basename(f)[:-6]
        cp = os.path.join(cache, name + ".parquet")
        fresh = os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(f)
        if (fresh and not a.force) or a.summarize_only:
            if os.path.exists(cp):
                parts.append(pd.read_parquet(cp)); print(f"cached {name}"); continue
            continue
        df = pd.read_json(f, lines=True, dtype={"seed_id": str})
        if a.limit:
            df = df.head(a.limit)
        print(f"scoring {name}: {len(df)} rows", flush=True)
        t0 = time.time()
        sc = score_file(df, a.nproc, log=lambda s: print(s, flush=True))
        sc.to_parquet(cp)
        print(f"  done {name} {time.time() - t0:.0f}s", flush=True)
        parts.append(sc)
    sc = pd.concat(parts, ignore_index=True)
    sc.to_parquet(a.out)
    print(f"wrote {a.out} rows={len(sc)} models={sorted(sc.model.unique())}", flush=True)
    summ = _clean(summarise(sc, B=a.B))
    with open(a.summary, "w") as fh:
        json.dump(summ, fh, indent=1)
    print(f"wrote {a.summary}", flush=True)


if __name__ == "__main__":
    main()
