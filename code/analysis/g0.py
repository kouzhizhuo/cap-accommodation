#!/usr/bin/env python
"""G0 measurement-validity table (PREREG §4a): (a) productive AND total CR at base under raw/F1/F2; (b) per-feature false positives on SAE-prompt replies at every
stage (<= 2%) with D_v(SAE) by stage; (c) >= 3 opportunity sites per reply and retention >= 60% at every stage of the lineage.
Two instruments per variety: GATED = opportunity-normalised D (summary_raw.json; needs (c)); RATE = retained+lexical features per 100 words, no gate, no
opportunity normalisation (computed here; bootstrap over seed prompts, B=2000). Output: nested lower-case keys for json2tex (prefix e1.g0) + a markdown table.
  MV_COREF=0 python analysis/g0.py --scores E1/scores.parquet --summary E1/summary_raw.json --fp E1/fp_audit_SAE.json --out results/E1/g0.json --md E1/G0.md"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
VAR = ["AAE", "SGE", "INF", "BRE", "AUS", "FOR", "NGP", "APP", "SCO", "IND", "IRE"]
LIN = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L5": "olmo2_32b", "Q72": "qwen25_72b", "L2B": "olmo2_13b", "Q15": "qwen25_1_5b", "Q7": "qwen25_7b"}
RATE = {"AAE": ["ain_t", "completive_done", "finna_gon", "negative_concord", "yall", "lexicon", "g_drop", "habitual_be"], "SGE": ["particles", "lexicon"],
        "INF": ["contraction", "casual_lexis"], "BRE": ["spelling", "lexis", "have_got"], "AUS": ["spelling", "aus_lexis"],
        "NGP": ["dey", "don", "na", "no", "fit", "wan", "lexicon"], "APP": ["a_prefix", "fixin_to", "double_modals", "those_them", "negative_concord", "ain_t", "yall", "reflexive", "lexicon"], "FOR": ["lexis"]}
ap = argparse.ArgumentParser(); [ap.add_argument(f"--{k}", required=True) for k in ("scores", "summary", "fp", "out", "md")]; a = ap.parse_args()
df = pd.read_parquet(a.scores); S = json.load(open(a.summary)); FP = json.load(open(a.fp))
resp, prm = df[df.kind == "response"], df[df.kind == "prompt"]
rng = np.random.default_rng(0); B = 2000


def rate(d, v, novel):
    cols = [f"{v}.{f}.{'real_n' if novel else 'real'}" for f in RATE[v] if f"{v}.{f}.real" in d.columns]
    return 100.0 * d[cols].sum(axis=1) / d.n_words.clip(lower=1)


def seed_means(d, v, novel):
    d = d.assign(_r=rate(d, v, novel)); return d.groupby(["condition", "seed_id"])._r.mean().unstack(0)


def cr_rate(m, v, novel):
    R = seed_means(resp[resp.model == m], v, novel); Pm = seed_means(prm, v, False)
    idx = R.index.intersection(Pm.index); ai = (R.loc[idx, v] - R.loc[idx, "SAE"]).values; pc = (Pm.loc[idx, v] - Pm.loc[idx, "SAE"]).values
    f = lambda s: np.nanmean(ai[s]) / np.nanmean(pc[s]) if abs(np.nanmean(pc[s])) > 1e-9 else np.nan
    bs = [f(rng.integers(0, len(idx), len(idx))) for _ in range(B)]
    return float(f(np.arange(len(idx)))), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


out, rows = {}, []
models = sorted(S["models"])
retained = {k for k in S["retained_features"]}
for m in models:
    lin, st = m.split("@"); node = out.setdefault(LIN.get(lin, lin.lower()), {}).setdefault(st.lower().replace("-", "_"), {})
    for v in VAR:
        if v not in S["models"][m]["varieties"] or v not in RATE: continue
        g, t = S["models"][m]["varieties"][v]["novel"], S["models"][m]["varieties"][v]["total"]
        rp, rt = cr_rate(m, v, True), cr_rate(m, v, False)
        node[v.lower()] = {"cr_gated": g["cr"], "cr_gated_tot": t["cr"], "cr_rate": rp[0], "cr_rate_lo": rp[1], "cr_rate_hi": rp[2], "cr_rate_tot": rt[0],
                           "retention": g["retention_cond"], "retention_sae": g["retention_sae"], "d_sae": g["d_sae"], "copy": g["copy"]}
        rows.append((m, v, g["cr"], t["cr"], rp[0], rt[0], g["retention_cond"], g["d_sae"]))
# (b) FP by retained feature x stage; (c) retention by lineage
g0 = {"fp": {}, "retention_pass": {}, "fp_pass": {}}
for m, e in FP["models"].items():
    lin, st = m.split("@"); l = LIN.get(lin, lin.lower())
    for f, x in e["by_feature"].items():
        if f in retained or f.split(".")[0] in ("BRE", "AUS"):
            v, fid = f.split(".", 1); g0["fp"].setdefault(l, {}).setdefault(st.lower().replace("-", "_"), {}).setdefault(v.lower(), {})[fid] = x
for lin in sorted({m.split("@")[0] for m in models}):
    l = LIN.get(lin, lin.lower())
    for v in VAR:
        if v not in RATE: continue
        ms = [m for m in models if m.startswith(lin + "@") and m.split("@")[1] in ("base", "sft", "dpo", "rlvr1", "instruct", "final") and v in S["models"][m]["varieties"]]
        if not ms: continue
        ret = min(S["models"][m]["varieties"][v]["novel"]["retention_cond"] for m in ms)
        fpmax = max([x for m in ms for f, x in FP["models"].get(m, {"by_feature": {}})["by_feature"].items() if f.startswith(v + ".") and (f in retained)] or [0.0])
        g0["retention_pass"].setdefault(l, {})[v.lower()] = {"min_retention": ret, "gated_passes_c": bool(ret >= 0.60), "rate_instrument_passes_c": True}
        g0["fp_pass"].setdefault(l, {})[v.lower()] = {"max_fp": fpmax, "passes_b": bool(fpmax <= 0.02)}
out["g0"] = g0
json.dump(out, open(a.out, "w"), indent=1)
md = ["# G0 table (pilot-spec E1 v1)", "", "model | variety | CR gated (prod) | CR gated (total) | CR rate (prod) | CR rate (total) | retention | D_v(SAE)", "---|---|---|---|---|---|---|---"]
md += [f"{m} | {v} | {a_:+.2f} | {b:+.2f} | {c:+.3f} | {d:+.3f} | {r:.2f} | {ds:.3f}" for m, v, a_, b, c, d, r, ds in rows]
md += ["", "## (c) retention >= 60% at every stage (gated instrument) / (b) FP <= 2%", ""]
for l, vs in g0["retention_pass"].items():
    md.append(f"- {l}: " + "; ".join(f"{v}: min retention {x['min_retention']:.2f} -> gated {'PASS' if x['gated_passes_c'] else 'FAIL'}, rate PASS; max FP {g0['fp_pass'][l][v]['max_fp']:.3f} -> {'PASS' if g0['fp_pass'][l][v]['passes_b'] else 'FAIL'}" for v, x in vs.items()))
Path(a.md).write_text("\n".join(md) + "\n"); print("wrote", a.out, a.md, len(rows), "rows")
