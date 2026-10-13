#!/usr/bin/env python
"""E4 Day-0 regime check, stage 3: aggregate bulk/*.parquet (+ fullsample) -> prefdata_counts.json, prefdata_examples.jsonl.
Written incrementally (prefdata_counts.json is rewritten after every dataset).  CPU only.

  cd .../exp && source ../../../envs/p3mv/bin/activate && MV_COREF=0 python E4/prefdata_analyze.py
"""
import json, os, re, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import detectors  # noqa: F401
from prefdata_bulk import RETAINED_MORPH, LEXRATE, FNAMES, jobs_pref, FEATS

PREF = ["olmo-2-0425-1b-preference-mix", "olmo-2-1124-7b-preference-mix", "llama-3.1-tulu-3-8b-preference-mixture"]
B = 2000
MORPH = {v: [f"{v}.{f}" for f in fs] for v, fs in RETAINED_MORPH.items()}
LEX = {v: [f"{v}.{f}" for f in fs] for v, fs in LEXRATE.items()}
ALLV = {v: MORPH[v] + LEX[v] for v in MORPH}
BREF = ["BRE.spelling", "BRE.lexis", "BRE.have_got"]
AUSF = ["AUS.aus_lexis"]


def cols(tag, names):
    return [f"{tag}.{n}" for n in names]


def hit_masks(df, tag):
    """dict name -> boolean array (>=1 hit) for a text tag in {p, c, r, craw, rraw}."""
    m = {}
    for v in ("AAE", "SGE"):
        m[f"{v}.morph"] = (df[cols(tag, MORPH[v])].sum(axis=1) > 0).values
        m[f"{v}.lex"] = (df[cols(tag, LEX[v])].sum(axis=1) > 0).values
        m[f"{v}.any"] = m[f"{v}.morph"] | m[f"{v}.lex"]
    m["union.morph"] = m["AAE.morph"] | m["SGE.morph"]
    m["union.lex"] = m["AAE.lex"] | m["SGE.lex"]
    m["union.any"] = m["union.morph"] | m["union.lex"]
    m["BRE.any"] = (df[cols(tag, BREF)].sum(axis=1) > 0).values
    m["BRE.spelling"] = (df[f"{tag}.BRE.spelling"] > 0).values
    m["BRE.lexis"] = (df[f"{tag}.BRE.lexis"] > 0).values
    m["AUS.lexis_specific"] = (df[f"{tag}.AUS.aus_lexis"] > 0).values
    m["BRE_or_AUS.any"] = m["BRE.any"] | m["AUS.lexis_specific"]
    return m


def shares(df, tag):
    n = len(df)
    return {k: {"n": int(v.sum()), "share": float(v.sum() / n)} for k, v in hit_masks(df, tag).items()}


def dens(df, tag, v, words_col):
    """hits per 100 words of the variety's retained-morph + lexical/rate features; v in AAE|SGE|union."""
    vs = ("AAE", "SGE") if v == "union" else (v,)
    h = sum(df[cols(tag, ALLV[x])].sum(axis=1) for x in vs)
    return (100.0 * h / np.maximum(1, df[words_col])).values


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": int(len(x))}


def asym(df, mask_bearing, v):
    d = df[mask_bearing]
    out = {}
    for label, ct, rt in (("novel", "c", "r"), ("raw", "craw", "rraw")):
        dc, dr = dens(d, ct, v, "c_words"), dens(d, rt, v, "r_words")
        diff = dc - dr
        out[label] = {"D_chosen_mean": float(dc.mean()) if len(d) else None, "D_rejected_mean": float(dr.mean()) if len(d) else None,
                      "D_chosen_minus_rejected": boot_ci(diff),
                      "share_chosen_gt_rejected": float((diff > 0).mean()) if len(d) else None,
                      "share_chosen_lt_rejected": float((diff < 0).mean()) if len(d) else None,
                      "share_tie": float((diff == 0).mean()) if len(d) else None}
    out["mean_words_chosen"] = float(d["c_words"].mean()) if len(d) else None
    out["mean_words_rejected"] = float(d["r_words"].mean()) if len(d) else None
    return out


def perfeature(df, tags):
    out = {}
    for t in tags:
        out[t] = {n: {"texts_with_hit": int((df[f"{t}.{n}"] > 0).sum()), "total_hits": int(df[f"{t}.{n}"].sum())} for n in FNAMES}
    return out


def dataset_block(df, name, is_pref=True, clean=False):
    b = {"n_rows": int(len(df)), "prompt": shares(df, "p")}
    if is_pref:
        b["chosen_novel"] = shares(df, "c"); b["rejected_novel"] = shares(df, "r")
        b["chosen_raw"] = shares(df, "craw"); b["rejected_raw"] = shares(df, "rraw")
        pm = hit_masks(df, "p")
        b["dialect_bearing_prompts"] = {}
        b["all_prompts_reference"] = {}
        for v in ("AAE", "SGE", "union"):
            m = pm[f"{v}.morph"]
            b["dialect_bearing_prompts"][v] = {"n": int(m.sum()), **asym(df, m, v)}
            b["all_prompts_reference"][v] = {"n": int(len(df)), **asym(df, np.ones(len(df), bool), v)}
    return b


def fullsample_block(name):
    fn = f"{HERE}/bulk/{name}.fullsample.parquet"
    if not os.path.exists(fn):
        return None
    df = pd.read_parquet(fn)
    if "error" in df: df = df[df["error"].isna()]
    from detectors.base import FEATURES
    out = {"n_scored": int(len(df)), "meta": json.load(open(fn.replace(".parquet", ".meta.json")))}
    for v in ("AAE", "SGE"):
        feats = [f for f in FEATURES[v] if f"{v}.{f.id}" in MORPH[v] and f.mv_rules]
        r = sum(df[f"{v}.{f.id}.real_n"] for f in feats); o = sum(df[f"{v}.{f.id}.opp_n"] for f in feats)
        ok = (r + o) >= 3
        Dm = (r / (r + o).replace(0, np.nan))
        out[v] = {"retained_morph_features_with_opportunity_rules": [f.id for f in feats],
                  "share_prompts_with_retained_hit": float((r > 0).mean()), "n_prompts_with_retained_hit": int((r > 0).sum()),
                  "share_prompts_with_>=3_sites": float(ok.mean()), "n_prompts_with_>=3_sites": int(ok.sum()),
                  "D_morph_mean_over_prompts_with_>=3_sites": float(Dm[ok].mean()) if ok.any() else None,
                  "D_morph_mean_over_prompts_with_>=1_hit": float(Dm[r > 0].mean()) if (r > 0).any() else None,
                  "share_prompts_with_D_morph>0": float(((Dm > 0) & ok).mean())}
    return out


def context(text, spans_, k=70):
    for s, e in spans_:
        return text[max(0, s - k):e + k].replace("\n", " ")
    return ""


def make_examples(dfs, seed=0, n_hit=40, n_non=20):
    rng = np.random.default_rng(seed)
    per_hit = [14, 13, 13]; per_non = [7, 7, 6]
    picks = {}
    for i, name in enumerate(PREF):
        df = dfs[name]; pm = hit_masks(df, "p")["union.morph"]
        h = df.index[pm].values; nh = df.index[~pm].values
        picks[name] = (set(df.loc[rng.choice(h, size=min(per_hit[i], len(h)), replace=False), "idx"].tolist()),
                       set(df.loc[rng.choice(nh, size=min(per_non[i], len(nh)), replace=False), "idx"].tolist()))
    rows = []
    for name in PREF:
        hs, ns = picks[name]
        df = dfs[name].set_index("idx")
        for (i, pid, src, p, c, rj) in jobs_pref(name):
            if i not in hs and i not in ns:
                continue
            kind = "dialect_bearing" if i in hs else "non_hit"
            feats = []; ctx = []
            for fn in FNAMES:
                if fn in FEATS and fn.split(".")[0] in ("AAE", "SGE") and df.loc[i, f"p.{fn}"] > 0 and fn in MORPH["AAE"] + MORPH["SGE"]:
                    feats.append(fn); ctx.append({fn: context(p, FEATS[fn](p[:30000]))})
            rows.append({"dataset": name, "kind": kind, "idx": int(i), "id": pid, "source": src,
                         "matched_retained_features": feats, "hit_context": ctx,
                         "prompt_200": p[:200], "chosen_200": c[:200], "rejected_200": rj[:200]})
    rows.sort(key=lambda r: (r["kind"] != "dialect_bearing", r["dataset"], r["idx"]))
    with open(f"{HERE}/prefdata_examples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    t0 = time.time()
    val = json.load(open(os.path.join(os.path.dirname(HERE), "detectors/validation_synthetic.json")))["features"]
    fpa = json.load(open(os.path.join(os.path.dirname(HERE), "detectors/fp_audit_SAE.json")))
    fp_max = {}
    for m, e in fpa["models"].items():
        for k, x in e["by_feature"].items():
            fp_max[k] = max(fp_max.get(k, 0.0), x)
    used = MORPH["AAE"] + MORPH["SGE"] + LEX["AAE"] + LEX["SGE"]
    out = {"spec": {
        "retained_morph": RETAINED_MORPH,
        "retained_rule": "validation_synthetic.json gate_pass AND fp_audit_SAE.json by_feature <= 0.02 at every audited stage (== pilot_summary.retained_set(); verified to yield exactly these 7 morph features)",
        "lexical_rate_features": LEXRATE,
        "lexical_rate_fp_audit_max_over_stages": {k: fp_max.get(k, 0.0) for k in LEX["AAE"] + LEX["SGE"]},
        "dialect_bearing_prompt": ">=1 hit of a retained morphosyntactic feature of the variety (union: AAE or SGE) in the prompt (all user turns)",
        "D": "hits per 100 words of the variety's retained-morph + lexical/rate features (union = AAE + SGE); responses: 'novel' = spans inside >=4-gram copies of the prompt and quotations dropped (primary), 'raw' = no filter",
        "bootstrap": f"mean of paired per-prompt difference, B={B}, percentile 95% CI, seed 0",
        "text_truncation_chars": 30000,
        "validation_precision_recall_of_retained_features": {k: {kk: val[k][kk] for kk in ("n_pos", "recall", "FPR", "FP_all", "n_neg", "precision_balanced")} for k in MORPH["AAE"] + MORPH["SGE"]},
        "fp_audit_SAE_max_over_stages": {k: fp_max.get(k, 0.0) for k in used},
        "fp_audit_note": f"target {fpa['target']}, n={fpa['n_per_model']} SAE-condition replies per model-stage: the rate of replies to SAE prompts on which the feature fires",
    }, "datasets": {}}
    dfs = {}
    for name in PREF:
        fn = f"{HERE}/bulk/{name}.parquet"
        if not os.path.exists(fn):
            continue
        df = pd.read_parquet(fn); dfs[name] = df
        blk = dataset_block(df, name)
        clean = df[(df["p_code"] == 0) & (df["p_nonascii"] < 0.05)]
        blk["sensitivity_no_code_fence_and_<5pct_nonascii_prompt"] = dataset_block(clean, name)
        blk["per_feature_counts"] = perfeature(df, ["p", "c", "r"])
        top = df[hit_masks(df, "p")["union.morph"]]["source"].value_counts().head(12)
        blk["dialect_bearing_prompts_by_source_top12"] = {k: int(v) for k, v in top.items()}
        blk["n_pairs"] = int(len(df))
        blk["fullsample_score_text"] = fullsample_block(name)
        out["datasets"][name] = blk
        json.dump(out, open(f"{HERE}/prefdata_counts.json", "w"), indent=1)
        print("done", name, f"{time.time()-t0:.0f}s", flush=True)
    # controls
    ctrl = {}
    for name, label in (("dolly-15k", "dolly_15k_prompts_SAE_reference"), ("tulu-3-sft-mixture", "tulu3_sft_mixture_all_prompts")):
        fn = f"{HERE}/bulk/{name}.parquet"
        if os.path.exists(fn):
            df = pd.read_parquet(fn)
            ctrl[label] = dataset_block(df, name, is_pref=False)
            ctrl[label]["per_feature_prompt_counts"] = perfeature(df, ["p"])["p"]
            if name.startswith("tulu"):
                s = df.sample(n=5000, random_state=0)
                ctrl["tulu3_sft_mixture_random_5000_prompts_seed0"] = dataset_block(s, name, is_pref=False)
                cl = df[(df["p_code"] == 0) & (df["p_nonascii"] < 0.05)]
                ctrl["tulu3_sft_mixture_all_prompts_no_code_ascii"] = dataset_block(cl, name, is_pref=False)
            if name.startswith("dolly"):
                pass
    out["controls"] = ctrl
    # net-of-floor shares (floor = union.morph share on the Dolly SAE-reference prompts)
    fl = ctrl.get("dolly_15k_prompts_SAE_reference", {}).get("prompt", {})
    out["net_of_floor"] = {"floor_definition": "share of Dolly-15k instructions (human-written, mostly SAE) with the same detector hit; an upper-bound proxy for the false-positive floor on plain English, NOT for code/multilingual prompts",
                           "floor": {k: fl[k]["share"] for k in ("AAE.morph", "SGE.morph", "union.morph", "union.lex", "union.any")} if fl else None, "datasets": {}}
    for name, blk in out["datasets"].items():
        ps = blk["prompt"]; cs = blk["sensitivity_no_code_fence_and_<5pct_nonascii_prompt"]["prompt"]
        out["net_of_floor"]["datasets"][name] = {
            k: {"prompt_share": ps[k]["share"], "net_share": ps[k]["share"] - fl[k]["share"],
                "clean_subset_share": cs[k]["share"], "clean_subset_net_share": cs[k]["share"] - fl[k]["share"]}
            for k in ("AAE.morph", "SGE.morph", "union.morph")} if fl else None
    json.dump(out, open(f"{HERE}/prefdata_counts.json", "w"), indent=1)
    if all(n in dfs for n in PREF):
        print("examples", make_examples(dfs), flush=True)
    print("ALL WRITTEN", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
