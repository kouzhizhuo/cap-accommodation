#!/usr/bin/env python
"""Authentic-twin E1 post-processing (expansion A; lead 2026-10-04). For each set (auth_gro, auth_endive_aae, auth_endive_sge) and every model-stage with gens in E1/gen_auth/<set>/:
  1 refusal flag per reply = REFUSAL-PHRASE RULE (no refusal classifier exists in the pipeline; the PREREG judge-based flag (Qwen3-8B) has not been run on these gens): a reply refuses when its first 300 characters match a refusal pattern
    (apology + can't/won't/unable; "I can't/cannot/won't/am unable to help|assist|comply|provide|generate|write|..."; "as an AI ... I can't"; "I must decline/refuse"; "I don't feel comfortable"; "I'm not able to ...").
  2 PAIR EXCLUSION: for every (model, seed_id, sampling_seed, sample), if EITHER twin's reply (dialect side or SAE side) is a refusal, BOTH replies are dropped from the CR estimates; all prompts are still generated.
  3 detectors.score output (already scored: --scores-dir) is filtered accordingly, pilot_summary + emit_results are run on the filtered scores per set, and everything is merged into results/E1/authentic.json
    keyed  <lineage>.<stage>.<set>.<var>.<stat>  (stats as in summary.json: cr, cr_lo/hi, cr_rate, cr_rate_lo/hi, cr_tot, retention, ai, d_cond, d_sae ...)  plus  refusal.<lineage>.<stage>.<set>.{dialect,sae,pair_excluded_share,n_pairs}.
  python analysis/authentic_post.py --gen-root E1/gen_auth --scores-root E1 --out results/E1/authentic.json --work E1/auth_work"""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
EXP = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--gen-root", required=True); ap.add_argument("--scores-root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--py", default=sys.executable); a = ap.parse_args()
SETS = {"auth_gro": "AAE", "auth_endive_aae": "AAE", "auth_endive_sge": "SGE"}
REF = [re.compile(p, re.I) for p in [
    r"\b(i'?m|i am|we are|sorry)\b[^.\n]{0,25}\b(sorry|apolog)", r"\bsorry,? (but )?(i|we)\b[^.\n]{0,30}\b(can'?t|cannot|can not|won'?t|unable|not able|am not|do not|don'?t)",
    r"\bi (can'?t|cannot|can not|won'?t|will not|am unable to|am not able to|'m unable to|'m not able to) (help|assist|comply|fulfil|fulfill|provide|generate|create|write|engage|continue|respond|do that|answer|support|discuss|participate)",
    r"\bas an ai( language model)?,? i (can'?t|cannot|do not|don'?t|am unable|am not able)", r"\bi must (decline|refuse)", r"\bi (do not|don'?t) feel comfortable", r"\bi'?m not (able|going) to (help|assist|comply|provide|generate|create|write)",
    r"\b(this|that) (request|content|message|language) (is|contains|violates|goes against)", r"\bi (apologi[sz]e),? but\b"]]
refuses = lambda t: any(p.search((t or "")[:300]) for p in REF)
LINEAGE = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L5": "olmo2_32b", "L2B": "olmo2_13b", "Q7": "qwen25_7b", "Q15": "qwen25_1_5b", "Q72": "qwen25_72b"}
out = {"refusal_rule": "refusal-phrase rule on the first 300 characters of each reply (patterns in analysis/authentic_post.py); pair excluded if either twin's reply refuses; no classifier/judge flag", "sets": {}}
work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
for st, var in SETS.items():
    gens = sorted(Path(a.gen_root, st).glob("*@*.jsonl"))
    if not gens: continue
    key = ["model", "seed_id", "sampling_seed", "sample"]; flags = []
    for g in gens:
        for l in open(g):
            r = json.loads(l); flags.append({"model": r["model"], "condition": r["condition"], "seed_id": r["seed_id"], "sampling_seed": r["sampling_seed"], "sample": r["sample"], "ref": refuses(r["response"])})
    f = pd.DataFrame(flags); piv = f.pivot_table(index=key, columns="condition", values="ref", aggfunc="max").fillna(False)
    piv["drop"] = piv[var] | piv["SAE"]
    ref_rates = f.groupby(["model", "condition"]).ref.mean().unstack(1)
    pair_rates = piv.groupby(level="model")["drop"].mean(); npairs = piv.groupby(level="model").size()
    sp = Path(a.scores_root) / f"auth_scores_{st}.parquet"
    df = pd.read_parquet(sp); resp = df[df.kind == "response"].merge(piv[["drop"]].reset_index(), on=key, how="left"); resp = resp[~resp["drop"].fillna(False).astype(bool)].drop(columns="drop")
    filt = pd.concat([df[df.kind != "response"], resp]); fp = work / f"scores_{st}_filtered.parquet"; filt.to_parquet(fp, index=False)
    raw = work / f"summary_raw_{st}.json"; emit = work / f"emit_{st}.json"
    env = {"MV_COREF": "0"}
    subprocess.run([a.py, "-m", "detectors.pilot_summary", "--scores", str(fp), "--out", str(raw)], cwd=EXP, check=True, env={**__import__("os").environ, **env})
    subprocess.run([a.py, "analysis/emit_results.py", "--summary", str(raw), "--out", str(emit)], cwd=EXP, check=True, env={**__import__("os").environ, **env})
    e = json.load(open(emit))
    for lin, stages in e.items():
        if lin in ("meta", "prompts", "validation") or not isinstance(stages, dict): continue
        for stage, vs in stages.items():
            out.setdefault(lin, {}).setdefault(stage, {}).setdefault(st, vs)
    for m in ref_rates.index:
        lin, stg = m.split("@"); lk = LINEAGE.get(lin, lin.lower()); sk = stg.lower().replace("-", "_")
        out.setdefault("refusal", {}).setdefault(lk, {}).setdefault(sk, {})[st] = {"dialect": float(ref_rates.loc[m, var]), "sae": float(ref_rates.loc[m, "SAE"]), "pair_excluded_share": float(pair_rates.get(m, float("nan"))), "n_pairs": int(npairs.get(m, 0))}
    out["sets"][st] = {"conditions": [var, "SAE"], "models": len(ref_rates), "prompts": e.get("prompts")}
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out, list(out.get("sets", {})))
