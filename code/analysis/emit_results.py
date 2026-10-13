#!/usr/bin/env python
"""Turn detectors/pilot_summary.py output into the json2tex key scheme of notes/STATS_PLAN.md §9.

  python analysis/emit_results.py --summary P0/pilot_summary.json --out results/P0/summary.json
  python3 paper/tools/json2tex.py exp/results/P0/summary.json:p0 exp/results/design/summary.json:design --wrap pilot -o paper/numbers/pilot.tex

Nested lower-case keys, NO prefix inside the file (the prefix p0 / e1 is given on the json2tex command line):
  <lineage>.<stage>.<variety>.<stat>      lineage in {olmo2_1b, olmo2_7b, ...}; stage in {base, base_raw, base_f1, sft, dpo, rlvr1, instruct, dpo_request, ...}
  <lineage>.<stage>.s / s_tot / len_share / mean_tokens
  prompts.<variety>.<condition>.d_pooled / lex_per100w ;  meta.*
stat (productive = novel spans; `_tot` = total, no filter):
  cr cr_lo cr_hi | cr_tot cr_tot_lo cr_tot_hi | cr_lex cr_lex_lo cr_lex_hi | ai ai_lo ai_hi | d_cond d_sae lex_cond lex_sae
  retention retention_sae | df df_lo df_hi df_minus_inf cr_minus_inf | copy
"""
import argparse, hashlib, json, math
from pathlib import Path

LINEAGE = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L5": "olmo2_32b", "Q72": "qwen25_72b", "L2B": "olmo2_13b", "Q15": "qwen25_1_5b", "Q7": "qwen25_7b", "Z7": "zephyr_7b", "T27": "tulu2_7b", "T213": "tulu2_13b", "O31": "olmo31_32b", "T370": "tulu3_70b", "opangu7b": "openpangu_embedded_7b_v11", "T270": "tulu2_70b", "O3T": "olmo3_32b_think"}   # names as in notes/RES_KEYS.md


def clean(x):
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 5)
    return x


STAGE = {"rlvr1": "rlvr", "final": "rlvr"}   # RES_KEYS stages: base/sft/dpo/rlvr/instruct


def stage_key(s):
    s = s.lower()
    return STAGE.get(s, s.replace("-", "_"))


def emit(summary, design=False, stimuli_version="v1"):
    out = {}; nseeds = 0
    for model, M in summary["models"].items():
        lin, stage = model.split("@")
        node = out.setdefault(LINEAGE.get(lin, lin.lower()), {}).setdefault(stage_key(stage), {})
        node["len_share"] = clean(M["finish_length_share"]); node["mean_tokens"] = clean(M["mean_tokens"])
        for k, v in (M.get("S") or {}).items():
            node["split" if k == "novel" else "split_tot"] = clean(v)
        for v, D in M["varieties"].items():
            vn = node.setdefault(v.lower(), {})
            n, t = D["novel"], D["total"]; nseeds = max(nseeds, n.get("n_seeds", 0))
            for stat in ("cr", "cr_lo", "cr_hi", "cr_rate", "cr_rate_lo", "cr_rate_hi", "cr_lex", "cr_lex_lo", "cr_lex_hi", "ai", "ai_lo", "ai_hi", "d_cond", "d_sae", "lex_cond", "lex_sae",
                         "df", "dF", "dF_lo", "dF_hi", "dF_minus_inf", "cr_minus_inf", "copy", "prompt_contrast"):
                if stat in n:
                    vn[stat.lower()] = clean(n[stat])
            vn["retention"] = clean(n["retention_cond"]); vn["retention_sae"] = clean(n["retention_sae"])
            vn["cr_rate_tot"] = clean(t.get("cr_rate"))
            for stat in ("cr", "cr_lo", "cr_hi"):
                vn[stat + "_tot" if stat == "cr" else "cr_tot_" + stat.split("_")[1]] = clean(t[stat])
    out["prompts"] = {v.lower(): {c.lower(): {"d_pooled": clean(x["D_pooled"]), "lex_per100w": clean(x["lex_per100w"])} for c, x in cs.items()}
                      for v, cs in summary["prompts"].items()}
    out["validation"] = {k.lower(): v for k, v in summary.get("validation_flags", {}).items()}
    out["meta"] = {"n_responses": summary["n_responses"], "b": summary["B"], "rng_seed": summary["rng_seed"], "min_sites": summary["min_sites"],
                   "fp_max": summary["fp_max"], "stimuli_version": stimuli_version, "n_seeds": nseeds, "spec": summary["spec"],
                   "retained_features": ",".join(summary["retained_features"]), "code_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]}
    return out


def design_keys():
    """Planned design counts for the paper text (EXPERIMENT_PLAN v1.x); update when the plan changes."""
    return {"n_lineages": 4, "n_open_lineages": 3, "n_seeds": 300, "n_varieties": 10, "n_registers": 2, "n_prior_measures": 7,
            "note": "planned design values (plan v1), not measured results"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--design-out", default=None); ap.add_argument("--stimuli-version", default="v1")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(emit(s, stimuli_version=a.stimuli_version), open(a.out, "w"), indent=1)
    if a.design_out:
        Path(a.design_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(design_keys(), open(a.design_out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
