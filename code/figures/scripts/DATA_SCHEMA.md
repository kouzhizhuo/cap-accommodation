# Data schema read by figures/scripts/*.py (all under `exp/results/`)

Nothing is typed into the scripts. A missing file makes the script exit with "no results yet". Files named `PILOT_*` or under `PILOT_P0/`
mark pilot data; the outputs then carry `_PILOT` in the file name. `NaN` is allowed (undefined ratio, e.g. zero prompt contrast).

## E1 (accommodation index) — `E1/summary.json`  (pilot: `PILOT_P0/summary_L{1,2}.json`, already in this shape)
```
{"models": {"<lineage>@<stage>": {"varieties": {"<VAR>": {"<scoring: novel|raw>": {
      "CR_morph": x, "CR_morph_ci": [lo, hi], "CR_lex": x, "CR_lex_ci": [lo, hi],
      "dF": x, "dF_ci": [lo, hi], "dF_minus_dF_INF": x, "n_seeds": n }}}}}}
```
stage ∈ base | sft | dpo | rlvr1 | instruct. Optional per-variety `"tier": "P|M|G"`.
Requested additions (for Fig 3b when they exist): per variety and stage `LSM`, `blevins_style` (+ `_ci`) on the SAME replies.

## E4 (mechanism) — `E4/summary.json`
```
{"models": {"<lineage>@<stage>": {"varieties": {"<VAR>": {"MB": x, "MB_ci": [lo, hi], "RB": x (optional), "tier": "P|M|G" (optional)}}}}}
```
MB_v = implicit-reward matching bonus (EXPERIMENT_PLAN §3.9), full responses, teacher-forced. The scatter joins MB_v with CR_v of `E1/summary.json` for the same model-stage.

## E5 (causal DPO re-run) — `E5/summary.json`
```
{"outcome": "CR_stig" | "S", "dose_unit": "pairs removed (%)" | "pairs added (k)",
 "runs": [{"arm": "treatment|control|anti", "dose": d, "seed": s, "value": y, "ci": [lo, hi] (optional; single-seed runs)}],
 "fit": {"arm": "treatment", "slope": b, "slope_ci": [lo, hi], "x": [..], "y": [..], "y_lo": [..], "y_hi": [..]} (optional)}
```
A dose with ≥ 3 seeds is drawn filled (mean ± std over seeds); < 3 seeds hollow, single run, no std (coordinator ruling Sep 29).
