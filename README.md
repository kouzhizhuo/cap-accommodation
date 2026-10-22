# Answering in the User’s Variety: Calibrated Accommodation Pairs for Post-Training

Code and data for an anonymous submission under review.

Chat models answer users of African American English, Singlish and Nigerian Pidgin in Standard English. We trace open model lineages through every released post-training stage (SFT, DPO, RLVR), measure how far replies converge to the user's variety with a copy-controlled convergence ratio, and read what each update prefers from content-matched replies (the matching bonus). Post-training standardizes replies for every user, yet the DPO update still favours matching prestige spelling over stigmatized varieties. We then introduce calibrated accommodation pairs (CAP), preference data that reward replies at the user's own feature density and keep standard replies for Standard-English users.

## Repository map
- `code/harness/`: generation and scoring (vLLM), DPO re-runs (`e5/`), IFEval/GSM8K scoring, deployed-chatbot querying (`run_api.py`, needs `OPENAI_BASE_URL`/`OPENAI_API_KEY`).
- `code/detectors/`, `code/measures/`: variety-feature detectors, convergence ratio, matching bonus, prior measures.
- `code/analysis/`: statistics and derived results (`lead_derived.py`, `cap_tests.py`, `cap_m4.py`, `lead_mb_adjusted.py`, ...).
- `code/paper_tools/`: `json2tex.py` (result keys), `make_tables.py`, `style_tables.py`, `check_keys.py`, `regen.sh`.
- `code/figures/`: figure scripts; `code/E4/`: preference-data audit; `code/gold/`: human-audit sheet builders and protocol; `code/prereg/`: pre-registration and the CAP pre-declaration.
- `data/stimuli/`: the variety-twin requests and their renderings (ours) and the rendering rules; WildChat items as identifiers and source hashes only.
- `results/`: the small JSON/CSV files every number in the paper comes from; `results/human_audit/`: blind audit items and consensus labels from a panel of five or more annotators with PhD-level training in related fields, agreed through discussion (no annotator identity; no inter-annotator agreement, since labels are consensus).

## Install
Python 3.9+ for analysis and tables (`pip install -r requirements.txt`); a separate Python 3.12 environment for IFEval scoring; vLLM on NVIDIA GPUs for generation. See `ENV.md`.

## Reproduce the tables and figures
Every number is a key in the result files: `python code/analysis/lead_derived.py`, then `code/paper_tools/regen.sh` (json2tex + make_tables + style_tables).

| Paper item | Script | Input result files |
|---|---|---|
| fig:teaser (Fig. 1) | drawn (figures/specs/P3_fig1.md, fig1_teaser.figma.*.js) | none |
| fig:method (Fig. 2) | drawn (figures/specs/P3_fig_method.md) | none |
| fig:heatmap (Fig. 3) | code/figures/hero_standardisation.py | exp/results/E1/summary.json, breadth/newgen summaries |
| fig:models (Fig. new) | code/figures/fig5_models_dumbbell.py | exp/results/derived/fig_models.json (written by code/paper_tools/make_tables.py; same values as tab:main) |
| fig:heatmap (main, panel a) | code/figures/hero_standardisation.py (HERO_PANEL_A_ONLY=1) | exp/results/E1/summary.json |
| fig:heatmap_full (appendix) | code/figures/hero_standardisation.py | E1/summary.json, E4/interaction.json, derived/e4_adjusted.json |
| tab:contrast | code/paper_tools/make_tables.py | E4/interaction.json, derived/e4_adjusted.json |
| fig:models (main, compact) | code/figures/fig4_models_change.py | derived/fig_models.json |
| fig:models_full (appendix) | code/figures/fig5_models_dumbbell.py | derived/fig_models.json |
| fig:e5 (dose) | code/figures/fig4_e5_dose.py | exp/results/E5/summary.json |
| fig:cap | drawn (figures/specs) | none |
| tab:main | code/paper_tools/make_tables.py | E1/summary.json, breadth/summary.json, newgen/summary.json, E1/newgen_q38, derived/openpangu_*, E4/mbv*.json, E4/interaction.json, derived/e4_adjusted.json |
| tab:chatonly | code/paper_tools/make_tables.py (split from tab:main, 2026-10-09) | as tab:main (newgen_q38, derived/openpangu_*) |
| tab:update | code/paper_tools/make_tables.py | E4/mbv*.json, derived/e4_adjusted.json (exp/analysis/lead_mb_adjusted.py) |
| tab:cap | code/paper_tools/make_tables.py | derived/cap_table.json (exp/analysis/cap_tests.py, cap_m4.py; lead_derived.py) |
| tab:capabl | code/paper_tools/make_tables.py | derived/cap_abl.json, METHOD/cap_tests_1b_*.json, METHOD/cap_m4.json |
| tab:capsetup | hand-written | none |
| tab:sensitivity | hand-written with keys | derived/lead_derived.json, METHOD/sens_*.json |
| tab:comp | hand-written with keys | E1/comp.json, audit/audit_v4.json (dv.comp_corr) |
| tab:audit | hand-written with keys | audit/audit_v4.json (exp/analysis/score_audit_v4.py) |
| tab:mainfull, tab:sizes, tab:revisions, tab:authentic | code/paper_tools/make_tables.py | E1/summary.json, E4/*, design/model_revisions.json, E1/authentic.json |
| tab:controls, tab:measures, tab:detectors, tab:mechanism, tab:mitigation, tab:operationalization, tab:outcomes, tab:licences | hand-written with keys | E1/E2/E4/E5 summaries, derived/*.json |

## Data sources and licences
- Our requests, renderings, detectors and audit labels: CC BY 4.0.
- WildChat-1M (ODC-BY): identifiers and hashes only, no text; rebuild with the scripts in `data/stimuli/`.
- Human-written AAE tweets with translations and EnDive: not redistributed (licences do not state redistribution); scripts rebuild the twins from the original sources.
- Models: loaded from their official Hugging Face repositories under their own licences; no weights or checkpoints are included.
- Raw generations (several GB) are not included; they are regenerated by `code/harness/gen.py` from the requests.

## Citation
Anonymous submission under review.
