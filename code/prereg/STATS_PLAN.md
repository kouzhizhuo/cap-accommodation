# P3 STATS_PLAN — estimands, models, intervals, seeds, reporting conventions

Owner: P3 analyst/writer `<agent>` (Opus). v1, 2026-09-29.
- Measure definitions follow EXPERIMENT_PLAN §3 (lead `9e4f338d`). The notation is identical: m = model-stage,
  v/c = condition, i = seed prompt, s = sampling seed.
- This file fixes *how every number is estimated and reported*. It feeds `notes/PREREG.md`: §2 (primary
  estimands) and §6 (exclusions) are copied there verbatim before Sep 30 22:00 HKT.
- Nothing here is a result.

---

## 1. Design and units of inference

Sources of variation, in order of importance for generalisation:
1. **Seed prompts i** (n = 300 content-matched seeds, 6 task types × 50; Arm N: ≥ 100 per variety). Each seed
   is rendered in all 12 conditions. The **prompt is the unit of resampling**: claims are about "prompts like
   these", so every CI in the paper is a CI over prompts.
2. **Sampling seeds s ∈ {0, 1, 2}, with n = 3 samples per seed** (EXPERIMENT_PLAN E1).
   - This gives 9 responses per (m, c, i).
   - Decoding: T = 0.7, top-p = 0.95, max 400 new tokens. A greedy subset of 100 seeds is a robustness row.
   - The vLLM request seed is derived from (i, c, s) and is stable across stages, so a rerun reproduces the
     same responses.
   - The ± std convention (§4.2) is over the 3 sampling seeds; the 3 samples within a seed enter its mean.
3. **Model-stages m** within a lineage ℓ (base → SFT → DPO → RLVR (+ branches)). Stages are fixed effects:
   they are the object of the claim, not a sample.
4. **Varieties v** (10 + 2 register controls). These are fixed effects in per-variety claims. In the
   prejudice-link correlation (C6) varieties are the *observations* (n = 10), which bounds that claim (§5.6).
5. **Training seeds t** (only for our own DPO re-runs, E5). **Ruling of the coordinator, 2026-09-29
   (`DECISIONS.md` D4):**
   - **3 seeds** (t ∈ {0, 1, 2}) at the doses {0, mid, max}: Regime B 0 / 5k / 20k, Regime A 0 / 50% / 100%;
   - **1 seed** at the remaining dose;
   - matched controls: 2 seeds at mid and max; anti-accommodation and flip arms: 1 seed each
     (lead, D5; ≈ 17 runs);
   - if compute allows after the Oct 3 gate, the intermediate doses are topped up to 3 seeds.
   Reporting follows from this:
   - ± std over 3 seeds wherever n = 3;
   - rows and points with < 3 seeds carry a **bootstrap CI over prompts**, and the caption says so. In plots
     they are hollow markers with no seed error bar;
   - the dose–response claim rests on the fit across **all** doses (§3.4), not on any single-seed point.

Pairing: the design is fully crossed and paired (the same i in every c, m, s). Every contrast is computed
**within prompt** first and then averaged. This removes content variance, which is the point of the
content-matched design.

---

## 2. Primary estimands (copied to PREREG)

All densities are computed on **novel spans** (EXPERIMENT_PLAN §3). Responses excluded under §6 are removed
before any mean is taken.

### E-1. Convergence ratio (primary quantity)
- **Per prompt:** AI_{i,v}(m) = D̄_v(r(m, v, i, ·)) − D̄_v(r(m, SAE, i, ·)), where D̄ is the mean over sampling
  seeds, in percentage points.
- **Variety-level user gap:** Δp_v = mean_i [D_v(p_{v,i}) − D_v(p_{SAE,i})].
  - This is a property of the fixed stimuli.
  - It is held **constant** as the denominator, so a per-prompt ratio never divides by a tiny per-prompt gap.
- **CR_v(m) = mean_i AI_{i,v}(m) / Δp_v.**
  - Unit: fraction of the user's density that the reply mirrors; reported ×100 as "% mirrored", 1 decimal.
  - 1 = full matching, 0 = none, < 0 = fewer v-features to a v-speaker than to an SAE speaker.
  - Also CR_v^morph and CR_v^lex (feature subsets), with their own Δp.
- **Register controls (INF, FOR):** the "density" is the formality score, not feature density.
  - CR_INF = −ΔF_INF / (F(p_INF) − F(p_SAE)), i.e. the fraction of the user's informality mirrored.
  - Sign convention: positive = the reply moves toward the user.

### E-2. Split index and prestige ordering
- **S(m) = mean_{v ∈ P} CR_v(m) − mean_{v ∈ G} CR_v(m).**
  - P = prestige cues: INF, BRE, AUS (and FOR in a robustness row).
  - G = stigmatized varieties per the PREREG tiers.
  - Mid-tier varieties (IRE, and IND if it is frozen as mid) are excluded from S and appear only in ρ.
- **ρ_prestige(m) = Spearman(CR_v(m), tier_v)** over the 10 varieties (tiers are ordinal; ties allowed).

### E-3. Stage deltas (C1, locus)
- **Δ_{a→b}S = S(b) − S(a)** for consecutive stages a → b of one lineage.
- The **locus contrast** is **L(ℓ) = Δ_{SFT→DPO}S − Δ_{base→SFT}S** (a difference-in-differences within
  lineage).
- **C1 holds on lineage ℓ** iff all three conditions hold:
  1. L(ℓ) > 0 with its 95% CI excluding 0;
  2. Δ_{SFT→DPO}S's CI excludes 0;
  3. Δ_{DPO→RLVR}S ≥ −½·Δ_{SFT→DPO}S ("RLVR does not reverse it"; the point estimate is compared, and the CI is
     reported).
- **The alternatives are coded exactly as in EXPERIMENT_PLAN §1:**
  - α: S(SFT) − S(base) ≥ 0.8·[S(final) − S(base)];
  - β: none of the three consecutive Δs exceeds either of the others by more than the CI half-width of the
    difference;
  - γ: \|S(m)\| < its CI half-width at every stage while the CR of every condition falls.

### E-4. Register divergence (C2)
- **ΔF_v(m) = mean_i [F̄(r(m, v, i)) − F̄(r(m, SAE, i))].**
- The divergence contrast is **ΔF_v − ΔF_INF**: positive means the model is more formal to a v-speaker than
  to an equally informal SAE speaker. It is tested by a paired sign-flip permutation over prompts (§5.2).

### E-5. Competence and the "can but won't" gap (C3)
- **COMP_v(m) = mean_f min(1, ρ_f(r | request) / ρ_f(authentic))**, over retained features f.
- **G_v(m) = COMP_v(m) − CR_v(m).**
- **CAR_v** is the unclipped ratio.
- The C3 criterion is evaluated per variety × stage: the share of features with ratio ≥ 0.5, which must be
  ≥ 70%.

### E-6. Matching bonus (C4)
- MB_v(m) and RB_v are as in EXPERIMENT_PLAN §3.9, computed per prompt and averaged over prompts.
- **Length-normalised** implicit reward, (1/|y|)·log π_m/π_SFT, is **primary** (lead D8, 2026-09-30): the released
  OLMo-2 DPO stages were trained with open-instruct's length-normalised `dpo_norm` loss. The unnormalised version
  is a robustness row.

### E-7. Dose–response (C5)
- **CR_G(dose, t)** = mean CR over stigmatized varieties for the DPO re-run at dose d and training seed t.
- The monotonicity statistic is Spearman(dose, CR_G) over all (d, t) runs, together with a random-intercept
  slope (§3.4).

### E-8. Prejudice link (C6)
- **ρ_C6(m) = Spearman over varieties of (CR_v(m), stereo_v(m))**, where stereo_v is the Hofmann-protocol
  covert association score (EXPERIMENT_PLAN §4.6).
- The same with the quality gap Q_v(m).
- **Dissociation statistic:** across the stages of a lineage, Δ_stage CR_G vs. Δ_stage stereo_G. It is
  reported as two parallel lines with CIs, not as a correlation (n = 4 stages).

### E-9. Instrument validity (C7)
Per measure B1–B11 (`PRIOR_WORK_measures.md` §2), criteria as in EXPERIMENT_PLAN E3:
- (i) Pearson r with synthetic gold density (1,000 transformed responses with known feature counts);
- (ii) Spearman with the LLM reference ratings (300 responses, §8), and with the human audit ratings on the audited subset (reported separately);
- (iii) copy sensitivity = the change in the score when a copied ≥ 4-gram span is spliced in (smaller is
  better);
- (iv) length sensitivity = \|partial Spearman with response length\| (smaller is better);
- (v) stage sensitivity = Cohen's d between SFT and DPO on stigmatized varieties (larger is better;
  direction-agnostic);
- (vi) seed reliability = ICC(2,1) across the 3 sampling seeds.

---

## 3. Mixed-effects models (the tests behind the tables)

The bootstrap (§4) gives the reported intervals of the estimands. The mixed models give the **tests of the
interactions** and a check that the conclusions do not depend on the aggregation.
- Fits are **per lineage**. Headlines are stated as "on k of 3 lineages" (EXPERIMENT_PLAN §6).
- A pooled fit with lineage as a fixed effect appears only as an appendix robustness row.

### 3.1 Primary model: variety × stage (C1, C2)
- **Unit:** the per-prompt convergence contribution y_{i,v,m} = AI_{i,v}(m) / Δp_v, whose mean over i is
  CR_v(m).
- **Rows:** one per (i, v, m). Rows for all 10 varieties plus INF and BRE/AUS.
- Model (lme4 notation; statsmodels `MixedLM` or `pymer4`/lme4 via R, whichever the analysis env has, with the
  same formula):

```
y ~ tier * stage + (1 + stage | seed) + (1 | variety)
```

- `tier` has 3 levels: prestige cue (reference), mid, stigmatized.
- `stage` is a factor with treatment coding: base (reference), SFT, DPO, RLVR.
- `(1 + stage | seed)` lets each prompt have its own stage profile. If it fails to converge, drop to
  `(1 | seed) + (1 | seed:stage)` and report the change.
- `(1 | variety)` absorbs variety differences within a tier.

**Tests:**
- **H1/C1 locus:** the contrast (tier_stig:stage_DPO − tier_stig:stage_SFT) − (tier_stig:stage_SFT − 0),
  i.e. the stigmatized-minus-prestige gap grows more at SFT→DPO than at base→SFT. This is a linear contrast
  on the fixed effects with a Wald test (Satterthwaite df if lme4). It is the model-based counterpart of
  L(ℓ).
- **H2/C2 selectivity:** tier_stig:stage_DPO < 0 and tier_stig:stage_RLVR < 0 (one-sided, Holm over the two
  aligned stages).

### 3.2 Response-level GLMM (robustness; appendix)
Count data directly, at the level of the individual response:

```
cbind(real, opp) ~ is_v * stage + (1 + is_v | seed) + (1 | response_id)    # binomial, logit link
```

- The model is fitted per variety.
- `is_v` = the response is to the v-condition (vs. SAE) of the same seed.
- `(1 | response_id)` is an observation-level random effect for overdispersion.
- The is_v:stage coefficients must agree in sign with CR_v(m) − CR_v(base). Disagreements are listed in the
  appendix.

### 3.3 Register model (C2)
```
F ~ condition * stage + (1 + condition | seed)
```
- `condition` ∈ {SAE, INF, v}.
- The contrast of interest is (v − SAE) − (INF − SAE) at each aligned stage.

### 3.4 Dose model (C5)
```
CR_G ~ dose + (1 | train_seed)
```
- dose is numeric on [0, 1].
- The fit uses **all** runs: 3 seeds at {0, mid, max} and 1 seed at the intermediate doses (D4).
- The primary interval for the slope is a **bootstrap CI**: resample prompts (stratified, B = 2000) and, within
  each dose with n = 3, the training seeds. The profile CI is reported alongside, together with
  Spearman(dose, CR_G) over all runs.
- The C5 dose–response claim is evaluated on this slope. Single-seed points never carry it alone.
- Helpfulness (IFEval, AlpacaEval-lite) uses the same model; the C5 falsifier is a drop of more than 2 pp at
  the effective dose.

### 3.5 Multi-turn (E7, optional)
```
y ~ turn * tier + (1 + turn | seed)
```
- turn ∈ {1, 2, 3} is numeric.
- CAT predicts that convergence rises over turns for prestige cues. Mitiaeva & Xiao 2026 report that it
  declines in human–ChatGPT dialogue.

Model hygiene:
- Report singular fits.
- No model selection on the fixed effects: the formula above is pre-registered.
- Random-effect simplification only for non-convergence, in the order stated.

---

## 4. Intervals, seeds and the ± conventions

### 4.1 Bootstrap (all reported CIs)
- **Cluster bootstrap over seed prompts**, **stratified by task type** (6 strata × 50; EXPERIMENT_PLAN §6).
  B = 2000, percentile 95% CIs, numpy RNG seed `20261012`.
- Resample prompt ids with replacement, keeping every condition, stage and sample of a drawn prompt. This
  keeps the pairing and the cross-stage dependence intact.
- Δp_v is recomputed inside each replicate (it is also a mean over prompts).
- Estimands that are functions of several varieties or stages (S, L, ρ_prestige) are recomputed per
  replicate from the same resampled prompt set.
- Arm N (authentic, different prompts per variety) is resampled within variety.
- For C7 the unit is the rated response (§8), B = 2000.

### 4.2 Sampling-seed spread (the ± in tables)
- For each s ∈ {0, 1, 2}, compute the estimand using only responses with sampling seed s (all prompts).
- Report the **mean ± std (ddof = 1) over the 3 values**.
- This quantifies decoding noise and is what TEAM_RULES §1.4 asks for. It is **not** a CI and is never
  described as one.

### 4.3 Per-table conventions

| Table (paper) | Cell content | ± / interval | Marks |
|---|---|---|---|
| T1 main (`tab_main`): CR_v by variety (rows grouped by tier) × lineage@stage (grouped headers per lineage) | CR ×100, 1 decimal | `\pms{mean}{std}` over 3 sampling seeds | `\sig{*}`/`\sig{**}`/`\sig{***}` = 95/99/99.9% bootstrap CI excludes 0, Holm-adjusted within the column; `\best`/`\second` **not used** (analysis table; see STYLE_GUIDE §6.1.3). S and ρ_prestige as the last two rows |
| T2 stage deltas: Δ_{base→SFT}S, Δ_{SFT→DPO}S, Δ_{DPO→RLVR}S, L(ℓ) per lineage | S points ×100, 1 decimal | `\pms` over 3 sampling seeds; the 95% bootstrap CI in brackets for L only | `\up`/`\down` coloured by desirability (a *growing* split is undesirable → `\upbad`); `\sig` from the bootstrap CI |
| T3 register and competence: ΔF_v − ΔF_INF, COMP_v, G_v, CAR_v | ΔF in formality points (2 dec.); COMP/CR in % | `\pms` over 3 sampling seeds | `\sig` from the paired sign-flip permutation (ΔF) or the bootstrap CI (G) |
| T4 instrument validity (C7): measures B1–B11 × criteria (i)–(v) | Spearman ρ (2 dec.); sensitivities (2 dec.) | `\pms{est}{se}` with **± = bootstrap s.e. over rated items**, stated in the caption (no seeds apply) | `\best`/`\second` per column (a real comparison), direction arrows in headers |
| T5 mechanism: MB_v, RB_v per stage/RM | nats (2 dec.) | `\pms` over 3 seeds of the sampled y_std; 95% CI in the appendix | `\sig` from the bootstrap CI |
| T6 mitigation (C5): CR_G, CR_INF, S, IFEval, AlpacaEval-lite for SFT, released DPO, AA-DPO doses, baselines X1–X5 | as T1/T2; helpfulness in % | `\pms` over **3 training seeds** for our runs at doses {0, mid, max}; rows with < 3 seeds (intermediate dose, controls, anti/flip arms) show the 95% bootstrap CI over prompts instead, marked in the caption; over 3 sampling seeds for released checkpoints and inference-time baselines (the caption says which rows are which) | `\best`/`\second` per column among mitigations; `\up`/`\down` vs. the released DPO row; † for any copied number |
| T7 (or figure) prejudice link | Spearman ρ_C6 per stage (2 dec.) | 95% bootstrap CI over prompts `[lo, hi]` (n = 10 varieties stated) | none beyond the CI; verb *suggest* |
| Operationalisation table | text only | — | — |

Text convention: every number in prose carries "95% CI [lo, hi], bootstrap over prompts" at first use in a
subsection; later mentions in that subsection may give the point estimate only.

### 4.4 Precision
- CR, S and COMP are percentages with 1 decimal.
- ΔF, MB and ρ have 2 decimals.
- p-values have 2 significant digits, or "< 10⁻⁴" (the bootstrap resolution is 1/2000, so exact small
  p-values come from permutation tests only).
- Decimals are the same within a column (STYLE_GUIDE §5.2).

---

## 5. Hypothesis tests and multiplicity

1. **Is CR_v(m) ≠ 0?** Bootstrap CI (§4.1). The two-sided p comes from the bootstrap distribution (the
   percentile of 0).
   - Family = all (v, m) cells of one lineage (≤ 12 × 5 = 60).
   - Correction: **Holm** within the family. Stars in T1 use Holm-adjusted p.
2. **Selectivity contrasts (EXPERIMENT_PLAN P2).** Both use a paired sign-flip permutation over prompts
   (10,000 permutations, one-sided, Holm over the stigmatized varieties, per aligned model):
   - CR_v − CR_INF < 0 for each stigmatized v;
   - ΔF_v − ΔF_INF > 0.
   - C2 decision rule (verbatim from the plan): **≥ 6 of 8 stigmatized–INF contrasts < 0 with Holm-adjusted
     p < .05, and ρ_prestige ≥ 0.6.**
   - The falsifier (ρ < 0.3 in a majority of aligned models, or a uniform drop) is evaluated as well. The
     region 0.3 ≤ ρ < 0.6 is "not supported, not falsified" and is reported as such.
3. **Locus L(ℓ) > 0.** Bootstrap CI plus the §3.1 Wald contrast.
   - Family = the ≤ 4 lineages; Holm.
   - C1 needs ≥ 2 of 3 fully open lineages to pass *after* correction.
4. **Prestige ordering.** Spearman ρ_prestige with an exact permutation p over tier labels (10! is too many,
   so 100,000 random permutations; ties handled by average ranks), plus the bootstrap CI over prompts.
   - The falsifier (ρ < 0.3 in a majority of aligned models) is evaluated on the point estimate. The CI is
     reported.
5. **Competence criterion (C3).** A descriptive threshold rule (no test). A bootstrap CI for the share of
   features ≥ 0.5 is reported.
6. **Prejudice link (C6).** Spearman over n = 10 varieties.
   - CI by the prompt bootstrap (CR and stereo both recomputed per replicate), and a variety jackknife
     (leave one variety out) for robustness.
   - Report both. With n = 10, only \|ρ\| ≥ 0.65 reaches p < .05, so the verb is *suggest* in all cases.
7. **Dose–response (C5).** Spearman over the (dose, seed) runs and the §3.4 slope; the falsifier is from
   EXPERIMENT_PLAN C5.
8. **Per-feature and per-variety appendix tables.** Benjamini–Hochberg (q = 0.05), labelled as exploratory.

§4 "Statistics" of the paper states this once: "Intervals are 95% cluster-bootstrap CIs over prompts (B = 2000);
table ± are std over 3 sampling seeds (3 training seeds for our DPO runs); tests are Holm-corrected within each
hypothesis."

---

## 6. Exclusions, retention and sensitivity (copied to PREREG)

Exclusions are applied in this order, and a count is logged per (m, c):
1. **Refusals.** Judge-flagged refusals (Qwen3-8B; consistency with the 200 LLM reference labels reported, human agreement on the audit subset if audited) are excluded (Haq et al.
   2026 show that dialect cues change refusal rates).
2. **Non-comprehension.** Responses that do not address the request (the judge with the SAE prompt as
   reference) are excluded.
3. **Incoherent base outputs.** The coherence filter of EXPERIMENT_PLAN §7 applies (e.g. degenerate
   repetition, off-dialogue continuation).
4. **Too few opportunities.** Σ_f (real + opp) < 3 on the novel spans: the response is dropped for that
   variety's density.

Reporting and sensitivity:
- Retention per cell goes in the appendix. **Differential retention is itself reported**: refusal and
  comprehension rates by condition × stage are a result, and they go in T3 or the appendix.
- **Sensitivity (appendix):**
  - every headline estimand with (a) no exclusions, (b) refusals kept as D = 0, and (c) the leave-one-feature-out
    maximum change;
  - novel-span threshold n ∈ {3, 4, 6};
  - base models with and without the chat scaffold (F1/F2).
- A headline is **robust** if its sign and the "CI excludes 0" status survive every variant. Otherwise the
  text says which variant breaks it.

---

## 7. Precision planning (pilot → final n)

- From the pilot (≥ 50 prompts × 12 conditions × 1 lineage × 4 stages × 3 seeds), estimate SD_i of the
  per-prompt contribution y.
- The CI half-width of CR_v at n prompts is ≈ 1.96 · SD_i / √n. Report the projected half-width at n = 300 to
  the lead **before** the main run.
- The target is a half-width ≤ 5 points of CR (×100) for each variety, and ≤ 4 points for S.
- If the target is missed, raise the number of samples per cell (this reduces within-prompt noise) before
  adding prompts, since new prompts need a naturalness check.

---

## 8. Reference labels and the human audit (TEAM_RULES §1.7/§1.7a; details in ANNOTATION_PROTOCOL v2)

- **LLM reference labels** by two LLMs from different model families (Mistral-Small-3.2-24B, Phi-4), with
  disagreements adjudicated by a third (Claude):
  - Pools: 50 authentic texts per variety (feature presence); 300 responses for C7 (0–4 accommodation);
    200 for comprehension/refusal; 200 for the stereotyping dimensions.
  - Inter-LLM κ/α is reported before adjudication. Detector-vs-reference and judge-vs-reference agreement are
    reported as **consistency**, not validity.
- **Human audit** (the user, ≈ 60 items, ≤ 40 min), stratified random from the same pools with seed 20261001:
  - ≈ 40 feature-presence items, stratified by variety × source × detector outcome;
  - ≈ 20 accommodation ratings, if replies exist by Oct 1 noon.
- **Human-audit statistics:** human–detector P/R with Wilson 95% CIs (small n), human–reference κ and
  human–judge κ with bootstrap CIs.
  - Only these may be described as human agreement.
  - With n ≈ 40, the P/R CIs are wide (about ±15 pp). The paper states this; it does not pool them into a
    headline.
- No crowdworkers (ARR checklist D: the audit is by one author, unpaid). LLM labelling is disclosed in E1.

---

## 9. Pipeline and traceability

- The analysis code lives in `exp/analysis/` (coder), with one script per estimand family.
- Each writes `exp/results/<E>/summary.json` with flat keys that `tools/json2tex.py` turns into `\resdef` lines:
  - key pattern: `<exp>.<lineage>.<stage>.<variety>.<stat>` (lower case, no spaces);
  - stat ∈ {`cr`, `cr_std`, `cr_lo`, `cr_hi`, `p_holm`, …};
  - examples: `e1.olmo2_7b.dpo.aae.cr`, `e1.olmo2_7b.dpo.aae.cr_lo`, `e1.tulu3_8b.locus.l`,
    `e1.tulu3_8b.locus.l_lo`.
- Bootstrap replicate arrays are saved (`.npz`) next to each summary, so that any new contrast can be computed
  without refitting.
- Every summary records: git hash of the analysis code, the stimuli version (`exp/stimuli/v1`), the exclusion
  counts, B, and the RNG seed.

## 10. Open decisions (for the lead)
1. ~~Training seeds for E5~~: resolved by the coordinator (D4): 3 seeds at {0, mid, max}, 1 seed at the intermediate doses.
2. Samples per cell: 3 (this plan) or 5 (the proposal).
3. Whether mid-tier varieties enter S (currently excluded) once `PREREG.md` freezes the tiers. Varieties with UNVERIFIED tier sources at the freeze are excluded from ρ_prestige until they are verified (D5).
4. FOR in P (currently a robustness row only, because formal register is a prestige *register*, not a
   variety).
