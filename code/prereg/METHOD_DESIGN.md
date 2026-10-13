# P3 METHOD design: Calibrated Accommodation Pairs (CAP), working name

Status: v4 = D24 FINAL (19:05). (Qwen3-8B as the main repair model, per the coordinator's popular-models decision of ~15:35; v3 + astra red-team otherwise unchanged). Lead <agent>, 2026-10-05 15:45 HKT.
- Inputs: the coordinator's binding triage (15:20) and the astra red-team (notes/ASTRA_method_redteam.md, adopted where
  sound; §10).
- To be pre-declared as D24 after the coder's feasibility check. Final design to the coordinator by 20:00.
- Direction: the user, Oct 5 15:10: "we need have our methods to make model better not a position paper".
- Paper shape: analysis (§3–5) → method → results.

## 0. Motivation: the analysis, plus a measured failure profile

- **§4.** Post-training standardizes replies to every informal cue. After it, AAE convergence is ≈ 0.
- **§5.** The DPO update selects: prestige matching gains relative to AAE in 10/10 lineages. Prompted production of AAE
  features is retained in 2 of 3 families, which suggests the capacity remains.
- **The E5 one-sided recipe** (chosen = the SFT reply transformed at fixed per-sentence rates; rejected = the untransformed
  reply) raises held-out-feature convergence at 60k (pooled +.042 [.031, .058]).
- **It is not calibrated** (results/derived/e5_calibration.json; existing runs, v1 stimuli, F_train features, 4 runs):
  - **inserts dialect for Standard-English users:** 0.10–1.06 features per 100 words, against 0.00 for every control and
    plain run;
  - **over-produces Singlish:** 1.07–9.47× the user's density; 31–79% of requests get more than 2× the user's density;
  - **under-produces AAE:** 0.17–0.47× the user's density;
  - **barely conditions on the user:** Spearman ρ of reply rate with user rate is only .20–.44, and this figure is
    content-confounded.
- **Target behaviour, in CAT terms:** respond to the user's own cues, do not overshoot (over-accommodation reads as
  patronizing), and do not impose a variety on a user who did not use it.
- **Framing:** CAP improves bounded, user-controllable style responsiveness while preserving task performance. Density
  matching is an operational proxy, not a universal communicative ideal.

## 1. Method: CAP (data recipe only; trainer, dpo_norm, β, lr and epochs as released)

**Candidates.** All candidates are SFT-derived and content-anchored: the SFT reply r0 to the Standard-English twin, then
re-rendered. They are not on-policy samples, and DPO does not need them to be. Rendering uses F_train features only.

| Pair | Prompt | Chosen | Rejected |
|---|---|---|---|
| U (`dial_matched`) | variety v; user density d_p (F_train hits per 100 words) | MIRROR | r0 |
| O (`dial_over`, new) | as U | MIRROR | FULL (every sentence transformed), kept only if FULL ≥ 2× MIRROR's hits |
| A (`sae_cond`) | the SAE twin | r0 | FULL |

- **MIRROR:** r0 with the sentence subset whose achieved density is closest to d_p, with overshoot capped at ≤ 1.25×
  target. Zero targets and identical pairs are rejected, and the target–achieved error is logged.
- **Claims (wording fixed by the red-team).** The pairs express preferences against *sampled* under-accommodation (U),
  over-accommodation (O) and imposition (A). We do NOT claim a monotone or peaked reward.
- **Mix.** U : O : A = 2 : 1 : 1, over AAE and SGE.
- **Dose, FIXED at 33% of DPO pairs:**
  - 1B: the 120k base subset + 60k CAP, step-matched to the existing C60k and T60k runs;
  - Qwen3-8B: 40k general pairs + 20k CAP pairs (§2);
  - 20k at 1B is exploratory only.

## 2. Runs (6 GPUs: <host> 0–3 = Qwen3-8B, one arm per GPU; 4 and 6 = OLMo-2-1B)
Paper parts (coordinator): (1) behavioural breadth on popular models (Qwen2.5-7B, Qwen3-8B, Qwen3.5 pairs, DeepSeek-LLM-7B base→chat [new], openPangu descriptive) — base→release contrasts measure the AGGREGATE post-training change; (2) stage-resolved mechanism on open-recipe lineages (OLMo/Tülu SFT→DPO) — only these attribute a change to DPO; (3) REPAIR = CAP:
- **Main repair: Qwen3-8B** (official, released instruct, NON-thinking mode for data, training and eval), continued DPO. Fallback (pre-declared): Qwen2.5-7B-Instruct if the Qwen3 DPO pipeline does not train end to end by Oct 6 12:00.
  - Fixed budget in every trained arm: 60k pairs = the same 40k GENERAL pairs (seeded subset of the open OLMo-2-7B preference mix) + 20k ARM pairs on the same prompts (33% share, as at 1B).
  - Arms: (i) INSTRUCT (no training); (ii) CONTROL = density-matched non-accommodative budget control (CAP's dialect and SAE prompts; chosen r0 vs rejected r1, both standard); (iii) UNMATCHED = the E5 one-sided recipe (FULL ≻ r0); (iv) CAP (U:O:A = 2:1:1). Candidates are Qwen3-8B's own non-thinking replies to the SAE twins.
  - Trainer: LoRA r 64 (α 128, all linear projections), dpo_norm, β 5, lr 1e-5, 1 epoch, effective batch 128; same for every arm. lr may be tuned ONCE on CONTROL only if the smoke test diverges or does not move, then frozen (logged).
  - Seeds 0, 1, 2 for (ii)–(iv), if they fit; order s0 of all arms first.
- **OLMo-2-0425-1B** (E5 pipeline; stage-resolved repair with the existing comparators C60k ×4, T60k ×4, d0 ×3): CAP-60k s0 (= PILOT/GATE), CAP s1, s2; −O (U fills the O slots) s0 (s1 if time); mirrored-target SFT ×1 (SFT from -SFT on the same 60k sources, then DPO on 120k base + 60k control pairs).
- Dropped/not run (compute): OLMo-2-7B arm (replaced by Qwen3-8B), 70B lineages, Tülu-3-8B, further human-written-AAE expansion.

## 3. Evaluation sets

- **PRIMARY TEST F300.** 300 requests from the disjoint Dolly/OASST pool: in no training pair, not in validation, not in
  v1. Rendered with the v1 renderer (all retained features) in AAE, SGE, NGP and APP, plus the SAE twin. 3 sampling seeds.
  - Why: v1 has informed the project, so it is reported as a previously used benchmark (continuity with §4–5).
  - If the v1 renderer cannot run on pool seeds by tonight, v1 becomes the primary set and this is stated.
- **VALIDATION V300.** Disjoint, F_train renderings with SAE twins. Used for the pilot gate only.
- **DENSITY INTERVENTION D.** On F300 requests in AAE and SGE, each request is rendered at two prespecified F_train target
  densities, LOW and HIGH (the 25th and 75th percentiles of the CAP prompt-density distribution).
  - Same renderer policy; matched decoding seeds.
  - A request is excluded if its achieved low–high contrast is below 0.5 hits per 100 words.
- **OVERRIDES O.**
  - 100 F300 dialect prompts, each with an appended instruction: "Please answer in Standard English." / "Please write
    formally." / "Please don't imitate my wording."
  - 100 F300 SAE prompts, each with: "Please reply in African American English." / "Please reply in Singlish."

## 4. Baselines

| # | Baseline | Applied to |
|---|---|---|
| B1 | step-matched control DPO (1B: C60k ×4, d0 ×3; Qwen3-8B: CONTROL ×3) and the unchanged INSTRUCT model | 1B, Qwen3-8B |
| B2 | released checkpoints as reference rows (OLMo-2-1B-DPO/Instruct) | 1B |
| B3 | system prompt, verbatim: "Reply in the same variety and register the user writes in, matching how much they use it; never exaggerate it." | plain checkpoints |
| B4 | few-shot: 3 fixed demonstration turns from the training pool (AAE, SGE and SAE users) | plain checkpoints |
| B34 | B3 + B4 combined (the strongest prompting baseline) | 1B plain, Qwen3-8B INSTRUCT |
| B7 | self-rewrite: the plain model rewrites its own draft given the user's message and the B3 instruction (two passes) | F300 |
| B8 | mirrored-target SFT (§2) | 1B |

B6 (the rule rewrite) is dropped: it adds nothing that −O and B7 do not cover, and it does not "score 0 by construction".

## 5. Metrics

Units are those of the paper. Every per-run value is reported.

| Metric | Definition |
|---|---|
| crheld (PRIMARY) | CR on the held-out features of AAE and SGE, F300 |
| crg | CR on the held-out varieties NGP and APP |
| calibration | reply/user density ratio (ratio of means) per variety; \|ratio − 1\|; overshoot tail = share of requests with reply > 2× user density; all by user-density tertile |
| insertion | F_train + F_heldout features per 100 words in replies to SAE twins |
| tracking β | from y_ij = α_i + β x_ij (achieved user density x, reply density y, request fixed effects, request-clustered bootstrap), on set D, per training seed |
| override compliance | variety-feature rate under each override; feature rate under the explicit requests |
| task fidelity (noisy diagnostic) | open-weight judges (gpt-oss-20b, Qwen3-8B) rate adequacy (answers the request) and naturalness (standard / natural / overdone) on 200 F300 replies per arm, with judge disagreement reported |

- **Human check.** A blind human check of 40 replies, ONLY if the user agrees. It is described as a limited diagnostic by
  one author with doctoral-level training; no community validation is claimed.
- **Capability:** IFEval (the E5 metric) and GSM8K (0-shot CoT, greedy).
- **Cost:** GPU-hours per run, and inference passes for B7.

**Leakage and artefact audit** (CPU, before training; reported):
- held-out-feature hits in chosen vs rejected for every pair type (must not differ);
- length, sentence count, punctuation and formatting by pair type;
- SFT log-prob per token of MIRROR vs FULL, to check fluency. If FULL is much less fluent, O may teach fluency rather than
  density, and we say so;
- target–achieved density error;
- source-request disjointness of train, V300, F300 and v1.

## 6. Pre-declared estimands and success criteria (every outcome reported)

- **Main contrast:** Δ = mean over CAP runs − mean over control runs (1B: C60k; Qwen3-8B: CONTROL); INSTRUCT is reported alongside.
- **CI:** a two-level bootstrap (training runs within arm, then requests, paired across arms), B = 2000, seed 20261012.
- **Seed counts are fixed in §2.** If a run is lost, that is reported, and the estimand uses the runs that exist.

| | Criterion |
|---|---|
| M1 (primary, per lineage) | crheld Δ: CI lower bound > 0 AND Δ ≥ +.02. MIN_GAIN = half the pooled E5 60k effect, set from the existing scale, not from the pilot. |
| M2 (calibration) | for each variety, \|ratio − 1\|_CAP < \|ratio − 1\|_T60k (seed-aware CI of the difference < 0); overshoot tail CAP < T60k; insertion CAP − control: paired CI upper bound ≤ +0.05 per 100 words (half the smallest T60k run) |
| M3 (conditions on the user) | β_CAP CI > 0 and β_CAP − β_B5 CI > 0 (1B: T60k; Qwen3-8B: UNMATCHED) |
| M4 (non-inferiority) | IFEval Δ lower CI bound > −2 points (two-level, paired over prompts); otherwise "non-inferiority not established". GSM8K is secondary, with the same margin. |
| M5 (control) | under "answer in Standard English", CAP's feature rate ≤ control's + 0.05 per 100 words; explicit requests are reported |

- **Secondary:** CAP vs B34, B7 and B8 on crheld at their calibration and inference cost. These are trade-offs, not
  superiority claims. Also crg, judge-rated overdone and adequacy, and the split S.
- **Publishable fallbacks, stated in advance:**
  - O reduces overshoot without raising crheld;
  - CAP matches prompting with no inference overhead;
  - the 1B effect fails on Qwen3-8B, with a diagnosis;
  - if SFT or prompting matches CAP everywhere, the CAP-specific claim is dropped.

## 7. Gate (triage)

- **Executable pipeline tonight:** the CAP and −O sets, V300, F300, D and O, and the audit.
- **Pilot:** 1B CAP-60k s0 by Oct 6 12:00, scored on V300 against C60k s0 (or d0 s0 if C60k's checkpoint is gone).
- **Pass conditions:**
  1. it runs end to end;
  2. CAP − control on the trained-feature CR has a request-bootstrap CI lower bound > 0;
  3. the reply/user ratio is in [0.5, 1.5] for both varieties;
  4. insertion on the SAE twins has a CI upper bound ≤ 0.05 per 100 words.
- **On a failure:**
  - (1) or (2) fails → stop scaling and tell the coordinator;
  - (3) or (4) fails → tell the coordinator with the numbers before scaling.
- **The gate outcome is reported in the paper.**

## 8. Timeline

| Lanes | When | Work |
|---|---|---|
| GPUs 4, 6 | Oct 5 evening | baseline evaluations on existing 1B checkpoints while the pairs build |
| GPUs 4, 6 | ~21:00 – Oct 6 04:00 | 1B CAP s0 (PILOT) + CAP s1 |
| GPUs 4, 6 | Oct 6 04:00 – 11:00 | −O s0 + CAP s2 |
| GPUs 4, 6 | Oct 6 11:00 – 19:00 | mirrored SFT + −O s1 |
| GPUs 4, 6 | afterwards | Qwen3-8B evaluations and judges |
| GPUs 0–3 | Oct 5 16:00 – 21:00 | Qwen3-8B non-thinking candidates on the pool (2 samples), pairs, LoRA-DPO smoke test, INSTRUCT + B3/B4/B34/B7 evaluations |
| GPUs 0–3 | Oct 5 ~22:00 → | wave 1 = CONTROL/UNMATCHED/CAP s0 (+ 1 spare GPU: evaluations); waves 2–3 = s1, s2 (≈ 5–8 h per LoRA run, to be measured) |
| GPUs 0–3 | done by | about Oct 7 morning; evaluations of 9 checkpoints after |
| all | Oct 8 18:00 | freeze |

Budget ≈ 200 of ~390 GPU-hours (to be updated with the measured smoke-test time).

## 9. Writing (the user's instruction of Oct 5)

Until results land, the method is a short design subsection. It enters the main body only when its experiments are
complete; partial results go to the appendix.

## 10. Astra red-team

**Adopted:**
- no reward-shape or on-policy claims;
- fixed 33% dose, with 20k exploratory;
- the seed-aware estimand, MIN_GAIN, and a non-inferiority M4;
- M2 as distance, tail and insertion (not "≤ 1");
- the two-level density intervention with a within-request regression;
- combined prompting, self-rewrite, mirrored SFT, overrides, and a fidelity diagnostic;
- the leakage and artefact audit; F300 as a fresh test slice;
- a seeded main model with replicated arms (Qwen3-8B ×3 per trained arm; astra had suggested 7B 2 + 2); an ablation that keeps the count of mirrored chosen fixed;
- the "user-controllable responsiveness" framing.

**Not adopted:**
- full SteerLM or CAA reproductions: discussed, not run, as astra also suggests;
- community validation: impossible without new human data, which the triage excludes; stated as a limitation.
