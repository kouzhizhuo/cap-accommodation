FROZEN 2026-10-01 02:51 CST(HKT) by the P3 lead <agent> (sha256 of the content below this line: a3cb680dda970a2b29b2d552fcfefb71892239857dcb207c938b8f79d4fc3486)
# P3 PREREG — pre-registration (DRAFT; freezes Sep 30 2026, 22:00 HKT)

Status: **FROZEN** (writer `<agent>`, 2026-09-29 night). The lead signs off; it is frozen by committing it
with the line `FROZEN <timestamp>` at the top. After the freeze, any change goes in §10 with a date and a reason,
and the paper reports deviations (App. H).

Provenance:
- **Verbatim** passages (§3, §4, §5.2, §6, §7, §8.1–8.3) are extracted by script from `EXPERIMENT_PLAN.md`
  v1.1, so they cannot drift.
- The other sections are written for this file and reference `STATS_PLAN.md` (estimands) and `DECISIONS.md`
  (closest work).

**Label wording (TEAM_RULES §1.7, 2026-09-30).**
- Wherever the verbatim plan passages below say "author labels", "author-validated" or "hand-checked", read
  "LLM reference labels" and "consistency". The raters are gpt-oss-20b + a Claude-family rater, adjudicated by
  the Claude rater; the judges are Qwen, with no family overlap (ANNOTATION_PROTOCOL v2 §1).
- Human agreement is claimed only on the user's audit subset (ANNOTATION_PROTOCOL v2 §4).
- The passages are re-extracted once the lead updates EXPERIMENT_PLAN wording, before the freeze.

Nothing in this file is a result. At the time of writing, no main-run output exists and only the pilot P0 is
running.

---

## 1. Thesis (verbatim, EXPERIMENT_PLAN §0.2)
**Thesis:** *Chatbots can accommodate but are trained not to. Preference optimisation converts accommodation from a competence of the pretrained model into a selectively withheld choice: aligned models keep converging to prestige varieties and registers (Giles' upward convergence) while diverging from stigmatized varieties they can still produce on request. The default is installed at the preference stage, so removing it from the preference data removes it from the model.*

## 2. Varieties and external prestige tiers

### 2.1 Conditions and provisional tiers (verbatim, EXPERIMENT_PLAN §2.2)
| Code | Condition | Tier (provisional) | Build (§2.4) | Authentic source (status in `RESOURCES.md`) |
|---|---|---|---|---|
| SAE | Standard American English, neutral register | reference | seeds | — |
| FOR | Formal Standard English (register control) | prestige register | rule + open-LLM rewrite, formality-checked | — |
| INF | Informal Standard American English (contractions, casual lexis, lowercase; no dialect features), formality-matched to the vernaculars | **the reference for "divergence"** | rule + open-LLM rewrite, formality-matched | — |
| BRE | Standard British English (spelling, lexis, *whilst*, *have got*) | prestige | orthographic/lexical rules | — |
| AUS | Standard Australian English (spelling, lexis) | prestige/mid | orthographic/lexical rules | — |
| IRE | Irish English (eWAVE Irish morphosyntax; lexis) | mid | Multi-VALUE + lexicon | — |
| IND | Indian English (eWAVE Indian; lexis) | mid/stigmatized (freeze from sources) | Multi-VALUE + lexicon | EnDive (IndE), Demszky 2021 features |
| AAE | African American English (eWAVE Urban AAE + lexis) | stigmatized | Multi-VALUE + lexicon | **EnDive, ReDial** (parallel, human-verified), Groenwold pairs (Hofmann repo), TwitterAAE, CORAAL |
| SGE | Colloquial Singapore English (particles, copula deletion, *can/cannot*, *already*, *kena*; lexis) | stigmatized | Multi-VALUE + particle/lexicon rules | EnDive (CSE), Liang 2025 formal/informal Singlish, NUS SMS |
| NGP | Nigerian Pidgin (*dey/don/na/go/wan/sabi/abeg/wetin*) | stigmatized | Multi-VALUE (Nigerian) + rules | MAFAND-MT pcm–en, BBC Pidgin |
| SCO | Scots / Scottish English vernacular (*dinnae/cannae*, *aye*, *ken*, *wee*, -in') | stigmatized (overt) | Multi-VALUE (Scottish) + lexicon/orthography | SCOTS corpus (licence check); never Scots Wikipedia |
| APP | Appalachian English (a-prefixing, double modals, *fixin' to*, demonstrative *them*) | stigmatized | Multi-VALUE (Appalachian) + lexicon | authentic source to find; else synthetic-only and flagged |

### 2.2 Tier sources (page-level; TO FILL from the support's evidence notes before the freeze)
Rule: a variety's tier is fixed by *attitude studies of that variety or accent label*, never by our results. If
the sources disagree, the lower-prestige tier is used and both sources are listed. Tier codes used by S:
**P** (prestige cue), **M** (mid, excluded from S, included in ρ_prestige), **G** (stigmatized).

Core varieties (plan v1.4): BRE and AUS spelling/lexis detectors are in the core. The lexical-vs-morphosyntactic contrast of H2 is carried by BRE/AUS spelling and SGE particles, since the AAE lexical tier was silent in the pilot.

| Code | Frozen tier | Grade | Source(s), page (all read in the source by support; `PRESTIGE_EVIDENCE.md`) | Population / method | In ρ_prestige? |
|---|---|---|---|---|---|
| FOR | register control | — | — (formal Standard English; not a variety tier) | — | no (register control; robustness row in S) |
| INF | register control | — | — (informal Standard American English; the reference for divergence) | — | no (register control; in P for S) |
| BRE | **P (prestige)** | A+B | Coupland & Bishop 2007, *J. Sociolinguistics* 11(1):74–93, Table 2 p. 79 (Standard English 5.44, rank 2; Queen's English 5.59, rank 1) and abstract p. 74 ("standard" accents "strongly favoured in the prestige and attractiveness dimensions"); Bernaisch & Koch 2016, *World Englishes* 35(1):118–132, abstract p. 118 | UK respondents; Indian respondents | **yes** |
| IND | **M (mid)** | B | Bernaisch & Koch 2016, abstract p. 118: IndE rated positively, but Indian raters "view British English in a more favourable light" | Indian respondents | **yes** |
| SGE | **G (stigmatized)** | A (+B) | Cavallaro, Ng & Seilhamer 2014, *World Englishes* 33(3):378–397, manuscript p. 9: status means SSE 4.89 > SCE 3.67 > basilectal SCE 3.08 (Bonferroni-corrected), SSE also higher on solidarity; Cavallaro & Ng 2009, *World Englishes* 28(2):143–159, abstract p. 143 | Singapore, matched guise | **yes** |
| AAE | **G** | B+B | Purnell, Idsardi & Baugh 1999, *JLSP* 18(1):10–30, abstract p. 10 (housing discrimination by voice); Rickford & King 2016, *Language* 92(4):948–988, abstract p. 948 (AAVE testimony "dismissed as incomprehensible and not credible") | US | **yes** |
| APP | **G** | B | Luhman 1990, *Language in Society* 19(3):331–348, abstract p. 331: speakers "partially accept low status evaluations of their dialect" | Kentucky | **yes** |
| NGP | **G** | C (qualitative) | Obi 2014, *IJAH* 3(4):34–46, p. 41 ("low status accorded to the language as the language of the illiterate"), p. 43 ("NP is regarded as an inferior language"). Akande & Salami 2010 and Deuber 2005 were not accessible; Akande 2010 (*Dialectologia*) is a grammar paper and is **not** cited for prestige | Nigeria | **yes** |
| AUS | **M** | B | Coupland & Bishop 2007, Table 2 p. 79 / App. 1 pp. 88–90: "Australian" 3.51 (rank 11 of 34; UK raters, 1–7 prestige) | UK raters | **yes** (upgraded before the freeze) |
| IRE | **M** | B | Coupland & Bishop 2007, Table 2 p. 79: "Southern Irish" 3.63 (rank 10) | UK raters | **yes** (upgraded) |
| SCO | **M** (not stigmatized as a national accent) | B | Coupland & Bishop 2007, Table 2 p. 79: "Scottish" 3.98 (rank 5); non-Scottish raters also ≈ 3.9; only "Glasgow" is low (2.93, rank 29) → a Glasgow = G sensitivity row | UK raters | **yes** (upgraded) |

**Frozen ordering test (lead rule 2026-09-30; upgrade by support before the freeze, `PRESTIGE_EVIDENCE.md` §2):**
- ρ_prestige is computed over **10 varieties in 3 ordinal tiers**: 1 = BRE; 2 = SCO, IRE, AUS, IND; 3 = SGE, AAE,
  APP, NGP. Tier codes are ordinal across studies, never pooled rating values. SAE is the reference (North
  American 3.80, rank 8 in CB2007).
- Coupland & Bishop 2007 has no "Indian" label, only "Asian" (2.74, rank 33). That is **not** used for IND's
  tier; it is a sensitivity row (IND = 3) only.
- **Sensitivity orderings:** (i) SCO = 3 (the Glasgow reading); (ii) IND = 3; (iii) CB2007 numeric ratings for
  the varieties it covers.
- For S (lead, 2026-09-30): P = {BRE} + register cues {INF, FOR}; G = {AAE, SGE, NGP, APP}. Mid-tier varieties are excluded from S. The E5 held-out varieties are NGP (stigmatized), SCO and IND (mid). The tier-2 varieties are excluded from S and included in
  ρ_prestige.
- Any later change goes in §10 with its date and source.

**SGE tier rule (lead <agent>, 2026-09-30, from the pilot's retention; before the freeze):**
- SGE is carried by **CR_lex** (discourse particles and SGE lexis), because SGE morphosyntactic retention was
  4–16% in the pilot (below G0's 60%).
- CR_SGE^morph is reported descriptively only. It enters no test unless it passes G0 retention on the main run.
- The same rule applies to any variety whose morph tier fails G0 retention. Such a variety is carried by its
  lexical tier if that tier passes, otherwise it is dropped from the tests, with the reason logged in §10.

**BRE/AUS stimuli:** v1 must pass a US↔UK/AU opportunity check. At least one transformable item per seed is
required; the seeds that fail it are listed. (In v0, 72% of the BRE prompts were identical to SAE.)

**MB_v variant (lead D8, before the freeze):** the primary implicit reward is length-normalised, (1/|y|)·log π_m/π_SFT, matching the `dpo_norm` loss of the released OLMo-2 DPO stages (`E5_DPO_RECIPE.md`). The unnormalised version is a robustness row. E5 uses a fixed 120k-pair subset for the sweep; the full-mix dose-0 run must first reproduce the released DPO checkpoint (CR/S within CI) before the sweep is interpreted.

**Scale rule (lead, plan v1.5 §8.5a):** headline claims are asserted only where the ≥ 7B lineages agree in sign.
The 1B lineage serves the causal re-run (E5) and the dose axis. Claims that rest on 1B only are named
scale-limited.

Ordinal tier values for ρ_prestige: P = 3, M = 2, G = 1 (ties allowed; average ranks).

**Freeze rule for tiers (DECISIONS D5).**
- Only verified page-level sources count.
- A variety whose source is still UNVERIFIED at the freeze keeps a **provisional** tier and is flagged
  `PROVISIONAL` in the table above.
- It is **excluded from ρ_prestige** until the source is verified; the verification is logged in §10.
- The INF/BRE/AUS-vs-stigmatized contrasts of H2 and the split S use the P/G assignment of the standard vs.
  vernacular conditions. Those do not depend on the attitude sources.

## 3. Hypotheses and falsifiers (verbatim, EXPERIMENT_PLAN §1, v1.5a)
| # | Claim (paper wording; verb calibrated) | Evidence (experiment) | Falsified if |
|---|---|---|---|
| **C1 locus** (H1) | Along one lineage, dialect convergence for stigmatized varieties drops or flips sign at the preference stage (SFT→DPO), not at SFT (base→SFT), and RLVR does not reverse it. The SFT→DPO→RLVR contrasts are same-template and carry the locus claim; base→SFT is the only cross-template contrast and is reported with both scaffolds F1 and F2: Δ_{SFT→DPO}(S) > Δ_{base→SFT}(S) with non-overlapping CIs on ≥ 2 of 3 fully open lineages (*show*: stages of one lineage differ only in training). | E1 §3 on L1 OLMo-2-1B (base/SFT/DPO/RLVR1/Instruct + 26 branches), L2 Tülu-3-8B (base/SFT/DPO/RLVR + 61 GRPO branches), L3 OLMo-2-7B, L4 OLMo-3-7B-Instruct. | α: SFT already carries ≥ 80% of the final split on ≥ 2 lineages; β: the split grows by comparable amounts at every stage; γ: no split anywhere (uniform drop). Each alternative is reported as the finding. |
| **C2 selectivity** (H2) | At the aligned stages, convergence is retained for prestige cues and withdrawn for stigmatized morphosyntax at matched informality: CR_INF, CR_BRE, CR_AUS > 0 while CR_v ≤ 0 for stigmatized v, with ΔF_v > ΔF_INF (the model is *more* formal to an AAE speaker than to an equally informal Standard-English speaker); within stigmatized varieties CR_lex > CR_morph; CR is ordered by pre-registered prestige (*find*; the within-prompt contrasts are controlled). | E1 + E2 §4.1–4.2 (register, prestige, lexical/morphosyntactic contrasts; CAT categories). | Spearman ρ(CR_v, tier_v) < 0.3 in a majority of aligned models, or stigmatized and INF fall by the same amount (γ). |
| **C3 can but won't** (H3) | The same models realise stigmatized features on explicit request at ≥ 50% of authentic-corpus rates for ≥ 70% of retained features at every stage, so the withdrawal is a choice; the gap (COMP − CR) widens across stages (*show*). | E2 §4.3 competence + generic "match the user's style" instruction; Fleisig stereotyping measure on imitation prompts (open-weight judge, author-validated). | COMP < 50% for a majority of features in a majority of stigmatized varieties at the aligned stages (then "competence gap" is reported instead). |
| **C4 mechanism** (H4) | The preference stage rewards standardisation selectively: the DPO implicit reward (log π_DPO/π_SFT) and released reward models give a *matching bonus* MB_v > 0 for prestige cues and ≤ 0 for stigmatized varieties at identical content; in Regime A, chosen responses are less accommodating than rejected ones on dialect-bearing prompts (*find*; generalises Mire et al. beyond AAL and to the policy). | E4 §4.4 on L1–L4 (implicit reward), Tülu-3-8B-RM and 1–2 general RMs (**VERIFY** ids), preference-data counts. | MB_v is ≥ 0 for stigmatized varieties or shows no prestige ordering (CI overlap). |
| **C5 causal / constructive** (H5) | Re-running DPO from the released SFT checkpoint with the accommodation signal removed (Regime A) or added (Regime B) changes the split in the direction and roughly in proportion to the dose, at matched helpfulness; the fix beats inference-time baselines (system prompt, Logit-Constrained Generation) and matches or beats adapter/alignment baselines (TADA, DiaLLM) on CR without quality loss (*show*: trained intervention with dose). | E5 §4.5 on OLMo-2-1B with open-instruct; 4 doses × 2 seeds. | No monotone dose–response (Spearman over doses < 0.8 or CIs overlap between dose 0 and max), or IFEval/AlpacaEval-lite drop > 2 pp at the effective dose. |
| **C6 prejudice link** (H6) | Per-variety withdrawal tracks covert prejudice: CR_v (aligned) correlates negatively with the matched-guise stereotype score and with the condescension/quality gap across ≥ 10 varieties; **or** the two dissociate across stages (style standardised while covert associations stay flat, as Hofmann's alignment result suggests) — both pre-registered as informative (*suggest*). | E6 §4.6: Hofmann protocol generalised to 10 varieties on every stage; Fleisig-style quality gap with an open-weight judge validated on author labels. | Neither pattern: |ρ| < 0.3 *and* no stage dissociation (then C6 is dropped from the abstract). |
| **C7 instrument** (H7) | The opportunity-normalised, novel-span accommodation index is more valid than prior style-matching measures: highest agreement with synthetic gold and author ratings, least sensitivity to copying and length, highest stage sensitivity (*find*). | E3 §4.7, Table 4 (Blevins stylometric convergence, Durandard/DNM LSM & coordination, TwitterAAE P(AAE), style embeddings, raw Multi-VALUE density, judge). | AI is neither best nor second on ≥ 3 of 5 criteria. |
| **Anti-claims to rule out** | (a) *Copying*: novel-span scoring (≥ 4-gram overlap removed), copy rate reported (§3.7). (b) *Register normalisation*: INF is the reference for "divergence" (§4.1). (c) *Synthetic caricature*: density-calibrated transforms + authentic arm (EnDive, ReDial, Groenwold, NUS SMS, MAFAND) (§2.4). (d) *Detector bias*: per-variety P/R on synthetic and hand-checked authentic text; leave-one-feature-out (§2.3). (e) *Base "convergence" is continuation*: chat scaffold F1/F2 reported with and without (§5). (f) *Comprehension failure, not choice*: comprehension rate per condition; CR on comprehended responses (§7). (g) *Refusals* (Haq 2026): refusal rate per condition; excluded and reported (§7). (h) *Prestige post hoc*: PREREG. (i) *Judge bias*: rule-based detectors primary; judges validated on author labels (§2.3, §4.6). | | |
| **Rival account** | *Helpfulness normalisation*: preference data rewards clarity/formality per se → all non-standard input standardised equally (γ). This is the null of C2 and, if it wins, the paper's finding. | | |

Clarifications (writer; they do not change any threshold):
- H1 is stated on the **split S**, not on the level of CR. A uniform base→SFT drop in CR is expected from
  Blevins et al. 2026a and is not by itself evidence against H1 (`DECISIONS.md` D1).
- H4 is evaluated only on **full, content-identical responses** (MB_v, RB_v). Token-level reward-model
  preferences (Elle 2026) are a different quantity (D1).
- The optional E7 (multi-turn) is pre-registered **two-sided**: CAT predicts rising convergence and Mitiaeva &
  Xiao 2026 report a decline. It is reported as exploratory.

## 4. Pre-registered alternatives and outcomes that are reported as findings
**Pre-registered alternatives:** (α) selectivity already at SFT → locus = imitation (still a locus result, weaker ideology story); (β) split accumulates monotonically across stages (Dialect-Tax-like) → no single locus; (γ) uniform drop at every stage → register normalisation, not ideology; (δ, added Sep 30 after the pilot, before any main run) no spontaneous dialect-grammar convergence at any stage including base under every scaffold, while register convergence is present throughout → selectivity is a property of the pretrained model's chat behaviour, not installed by preference tuning; the locus claim then moves to where the *ability* on request (C3) and the *reward* for accommodation (C4) change, and E5 tests whether preference data can *install* accommodation.

- **(δ; added Sep 30 after the pilot, before any main run; verbatim, EXPERIMENT_PLAN §0.2)** no spontaneous
  dialect-grammar convergence at any stage including base under every scaffold, while register convergence is
  present throughout → selectivity is a property of the pretrained model's chat behaviour, not installed by
  preference tuning; the locus claim then moves to where the *ability* on request (C3) and the *reward* for
  accommodation (C4) change, and E5 tests whether preference data can *install* accommodation.
  - **Operational test (writer):** δ holds if (i) CR_v^morph has a 95% CI including 0 or lying below 0 for
    every stigmatized v at **every** stage, including base under raw, F1 and F2 scaffolds; and (ii) CR_INF > 0
    (CI excluding 0) at every stage.
  - **Consequence sentence (for the abstract if δ wins):** "Chat models do not converge to dialect grammar at
    any training stage, while they do converge to informal Standard English throughout; the selectivity is
    already present in the pretrained model's chat behaviour, and preference optimization changes what models
    can be asked to do (C3) and what they are rewarded for (C4), not whether they spontaneously accommodate."
  - **Why it was added:** the L2 pilot (`RESULTS_LOG.md` 2026-09-30, 50 seeds, detectors v0, F2 only) showed
    CR_morph ≈ 0 at every stage. The pilot is not evidence, and δ is adjudicated only on the main runs with
    detector v1.
- **Under δ (plan v1.4; lead's message 2026-09-30):**
  - The locus instruments become **MB_v (DPO vs. SFT)** and the **SFT-vs-base implicit reward**.
    Prediction: MB_v for stigmatized varieties is more negative at DPO while CR stays at 0.
  - The anti-accommodation arm of E5 is demoted to the question "does anti-data make MB_v more negative".
  - The δ title and abstract below replace the working title only if δ wins at the gate.
- **δ title and abstract (verbatim, EXPERIMENT_PLAN §0.2):** **Pre-registered δ title and abstract (decided at the Oct 3 gate):** "Can, But Won't: Language Models Mirror Register but Not Dialect Grammar, and Alignment Rewards the Default" — LMs mirror register but not dialect grammar at every stage (productive and total CR ≈ 0 at base under raw and F1); they can produce the varieties on request; the preference stage does not install the default but *rewards* it (MB_v for stigmatized varieties more negative at DPO while CR stays at 0); a few thousand accommodation-consistent pairs switch accommodation on, and either transfer to held-out varieties (a general disposition) or do not (variety-specific mimicry) — both are findings. The paradox hook survives δ in a stronger form: a property of LMs at every stage, with alignment measured in the reward.
- **H6 dissociation outcome.** If covert matched-guise associations stay flat across stages (CI overlap between
  base and DPO) while S grows at SFT→DPO, the paper reports *dissociation*: alignment standardizes behaviour
  without changing covert attitudes. This is informative and pre-registered; the verb is *suggest*.
- **Rival account (helpfulness normalization) = γ.** If it wins, it is the paper's finding for H2.

## 4a. Oct 3 gate (verbatim, EXPERIMENT_PLAN §8.4); G0 is read first
- **G0 (measurement validity; must pass before G1–G3 are read):** (a) productive *and* total CR reported at base under raw, F1 and F2; (b) every retained feature fires on ≤ 2% of SAE-prompt replies at every stage after clause gating, with D_v(SAE) tabulated by stage; (c) ≥ 3 opportunity sites per reply and retention ≥ 60% at every stage of the lineage. A lineage failing G0 supports no locus claim.
- **G1 (locus):** on ≥ 1 of L1/L2, a stage at which S drops or CR of ≥ 3 stigmatized varieties flips sign, with CIs excluding 0, and the SFT→DPO step identified as larger than base→SFT (or α/β explicitly recorded as the outcome).
- **G2 (selectivity):** at the aligned stage, CR_v − CR_INF < 0 (Holm-adjusted) on ≥ 4 stigmatized varieties while CR_INF, CR_BRE > 0, on ≥ 1 lineage, Arm S and Arm N agreeing in sign for ≥ 2 varieties.
- **G3 (mechanism feasibility):** regime determined with counts; MB_v measurable with prestige ordering on ≥ 1 lineage; E5 throughput test done.
- Pass G1+G2(+G3) → full plan (E5 sweep, L3/L4, E6, breadth). G2 only (γ-like uniform drop at one stage) → the paper becomes "alignment installs register normalisation at stage X; no dialect selectivity" plus C4/C5 as the mechanism of the uniform drop. G1 only → locus paper with the mechanism and re-run; selectivity reported as null. Neither → fold P3's people into P1/P2 (coordinator).

## 5. Primary estimands and tests

### 5.1 Definitions
The definitions are in `STATS_PLAN.md` §2 (E-1 to E-9) and copied into the frozen version below them at the
freeze. Summary:
- **CR_v(m)** = mean_i AI_i,v(m) / Δp_v on novel spans. Δp_v is held constant as the denominator.
- **S(m)**: mean CR over P minus mean CR over G.
- **L(ℓ) = Δ_SFT→DPO S − Δ_base→SFT S.**
- **ΔF_v − ΔF_INF.**
- **COMP_v** and **G_v = COMP_v − CR_v**.
- **MB_v / RB_v** as in EXPERIMENT_PLAN §3.9.
- **ρ_prestige**.
- **ρ_C6**.

### 5.2 Statistics (verbatim, EXPERIMENT_PLAN §6, v1.5a)
- **Unit of analysis:** seed prompt (300 Arm S; ≥ 100 Arm N). Responses are averaged within seed × condition × seed-of-sampling first.
- **Intervals:** tables: mean ± std over 3 sampling seeds (training seeds in Table 6); text and figures: 95% bootstrap CI over seeds (B = 2000), stratified by task type; cluster bootstrap over seeds for pooled analyses.
- **Primary estimands:** (P1) Δ_{SFT→DPO}(S) − Δ_{base→SFT}(S) per lineage, CI by seed-level bootstrap; C1 supported if > 0 with CI excluding 0 on ≥ 2 of 3 open lineages. (P2) per aligned model: CR_v − CR_INF for each stigmatized v (paired permutation over seeds, 10k permutations, Holm over varieties) and ρ_prestige(m) with CI; C2 supported if ≥ 6 of 8 stigmatized–INF contrasts are < 0 with Holm-adjusted p < .05 and ρ_prestige ≥ 0.6. (P3) COMP_v ≥ 0.5 for ≥ 70% of features, per variety and stage (binomial CI). (P4) MB_v ordering: mixed model MB ~ tier + (1 | seed) + (1 | variety). (P5) dose–response: the slope of S (treatment − control) on dose, fitted across all doses, with a bootstrap CI over training seeds and prompts (primary); Spearman over doses and the mixed model S ~ dose + (1 | seed) as secondary. (P6) Spearman(CR_v, stereotype_v) with bootstrap CI; dissociation = stage × measure interaction.
- **Tests named in captions**; effect sizes (d, CR in pp) always beside p; Holm within each hypothesis family; no "significantly" without a test.
- **Mixed models:** `statsmodels` MixedLM (CR at seed level) with lineage-specific fits; never pooled across lineages for a headline ("on k of 3 lineages").
- **Power (sketch):** with 300 seeds × 9 samples, the SE of a per-condition mean CR is ≈ 0.03–0.05 (pilot to confirm), so a stage difference of 0.15 in S is detectable at > 0.9 power.

Intervals and seeds as in STATS_PLAN §4:
- cluster bootstrap over prompts, stratified by task type, B = 2000, RNG seed 20261012;
- table ± = std over 3 sampling seeds (training seeds for E5; see §8.4).

## 6. Exclusions and confounds (verbatim, EXPERIMENT_PLAN §7, v1.5a)
1. **Copying:** novel-span filter (n ≥ 4) + COPY rate; CR also with n ≥ 6 and with no filter (appendix).
2. **Register vs dialect:** INF is the divergence reference; formality-matching of INF to each vernacular verified with the formality ranker (within ± 0.05 of the vernacular's F).
3. **Comprehension:** judge-rated comprehension per condition (consistency against 200 reference labels (two LLM raters from different families with no family overlap with the judge under validation: `openai/gpt-oss-20b` and a Claude-family model, i.e. a team pane; adjudicated; inter-rater κ reported; any family overlap disclosed)); CR on comprehended responses as the main row, all responses as a robustness row; comprehension gaps themselves reported (they are Fleisig's finding, not ours).
4. **Refusals** (Haq 2026): judge-flagged refusals excluded and their rate reported by condition and stage.
5. **Length and structure:** rates are per opportunity, so length cancels; markdown/list structure tracked (opportunity counts per response reported); a matched-opportunity subsample as a robustness row.
6. **Detector bias:** per-variety P/R gates (§2.3); leave-one-feature-out; two detector tiers reported separately.
7. **Synthetic vs authentic:** Arm S and Arm N both reported for C1/C2; density calibration; naturalness ratings.
8. **Judge bias:** rule-based detectors primary; every judge (Qwen3-8B / Qwen3-32B / Qwen2.5-7B-Instruct) is checked for **consistency** against reference labels from raters of other families (gpt-oss-20b + a Claude-family pane; inter-rater κ reported; agreement is consistency, not human validity; the user's audit is the validity evidence; family overlaps disclosed in checklist D and E1); model id + date logged in `exp/JUDGES.md`. **Judge family ≠ subject family (reviewer fix 4):** OLMo/Tülu subjects are judged by Qwen judges; Qwen subject pairs are judged by `openai/gpt-oss-20b` or `allenai/Olmo-3-7B-Instruct`; no model judges its own family.
9. **Scaffold:** raw / F1 / F2 for base (§5).
10. **Prestige order:** pre-registered (§0.3.2); a sensitivity row with an alternative ordering (Coupland & Bishop numeric scores where available).
11. **Multiple comparisons:** Holm within families; primary estimands fixed in `PREREG.md`.
12. **Heterogeneity:** effects reported per lineage; the claim is "on k of 3".

Added reporting rule (DECISIONS D1, after Haq & Saldías 2026): refusal and comprehension rates per condition ×
stage are **reported as outcomes** (a table in the appendix), not only used as exclusions.

## 7. Regime A / B (the preference-data check; section filled by Oct 1 from `exp/E4/prefdata_counts.json`)

Decision rule (EXPERIMENT_PLAN §0.3.1): **Regime A** if dialect-bearing prompts are ≥ 0.5% of the lineage's
preference mix; otherwise **Regime B**.

**Pre-registered Regime-B abstract wording (verbatim, EXPERIMENT_PLAN §0.3.1):**
> The lineage's preference data contain almost no dialect-bearing prompts, so the default is not learned from dialect-specific pairs; instead the preference stage's implicit reward penalises variety-matched replies at identical content selectively (MB_v), and as few as [≤ 5k] explicit anti-accommodation pairs install the split, while [dose] accommodation-consistent pairs remove it at matched helpfulness.

In Regime B the headline is the anti-accommodation arm ("≤ 5k pairs install the split") plus MB_v.
**Removal is untestable in Regime B and must not be claimed.**

| Mix | n pairs | % prompts dialect-bearing (per variety) | D_v(chosen) − D_v(rejected) on those | Regime | Filled on |
|---|---|---|---|---|---|
| allenai/olmo-2-0425-1b-preference-mix | | | | | |
| allenai/olmo-2-1124-7b-preference-mix | | | | | |
| allenai/llama-3.1-tulu-3-8b-preference-mixture | | | | | |
| allenai/tulu-3-sft-mixture (SFT alternative α) | | | | | |

## 8. E5 (causal re-run): controls and the held-out features

### 8.1 Dose controls (verbatim, EXPERIMENT_PLAN §E5)
- **Controls for the dose (reviewer fix 1, CRITICAL).** Regime A: a *dose-matched random-removal* control drops the same number of randomly chosen pairs at every dose (2 seeds), so the dose–response is not data quantity. Regime B: a *matched standard-pair addition* control adds the same number of pairs whose prompts are in v but whose chosen/rejected are both standard (content-matched, no accommodation signal), so the effect is not extra data on dialect-bearing prompts per se. Every dose point is reported as (treatment − control) with seed-level CIs.

### 8.2 Feature split against detector circularity (verbatim, EXPERIMENT_PLAN §E5)
- **Breaking detector circularity (reviewer fix 2, CRITICAL).** The counterfactual chosen responses are Multi-VALUE/lexicon transforms scored by the same features. Therefore each variety's retained features are split once, pre-registered, into F_train (used to build E5 pairs; ≈ 60% of features, the most frequent ones) and F_heldout (never used in pair construction; ≈ 40%). **Primary E5 outcomes:** CR_v on F_heldout and CR_v on Arm N (authentic prompts); CR on F_train is reported as the secondary (in-distribution) row. CAR_v (caricature) is reported for every run so that over-accommodation from training is visible.

### 8.2a Transfer design (verbatim, EXPERIMENT_PLAN §E5)
**Transfer design (reviewer, Sep 30; required for E5 to be a headline under δ):** pairs are built for **two training varieties only (AAE, SGE)**; CR is evaluated (i) on F_heldout of AAE/SGE, (ii) on Arm N, and (iii) on **held-out varieties NGP, SCO, IND** that never appear in the pairs. Transfer (CR rises on held-out varieties) = a general accommodation disposition that alignment data can switch on; no transfer = variety-specific mimicry. Either is a finding and is pre-registered as such.

- F_train/F_heldout (§8.3) is needed for **AAE and SGE** (the training varieties).
- NGP, SCO and IND are held out **as whole varieties**, so they need no split. Their features must still be
  frozen (§9) before any E5 run.

### 8.3 The frozen split per variety (STEP 1 frozen with this file; lead <agent>, 2026-09-30)

**Rule (amended before the freeze; replaces the frequency-ranked 60/40 rule written earlier and the "≈ 60%, most
frequent" wording in the verbatim §8.2 passage):**
- Within each (variety, tier), the eligible units are ranked by `sha1('P3-split-v1|variety|tier|unit')` and assigned
  alternately to F_train and F_heldout. This is deterministic, uses no data, and involves no hand-picking.
- **Why:** authentic frequency estimates for the small lexicons are floor-limited and unstable, and a hash split
  cannot be tuned toward an outcome.
- **Eligible units:**
  - retained morphosyntactic features;
  - lexical items with lexicon status `VARIETY`;
  - AAE *done/been/stay/finna* are detected morphosyntactically (DECISIONS D7).
  - Lexical copies of morph features are **dropped**: `gon` (= `finna_gon`) and `y'all` (= `yall`).
- **Source file:** `exp/detectors/feature_split_v1.1.json`, sha256 `9a2c751d24bb539635932073819a3fff9e0c53a767577a9ebf5e983eb620c68d` (v1 prefix e62a17ade666aecc had the two duplicates).

| Variety / tier | F_train | F_heldout |
|---|---|---|
| AAE morph | finna_gon, completive_done, ain_t | negative_concord, yall |
| AAE lex | talmbout, bruh, tryna, yo | whip, ion, fam, mane |
| SGE particles (lex) | sia, ah | lah, liao |
| SGE lex | sian, siao, walao | steady, makan, paiseh |
| SGE morph | — (descriptive only, §2.2 SGE rule) | — |

- Unretained features (for example copula_absence and uninflect, which fail the FP gate) are in neither list.
- **Step 2** (IND, SCO, IRE, APP, AUS, NGP if used in E5) is frozen here before any E5 run uses them.
- The held-out *varieties* (NGP, SCO, IND) need no split (§8.2a).
- Primary E5 outcomes: CR_v on F_heldout and on Arm N; CR on F_train is secondary; CAR_v for every run.

### 8.4 Training seeds (decided 2026-09-29; DECISIONS D4 + D5)

| Arm | Doses | Training seeds | Reported as |
|---|---|---|---|
| Treatment, Regime B | 0 / 5k / 20k | 3 | mean ± std over 3 seeds |
| Treatment, Regime B | 1k | 1 | bootstrap CI over prompts; hollow marker in plots |
| Treatment, Regime A | 0 / 50% / 100% | 3 | mean ± std over 3 seeds |
| Treatment, Regime A | 25% | 1 | bootstrap CI over prompts; hollow marker |
| Matched control (random removal / matched standard-pair addition) | mid, max | 2 | bootstrap CI over prompts (+ half-range) |
| Anti-accommodation arm (B), flip arm (A) | as in §8.1 | 1 | bootstrap CI over prompts |

- The dose–response test is the slope fitted across **all** doses on (treatment − control) with a bootstrap
  CI (STATS_PLAN §3.4). No claim rests on a single-seed point.
- Intermediate doses are topped up to 3 seeds after the Oct 3 gate if compute allows. That top-up is logged
  in §10.
- Total ≈ 17 runs.

## 9. Feature inventories per variety (step 1 frozen; the source of truth is the detector files at the hashes below)
- **AAE (morph, retained after the gate):** finna_gon, completive_done, ain_t, negative_concord, yall.
  - Excluded by the G0 FP gate: copula_absence, uninflect (3sg -s), and the others listed in
    `exp/detectors/feature_split_v1.1.json`.
  - **AAE lex:** talmbout, bruh, tryna, yo, whip, ion, fam, mane (lexicon status `VARIETY`).
- **SGE (lex; carries SGE per §2.2):**
  - particles sia, ah, lah, liao;
  - lexis sian, siao, walao, steady, makan, paiseh.
  - SGE morph (can_cannot, finish_completive) is descriptive only.
- **INF, BRE, NGP, IND, SCO, IRE, APP, AUS:** detector v1 lists were **not delivered** by the freeze (the outage stopped
  the panes at 04:04 on Sep 30). They are frozen as step 2, **before any main-run CR for that variety is computed**,
  and logged in §10. Until then, no hypothesis test uses these varieties.
- Detector file: `exp/detectors/feature_split_v1.1.json`, sha256
  9a2c751d24bb539635932073819a3fff9e0c53a767577a9ebf5e983eb620c68d. The detector code version is recorded with
  each summary (STATS_PLAN §9).

## 10. Deviations log (after the freeze)
| Date | Section | Change | Reason | Affects a falsifier? |
|---|---|---|---|---|
| 2026-10-01 02:51 CST(HKT) | all | Freeze happened at this time instead of Sep 30 22:00 | the API/network outage interrupted all panes; **no main-run (E1-E6) output existed at the freeze**, only the P0 pilot (exp/results/P0, PILOT_P0) that motivated δ (§4, logged before) | no |
| 2026-10-01 02:51 CST(HKT) | §9, §8.3 | Step-2 feature lists (INF, BRE, NGP, IND, SCO, IRE, APP, AUS) pending | not delivered before the freeze | no; they are frozen before first use |
