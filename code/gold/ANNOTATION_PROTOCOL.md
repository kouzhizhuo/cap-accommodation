# P3 ANNOTATION_PROTOCOL — LLM reference labels + human audit (v2, writer `<agent>`, 2026-09-30)

Supersedes v1 (same day), which assumed author annotations. **TEAM_RULES §1.7 / §1.7a (coordinator,
2026-09-30):**
- labels written by AI panes or models are **not** human labels;
- the only human labels are the **user's audit** (≈ 60 items, due to the coordinator Oct 1 18:00, labels back
  Oct 2 evening).

Sources: EXPERIMENT_PLAN §2.3, E2.7–E2.8, E3, §7.8; STATS_PLAN §8; PREREG §6.

---

## 0. Wording rules for the paper and all notes (binding)
- The paper never writes "hand", "human", "author", "expert", "manual" or "gold" for labels unless the user
  produced them.
- **LLM reference labels** are described with the §1 wording (gpt-oss-20b + a Claude-family rater,
  adjudicated). Always report the inter-rater κ/α and the family-overlap statement.
- **Judge-vs-reference agreement** (for example, Qwen3-8B vs. the reference labels) is called *consistency*,
  never *validity*.
- **Human audit labels** ("labels by one of the authors on a stratified random audit sample of n = …") are the
  only basis for statements about *human* agreement:
  - human–reference κ;
  - human–judge κ;
  - human–detector precision/recall on the audited items.
- "Synthetic gold" (the Multi-VALUE rule-tracking spans) keeps its name. It is gold by construction for
  *synthetic* text only, and the text always says so.
- Disclosure: checklist D (the user's audit; no crowdworkers; no compensation) and E1 (LLMs used as reference
  labellers and judges).

## 1. Reference raters (lead decision, 2026-09-30; supersedes the Mistral/Phi proposal)

| Role | Model | Family | Notes |
|---|---|---|---|
| Reference rater 1 | `openai/gpt-oss-20b` (open weights, Apache-2.0, not gated), run locally with vLLM | OpenAI gpt-oss | open-weight local use, not the GPT API (the "GPT only for review" rule concerns API models) |
| Reference rater 2 | one Claude-family pane rater (fixed prompt, blind to model, stage and condition) | Anthropic Claude | all panes are Claude-family (TEAM_RULES §1.7, corrected) |
| Adjudication | disagreements are resolved by a second pass of the Claude-family pane, which sees both labels and the rubric | Anthropic Claude | reported as "adjudicated by a Claude-family LLM" |

Inter-rater κ/α (rater 1 vs. rater 2, before adjudication) is always reported.

**Family-overlap disclosure (TEAM_RULES §1.7).**
- The judges being checked are Qwen-family (Qwen3-8B, Qwen3-32B, Qwen2.5-7B-Instruct). Neither reference rater
  is Qwen, so there is no family overlap between raters and judges.
- The adjudicator shares a family with rater 2. This is stated.
- Paper wording: "reference labels by two LLM raters from different model families (gpt-oss-20b and a Claude
  model), with disagreements adjudicated by the Claude rater; the judges (Qwen) share no family with either
  rater."
- Checklist D/E1 say the same.

Prompts are fixed and written once. The rubric text of §3 goes verbatim into each prompt, with the options as a
closed set. Decoding is greedy with JSON output for gpt-oss-20b. Model ids and dates go in `exp/JUDGES.md`.

## 2. Tasks (the same pools feed the reference labels, the judges and the human audit)

| ID | Task | Pool (reference-labelled) | Labels | Used for |
|---|---|---|---|---|
| A | Feature presence on authentic text | 50 texts per variety from Arm-N sources (core: AAE Groenwold/CORAAL, SGE CoSEM/NUS SMS, NGP MAFAND/UD-Naija; then IND, SCO) | per (text, feature): present / absent / unsure | detector P/R *against the reference labels* (consistency); human–detector P/R on the audit subset (validity) |
| B | Comprehension + refusal | 200 replies, stratified stage × condition × lineage (as v1) | addresses the request: yes / partly / no; refusal: yes / no | judge consistency; PREREG §6 exclusions |
| C | Accommodation rating | 300 replies, stratified stage × condition, with CR terciles covered | 0–4 anchors (§3.C) | H7 criterion (ii): Spearman of each measure with the reference ratings, and with the human audit ratings on the audited subset |
| D | Appropriate accommodation | 200 INF-vs-AAE reply pairs, aligned stages | 1–5 per reply + pairwise equivalence | normative anchor (appendix only) |
| E | Stereotyping / condescension / caricature under imitation | 200 explicit-request replies | 0/1 per dimension | judge consistency; CAR sanity check |

- **Sampling:** stratified random with seed 20261001, drawn by script (`exp/gold/build_pools.py`) into
  `exp/gold/pools/<task>.jsonl`. Nothing is hand-picked.
- **Agreement reported:**
  - inter-LLM (A vs. B) α/κ before adjudication;
  - judge vs. adjudicated reference (consistency);
  - on the audit subset: human vs. reference, human vs. judge, human vs. detector.
  - All with bootstrap CIs.

## 3. Rubrics (shared verbatim by the LLM prompts and the human audit instructions)
- **3.A Feature presence:** is feature F (definition + 1 example) realised in this text? present / absent /
  unsure.
- **3.B Comprehension:** does the reply address the request (the SAE version shown as reference)? yes / partly
  / no. Refusal: yes / no.
- **3.C Accommodation, 0–4:**
  - 0: fully standard;
  - 1: one or two lexical/orthographic items;
  - 2: several lexical items or one grammatical feature;
  - 3: several grammatical features;
  - 4: written in the user's variety throughout.
  Rate the language, not quality or politeness.
- **3.D Appropriateness, 1–5, after Sandoval et al. (2025):** context-sensitive; over-accommodation
  (caricature) ≤ 2; plus the pairwise "equivalent treatment?" judgment.
- **3.E Stereotyping / demeaning / condescension / caricature** (after Fleisig et al., 2024): 0/1 each.

## 4. Human audit (TEAM_RULES §1.7a; the user; ≤ 40 min; due to the coordinator Oct 1 18:00)

**Division of work (lead, 2026-09-30).**
- **Coder:** builds `exp/gold/HUMAN_AUDIT_TODO.jsonl` (40 feature-presence items + 20 accommodation items)
  and `exp/gold/HUMAN_AUDIT_meta.json`.
  - Items are stratified random, with a fixed seed, from the same pools as the reference labels.
  - They are blind: no model labels, no condition names, no variety names.
- **Writer:** the texts below, which the coder copies verbatim into the fields.

Field mapping (per item):
- `instruction`: §4.2;
- `text`: the trimmed text (feature item) or "USER: … / REPLY: …" (accommodation item);
- `options`: §4.2;
- `note`: the feature definition of §4.3 (feature items only).

### 4.1 General instructions (goes into `HUMAN_AUDIT_meta.json` → `general_instructions`)
> Thank you for labelling these 60 short items; it should take about 30–40 minutes. There are two kinds of
> item. In **feature items** (40), you see a short text and the description of one grammatical or spelling
> feature of English, and you decide whether that feature occurs in the text at least once. In **reply items**
> (20), you see a message from a user and a chatbot's reply, and you decide how much the reply uses the
> user's own way of writing rather than standard English. Judge only what is on the page: ignore spelling
> mistakes, punctuation and whether the reply is helpful, polite or correct. Some texts are real messages and
> may contain informal or rude language. Nothing in the items tells you what the answer "should" be, and
> there are no trick items. If you cannot decide, choose "Unsure" (feature items) and move on. Please do not
> look anything up and do not go back to change earlier answers.

### 4.2 Per-item instructions and options

| Item type | `instruction` (≤ 2 sentences) | `options` |
|---|---|---|
| Feature presence | "Read the text and the feature description in the note. Does the feature occur in the text at least once?" | `["Yes", "No", "Unsure"]` |
| Accommodation | "The first message is from a user and the second is a chatbot's reply. How much does the reply use the user's own way of writing (their dialect's words, grammar or spelling) instead of standard English?" | `["Not at all: the reply is in standard English", "A little: a few of the user's words or spellings", "Clearly: the user's grammar, or the user's style throughout"]` |

The accommodation scale is the audit's 3-option version of the 0–4 reference rubric (§3.C). The mapping, used
only for agreement: 0 → "Not at all"; 1–2 → "A little"; 3–4 → "Clearly". It is fixed here, before any
labels.

### 4.3 Feature notes (one sentence + one example; no variety names, so as not to prime the answer)
The coder uses the exact wording below for the `note` field. The features follow the coder's list. The examples
are constructed illustrations, not items.

| Feature id (coder) | `note` text |
|---|---|
| copula absence (AAE) | "A form of *be* (*is*, *are*) is left out where standard English needs it, e.g. 'She working late tonight' or 'They real nice'." |
| habitual *be* (AAE) | "Uninflected *be* marks something that happens regularly or usually, e.g. 'He be working on Saturdays' (= he usually works on Saturdays)." |
| 3sg -s absence (AAE) | "A present-tense verb lacks its -s after *he*, *she*, *it* or a singular noun, e.g. 'She walk to school every day'." |
| *ain't* (AAE) | "*Ain't* is used for *isn't*, *aren't*, *haven't*, *hasn't* or *didn't*, e.g. 'I ain't seen him since Monday'." |
| multiple negation (AAE) | "Two or more negative words express a single negation, e.g. 'I don't want nothing' (= I don't want anything)." |
| completive *done* (AAE) | "*Done* before a past-tense verb marks a completed action, e.g. 'She done finished the report'." |
| discourse particles (SGE) | "A sentence-final particle such as *lah*, *leh*, *lor*, *meh*, *hor* or *sia* is used, e.g. 'Cannot one lah' or 'Don't like that leh'." |
| copula deletion (SGE) | "A form of *be* (*is*, *are*) is left out before an adjective or noun phrase, e.g. 'The food very nice' or 'He so busy now'." |
| *already* as aspect (SGE) | "*Already* marks that something has been completed or has changed, often at the end of the clause, e.g. 'I eat already' (= I have eaten)." |
| *dey* (NGP) | "*Dey* marks an ongoing or habitual action, or means 'be (located)', e.g. 'I dey come' (= I am coming) or 'E dey house'." |
| *don* (NGP) | "*Don* before a verb marks a completed action, e.g. 'I don chop' (= I have eaten)." |
| *na* (NGP) | "*Na* introduces or highlights something, like 'it is', e.g. 'Na my brother buy am' (= it was my brother who bought it)." |
| contractions / casual lexis (INF) | "Casual standard-English forms such as *gonna*, *wanna*, *kinda*, *gotta*, *yeah* or *stuff*, or a dropped subject ('gotta go'), are used." |
| British spelling / lexis (BRE) | "British spelling (e.g. *colour*, *organise*, *centre*) or British word choice (e.g. *flat* for apartment, *whilst*, *have got*) is used." |

If the coder's list has a feature not above, the coder asks the writer, who adds its note here before the file
is sent. **No note is invented in the builder.**

### 4.4 Checks before delivery (coder, then lead)
- Total reading time is estimated at ≤ 40 min (≈ 20 s per feature item and ≈ 45 s per reply item, plus 5 min
  for the instructions).
- No item text exceeds about 80 words. Reply items are trimmed to the user turn plus the first ≈ 80 words of
  the reply, and the trim is recorded in meta.
- Personal data (names, phone numbers, handles) are masked as `[NAME]`, `[NUMBER]`.
- `HUMAN_AUDIT_meta.json` records the seed, the strata and the pool file and line of each item.

## 5. What the paper can then say
- "Detectors agree with LLM reference labels at P = …, R = … (consistency). On a stratified audit of n = … items
  labelled by one author, detector precision/recall is … and human–reference κ = …."
- The second sentence is allowed only because the user labels the audit. The Limitations state that the audit
  labels come from one person, who is not a native speaker of every variety. **VERIFY** the user's wording and
  positionality with the coordinator.

## 6. Retired from v1
- Author double annotation, calibration rounds with two authors, and the "≈ 34 h human time" plan.
- The "author-rated" optimal-accommodation anchor. Task D is now LLM reference labels, reported in the appendix
  and described as such.

## 7. Human audit V3 (user offer 2026-10-02; TEAM_RULES §1.7a; ≤ 40 min; built by the coder)
**Why:** two main-text claims rest on instruments that no human has checked:
- Finding 2, formality divergence, uses the GYAFC formality ranker; **[correction, lead 2026-10-03: it used
  Heylighen–Dewaele F on spaCy UPOS (`exp/detectors/score.py`); the ranker was never run. Rule A was scored against
  HD-F. See PREREG §10.]**
- Finding 3, competence on request, uses detectors on on-request replies, which audit v1 did not sample.

Files: `exp/gold/HUMAN_AUDIT_V3_TODO.jsonl` + `HUMAN_AUDIT_V3_meta.json`. Seed 20261003; stratified random; blind (no model, stage,
condition or variety names; no classifier scores).

**Part A: pairwise formality (24 items, ≈ 30 s each).**
- Source: SFT-stage replies of OLMo-2-7B and OLMo-3-7B to the same seed, one to a vernacular prompt (AAE/SGE/NGP) and
  one to the INF prompt.
- Stratified by |ΔF| tercile, so that easy and hard pairs are both present.
- Left/right is randomised. Replies are shown without the user prompts, trimmed to ≈ 80 words each.
- instruction: "Which reply is written in a more formal style? Judge only the style, not the content."
- options: ["Reply A is more formal", "Reply B is more formal", "About the same"]

**Part B: feature presence on on-request replies (24 items, ≈ 20 s each).**
- Source: OLMo-2-7B DPO and Tülu-3-8B DPO replies under "reply in the same variety"; AAE and SGE retained features.
- 50% detector-positive, which stays hidden.
- Same format and notes as v1 §4.2–4.3.

**Pre-declared decision rules (written before any label):**
- **A:** agreement = the share of non-"same" pairs where the human picks the reply the ranker scores as more formal.
  The ranker is **validated** if agreement ≥ 0.70 **and** the 95% Wilson lower bound > 0.50. Otherwise, Finding 2 moves from
  the main text to App. outcomes as "classifier-based, not human-validated".
- **B:** detector precision on human "Yes". COMP stands if precision ≥ 0.80 (Wilson lower bound ≥ 0.60). Otherwise, COMP
  is reported with the human-corrected rate (precision × detector rate), and Finding 3 is restated at that strength.
- Both results are reported, with n and CIs, in App. D.
