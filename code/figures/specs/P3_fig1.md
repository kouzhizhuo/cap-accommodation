> **REVISION Sep 30 (lead), supersedes the layout below where they differ.** The pilot shows that aligned-stage replies to the *informal* prompt are not informal either (register mirroring is real but tiny per reply), so "mirrored" chips on reply A cannot be drawn honestly. New story, true under C1 and under alternative δ: **"Can, but won't."** Setting panel = one real AAE user turn (orange chips only where the detector fires) + the same aligned model's **default reply** (Standard English; ghost outlines where features could have appeared) and its **on-request reply** (system prompt "The user writes in AAE; reply in the same variety"; orange chips where features are realised). Seed selection rule (pre-registered): the seed whose default-reply ΔF is closest to the median among seeds whose on-request reply realises ≥ 2 retained features. Existing panel unchanged (grey, "alignment removes accommodation"). Ours panel: stage rail base → SFT → DPO → RLVR, a reward-gap glyph (two reply cards on a balance: standard reply weighed heavier = MB_v < 0, schematic) and the lock glyph "can, but won't"; the behavioural curve stays an editable group and its DPO drop is drawn only if C1 survives the Oct 3 gate. Source for the on-request replies: `exp/results/PILOT_P0/gen/L2@dpo_request.jsonl` (coder, today) until E2 lands. Everything else (chassis, colours, labels ≤ 18, caption first sentence bold) as below.

# P3 Figure 1 — teaser (column, 520 × 460 px; `figure`, page 1 right column)

Owner of the spec: P3 lead `9e4f338d` (2026-09-29). Builder: `<agent>` (Figma page ARR `532:2`, Section P3 `544:577`, frame `P3_fig1_teaser`). Standard: `figures/FIGURE_STANDARD.md` §3, §6.1 (no measured numbers; real text only; ≤ 18 labels). Colour roles (P3 remap agreed with <agent> on Sep 30, to be recorded in FIGURE_STANDARD §5.3): **ours / contribution = blue `#0072B2`** (the re-run block, our index); **prestige cues and style convergence = green `#009E73`**; **stigmatized dialect features = orange `#E69F00`**; base grey `#8C8C8C`; intervention / withheld = vermilion `#D55E00` (dashed); pill `#2F6B75`. The same mapping is used in Fig. 2 and in every matplotlib plot (Fig. 3/4).

## Story sentence
A chat model that mirrors an informal Standard-English user standardises an AAE user's *same* request after preference tuning, although it can still write AAE on request: it converges on how you write, not on who you are.

## Layout (frame-local px)
```
┌ Setting: same request, two users ─────────────────────────────── (6,18)–(514,190) grey tint ┐
│ ┌ card A: user (informal Std. English) ┐  ┌ card B: user (AAE) ─────────────────┐            │
│ │ "<real INF prompt excerpt…>"          │  │ "<real AAE prompt excerpt…>"        │            │
│ │  chips: [contraction] [casual lexis] (green) │  │  chips: [copula absence] [habitual be] (orange)  │
│ │ ── reply (DPO stage) ──               │  │ ── reply (DPO stage) ──             │            │
│ │ "<real reply excerpt…>" chips kept    │  │ "<real reply excerpt…>" no chips;   │            │
│ │  (green: mirrored)                     │  │  ghost outlines where features fit  │            │
│ └───────────────────────────────────────┘  └─────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
┌ Existing: alignment removes ─── (6,204)–(254,454) ┐ ┌ Ours: trained not to accommodate ─ (266,204)–(514,454) blue tint ┐
│ rail: base ──────────► instruct                  │ │ rail: base ─► SFT ─► DPO ─► RLVR   (step-rail, 4 ticks)         │
│ one grey curve falling ("converges less")        │ │ green curve (style / prestige cues) stays high                      │
│ label: "style and dialect alike"                 │ │ orange curve (stigmatized dialect) drops below the 0 rule at DPO,  │
│                                                  │ │   continuing dashed vermilion ("withheld")                         │
│                                                  │ │ lock glyph + chip "can, but won't"; small chip "on request ✓"      │
└──────────────────────────────────────────────────┘ └────────────────────────────────────────────────────────────────────┘
```
Panel geometry: r = 14, 2-px stroke, tints per §5.2 (grey / grey-vermilion for "Existing" / blue for "Ours"). Pills: "Setting: same request, two users", "Existing: alignment removes", "Ours: trained not to accommodate" (≤ 4 words each; re-fit width to text + 28).

## Elements and exact labels (≤ 18 labels)
1. Card A header chip: `informal Standard English` (grey chip). 2. Card B header chip: `AAE` (orange chip).
3. Card A prompt excerpt (real, ≤ 2 lines; from `exp/stimuli/v1/INF.jsonl`, seed id recorded below). 4. Card B prompt excerpt (real, same seed; `exp/stimuli/v1/AAE.jsonl`).
5. Card A reply excerpt (real, ≤ 2 lines; `exp/E1/gen/L2@DPO/INF.jsonl`, same seed, sampling seed 0). 6. Card B reply excerpt (real; `exp/E1/gen/L2@DPO/AAE.jsonl`).
7. Feature chips on the AAE prompt: `copula absence`, `habitual be` (orange, ink edge). 8. Style chips on the INF prompt: `contraction`, `casual lexis` (green).
9. Reply chips: green `mirrored` on reply A; ghost (dashed grey) outline `standardised` on reply B.
10. "Existing" rail labels: `base`, `instruct`; curve label `converges less`. 11. "Ours" rail labels: `base`, `SFT`, `DPO`, `RLVR`.
12. Curve labels: `style, prestige cues` (green), `stigmatized dialect` (orange → dashed vermilion `withheld`). 13. Zero rule label `0`. 14. Lock glyph + chip `can, but won't`. 15. Chip `on request ✓` at the DPO tick.
Glyphs to instance from LIB: `arr/cot-card` (header = user chip; body = real excerpt with chips), `arr/step-rail` (4 ticks), `arr/llm-block` frozen (lock), token chips from `arr/token-strip` (state = highlight / deleted).

## Text-excerpt sources (mandatory, §4 rule 5)
- Prompts: `P3_accommodation/exp/stimuli/v1/{INF,AAE}.jsonl`, field `text`, seed id `<to be chosen by the lead after Oct 1; recorded here>`; both are our stimuli (synthetic, density-calibrated; caption says so) — if an authentic Arm-N pair (Groenwold/EnDive) exists for the chosen seed, prefer it and name the source.
- Replies: `P3_accommodation/exp/E1/gen/L2@DPO/{INF,AAE}.jsonl` (OLMo-2-7B-DPO, `allenai/OLMo-2-1124-7B-DPO`, T = 0.7, seed 0), field `response`, verbatim, elided with "…" only. Until Oct 1 the frame carries `\figplaceholder`-style grey text "real excerpt pending (exp/E1)".

## Caption draft (LaTeX, first sentence bold = the finding)
\textbf{Chat models are trained not to accommodate.} Top: the same request from an informal Standard-English user and from an AAE user (our content-matched stimuli), and the verbatim replies of OLMo-2-7B after DPO: the model mirrors the informal register but standardises the AAE request. Bottom: prior work reads alignment as a general loss of accommodation (left); we find the loss is selective and installed at the preference stage, while the ability to write the variety on request remains (right; schematic, no measured values).

## Notes for the builder
- The reviewer asked for a heat-map (varieties × stages) as "Figure 1"; per T6/T5 rules the teaser carries no numbers, so the heat-map is **Figure 3a** (matplotlib, `figures/plots/P3/`), and this teaser carries the paradox with real text. Flag to the lead if you disagree.
- Height may grow to 480 px; width never changes. Grayscale check: green vs orange curves also differ by dash (orange becomes dashed vermilion after DPO) and by direct end labels.
