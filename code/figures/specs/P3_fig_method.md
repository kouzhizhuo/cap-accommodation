# P3 Figure 2 — method / framework (`figure*`, 1080 × 540 px; top of the page where §3.4 starts)

Owner of the spec: P3 lead `9e4f338d` (2026-09-29). Builder: `<agent>` (Figma Section P3 `544:577`, frame `P3_fig2_method`). Standard: `figures/FIGURE_STANDARD.md` §3, §6.2 (3–4 stage panels, every arrow labelled with what flows, ≤ 1 formula pill per zone, frozen vs trained distinguished, bottom strip = the intervention loop). Notation must match `paper/macros.tex` once the writer adds the symbols (CR, ΔF, MB; see below).

## Story sentence
Content-matched prompts in twelve varieties and registers → replies from every stage of an open alignment lineage → novel-span, opportunity-normalised feature scoring → a convergence ratio per variety and stage; the preference stage's implicit reward and a dosed DPO re-run then show where and why accommodation is withheld.

## Layout (frame-local px; four stage panels left→right, bottom strip = the causal intervention)
```
y=18 ┌① Stimuli (6–262)┐ ┌② Lineage stages (274–530)┐ ┌③ Scoring (542–798)┐ ┌④ Mechanism (810–1074)┐
     │ seed (SAE) card    │ │ base ─► SFT ─► DPO ─► RLVR│ │ reply card         │ │ pill: MB_v formula     │
     │  ↓ "12 conditions" │ │ (4 llm-blocks, lock =     │ │  ↓ scissors: shared│ │ dot plot glyph:        │
     │ Multi-VALUE rules  │ │  frozen; step ticks under │ │    4-grams removed │ │  prestige ● above 0    │
     │ + lexicon (calibr.)│ │  RLVR = branches)         │ │  ↓ detectors:      │ │  stigmatized ● below 0 │
     │ authentic pairs    │ │ arrows in: prompts ×12    │ │   real / (real+opp)│ │ arrow in: replies y_v, │
     │ (3 offset cards)   │ │ arrows out: replies       │ │  ↓ CR_v, ΔF_v, CAT │ │   y_std (same content) │
     └────────────────────┘ └───────────────────────────┘ └────────────────────┘ └────────────────────────┘
y=400 ┌ Causal re-run (E5): preference mix ± counterfactual pairs (dose 0 → 20k) ─► DPO from SFT ─► CR_v ┐  (6–1074, 400–534; vermilion-tinted strip, dashed feedback arrow with a clock = "trained")
      └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Elements and labels
- ① **Stimuli**: `seed (Standard American English)` card; arrow labelled `12 conditions`; two boxes `Multi-VALUE morphosyntax` and `lexicon / spelling` with a small gauge glyph `density-calibrated`; three offset cards `authentic pairs (EnDive, Groenwold)`; register chips `formal`, `informal`; variety chips `BRE`, `AAE`, `SGE`, `NGP`, `SCO`, … (max 6 shown, "…" for the rest).
- ② **Lineage stages**: four `arr/llm-block` glyphs with lock (frozen; we only sample): `base`, `SFT`, `DPO`, `RLVR`; a `step-rail` under RLVR labelled `checkpoints`; arrows in `prompts` (from ①), arrows out `replies` (to ③); a small chip at DPO `preference mix` (data glyph) feeding the bottom strip.
- ③ **Scoring**: reply card → scissors glyph `shared 4-grams removed (novel spans)` → detector box `feature f: realised / (realised + opportunity)` → outputs `CR_v` (convergence ratio), `ΔF_v` (register shift), `CAT: converge / maintain / diverge`; control chips `INF reference`, `BRE`, `on request (competence)`.
- ④ **Mechanism**: one formula pill (STIX): `MB_v = Δ_v [ log π_DPO(y|x) − log π_SFT(y|x) ]` with sub-labels `y_v vs y_std, same content`; dot-plot glyph with a zero rule: blue dots above (prestige cues), orange dots below (stigmatized); arrow in from ③ `content-identical replies`.
- **Bottom strip (causal re-run)**: data glyph `preference mix` → `± counterfactual pairs, dose 0…20k` (3 offset cards, one dashed) → llm-block `DPO from SFT` (contribution colour, heaviest stroke = trained) → `CR_v` output; dashed vermilion feedback arrow with clock glyph looping to ② DPO labelled `replaces the stage`.
- Pills: `① Stimuli`, `② Lineage stages`, `③ Scoring`, `④ Mechanism`, strip pill `Causal re-run`.

## Notation to add to `paper/macros.tex` (writer; the builder reads the macro before typing a symbol)
`\CR` (convergence ratio CR_v), `\dF` (ΔF_v), `\MB` (matching bonus MB_v), `\pistage{DPO}` (π_DPO), `\opp`/`\real` (opportunity / realised counts).

## Caption draft
\textbf{Framework.} (1) Each Standard-English seed is rendered in twelve varieties and registers (Multi-VALUE morphosyntax plus lexicon, density-calibrated to authentic corpora) and paired with authentic parallel prompts; (2) every stage of an open lineage answers the same prompts; (3) replies are scored on novel spans with opportunity-normalised feature detectors, giving a convergence ratio, a register shift and a CAT category per variety and stage; (4) the preference stage's implicit reward on content-identical replies, and a DPO re-run from the SFT checkpoint with a dose of counterfactual pairs, locate and test the cause. Schematic; no measured values.

## Notes for the builder
Nothing flows back except the dashed re-run loop. Frozen vs trained: locks on ②; contribution colour only on the re-run block. Density diagnostic: ≤ ~47 words of labels; draw the mechanism (scissors, gauge, dots) rather than name it.
