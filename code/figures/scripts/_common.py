"""Shared loaders for the P3 result plots. Reads ONLY <paper>/exp/results/** (no typed numbers).

Sources (first that exists wins; the PILOT tag is added to output file names when a pilot file is used):
  main run : exp/results/E1/summary.json   (E4/summary.json, E5/summary.json for the mechanism and dose plots)
  pilot    : exp/results/PILOT_P0/summary_L{1,2}.json
Schemas: see DATA_SCHEMA.md.  `--fixture` swaps in figures/scripts/_fixtures/SYNTHETIC_*.json and writes to a
scratch dir, never to figures/plots (fixtures test the layout only; they are not results).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]                       # P3_accommodation/
RESULTS = PAPER / "exp" / "results"
PLOTS = PAPER / "figures" / "plots"
FIXTURES = HERE / "_fixtures"
sys.path.insert(0, str(PAPER.parent / "figures" / "style"))
import palette as P  # noqa: E402

# Role colours fixed by the P3 lead (FIGURE_STANDARD §5.3): ours/re-run = blue, prestige + style
# convergence = green, stigmatized dialect = orange, withheld/anti-arm = vermilion dashed, base = grey.
C_OURS, C_PRESTIGE, C_STIG, C_WITHHELD, C_BASE = P.BLUE, P.GREEN, P.ORANGE, P.VERMILION, P.GREY

# Ordinal tier per PREREG §2.2 (P = prestige cue, M = mid, G = stigmatized); provisional until the freeze.
# A results file may override it with a "tier" field per variety.
TIER = {"FOR": "P", "INF": "P", "BRE": "P", "AUS": "P", "IRE": "M", "IND": "M",
        "AAE": "G", "SGE": "G", "NGP": "G", "SCO": "G", "APP": "G"}
TIER_RANK = {"P": 0, "M": 1, "G": 2}
STAGE_ORDER = ["base", "sft", "dpo", "rlvr", "final"]
STAGE_LABEL = {"base": "base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR", "final": "final"}
NAN = float("nan")
PRELIM = ""             # E1/E4 declared final by lead <agent> (2026-10-04); set to "_PRELIM" to stamp main-run files again


def tier_of(v: str, blob: dict | None = None) -> str:
    return (blob or {}).get("tier") or TIER.get(v, "G")


def order_varieties(vs, tiers=None):
    """Prestige first, stigmatized last; ties keep alphabetical order."""
    return sorted(vs, key=lambda v: (TIER_RANK[(tiers or {}).get(v) or tier_of(v)], v))


def color_of(v: str, tiers=None) -> str:
    t = (tiers or {}).get(v) or tier_of(v)
    return {"P": C_PRESTIGE, "M": C_BASE}.get(t, C_STIG)     # mid tier (excluded from S) in neutral grey


def args(desc: str):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--fixture", action="store_true", help="layout test on SYNTHETIC fixture; writes to scratch, not figures/plots")
    ap.add_argument("--lineages", default="l1,l2", help="comma list of lineage tokens (E1/summary.json keys)")
    ap.add_argument("--out", default=None, help="output dir (default figures/plots)")
    return ap.parse_args()


def outdir(a) -> Path:
    if a.out:
        return Path(a.out)
    return FIXTURES / "_out" if a.fixture else PLOTS


def jload(path: Path):
    with open(path) as f:
        return json.load(f, parse_constant=lambda c: NAN)


def find(a, main: str, pilot: list[str] | None, fixture: str):
    """Return ([paths], tag). tag = '' for main-run files, '_PILOT' for pilot files, '_SYNTHETIC' for fixtures."""
    if a.fixture:
        return [FIXTURES / fixture], "_SYNTHETIC"
    if (RESULTS / main).exists():
        return [RESULTS / main], PRELIM
    ps = [RESULTS / p for p in (pilot or []) if (RESULTS / p).exists()]
    if ps:
        return ps, "_PILOT"
    sys.exit(f"no results yet: neither exp/results/{main} nor a pilot file exists ({pilot}). Nothing plotted.")


def split_key(k: str):
    lin, st = k.split("@", 1)
    return lin, st


LINEAGE = {"L1": "l1", "L2": "l2", "L3": "l3", "L4": "l4", "L2B": "olmo2_13b"}   # pilot / style-file keys -> lineage tokens of E1/summary.json
TITLE = {"l1": "OLMo-2-1B", "l2": "OLMo-2-7B", "l3": "OLMo-3", "l4": "Tülu 3", "olmo2_13b": "OLMo-2-13B", "olmo2_32b": "OLMo-2-32B"}
STAGE_ALIAS = {"rlvr1": "rlvr", "instruct": "final"}  # pilot stage keys -> plot stage tokens


class Res:
    """Flat table {'e1.olmo2_7b.dpo.aae.cr': x} in the STATS_PLAN §9 key scheme <exp>.<lineage>.<stage>.<variety>.<stat>.
    Main-run summary.json files are already flat. Pilot summaries (nested by model-stage, coder's P0 format) are flattened on load:
    CR_morph -> cr_morph, CR_lex -> cr_lex, *_ci -> *_lo/_hi, dF -> df."""

    def __init__(self, paths, exp="e1"):
        self.d = {}
        for p in paths:
            j = jload(p)
            if "models" in j:                       # pilot layout
                for mk, m in j["models"].items():
                    lin, st = split_key(mk)
                    lin = LINEAGE.get(lin, lin.lower()); st = STAGE_ALIAS.get(st, st)
                    for v, sc in m.get("varieties", {}).items():
                        n = sc.get("novel", {})
                        base = f"{exp}.{lin}.{st}.{v.lower()}."
                        for stat, key in (("cr_morph", "CR_morph"), ("cr_lex", "CR_lex"), ("df", "dF")):
                            if key in n:
                                self.d[base + stat] = n[key]
                            if key + "_ci" in n:
                                self.d[base + stat + "_lo"], self.d[base + stat + "_hi"] = n[key + "_ci"]
                            elif key.lower() + "_ci" in n:
                                self.d[base + stat + "_lo"], self.d[base + stat + "_hi"] = n[key.lower() + "_ci"]
                        if "dF_ci" in n:
                            self.d[base + "df_lo"], self.d[base + "df_hi"] = n["dF_ci"]
                        self.d[base + "n_seeds"] = n.get("n_seeds", NAN)
            elif any(k in j for k in ("l1", "olmo2_32b", "olmo31_32b", "tulu2_7b")) and all(isinstance(v, dict) for k, v in j.items() if k in ("l1", "l2", "olmo2_32b", "olmo31_32b", "tulu2_7b")):   # E1/summary.json: {lineage: {stage: {variety: {stat: x}}}}
                for lin, stg in j.items():
                    if not (isinstance(stg, dict) and lin not in ("prompts", "meta")):
                        continue
                    for st, vs in stg.items():
                        if not isinstance(vs, dict):
                            continue
                        stn = STAGE_ALIAS.get(st, st)
                        for v, stats in vs.items():
                            if isinstance(stats, dict):
                                for k, x in stats.items():
                                    if isinstance(x, (int, float)):
                                        self.d[f"{exp}.{lin}.{stn}.{v}.{k}"] = x
            else:
                self.d.update({k: v for k, v in j.items() if isinstance(v, (int, float)) or v is None})

    def get(self, exp, lin, st, var, stat, default=NAN):
        v = self.d.get(f"{exp}.{lin}.{st}.{var.lower()}.{stat}", default)
        return NAN if v is None else v

    def triple(self, exp, lin, st, var, stat):
        return (self.get(exp, lin, st, var, stat), self.get(exp, lin, st, var, stat + "_lo"), self.get(exp, lin, st, var, stat + "_hi"))

    def has(self, exp, lin, st, var, stat):
        return f"{exp}.{lin}.{st}.{var.lower()}.{stat}" in self.d

    def tokens(self, exp, pos):
        """distinct tokens at key position pos (0=exp, 1=lineage, 2=stage/run, 3=variety)"""
        return sorted({k.split(".")[pos] for k in self.d if k.startswith(exp + ".")}) if pos else [exp]

    def stages(self, exp, lin):
        have = {k.split(".")[2] for k in self.d if k.startswith(f"{exp}.{lin}.")}
        return [s for s in STAGE_ORDER if s in have]

    def varieties(self, exp, lin):
        return sorted({k.split(".")[3] for k in self.d if k.startswith(f"{exp}.{lin}.") and len(k.split(".")) == 5})


def stat_for(res, exp, lin, st, var, want="cr"):
    """primary 'cr' when the main run provides it, else the morphosyntactic tier (pilot)."""
    return want if res.has(exp, lin, st, var, want) else want + "_morph"


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def stamp(fig, tag: str, src: str = "", top: bool = False):
    """Small provenance line under the figure: pilot/synthetic marking + source file (kept >= 6.5 pt)."""
    if tag:
        fig.text(1.0, 1.0 if top else 0.0, {"_PILOT": "PILOT · 50 seeds · detectors v0", "_PRELIM": "PRELIM · not gate-final", "_SYNTHETIC": "SYNTHETIC LAYOUT TEST · not results"}[tag],
                 ha="right", va="top" if top else "bottom", fontsize=P.SIZE["sub"], color=P.MUTED, style="italic")


def style_delta(path, measure="lsm_novel"):
    """{(lineage, stage, VARIETY): (delta, lo, hi)} from E1/style_summary.json 'style_ai' (mean(cond) - mean(SAE) on the same replies)."""
    j = jload(path)["style_ai"]; out = {}
    for mk, conds in j.items():
        if "@" not in mk or "-" in mk.split("@")[1]:
            continue                       # skip base-raw / base-f1 scaffolds
        lin, st = split_key(mk)
        for cond, ms in conds.items():
            if measure in ms:
                out[(LINEAGE.get(lin, lin.lower()), STAGE_ALIAS.get(st, st), cond)] = (ms[measure]["delta"], *ms[measure]["ci"])
    return out
