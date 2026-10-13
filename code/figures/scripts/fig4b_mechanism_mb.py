"""Fig 4b — mechanism: update diagnostic MB_v by variety (preference-tuned policy vs SFT; EXPERIMENT_PLAN §3.9).
(a) MB_v per variety, ordered prestige -> stigmatized, dots with 95% CIs, one marker per lineage; 0 = no matching bonus.
(b) scatter: MB_v (x) vs the accommodation index CR_v (y) of the same aligned model, one dot per variety (E1 + E4).
Reads exp/results/E4/mbv.json ({lineage: {contrast: {variety: {mb, mb_lo, mb_hi}}}}; contrast dpo_vs_sft = DPO implicit reward vs the SFT
reference) and E1/summary.json for (b). --lineages selects lineages (default l1,l2).
"""
import numpy as np
from _common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

a = args(__doc__)
p4, tag = find(a, "E4/mbv.json", None, "SYNTHETIC_e4.json")
p1, _ = find(a, "E1/summary.json", ["PILOT_P0/summary_L1.json", "PILOT_P0/summary_L2.json"], "SYNTHETIC_e1.json")
R4, R1 = Res(p4, "e4"), Res(p1)
if any("PILOT" in str(p) for p in p1) and tag == "":
    tag = "_PILOT"
lins = [l for l in a.lineages.split(",") if l in R4.tokens("e4", 1)]
stage = "dpo_vs_sft"
mk = {"l1": "o", "l2": "s", "l3": "D", "l4": "^", "olmo2_13b": "v"}
title = TITLE
P.use()
fig, axs = plt.subplots(1, 2, figsize=P.figsize("text", height=1.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
vs = order_varieties([v.upper() for v in R4.varieties("e4", lins[0])])
x = np.arange(len(vs))
ax.axhline(0, color=P.INK_2, lw=P.LW["axis"], zorder=1)
for k, lin in enumerate(lins):
    off = (k - (len(lins) - 1) / 2) * 0.22
    for i, v in enumerate(vs):
        m, lo, hi = R4.triple("e4", lin, stage, v, "mb")
        if isnan(m):
            continue
        c = color_of(v)
        ax.errorbar(x[i] + off, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=P.INK, elinewidth=P.LW["thin"], capsize=1.5, zorder=2)
        ax.plot(x[i] + off, m, ls="none", **P.marker_kw(c, marker=mk.get(lin, "o"), markersize=4.2), zorder=3)
ax.set_xticks(x, vs, rotation=0)
ax.tick_params(axis="x", labelsize=P.SIZE["sub"] + .5)
ax.set_ylabel(r"MB$_v$ (update diagnostic)")
ax.spines["bottom"].set_bounds(-.5, len(vs) - .5)
for i, v in enumerate(vs):
    ax.get_xticklabels()[i].set_color(P.INK)
hs = [ax.plot([], [], ls="none", **P.marker_kw(C_PRESTIGE), label="prestige cue")[0], ax.plot([], [], ls="none", **P.marker_kw(C_BASE), label="mid")[0],
      ax.plot([], [], ls="none", **P.marker_kw(C_STIG), label="stigmatized")[0]]
hs += [ax.plot([], [], ls="none", marker=mk.get(lin, "o"), markerfacecolor="white", markeredgecolor=P.INK, markeredgewidth=.5, label=title.get(lin, lin))[0] for lin in lins]
fig.legend(handles=hs, loc="lower left", ncols=len(hs), handletextpad=0.1, columnspacing=0.9, borderaxespad=0.1)
P.panel_title(ax, "(a) Update diagnostic by variety")
ax = axs[1]
ax.axhline(0, color=P.INK_2, lw=P.LW["axis"], zorder=1); ax.axvline(0, color=P.INK_2, lw=P.LW["axis"], zorder=1)
for lin in lins:
    for v in vs:
        m = R4.get("e4", lin, stage, v, "mb")
        crv = R1.get("e1", lin, "dpo", v, stat_for(R1, "e1", lin, "dpo", v))
        if isnan(m) or isnan(crv):
            continue
        ax.plot(m, crv, ls="none", **P.marker_kw(color_of(v), marker=mk.get(lin, "o"), markersize=4.2), zorder=3)
ax.set_xlabel(r"MB$_v$"); ax.set_ylabel(r"CR$_v$ (DPO)")
P.panel_title(ax, "(b) Update predicts behaviour")
fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.91))
stamp(fig, tag, "exp/results/" + ", ".join(p.name for p in p4))
out = outdir(a) / f"fig4b_mechanism_mb{tag}"
print([str(p) for p in P.save(fig, out)])
