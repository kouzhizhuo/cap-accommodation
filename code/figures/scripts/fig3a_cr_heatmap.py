"""Fig 3a — accommodation index CR_v by variety x stage (heat-map, diverging map centred at 0), one panel per lineage.
Rows = variety (retained-feature CR, productive = novel spans) and, below it, the same variety's lexical-item CR; ordered prestige ->
stigmatized (PREREG §2.2). Cell text = CR_v; bold when the 95% CI excludes 0. Hatched cells: undefined (zero contrast) or, for the
first row of a variety, opportunity-based retention < 0.6 (G0(c) fails: no locus claim rests on that cell). Reads exp/results only.
    python3 figures/scripts/fig3a_cr_heatmap.py [--lineages l1,l2]
"""
import numpy as np
from matplotlib.patches import Rectangle
from _common import *  # noqa: F401,F403
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# converge = green (prestige/style role), diverge = vermilion; blue is reserved for the contribution (FIGURE_STANDARD §5.3, QA 2026-10-01)
CMAP = LinearSegmentedColormap.from_list('cr_gv', [P.VERMILION, '#EDB895', '#F2F2F2', '#8FD1B9', P.GREEN])

a = args(__doc__)
paths, tag = find(a, "E1/summary.json", ["PILOT_P0/summary_L1.json", "PILOT_P0/summary_L2.json"], "SYNTHETIC_e1.json")
R = Res(paths)
lins = [l for l in a.lineages.split(",") if l in R.tokens("e1", 1)]
P.use()
fig, axs = plt.subplots(1, len(lins), figsize=P.figsize("text", height=2.35), squeeze=False,
                        gridspec_kw={"width_ratios": [len(R.stages("e1", l)) for l in lins]})
allv = [abs(R.get("e1", l, s, v, stat_for(R, "e1", l, s, v) if t == "cr" else "cr_lex"))
        for l in lins for s in R.stages("e1", l) for v in R.varieties("e1", l) for t in ("cr", "lex")]
vmax = float(np.ceil(max(0.02, np.nanmax(allv)) * 20) / 20)
for ax, lin, letter in zip(axs[0], lins, "abcd"):
    st = R.stages("e1", lin)
    vs = order_varieties([v.upper() for v in R.varieties("e1", lin)])
    rows = []
    for v in vs:
        rows.append((v, "prim", stat_for(R, "e1", lin, st[0], v)))
        rows.append((v, "lex", "cr_lex"))
    rows = [r for r in rows if any(not isnan(R.get("e1", lin, s_, r[0], r[2])) for s_ in st)]      # drop all-undefined rows
    M = np.full((len(rows), len(st)), np.nan); SIG = np.zeros_like(M, bool); LOWRET = np.zeros_like(M, bool)
    for i, (v, kind, stat) in enumerate(rows):
        for j, s in enumerate(st):
            val, lo, hi = R.triple("e1", lin, s, v, stat)
            M[i, j] = val; SIG[i, j] = (not isnan(lo)) and (lo > 0 or hi < 0)
            ret = R.get("e1", lin, s, v, "retention")
            LOWRET[i, j] = kind == "prim" and (not isnan(ret)) and ret < 0.6
    ax.imshow(np.where(np.isnan(M), 0, M), cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(st)):
            if np.isnan(M[i, j]) or LOWRET[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=P.GREY_LIGHT, hatch="////", lw=0))
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, (f"{M[i, j]:.2f}".replace("-", "−") if abs(M[i, j]) >= 0.005 else "0.00"), ha="center", va="center",
                    fontsize=P.SIZE["sub"] + .5, fontweight="bold" if SIG[i, j] else "normal", color=P.INK)
    ax.set_xticks(range(len(st)), [STAGE_LABEL[s] for s in st])
    ax.set_yticks(range(len(rows)), [v if k == "prim" else f"{v} lexis" for v, k, _ in rows])
    for i in range(1, len(rows)):
        if rows[i][0] != rows[i - 1][0]:
            ax.axhline(i - .5, color="white", lw=1.6)
    for i, (v, k, _) in enumerate(rows):
        ax.add_patch(Rectangle((-0.62, i - .5), .09, 1, transform=ax.transData, clip_on=False, fc=color_of(v), ec="none"))
    ax.set_xlim(-.5, len(st) - .5)
    ax.tick_params(length=0); ax.tick_params(axis="y", pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    P.panel_title(ax, f"({letter}) {TITLE.get(lin, lin)}")
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vmax, vmax))
cb = fig.colorbar(sm, ax=list(axs[0]), fraction=0.025, pad=0.02)
cb.outline.set_visible(False); cb.set_label(r"CR$_v$", fontsize=P.SIZE["label"]); cb.ax.tick_params(length=2, width=P.LW["hair"])
fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
fig.text(0.0, 0.0, 'Bold: 95% CI excludes 0 · hatched: undefined or opportunity-based retention < 0.6', ha='left', va='bottom', fontsize=P.SIZE['sub'], color=P.MUTED, style='italic')
stamp(fig, tag)
out = outdir(a) / f"fig3a_cr_heatmap{tag}"
print([str(p) for p in P.save(fig, out)])
