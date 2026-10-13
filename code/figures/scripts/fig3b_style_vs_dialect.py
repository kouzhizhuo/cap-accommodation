"""Fig 3b — style vs dialect convergence on the SAME replies, per training stage, one column per lineage.
Row 1: CR_v (share of the user's feature density the reply mirrors; productive = novel spans; retained detectors only).
       Prestige cues in green (INF register, BRE/AUS spelling+lexis), stigmatized dialects in orange (AAE, SGE morphosyntax).
Row 2: style matching on the same replies = LSM (function-word style) of replies to the variety minus replies to SAE (style_ai, novel spans).
Bands = 95% bootstrap CI over seeds. Reads exp/results/E1/{summary,style_summary}.json only.
    python3 figures/scripts/fig3b_style_vs_dialect.py [--lineages l1,l2]
"""
import numpy as np
from _common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

a = args(__doc__)
paths, tag = find(a, "E1/summary.json", ["PILOT_P0/summary_L1.json", "PILOT_P0/summary_L2.json"], "SYNTHETIC_e1.json")
R = Res(paths)
sp = RESULTS / "E1" / "style_summary.json"
SD = style_delta(sp) if sp.exists() and not a.fixture else {}
lins = [l for l in a.lineages.split(",") if l in R.tokens("e1", 1)]
nrow = 2 if SD else 1
P.use()
fig, axs = plt.subplots(nrow, len(lins), figsize=P.figsize("text", height=1.55 * nrow + 0.35), squeeze=False, sharex="col")
MK = {"INF": "o", "BRE": "D", "AUS": "v", "AAE": "s", "SGE": "^"}
DASH = {"AAE": "-", "SGE": (0, (3.2, 1.8))}
NAME = {"INF": "INF register", "BRE": "BRE", "AUS": "AUS", "AAE": "AAE", "SGE": "SGE"}


def decollide(ax, ends, frac=0.085):
    """End labels at (or as near as possible to) their final values: iterative pairwise repel around the true y, leader line to the marker."""
    lo_, hi_ = ax.get_ylim(); gap = frac * (hi_ - lo_)
    ends.sort(key=lambda e: e[0]); y0s = np.array([e[0] for e in ends], float); pos = y0s.copy()
    for _ in range(200):
        moved = False
        for q in range(1, len(pos)):
            d = pos[q] - pos[q - 1]
            if d < gap - 1e-9:
                push = (gap - d) / 2; pos[q] += push; pos[q - 1] -= push; moved = True
        if not moved:
            break
    for (y0, lab, xe), yp in zip(ends, pos):
        ax.annotate(lab, (xe, y0), xytext=(xe + 0.25, yp), textcoords="data", va="center", ha="left", fontsize=P.SIZE["legend"], color=P.INK,
                    arrowprops=dict(arrowstyle="-", lw=P.LW["hair"], color=P.MUTED, shrinkA=0, shrinkB=2.5) if abs(yp - y0) > 0.15 * gap else None)


for k, lin in enumerate(lins):
    st = R.stages("e1", lin); x = np.arange(len(st))
    vs = order_varieties([v.upper() for v in R.varieties("e1", lin)])
    for row in range(nrow):
        ax = axs[row, k]; ax.axhline(0, color=P.INK_2, lw=P.LW["axis"], zorder=1); ends = []
        for v in vs:
            if row == 0:
                stat = stat_for(R, "e1", lin, st[0], v)
                ys = np.array([R.triple("e1", lin, s, v, stat) for s in st], float)
            else:
                if v not in {c for (_, _, c) in SD}:
                    continue
                ys = np.array([SD.get((lin, s, v), (NAN, NAN, NAN)) for s in st], float)
            if np.all(np.isnan(ys[:, 0])):
                continue
            c = color_of(v)
            ax.plot(x, ys[:, 0], color=c, ls=DASH.get(v, "-"), **P.marker_kw(c, marker=MK.get(v, "o")), zorder=3)
            P.band(ax, x, ys[:, 1], ys[:, 2], c)
            last = np.where(~np.isnan(ys[:, 0]))[0][-1]
            ends.append([ys[last, 0], NAME.get(v, v), x[last]])
        ax.set_xticks(x, [STAGE_LABEL[s] for s in st]); ax.set_xlim(-0.3, len(st) - 1 + 1.7)
        ax.spines["bottom"].set_bounds(0, len(st) - 1)
        decollide(ax, ends)
        if k == 0:
            ax.set_ylabel([r"CR$_v$ (feature mirroring)", r"$\Delta$LSM (style matching)"][row])
        if row == 0:
            P.panel_title(ax, f"({'abcd'[k]}) {TITLE.get(lin, lin)}")
        if row == nrow - 1:
            ax.set_xlabel("training stage")
fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
stamp(fig, tag)
out = outdir(a) / f"fig3b_style_vs_dialect{tag}"
print([str(p) for p in P.save(fig, out)])
