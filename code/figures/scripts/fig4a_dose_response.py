"""Fig 4a — causal DPO re-run: CR of the stigmatized varieties vs dose (EXPERIMENT_PLAN §4.5, STATS_PLAN P5).
Doses with >= 3 training seeds (anchors): filled marker = mean, bar = ± std over seeds, individual seeds as small dots.
Doses with < 3 seeds: HOLLOW marker, single-run value, no error bar (coordinator ruling 2026-09-29).
Treatment = blue (the re-run), matched control = grey dashed, anti-accommodation arm = vermilion dashed.
Fitted slope of (treatment) on dose is drawn as a thin line, its bootstrap CI printed from the file (slope, slope_lo, slope_hi).
Keys (STATS_PLAN §9 scheme): e5.<lineage>.<arm>-d<dose>-s<seed>.<group>.cr ; e5.<lineage>.fit.<group>.{slope,slope_lo,slope_hi,intercept}
NO pilot data exist for E5 yet -> exits until they do.   --unit sets the x-axis label.
"""
import re
from collections import defaultdict
import numpy as np
from _common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

ap_unit = "dose"
if "--unit" in __import__("sys").argv:
    i = __import__("sys").argv.index("--unit"); ap_unit = __import__("sys").argv[i + 1]; del __import__("sys").argv[i:i + 2]
a = args(__doc__)
paths, tag = find(a, "E5/summary.json", None, "SYNTHETIC_e5.json")
R = Res(paths, "e5")
lin = R.tokens("e5", 1)[0]
grp = "stig"
runs = defaultdict(lambda: defaultdict(list))            # arm -> dose -> [values per seed]
pat = re.compile(rf"^e5\.{lin}\.(treat|ctrl|anti)-d([0-9.]+)-s(\d+)\.{grp}\.cr$")
for k, v in R.d.items():
    m = pat.match(k)
    if m and not isnan(v):
        runs[m.group(1)][float(m.group(2))].append(v)
if not runs["treat"]:
    raise SystemExit("no treatment runs in E5 summary; nothing plotted.")
STY = {"treat": (C_OURS, "-", "re-run (treatment)"), "ctrl": (C_BASE, (0, (3.2, 1.8)), "matched control"), "anti": (C_WITHHELD, (0, (3.2, 1.8)), "anti-accommodation")}
P.use()
fig, ax = P.subplots("col", aspect=0.72)
for arm in ("ctrl", "anti", "treat"):
    if not runs[arm]:
        continue
    c, ls, lab = STY[arm]
    ds = sorted(runs[arm]); mean = np.array([np.mean(runs[arm][d]) for d in ds]); n = np.array([len(runs[arm][d]) for d in ds])
    if len(ds) > 1:
        ax.plot(ds, mean, color=c, ls=ls, lw=P.LW["data"] if arm == "treat" else P.LW["thin"], zorder=2, label=lab)
    else:
        ax.plot([], [], color=c, ls=ls, label=lab)
    for d, m_, n_ in zip(ds, mean, n):
        if n_ >= 3:
            sd = np.std(runs[arm][d], ddof=1)
            ax.errorbar(d, m_, yerr=sd, fmt="none", ecolor=P.INK, elinewidth=P.LW["thin"], capsize=1.8, zorder=3)
            ax.plot([d] * n_, runs[arm][d], ls="none", marker="o", markersize=1.6, color=P.INK, zorder=3, alpha=.6)
            ax.plot(d, m_, ls="none", **P.marker_kw(c, markersize=4.6), zorder=4)
        else:
            ax.plot(d, m_, ls="none", marker="o", markersize=4.6, markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.1, zorder=4)
# fitted slope
if R.has("e5", lin, "fit", grp, "slope"):
    b, blo, bhi = R.triple("e5", lin, "fit", grp, "slope"); b0 = R.get("e5", lin, "fit", grp, "intercept")
    xs = np.array([min(runs["treat"]), max(runs["treat"])])
    if not isnan(b0):
        ax.plot(xs, b0 + b * xs, color=C_OURS, lw=P.LW["hair"], alpha=.55, zorder=1)
    ax.text(0.03, 0.88, rf"slope {b:.3f} [{blo:.3f}, {bhi:.3f}]", transform=ax.transAxes, ha="left", va="top", fontsize=P.SIZE["legend"], color=P.INK)
ax.axhline(0, color=P.INK_2, lw=P.LW["axis"], zorder=0)
ax.set_xlabel(ap_unit); ax.set_ylabel(r"CR$_v$, stigmatized varieties")
h, l = ax.get_legend_handles_labels(); h, l = h[::-1], l[::-1]
fig.legend(h, l, loc="lower left", ncols=len(l), handlelength=1.4, columnspacing=0.8, borderaxespad=0.1)
fig.get_layout_engine().set(rect=(0, 0.11, 1, 0.94))
stamp(fig, tag, top=True)
out = outdir(a) / f"fig4a_dose_response{tag}"
print([str(p) for p in P.save(fig, out)])
