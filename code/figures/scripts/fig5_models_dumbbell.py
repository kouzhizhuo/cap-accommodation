import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "exp" / "results" / "derived" / "fig_models.json"
OUT_PDF = ROOT / "paper" / "figures" / "fig5_models_dumbbell.pdf"
OUT_PNG = ROOT / "paper" / "figures" / "fig5_models_dumbbell.png"

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})

POP_COLOR = "#0072B2"
OPEN_COLOR = "#009E73"
ADJ_COLOR = "#E69F00"
POP_BAND = "#0072B2"


def clean_name(name):
    name = name.replace("$^{\\S}$", "").replace("$^{\\dagger}$", "")   # before makecell parsing (braces)
    m = re.search(r"\\makecell\[l\]\{(.*?)\}", name)
    if m:
        inner = m.group(1).replace("\\\\", "")
        name = name[: m.start()] + inner + name[m.end():]
    name = name.replace("$^{\\S}$", "").replace("$^{\\dagger}$", "")
    name = name.replace('T\\"ulu', "Tülu")
    name = name.replace("$\\beta$", "β")
    return name.strip()


with open(DATA) as f:
    records = json.load(f)

pop = [r for r in records if r["group"] == "popular"]
opn = [r for r in records if r["group"] == "open"]
ordered = pop + opn
sep_idx = len(pop)

names = [clean_name(r["model"]) for r in ordered]
n = len(ordered)
ys = np.arange(n)[::-1]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True)

panels = [
    ("bre_first", "bre_last", axes[0], "(a) British spelling"),
    ("inf_first", "inf_last", axes[1], "(b) Informal register"),
]

# compute median first->last drop for panel (b) takeaway
inf_drops = []
for r in ordered:
    first = r.get("inf_first")
    last = r.get("inf_last")
    if first is not None and last is not None:
        inf_drops.append(first - last)
median_drop = np.median(inf_drops) if inf_drops else None

# y-range for popular block band (across all panels)
pop_ys = ys[:sep_idx]
if len(pop_ys) > 0:
    band_top = pop_ys.max() + 0.5
    band_bottom = pop_ys.min() - 0.5

for first_key, last_key, ax, title in panels:
    if len(pop_ys) > 0:
        ax.axhspan(band_bottom, band_top, color=POP_BAND, alpha=0.08, zorder=0)
    for r, y in zip(ordered, ys):
        color = POP_COLOR if r["group"] == "popular" else OPEN_COLOR
        first = r.get(first_key)
        last = r.get(last_key)
        if last is None:
            continue
        if first is not None:
            ax.plot([first, last], [y, y], color="grey", lw=0.8, zorder=1)
            ax.scatter([first], [y], facecolors="white", edgecolors="grey",
                       s=14, zorder=2, linewidths=0.7)
            ax.scatter([last], [y], facecolors=color, edgecolors=color,
                       s=14, zorder=3)
        else:
            ax.scatter([last], [y], facecolors=color, edgecolors=color,
                       s=14, zorder=3)
            ax.text(last, y, " §", fontsize=6, va="center", ha="left",
                    color="grey")
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("Convergence \u00d7100")
    ax.axhline(n - sep_idx - 0.5, color="grey", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="lightgrey", lw=0.4, zorder=0)

# takeaway annotation in panel (b): median first->last drop
if median_drop is not None:
    axb = axes[1]
    xlim = axb.get_xlim()
    x_text = xlim[0] + 0.55 * (xlim[1] - xlim[0])
    y_text = n - 1
    axb.annotate(
        f"median drop {median_drop:.1f}",
        xy=(x_text, y_text), xycoords="data",
        fontsize=6, color="black", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgrey", lw=0.5),
    )

fig.text(0.005, 0.005, "\u25cb first stage\n\u25cf last stage\nbold: popular families", ha="left", va="bottom", fontsize=6)

ax3 = axes[2]
ax3.axhspan(band_bottom, band_top, color=POP_BAND, alpha=0.08, zorder=0) if len(pop_ys) > 0 else None
for r, y in zip(ordered, ys):
    adj = r.get("adj")
    lo = r.get("adj_lo")
    hi = r.get("adj_hi")
    if adj is None:
        continue
    xerr = None
    ci_excludes_zero = True
    if lo is not None and hi is not None:
        xerr = [[adj - lo], [hi - adj]]
        ci_excludes_zero = not (lo <= 0 <= hi)
    facecolor = ADJ_COLOR if ci_excludes_zero else "white"
    ax3.errorbar([adj], [y], xerr=xerr, fmt="none",
                 ecolor=ADJ_COLOR, elinewidth=0.8, capsize=1.5, zorder=2)
    ax3.scatter([adj], [y], facecolors=facecolor, edgecolors=ADJ_COLOR,
                s=16, linewidths=0.8, zorder=3)

ax3.axvline(0, color="grey", ls="--", lw=0.8)
xlim3 = ax3.get_xlim()
if xlim3[1] > 0:
    ax3.axvspan(0, xlim3[1], color=POP_COLOR, alpha=0.06, zorder=0)
    ax3.text(xlim3[1] * 0.98, n - 0.3, "favours prestige", fontsize=5.5,
              color="grey", ha="right", va="top", style="italic")
ax3.set_xlim(xlim3)
ax3.set_title("(c) Update (adjusted)", fontsize=7.5)
ax3.set_xlabel("Prestige \u2212 stigmatized \u00d7100")
ax3.axhline(n - sep_idx - 0.5, color="grey", lw=0.6)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(axis="x", color="lightgrey", lw=0.4, zorder=0)

axes[0].set_yticks(ys)
axes[0].set_yticklabels(names)
for i, lbl in enumerate(axes[0].get_yticklabels()):
    if i < sep_idx:
        lbl.set_fontweight("bold")

# block labels at left margin
if len(pop_ys) > 0:
    fig.text(0.005, (band_top + 0.0) / n * 0.0, "", fontsize=0)  # placeholder no-op
    y_pop_mid = (band_top + band_bottom) / 2.0
    y_open_mid = (ys[sep_idx:].max() + ys[sep_idx:].min()) / 2.0 if len(ys) > sep_idx else None
    fig.canvas.draw()
    trans = axes[0].transData
    disp_pop = trans.transform((0, y_pop_mid))
    fig_pop = fig.transFigure.inverted().transform(disp_pop)
    fig.text(0.005, fig_pop[1], "Popular\nfamilies", fontsize=5.5, color=POP_COLOR,
              ha="left", va="center", rotation=90)
    if y_open_mid is not None:
        disp_open = trans.transform((0, y_open_mid))
        fig_open = fig.transFigure.inverted().transform(disp_open)
        fig.text(0.005, fig_open[1], "Open-recipe\nlineages", fontsize=5.5, color=OPEN_COLOR,
                  ha="left", va="center", rotation=90)

fig.tight_layout(pad=0.3)
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.02, dpi=150)
