"""Compact Fig. 4 (2026-10-12): change in convergence from first to last stage per model (British spelling, informal register),
x100, from exp/results/derived/fig_models.json (same values as Table 2). Chat-only releases have no first stage and are omitted
here (Table chatonly). Full three-panel version: fig5_models_dumbbell.py (appendix)."""
import json, re
from pathlib import Path
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "exp/results/derived/fig_models.json").read_text())
plt.rcParams.update({"font.size": 6.5, "font.family": "sans-serif"})
BLUE, GREEN, GREY = "#0072B2", "#009E73", "#8C8C8C"
def clean(n):
    n = n.replace("$^{\\S}$", "").replace("$^{\\dagger}$", "")
    m = re.search(r"\\makecell\[l\]\{(.*?)\}", n)
    if m: n = n[:m.start()] + m.group(1).replace("\\\\", "") + n[m.end():]
    return n.replace('T\\"ulu', "Tülu").replace("$\\beta$", "β").strip()
rows = [r for r in R if r["first_stage"] and r["bre_first"] is not None and r["inf_first"] is not None]
n = len(rows); npop = sum(r["group"] == "popular" for r in rows)
fig, ax = plt.subplots(figsize=(3.15, 2.75))
ys = list(range(n))[::-1]
ax.axhspan(n - npop - 0.5, n - 0.5, color=BLUE, alpha=0.06, lw=0)
ax.axvline(0, color=GREY, lw=0.6)
for y, r in zip(ys, rows):
    db, di = r["bre_last"] - r["bre_first"], r["inf_last"] - r["inf_first"]
    ax.plot([min(db, di), max(db, di)], [y, y], color="#DDDDDD", lw=0.8, zorder=1)
    ax.plot(db, y, "D", ms=3.2, color=GREEN, zorder=3)
    ax.plot(di, y, "o", ms=3.2, color=BLUE, zorder=3)
ax.set_yticks(ys, [clean(r["model"]) for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r["group"] == "popular": t.set_fontweight("bold")
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel("Change, first → last stage (×100)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#EEEEEE", lw=0.5); ax.set_axisbelow(True)
ax.text(0.0, 1.015, "◆ British spelling", transform=ax.transAxes, fontsize=6.3, color=GREEN, ha="left", va="bottom")
ax.text(0.42, 1.015, "● informal register", transform=ax.transAxes, fontsize=6.3, color=BLUE, ha="left", va="bottom")
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/figures/fig4_models_change.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
