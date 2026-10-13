#!/usr/bin/env python
"""Fig. 3a (pilot): heat-map of productive CR_v by stage, one panel per lineage, rows in prestige order (INF > AAE > SGE tonight).
  python analysis/plot_pilot.py --summary P0/pilot_summary.json --style <dir with palette.py> --out results/P0/plots/P3_fig3a_pilot
Marked pilot: v0 stimuli (50 seeds), detector v0.x, no claim."""
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--summary"); ap.add_argument("--style"); ap.add_argument("--out"); a = ap.parse_args()
sys.path.insert(0, a.style)
import numpy as np, matplotlib
matplotlib.use("Agg")
import palette as P
import matplotlib.pyplot as plt
P.use()
S = json.load(open(a.summary))["models"]
ROWS = ["INF", "AAE", "SGE"]
lineages = sorted({m.split("@")[0] for m in S})
STAGES = ["base", "base-F1", "base-raw", "sft", "dpo", "rlvr1", "instruct"]
fig, axes = plt.subplots(1, len(lineages), figsize=(P.TEXT_W, 1.9), squeeze=False)
for ax, lin in zip(axes[0], lineages):
    st = [s for s in STAGES if f"{lin}@{s}" in S]
    Z = np.array([[S[f"{lin}@{s}"]["varieties"][v]["novel"]["cr"] for s in st] for v in ROWS], float)
    im = ax.imshow(Z, cmap="arr_div", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st, rotation=35, ha="right"); ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, "" if np.isnan(Z[i, j]) else f"{Z[i, j]:+.2f}", ha="center", va="center", fontsize=P.SIZE["sub"], color=P.INK)
    ax.set_title(f"{lin} (pilot)", fontsize=P.SIZE["pill"])
fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="CR (productive)")
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=300)
open(str(out) + ".txt", "w").write(f"script: exp/analysis/plot_pilot.py ; data: {a.summary} ; pilot (v0 stimuli), no claim\n")
print("wrote", out)
