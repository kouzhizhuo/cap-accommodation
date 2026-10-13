"""Fig 4 (main §5.2) — added accommodation pairs vs convergence, causal DPO re-run (E5). One column, two panels.
(a) held-out features (rate-based `crheld`, mean over {AAE, SGE}); (b) trained features `ftrain`.
Data: exp/results/E5/summary.json 'runs' (arm -> run -> crheld / ftrain / ...), exp/results/E5/tests.json ('ftrain' effects).
  treatment aa{1k,5k,20k,60k} (T*), matched control ctl{5k,20k,60k} (C*), anti arm anti{1k,5k} (A*), dose-0 d0 (shared start),
  released DPO (1B) = runs.dpo['L1@dpo'] as a horizontal reference.
Points = run means, bars = +-1 std over runs (ddof=1); a dose with a single run is a HOLLOW marker without a bar and is
annotated "1 run" in the treatment colour. Small dots = individual runs. y is symlog (linear below 1e-3) because the 60k arm
sits two orders of magnitude above the others.
Panel (b): if summary.json carries per-run `ftrain` it is plotted exactly like (a); until then the T - C effect of the E5
test file (`tests.json` -> ftrain.effect_*_T_minus_C, prompt-bootstrap 95% CI) is shown instead and the panel says so.
Share-of-training-pairs tick labels are read from exp/results/derived/lead_derived.json (e5dose.share_*k); every plotted number is read from a JSON.
    python3 figures/scripts/fig4_e5_dose.py      -> figures/plots/fig4_e5_dose.{pdf,png}
"""
import os
import time
import numpy as np
from _common import *  # noqa: F401,F403
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

a = args(__doc__)
if a.fixture:
    raise SystemExit("no fixture for E5 dose; use the real file.")
SUM, TESTS = RESULTS / "E5" / "summary.json", RESULTS / "E5" / "tests.json"
for q in (SUM, TESTS):
    print("input", q.name, time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(q))))
D, T = jload(SUM), jload(TESTS)
RUNS = D["runs"]
DOSES = [0, 1, 5, 20, 60]                                       # thousand added pairs
XPOS = {d: i for i, d in enumerate(DOSES)}
_LD = jload(RESULTS / "derived" / "lead_derived.json")["e5dose"]
SHARE = {d: (f"{_LD[f'share_{d}k']:.1f}" if _LD[f"share_{d}k"] < 1 else f"{_LD[f'share_{d}k']:.0f}") for d in (1, 5, 20, 60)}   # % of training pairs (derived/lead_derived.json)
ARMS = {"treat": ("aa", None), "ctl": ("ctl", None), "anti": ("anti", None)}


def runvals(group, key):
    """{dose_k: [value per run]} for a run group like 'aa5k'; skips NaN/None."""
    out = []
    for r, rec in RUNS.get(group, {}).items():
        v = rec.get(key)
        if v is not None and not isnan(v):
            out.append(float(v))
    return out


def arm_series(prefix, key):
    s = {}
    for d in DOSES[1:]:
        vals = runvals(f"{prefix}{d}k", key)
        if vals:
            s[d] = vals
    d0 = runvals("d0", key)
    if d0:
        s[0] = d0
    return dict(sorted(s.items()))


def draw(ax, key, ref_dpo=True):
    handles = {}
    sty = {"treat": (P.ORANGE, "-", "treatment"), "ctl": (P.GREY, (0, (3.2, 1.8)), "matched control"), "anti": (P.VERMILION, (0, (1.2, 1.4)), "anti-arm")}
    for arm, prefix in (("ctl", "ctl"), ("anti", "anti"), ("treat", "aa")):
        s = arm_series(prefix, key)
        if arm != "treat" and 0 in s:                           # the dose-0 point is shared; draw it once in the treatment colour
            s0 = s[0]
        else:
            s0 = None
        c, ls, lab = sty[arm]
        ds = [d for d in s if d != 0 or arm == "treat"]
        if arm != "treat":
            ds = [0] + [d for d in s if d != 0]
            s = dict(s); s[0] = arm_series("aa", key)[0]
        xs = [XPOS[d] for d in ds]; ms = [float(np.mean(s[d])) for d in ds]
        (ln,) = ax.plot(xs, ms, color=c, ls=ls, lw=P.LW["data"] if arm == "treat" else P.LW["thin"], zorder=3 if arm == "treat" else 2)
        handles[arm] = Line2D([], [], color=c, ls=ls, lw=P.LW["data"] if arm == "treat" else P.LW["thin"], label=lab)
        for d in ds:
            v = s[d]; n = len(v); x = XPOS[d]; m = float(np.mean(v))
            if arm != "treat" and d == 0:
                continue
            if n >= 2:
                sd = float(np.std(v, ddof=1))
                ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=c, elinewidth=P.LW["thin"], capsize=1.6, zorder=4)
                ax.plot([x] * n, v, ls="none", marker="o", markersize=1.5, color=P.INK, alpha=.55, zorder=4)
                ax.plot(x, m, ls="none", marker="o", markersize=4.2, markerfacecolor=c, markeredgecolor=P.INK, markeredgewidth=0.5, zorder=5)
            else:
                ax.plot(x, m, ls="none", marker="o", markersize=4.2, markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.1, zorder=5)
                if arm == "treat" and d == max(ds):
                    ax.annotate("1 run", (x, m), xytext=(5, -2), textcoords="offset points", ha="left", va="center", fontsize=P.SIZE["sub"], color=c)
    if ref_dpo and "L1@dpo" in RUNS.get("dpo", {}):
        v = RUNS["dpo"]["L1@dpo"].get(key)
        if v is not None and not isnan(v):
            ax.axhline(v, color=P.INK_2, lw=P.LW["hair"], ls=(0, (4, 2)), zorder=1)
            handles["dpo"] = Line2D([], [], color=P.INK_2, lw=P.LW["hair"], ls=(0, (4, 2)), label="released DPO (1B)")
    return handles


def style_axes(ax):
    ax.set_yscale("symlog", linthresh=1e-3, linscale=0.7)
    ax.set_yticks([0, 1e-3, 1e-2, 1e-1], ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.axhline(0, color=P.INK_2, lw=P.LW["axis"], zorder=0)
    ax.set_xticks(range(len(DOSES)), [("0" if d == 0 else f"{d}") + ("" if d == 0 else "\n" + SHARE[d]) for d in DOSES])
    ax.set_xlim(-0.4, len(DOSES) - 0.1)


P.use()
fig, (axa, axb) = plt.subplots(1, 2, figsize=P.figsize("col", height=2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
hs = draw(axa, "crheld"); style_axes(axa); axa.set_ylim(-3e-4, 0.2)
axa.set_ylabel(r"CR$_v$, held-out features"); P.panel_title(axa, "held-out")
# panel (b)
has_run_ftrain = any("ftrain" in rec for g in RUNS.values() for rec in g.values())
if has_run_ftrain:
    draw(axb, "ftrain"); style_axes(axb); axb.set_ylabel(r"CR$_v$, trained features")
    axb.set_ylim(-3e-4, 12); axb.set_yticks([0, 1e-3, 1e-2, 1e-1, 1, 10], ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "10"])
    P.panel_title(axb, "trained")
else:
    eff = T["ftrain"]
    xs, ms, los, his = [], [], [], []
    for d in (5, 20, 60):
        e = eff[f"effect_{d}k_T_minus_C"]; xs.append(XPOS[d]); ms.append(e["mean"]); los.append(e["lo"]); his.append(e["hi"])
    axb.plot(xs, ms, color=P.ORANGE, lw=P.LW["data"], zorder=3)
    axb.errorbar(xs, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="none", ecolor=P.INK, elinewidth=P.LW["thin"], capsize=1.6, zorder=4)
    n60 = T["ftrain"]["n_runs"].get("T60k", 2)
    for x, m, d in zip(xs, ms, (5, 20, 60)):
        hollow = d == 60 and n60 < 2
        axb.plot(x, m, ls="none", marker="o", markersize=4.2, markerfacecolor="white" if hollow else P.ORANGE, markeredgecolor=P.ORANGE if hollow else P.INK, markeredgewidth=1.1 if hollow else 0.5, zorder=5)
    style_axes(axb); axb.set_ylim(-3e-4, 20); axb.set_yticks([0, 1e-3, 1e-2, 1e-1, 1, 10], ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "10"])
    _ds = [5, 20, 60]; axb.set_xticks([XPOS[d] for d in _ds], [f"{d}\n{SHARE[d]}" for d in _ds]); axb.set_xlim(XPOS[5] - 0.6, XPOS[60] + 0.9)   # no data at 0/1k: drop those positions
    axb.set_ylabel(r"$\Delta$ treatment$-$control, trained")
    P.panel_title(axb, "trained")
    axb.text(0.04, 0.96, "effect, 95% CI", transform=axb.transAxes, ha="left", va="top", fontsize=P.SIZE["sub"], color=P.MUTED, style="italic")
    axb.annotate("1 run", (XPOS[60], ms[-1]), xytext=(5, -2), textcoords="offset points", ha="left", va="center", fontsize=P.SIZE["sub"], color=P.ORANGE)
fig.supxlabel("added pairs, thousands (second row: % of training pairs)", fontsize=P.SIZE["sub"])
fig.legend([hs[k] for k in ("treat", "ctl", "anti", "dpo") if k in hs], [hs[k].get_label() for k in ("treat", "ctl", "anti", "dpo") if k in hs],
           loc="outside upper center", ncols=2, handlelength=1.3, columnspacing=0.7, handletextpad=0.4, fontsize=P.SIZE["legend"])
out = PLOTS / "fig4_e5_dose" if not a.out else Path(a.out) / "fig4_e5_dose"
print([str(p) for p in P.save(fig, out)])
