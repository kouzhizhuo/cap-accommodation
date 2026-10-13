"""HERO result figure — "Standard Replies, Selective Updates" (P3), text width.
COMPACT (<= 330 pt tall): (a) ten lineages as 2 x 5 small multiples, ONE axis per lineage with the four cues the text uses (BRE rate, INF opp., AAE opp., SGE lexis rate), accommodation index CR_v vs training stage, from
  exp/results/E1/summary.json; ONE estimator per row (lead <agent>, 2026-10-02):
    row 1 "opportunity-based":  INF, AAE, APP   -> `cr`   (open marker = retention < 0.6)
    row 2 "rate-based":         BRE, AUS spelling -> `cr_rate`; SGE lexis -> `cr_lex`   (the opportunity-based spelling variant goes to the appendix)
  prestige cues in green (BRE, AUS, INF), stigmatized in orange (AAE, APP, SGE); bands = 95% CI over seeds.
Row 3: update diagnostic MB_v (DPO vs its SFT reference), exp/results/E4/mbv.json (`dpo_vs_sft`): one dot per lineage with
  95% CI per variety, mean over lineages as a wide bar in the variety colour.
Values are read from the JSON files by key (no typed numbers); input mtimes are printed. PRELIM tag until the lead declares E1/E4 gate-final.
    python3 figures/scripts/hero_standardisation.py --lineages l1,l2,olmo2_13b,l3,l4      -> figures/fig_hero_standardisation.{pdf,png}
"""
import os, sys
import time
import numpy as np
from _common import *  # noqa: F401,F403
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

a = args(__doc__)
p1, tag = find(a, "E1/summary.json", ["PILOT_P0/summary_L1.json", "PILOT_P0/summary_L2.json"], "SYNTHETIC_e1.json")
p4, _ = find(a, "E4/mbv.json", None, "SYNTHETIC_e4.json")
if not a.fixture:
    for extra in ("E1/newlineages/summary.json",):
        if (RESULTS / extra).exists():
            p1 = list(p1) + [RESULTS / extra]                      # OLMo-3.1-32B, Tülu-2 7B/13B, Zephyr-7B-β
    for extra in ("E4/mbv_32b.json", "E4/mbv_new.json"):
        if (RESULTS / extra).exists():
            p4 = list(p4) + [RESULTS / extra]                      # same dpo_vs_sft layout
for q in list(p1) + list(p4):
    print("input", q.name, time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(q))))
R1, R4 = Res(p1), Res(p4, "e4")
# OLMo "instruct" checkpoint = the released RLVR model (Table 1): plot it at the RLVR slot unless the lineage has its own `rlvr` stage (OLMo-2-1B keeps both).
for lin in {k.split(".")[1] for k in R1.d if k.startswith("e1.")}:
    have = {k.split(".")[2] for k in R1.d if k.startswith(f"e1.{lin}.")}
    if "final" in have and "rlvr" not in have:
        for k in [k for k in R1.d if k.startswith(f"e1.{lin}.final.")]:
            R1.d[k.replace(f"e1.{lin}.final.", f"e1.{lin}.rlvr.", 1)] = R1.d.pop(k)
LIN_ALL = ["l1", "l2", "olmo2_13b", "olmo2_32b", "l3", "olmo31_32b", "l4", "tulu2_7b", "tulu2_13b", "zephyr_7b"]
LIN = [l for l in (a.lineages.split(",") if a.lineages != "l1,l2" else LIN_ALL) if l in R1.tokens("e1", 1)]
LABEL = {"l1": "OLMo-2-1B", "l2": "OLMo-2-7B", "olmo2_13b": "OLMo-2-13B", "olmo2_32b": "OLMo-2-32B", "l3": "OLMo-3-7B", "olmo31_32b": "OLMo-3.1-32B",
         "l4": "Tülu-3-8B", "tulu2_7b": "Tülu-2-7B", "tulu2_13b": "Tülu-2-13B", "zephyr_7b": "Zephyr-7B-β"}
VCOL = {"bre": P.GREEN, "aus": "#4DB896", "inf": P.SKY, "aae": P.ORANGE, "app": "#EDB44A", "sge": "#F2C46D"}
VNAME = {"bre": "BRE", "aus": "AUS", "inf": "INF", "aae": "AAE", "app": "APP", "sge": "SGE"}
MK = {"bre": "D", "aus": "v", "inf": "o", "aae": "s", "app": "P", "sge": "^"}
LS = {"aae": "-", "bre": "-", "aus": "-", "sge": (0, (3.2, 1.8)), "app": (0, (5, 1.8, 1, 1.8)), "inf": (0, (1, 1.6))}
CUES = [("bre", "cr_rate"), ("inf", "cr"), ("aae", "cr"), ("sge", "cr_lex")]
STG = ["base", "sft", "dpo", "rlvr"]
xi = {s: i for i, s in enumerate(STG)}
NCOL, YMAX = 5, 0.26
GROUPS = [LIN[i:i + NCOL] for i in range(0, len(LIN), NCOL)]
P.use()
A_ONLY = bool(os.environ.get("HERO_PANEL_A_ONLY"))   # 2026-10-12: main text keeps panel (a); panel (b) -> table tab:contrast
if A_ONLY:
    fig = plt.figure(figsize=(P.TEXT_W, 2.45), constrained_layout=True); gs = fig.add_gridspec(len(GROUPS), NCOL)
else:
    fig = plt.figure(figsize=(P.TEXT_W, 4.55), constrained_layout=True)
    gs = fig.add_gridspec(len(GROUPS) + 1, NCOL, height_ratios=[1, 1, 1.75])
anyhollow = False; first = None; clipped = []
for g, grp in enumerate(GROUPS):
    for j, lin in enumerate(grp):
        ax = fig.add_subplot(gs[g, j], sharey=first); first = first or ax
        ax.axhline(0, color=P.INK_2, lw=P.LW["axis"], zorder=1)
        present = set()
        for v, stat in CUES:
            c = VCOL[v]; pts = []
            for s_ in STG:
                val, lo, hi = R1.triple("e1", lin, s_, v, stat)
                ret = R1.get("e1", lin, s_, v, "retention") if stat == "cr" else 1.0
                if not isnan(val):
                    pts.append((xi[s_], val, (not isnan(ret)) and ret < 0.6)); present.add(s_)
            if not pts:
                continue
            xs_, ys, low = map(np.array, zip(*pts)); ys = ys.astype(float)
            ax.plot(xs_, np.minimum(ys, YMAX - 0.012), color=c, ls=LS[v], lw=P.LW["data"], zorder=3)
            for x_, y_, o_ in zip(xs_, ys, low):
                anyhollow |= bool(o_)
                if y_ > YMAX - 0.012:
                    ax.plot(x_, YMAX - 0.012, ls="none", marker="^", markersize=3.6, markerfacecolor=c, markeredgecolor=P.INK, markeredgewidth=0.45, zorder=5, clip_on=False)
                    ax.text(x_ + 0.18, YMAX - 0.012, f"{y_:.2f}", ha="left", va="center", fontsize=P.SIZE["sub"], color=P.INK, zorder=6); clipped.append((lin, v, s_, float(y_)))
                else:
                    ax.plot(x_, y_, ls="none", marker=MK[v], markersize=3.2, markerfacecolor="white" if o_ else c, markeredgecolor=P.INK, markeredgewidth=0.45, zorder=4)
            # direct end-of-line labels replace the top legend row (critique #1)
            if g == 0 and j == 0 and len(xs_):
                lx = float(xs_[-1]); ly = float(min(ys[-1], YMAX - 0.012))
                ax.annotate(VNAME[v], (lx, ly), xytext=(3.5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=P.SIZE["sub"] - 0.4, color=c, clip_on=False, zorder=6)
        keep = [i for i, s_ in enumerate(STG) if s_ in present]           # no tick (and no "--") where the lineage lacks the stage; positions stay aligned
        ax.set_xticks(keep, [STAGE_LABEL[STG[i]] for i in keep], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.35, len(STG) - 0.35); ax.set_ylim(-0.012, YMAX); ax.set_yticks([0, 0.1, 0.2])
        ax.tick_params(axis="x", labelsize=P.SIZE["sub"]); ax.tick_params(axis="y", labelsize=P.SIZE["sub"])
        ttl = ("(a) " + LABEL.get(lin, lin)) if (g == 0 and j == 0) else LABEL.get(lin, lin)
        P.panel_title(ax, ttl); ax.title.set_fontsize(P.SIZE["sub"] + 1.0)
        if g == 0:
            plt.setp(ax.get_xticklabels(), visible=False)
        if j == 0:
            ax.set_ylabel(r"CR$_v$", fontsize=9.3)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
# small note on the open-marker convention (replaces legend handle)
if anyhollow:
    fig.text(0.992, 0.0, "hollow marker: retention < 0.6", ha="right", va="bottom",
              fontsize=P.SIZE["sub"] - 0.5, color=P.INK_2, style="italic")
# one-line takeaway for the top grid (critique #2)
pass   # takeaway lives in the caption (overlapped panel titles)
if A_ONLY:
    print([str(p) for p in P.save(fig, PAPER / "figures" / "fig_hero_panel_a")]); sys.exit(0)
# ---- panel (b): forest / dumbbell, raw vs adjusted DPO-vs-SFT contrast, 10 lineages ----
ADJ = jload(RESULTS / "derived" / "e4_adjusted.json"); INT = jload(RESULTS / "E4" / "interaction.json")["dpo_vs_sft"]["per_lineage"]
for q in (RESULTS / "derived" / "e4_adjusted.json", RESULTS / "E4" / "interaction.json"):
    print("input", q.name, time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(q))))
rows = [(l, INT[l]["raw"], INT[l]["raw_lo"], INT[l]["raw_hi"], ADJ[l]["adj"], ADJ[l]["adj_lo"], ADJ[l]["adj_hi"]) for l in LIN if l in INT and l in ADJ]
rows.sort(key=lambda r: -r[4])                          # sorted by adjusted contrast, largest on top
ax = fig.add_subplot(gs[-1, :]); ax.axvline(0, color=P.INK_2, lw=P.LW["axis"], zorder=1)
xlo = min(min(r[2], r[5]) for r in rows); xhi = max(max(r[3], r[6]) for r in rows)
pad = 0.08 * (xhi - xlo if xhi > xlo else 1.0)
ax.axvspan(0, xhi + pad, color=P.GREEN, alpha=0.05, zorder=0)
ax.axvspan(xlo - pad, 0, color=P.ORANGE, alpha=0.05, zorder=0)
for k, (l, raw, rlo, rhi, adj, alo, ahi) in enumerate(rows):
    y = len(rows) - 1 - k
    sig = not (alo <= 0 <= ahi)                          # critique #2/#3: fill adjusted marker when its CI excludes 0
    fc = P.BLUE if sig else "white"
    ew = 0.45 if sig else 0.9
    ax.plot([raw, adj], [y, y], color=P.GREY, lw=P.LW["hair"], zorder=2)
    ax.plot([rlo, rhi], [y + 0.16, y + 0.16], color=P.GREY, lw=P.LW["thin"], zorder=2, solid_capstyle="butt")
    ax.plot(raw, y + 0.16, ls="none", marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=P.GREY, markeredgewidth=0.9, zorder=3)
    ax.plot([alo, ahi], [y - 0.16, y - 0.16], color=P.BLUE, lw=P.LW["data"], zorder=3, solid_capstyle="butt")
    ax.plot(adj, y - 0.16, ls="none", marker="o", markersize=3.8, markerfacecolor=fc, markeredgecolor=P.BLUE if not sig else P.INK, markeredgewidth=ew, zorder=4)
    if l == "olmo2_32b":
        fm = lambda v: f"{v:+.3f}".replace("-", "−") if v < 0 else f"{v:.3f}"
        ax.annotate(f"raw {fm(raw)} → adjusted {fm(adj)}", (ahi, y - 0.16), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=P.SIZE["sub"] + 0.5, color=P.INK)
ax.set_yticks(range(len(rows)), [LABEL[r[0]] for r in rows][::-1], fontsize=P.SIZE["sub"] + 0.3); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlim(xlo - pad, xhi + pad)
ax.tick_params(axis="x", labelsize=P.SIZE["tick"]); ax.set_xlabel("prestige − stigmatized matching bonus (nats/token)", fontsize=P.SIZE["sub"] + 1.0)
ax.text(xhi + pad * 0.9, len(rows) - 0.15, "favours prestige", ha="right", va="top", fontsize=P.SIZE["sub"] - 0.3, color=P.GREEN, style="italic")
ax.text(xlo - pad * 0.9, len(rows) - 0.15, "favours stigmatized", ha="left", va="top", fontsize=P.SIZE["sub"] - 0.3, color=P.ORANGE, style="italic")
hl = [Line2D([], [], ls="none", marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=P.GREY, markeredgewidth=0.9, label="raw, 95% CI"),
      Line2D([], [], ls="none", marker="o", markersize=3.8, markerfacecolor=P.BLUE, markeredgecolor=P.INK, markeredgewidth=0.45, label="adjusted, CI excl. 0"),
      Line2D([], [], ls="none", marker="o", markersize=3.8, markerfacecolor="white", markeredgecolor=P.BLUE, markeredgewidth=0.9, label="adjusted, CI incl. 0")]
ax.legend(handles=hl, loc="upper left", ncols=1, fontsize=P.SIZE["legend"], frameon=False, handletextpad=0.3, borderaxespad=0.2)
P.panel_title(ax, "(b) After adjustment, every DPO update favours prestige matching"); ax.title.set_fontsize(P.SIZE["label"])
print("clipped:", clipped)
out = (PAPER / "figures" / "fig_hero_standardisation") if not a.fixture else outdir(a) / "fig_hero_standardisation"
print([str(p) for p in P.save(fig, out)])
