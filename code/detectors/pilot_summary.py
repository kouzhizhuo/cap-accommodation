#!/usr/bin/env python
"""Summary of dialect accommodation per model-stage (v1.4 measurement spec). Env p3mv.

Per cell (model, variety v, filter in {novel, total}):
  D_morph  per REPLY = sum real / sum (real + opp) over RETAINED morph features; a reply is dropped when it has < 3 sites (plan §3.1);
           per seed prompt = mean over its retained replies; `retention` = share of replies kept.  (pooled-per-seed D is kept as `*_pooled`.)
  AI_v = mean_i D_v(r(v,i)) - mean_i D_v(r(SAE,i))         CR_v = AI_v / (P_v(cond) - P_v(SAE)),  P_v = pooled D_v of the PROMPTS
  novel = productive CR (spans in shared >=4-grams and quotes removed); total = no filter.  Both are always emitted.
RETAINED = features that (i) pass the synthetic-gold gate (validation_synthetic.json) and (ii) have <= 2% false positives on SAE-condition
replies at EVERY stage (fp_audit_SAE.json), so a feature that fires on list fragments is excluded automatically.
Bootstrap over seed prompts, B = 2000, RNG seed 0.
  python -m detectors.pilot_summary --scores P0/pilot_scores.parquet --out P0/pilot_summary.json
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
VARIETIES = ["AAE", "SGE", "INF", "BRE", "AUS", "FOR", "NGP", "APP", "SCO", "IND", "IRE"]   # v1 seeds carry US/UK opportunities (seeds_select_v1.py); with v0 stimuli BRE/AUS have no rows and are skipped
MIN_SITES, B, FP_MAX = 3, 2000, 0.02
SUBSET_LEX = {"particle_lah", "particle_lor", "particle_leh", "particle_meh", "particle_hor"}   # already inside `particles`
_RET = None


def _newest_fp():
    c = [Path(x) for x in ("detectors/fp_audit_SAE.json", "../E1/fp_audit_SAE.json") if Path(x).exists()]
    return str(max(c, key=lambda q: q.stat().st_mtime)) if c else "detectors/fp_audit_SAE.json"


def retained_set(val="detectors/validation_synthetic.json", fp=None):
    global _RET
    fp = fp or _newest_fp()
    if _RET is None:
        ret = {k for k, x in json.load(open(val))["features"].items() if x["gate_pass"]}
        flagged = {}
        if Path(fp).exists():
            for m, e in json.load(open(fp))["models"].items():
                for k, x in e["by_feature"].items():
                    if x > FP_MAX:
                        flagged.setdefault(k, []).append(m)
        import detectors.aae, detectors.sge, detectors.inf, detectors.bre, detectors.ext  # noqa: F401
        from detectors.base import FEATURES as _FT
        paired = {f"{v}.{f.id}" for v, fs in _FT.items() for f in fs if f.mv_rules and not f.mv_dialect}   # paired recognisers (INF/BRE/AUS/NGP/FOR): unit-tested, FP audited
        _RET = ({k for k in ret if k not in flagged} | {k for k in paired if k not in flagged or k.split(".")[0] in ("INF", "FOR")}, flagged)
    return _RET


def tier_split(v):
    import detectors.aae, detectors.sge, detectors.inf, detectors.bre, detectors.ext  # noqa: F401
    from detectors.base import FEATURES
    ret, _ = retained_set()
    if v not in FEATURES: return [], []
    morph = [f"{v}.{f.id}" for f in FEATURES[v] if f.tier == "morph" and f.mv_rules and f"{v}.{f.id}" in ret]
    _, flagged = retained_set()   # a lexical/rate feature that fires on > 2% of SAE replies at some stage is excluded too (D7)
    lex = [f"{v}.{f.id}" for f in FEATURES[v] if (f.tier == "lex" or not f.mv_rules) and f.id not in SUBSET_LEX and f"{v}.{f.id}" not in flagged]
    return morph, lex


RATE = {"AAE": ["ain_t", "completive_done", "finna_gon", "negative_concord", "yall", "lexicon", "g_drop", "habitual_be"], "SGE": ["particles", "lexicon"],
        "INF": ["contraction", "casual_lexis"], "BRE": ["spelling", "lexis", "have_got"], "AUS": ["spelling", "aus_lexis"],
        "NGP": ["dey", "don", "na", "no", "fit", "wan", "lexicon"], "APP": ["a_prefix", "fixin_to", "double_modals", "negative_concord", "ain_t", "yall", "reflexive", "lexicon"],
        "FOR": ["lexis"], "SCO": [], "IND": [], "IRE": []}


def rate_seed_means(d, v, novel):
    """per-seed mean of (retained+lexical features per 100 words); features flagged by the SAE-FP audit are excluded; no opportunity normalisation, no gate"""
    _, flagged = retained_set()
    cols = [f"{v}.{f}.{'real_n' if novel else 'real'}" for f in RATE.get(v, []) if f"{v}.{f}" not in flagged and f"{v}.{f}.real" in d.columns]
    r = 100.0 * d[cols].sum(axis=1) / d.n_words.clip(lower=1) if cols else d.n_words * 0.0
    return d.assign(_r=r).groupby(["condition", "seed_id"])._r.mean().unstack(0)


def per_seed(df, v, novel=True, gate=True):
    morph, lex = tier_split(v)
    r, o = ("real_n", "opp_n") if novel else ("real", "opp")
    d = df.copy()
    d["_r"] = d[[f"{m}.{r}" for m in morph]].sum(axis=1); d["_o"] = d[[f"{m}.{o}" for m in morph]].sum(axis=1)
    d["_ok"] = (d._r + d._o) >= (MIN_SITES if gate else 1)
    d["_D"] = np.where(d._ok, d._r / (d._r + d._o).replace(0, np.nan), np.nan)
    d["_lex"] = d[[f"{m}.{r}" for m in lex]].sum(axis=1)
    g = d.groupby(["model", "condition", "seed_id"])
    a = g.agg(D=("_D", "mean"), ret=("_ok", "mean"), r=("_r", "sum"), o=("_o", "sum"), lex=("_lex", "sum"), words=("n_words", "sum"),
              F=("formality", "mean"), copy=("copy_rate", "mean"), n=("n_words", "size")).reset_index()
    a["D_pooled"] = np.where(a.r + a.o >= 1, a.r / (a.r + a.o).replace(0, np.nan), np.nan)
    a["D_lex"] = 100.0 * a.lex / a.words.clip(lower=1)
    return a


def ci(x):
    x = np.asarray(x, float)
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.scores)
    resp, prm = df[df.kind == "response"], df[df.kind == "prompt"]
    rng = np.random.default_rng(0)
    ret, flagged = retained_set()
    import detectors.aae, detectors.sge, detectors.inf, detectors.bre  # noqa: F401
    try:
        import detectors.ext  # noqa: F401  (FOR/NGP/APP/... detectors, when present)
    except ImportError:
        pass
    from detectors.base import FEATURES as _F
    present = set(df.condition)
    VARS = [v for v in VARIETIES if v in present and v in _F]
    out = {"validation_flags": {"NGP": "unit gold only (no synthetic gold)", "FOR": "register control; unit gold only", "APP": "synthetic gold (Multi-VALUE) for morph features; lexis dictionary-only"},
           "spec": "v1.4: gated per-reply D (>=3 sites), retained features only, productive=novel spans, total=no filter, B=2000, seed 0",
           "retained_features": sorted(ret), "fp_flagged_features": flagged, "n_responses": int(len(resp)), "rng_seed": 0, "B": B,
           "min_sites": MIN_SITES, "fp_max": FP_MAX, "models": {}, "prompts": {}}
    for v in VARS:   # prompt-side densities (pooled over the seeds, no gate)
        ps = per_seed(prm, v, novel=False, gate=False)
        out["prompts"][v] = {c: {"D_pooled": float(x.r.sum() / max(1, (x.r + x.o).sum())), "lex_per100w": float(100 * x.lex.sum() / max(1, x.words.sum()))}
                             for c, x in ps.groupby("condition")}
    for model, dm in resp.groupby("model"):
        M = out["models"].setdefault(model, {"scaffold": sorted(dm.scaffold.unique()), "finish_length_share": float((dm.finish == "length").mean()),
                                            "mean_tokens": float(dm.n_tokens.mean()), "varieties": {}})
        for v in VARS:
            if v not in set(dm.condition): continue
            prm_ps = per_seed(prm, v, novel=False, gate=False)
            for novel in (True, False):
                key = "novel" if novel else "total"
                ps = per_seed(dm, v, novel=novel)
                piv = lambda col: ps.pivot(index="seed_id", columns="condition", values=col)
                pD, pL, pF, pC, pR = piv("D"), piv("D_lex"), piv("F"), piv("copy"), piv("ret")
                pp_r = prm_ps.pivot(index="seed_id", columns="condition", values="r"); pp_o = prm_ps.pivot(index="seed_id", columns="condition", values="o")
                pp_l = prm_ps.pivot(index="seed_id", columns="condition", values="D_lex")
                idx = pD.index
                ai = (pD[v] - pD["SAE"]).values; ai_l = (pL[v] - pL["SAE"]).values; dF = (pF[v] - pF["SAE"]).values
                r_c, o_c, r_s, o_s = (pp_r.loc[idx, v].values, pp_o.loc[idx, v].values, pp_r.loc[idx, "SAE"].values, pp_o.loc[idx, "SAE"].values)
                lc, ls = pp_l.loc[idx, v].values, pp_l.loc[idx, "SAE"].values

                def cr_of(sel):
                    P = r_c[sel].sum() / max(1, (r_c[sel] + o_c[sel]).sum()) - r_s[sel].sum() / max(1, (r_s[sel] + o_s[sel]).sum())
                    return np.nanmean(ai[sel]) / P if abs(P) > 1e-9 else np.nan

                def crl_of(sel):
                    P = np.nanmean(lc[sel]) - np.nanmean(ls[sel])
                    return np.nanmean(ai_l[sel]) / P if abs(P) > 1e-9 else np.nan

                allsel = np.arange(len(idx))
                bs = [rng.integers(0, len(idx), len(idx)) for _ in range(B)]
                cr, crl = cr_of(allsel), crl_of(allsel)
                bcr, bcl = [cr_of(s) for s in bs], [crl_of(s) for s in bs]
                P_pool = r_c.sum() / max(1, (r_c + o_c).sum()) - r_s.sum() / max(1, (r_s + o_s).sum())
                Rr, Rp = rate_seed_means(dm, v, novel), rate_seed_means(prm, v, False)
                ridx = Rr.index.intersection(Rp.index)
                rai = (Rr.loc[ridx, v] - Rr.loc[ridx, "SAE"]).values; rpc = (Rp.loc[ridx, v] - Rp.loc[ridx, "SAE"]).values
                rf = lambda sel: np.nanmean(rai[sel]) / np.nanmean(rpc[sel]) if abs(np.nanmean(rpc[sel])) > 1e-9 else np.nan
                rbs = [rf(rng.integers(0, len(ridx), len(ridx))) for _ in range(B)]
                M["varieties"].setdefault(v, {})[key] = {
                    "cr_rate": float(rf(np.arange(len(ridx)))), "cr_rate_lo": ci(rbs)[0], "cr_rate_hi": ci(rbs)[1], "prompt_rate_contrast": float(np.nanmean(rpc)),
                    "cr": float(cr), "cr_lo": ci(bcr)[0], "cr_hi": ci(bcr)[1], "ai": float(np.nanmean(ai)),
                    "ai_lo": ci([np.nanmean(ai[s]) for s in bs])[0], "ai_hi": ci([np.nanmean(ai[s]) for s in bs])[1],
                    "prompt_contrast": float(P_pool), "cr_lex": float(crl), "cr_lex_lo": ci(bcl)[0], "cr_lex_hi": ci(bcl)[1],
                    "d_cond": float(np.nanmean(pD[v])), "d_sae": float(np.nanmean(pD["SAE"])),
                    "lex_cond": float(np.nanmean(pL[v])), "lex_sae": float(np.nanmean(pL["SAE"])),
                    "retention_cond": float(np.nanmean(pR[v])), "retention_sae": float(np.nanmean(pR["SAE"])),
                    "dF": float(np.nanmean(dF)), "dF_lo": ci([np.nanmean(dF[s]) for s in bs])[0], "dF_hi": ci([np.nanmean(dF[s]) for s in bs])[1],
                    "copy": float(np.nanmean(pC[v])), "n_seeds": int(len(idx)), "seeds_without_kept_replies": int(np.isnan(ai).sum())}
    for model, M in out["models"].items():                        # register-referenced quantities and split index S
        for key in ("novel", "total"):
            vs = M["varieties"]
            if "INF" not in vs: continue                      # instruction/persona models carry only SAE + their own condition
            for v in [x for x in vs if x != "INF"]:
                vs[v][key]["dF_minus_inf"] = vs[v][key]["dF"] - vs["INF"][key]["dF"]
                vs[v][key]["cr_minus_inf"] = vs[v][key]["cr"] - vs["INF"][key]["cr"]
            stig = [vs[v][key]["cr"] for v in ("AAE", "SGE", "NGP", "APP", "SCO", "IND") if v in vs]
            pres = [vs[v][key]["cr"] for v in ("INF", "BRE", "AUS") if v in vs]
            M.setdefault("S", {})[key] = float(np.nanmean(pres) - np.nanmean(stig))   # prestige cues {INF, BRE, AUS} minus stigmatized {AAE, SGE}
    json.dump(out, open(a.out, "w"), indent=1)
    for lin in sorted({m.split("@")[0] for m in out["models"]}):
        sub = dict(out); sub["models"] = {m: x for m, x in out["models"].items() if m.startswith(lin + "@")}
        json.dump(sub, open(str(Path(a.out).parent / f"summary_{lin}.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
