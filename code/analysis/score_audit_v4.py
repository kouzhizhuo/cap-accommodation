#!/usr/bin/env python3
"""Score the V4 human audit (user labels, 84 items) by the pre-declared rules of ANNOTATION_PROTOCOL §7.

Choices fixed here before the first run (lead <agent>, 2026-10-03):
- Who counts: only rows with source == "human" (the user's V4 sheet). By the pooling rule V4 is pooled with V1/V3 items of the
  same type, but V1 holds GPT-6 (Codex) labels (not human; DECISIONS D13 retracted) and V3 is unlabelled, so the pool is V4 alone.
- Rule A instrument: Heylighen-Dewaele F (the formality score E1/G2 use; the GYAFC ranker was never run), computed on the
  TRIMMED texts the annotator saw (primary) and on the full replies (secondary). Pairs the annotator calls "About the same" are
  excluded, as the rule says; exact F ties would be excluded too.
- Rule B: precision = share of human "Yes" among detector-positive Part B (on-request) items; "Unsure" excluded (primary) and
  counted as "No" (sensitivity). Part C (main replies, all detector-positive) has no pre-declared rule and is reported descriptively.
- Agreement statistics: Wilson 95% intervals for shares; Cohen's kappa and Gwet's AC1 with percentile bootstrap intervals over
  items (B = 2000, seed 20261012, STATS_PLAN §8), per item family.
Added AFTER the labels were seen (reported as a sensitivity, never as the decision): rule B restricted to the features the
protocol specifies for Part B ("AAE and SGE retained features", §7) and COMP uses (analysis/comp.py); the builder's two SGE
features (can_cannot, finish_completive) are "descriptive only" in PREREG §9 and enter no COMP or CR estimate.
No LLM labels exist for the V4 items (no judge or reference-rater run; the GPT-6 sheet covers V1 items, 0 overlap), so no
human-model agreement is computed here.
"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
GOLD = EXP / "gold"
OUT = EXP / "results" / "audit" / "audit_v4.json"
Z, B, SEED = 1.959964, 2000, 20261012


def wilson(k, n):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def kappa_ac1(pairs):
    """pairs: list of (x, y) binary labels. Returns (kappa, ac1); None where undefined."""
    n = len(pairs)
    if n == 0:
        return None, None
    pa = sum(x == y for x, y in pairs) / n
    px = sum(x for x, _ in pairs) / n
    py = sum(y for _, y in pairs) / n
    pe_k = px * py + (1 - px) * (1 - py)
    pi = (px + py) / 2
    pe_g = 2 * pi * (1 - pi)
    k = (pa - pe_k) / (1 - pe_k) if pe_k < 1 else None
    a = (pa - pe_g) / (1 - pe_g) if pe_g < 1 else None
    return k, a


def boot(pairs, rng):
    ks, acs = [], []
    n = len(pairs)
    for _ in range(B):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        k, a = kappa_ac1(s)
        if k is not None:
            ks.append(k)
        if a is not None:
            acs.append(a)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else None
    return (q(ks, .025), q(ks, .975)), (q(acs, .025), q(acs, .975))


def share(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else None), "lo": lo, "hi": hi}


def agreement_block(pairs, rng):
    """pairs of (human, instrument) binary labels -> agreement share + kappa/AC1 with CIs."""
    k, a = kappa_ac1(pairs)
    (klo, khi), (alo, ahi) = boot(pairs, rng) if pairs else ((None, None), (None, None))
    out = share(sum(x == y for x, y in pairs), len(pairs))
    out.update({"kappa": k, "kappa_lo": klo, "kappa_hi": khi, "ac1": a, "ac1_lo": alo, "ac1_hi": ahi})
    return out


def main():
    rng = random.Random(SEED)
    meta = json.load(open(GOLD / "HUMAN_AUDIT_V4_meta.json"))
    prov = meta["item_provenance"]
    rows = [json.loads(l) for l in open(GOLD / "HUMAN_AUDIT_V4_LABELS.jsonl")]
    rows = [r for r in rows if r.get("source") == "human"]
    assert len(rows) == meta["n_items"] == 84, len(rows)

    A, Bp, C = [], [], []
    for r in rows:
        p = prov[r["id"]]
        if "tercile" in p:
            A.append((r, p))
        elif p["stage"] == "dpo-request":
            Bp.append((r, p))
        else:
            C.append((r, p))
    assert (len(A), len(Bp), len(C)) == (meta["n_partA"], meta["n_partB"], meta["n_partC"])

    res = {"n_items": len(rows), "n_a": len(A), "n_b": len(Bp), "n_c": len(C)}

    # ---------------- Part A: pairwise formality vs HD-F
    a = {"n_same": 0, "n_nonsame": 0}
    for which in ("trimmed", "full"):
        agree, pairs_b, pick_vern_h, pick_vern_f, ties = [], [], 0, 0, 0
        by_t = defaultdict(lambda: [0, 0, 0])      # tercile -> [n_items, n_same, n_agree]
        by_lin = defaultdict(lambda: [0, 0])       # lineage -> [n_nonsame, n_agree]
        for r, p in A:
            fv, fi = (p["F_vern_trimmed"], p["F_inf_trimmed"]) if which == "trimmed" else (p["F_vern_full"], p["F_inf_full"])
            t = p["tercile"]
            by_t[t][0] += 1
            if r["letter"] == "C":
                by_t[t][1] += 1
                continue
            if fv == fi:
                ties += 1
                continue
            human_vern = (r["letter"] == "A") == p["vern_is_A"]     # True: the annotator picked the vernacular-prompt reply
            f_vern = fv > fi
            ok = human_vern == f_vern
            agree.append(ok)
            pairs_b.append((int(human_vern), int(f_vern)))
            pick_vern_h += human_vern
            pick_vern_f += f_vern
            by_t[t][2] += ok
            by_lin[p["lineage"]][0] += 1
            by_lin[p["lineage"]][1] += ok
        blk = agreement_block(pairs_b, rng)
        blk["ties"] = ties
        blk["validated"] = bool(blk["rate"] is not None and blk["rate"] >= 0.70 and blk["lo"] > 0.50)
        blk["human_picks_vern"] = share(pick_vern_h, len(pairs_b))
        blk["hdf_picks_vern"] = share(pick_vern_f, len(pairs_b))
        blk["by_tercile"] = {f"t{t}": {**share(v[2], v[0] - v[1]), "n_items": v[0], "n_same": v[1], "same_share": v[1] / v[0]}
                             for t, v in sorted(by_t.items())}
        blk["by_lineage"] = {l.lower(): share(v[1], v[0]) for l, v in sorted(by_lin.items())}
        a[which] = blk
    a["n_same"] = sum(r["letter"] == "C" for r, _ in A)
    a["n_nonsame"] = len(A) - a["n_same"]
    a["same_share"] = a["n_same"] / len(A)
    a["rule_validated"] = a["trimmed"]["validated"]          # primary, fixed above
    res["a"] = a

    # ---------------- Part B / C: feature presence vs detectors
    def yes(r):
        return {"A": 1, "B": 0, "C": None}[r["letter"]]          # Yes / No / Unsure

    def feat_block(items, rng, with_kappa=True):
        out, fam = {}, defaultdict(list)
        n_unsure = sum(yes(r) is None for r, _ in items)
        for unsure_as_no in (False, True):
            tag = "unsure_no" if unsure_as_no else "primary"
            pos = [(r, p) for r, p in items if p["detector_positive_trimmed"]]
            neg = [(r, p) for r, p in items if not p["detector_positive_trimmed"]]
            f = lambda lst: [0 if yes(r) is None else yes(r) for r, _ in lst] if unsure_as_no else [yes(r) for r, _ in lst if yes(r) is not None]
            blk = {"precision": share(sum(f(pos)), len(f(pos))), "miss_on_neg": share(sum(f(neg)), len(f(neg)))}
            if with_kappa and neg:
                pairs = [((0 if yes(r) is None else yes(r)) if unsure_as_no else yes(r), int(p["detector_positive_trimmed"]))
                         for r, p in items if unsure_as_no or yes(r) is not None]
                blk["agreement"] = agreement_block(pairs, rng)
            out[tag] = blk
        out["n_unsure"] = n_unsure
        for r, p in items:
            fam[p["feature"]].append((r, p))
        out["by_feature"] = {}
        for ft, lst in sorted(fam.items()):
            pos = [yes(r) for r, p in lst if p["detector_positive_trimmed"] and yes(r) is not None]
            neg = [yes(r) for r, p in lst if not p["detector_positive_trimmed"] and yes(r) is not None]
            out["by_feature"][ft] = {"n": len(lst), "n_det_pos": sum(p["detector_positive_trimmed"] for _, p in lst),
                                     "precision": share(sum(pos), len(pos)), "miss_on_neg": share(sum(neg), len(neg)),
                                     "note_approved": meta["note_status"][ft].startswith("protocol")}
        by_var = defaultdict(list)
        for r, p in items:
            by_var[p["variety"]].append((r, p))
        out["by_variety"] = {}
        for v, lst in sorted(by_var.items()):
            pos = [yes(r) for r, p in lst if p["detector_positive_trimmed"] and yes(r) is not None]
            neg = [yes(r) for r, p in lst if not p["detector_positive_trimmed"] and yes(r) is not None]
            blk = {"n": len(lst), "precision": share(sum(pos), len(pos)), "miss_on_neg": share(sum(neg), len(neg))}
            if with_kappa and neg and pos:
                blk["agreement"] = agreement_block([(yes(r), int(p["detector_positive_trimmed"])) for r, p in lst if yes(r) is not None], rng)
            out["by_variety"][v.lower()] = blk
        return out

    b = feat_block(Bp, rng)
    pr = b["primary"]["precision"]
    b["comp_stands"] = bool(pr["rate"] is not None and pr["rate"] >= 0.80 and pr["lo"] >= 0.60)
    b["correction_factor"] = pr["rate"]                       # human-corrected rate = precision x detector rate (rule B fallback)
    RETAINED = {"ain_t", "completive_done", "finna_gon", "negative_concord", "yall"}   # PREREG §9 AAE morph; SGE morph descriptive only
    rp = [yes(r) for r, p in Bp if p["feature"] in RETAINED and p["detector_positive_trimmed"] and yes(r) is not None]
    b["retained_only"] = {"precision": share(sum(rp), len(rp)), "n_items": sum(p["feature"] in RETAINED for _, p in Bp)}
    b["retained_only"]["comp_stands"] = bool(rp and sum(rp) / len(rp) >= 0.80 and b["retained_only"]["precision"]["lo"] >= 0.60)
    res["b"] = b
    res["c"] = feat_block(C, rng, with_kappa=False)
    allpos = [(r, p) for r, p in Bp + C if p["detector_positive_trimmed"] and yes(r) is not None]
    res["bc_pooled_precision"] = share(sum(yes(r) for r, _ in allpos), len(allpos))
    res["models_compared"] = 0                                # no judge / reference-rater labels on V4 items

    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
