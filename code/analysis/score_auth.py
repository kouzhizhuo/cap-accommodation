#!/usr/bin/env python3
"""Precision of the WildChat dialect-feature filter (Arm N) from the user's blind AUTH spot-check (20 items).

Decision rule fixed in gold/HUMAN_AUDIT_AUTH_meta.json before labelling: filter precision = share of detector-flagged items with
human "Yes" (Wilson 95% CI); controls expected "No". "Unsure" would be excluded (primary) and counted as "No" (sensitivity); the
sheet has none. Output: results/audit/auth_precision.json (keys audit.auth.*). Lead <agent>, 2026-10-04.
"""
import ast, json, math
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
G = EXP / "gold"
Z = 1.959964


def share(k, n):
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return {"k": k, "n": n, "rate": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h),
            "rate__digits": 3, "lo__digits": 3, "hi__digits": 3}


def main():
    meta = json.load(open(G / "HUMAN_AUDIT_AUTH_meta.json"))
    prov = meta["item_provenance"]
    prov = ast.literal_eval(prov) if isinstance(prov, str) else prov
    lab = {r["id"]: r for r in map(json.loads, open(G / "HUMAN_AUDIT_AUTH_LABELS.jsonl")) if r.get("source") == "human"}
    assert set(lab) == set(prov) and len(lab) == meta["n_items"] == 20
    yes = lambda i: lab[i]["label"] == "Yes"
    known = lambda i: lab[i]["label"] != "Unsure"
    res = {"n_items": len(lab), "n_unsure": sum(not known(i) for i in lab)}
    by = defaultdict(list)
    for i, p in prov.items():
        by[p["set"]].append(i)
    for s, ids in by.items():
        ids_k = [i for i in ids if known(i)]
        res[s.lower()] = share(sum(map(yes, ids_k)), len(ids_k))
    flagged = [i for i, p in prov.items() if p["flagged_by_detector"] and known(i)]
    res["flagged"] = share(sum(map(yes, flagged)), len(flagged))
    res["by_feature"] = {}
    for f in sorted({p["feature"] for p in prov.values()}):
        ids = [i for i, p in prov.items() if p["feature"] == f and p["flagged_by_detector"] and known(i)]
        res["by_feature"][f.replace(".", "_").lower()] = share(sum(map(yes, ids)), len(ids))
    # human-corrected WildChat prevalence = recogniser share x precision (point estimate; bounds use the Wilson bounds only)
    W = json.load(open(EXP / "stimuli" / "armN" / "mining_stats.json"))["prevalence_share"]
    a = W["AAE_bearing(>=1 retained AAE morph hit)"] * 100
    res["wild_aae_corr"] = {"est": a * res["aaen"]["rate"], "lo": a * res["aaen"]["lo"], "hi": a * res["aaen"]["hi"],
                            "est__digits": 2, "lo__digits": 2, "hi__digits": 2}
    res["note"] = "precision of the recogniser filter on WildChat first turns; templated creative-writing prompts and code dominate the rejected hits"
    (EXP / "results" / "audit" / "auth_precision.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_feature"}, indent=1))
    print({k: (v["k"], v["n"]) for k, v in res["by_feature"].items()})


if __name__ == "__main__":
    main()
