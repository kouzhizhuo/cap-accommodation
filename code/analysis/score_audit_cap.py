#!/usr/bin/env python3
"""Score the blind 40-reply CAP naturalness check (D24; gold/build_audit_cap.py; HUMAN_AUDIT_CAP_LABELS.jsonl, imported 2026-10-06).
Pre-declared rule (HUMAN_AUDIT_CAP_meta.json): exaggerated share CAP - UNMATCHED with a Newcombe (hybrid Wilson score) 95% CI;
"Unclear" excluded (primary) and counted as not exaggerated (sensitivity); descriptive diagnostic, never a gate.
Output: results/audit/audit_cap.json (keys audit.cap.*). Lead <agent>."""
import json, math
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; G = EXP / "gold"
Z = 1.959964


def wilson(k, n):
    if n == 0: return (None, None)
    p = k / n; den = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / den; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


meta = json.load(open(G / "HUMAN_AUDIT_CAP_meta.json")); prov = meta["item_provenance"]
lab = {r["id"]: r for r in map(json.loads, open(G / "HUMAN_AUDIT_CAP_LABELS.jsonl")) if r.get("source") == "human"}
assert set(lab) == set(prov) and len(lab) == meta["n_items"] == 40
res = {"n_items": len(lab), "annotator": "one author with doctoral-level training (user attestation, openpyxl entry)"}
for arm in ("CAP", "UNMATCHED"):
    ids = [i for i, p in prov.items() if p["arm"] == arm]
    lbl = [lab[i]["label"] for i in ids]
    known = [x for x in lbl if x != "Unclear"]
    k = sum(x == "Exaggerated" for x in known); lo, hi = wilson(k, len(known))
    res[arm.lower()] = {"n": len(ids), "n_natural": lbl.count("Natural"), "n_exaggerated": lbl.count("Exaggerated"), "n_unclear": lbl.count("Unclear"),
                        "exag_share": k / len(known) if known else None, "exag_lo": lo, "exag_hi": hi,
                        "exag_share__digits": 2, "exag_lo__digits": 2, "exag_hi__digits": 2}
c, u = res["cap"], res["unmatched"]
kc, nc = c["n_exaggerated"], c["n"] - c["n_unclear"]; ku, nu = u["n_exaggerated"], u["n"] - u["n_unclear"]
d, lo, hi = newcombe(kc, nc, ku, nu)
res["diff"] = {"est": d, "lo": lo, "hi": hi, "est__digits": 2, "lo__digits": 2, "hi__digits": 2}
d2, lo2, hi2 = newcombe(kc, c["n"], ku, u["n"])   # sensitivity: Unclear counted as not exaggerated
res["diff_sens"] = {"est": d2, "lo": lo2, "hi": hi2, "est__digits": 2, "lo__digits": 2, "hi__digits": 2}
res["discriminates"] = bool(kc + ku > 0)
res["note"] = ("descriptive diagnostic, never a gate (D24); with no exaggerated reply in either arm the check does not discriminate between "
               "the arms and is not evidence that CAP reduces caricature; over-production evidence stays with the automatic calibration metrics (M2)")
(EXP / "results" / "audit" / "audit_cap.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1))
