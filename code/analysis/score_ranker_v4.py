#!/usr/bin/env python3
"""Rule A (ANNOTATION_PROTOCOL §7) applied to the GYAFC-trained formality ranker on the V4 Part A pairs.

Post-audit run (PREREG §10 deviation row 2026-10-03, "Post-audit run of the GYAFC ranker"); the coder scored every Part A reply
with s-nlp/roberta-base-formality-ranker (results/audit/ranker_partA.json: A_shown/B_shown = trimmed texts as shown,
vern_full/inf_full = full replies). Scoring mirrors score_audit_v4.py Part A exactly, with the ranker's score in place of HD-F
(lead <agent>, 2026-10-04, fixed before the first run):
- pairs the annotator called "About the same" are excluded, as are exact score ties;
- agreement = the ranker's more formal reply equals the annotator's; Wilson 95% interval; Cohen's kappa and Gwet's AC1 on
  (annotator picks the vernacular-prompt reply, ranker does) with percentile bootstrap (B = 2000, seed 20261012);
- Rule A: agreement >= .70 and Wilson lower bound > .50; primary = the trimmed texts the annotator saw, as for HD-F.
"""
import json, random
from pathlib import Path
import importlib.util

EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sa", EXP / "analysis" / "score_audit_v4.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)                                   # reuse wilson / agreement_block / share (no side effects)
OUT = EXP / "results" / "audit" / "ranker_v4.json"


def main():
    rng = random.Random(sa.SEED)
    meta = json.load(open(sa.GOLD / "HUMAN_AUDIT_V4_meta.json"))
    prov = meta["item_provenance"]
    rows = [json.loads(l) for l in open(sa.GOLD / "HUMAN_AUDIT_V4_LABELS.jsonl")]
    human = {r["id"]: r["letter"] for r in rows if r.get("source") == "human"}
    R = json.load(open(EXP / "results" / "audit" / "ranker_partA.json"))
    A = [i for i in human if "tercile" in prov[i]]
    assert set(A) <= set(R), sorted(set(A) - set(R))
    res = {"n_a": len(A), "n_same": sum(human[i] == "C" for i in A)}
    for which in ("trimmed", "full"):
        pairs, ties = [], 0
        for i in A:
            if human[i] == "C":
                continue
            p, r = prov[i], R[i]
            if which == "trimmed":
                sv, si = (r["A_shown"], r["B_shown"]) if p["vern_is_A"] else (r["B_shown"], r["A_shown"])
            else:
                sv, si = r["vern_full"], r["inf_full"]
            if sv == si:
                ties += 1
                continue
            human_vern = (human[i] == "A") == p["vern_is_A"]
            pairs.append((int(human_vern), int(sv > si)))
        blk = sa.agreement_block(pairs, rng)
        blk["ties"] = ties
        blk["validated"] = bool(blk["rate"] is not None and blk["rate"] >= 0.70 and blk["lo"] > 0.50)
        blk.update({f"{x}__digits": 2 for x in ("rate", "lo", "hi", "kappa", "kappa_lo", "kappa_hi", "ac1", "ac1_lo", "ac1_hi")})
        res[which] = blk
    res["rule_validated"] = res["trimmed"]["validated"]
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
