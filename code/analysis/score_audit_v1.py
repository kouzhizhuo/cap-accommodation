"""Score the returned human audit v1 by the PRE-DECLARED rules (ANNOTATION_PROTOCOL v2 §4-5; STATS_PLAN §8).
Lead <agent>, 2026-10-02. Human labels: exp/gold/HUMAN_AUDIT_LABELS.jsonl (provenance confirmed, DECISIONS D13).
Feature items: detector for THAT item's feature run on the item text. Precision = P(human Yes | detector+),
recall = P(detector+ | human Yes); Wilson 95% CIs; 'Unsure' excluded (sensitivity: Unsure counted as No).
Retained features (used by the paper) are reported separately from features excluded by the G0 FP gate.
Reply items: human 3-way rating vs. 'any retained feature of the user's variety realised in the reply'."""
import json, math, re, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP))
import detectors  # noqa
import detectors.aae as A, detectors.sge as S, detectors.inf as I, detectors.bre as B
from detectors import shared
FEAT = {"copula_absence_aae": shared.copula_absence, "habitual_be": A.habitual_be, "uninflect": shared.uninflect, "ain_t": A.ain_t,
        "negative_concord": A.negative_concord, "completive_done": A.completive_done, "particles": S.particles,
        "copula_absence_sge": shared.copula_absence, "already": S.already,
        "dey": lambda t: re.findall(r"(?i)\bdey\b", t), "don": lambda t: re.findall(r"(?i)\b(?:i|we|dem|he|she|e|you|una|they)\s+don\s+[a-z]+", t),
        "na": lambda t: re.findall(r"(?i)\bna\b", t), "inf": lambda t: I.contraction(t) + I.casual_lexis(t),
        "bre": lambda t: B.uk_spelling(t) + B.uk_lexis(t) + B.have_got(t)}   # identical to gold/build_audit.py
RETAINED = {"ain_t", "negative_concord", "completive_done", "particles", "dey", "don", "na", "inf", "bre"}   # PREREG §9 / step-2 lists
def wilson(k, n, z=1.96):
    if n == 0: return (None, None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(c - h, 3), round(c + h, 3))
G = EXP / "gold"
meta = json.load(open(G / "HUMAN_AUDIT_meta.json")); prov = meta["item_provenance"]
todo = {json.loads(l)["id"]: json.loads(l) for l in open(G / "HUMAN_AUDIT_TODO.jsonl")}
LABELS = sys.argv[1] if len(sys.argv) > 1 else "HUMAN_AUDIT_LABELS.jsonl"   # e.g. LLM_GPT6_AUDIT_LABELS.jsonl
RATER = sys.argv[2] if len(sys.argv) > 2 else "human"                      # e.g. gpt6_reviewed (NEVER called human)
lab = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(G / LABELS)}
out = {"n_labels": len(lab), "rater": RATER, "label_file": LABELS}
rows = []
for i, it in todo.items():
    key = prov[i]["key"]
    if key.startswith("accommodation"): continue
    det = len(FEAT[key](it["text"])) > 0
    rows.append((key, det, lab[i]))
def pr(rows, unsure_as_no=False):
    r = [(k, d, ("No" if (l == "Unsure" and unsure_as_no) else l)) for k, d, l in rows]
    r = [x for x in r if x[2] in ("Yes", "No")]
    tp = sum(d and l == "Yes" for _, d, l in r); fp = sum(d and l == "No" for _, d, l in r); fn = sum((not d) and l == "Yes" for _, d, l in r)
    tn = sum((not d) and l == "No" for _, d, l in r)
    P = wilson(tp, tp + fp); R = wilson(tp, tp + fn)
    return {"n": len(r), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": P[0], "precision_lo": P[1], "precision_hi": P[2],
            "recall": R[0], "recall_lo": R[1], "recall_hi": R[2], "agreement": round((tp + tn) / len(r), 3) if r else None}
out["retained"] = pr([x for x in rows if x[0] in RETAINED]); out["retained_unsure_as_no"] = pr([x for x in rows if x[0] in RETAINED], True)
out["excluded"] = pr([x for x in rows if x[0] not in RETAINED]); out["all_features"] = pr(rows)
out["n_unsure"] = sum(l == "Unsure" for _, _, l in rows)
# reply items: human rating vs detector presence of any retained feature of the user's variety in the reply
pool = {json.loads(l)["pool_id"]: json.loads(l) for l in open(G / "pools" / "accommodation_pool.jsonl")}
VF = {"AAE": ["ain_t", "negative_concord", "completive_done"], "SGE": ["particles"], "INF": ["inf"], "BRE": ["bre"]}
acc = []
for i, it in todo.items():
    key = prov[i]["key"]
    if not key.startswith("accommodation"): continue
    v = key.split(":")[1]; reply = it["text"].split("REPLY:", 1)[1]
    det = any(len(FEAT[f](reply)) > 0 for f in VF[v]); h = lab[i]
    acc.append((v, det, h.startswith("Not at all")))
agree = sum((not d) == none for _, d, none in acc)
out["reply_items"] = {"n": len(acc), "human_not_at_all": sum(n for *_, n in acc), "detector_none": sum(not d for _, d, _ in acc),
                      "agreement": wilson(agree, len(acc))[0], "agreement_lo": wilson(agree, len(acc))[1], "agreement_hi": wilson(agree, len(acc))[2]}
json.dump({RATER: out}, open(EXP / "results" / "audit" / f"audit_v1_{RATER}.json", "w"), indent=1); print(json.dumps(out, indent=1))
