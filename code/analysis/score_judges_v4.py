#!/usr/bin/env python3
"""Agreement of the two blind LLM judges (gpt-oss-20b, Qwen3-8B) with the V4 human labels and with each other.

Post-audit run (PREREG §10 deviation; judges ran 2026-10-03 on <host>, outputs in _incoming/<host>/results/audit/judges_v4/).
Descriptive only: judge labels never enter a validation rule (rules A/B use the human labels alone, score_audit_v4.py) and no
reported effect uses them. Choices fixed here before the first run (lead <agent>, 2026-10-04):
- Parts as in score_audit_v4.py: A = register pairs (3 options: Reply A / Reply B / About the same), BC = feature presence
  (Parts B and C pooled; Yes / No; "Unsure" excluded, which neither the annotator nor the judges used).
- Exact-option agreement with Wilson 95% intervals; Cohen's kappa and Gwet's AC1 over the option categories (K = 3 for A,
  K = 2 for BC) with percentile bootstrap intervals over items (B = 2000, seed 20261012, as in STATS_PLAN §8).
- Part A also reports direction agreement on the pairs where both raters picked a reply.
"""
import json, math, random
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
GOLD = EXP / "gold"
JDIR = EXP / "_incoming" / "<host>" / "results" / "audit" / "judges_v4"
OUT = EXP / "results" / "audit" / "judges_v4.json"
Z, B, SEED = 1.959964, 2000, 20261012
JUDGES = {"gptoss": "gptoss20b", "qwen": "qwen3_8b"}


def wilson(k, n):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def kappa_ac1(pairs, cats):
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    p1 = Counter(x for x, _ in pairs)
    p2 = Counter(y for _, y in pairs)
    pe_k = sum(p1[c] * p2[c] for c in cats) / n / n
    pe_g = sum(((p1[c] + p2[c]) / (2 * n)) * (1 - (p1[c] + p2[c]) / (2 * n)) for c in cats) / (len(cats) - 1)
    k = (po - pe_k) / (1 - pe_k) if pe_k < 1 else None
    a = (po - pe_g) / (1 - pe_g) if pe_g < 1 else None
    return k, a


def block(pairs, cats, rng):
    n = len(pairs)
    agree = sum(x == y for x, y in pairs)
    lo, hi = wilson(agree, n)
    k, a = kappa_ac1(pairs, cats)
    ks, acs = [], []
    for _ in range(B):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        bk, ba = kappa_ac1(s, cats)
        if bk is not None:
            ks.append(bk)
        if ba is not None:
            acs.append(ba)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else None
    out = {"k": agree, "n": n, "rate": agree / n, "lo": lo, "hi": hi, "kappa": k, "kappa_lo": q(ks, .025),
           "kappa_hi": q(ks, .975), "ac1": a, "ac1_lo": q(acs, .025), "ac1_hi": q(acs, .975)}
    out.update({f"{x}__digits": 2 for x in ("rate", "lo", "hi", "kappa", "kappa_lo", "kappa_hi", "ac1", "ac1_lo", "ac1_hi")})
    return out


def main():
    rng = random.Random(SEED)
    meta = json.load(open(GOLD / "HUMAN_AUDIT_V4_meta.json"))
    prov = meta["item_provenance"]
    human = {r["id"]: r["letter"] for r in map(json.loads, open(GOLD / "HUMAN_AUDIT_V4_LABELS.jsonl")) if r.get("source") == "human"}
    assert len(human) == 84
    lab = {"human": human}
    for tag, f in JUDGES.items():
        rows = [json.loads(l) for l in open(JDIR / f"{f}.jsonl")]
        lab[tag] = {r["id"]: r["letter"] for r in rows}
        assert set(lab[tag]) == set(human), tag
    part = lambda i: "a" if "tercile" in prov[i] else "bc"
    ids = {p: sorted(i for i in human if part(i) == p) for p in ("a", "bc")}
    cats = {"a": ["A", "B", "C"], "bc": ["A", "B"]}
    res = {"n_items": len(human), "n_a": len(ids["a"]), "n_bc": len(ids["bc"]),
           "label_counts": {r: {p: dict(Counter(lab[r][i] for i in ids[p])) for p in ids} for r in lab}}
    for x, y in (("human", "gptoss"), ("human", "qwen"), ("gptoss", "qwen")):
        key = ("h" if x == "human" else x) + "_" + y
        res[key] = {}
        for p in ("a", "bc"):
            pairs = [(lab[x][i], lab[y][i]) for i in ids[p] if lab[x][i] in cats[p] and lab[y][i] in cats[p]]
            res[key][p] = block(pairs, cats[p], rng)
        d = [(lab[x][i], lab[y][i]) for i in ids["a"] if lab[x][i] != "C" and lab[y][i] != "C"]
        k = sum(u == v for u, v in d)
        lo, hi = wilson(k, len(d))
        res[key]["a_dir"] = {"k": k, "n": len(d), "rate": k / len(d) if d else None, "lo": lo, "hi": hi}
        res[key]["a"]["n_same_x"] = sum(lab[x][i] == "C" for i in ids["a"])
        res[key]["a"]["n_same_y"] = sum(lab[y][i] == "C" for i in ids["a"])
    # descriptive: judge "Yes" share on detector-positive on-request (Part B) items; never a validation figure
    bpos = [i for i in ids["bc"] if prov[i]["stage"] == "dpo-request" and prov[i]["detector_positive_trimmed"]]
    res["b_yes_on_detpos"] = {}
    for r in lab:
        k = sum(lab[r][i] == "A" for i in bpos)
        lo, hi = wilson(k, len(bpos))
        res["b_yes_on_detpos"][r] = {"k": k, "n": len(bpos), "rate": k / len(bpos), "lo": lo, "hi": hi}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
