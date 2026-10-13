#!/usr/bin/env python
"""Synthetic-gold validation of the detectors (plan §2.3 i): per-feature precision/recall on Multi-VALUE-transformed sentences.

For each Standard English pool sentence s and variety V:  d.update(s); run every rule method once -> d.rules[type][(a,b)] = {"value":..}.
For feature f with opportunity rules R_f:
  positive  = s with ONLY the R_f rule spans applied (gold: >=1 R_f rule fired)      -> recall = P(recogniser fires | positive)
  negatives = (a) s itself where no R_f rule fires (no opportunity)                   -> FP if the recogniser fires
              (b) s with ALL OTHER whitelisted rule spans applied where no R_f fired   -> FP (cross-feature confusion)
Precision is class-balanced (equal numbers of positives and negatives, negatives split (a)/(b)); FPR is reported too.
Gate (lead): P >= 0.85 and R >= 0.70 and n_pos >= 30, else the feature is not retained (unit-tested rate features are exempt: no gold).

  MV_COREF=0 python -m detectors.validate --pool <HOME>/databricks-dolly-15k.jsonl --n 1500 --out detectors/validation_synthetic.json
"""
import argparse, json, random, re, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detectors  # noqa: F401
import detectors.aae, detectors.sge, detectors.ext  # noqa: F401
from detectors.base import FEATURES, _dialect

VARIETIES = {"AAE": "AfricanAmericanVernacular", "SGE": "ColloquialSingaporeDialect", "APP": "AppalachianDialect"}


def pool_sentences(path, n, seed=0):
    rng = random.Random(seed)
    sents = []
    for l in open(path):
        r = json.loads(l)
        for field in ("response", "instruction"):
            for s in re.split(r"(?<=[.?!])\s+", r[field].strip()):
                w = s.split()
                if 6 <= len(w) <= 35 and s.isascii() and s[0].isupper() and s[-1] in ".?!" and not re.search(r"\d{3}|http|\(|\)|\[|\]|;|:", s):
                    sents.append(s)
    rng.shuffle(sents)
    return sents[:n]


def compile_apply(d, s, spans_by_type, types):
    """Apply rule spans of the given types to s (skip overlaps), then the dialect's surface cleanup."""
    items = []
    for t in types:
        for (a, b), v in spans_by_type.get(t, {}).items():
            items.append((a, b, v["value"], t))
    items.sort()
    out, prev, last_end, fired = "", 0, -1, set()
    for a, b, val, t in items:
        if a <= last_end:
            continue
        out += s[prev:a] + val; prev = b + 1; last_end = b; fired.add(t)
    out += s[prev:]
    try:
        out = d.surface_sub(out)
    except Exception:
        pass
    return out, fired


def work(args):
    variety, sents = args
    import mv_env  # noqa: F401
    d = _dialect(VARIETIES[variety])
    feats = [f for f in FEATURES[variety] if f.mv_rules and f.mv_dialect]
    all_types = sorted({r for f in feats for r in f.mv_rules})
    res = {f.id: {"pos": [], "neg_a": [], "neg_b": []} for f in feats}
    for s in sents:
        d.update(s)
        for rt in all_types:
            fn = getattr(d, rt, None)
            if fn:
                try: fn()
                except Exception: pass
        rules = {k: dict(v) for k, v in d.rules.items()}
        for f in feats:
            mine = set(f.mv_rules)
            if any(rules.get(t) for t in mine):
                txt, fired = compile_apply(d, s, rules, sorted(mine))
                if fired and txt != s and (f.gold_filter is None or f.gold_filter(s, txt)):
                    res[f.id]["pos"].append(bool(f.recogniser(txt)))
            else:
                res[f.id]["neg_a"].append(bool(f.recogniser(s)))
                others = [t for t in all_types if t not in mine and rules.get(t)]
                if others:
                    txt, fired = compile_apply(d, s, rules, others)
                    if fired:
                        res[f.id]["neg_b"].append(bool(f.recogniser(txt)))
    return variety, res


def audit_fp(scores_path, val_path, out_path, target=0.02):
    """Per-stage false-positive audit on SAE-condition replies (the prompt has no dialect features, so any hit is a false alarm or
    spontaneous use). Reports, per model-stage: share of replies with >=1 hit of (a) any RETAINED morph feature of the variety,
    (b) each feature separately (retained or not), so the culprits of a high rate are visible. Target for (a): <= 2%."""
    import pandas as pd
    df = pd.read_parquet(scores_path)
    r = df[(df.kind == "response") & (df.condition == "SAE")]
    val = json.load(open(val_path))["features"]
    ret = {k for k, x in val.items() if x["gate_pass"]}
    feats = sorted({c[:-len(".real_n")] for c in r.columns if c.endswith(".real_n")})
    out = {"target": target, "n_per_model": int(r.groupby("model").size().iloc[0]), "models": {}}
    for m, d in r.groupby("model"):
        e = {"any_retained": {}, "by_feature": {}}
        for v in ("AAE", "SGE", "BRE", "AUS", "NGP", "APP", "FOR"):
            cols = [f + ".real_n" for f in feats if f.startswith(v + ".") and (f in ret or v in ("BRE", "AUS", "NGP", "FOR"))]
            e["any_retained"][v] = round(float((d[cols].sum(axis=1) > 0).mean()), 4) if cols else None
        for f in feats:
            if f.split(".")[0] in ("AAE", "SGE", "BRE", "AUS", "NGP", "APP", "FOR"):
                e["by_feature"][f] = round(float((d[f + ".real_n"] > 0).mean()), 4)
        e["pass"] = all(x is None or x <= target for k, x in e["any_retained"].items() if k not in ("FOR",))
        out["models"][m] = e
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"{'model':16s} AAE_ret SGE_ret BRE_all AUS_all pass | worst non-retained features")
    for m, e in out["models"].items():
        worst = sorted(((v, k) for k, v in e["by_feature"].items() if k not in ret and k.split(".")[0] in ("AAE", "SGE")), reverse=True)[:3]
        a_ = e["any_retained"]
        print(f"{m:16s} {a_['AAE']:.3f}   {a_['SGE']:.3f}   {a_['BRE']:.3f}   {a_['AUS']:.3f}  {'ok' if e['pass'] else 'FAIL'}  | " + ", ".join(f"{k}={v:.2f}" for v, k in worst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-scores", default=None, help="run only the SAE-reply false-positive audit on this scores parquet")
    ap.add_argument("--audit-out", default="detectors/fp_audit_SAE.json")
    ap.add_argument("--pool", default=None); ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="detectors/validation_synthetic.json"); ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.audit_scores:
        return audit_fp(a.audit_scores, a.out, a.audit_out)
    sents = pool_sentences(a.pool, a.n)
    out = {"n_pool_sentences": len(sents), "pool": "databricks-dolly-15k (CC BY-SA 3.0), sentences 6-35 words", "features": {}}
    jobs = []
    for v in VARIETIES:
        chunk = max(1, len(sents) // a.workers)
        jobs += [(v, sents[i:i + chunk]) for i in range(0, len(sents), chunk)]
    with Pool(a.workers) as pool:
        parts = pool.map(work, jobs)
    agg = {}
    for v, res in parts:
        for fid, r in res.items():
            e = agg.setdefault((v, fid), {"pos": [], "neg_a": [], "neg_b": []})
            for k in e: e[k] += r[k]
    rng = random.Random(0)
    for (v, fid), e in sorted(agg.items()):
        pos, na, nb = e["pos"], e["neg_a"], e["neg_b"]
        npos = len(pos); tp = sum(pos); fn = npos - tp
        neg = na + nb
        # class-balanced precision: sample npos negatives half from (a), half from (b) where available
        k = min(npos, len(neg))
        sample = rng.sample(na, min(len(na), k // 2 if nb else k)) + rng.sample(nb, min(len(nb), k - (k // 2 if nb else k))) if k else []
        fp_bal = sum(sample); n_bal = len(sample)
        den = tp + fp_bal * (npos / max(1, n_bal))
        prec = tp / den if npos and n_bal and den > 0 else float("nan")
        rec = tp / npos if npos else float("nan")
        fpr = sum(neg) / len(neg) if neg else float("nan")
        gate = bool(npos >= 30 and prec == prec and prec >= 0.85 and rec >= 0.70)
        out["features"][f"{v}.{fid}"] = {"n_pos": npos, "TP": tp, "FN": fn, "n_neg": len(neg), "FP_all": int(sum(neg)), "FPR": round(fpr, 4),
                                         "precision_balanced": round(prec, 3), "recall": round(rec, 3), "gate_pass": gate,
                                         "gate_note": "n_pos<30" if npos < 30 else ""}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'feature':30s} n_pos  P(bal)  R     FPR   gate")
    for k, r in out["features"].items():
        print(f"{k:30s} {r['n_pos']:5d}  {r['precision_balanced']:.2f}   {r['recall']:.2f}  {r['FPR']:.3f}  {'PASS' if r['gate_pass'] else 'fail ' + r['gate_note']}")


if __name__ == "__main__":
    main()
