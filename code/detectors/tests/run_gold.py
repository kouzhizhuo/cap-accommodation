"""Run the author gold cases: per-feature TP/FN/FP/TN on sentences (feature fires >=1 time). Usage: python -m detectors.tests.run_gold [--min-p .85 --min-r .7]"""
import json, sys
sys.path.insert(0, __file__.rsplit("/detectors/", 1)[0])
import detectors  # noqa: F401
import detectors.aae, detectors.sge, detectors.inf, detectors.bre, detectors.ext  # noqa: F401
from detectors.base import FEATURES
from detectors.tests.gold_cases import G

byid = {f"{v}.{f.id}": f for v, fs in FEATURES.items() for f in fs}
rows, fails = [], []
for key, (pos, neg) in G.items():
    f = byid[key]
    tp = sum(bool(f.recogniser(t)) for t in pos); fn = len(pos) - tp
    fp = sum(bool(f.recogniser(t)) for t in neg); tn = len(neg) - fp
    for t in pos:
        if not f.recogniser(t): fails.append(("FN", key, t))
    for t in neg:
        if f.recogniser(t): fails.append(("FP", key, t))
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn)
    rows.append({"feature": key, "n_pos": len(pos), "n_neg": len(neg), "TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": round(p, 3), "recall": round(r, 3)})
for r in rows:
    print(f"{r['feature']:28s} P={r['precision']:.2f} R={r['recall']:.2f}  (TP{r['TP']} FN{r['FN']} FP{r['FP']} TN{r['TN']})")
print()
for k, key, t in fails:
    print(k, key, "|", t)
json.dump({"rows": rows, "failures": fails}, open("detectors/tests/gold_results.json", "w"), indent=1)
