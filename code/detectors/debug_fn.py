"""Dump false-negative examples per feature from the synthetic gold. python -m detectors.debug_fn --pool ... --n 600 --feat AAE.copula_absence,..."""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detectors  # noqa
import detectors.aae, detectors.sge  # noqa
from detectors.base import FEATURES, _dialect
from detectors.validate import pool_sentences, compile_apply, VARIETIES
ap = argparse.ArgumentParser(); ap.add_argument("--pool"); ap.add_argument("--n", type=int, default=600); ap.add_argument("--feat"); ap.add_argument("--k", type=int, default=6)
a = ap.parse_args()
import mv_env  # noqa
sents = pool_sentences(a.pool, a.n)
for key in a.feat.split(","):
    v, fid = key.split(".")
    d = _dialect(VARIETIES[v]); f = [x for x in FEATURES[v] if x.id == fid][0]
    shown = 0
    print("=====", key, f.mv_rules)
    for s in sents:
        d.update(s)
        for rt in f.mv_rules:
            try: getattr(d, rt)()
            except Exception: pass
        rules = {k: dict(x) for k, x in d.rules.items()}
        if any(rules.get(t) for t in f.mv_rules):
            txt, fired = compile_apply(d, s, rules, sorted(f.mv_rules))
            if fired and txt != s and not f.recogniser(txt):
                ex = [(t, s[a_:b+1], v_["value"]) for t in f.mv_rules for (a_, b), v_ in rules.get(t, {}).items()][:3]
                print("  S:", s); print("  T:", txt); print("  rules:", ex); shown += 1
                if shown >= a.k: break
