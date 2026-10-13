"""Writes SYNTHETIC layout-test fixtures (random numbers, NOT results) to figures/scripts/_fixtures/. Used only with --fixture;
output goes to a scratch dir, never to figures/plots. Keys follow STATS_PLAN §9 (<exp>.<lineage>.<stage>.<variety>.<stat>)."""
import json
import numpy as np
from _common import FIXTURES

rng = np.random.default_rng(7)
FIXTURES.mkdir(exist_ok=True)
V = ["inf", "bre", "aus", "ire", "ind", "aae", "sge", "ngp", "sco", "app"]
G = {"aae", "sge", "ngp", "sco", "app"}
e1, e4 = {}, {}
for lin in ("olmo2_1b", "olmo2_7b"):
    for st in ("base", "sft", "dpo", "rlvr", "final"):
        for v in V:
            m = (0.3 if v not in G else 0.15) * (0.5 if st in ("dpo", "rlvr", "final") else 1.0) * (0.2 if v in G and st in ("dpo", "rlvr", "final") else 1) + rng.normal(0, .02)
            e1.update({f"e1.{lin}.{st}.{v}.cr": m, f"e1.{lin}.{st}.{v}.cr_lo": m - .05, f"e1.{lin}.{st}.{v}.cr_hi": m + .05})
    for v in V:
        mb = (1.0 if v not in G else -0.6) + rng.normal(0, .25)
        e4.update({f"e4.{lin}.dpo.{v}.mb": mb, f"e4.{lin}.dpo.{v}.mb_lo": mb - .3, f"e4.{lin}.dpo.{v}.mb_hi": mb + .3})
json.dump(e1, open(FIXTURES / "SYNTHETIC_e1.json", "w"), indent=0)
json.dump(e4, open(FIXTURES / "SYNTHETIC_e4.json", "w"), indent=0)
# E5: flat keys  e5.<lineage>.<arm>-d<dose>-s<seed>.<group>.<stat>  (group: stig | prest)
e5 = {}
doses = [0, 5, 20]; anchor = {0, 20}
for arm, slope in (("treat", 0.008), ("ctrl", 0.0), ("anti", -0.004)):
    for d in doses if arm != "anti" else [5]:
        for s in (1, 2, 3) if (d in anchor and arm == "treat") else ((1,) if arm == "anti" else (1, 2) if arm == "ctrl" and d else (1,)):
            y = 0.02 + slope * d + rng.normal(0, .01)
            e5[f"e5.olmo2_1b.{arm}-d{d}-s{s}.stig.cr"] = y
e5["e5.olmo2_1b.fit.stig.slope"] = 0.008; e5["e5.olmo2_1b.fit.stig.slope_lo"] = 0.005; e5["e5.olmo2_1b.fit.stig.slope_hi"] = 0.011
e5["e5.olmo2_1b.fit.stig.intercept"] = 0.02
json.dump(e5, open(FIXTURES / "SYNTHETIC_e5.json", "w"), indent=0)
print("fixtures written")
