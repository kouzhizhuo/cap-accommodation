#!/usr/bin/env python3
"""Which recogniser hits make up the AAE and Singlish signal in model REPLIES (E1/scores.parquet, novel spans: *.real_n).

Answers the coordinator's check (2026-10-04) after the WildChat spot-check (precision 3/18 on raw user requests): do the
low-precision features there (AAE multiple negation, Singlish particles) also drive the main reply measurements?
Aligned = SFT/DPO/RLVR/Instruct/final stages; base = every base framing. Retained AAE morph features only.
Output: results/derived/reply_hits.json (keys rh.*). Lead <agent>, 2026-10-04.
"""
import json
from pathlib import Path
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
F = ["ain_t", "completive_done", "finna_gon", "negative_concord", "yall"]
cols = ["condition", "kind", "stage"] + [f"AAE.{f}.real_n" for f in F] + ["SGE.particles.real_n", "SGE.lexicon.real_n"]
d = pd.read_parquet(EXP / "E1" / "scores.parquet", columns=cols)
r = d[d.kind == "response"]
groups = {"aligned": r[r.stage.isin(["sft", "dpo", "instruct", "rlvr1", "final"])], "base": r[r.stage.str.startswith("base")]}
out = {}
for g, sub in groups.items():
    o = out.setdefault(g, {"n_replies": int((sub.condition == "AAE").sum())})
    for c in ("AAE", "SAE", "SGE"):
        s = sub[sub.condition == c]
        hits = {f: int(s[f"AAE.{f}.real_n"].sum()) for f in F}
        tot = sum(hits.values())
        o[c.lower()] = {"aae_hits": tot, "nc_share": (hits["negative_concord"] / tot) if tot else None,
                        "yall_aint_share": ((hits["yall"] + hits["ain_t"]) / tot) if tot else None,
                        "sge_particles": int(s["SGE.particles.real_n"].sum()), "sge_lexicon": int(s["SGE.lexicon.real_n"].sum()),
                        "nc_share__digits": 2, "yall_aint_share__digits": 2,
                        "nc_pct": round(100 * hits["negative_concord"] / tot) if tot else None}
(EXP / "results" / "derived" / "reply_hits.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
