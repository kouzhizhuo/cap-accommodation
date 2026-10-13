#!/usr/bin/env python
"""F_train / F_heldout STEP-1 split (E5 transfer design; lead <agent>). Deterministic, no hand-picking:
units = retained morph features (AAE) and lexical items (AAE lexicon; SGE particles + lexicon, since SGE is carried by CR_lex, PREREG §2.2);
within each (variety, tier) rank units by sha1('P3-split-v1|variety|tier|unit') and alternate F_train, F_heldout in rank order (balance +-1).
F_train builds the DPO pairs; F_heldout is evaluation only. Features not retained (gate/FP audit) are in neither list.
  python -m detectors.make_split --out detectors/feature_split_v1.json"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import re

AAE_MORPH = ["ain_t", "completive_done", "finna_gon", "negative_concord", "yall"]          # gate_pass=true AND FP<=2% (validation_synthetic.json, fp_audit_SAE.json)
LEXD = Path(__file__).resolve().parents[2] / "notes" / "lexicons"


def gated(variety, exclude=()):
    """VARIETY-status items of notes/lexicons/<v>.json (dictionary label AND genre-matched corpus ratio >=5, >=3 authentic hits; D7)."""
    items = [i for i in json.load(open(LEXD / f"{variety}.json")) if i["status"] == "VARIETY" and i["form"] not in exclude]
    return items


_aae = gated("aae", exclude={"finna", "done", "been", "stay"})   # finna is covered by AAE.finna_gon; done/BIN/stay are morphosyntactic (D7)
_sge = gated("sge")
AAE_LEX = [i["form"] for i in _aae]
SGE_PART = [i["form"] for i in _sge if i["type"] == "particle"]
SGE_LEX = [i["form"] for i in _sge if i["type"] != "particle"]
UNITS = {("AAE", "morph"): AAE_MORPH, ("AAE", "lex"): AAE_LEX, ("SGE", "particle"): SGE_PART, ("SGE", "lex"): SGE_LEX}


def split(variety, tier, units):
    ranked = sorted(units, key=lambda u: hashlib.sha1(f"P3-split-v1|{variety}|{tier}|{u}".encode()).hexdigest())
    return {"F_train": [u for i, u in enumerate(ranked) if i % 2 == 0], "F_heldout": [u for i, u in enumerate(ranked) if i % 2 == 1]}


ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = {"version": "step-1 (proposed; PREREG freezes it as step 2 before any E5 run)", "rule": __doc__.split("\n")[1:4], "lexicon_source": "notes/lexicons/{aae,sge}.json, status==VARIETY only (CHECK/dictionary-only items may be used in transforms but are not detector features until they pass P/R + FP audit; D7)",
       "excluded_not_retained": {"AAE": ["copula_absence", "uninflect", "dont_3sg", "those_them", "were_was", "existential_it", "habitual_be(no gold)"],
                                 "SGE": ["copula_absence", "uninflect", "zero_plural", "already", "bare_past", "existential_got", "inverted_indirect", "finish_completive/can_cannot are retained but thin: morph descriptive only"]},
       "splits": {f"{v}.{t}": split(v, t, u) for (v, t), u in UNITS.items()}}
json.dump(out, open(a.out, "w"), indent=1)
for k, x in out["splits"].items():
    print(k, "TRAIN:", x["F_train"], "| HELDOUT:", x["F_heldout"])
