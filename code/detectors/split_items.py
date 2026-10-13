"""Per-item recognisers for the F_train / F_heldout split (detectors/feature_split_v1.1.json). Counted separately from the feature columns so any split can be
evaluated (lead <agent>). Morph groups reuse the gated recognisers; lexical items are plain word regexes (dictionary/lexicon items; FP risk is reported by the
SAE-reply audit in the E5 export: e.g. 'ion', 'whip', 'mane', 'steady' are ordinary English words)."""
import re
from .base import spans
from . import aae as A

def _w(pattern): return lambda t: spans(pattern, t)

ITEMS = {   # group -> {item name: recogniser}
    "AAE.morph.train": {"finna_gon": A.finna_gon, "completive_done": A.completive_done, "ain_t": A.ain_t},
    "AAE.morph.held": {"negative_concord": A.negative_concord, "yall": A.yall},
    "AAE.lex.train": {"talmbout": _w(r"\btalm?bout\b"), "bruh": _w(r"\bbruh\b"), "tryna": _w(r"\btryna\b"), "yo": _w(r"\byo\b")},
    "AAE.lex.held": {"whip": _w(r"\bwhip\b"), "ion": _w(r"\bion\b"), "fam": _w(r"\bfam\b"), "mane": _w(r"\bmane\b")},
    "SGE.part.train": {"sia": _w(r"\bsia+\b"), "ah": _w(r"(?<=[A-Za-z0-9])\s+ah(?=\s*[?.!,]|\s*$)")},
    "SGE.part.held": {"lah": _w(r"\blah+\b"), "liao": _w(r"\bliao\b")},
    "SGE.lex.train": {"sian": _w(r"\bsian\b"), "siao": _w(r"\bsiao\b"), "walao": _w(r"\bwalao\b")},
    "SGE.lex.held": {"steady": _w(r"\bsteady\b"), "makan": _w(r"\bmakan\b"), "paiseh": _w(r"\bpaiseh\b")},
}
GROUPS = {g: list(d.values()) for g, d in ITEMS.items()}


def spans_of(group, text):
    out = []
    for r in GROUPS[group]: out += r(text)
    return out


def item_spans(group, name, text):
    return ITEMS[group][name](text)
