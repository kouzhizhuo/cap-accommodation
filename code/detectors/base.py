"""Detector core: Feature registry, novel-span filter, Multi-VALUE opportunity extraction, scoring.

A feature f of variety v has
  * recogniser(text) -> [(start, end)] : realised variety variants  (real_f)
  * mv_rules (tuple)                    : Multi-VALUE rule types whose firing on the text = a site where the STANDARD variant
                                          was used but f could apply (opp_f).  Empty => rate feature (no opportunity notion).
  * tier in {"morph", "lex"}
Density (plan §3.1): D_v^morph = sum real_f / sum (real_f + opp_f) over morph features with mv_rules;
D_v^lex = lexical hits per 100 words (rate features).  Spans inside copied >=4-grams / quotes are dropped from real AND opp.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

FEATURES = {}  # variety -> [Feature]


@dataclass
class Feature:
    id: str
    variety: str
    tier: str                     # "morph" | "lex"
    recogniser: Callable
    mv_rules: tuple = ()          # opportunity rule types (Multi-VALUE 0.3 names); () => rate feature
    mv_dialect: str = None        # dialect class whose rule machinery provides the opportunities
    doc: str = ""
    needs_spacy: bool = False
    gold_filter: Callable = None  # (source_sentence, transformed_sentence) -> bool : keep only Multi-VALUE outputs that are valid variety forms


def register(f):
    FEATURES.setdefault(f.variety, []).append(f)
    return f


def spans(pattern, text, flags=re.I):
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]


# ---------------------------------------------------------------- novel-span filter
_TOK = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def _tokens(text):
    return [(m.group(0).lower().replace("’", "'"), m.start(), m.end()) for m in _TOK.finditer(text)]


def copied_ranges(response, prompt, n=4):
    """Char ranges of `response` covered by word n-grams (n>=4) also present in `prompt`, plus quoted spans."""
    r, p = _tokens(response), _tokens(prompt)
    pgrams = {tuple(t[0] for t in p[i:i + n]) for i in range(len(p) - n + 1)}
    covered = [False] * len(r)
    for i in range(len(r) - n + 1):
        if tuple(t[0] for t in r[i:i + n]) in pgrams:
            for j in range(i, i + n):
                covered[j] = True
    rng = []
    for i, (_, s, e) in enumerate(r):
        if covered[i]:
            rng.append((s, e))
    for m in re.finditer(r"[\"“”]([^\"“”]{1,200})[\"“”]", response):   # metalinguistic mention / quotation
        rng.append((m.start(), m.end()))
    copy_rate = sum(covered) / max(1, len(r))
    return rng, copy_rate


def in_ranges(span, ranges):
    s, e = span
    return any(s < re_ and rs < e for rs, re_ in ranges)


# ---------------------------------------------------------------- Multi-VALUE opportunities
_MV = {}


def _dialect(cls):
    if cls not in _MV:
        import mv_env  # noqa: F401  (env shim must be imported first)
        _MV[cls] = mv_env.get_dialect(cls)
    return _MV[cls]


def mv_opportunities(text, dialect_cls, rule_types):
    """{rule_type: [(start, end)]}: sites in `text` where Multi-VALUE's rule would fire (i.e. standard variant used)."""
    d = _dialect(dialect_cls)
    d.update(text)
    for rt in rule_types:
        fn = getattr(d, rt, None)
        if fn is None:
            continue
        try:
            fn()
        except Exception:  # a rule crashing on odd input must not kill scoring; count as no opportunity
            pass
    out = {}
    for rt in rule_types:
        out[rt] = sorted((k[0], k[1] + 1) for k in d.rules.get(rt, {}).keys())
    return out


def score_text(text, prompt="", varieties=("AAE", "SGE", "INF", "BRE", "AUS", "NGP", "APP", "FOR"), use_filter=True):
    """Per-feature real/opp counts (raw and novel) for one text. Returns dict with feature-level rows and totals."""
    import detectors.aae, detectors.sge, detectors.inf, detectors.bre, detectors.ext  # noqa: F401  (registers features)
    rng, copy_rate = copied_ranges(text, prompt) if (use_filter and prompt) else ([], 0.0)
    n_words = len(_tokens(text))
    res = {"n_words": n_words, "copy_rate": copy_rate, "feat": {}, "_ranges": rng}
    for v in varieties:
        from detectors.inf import PAIRED_OPP
        by_dialect = {}
        for f in FEATURES[v]:
            if f.mv_rules and f.mv_dialect:
                by_dialect.setdefault(f.mv_dialect, set()).update(f.mv_rules)
        opps = {cls: mv_opportunities(text, cls, sorted(rules)) for cls, rules in by_dialect.items()} if by_dialect else {}
        for f in FEATURES[v]:
            real = f.recogniser(text)
            if f.mv_rules and not f.mv_dialect:
                opp = [s for r in f.mv_rules for s in PAIRED_OPP[r](text)]
            else:
                opp = [s for r in f.mv_rules for s in opps.get(f.mv_dialect, {}).get(r, [])] if f.mv_rules else []
            real_n = [s for s in real if not in_ranges(s, rng)]
            opp_n = [s for s in opp if not in_ranges(s, rng)]
            res["feat"][f"{v}.{f.id}"] = {"real": len(real), "opp": len(opp), "real_n": len(real_n), "opp_n": len(opp_n),
                                          "tier": f.tier, "has_opp": bool(f.mv_rules)}
    return res


def densities(res, variety, novel=True, min_den=3):
    """(D_morph in [0,1] or None, D_lex hits per 100 words) for one variety from a score_text result."""
    k = ("real_n", "opp_n") if novel else ("real", "opp")
    r = o = lex = 0
    for name, v in res["feat"].items():
        if not name.startswith(variety + "."):
            continue
        if v["tier"] == "morph" and v["has_opp"]:
            r += v[k[0]]; o += v[k[1]]
        elif v["tier"] == "lex" or not v["has_opp"]:
            lex += v[k[0]]
    d_morph = r / (r + o) if (r + o) >= min_den else None
    return d_morph, 100.0 * lex / max(1, res["n_words"])
