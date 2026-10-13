#!/usr/bin/env python
"""Build stimuli: SAE seeds -> conditions {SAE, INF, AAE, SGE} (v0). Env `p3mv` (<host>, CPU).

  MV_COREF=0|1 python build.py --seeds seeds_v0.jsonl --out v0 --conds SAE INF AAE SGE

Morphosyntax: Multi-VALUE 0.3 (`Dialects.<Name>`), applied PER SENTENCE with rule tracking (gold: `executed_rules` -> feature, char span).
Content check (QC) per sentence: negation polarity preserved, question mark preserved, length ratio, no doubled tokens. On failure the
sentence is re-sampled with the offending rule types blacklisted (Multi-VALUE interplay bugs otherwise flip negation, e.g. "am" -> "ain't").
Lexical layer: SGE particles/interjections and AAE 'tryna' (v0 mini-lexicon; the full lexicons come from notes/lexicons in v1).
Everything is deterministic given (seed_id, condition, VERSION). Nothing here is density-calibrated yet (v0)."""
import argparse, json, random, re, sys
from pathlib import Path

import mv_env
from inf_rules import inf_transform

VERSION = "v0.1"
DIALECT = {"AAE": "AfricanAmericanVernacular", "SGE": "ColloquialSingaporeDialect"}
# v0 rule WHITELISTS (eyeballed on the 50 seeds: rules outside these produced ungrammatical / meaning-changing text, e.g. clefting,
# referential_thing "the thing", double_modals "might should", regularized_plurals "companys", one_relativizer "landlord one did").
# Only rules for which we also have realisation detectors are kept, so every applied feature is measurable.
WHITELIST = {
    "AAE": {"drop_copula_be_NP", "drop_copula_be_AP", "drop_copula_be_locative", "drop_aux_be_progressive", "drop_aux_have",
            "negative_concord", "aint_have", "aint_before_main", "dont", "uninflect", "completive_done", "yall", "those_them",
            "finna_future", "future_sub_gon", "existential_it", "null_genitive", "were_was", "conditional_were_was",
            "invariant_tag_non_concord", "regularized_reflexives_aave"},
    "SGE": {"drop_copula_be_NP", "drop_copula_be_AP", "drop_copula_be_locative", "drop_aux_be_progressive", "drop_aux_have",
            "remove_det_definite", "remove_det_indefinite", "zero_plural", "bare_past_tense", "perfect_already", "completive_finish",
            "invariant_tag_can_or_not", "existential_got", "plural_to_singular_human", "reflex_number",
            "never_negator", "uninflect", "present_for_neutral_future"},
}
RATE_SCALE = 0.8  # v0 "calibration-lite": attestation weights x 0.8 (real density calibration to authentic corpora is v1)
GLOBAL_BLACKLIST = {"AAE": set(), "SGE": set()}
PLURAL_RULES = {"zero_plural", "regularized_plurals", "plural_to_singular_human", "mass_noun_plurals"}
PLURAL_STOP = {"his", "this", "its", "is", "was", "has", "us", "yes", "always", "perhaps", "sometimes", "thanks", "news", "series", "analysis", "boss", "class", "process", "access", "business", "stress", "mess", "focus", "less", "unless", "across", "pass", "miss", "guess", "discuss", "address", "dress", "kiss", "loss", "success", "illness", "happiness", "kindness"}
RULE_SRC_ALLOW = {"aint_before_main": {"did", "have", "has", "had"}}  # "ain't" replaces didn't/haven't, not present-tense do/does/am
RISKY = {"zero_plural", "regularized_plurals", "plural_to_singular_human", "aint_be", "aint_have", "aint_before_main", "dont", "negative_concord", "never_negator", "uninflect", "one_relativizer",
         "will_would", "present_for_neutral_future", "drop_aux_yn", "drop_aux_wh", "null_prepositions", "chaining_main_verbs"}
NEG_RE = re.compile(r"\b(?:not|no|never|nothing|nobody|nowhere|none|neither|nor|cannot|ain't|nuh)\b|n't", re.I)
SGE_PARTICLES = [("lah", 0.35), ("lor", 0.15), ("leh", 0.15)]
SGE_Q_PARTICLES = [("ah", 0.25)]
SGE_INTERJ = [("Wah, ", 0.12), ("Aiyo, ", 0.08)]


def split_sents(text):
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def n_neg(s):
    return len(NEG_RE.findall(s))


def qc_ok(src, out):
    if not out.strip():
        return False
    if (n_neg(src) > 0) != (n_neg(out) > 0):
        return False
    if n_neg(src) > 0 and n_neg(out) < n_neg(src):
        return False
    if src.strip().endswith("?") != out.strip().endswith("?"):
        return False
    ls, lo = len(src.split()), len(out.split())
    if not (0.6 * ls <= lo <= 1.6 * ls):
        return False
    if re.search(r"\b(\w+)\s+\1\b", out, re.I):
        return False
    return True


def transform_sentence(dialect, cond, src, rng_key):
    black = set(GLOBAL_BLACKLIST[cond])
    base_rules = dict(dialect.morphosyntax_transforms)
    best = (src, [], 0, sorted(black))
    for attempt in range(1, 9):
        dialect.morphosyntax_transforms = {k: v for k, v in base_rules.items() if k not in black}
        random.seed(f"{rng_key}:{attempt}")
        out = dialect.transform(src)
        rules = [{"feature": v["type"], "tier": "morph", "char": list(k), "src": src[k[0]:k[1] + 1], "out": v["value"]}
                 for k, v in sorted(dialect.executed_rules.items())]
        bad_plural = any(r["feature"] in PLURAL_RULES and re.sub(r"\W", "", r["src"].lower()) in PLURAL_STOP for r in rules)
        bad_src = any(r["feature"] in RULE_SRC_ALLOW and r["src"].strip().lower() not in RULE_SRC_ALLOW[r["feature"]] for r in rules)
        bad_plural = bad_plural or bad_src
        if qc_ok(src, out) and not bad_plural and (rules or attempt >= 3):
            dialect.morphosyntax_transforms = base_rules
            return out, rules, attempt, sorted(black)
        types = {r["feature"] for r in rules}
        black |= (types & RISKY) or types
    dialect.morphosyntax_transforms = base_rules
    return src, [], 8, sorted(black)  # give up: keep the SAE sentence (rare); recorded via attempts=8


def lex_layer(cond, sents, rng):
    feats = []
    out = []
    for i, s in enumerate(sents):
        if cond == "SGE":
            if s.endswith("?"):
                for p, pr in SGE_Q_PARTICLES:
                    if rng.random() < pr:
                        s = s[:-1] + f" {p}?"; feats.append({"feature": f"particle_{p}", "tier": "lex", "sent": i, "out": p}); break
            elif s.endswith("."):
                r, acc = rng.random(), 0.0
                for p, pr in SGE_PARTICLES:
                    acc += pr
                    if r < acc:
                        s = s[:-1] + f" {p}."; feats.append({"feature": f"particle_{p}", "tier": "lex", "sent": i, "out": p}); break
            if i == 0:
                for w, pr in SGE_INTERJ:
                    if rng.random() < pr:
                        s = w + (s if s.startswith("I ") or s.startswith("I'") else s[0].lower() + s[1:]); feats.append({"feature": "interjection", "tier": "lex", "sent": i, "out": w.strip(", ")}); break
        if cond == "AAE" and re.search(r"\btrying to\b", s):
            s = re.sub(r"\btrying to\b", "tryna", s, count=1); feats.append({"feature": "tryna", "tier": "lex", "sent": i, "out": "tryna"})
        out.append(s)
    return out, feats


def _case(src, new):
    return new.capitalize() if src[:1].isupper() else new


def uk_transform(text, cond, seed_id):
    """BRE / AUS: US -> UK spelling + lexis (+ have-got, whilst for BRE; Australian lexis for AUS). v0 mini-lists from detectors/bre.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from detectors import bre
    rng = random.Random(f"{VERSION}:{seed_id}:{cond}")
    feats = []

    def swap(m, table, feat):
        w = m.group(0); new = table.get(w.lower())
        if new is None or rng.random() > 0.9:
            return w
        feats.append({"feature": feat, "tier": "morph", "src": w, "out": _case(w, new)}); return _case(w, new)

    out = re.sub(r"\b[A-Za-z]+\b", lambda m: swap(m, bre.SPELL_US, "spelling"), text)
    lex = {k: v for k, v in bre.LEX_SWAP.items() if k not in {"gas", "fall", "cart", "stove", "while", "among", "resume"}}
    if cond == "AUS":
        lex.update({"apartment": "unit", "apartments": "units", "soccer": "soccer"})
    out = re.sub(r"\b[A-Za-z]+\b", lambda m: swap(m, lex, "lexis"), out)
    if cond == "BRE":
        for pat, rep, feat in [(r"\bI have (?=(?:a|an|no|some|more|two|three|many|the)\b)", "I've got ", "have_got"),
                               (r"\bwe have (?=(?:a|an|no|some|more|two|three|many|the)\b)", "we've got ", "have_got"),
                               (r"\bWhile (?=(?:I|we|you|he|she|they|it|my|the)\b)", "Whilst ", "lexis"),
                               (r"\bwhile (?=(?:I|we|you|he|she|they|it|my|the)\b)", "whilst ", "lexis")]:
            out, n = re.subn(pat, rep, out)
            feats += [{"feature": feat, "tier": "morph", "src": pat, "out": rep}] * n
    if cond == "AUS":
        for pat, rep, feat in [(r"\bI think\b", "I reckon", "aus_lexis"), (r"\ba lot of\b", "heaps of", "aus_lexis"),
                               (r"\bmy friends\b", "my mates", "aus_lexis"), (r"\bmy friend\b", "my mate", "aus_lexis"),
                               (r"\bbreakfast\b", "brekkie", "aus_lexis"), (r"\bbarbecue\b", "barbie", "aus_lexis"),
                               (r"\bvery\b", "heaps", "aus_lexis")]:
            out, n = re.subn(pat, rep, out)
            feats += [{"feature": feat, "tier": "morph", "src": pat, "out": rep}] * n
    return out, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="seeds_v0.jsonl"); ap.add_argument("--out", default="v0")
    ap.add_argument("--conds", nargs="+", default=["SAE", "INF", "AAE", "SGE"])
    a = ap.parse_args()
    seeds = [json.loads(l) for l in open(a.seeds)]
    Path(a.out).mkdir(exist_ok=True)
    import os
    coref = os.environ.get("MV_COREF", "0") == "1"
    dialects = {c: mv_env.get_dialect(DIALECT[c]) for c in a.conds if c in DIALECT}
    for c, d in dialects.items():
        d.morphosyntax_transforms = {k: min(1.0, float(v) * RATE_SCALE) for k, v in d.morphosyntax_transforms.items() if k in WHITELIST[c]}
        print(c, "rules:", sorted(d.morphosyntax_transforms), flush=True)
    stats = {}
    for cond in a.conds:
        rows, gaveup = [], 0
        for sd in seeds:
            feats, attempts, text = [], [], sd["text"]
            if cond == "SAE":
                pass
            elif cond in ("NGP", "APP", "FOR"):
                import ext_rules
                fn = {"NGP": ext_rules.ngp_transform, "APP": ext_rules.app_transform, "FOR": ext_rules.for_transform}[cond]
                text, feats = fn(sd["text"], f"{VERSION}:{sd['seed_id']}:{cond}")
            elif cond in ("BRE", "AUS"):
                text, feats = uk_transform(sd["text"], cond, sd["seed_id"])
            elif cond == "INF":
                text, feats = inf_transform(sd["text"], seed=f"{VERSION}:{sd['seed_id']}")
                feats = [dict(f, tier="morph" if f["feature"] == "contraction" else "lex") for f in feats]
            else:
                sents = split_sents(sd["text"]); outs = []
                for i, s in enumerate(sents):
                    o, r, att, black = transform_sentence(dialects[cond], cond, s, f"{VERSION}:{sd['seed_id']}:{cond}:{i}")
                    outs.append(o); attempts.append(att); gaveup += att == 8
                    feats += [dict(x, sent=i) for x in r]
                rng = random.Random(f"{VERSION}:{sd['seed_id']}:{cond}:lex")
                outs, lf = lex_layer(cond, outs, rng)
                feats += lf; text = " ".join(outs)
            rows.append({"seed_id": sd["seed_id"], "condition": cond, "arm": "S", "text": text, "source_text": sd["text"],
                         "task_type": sd["task_type"], "context": sd["context"], "features_applied": feats,
                         "n_morph": sum(f["tier"] == "morph" for f in feats), "n_lex": sum(f["tier"] == "lex" for f in feats),
                         "qc_attempts": attempts, "density_morph": None, "density_lex": None, "formality": None,
                         "naturalness": None, "build": {"version": VERSION, "coref": coref, "calibrated": False}})
        with open(Path(a.out) / f"{cond}.jsonl", "w") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        stats[cond] = {"n": len(rows), "mean_morph": sum(r["n_morph"] for r in rows) / len(rows),
                       "mean_lex": sum(r["n_lex"] for r in rows) / len(rows), "gave_up_sentences": gaveup,
                       "identical_to_source": sum(r["text"] == r["source_text"] for r in rows)}
        print(cond, stats[cond], flush=True)
    sp = Path(a.out) / "build_stats.json"
    old = json.load(open(sp)) if sp.exists() else {}
    old.update(stats); json.dump(old, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main()
