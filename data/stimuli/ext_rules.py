"""Stimulus transforms for the step-2 varieties (v1): NGP (Nigerian Pidgin rule set), APP (Appalachian: Multi-VALUE whitelist + dictionary lexis), FOR (formal register).
Each returns (text, features_applied). Deterministic given `key`. Sources: notes/lexicons/ngp.json (26 VARIETY items), ngp_grammar.md (eWAVE A/B + UD_Naija function profiles),
Multi-VALUE 0.3 AppalachianDialect rules, notes/lexicons/app.json (dictionary-only items: used in transforms, NOT detector features until they pass P/R + FP audit; D7).
Env p3mv (spaCy for lemma/tag, Multi-VALUE for APP)."""
import random, re
import spacy
import build  # qc_ok, transform_sentence, DIALECT, RATE_SCALE

_NLP = None
def nlp():
    global _NLP
    if _NLP is None: _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _sents(text): return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


# ------------------------------------------------------------------ NGP
PRON = {"he": "e", "she": "e", "it": "e", "they": "dem", "them": "dem", "their": "dem", "his": "im", "her": "im", "him": "am"}
LEXSWAP = [(r"\bplease\b", "abeg", "abeg"), (r"\bboss\b", "oga", "oga"), (r"\bproblems?\b", "wahala", "wahala"), (r"\bkids\b", "pikin dem", "pikin"), (r"\bchild\b", "pikin", "pikin"),
           (r"\bwhat\b", "wetin", "wetin"), (r"\bthis\b(?= [a-z])", "dis", "dis"), (r"\bpeople\b", "pipo", "pipo"), (r"\btalk(?:ing)?\b", "tok", "tok")]


def ngp_sentence(s, rng):
    doc = nlp()(s); feats = []; out = []; skip = 0
    toks = list(doc)
    for i, t in enumerate(toks):
        if skip: skip -= 1; continue
        lw = t.lower_; nxt = toks[i + 1] if i + 1 < len(toks) else None
        subj = lw in {"i", "you", "we", "they", "he", "she", "it"} or t.pos_ in {"NOUN", "PROPN"}
        # progressive: be + V-ing  -> dey + V
        if t.lemma_ == "be" and t.pos_ == "AUX" and t.dep_ in {"aux"} and t.head.tag_ == "VBG" and t.head.i > i and t.head.i == i + 1 and lw in {"am", "is", "are"}:
            out.append("dey"); out.append(t.head.lemma_); feats.append("dey"); skip = 1; continue
        # perfect: have/has + VBN -> don + V
        if lw in {"have", "has"} and t.dep_ == "aux" and t.head.tag_ == "VBN" and t.head.i == i + 1 and t.head.lemma_ not in {"be"}:
            out.append("don"); out.append(t.head.lemma_); feats.append("don"); skip = 1; continue
        # negation: do/does/did not V -> no V ; cannot -> no fit
        if lw in {"do", "does", "did"} and nxt is not None and nxt.lower_ == "not" and t.dep_ == "aux":
            out.append("no"); feats.append("no_neg"); skip = 1; continue
        if lw == "cannot": out.append("no fit"); feats.append("fit"); continue
        if lw == "can" and nxt is not None and nxt.lower_ == "not": out.append("no fit"); feats.append("fit"); skip = 1; continue
        if lw == "can" and t.dep_ == "aux": out.append("fit"); feats.append("fit"); continue
        if lw == "will" and t.dep_ == "aux": out.append("go"); feats.append("go"); continue
        if lw == "want" and nxt is not None and nxt.lower_ == "to": out.append("wan"); feats.append("wan"); skip = 1; continue
        if lw == "going" and nxt is not None and nxt.lower_ == "to" and t.dep_ in {"aux", "ROOT", "xcomp"} and i + 2 < len(toks) and toks[i + 2].pos_ == "VERB":
            out.append("go"); feats.append("go"); skip = 1; continue
        # copula before NP -> na ; before ADJ -> dropped
        if lw in {"is", "are", "am", "was"} and t.dep_ in {"ROOT", "cop"}:
            comp = next((c for c in t.children if c.dep_ in {"attr", "acomp"}), None) or (t.head if t.dep_ == "cop" else None)
            if comp is not None and comp.pos_ in {"NOUN", "PROPN"}: out.append("na"); feats.append("na"); continue
            if comp is not None and comp.pos_ == "ADJ": feats.append("cop_drop"); continue
        if lw == "know" and t.pos_ == "VERB" and t.tag_ == "VBP": out.append("sabi"); feats.append("sabi"); continue
        if lw in PRON and (t.dep_ in {"nsubj", "poss", "dobj", "pobj", "nsubjpass", "dative"}) and lw != "it":
            new = PRON[lw]
            if new == "am" and t.dep_ not in {"dobj", "pobj", "dative"}: new = "e"
            if lw in {"her"} and t.dep_ == "dobj": new = "am"
            out.append(new); feats.append("dem" if new == "dem" else "e_im"); continue
        # present 3sg -s on lexical verbs: "he says" -> "e talk/say": use lemma
        if t.tag_ == "VBZ" and t.pos_ == "VERB" and any(c.dep_ == "nsubj" for c in t.children): out.append(t.lemma_); feats.append("no_infl"); continue
        out.append(t.text)
        if nxt is None or not (nxt.is_punct and nxt.i == i + 1 and False): pass
    text = " ".join(out)
    text = re.sub(r"\bnot\b", "no", text)          # negator left behind by a dropped copula ("I not sure" -> "I no sure")
    text = re.sub(r"\s+([,.?!;:])", r"\1", text); text = re.sub(r"\s+'", "'", text)
    for pat, rep, f in LEXSWAP:
        text, n = re.subn(pat, rep, text, flags=re.I); feats += [f] * n
    if text.endswith("?") and rng.random() < 0.15: text = text[:-1] + " abi?"; feats.append("abi")
    elif text.endswith(".") and rng.random() < 0.10: text = text[:-1] + " o."; feats.append("o")
    return text[0].upper() + text[1:] if text else text, feats


def ngp_transform(text, key):
    rng = random.Random(key); outs, feats = [], []
    for s in _sents(text):
        for att in range(4):
            o, f = ngp_sentence(s, rng)
            if build.qc_ok(s, o) or att == 3: break
        outs.append(o if build.qc_ok(s, o) else s); feats += f if build.qc_ok(s, o) else []
    return " ".join(outs), [{"feature": f, "tier": "morph" if f in {"dey", "don", "na", "no_neg", "fit", "go", "wan"} else "lex"} for f in feats]


# ------------------------------------------------------------------ APP
APP_RULES = {"a_ing", "fixin_future", "double_modals", "completive_done", "those_them", "regularized_reflexives", "negative_concord", "aint_have", "dont", "were_was", "yall", "existential_it"}
_APPD = {}
def app_dialect():
    if "d" not in _APPD:
        import mv_env
        d = mv_env.get_dialect("AppalachianDialect")
        d.morphosyntax_transforms = {k: min(1.0, float(v) * build.RATE_SCALE) for k, v in d.morphosyntax_transforms.items() if k in APP_RULES}
        _APPD["d"] = d
    return _APPD["d"]
APP_LEX = [(r"\bkids\b", "young'uns", "youngun"), (r"\bchildren\b", "young'uns", "youngun"), (r"\bpants\b", "britches", "britches"), (r"\bbag\b", "poke", "poke"),
           (r"\bover there\b", "yonder", "yonder"), (r"\bcarry\b", "tote", "tote")]


def app_transform(text, key):
    rng = random.Random(key); outs, feats = [], []
    build.GLOBAL_BLACKLIST.setdefault("APP", set())
    for i, s in enumerate(_sents(text)):
        o, r, att, _ = build.transform_sentence(app_dialect(), "APP", s, f"{key}:{i}")
        for pat, rep, f in APP_LEX:
            if rng.random() < 0.8:
                o, n = re.subn(pat, rep, o); feats += [{"feature": f, "tier": "lex"}] * n
        o = re.sub(r"\b(might|may) (?:cannot|can't)\b", r"\1 could not", o)   # "might cannot" is not an attested double modal
        outs.append(o); feats += [{"feature": x["feature"], "tier": "morph"} for x in r]
    return " ".join(outs), feats


# ------------------------------------------------------------------ FOR
FORMAL = [(r"\bkids\b", "children", "children"), (r"\bbuy\b", "purchase", "purchase"), (r"\bbuying\b", "purchasing", "purchase"), (r"\bstart\b", "commence", "commence"),
          (r"\bI think\b", "I believe", "believe"), (r"\bI want to\b", "I wish to", "wish"), (r"\btell (me|him|her|them|us)\b", r"inform \1", "inform"), (r"\btry to\b", "attempt to", "attempt"),
          (r"\bfix\b", "repair", "repair"), (r"\bask for\b", "request", "request"), (r"\bhelp me\b", "assist me", "assist"), (r"\bhelp\b(?= (?:him|her|them|us)\b)", "assist", "assist"),
          (r"\bfinish\b", "complete", "complete"), (r"\bshow\b", "demonstrate", "demonstrate"), (r"\bhard\b", "difficult", "difficult"), (r"\bbig\b", "substantial", "substantial"),
          (r"\bkeep\b(?= \w+ing)", "continue", "continue"), (r"\bget (?=an? |the |more |some )", "obtain ", "obtain"), (r"\bsure\b", "certain", "certain"),
          (r"\b[Cc]an you\b", "Would you be able to", "would_you"), (r"\bHow do I\b", "How might I", "how_might"), (r"\bI do not know\b", "I am uncertain", "uncertain"),
          (r"\bboss\b", "supervisor", "supervisor"), (r"\bmom\b", "mother", "mother"), (r"\bcoworkers?\b", "colleagues", "colleague"), (r"\ba bit\b", "somewhat", "somewhat")]


def for_transform(text, key):
    feats = []
    for pat, rep, f in FORMAL:
        text, n = re.subn(pat, rep, text); feats += [{"feature": f, "tier": "lex"}] * n
    return text, feats
