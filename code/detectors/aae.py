"""African American English (AAE) detectors v0. Rules follow eWAVE features implemented by Multi-VALUE (Urban AAVE) and the
list in the P3 plan §2.3 (copula absence, habitual be, 3sg -s absence, ain't, multiple negation, completive done, lexicon).
Recognisers are regex (+ spaCy for 3sg -s absence). NOT a claim of complete AAE coverage; features are retained only if they pass the
synthetic-gold gate (validate.py) and the unit tests (tests/test_detectors.py)."""
import re
from .base import Feature, register, spans
from .shared import copula_absence, uninflect, g_drop, were_was_plural, them_det

D = "AfricanAmericanVernacular"
NEG = r"(?:ain['’]?t|\bnot|can['’]t|won['’]t|(?:do|does|did|is|are|was|were|could|would|should|have|has|had)n['’]?t|(?:do|does|did|can|could|would|should|will|have|has|had|is|are|was|were)\s+not|never|cannot)"
NWORD = r"(?:nothing|nobody|nowhere|none|no\s+one|no\s+(?!matter\b|longer\b|more\b)[a-z]+|neither)"


def negative_concord(text):
    """Negation + n-word in one clause. The gap may not contain list/option markers, colons, newlines, quotes or brackets (NLI templates such as
    'Options: - yes - it is not possible to tell - no' and code produced ~all preference-data false alarms; E4 audit 2026-10-01)."""
    return spans(rf"\b{NEG}\b[^.!?;:\n\"'()\[\]{{}}<>|/\\#-]{{0,40}}?\b{NWORD}\b", text)


def gold_neg_concord(src, out):
    """Keep only outputs where an n-word was inserted (drops MV's is-not -> isn't contraction artefacts)."""
    return len(re.findall(r"\b(?:no|nothing|nobody|nowhere|none|never)\b", out, re.I)) > len(re.findall(r"\b(?:no|nothing|nobody|nowhere|none|never)\b", src, re.I))


def gold_those_them(src, out):
    return bool(re.search(r"\bthose\s+[a-z]+s\b", src, re.I))


def ain_t(text):
    return spans(r"\bain['’]?t\b", text)


def dont_3sg(text):
    return spans(r"\b(?:he|she|it|that|nobody|everybody|somebody|anybody|everyone|someone|who|teacher|boss|wife|husband|mom|dad|man|woman|friend|landlord|neighbor|coworker|son|daughter|sister|brother|mother|father)\s+(?:don['’]?t|do\s+not)\b", text)


_PART = r"(?:\w+ed|been|seen|gone|had|took|got|did|made|told|said|known|eaten|written|wrote|found|left|heard|knew|thought|paid|lost|won|read|met|spent|brought|finished|forgot|forgotten|taken|given|come|become|run|shown|built|held|kept|led|sent|set|sat|stood|understood|worked|\w+en)"
_NOTB = (r"(?<!\bhave )(?<!\bhas )(?<!\bhad )(?<!\bam )(?<!\bis )(?<!\bare )(?<!\bwas )(?<!\bwere )(?<!\bbe )(?<!\bbeen )(?<!\bbeing )"
         r"(?<!\bget )(?<!\bgot )(?<!\bgets )(?<!\bwhen )(?<!\buntil )(?<!\bonce )(?<!\bafter )(?<!\bbefore )(?<!\bnot )(?<!\bwell )(?<!\bbe )(?<!\bit's )(?<!\bI'm )")


def completive_done(text):
    return spans(rf"{_NOTB}\bdone\s+(?:already\s+|previously\s+|just\s+|never\s+)?(?:been\s+)?{_PART}\b", text)


def finna_gon(text):
    return spans(r"\b(?:finna|fixna|fitna|gon['’]?|ima|imma|bouta|boutta)\b|\bfixin['’g]?\s+to\b", text)


def yall(text):
    return spans(r"\by['’]?all\b", text)


_PLURAL_N = (r"(?:boys|girls|guys|folks|kids|people|shoes|clothes|books|days|cars|dogs|cats|houses|men|women|ladies|apples|bills|"
             r"tomatoes|beans|plants|weeds|neighbors|coworkers|friends|messages|emails|teachers|things|ideas|rules|shows|songs|jokes)")


def those_them(text):
    return spans(rf"\bthem\s+{_PLURAL_N}\b", text) + them_det(text)


def existential_it(text):
    return spans(r"\bit(?:['’]s|s|\s+is|\s+was|\s+ain['’]?t)\s+(?:a\s+lot\s+of|some|many|too\s+many|plenty\s+of|no)\s+(?:people|folks|kids|students|cars|houses|places|things|ways|problems|options|questions|reasons|women|men|children|food|money|space|room|seats|place|way|one|body|use|point|reason|time)\b", text)


def were_was(text):
    return spans(r"\b(?:we|you|they|y['’]all)\s+was\b", text) + were_was_plural(text)


def habitual_be(text):
    return spans(r"(?<!\blet )(?<!\bmay )(?<!\bif )(?<!\bto )\b(?:i|you|he|she|it|we|they|who|[a-z]+s)\s+be\s+(?:[a-z]+ing|at|in|on|there|like|too|so|always|really|all|out|up|down|with|over)\b", text)


def reflexive_aave(text):
    return spans(r"\b(?:hisself|theirselves|theirself|themself|ourself|ourselfs|yourselfs|hersef)\b", text)


def tag_non_concord(text):
    return spans(r",\s*(?:ain['’]?t\s+(?:it|that|nobody)|innit|huh)\s*\?", text)


def aae_lexicon(text):
    return spans(r"\b(?:tryna|bruh|homie|dawg|aight|deadass|lowkey|highkey|fam|no\s+cap|on\s+god|bussin|cuz|nah|yo|finna|frfr|bet)\b", text)


for _f in [
    Feature("copula_absence", "AAE", "morph", copula_absence,
            ("drop_copula_be_NP", "drop_copula_be_AP", "drop_copula_be_locative", "drop_aux_be_progressive"), D, "be-absence"),
    Feature("ain_t", "AAE", "morph", ain_t, ("aint_have", "aint_before_main"), D, "ain't for didn't/haven't/don't"),
    Feature("negative_concord", "AAE", "morph", negative_concord, ("negative_concord",), D, "multiple negation", gold_filter=gold_neg_concord),
    Feature("dont_3sg", "AAE", "morph", dont_3sg, ("dont",), D, "he/she/it don't"),
    Feature("uninflect", "AAE", "morph", uninflect, ("uninflect",), D, "3sg -s absence", needs_spacy=True),
    Feature("completive_done", "AAE", "morph", completive_done, ("completive_done",), D, "completive done"),
    Feature("finna_gon", "AAE", "morph", finna_gon, ("finna_future", "future_sub_gon"), D, "finna / gon future"),
    Feature("yall", "AAE", "morph", yall, ("yall",), D, "y'all"),
    Feature("those_them", "AAE", "morph", those_them, ("those_them",), D, "demonstrative them", needs_spacy=True, gold_filter=gold_those_them),
    Feature("existential_it", "AAE", "lex", existential_it, (), D, "existential it (rate: not separable from referential 'it' without context)"),
    Feature("were_was", "AAE", "morph", were_was, ("were_was",), D, "we/you/they/NNS was", needs_spacy=True),
    Feature("habitual_be", "AAE", "morph", habitual_be, (), D, "habitual be (no MV rule: rate feature)"),
    Feature("reflexive_aave", "AAE", "morph", reflexive_aave, (), D, "hisself/theirselves (rate)"),
    Feature("tag_non_concord", "AAE", "morph", tag_non_concord, (), D, "ain't it / innit tags (rate)"),
    Feature("g_drop", "AAE", "lex", g_drop, (), D, "-in' spelling (rate)"),
    Feature("lexicon", "AAE", "lex", aae_lexicon, (), D, "AAE-associated lexicon (rate)"),
]:
    register(_f)
