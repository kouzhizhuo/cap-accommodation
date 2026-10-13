"""Informal Standard American English (INF; register control): contractions (opportunity-based), casual lexis, lowercase sentence
starts (opportunity-based)."""
import re
from .base import Feature, register, spans

_CONTR = r"\b(?:\w+n['’]t|(?:i|you|we|they|he|she|it|that|there|what|who|let)['’](?:m|re|ve|ll|d|s)|can['’]t|won['’]t)\b"
_UNCONTR = (r"\b(?:(?:do|does|did|is|are|was|were|have|has|had|would|should|could|will|must|need)\s+not|cannot|"
            r"(?:i)\s+(?:am|will|would|have|had)|(?:you|we|they)\s+(?:are|will|would|have|had)|(?:he|she|it|that|there|what|who)\s+(?:is|will|would|has|had)|let\s+us)\b")


def contraction(text):
    return spans(_CONTR, text)


def uncontracted(text):
    return spans(_UNCONTR, text)


def casual_lexis(text):
    return spans(r"\b(?:gonna|wanna|gotta|kinda|sorta|yeah|yep|nope|nah|super|awful|stuff|lol|ugh|hey|honestly|dude|guys|btw|tbh|imo|omg|totally|a\s+ton\s+of|way\s+more|mom|dad|pretty\s+(?:bad|good|sure|much))\b", text)


def lowercase_start(text):
    return [(m.start(2), m.end(2)) for m in re.finditer(r"((?:^|[.!?]\s+))([a-z][A-Za-z']*)", text)]


def uppercase_start(text):
    return [(m.start(2), m.end(2)) for m in re.finditer(r"((?:^|[.!?]\s+))([A-Z][A-Za-z']*)", text)]


# opportunity-based INF features need an 'opp' notion that is not a Multi-VALUE rule -> handled by paired recognisers
class _Paired:
    """INF morph features: opp spans come from the paired 'standard-variant' recogniser (no Multi-VALUE involved)."""


def _register_paired():
    from . import base
    register(Feature("contraction", "INF", "morph", contraction, ("__paired_contraction__",), None, "contracted vs uncontracted forms"))
    register(Feature("lowercase_start", "INF", "morph", lowercase_start, ("__paired_lowercase__",), None, "lowercase vs capitalised sentence starts"))
    register(Feature("casual_lexis", "INF", "lex", casual_lexis, (), None, "casual lexis (rate)"))


PAIRED_OPP = {"__paired_contraction__": uncontracted, "__paired_lowercase__": uppercase_start}
_register_paired()
