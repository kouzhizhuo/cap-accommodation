"""Colloquial Singapore English (Singlish) detectors v0: particles, copula deletion, can/cannot, already, zero plural, bare past,
existential got, inverted indirect questions, lexicon. Same conventions as aae.py."""
import re
from .base import Feature, register, spans
from .shared import copula_absence, uninflect

D = "ColloquialSingaporeDialect"
_PART = r"(?:lah+|lor+|leh+|meh+|hor+|sia+|liao|mah(?!\s+jong)|wor|loh|lar)"


def particles(text):
    out = spans(rf"\b{_PART}\b", text)
    out += spans(r"(?<=[A-Za-z0-9])\s+ah(?=\s*[?.!,]|\s*$)", text)   # clause-final 'ah' (not the interjection 'Ah,' at sentence start)
    return out


def particle_of(name):
    return lambda text: spans(rf"\b{name}\b", text)


def can_cannot(text):
    return spans(r"\bcan\s+or\s+not\b|\b(?:want|got|have|need|is|okay|ok|true|correct|possible)\s+or\s+not\b", text) + \
        spans(r"(?:^|(?<=[.!?]\s))(?:can|cannot)\s*(?=[.!?])", text)


def already(text):
    return spans(r"(?<=[A-Za-z]\s)already\b(?=\s*(?:[.!?,;]|$|lah|lor|leh))", text)


def finish_completive(text):
    return spans(r"(?<!\bto )(?<!\bwill )(?<!\bcan )(?<!\bcould )(?<!\bshould )(?<!\bwould )(?<!\band )(?<!\bor )(?<!\bmust )(?<!\bmay )(?<!\bdid )(?<!\bdo )(?<!\bdon't )(?<!\bthen )(?<!\bplease )(?<!\bjust )(?<!\bgo )(?<!\bonce )\b[a-z]+\s+finish\s+(?!(?:the|my|your|a|an|it|this|that|these|those|with|early|first|up|off|and|on|in|at|line|strong)\b)(?![a-z]+ing\b)[a-z]+\b", text)


def existential_got(text):
    return spans(r"\b(?:there|here|where)\s+got\b", text)


_V = (r"(?:go|come|see|eat|buy|call|meet|take|make|get|give|tell|say|think|know|want|try|work|talk|stay|bring|find|leave|start|finish|"
      r"watch|ask|forget|borrow|return|burn|spill|study|play|help|bake|follow|cook|pay|move)")
_PAST = r"(?:yesterday|last\s+(?:night|week|month|year|time|weekend)|ago|this\s+morning)"


def bare_past(text):
    return spans(rf"\b{_PAST}\b[^.!?]{{0,30}}\b(?:i|we|they|he|she)\s+{_V}\b", text) + \
        spans(rf"\b(?:i|we|they|he|she)\s+{_V}\b[^.!?]{{0,40}}\b{_PAST}\b", text)


_N = (r"(?:year|month|week|day|hour|minute|dollar|person|friend|kid|thing|time|message|email|idea|problem|question|step|way|option|mistake|"
      r"neighbor|coworker|people|book|tomato|bean)")


def zero_plural(text):
    return spans(rf"\b(?:two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|many|several|few|hundred|thousand)\s+{_N}\b(?!s|\w)", text)


def inverted_indirect(text):
    return spans(r"\b(?:know|ask|wonder|tell|sure|idea|understand|explain|see|say|show)\b[^.!?,]{0,25}?\b(?:what|where|why|how|when|who|which)\s+(?:should|do|did|does|can|will|would|are|is|am)\s+(?!not\b)[a-z]+\b", text)


def sge_lexicon(text):
    return spans(r"\b(?:kena|shiok|paiseh|sian|jialat|wah|aiyo|aiyah|alamak|walao|makan|tapao|chope|cheem|atas|kiasu|kiasi|sabo|siao|lepak|bo\s+jio|ang\s+moh|die\s+die)\b", text)


for _f in [
    Feature("copula_absence", "SGE", "morph", copula_absence,
            ("drop_copula_be_NP", "drop_copula_be_AP", "drop_copula_be_locative", "drop_aux_be_progressive"), D, "be-absence"),
    Feature("can_cannot", "SGE", "morph", can_cannot, ("invariant_tag_can_or_not",), D, "can/cannot, X or not"),
    Feature("already", "SGE", "lex", already, (), D, "clause-final already (rate: MV rule also emits preverbal 'already', which is valid SAE)"),
    Feature("finish_completive", "SGE", "morph", finish_completive, ("completive_finish",), D, "finish + V completive"),
    Feature("existential_got", "SGE", "lex", existential_got, (), D, "there got (rate)"),
    Feature("bare_past", "SGE", "lex", bare_past, (), D, "bare verb with past-time adverb (rate: bare past is not recognisable without context)"),
    Feature("zero_plural", "SGE", "lex", zero_plural, (), D, "numeral + singular noun (rate)"),
    Feature("uninflect", "SGE", "morph", uninflect, ("uninflect",), D, "3sg -s absence", needs_spacy=True),
    Feature("inverted_indirect", "SGE", "morph", inverted_indirect, ("inverted_indirect_question",), D, "inverted embedded question"),
    Feature("particles", "SGE", "lex", particles, (), D, "discourse particles lah/lor/leh/meh/hor/sia/... (rate)"),
    Feature("particle_lah", "SGE", "lex", particle_of("lah+"), (), D, "lah (rate; subset of particles)"),
    Feature("particle_lor", "SGE", "lex", particle_of("lor+"), (), D, "lor"),
    Feature("particle_leh", "SGE", "lex", particle_of("leh+"), (), D, "leh"),
    Feature("particle_meh", "SGE", "lex", particle_of("meh+"), (), D, "meh"),
    Feature("particle_hor", "SGE", "lex", particle_of("hor+"), (), D, "hor"),
    Feature("lexicon", "SGE", "lex", sge_lexicon, (), D, "Singlish lexicon (rate)"),
]:
    register(_f)
