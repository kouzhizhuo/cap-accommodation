"""Step-2 detectors v1: NGP (Nigerian Pidgin), APP (Appalachian English), FOR (formal register). Registered on import (detectors.ext).
NGP/FOR: paired recognisers (variety form vs the standard counterpart as opportunity), no Multi-VALUE. APP: Multi-VALUE rules (AppalachianDialect) with synthetic gold.
Lists follow notes/lexicons/{ngp,app}.json + ngp_grammar.md; dictionary-only APP lexis is registered as a RATE feature only (D7)."""
import re
from .base import Feature, register, spans
from . import inf as _inf
from . import aae as _aae
from .shared import them_det

_S = r"(?:i|you|we|e|he|she|dem|una|they|who|[a-z]+)"
_SP = r"(?:i|you|we|e|he|she|dem|una)"


def ngp_dey(t): return spans(rf"\b{_SP}\s+dey\s+[a-z]+", t)
def ngp_dey_opp(t): return spans(r"\b(?:i am|you are|we are|he is|she is|they are|it is)\s+[a-z]+ing\b", t)
def ngp_don(t): return spans(rf"\b{_SP}\s+don\s+[a-z]+", t)
def ngp_don_opp(t): return spans(r"\b(?:i|you|we|they|he|she)\s+(?:have|has)\s+(?:already\s+|just\s+)?[a-z]+(?:ed|en)\b", t)
def ngp_na(t): return spans(r"\bna\b", t)
def ngp_na_opp(t): return spans(r"\b(?:is|are|am|was|were)\s+(?:a|an|the|my|your|his|her|our|their)\b", t)
def ngp_no(t): return spans(rf"\b{_SP}\s+no\s+(?!one\b|more\b|idea\b|matter\b|doubt\b|problem\b|way\b|longer\b|sure\b)[a-z]+", t)
def ngp_no_opp(t): return spans(r"\b(?:do|does|did)\s+not\b|\b(?:don't|doesn't|didn't)\b", t)
def ngp_fit(t): return spans(rf"\b{_SP}\s+(?:no\s+)?fit\s+[a-z]+", t)
def ngp_fit_opp(t): return spans(r"\b(?:can|cannot|can't|could)\b", t)
def ngp_wan(t): return spans(rf"\b{_SP}\s+wan\s+[a-z]+", t)
def ngp_wan_opp(t): return spans(r"\b(?:want|wants)\s+to\b", t)
def ngp_lex(t): return spans(r"\b(?:abeg|wahala|wetin|oga|sef|sha|una|pikin|sabi|shey|waka|tori|ogbonge|sotay|wey|dem|dis|pipo)\b|\babi\s*\?|\bo\s*[.!?]", t)

_inf.PAIRED_OPP.update({"__paired_ngp_dey__": ngp_dey_opp, "__paired_ngp_don__": ngp_don_opp, "__paired_ngp_na__": ngp_na_opp, "__paired_ngp_no__": ngp_no_opp,
                        "__paired_ngp_fit__": ngp_fit_opp, "__paired_ngp_wan__": ngp_wan_opp})
for _id, _rec in (("dey", ngp_dey), ("don", ngp_don), ("na", ngp_na), ("no", ngp_no), ("fit", ngp_fit), ("wan", ngp_wan)):
    register(Feature(_id, "NGP", "morph", _rec, (f"__paired_ngp_{_id}__",), None, f"NGP {_id} vs standard counterpart"))
register(Feature("lexicon", "NGP", "lex", ngp_lex, (), None, "NGP function words / lexis (rate)"))

# ---- FOR (register control; robustness row only)
_FORM = r"\b(?:purchase[sd]?|purchasing|commence[sd]?|attempt(?:s|ed)? to|inform (?:me|him|her|them|us)|repair|request|assist|demonstrate|substantial|difficult|obtain|certain|I wish to|I believe|children|would you be able to|how might I|I am uncertain|supervisor|mother|colleagues?|somewhat)\b"
_INFM = r"\b(?:buy|buys|buying|start|starts|try to|tries to|tell (?:me|him|her|them|us)|fix|ask for|help|show|big|hard|get a|get an|get the|sure|I want to|I think|kids|can you|how do I|I do not know|boss|mom|coworkers?|a bit)\b"
_inf.PAIRED_OPP["__paired_for_lexis__"] = lambda t: spans(_INFM, t)
register(Feature("lexis", "FOR", "morph", lambda t: spans(_FORM, t), ("__paired_for_lexis__",), None, "formal lexis vs plain counterpart"))

# ---- APP (Multi-VALUE AppalachianDialect rules; recognisers regex/spaCy)
DA = "AppalachianDialect"
def a_prefix(t): return spans(r"\ba-[a-z]{3,}ing\b", t)
def fixin(t): return spans(r"\bfixin['’g]?\s+to\b", t)
def double_modal(t): return spans(r"\b(?:might|may|must|used to)\s+(?:could|can|would|should|will)\b", t)
def app_lex(t): return spans(r"\b(?:young['’]?uns?|yonder|britches|poke|tote[sd]?|afeared|hisself|theirselves|you['’]?uns)\b", t)
register(Feature("a_prefix", "APP", "morph", a_prefix, ("a_ing",), DA, "a-prefixing"))
register(Feature("fixin_to", "APP", "morph", fixin, ("fixin_future",), DA, "fixin' to"))
register(Feature("double_modals", "APP", "morph", double_modal, ("double_modals",), DA, "double modals"))
register(Feature("those_them", "APP", "morph", lambda t: _aae.those_them(t), ("those_them",), DA, "demonstrative them", needs_spacy=True, gold_filter=_aae.gold_those_them))
register(Feature("negative_concord", "APP", "morph", _aae.negative_concord, ("negative_concord",), DA, "multiple negation", gold_filter=_aae.gold_neg_concord))
register(Feature("ain_t", "APP", "morph", _aae.ain_t, ("aint_have",), DA, "ain't"))
register(Feature("yall", "APP", "morph", _aae.yall, ("yall",), DA, "y'all"))
register(Feature("reflexive", "APP", "morph", _aae.reflexive_aave, ("regularized_reflexives",), DA, "hisself/theirselves"))
register(Feature("lexicon", "APP", "lex", app_lex, (), DA, "Appalachian dictionary lexis (rate; dictionary-only items)"))
