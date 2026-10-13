"""Standard British (BRE) and Australian (AUS) English cues: spelling + lexis, opportunity = the US variant of the same item.
No Multi-VALUE involved (0.3 has no British/Australian standard class). v0 mini-lists written by the coder from common knowledge
(VarCon/support lists replace them in v1: notes/lexicons/bre.md, aus.md). D_v = real / (real + opp) over paired items, i.e. the share of
US/UK-alternating items that use the variety's form. Recognisers are exact word matches (case-insensitive) -> unit-testable."""
import re
from .base import Feature, register, spans
from . import inf as _inf

# ---- US -> UK spelling (word forms; inflections listed explicitly to keep matching exact)
SPELL = {
    "color": "colour", "colors": "colours", "colored": "coloured", "favorite": "favourite", "favorites": "favourites", "neighbor": "neighbour",
    "neighbors": "neighbours", "neighborhood": "neighbourhood", "honor": "honour", "humor": "humour", "labor": "labour", "behavior": "behaviour",
    "flavor": "flavour", "harbor": "harbour", "rumor": "rumour", "vapor": "vapour", "savior": "saviour", "odor": "odour", "rigor": "rigour",
    "realize": "realise", "realized": "realised", "realizing": "realising", "organize": "organise", "organized": "organised",
    "organizing": "organising", "organization": "organisation", "recognize": "recognise", "recognized": "recognised", "apologize": "apologise",
    "apologized": "apologised", "criticize": "criticise", "summarize": "summarise", "prioritize": "prioritise", "emphasize": "emphasise",
    "minimize": "minimise", "maximize": "maximise", "customize": "customise", "analyze": "analyse", "analyzed": "analysed", "analyzing": "analysing",
    "center": "centre", "centers": "centres", "meter": "metre", "theater": "theatre", "fiber": "fibre", "liter": "litre",
    "defense": "defence", "offense": "offence", "pretense": "pretence", "catalog": "catalogue", "dialog": "dialogue",
    "traveling": "travelling", "traveled": "travelled", "traveler": "traveller", "canceled": "cancelled", "canceling": "cancelling",
    "modeling": "modelling", "labeled": "labelled", "counselor": "counsellor", "jewelry": "jewellery", "fulfill": "fulfil",
    "skillful": "skilful", "enrollment": "enrolment", "gray": "grey", "tire": "tyre", "tires": "tyres", "curb": "kerb", "aluminum": "aluminium",
    "pajamas": "pyjamas", "plow": "plough", "mold": "mould", "sulfur": "sulphur", "cozy": "cosy", "donut": "doughnut", "skeptical": "sceptical",
    "aging": "ageing", "practice": "practise", "program": "programme", "learned": "learnt", "spelled": "spelt", "burned": "burnt",
    "dreamed": "dreamt", "toward": "towards", "amongst": "amongst", "check": "cheque",
}
# ambiguous US words that are also valid in UK use are excluded from opportunity counting (noun 'practice', 'check', 'program', 'tire' verb, 'learned' adj)
AMBIG = {"practice", "check", "program", "tire", "learned", "mold", "toward", "burned", "canceled", "canceling", "dialog", "catalog", "spelled", "dreamed", "aging", "analyzed"}
SPELL_US = {k: v for k, v in SPELL.items() if k not in AMBIG}
# ---- US -> UK lexis
LEX = {
    "apartment": "flat", "apartments": "flats", "vacation": "holiday", "truck": "lorry", "mom": "mum", "mommy": "mummy", "trash": "rubbish",
    "garbage": "rubbish", "cookie": "biscuit", "cookies": "biscuits", "elevator": "lift", "gas": "petrol", "sidewalk": "pavement", "fall": "autumn",
    "soccer": "football", "diaper": "nappy", "diapers": "nappies", "candy": "sweets", "faucet": "tap", "flashlight": "torch", "movie": "film",
    "movies": "films", "cellphone": "mobile", "pants": "trousers", "sneakers": "trainers", "cart": "trolley", "resume": "CV", "stove": "cooker",
    "closet": "wardrobe", "math": "maths", "gotten": "got", "while": "whilst", "among": "amongst",
}
LEX_AMBIG = {"gas", "fall", "cart", "stove", "while", "among", "resume", "mail",
             "apartment", "apartments", "vacation", "soccer", "movie", "movies", "cookie", "cookies", "candy", "faucet", "flashlight", "elevator", "sidewalk", "pants", "closet"}
# 'flat/lift/tap/film/holiday/sweets/rubbish/biscuit/torch/pavement/trousers/autumn' are ordinary US words too: the UK-side recogniser only counts
# items that are (near-)exclusively British; US-side opportunity words above stay in the swap table (used by the stimulus builder) but not as opportunities.
UK_ONLY = {"lorry", "mum", "mummy", "nappy", "nappies", "petrol", "trainers", "wardrobe", "cooker", "maths", "whilst", "amongst", "got_"}    # too ambiguous to count as opportunities
LEX_US = {k: v for k, v in LEX.items() if k not in LEX_AMBIG}                    # US-side opportunities (unambiguous US words)
LEX_SWAP = dict(LEX)                                                             # everything the stimulus builder may swap
# ---- Australian additions (diagnostic lexis; AUS also uses the British spellings above except 'program')
AUS_LEX = {"arvo": "afternoon", "ute": "pickup", "servo": "gas station", "brekkie": "breakfast", "barbie": "barbecue", "mate": "friend",
           "heaps": "a lot", "reckon": "think", "footy": "football", "sunnies": "sunglasses", "thongs": "flip-flops", "esky": "cooler",
           "bloke": "guy", "stoked": "thrilled", "keen": "eager", "mobile": "cellphone", "unit": "apartment", "rubbish": "trash",
           "holiday": "vacation", "petrol": "gas", "lift": "elevator", "biscuit": "cookie", "mum": "mom", "nappy": "diaper"}
AUS_ONLY = {k for k in AUS_LEX if k in {"arvo", "ute", "servo", "brekkie", "barbie", "mate", "heaps", "reckon", "footy", "sunnies", "thongs", "esky",
                                        "bloke", "stoked", "keen", "unit"}}


def _words(d, text):
    return spans(r"\b(?:" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b", text)


def uk_spelling(text): return _words({v: 1 for v in SPELL_US.values()}, text)
def us_spelling(text): return _words(SPELL_US, text)
def uk_lexis(text): return _words({v: 1 for v in UK_ONLY}, text)
def us_lexis(text): return _words({k: 1 for k, v in LEX_US.items() if v in UK_ONLY}, text)
def aus_lexis(text): return _words({k: 1 for k in AUS_ONLY}, text)
def aus_lexis_opp(text):   # the non-Australian near-equivalent used where an Australian item could stand
    return spans(r"\b(?:afternoon|pickup|gas\s+station|breakfast|barbecue|friend|a\s+lot|think|football|sunglasses|flip-flops|cooler|guy|thrilled|eager|apartment)\b", text)


def have_got(text): return spans(r"\b(?:i|we|you|they)['’]ve\s+got\s+(?:a|an|no|some|more|two|three|many|the)\b", text)
def have_plain(text): return spans(r"\b(?:i|we|you|they)\s+have\s+(?:a|an|no|some|more|two|three|many|the)\b", text)


# paired-opportunity registry (base.score_text looks these up)
_inf.PAIRED_OPP.update({"__paired_bre_spelling__": us_spelling, "__paired_bre_lexis__": us_lexis, "__paired_bre_havegot__": have_plain,
                        "__paired_aus_spelling__": us_spelling, "__paired_aus_lexis__": aus_lexis_opp})
for _v in ("BRE", "AUS"):
    register(Feature("spelling", _v, "morph", uk_spelling, (f"__paired_{_v.lower()}_spelling__",), None, "UK/AU spelling vs US spelling"))
register(Feature("lexis", "BRE", "morph", uk_lexis, ("__paired_bre_lexis__",), None, "UK lexis vs US lexis"))
register(Feature("have_got", "BRE", "morph", have_got, ("__paired_bre_havegot__",), None, "have got vs have (possession)"))
register(Feature("aus_lexis", "AUS", "morph", aus_lexis, ("__paired_aus_lexis__",), None, "Australian-specific lexis vs neutral near-equivalents"))
