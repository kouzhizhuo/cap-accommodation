"""Recognisers shared by several varieties (copula/auxiliary absence, 3sg -s absence, -in' spelling, ...)."""
import re
from .base import spans

PRON_SUBJ = r"(?:he|she|they|we|you|it|who|that|everybody|everyone|nobody|somebody|people|kids|dad|mom)"
_AUX = r"(?:was|were|is|are|am|been|be|being|will|would|can|could|should|shall|may|might|must|do|did|does|have|has|had|just|also|not|never|always|only|really|still|even|too|ever|then|now|all|both|each|got|get|gets|gon|gonna|finna|ain't|to|and|or|but|so|if|when|as|than|because|though|while|from)"
ADJ = (r"(?:tired|sick|mad|happy|sad|hungry|late|ready|busy|scared|nervous|upset|right|wrong|crazy|sure|good|bad|fine|okay|ok|cold|hot|"
       r"hard|easy|free|broke|gone|done|finished|angry|worried|afraid|stuck|lost|full|empty|different|same|important|worth|true|"
       r"real|clear|open|closed|alone|awake|asleep|sorry|proud|jealous|confused|excited|freezing|terrible|awful|great|nice|expensive|cheap|slow|fast)")


def copula_absence(text):
    """Realised copula/auxiliary-be absence: subject pronoun + V-ing / predicate adjective / predicate NP / locative with no be."""
    out = []
    # (1) subject + V-ing without aux ("she working", "they going")
    out += spans(rf"\b(?:he|she|they|we|you|it|who|everybody|nobody)\s+(?!{_AUX}\b)(?:[a-z]{{2,}}ing)\b", text)
    # (2) subject + predicate adjective ("he tired", "she upset", "they busy"); not "he tired me out"
    out += spans(rf"\b(?:he|she|they|we|you|it|that|who|everybody|nobody)\s+(?:(?:very|so|too|really|damn|quite|kinda)\s+)?{ADJ}\b(?!\s+(?:me|him|her|us|them|you|it|the|a|an|my|your|his|our|their)\b)", text)
    # (2b) adverb between subject and predicate ("they already gone", "he still mad")
    out += spans(rf"\b(?:he|she|they|we|you|it)\s+(?:already|just|still|always|really)\s+(?:gone|done|finished|[a-z]{{2,}}ing|{ADJ})\b", text)
    # (2c) subject + intensifier + adjective without be ("this one very nice", "he so tired"); not "thank you very much"
    out += spans(r"(?<!thank )(?<!thanks )\b(?:this|that|it|he|she|they|we|you)(?:\s+one)?\s+(?:very|so|too|damn|quite)\s+(?!much\b|many\b|far\b)[a-z]+\b", text)
    # (3) subject + predicate NP ("she a nurse", "he my brother")
    out += spans(r"\b(?:he|she|they|we|you)\s+(?:a|an|my|your|his|her|our|their)\s+[a-z]{3,}\b(?!\s+(?:and|or)?\s*(?:said|says|told|knows))", text)
    # (4) subject + locative ("he at the store")
    out += spans(r"\b(?:he|she|they|we|you|it)\s+(?:at|in|on|by|over|under)\s+(?:the|a|my|his|her|our|their|home|work|school|church)\b", text)
    out += copula_absence_dep(text)
    # dedupe overlapping
    out.sort()
    ded = []
    for s in out:
        if not ded or s[0] >= ded[-1][1]:
            ded.append(s)
    return ded


_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def copula_absence_dep(text):
    """Dependency branch: a clause head that is a NOUN/ADJ/ADP/participle with an nsubj and no cop/aux child ("A dog a pet", "This due to X")."""
    doc = nlp()(text)
    out = []
    for t in doc:
        if t.dep_ not in {"ROOT", "conj", "ccomp", "advcl", "relcl", "parataxis"}:
            continue
        kids = list(t.children)
        if any(c.dep_ in {"cop", "aux", "auxpass"} for c in kids):
            continue
        if not any(c.dep_ == "nsubj" for c in kids):
            continue
        if t.pos_ in {"NOUN", "PROPN", "ADJ", "ADP", "NUM"} or t.tag_ in {"VBG", "VBN"} and t.dep_ != "relcl":
            s_ = [c for c in kids if c.dep_ == "nsubj"][0]
            if s_.lower_ in {"i", "who", "which", "that"}:
                continue
            out.append((s_.idx, t.idx + len(t)))
    return out


def uninflect(text):
    """3sg present -s absence: a VBP verb whose subject is he/she/it or a singular noun ("she keep ignoring", "my boss never give")."""
    doc = nlp()(text)
    out = []
    for t in doc:
        if t.tag_ in {"VBP", "VB"} and t.pos_ in {"VERB", "AUX"} and t.lower_ not in {"am", "be"}:
            kids = list(t.children)
            if any(c.dep_ in {"aux", "auxpass"} or c.dep_ == "mark" and c.lower_ == "to" for c in kids):
                continue   # modal/aux/infinitive: bare form is grammatical
            subj = [c for c in kids if c.dep_ == "nsubj"]
            if not subj and t.dep_ == "conj":                   # shared subject: "He is X and have a sister"
                subj = [c for c in t.head.children if c.dep_ == "nsubj"]
            if any(c.lower_ in {"he", "she", "it"} or c.tag_ in {"NN", "NNP"} for c in subj):
                out.append((t.idx, t.idx + len(t)))
    return out


def were_was_plural(text):
    """'was' with a plural subject (we/you/they/y'all or NNS): "the scenes was so intense"."""
    doc = nlp()(text)
    out = []
    for t in doc:
        if t.lower_ == "was" and t.pos_ == "AUX":
            subj = [c for c in t.head.children if c.dep_ == "nsubj"] if t.head is not t else []
            subj += [c for c in t.children if c.dep_ == "nsubj"]
            if any(c.lower_ in {"we", "you", "they", "y'all"} or c.tag_ in {"NNS", "NNPS"} for c in subj):
                out.append((t.idx, t.idx + 3))
    return out


def them_det(text):
    """'them' as a determiner ("them boys")."""
    doc = nlp()(text)
    return [(t.idx, t.idx + 4) for t in doc if t.lower_ == "them" and t.dep_ == "det"]


def g_drop(text):
    return spans(r"\b[a-z]{2,}in['’](?=\W|$)", text)
