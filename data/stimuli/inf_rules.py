"""INF condition: informal *Standard* American English (plan §2.2) -- contractions, casual lexis, lowercase sentence starts;
NO dialect features. Deterministic given `seed`. Every applied edit is recorded in `features_applied`.
"""
import random, re

CONTRACTIONS = [  # (pattern, replacement, feature)
    (r"\bcannot\b", "can't", "contraction"), (r"\bcan not\b", "can't", "contraction"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|would|should|could|will) not\b", lambda m: {
        "do": "don't", "does": "doesn't", "did": "didn't", "is": "isn't", "are": "aren't", "was": "wasn't",
        "were": "weren't", "has": "hasn't", "have": "haven't", "had": "hadn't", "would": "wouldn't",
        "should": "shouldn't", "could": "couldn't", "will": "won't"}[m.group(1).lower()], "contraction"),
    (r"\bI am\b", "I'm", "contraction"), (r"\bI have\b(?= (?:been|already|never|not|a lot|to)\b)", "I've", "contraction"),
    (r"\b(we|they|you) are\b", lambda m: m.group(1) + "'re", "contraction"),
    (r"\b(he|she|it|that|there|what|who) is\b", lambda m: m.group(1) + "'s", "contraction"),
    (r"\b(I|we|they|you) will\b", lambda m: m.group(1) + "'ll", "contraction"),
    (r"\b(I|we|they|you) would\b", lambda m: m.group(1) + "'d", "contraction"),
    (r"\b(We|They|You) are\b", lambda m: m.group(1) + "'re", "contraction"),
    (r"\b(It|That|There|What|Who) is\b", lambda m: m.group(1) + "'s", "contraction"),
]
CASUAL = [  # (pattern, replacement, feature)
    (r"\bgoing to\b(?= [a-z]+)", "gonna", "casual_lexis"), (r"\bwant to\b", "wanna", "casual_lexis"),
    (r"\bhave to\b", "gotta", "casual_lexis"), (r"\bkind of\b", "kinda", "casual_lexis"),
    (r"\bvery\b", "super", "casual_lexis"), (r"\breally bad\b", "really bad", "casual_lexis"),
    (r"\ba lot of\b", "a ton of", "casual_lexis"), (r"\bmore than I can eat\b", "way more than I can eat", "casual_lexis"),
    (r"\bterrible\b", "awful", "casual_lexis"), (r"\bcrazy\b", "nuts", "casual_lexis"),
    (r"\bOkay\b", "Ok", "casual_lexis"), (r"\bkids\b", "kids", "casual_lexis"),
    (r"\bmy mother\b", "my mom", "casual_lexis"), (r"\bMy mother\b", "My mom", "casual_lexis"),
    (r"\bmy father\b", "my dad", "casual_lexis"), (r"\bunderstand\b", "get", "casual_lexis"),
    (r"\bapartment\b", "place", "casual_lexis"), (r"\bmoney\b(?= for)", "cash", "casual_lexis"),
    (r"\bstressed\b", "stressed out", "casual_lexis"),
]
FILLERS = ["Honestly, ", "So, ", "Okay so ", "Ugh, ", "Man, ", "Hey, "]


def inf_transform(text, seed=0, p_filler=0.5):
    rng = random.Random(f"{seed}:{text}")
    applied = []
    out = text
    for pat, rep, feat in CONTRACTIONS + CASUAL:
        def sub(m, rep=rep, feat=feat):
            new = rep(m) if callable(rep) else re.sub(r"\\(\d)", lambda k: m.group(int(k.group(1))), rep)
            if new != m.group(0):
                applied.append({"feature": feat, "span": [m.group(0), new]})
            return new
        out = re.sub(pat, sub, out)
    if rng.random() < p_filler:
        f = rng.choice(FILLERS)
        out = f + out[0].lower() + out[1:] if out[:2] != "I " and not out.startswith("I'") else f + out
        applied.append({"feature": "filler", "span": ["", f.strip()]})
    # lowercase sentence starts (not the pronoun I, not proper names in the seeds: seeds have no names)
    def lc(m):
        w = m.group(2)
        if w in {"I"} or w.startswith("I'"):
            return m.group(0)
        applied.append({"feature": "lowercase", "span": [w, w[0].lower() + w[1:]]})
        return m.group(1) + w[0].lower() + w[1:]
    out = re.sub(r"((?:^|[.!?]\s+))([A-Z][\w']*)", lc, out)
    return out, applied
