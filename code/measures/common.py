"""Shared helpers for the style-convergence measures (P3 exp/measures).

* word tokenisation identical to detectors/base.py (`_TOK`), so copied-span logic aligns with the dialect index
* copied_ranges(): logic copied (not imported) from detectors/base.py so this package has no dependency on the
  detector package; tests/test_common.py checks parity with the original when it is importable.
* function-word lexicon: ConvoKit's bundled `data/coord-liwc-patterns.txt` (8 LIWC-derived categories, MIT licence,
  convokit 3.0.0; copy in measures/data/, identical sha256) + a small self-authored NEGATION list, because the
  ConvoKit file has no negation category.
"""
import os
import re

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN_FILE = os.path.join(_HERE, "data", "coord-liwc-patterns.txt")

# Ireland & Pennebaker (2010) nine categories; order fixed. ConvoKit ids, plus 'negate' (self-authored).
CATS = ["ppron", "ipron", "article", "conj", "preps", "auxverb", "adverb", "negate", "quant"]
CAT_LABEL = {"ppron": "personal pronouns", "ipron": "impersonal pronouns", "article": "articles",
             "conj": "conjunctions", "preps": "prepositions", "auxverb": "auxiliary verbs",
             "adverb": "high-frequency adverbs", "negate": "negations", "quant": "quantifiers"}

# Self-authored approximation of LIWC2007 "negate" (NOT from ConvoKit; NOT the licensed LIWC list).
NEGATE_WORDS = [
    "ain't", "aint", "aren't", "arent", "can't", "cant", "cannot", "couldn't", "couldnt", "didn't", "didnt",
    "doesn't", "doesnt", "don't", "dont", "hadn't", "hadnt", "hasn't", "hasnt", "haven't", "havent", "isn't", "isnt",
    "mustn't", "mustnt", "needn't", "neither", "never", "no", "none", "nope", "nor", "not", "nothing", "nowhere",
    "oughtn't", "shan't", "shant", "shouldn't", "shouldnt", "wasn't", "wasnt", "weren't", "werent", "won't", "wont",
    "wouldn't", "wouldnt", "without",
]
NEGATE_PREFIXES = ["nobod"]

_TOK = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def tokens(text):
    """[(lowercased word, start, end)] ; curly apostrophes normalised."""
    return [(m.group(0).lower().replace("’", "'"), m.start(), m.end()) for m in _TOK.finditer(text)]


def copied_ranges(response, prompt, n=4, quotes=True):
    """Char ranges of `response` covered by word n-grams also present in `prompt`, plus quoted spans.
    Logic of detectors/base.py:copied_ranges (quotes=True reproduces it exactly); returns (ranges, copy_rate)."""
    r, p = tokens(response), tokens(prompt)
    pgrams = {tuple(t[0] for t in p[i:i + n]) for i in range(len(p) - n + 1)}
    covered = [False] * len(r)
    for i in range(len(r) - n + 1):
        if tuple(t[0] for t in r[i:i + n]) in pgrams:
            for j in range(i, i + n):
                covered[j] = True
    rng = [(s, e) for i, (_, s, e) in enumerate(r) if covered[i]]
    if quotes:
        for m in re.finditer(r"[\"“”]([^\"“”]{1,200})[\"“”]", response):
            rng.append((m.start(), m.end()))
    return rng, sum(covered) / max(1, len(r))


def char_mask(text, ranges):
    """bool array over chars of `text`: True where inside a copied range."""
    m = np.zeros(len(text) + 1, dtype=bool)
    for s, e in ranges:
        m[s:e] = True
    return m


def novel_tokens(reply, prompt, n=4, quotes=True):
    """Reply word tokens that are NOT inside copied 4-grams / quotes: [(word, s, e)]."""
    rng, _ = copied_ranges(reply, prompt, n=n, quotes=quotes)
    if not rng:
        return tokens(reply)
    # same overlap rule as detectors.base.in_ranges (s < re_ and rs < e)
    return [t for t in tokens(reply) if not any(t[1] < re_ and rs < t[2] for rs, re_ in rng)]


# ---------------------------------------------------------------- lexicon
def _load_lexicon():
    exact, prefix = {c: set() for c in CATS}, {c: [] for c in CATS}
    with open(PATTERN_FILE) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cat, pat = line.split("\t")
            for item in pat.split("|"):
                w = item.replace("’", "'")
                if w.startswith("\\b"):
                    w = w[2:]
                if w.endswith("\\b"):
                    exact[cat].add(w[:-2])
                else:            # ConvoKit prefix pattern, e.g. \btheir, \banybod, \bthing
                    prefix[cat].append(w)
    exact["negate"].update(NEGATE_WORDS)
    prefix["negate"].extend(NEGATE_PREFIXES)
    return exact, {c: tuple(v) for c, v in prefix.items()}


LEX_EXACT, LEX_PREFIX = _load_lexicon()
assert all(LEX_EXACT[c] or LEX_PREFIX[c] for c in CATS)


def word_cats(word):
    """Set of category ids a (lowercased) word belongs to (categories overlap, as in LIWC)."""
    return [c for c in CATS if word in LEX_EXACT[c] or (LEX_PREFIX[c] and word.startswith(LEX_PREFIX[c]))]


_cache = {}


def word_cats_cached(word):
    r = _cache.get(word)
    if r is None:
        r = _cache[word] = word_cats(word)
    return r


def cat_counts(words):
    """{cat: count} over an iterable of lowercased words."""
    out = {c: 0 for c in CATS}
    for w in words:
        for c in word_cats_cached(w):
            out[c] += 1
    return out
