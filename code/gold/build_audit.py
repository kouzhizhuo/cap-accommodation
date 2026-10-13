#!/usr/bin/env python
"""Build the human-audit pools and file (TEAM_RULES §1.7a; notes/ANNOTATION_PROTOCOL.md §4). Env p3mv on <host>; fixed SEED.
Pools (the SAME pools the LLM reference raters label):  gold/pools/feature_pool.jsonl (50 real sentences x 5 varieties, ~50% detector-positive for the
audited features) and gold/pools/accommodation_pool.jsonl (300 prompt-reply pairs, stratified variety x stage class).
Audit file: gold/HUMAN_AUDIT_TODO.jsonl = 40 feature items + 20 accommodation items, shuffled, blind (ids carry no type; no model/condition/variety names).
Instruction texts, options and feature notes are parsed VERBATIM from ANNOTATION_PROTOCOL.md §4 (no notes invented here)."""
import glob, json, random, re, sys, xml.etree.ElementTree as ET
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP))
import detectors  # noqa: F401
import detectors.aae as A, detectors.sge as S, detectors.inf as I, detectors.bre as B
from detectors import shared

SEED = 20261001
rng = random.Random(SEED)
RAW = EXP / "data_raw_small"; PROTO = (EXP / "gold" / "ANNOTATION_PROTOCOL.md") if (EXP / "gold" / "ANNOTATION_PROTOCOL.md").exists() else EXP.parent / "notes" / "ANNOTATION_PROTOCOL.md"
GEN_GLOBS = [str(EXP.parent.parent / "runs" / "P3" / "E1" / "gen" / "*@*.jsonl"), str(EXP.parent.parent / "runs" / "P3" / "P0" / "gen" / "*@*.jsonl")] if False else None
PROFANE = re.compile(r"(?i)\b(fuck\w*|shit\w*|bitch\w*|nigg\w*|cunt|dick|pussy|whore|slut|ass|asshole|hoe|bastard|damn|dumbass|motherfuck\w*|cock|porn)\b")
pii = {"name": 0, "url": 0, "number": 0}


def mask(t):
    t, n1 = re.subn(r"@\w+", "[NAME]", t); t, n2 = re.subn(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", "[URL]", t); t, n3 = re.subn(r"\+?\d[\d\s\-]{6,}\d", "[NUMBER]", t)
    pii["name"] += n1; pii["url"] += n2; pii["number"] += n3
    return re.sub(r"\s+", " ", t).strip()


def sentences(text, lo=5, hi=40):
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip()
        if lo <= len(s.split()) <= hi and not PROFANE.search(s) and sum(c.isascii() for c in s) > 0.95 * len(s):
            out.append(mask(s))
    return out


# ---------------- detectors used to balance the pools and to choose feature items (NOT shown to the rater)
FEAT = {  # audit feature key -> (variety, recogniser)
    "copula_absence_aae": ("AAE", shared.copula_absence), "habitual_be": ("AAE", A.habitual_be), "uninflect": ("AAE", shared.uninflect), "ain_t": ("AAE", A.ain_t),
    "negative_concord": ("AAE", A.negative_concord), "completive_done": ("AAE", A.completive_done),
    "particles": ("SGE", S.particles), "copula_absence_sge": ("SGE", shared.copula_absence), "already": ("SGE", S.already),
    "dey": ("NGP", lambda t: re.findall(r"(?i)\bdey\b", t)), "don": ("NGP", lambda t: re.findall(r"(?i)\b(?:i|we|dem|he|she|e|you|una|they)\s+don\s+[a-z]+", t)),
    "na": ("NGP", lambda t: re.findall(r"(?i)\bna\b", t)),
    "inf": ("INF", lambda t: I.contraction(t) + I.casual_lexis(t)), "bre": ("BRE", lambda t: B.uk_spelling(t) + B.uk_lexis(t) + B.have_got(t)),
}
NOTE_KEY = {"copula_absence_aae": "copula absence (AAE)", "habitual_be": "habitual *be* (AAE)", "uninflect": "3sg -s absence (AAE)", "ain_t": "*ain't* (AAE)",
            "negative_concord": "multiple negation (AAE)", "completive_done": "completive *done* (AAE)", "particles": "discourse particles (SGE)",
            "copula_absence_sge": "copula deletion (SGE)", "already": "*already* as aspect (SGE)", "dey": "*dey* (NGP)", "don": "*don* (NGP)", "na": "*na* (NGP)",
            "inf": "contractions / casual lexis (INF)", "bre": "British spelling / lexis (BRE)"}
ALLOC = {"copula_absence_aae": 2, "habitual_be": 2, "uninflect": 2, "ain_t": 2, "negative_concord": 2, "completive_done": 2, "particles": 3, "copula_absence_sge": 3,
         "already": 3, "dey": 3, "don": 3, "na": 3, "inf": 5, "bre": 5}     # 40 feature items
assert sum(ALLOC.values()) == 40


def any_hit(variety, t):
    return any(bool(r(t)) for k, (v, r) in FEAT.items() if v == variety)


# ---------------- sources
def read_lines(p): return [l.rstrip("\n") for l in open(p, errors="ignore")]
def prompts_of(cond):
    return [(f"stimuli/v1/{cond}.jsonl:{i + 1}", s) for i, l in enumerate(open(EXP / "stimuli" / "v1" / f"{cond}.jsonl")) for s in sentences(json.loads(l)["text"])]
def gen_rows():
    files = sorted(glob.glob(str(EXP / "E1_gen_link" / "*@*.jsonl"))) or sorted(glob.glob(str(EXP / "P0" / "gen" / "*@*.jsonl")))
    rows = []
    for fn in files:
        for i, l in enumerate(open(fn)):
            r = json.loads(l); r["_src"] = f"{Path(fn).name}:{i + 1}"; rows.append(r)
    return rows, files
rows, gen_files = gen_rows()
def replies_of(cond):
    return [(r["_src"], s) for r in rows if r["condition"] == cond and r["sampling_seed"] == 0 and r["sample"] == 0 for s in sentences(r["response"])]
auth = {"AAE": [("groenwold2020/aave_samples.txt:%d" % (i + 1), s) for i, l in enumerate(read_lines(RAW / "aave_samples.txt")) for s in sentences(l)],
        "SGE": [], "NGP": []}
root = ET.parse(RAW / "smsCorpus_en_2015.03.09_all.xml").getroot()
for i, m in enumerate(root.iter("message")):
    if (m.findtext(".//country") or "").strip() in ("Singapore", "SG"):
        for s in sentences(m.findtext("text") or ""): auth["SGE"].append((f"nus_sms:msg{m.get('id')}", s))
for fn in sorted(glob.glob(str(RAW / "pcm_nsc-ud-*.conllu"))):
    for i, l in enumerate(read_lines(fn)):
        if l.startswith("# text_ortho = "):
            for s in sentences(l[len("# text_ortho = "):]): auth["NGP"].append((f"{Path(fn).name}:{i + 1}", s))
BUCKETS = {"AAE": [(auth["AAE"], 25), (prompts_of("AAE"), 13), (replies_of("AAE"), 12)], "SGE": [(auth["SGE"], 25), (prompts_of("SGE"), 13), (replies_of("SGE"), 12)],
           "NGP": [(auth["NGP"], 50)], "INF": [(prompts_of("INF"), 25), (replies_of("INF"), 25)], "BRE": [(prompts_of("BRE"), 25), (replies_of("BRE"), 25)]}
pool, short = [], []
for v, buckets in BUCKETS.items():
    for src, n in buckets:
        src = list(dict.fromkeys(src)); rng.shuffle(src); src = src[:800]   # random cap before detector testing (spaCy recognisers are slow)
        pos = [x for x in src if any_hit(v, x[1])]; neg = [x for x in src if not any_hit(v, x[1])]
        rng.shuffle(pos); rng.shuffle(neg)
        take = pos[:n // 2] + neg[:n - min(len(pos), n // 2)]
        if len(take) < n: short.append((v, n, len(take)))
        for p, s in take: pool.append({"variety": v, "text": s, "source": p, "detector_any": any_hit(v, s)})
for i, r in enumerate(pool): r["pool_id"] = f"F{i:03d}"
Path(EXP / "gold" / "pools").mkdir(parents=True, exist_ok=True)
with open(EXP / "gold" / "pools" / "feature_pool.jsonl", "w") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool))

# ---------------- accommodation pool: 300 pairs = {AAE,SGE,INF,BRE} x {base,sft,aligned} x 25 (fallback: conditions present)
def cls(r):
    st = r["stage"]
    return "base" if st == "base" and r["scaffold"] == "F2" else "sft" if st == "sft" else "aligned" if st in ("dpo", "instruct", "rlvr1", "final") else None
conds = [c for c in ("AAE", "SGE", "INF", "BRE") if any(r["condition"] == c for r in rows)]
per = 300 // (len(conds) * 3); acc = []
for c in conds:
    for k in ("base", "sft", "aligned"):
        cand = [r for r in rows if r["condition"] == c and cls(r) == k and r["sampling_seed"] == 0 and r["sample"] == 0]
        rng.shuffle(cand); seen = set()
        for r in cand:
            if r["seed_id"] in seen or len(seen) >= per + (1 if len(acc) < 300 - per * len(conds) * 3 else 0): continue
            seen.add(r["seed_id"]); acc.append(r)
def trim(t, n):
    w = t.split(); return (" ".join(w[:n]) + (" …" if len(w) > n else ""), len(w) > n)
accpool = []
for i, r in enumerate(acc[:300]):
    u, tu = trim(mask(r["prompt"]), 60); a, ta = trim(mask(r["response"]), 60)
    accpool.append({"pool_id": f"R{i:03d}", "condition": r["condition"], "stage_class": cls(r), "model": r["model"], "source": r["_src"], "user": u, "reply": a, "trimmed_user": tu, "trimmed_reply": ta})
with open(EXP / "gold" / "pools" / "accommodation_pool.jsonl", "w") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accpool))

# ---------------- audit items
proto = open(PROTO).read()
gen_instr = " ".join(l[2:].strip() for l in re.search(r"### 4\.1.*?\n(.*?)### 4\.2", proto, re.S).group(1).splitlines() if l.startswith("> "))
notes = {}
for l in re.search(r"### 4\.3.*?\n(.*?)### 4\.4", proto, re.S).group(1).splitlines():
    m = re.match(r"\|\s*(.+?)\s*\|\s*\"(.*)\"\s*\|\s*$", l)
    if m: notes[m.group(1)] = m.group(2)
INSTR_F = "Read the text and the feature description in the note. Does the feature occur in the text at least once?"
INSTR_A = ("The first message is from a user and the second is a chatbot's reply. How much does the reply use the user's own way of writing "
           "(their dialect's words, grammar or spelling) instead of standard English?")
OPT_F = ["Yes", "No", "Unsure"]
OPT_A = ["Not at all: the reply is in standard English", "A little: a few of the user's words or spellings", "Clearly: the user's grammar, or the user's style throughout"]
assert INSTR_F in proto and INSTR_A in proto and all(k in notes for k in NOTE_KEY.values()), "protocol text mismatch"
items, used = [], set()
for key, n in ALLOC.items():
    v, rec = FEAT[key]
    pv = [r for r in pool if r["variety"] == v and r["pool_id"] not in used]
    pos = [r for r in pv if rec(r["text"])]; neg = [r for r in pv if not rec(r["text"])]; rng.shuffle(pos); rng.shuffle(neg)
    npos = n // 2 + (n % 2 if rng.random() < 0.5 else 0); chosen = pos[:npos] + neg[:n - min(len(pos), npos)]
    if len(chosen) < n: chosen += [r for r in (pos[npos:] + neg[n - min(len(pos), npos):]) if r not in chosen][:n - len(chosen)]; short.append((key, n, len(chosen)))
    for r in chosen[:n]:
        used.add(r["pool_id"]); items.append({"kind": "feature", "pool": "feature_pool", "pool_id": r["pool_id"], "instruction": INSTR_F, "text": r["text"], "options": OPT_F, "note": notes[NOTE_KEY[key]], "_key": key})
strata = {c: 5 for c in conds} if len(conds) == 4 else {c: 20 // len(conds) for c in conds}
for c, n in strata.items():
    pc = [r for r in accpool if r["condition"] == c]; rng.shuffle(pc)
    for r in pc[:n]:
        items.append({"kind": "accommodation", "pool": "accommodation_pool", "pool_id": r["pool_id"], "instruction": INSTR_A, "text": f"USER: {r['user']}\nREPLY: {r['reply']}", "options": OPT_A, "note": "", "_key": "accommodation:" + c})
rng.shuffle(items)
for i, it in enumerate(items): it["id"] = f"A{i + 1:03d}"
with open(EXP / "gold" / "HUMAN_AUDIT_TODO.jsonl", "w") as f:
    for it in items: f.write(json.dumps({k: it[k] for k in ("id", "instruction", "text", "options", "note")}, ensure_ascii=False) + "\n")
words = [len(it["text"].split()) for it in items]
minutes = 5 + 40 * 20 / 60 + 20 * 45 / 60
json.dump({"seed": SEED, "n_items": len(items), "n_feature_items": 40, "n_accommodation_items": len(items) - 40, "general_instructions": gen_instr,
           "estimated_minutes": round(minutes, 1), "max_item_words": max(words), "strata": {"feature_items_per_feature": ALLOC, "accommodation_items_per_condition": strata,
           "feature_pool": "50 sentences x 5 varieties, ~50% detector-positive", "accommodation_pool": f"{len(accpool)} pairs = {len(conds)} conditions x 3 stage classes x {per}"},
           "pools": {"feature_pool": "exp/gold/pools/feature_pool.jsonl", "accommodation_pool": "exp/gold/pools/accommodation_pool.jsonl"},
           "item_provenance": {it["id"]: {"pool": it["pool"], "pool_id": it["pool_id"], "key": it["_key"]} for it in items},
           "gen_files_used": [Path(f).name for f in gen_files], "trim": "user turn and reply each trimmed to 60 words; trims flagged per pair in the accommodation pool",
           "pii_masked_counts": pii, "profanity_blocklist": "sentences matching a profanity/slur blocklist were excluded from the pools", "shortages": short,
           "blindness": "item ids are random A001..A060; fields carry no model, stage, condition or variety names; notes come verbatim from ANNOTATION_PROTOCOL.md §4.3",
           "sources_authentic": ["Groenwold et al. 2020 AAVE tweets (research use, not redistributed)", "NUS SMS Corpus (Singapore senders)", "UD-Naija NSC (text_ortho)"]},
          open(EXP / "gold" / "HUMAN_AUDIT_meta.json", "w"), indent=1, ensure_ascii=False)
print("items", len(items), "max words", max(words), "est min", round(minutes, 1), "pool", len(pool), "accpool", len(accpool), "shortages", short, "pii", pii, "gen files", len(gen_files))
