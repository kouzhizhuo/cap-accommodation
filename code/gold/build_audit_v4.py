#!/usr/bin/env python
"""Build the V4 blind human-audit batch (lead 2026-10-02): 40 formality pairs (V3-A spec) + 30 on-request feature items (V3-B spec) + 20 main-reply feature items (detector-positive). Env p3mv on <host> (needs the E1 gens); SEED 20261004; items already used in V3 are excluded.
(The V3 docstring follows.) V3: second blind human-audit file (notes/ANNOTATION_PROTOCOL.md §7); SEED 20261003.
Part A (24): pairwise formality. SFT replies of L2 (OLMo-2-7B) and L3 (OLMo-3-7B), same seed_id / sampling_seed 0 / sample 0, one reply to a vernacular prompt
  (AAE/SGE/NGP), one to the INF prompt; 8 pairs per |dF| tercile (dF = F_vern - F_INF on the FULL replies, Heylighen-Dewaele F; the GYAFC ranker was not run),
  balanced 4 L2 / 4 L3 per tercile, varieties 3/3/2 rotated; left/right randomised; each reply trimmed to ~80 words.
Part B (24): feature presence on on-request replies (L2@dpo-request, L4@dpo-request; condition AAE/SGE), 12 AAE + 12 SGE, one retained feature per item,
  50% detector-positive on the TRIMMED text (hidden), models alternated where both have candidates (all 3 sampling seeds pooled: SGE positives are rare); each reply trimmed to ~80 words.
Instruction/options text for Part B = v1 §4.2 feature wording (verbatim from the v1 protocol section); notes come from ANNOTATION_PROTOCOL §4.3 where present, otherwise from
gold/V3_NOTES_PROPOSED.json (flagged `note_status: proposed` in meta until the writer approves them)."""
import json, random, re, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP))
import detectors  # noqa: F401
import detectors.aae as A, detectors.sge as S
from detectors.score import hd_formality
from detectors.shared import nlp

SEED = 20261004
rng = random.Random(SEED)
V3 = json.load(open(Path(__file__).parent / "HUMAN_AUDIT_V3_meta.json"))["item_provenance"].values()
USED_A = {(p["lineage"], p["seed_id"]) for p in V3 if "tercile" in p}; USED_B = {(p["lineage"], p["seed_id"], p["feature"]) for p in V3 if "feature" in p}
P = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP.parents[0] / "runs" / "P3"    # runs/P3 root
GEN, REQ = P / "E1" / "gen", P / "E1" / "gen_request"
PROTO = EXP / "gold" / "ANNOTATION_PROTOCOL.md"
proto = open(PROTO).read()
PROFANE = re.compile(r"(?i)\b(fuck\w*|shit\w*|bitch\w*|nigg\w*|cunt|dick|pussy|whore|slut|asshole|bastard|motherfuck\w*|porn)\b")
pii = {"handle": 0, "url": 0, "number": 0}


def mask(t):
    t, a = re.subn(r"@\w+", "[NAME]", t); t, b = re.subn(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", "[URL]", t); t, c = re.subn(r"\+?\d[\d\s\-]{6,}\d", "[NUMBER]", t)
    pii["handle"] += a; pii["url"] += b; pii["number"] += c
    return t


def trim(t, n=80):
    """first ~n words, cut at the last sentence end after >= n*0.6 words if possible; keeps line breaks out (one paragraph)."""
    t = re.sub(r"[*#`]+", "", mask(t)); t = re.sub(r"\s+", " ", t).strip(); w = t.split()
    if len(w) <= n: return t, False
    cut = " ".join(w[:n]); m = list(re.finditer(r"[.!?](?=\s|$)", cut))
    if m and len(cut[:m[-1].end()].split()) >= int(n * 0.6): cut = cut[:m[-1].end()]
    return cut + " …", True


def load(fn):
    return [json.loads(l) for l in open(fn)]


def ok(r):
    return r["finish"] != "error" if "finish" in r else True


def usable(r):
    t = r["response"]; return 40 <= len(t.split()) and not PROFANE.search(t) and sum(c.isascii() for c in t) > 0.95 * len(t)


# ------------------------------------------------------------------ Part A
INSTR_A = "Which reply is written in a more formal style? Judge only the style, not the content."
OPT_A = ["Reply A is more formal", "Reply B is more formal", "About the same"]
rows = {}
for lin in ("L2", "L3"):
    for r in load(GEN / f"{lin}@sft.jsonl"):
        if r["sampling_seed"] == 0 and r["sample"] == 0 and r["condition"] in ("INF", "AAE", "SGE", "NGP") and usable(r):
            rows[(lin, r["condition"], r["seed_id"])] = r
for k in ("NGP",):   # NGP gens live in the extension files
    for lin in ("L2", "L3"):
        fn = GEN / f"{lin}@sft__{k}.jsonl"
        if fn.exists():
            for r in load(fn):
                if r["sampling_seed"] == 0 and r["sample"] == 0 and r["condition"] == k and usable(r): rows[(lin, k, r["seed_id"])] = r
pairs = []
for (lin, c, sid), r in rows.items():
    if c == "INF": continue
    inf = rows.get((lin, "INF", sid))
    if inf is None: continue
    fv, fi = hd_formality(nlp()(r["response"])), hd_formality(nlp()(inf["response"]))
    if fv is None or fi is None: continue
    pairs.append({"lin": lin, "variety": c, "seed_id": sid, "rv": r, "ri": inf, "dF": fv - fi, "Fv": fv, "Fi": fi})
absd = sorted(abs(p["dF"]) for p in pairs); q1, q2 = absd[len(absd) // 3], absd[2 * len(absd) // 3]
for p in pairs: p["tercile"] = 0 if abs(p["dF"]) <= q1 else 1 if abs(p["dF"]) <= q2 else 2
itemsA, shortA = [], []
VAR_ROT = [["AAE", "SGE", "NGP"], ["SGE", "NGP", "AAE"], ["NGP", "AAE", "SGE"]]
for t in range(3):
    plan = []   # 13/13/14 per tercile; models alternate; varieties cycle (AAE,SGE,NGP), offset by tercile
    vs = (VAR_ROT[t] * 5)[:(13, 13, 14)[t]]
    for i, v in enumerate(vs): plan.append(("L2" if i % 2 == 0 else "L3", v))
    used = set()
    for lin, v in plan:
        cand = [p for p in pairs if p["tercile"] == t and p["lin"] == lin and p["variety"] == v and p["seed_id"] not in used and (p["lin"], p["seed_id"]) not in USED_A]
        if not cand: shortA.append((t, lin, v)); continue
        p = rng.choice(cand); used.add(p["seed_id"]); USED_A.add((p["lin"], p["seed_id"])); p["_t"] = t; itemsA.append(p)
shortA = shortA
outA = []
for p in itemsA:
    tv, trv = trim(p["rv"]["response"]); ti, tri = trim(p["ri"]["response"])
    flip = rng.random() < 0.5            # True: vernacular-prompt reply is shown as A
    a, b = (tv, ti) if flip else (ti, tv)
    Ft = {"vern": hd_formality(nlp()(tv.replace(" …", ""))), "inf": hd_formality(nlp()(ti.replace(" …", "")))}
    outA.append({"kind": "formality_pair", "instruction": INSTR_A, "text": f"REPLY A: {a}\n\nREPLY B: {b}", "options": OPT_A, "note": "",
                 "_prov": {"lineage": p["lin"], "stage": "sft", "variety": p["variety"], "seed_id": p["seed_id"], "sampling_seed": 0, "sample": 0, "tercile": p["_t"],
                           "dF_full": round(p["dF"], 3), "F_vern_full": round(p["Fv"], 3), "F_inf_full": round(p["Fi"], 3), "F_vern_trimmed": round(Ft["vern"], 3),
                           "F_inf_trimmed": round(Ft["inf"], 3), "vern_is_A": flip, "trimmed_vern": trv, "trimmed_inf": tri,
                           "gen_files": [f"{p['lin']}@sft.jsonl", f"{p['lin']}@sft__NGP.jsonl"]}})

# ------------------------------------------------------------------ Part B
INSTR_F = "Read the text and the feature description in the note. Does the feature occur in the text at least once?"
OPT_F = ["Yes", "No", "Unsure"]
proto_notes = {}
for l in re.search(r"### 4\.3.*?\n(.*?)### 4\.4", proto, re.S).group(1).splitlines():
    m = re.match(r"\|\s*(.+?)\s*\|\s*\"(.*)\"\s*\|\s*$", l)
    if m: proto_notes[m.group(1)] = m.group(2)
assert INSTR_F in proto
proposed = json.load(open(EXP / "gold" / "V3_NOTES_PROPOSED.json"))
FEAT = {"ain_t": ("AAE", A.ain_t, "*ain't* (AAE)"), "negative_concord": ("AAE", A.negative_concord, "multiple negation (AAE)"),
        "completive_done": ("AAE", A.completive_done, "completive *done* (AAE)"),
        "can_cannot": ("SGE", S.can_cannot, None), "finish_completive": ("SGE", S.finish_completive, None)}
ALLOC = {"ain_t": 5, "negative_concord": 5, "completive_done": 5, "can_cannot": 8, "finish_completive": 7}
assert sum(ALLOC.values()) == 30
reqs = {}
for lin in ("L2", "L4"):
    for r in load(REQ / f"{lin}@dpo_request.jsonl"):
        if usable(r) and (lin, r["seed_id"]) not in {(a, b) for a, b, _ in USED_B}: reqs.setdefault(r["condition"], []).append((lin, r))    # all 3 sampling seeds: detector-positive replies are rare for SGE
note_status, outB, shortB, usedB = {}, [], [], set()
for key, n in ALLOC.items():
    v, rec, nk = FEAT[key]
    if nk and nk in proto_notes: note, st = proto_notes[nk], "protocol §4.3"
    else: note, st = proposed[key], "proposed (needs writer approval)"
    note_status[key] = st
    cand = []
    for lin, r in reqs[v]:
        t, tr = trim(r["response"]); cand.append((lin, r, t, tr, bool(rec(t.replace(" …", "")))))
    rng.shuffle(cand)
    npos = n // 2
    chosen = []
    for want, cnt in ((True, npos), (False, None)):
        if cnt is None: cnt = n - len(chosen)     # shortage of positives is filled with negatives (reported in meta shortages)
        got = []
        for lin in ("L2", "L4", "L2", "L4"):      # alternate models while both have candidates
            pool = [c for c in cand if c[4] == want and c not in got and (c[1]["seed_id"], c[0]) not in usedB and not any(c[1]["seed_id"] == g[1]["seed_id"] for g in got)]
            pm = [c for c in pool if c[0] == lin]; pick = (pm or pool)
            if pick and len(got) < cnt: got.append(pick[0])
        while len(got) < cnt:
            pool = [c for c in cand if c[4] == want and c not in got and not any(c[1]["seed_id"] == g[1]["seed_id"] for g in got)]
            if not pool: break
            got.append(pool[0])
        if len(got) < (npos if want else cnt): shortB.append((key, "positive" if want else "negative", npos if want else cnt, len(got)))
        chosen += got
    for lin, r, t, tr, det in chosen:
        usedB.add((r["seed_id"], lin))
        outB.append({"kind": "feature_on_request", "instruction": INSTR_F, "text": t, "options": OPT_F, "note": note,
                     "_prov": {"lineage": lin, "stage": "dpo-request", "variety": v, "feature": key, "seed_id": r["seed_id"], "sampling_seed": r["sampling_seed"], "sample": r["sample"],
                               "detector_positive_trimmed": det, "trimmed": tr, "note_source": st, "gen_file": f"{lin}@dpo_request.jsonl"}})

# ------------------------------------------------------------------ Part C: main-reply (no request) detector-positive feature items, aligned stages
ALLOC_C = {"ain_t": 4, "negative_concord": 4, "completive_done": 4, "can_cannot": 4, "finish_completive": 4}
assert sum(ALLOC_C.values()) == 20
mains = {}
MAIN_FILES = ["L1@dpo", "L1@instruct", "L1@rlvr1", "L2@dpo", "L2@instruct", "L2B@dpo", "L2B@instruct", "L3@dpo", "L3@instruct", "L4@dpo", "L4@final",
              "L5@dpo", "L5@instruct", "Q15@instruct", "Q7@instruct"]       # all aligned-stage main-condition gens (Q72 omitted: replies mostly truncated); positives are rare
for stem in MAIN_FILES:
    lin = stem.split("@")[0]
    for r in load(GEN / f"{stem}.jsonl"):
        if r["condition"] in ("AAE", "SGE") and usable(r): mains.setdefault(r["condition"], []).append((stem, r))
outC, shortC, usedC = [], [], set()
for key, n in ALLOC_C.items():
    v, rec, nk = FEAT[key]
    note, st = (proto_notes[nk], "protocol §4.3") if nk and nk in proto_notes else (proposed[key], "proposed (needs writer approval)")
    note_status[key] = st
    cand = []
    for lin, r in mains[v]:
        t, tr = trim(r["response"])
        if rec(t.replace(" …", "")): cand.append((lin, r, t, tr))
    rng.shuffle(cand); got = []
    for lin, r, t, tr in cand:
        if len(got) >= n: break
        if (r["seed_id"], lin) in usedC: continue
        usedC.add((r["seed_id"], lin)); got.append((lin, r, t, tr))
    if len(got) < n: shortC.append((key, n, len(got)))
    for lin, r, t, tr in got:
        outC.append({"kind": "feature_main_reply", "instruction": INSTR_F, "text": t, "options": OPT_F, "note": note,
                     "_prov": {"lineage": lin.split("@")[0], "stage": lin.split("@")[1], "variety": v, "feature": key, "seed_id": r["seed_id"], "sampling_seed": r["sampling_seed"], "sample": r["sample"],
                               "detector_positive_trimmed": True, "trimmed": tr, "note_source": st, "gen_file": f"{lin}.jsonl"}})

# ------------------------------------------------------------------ write
items = outA + outB + outC
rng.shuffle(items)
for i, it in enumerate(items): it["id"] = f"V{i + 1:03d}"
out = EXP / "gold"
with open(out / "HUMAN_AUDIT_V4_TODO.jsonl", "w") as f:
    for it in items: f.write(json.dumps({k: it[k] for k in ("id", "instruction", "text", "options", "note")}, ensure_ascii=False) + "\n")
pii.update({"handle": sum(it["text"].count("[NAME]") for it in items), "url": sum(it["text"].count("[URL]") for it in items), "number": sum(it["text"].count("[NUMBER]") for it in items)})   # counts in the delivered items
words = [len(it["text"].split()) for it in items]
minutes = 5 + 40 * 30 / 60 + 50 * 20 / 60
rules = re.search(r"\*\*Pre-declared decision rules.*", proto[proto.index("## 7."):], re.S).group(0).strip()
json.dump({"seed": SEED, "n_items": len(items), "n_partA": len(outA), "n_partB": len(outB), "n_partC": len(outC), "estimated_minutes": round(minutes, 1), "max_item_words": max(words),
           "general_instructions": ("Please choose every label yourself, without AI assistance; comments are optional. Thank you for labelling these {N} short items; it takes about {MIN} minutes. In the first kind of item you see two chatbot replies and decide which one is written "
                                    "in a more formal style, judging only the style and not the content. In the second kind you see a short text and the description of one feature of English, and "
                                    "decide whether the feature occurs in the text at least once. Nothing in the items tells you what the answer should be and there are no trick items. If you cannot decide, "
                                    "choose 'About the same' or 'Unsure' and move on. Please do not look anything up and do not go back to change earlier answers.").replace("{N}", str(len(items))).replace("{MIN}", str(int(round(minutes, -1)))),
           "general_instructions_status": "drafted by the coder from v1 §4.1; the writer owns the final wording",
           "strata": {"A": "3 |dF| terciles x 8 pairs; 4 L2 + 4 L3 per tercile; varieties AAE/SGE/NGP 3/3/2 rotated; left/right randomised",
                      "A_tercile_cuts_absdF": [round(q1, 3), round(q2, 3)], "B": ALLOC, "C": ALLOC_C, "C_source": "aligned-stage (DPO/instruct/final/rlvr1) replies of L1-L5, Qwen 7B/15B to the AAE/SGE prompts (no request), all sampling seeds; every item is detector-positive on the trimmed text (hidden), so human Yes-rate = precision on main replies; ","B_models": "L2@dpo-request and L4@dpo-request, alternated; replies from all 3 sampling seeds pooled, one reply per seed_id and feature; 50% detector-positive on trimmed text where available (see shortages)"},
           "formality_measure": "Heylighen-Dewaele F on spaCy UPOS (the GYAFC ranker has NOT been run); agreement for rule A must be computed against this F, or the ranker must be added first (flagged to the lead)",
           "pool_provenance": {"A": "runs/P3/E1/gen/{L2,L3}@sft.jsonl and @sft__NGP.jsonl, sampling_seed 0 sample 0, stimuli v1 (287 seeds)",
                               "B": "runs/P3/E1/gen_request/{L2,L4}@dpo_request.jsonl, all sampling seeds (system: 'The user writes in <variety>; reply in the same variety.')"},
           "decision_rules_verbatim": rules, "pooling_rule": "V4 items are pooled with V1/V3 items of the same type; estimates are reported pooled and by batch; the decision rules of §7 apply to the pooled estimate.", "pooling_rule_declared": "2026-10-02, before any V4 label exists", "labelling_rule": "Labels must be chosen personally by the annotator, without AI assistance; the sheet carries no prefilled answers and no AI text; only human-labelled V4/V1(user)/earlier user sheets count as human validation (LLM/GPT-6 agent labels never do).", "note_status": note_status, "shortages": {"A": shortA, "B": shortB, "C": shortC}, "trim": "each reply trimmed to ~80 words (sentence boundary when possible), markdown symbols stripped",
           "pii_masked_counts": pii, "profanity_blocklist": "replies matching a profanity/slur blocklist excluded; replies < 40 words excluded",
           "blindness": "ids V001.. are shuffled; fields carry no model, stage, condition, variety or score; provenance is only in this meta file",
           "item_provenance": {it["id"]: it["_prov"] for it in items}}, open(out / "HUMAN_AUDIT_V4_meta.json", "w"), indent=1, ensure_ascii=False)
print("items", len(items), "A", len(outA), "B", len(outB), "max words", max(words), "short", shortA, shortB, "pii", pii, "cuts", round(q1, 2), round(q2, 2))
