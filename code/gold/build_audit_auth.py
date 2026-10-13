#!/usr/bin/env python
"""20-item human spot-check of the WildChat-filtered requests (Arm N; coordinator 2026-10-04): precision of the detector filter. Blind (no model labels; ids shuffled), same item format as the other audits.
 13 AAE-flagged prompts (retained feature: ain't / negative concord / completive done), 5 SGE-flagged (discourse particle), 2 unflagged controls (asked about a random AAE note; expected No). Seed 20261005.
 Texts: first 100 words (the flagged span is kept inside the shown text), PII masked; prompts matching the profanity/slur blocklist are never shown (they stay in the generation set).
  python gold/build_audit_auth.py ; then  AUDIT_TAG=AUTH AUDIT_SHUFFLE=20261005 python infra/scripts/audit_xlsx.py to-xlsx <paper_dir>"""
import json, random, re, sys, os
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP)); sys.path.insert(0, str(EXP / "E4"))
os.environ.setdefault("MV_COREF", "0")
from prefdata_bulk import FEATS
rng = random.Random(20261005)
PROFANE = re.compile(r"(?i)\b(fuck\w*|shit\w*|bitch\w*|nigg\w*|cunt|dick|pussy|whore|slut|ass|asshole|hoe|bastard|damn|dumbass|motherfuck\w*|cock|porn|sex\w*|nude|naked|rape\w*|kill\w*|suicid\w*|hitler|nazi)\b")
proto = open(EXP / "gold" / "ANNOTATION_PROTOCOL.md").read(); notes = {}
for l in re.search(r"### 4\.3.*?\n(.*?)### 4\.4", proto, re.S).group(1).splitlines():
    m = re.match(r"\|\s*(.+?)\s*\|\s*\"(.*)\"\s*\|\s*$", l)
    if m: notes[m.group(1)] = m.group(2)
NOTE = {"AAE.ain_t": notes["*ain't* (AAE)"], "AAE.negative_concord": notes["multiple negation (AAE)"], "AAE.completive_done": notes["completive *done* (AAE)"], "SGE.particles": notes["discourse particles (SGE)"]}
INSTR = "Read the text and the feature description in the note. Does the feature occur in the text at least once?"
def load(c): return [json.loads(l) for l in open(EXP / "stimuli" / "armN" / f"{c}.jsonl")]
def mask(t):
    t = re.sub(r"@\w+", "[NAME]", t); t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", "[URL]", t); t = re.sub(r"\+?\d[\d\s\-]{6,}\d", "[NUMBER]", t); return re.sub(r"\s+", " ", t).strip()
def show(t, feat):
    w = mask(t).split(); s = " ".join(w[:100]); return s if len(FEATS[feat](s)) > 0 else None
def pick(cond, feats, n):
    pool = []
    for r in load(cond):
        if PROFANE.search(r["text"]): continue
        for f in feats:
            if len(FEATS[f](r["text"])) > 0:
                s = show(r["text"], f)
                if s: pool.append((r["seed_id"], f, s)); break
    rng.shuffle(pool); return pool[:n]
items = []
for sid, f, s in pick("AAEN", ["AAE.ain_t", "AAE.negative_concord", "AAE.completive_done"], 13): items.append({"src": "AAEN", "sid": sid, "key": f, "flagged": True, "text": s, "note": NOTE[f]})
for sid, f, s in pick("SGEN", ["SGE.particles"], 5): items.append({"src": "SGEN", "sid": sid, "key": f, "flagged": True, "text": s, "note": NOTE[f]})
ctrl = [r for r in load("CTRL") if not PROFANE.search(r["text"])]; rng.shuffle(ctrl)
for r in ctrl[:2]:
    f = rng.choice(["AAE.ain_t", "AAE.negative_concord"]); items.append({"src": "CTRL", "sid": r["seed_id"], "key": f, "flagged": False, "text": " ".join(mask(r["text"]).split()[:100]), "note": NOTE[f]})
rng.shuffle(items)
for i, it in enumerate(items): it["id"] = f"N{i + 1:03d}"
g = EXP / "gold"
with open(g / "HUMAN_AUDIT_AUTH_TODO.jsonl", "w") as f:
    for it in items: f.write(json.dumps({"id": it["id"], "instruction": INSTR, "text": it["text"], "options": ["Yes", "No", "Unsure"], "note": it["note"]}, ensure_ascii=False) + "\n")
json.dump({"seed": 20261005, "n_items": len(items), "purpose": "precision of the WildChat dialect-feature filter (Arm N); 13 AAE-flagged, 5 SGE-flagged, 2 unflagged controls",
           "general_instructions": "Thank you for labelling these 20 short items (about 8 minutes). Each item is a real user message and the description of one grammatical feature of English; decide whether the feature occurs in the message at least once. Some messages may contain informal language. Nothing in the items tells you the answer; there are no trick items. If you cannot decide, choose Unsure. Please do not look anything up.",
           "item_provenance": {it["id"]: {"set": it["src"], "seed_id": it["sid"], "feature": it["key"], "flagged_by_detector": it["flagged"]} for it in items},
           "source": "allenai/WildChat-1M (ODC-BY) first user turns; stimuli/armN/*.jsonl", "decision_rule": "filter precision = share of detector-flagged items with human Yes (Wilson 95% CI); controls expected No"}, open(g / "HUMAN_AUDIT_AUTH_meta.json", "w"), indent=1, ensure_ascii=False)
print(len(items), {k: sum(1 for i in items if i["src"] == k) for k in ("AAEN", "SGEN", "CTRL")})
