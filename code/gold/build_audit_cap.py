#!/usr/bin/env python
"""40-reply blind human check of CAP naturalness (D24 §5 "no caricature"; the user agreed 2026-10-05 ~20:10; lead <agent>, 2026-10-05,
written before any CAP generation existed). 20 replies from the CAP arm and 20 from the one-sided recipe (UNMATCHED / B5), both from
the same model, on F300 AAE and Singlish requests: per arm 10 AAE + 10 SGE, stratified by tertile of the user's density over
every counted feature of the variety (tertiles pooled over the F300 AAE/SGE prompts), distinct requests across arms (no request is shown twice), sampling seed 0; replies cut at the 400-token cap are kept
(only the first 150 words are shown), so the sample does not depend on reply length.
Blind: no arm, model or score shown; ids shuffled. Texts: the user's message (first 100 words) and the reply (first 150 words), PII
masked; items matching the profanity/slur blocklist are never shown.
Options (user decision via the coordinator, 2026-10-05): Natural / Exaggerated / Unclear. Decision rule (pre-declared, descriptive diagnostic,
never a gate): exaggerated share CAP - UNMATCHED with a Newcombe 95% CI; "Unclear" excluded (primary) and counted as not exaggerated
(sensitivity); agreement with the two open-weight judges reported (kappa).
  python gold/build_audit_cap.py --cap Q3/gen_f300/Q3@CAP_s0.jsonl --ref Q3/gen_f300/Q3@UNM_s0.jsonl
  then  AUDIT_TAG=CAP AUDIT_SHUFFLE=20261006 python infra/scripts/audit_xlsx.py to-xlsx <paper_dir>"""
import argparse, json, random, re, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP))
from detectors import split_items
ap = argparse.ArgumentParser(); ap.add_argument("--cap", required=True); ap.add_argument("--ref", required=True); a = ap.parse_args()
rng = random.Random(20261006)
TRAIN = {"AAE": ["AAE.morph.train", "AAE.lex.train"], "SGE": ["SGE.part.train", "SGE.lex.train"]}
COUNTED = {v: TRAIN[v] + [g.replace(".train", ".held") for g in TRAIN[v]] for v in TRAIN}   # the user's density over every counted feature of the variety
PROFANE = re.compile(r"(?i)\b(fuck\w*|shit\w*|bitch\w*|nigg\w*|cunt|pussy|whore|slut|asshole|motherfuck\w*|porn|nude|naked|rape\w*|suicid\w*)\b")
def mask(t):
    t = re.sub(r"@\w+", "[NAME]", t); t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", "[URL]", t); t = re.sub(r"\+?\d[\d\s\-]{6,}\d", "[NUMBER]", t); return re.sub(r"\s+", " ", t).strip()
def dens(text, v): return 100.0 * sum(len(split_items.spans_of(g, text)) for g in COUNTED[v]) / max(1, len(text.split()))
def rows(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r["condition"] in TRAIN and r.get("sampling_seed") == 0 and r.get("sample", 0) == 0 \
           and not PROFANE.search(r["prompt"] + " " + r["response"]): out[(r["condition"], r["seed_id"])] = r
    return out
C, R = rows(a.cap), rows(a.ref)
allp = {k: dens(r["prompt"], k[0]) for k, r in {**R, **C}.items()}
cut = {}
for v in TRAIN:
    d = sorted(x for (vv, _), x in allp.items() if vv == v); cut[v] = (d[len(d) // 3], d[2 * len(d) // 3])
tert = lambda v, x: 0 if x <= cut[v][0] else (1 if x <= cut[v][1] else 2)
used, items = set(), []
for arm, src in (("CAP", C), ("UNMATCHED", R)):
    for v in ("AAE", "SGE"):
        pool = [k for k in src if k[0] == v and k[1] not in used]; rng.shuffle(pool)
        per = [[k for k in pool if tert(v, allp[k]) == t] for t in range(3)]; take = []
        for i in range(10): take.append(per[i % 3].pop()) if per[i % 3] else None
        for k in take:
            used.add(k[1]); r = src[k]
            items.append({"arm": arm, "run": r["model"], "seed_id": k[1], "variety": v, "user_density": allp[k], "tertile": tert(v, allp[k]),
                          "text": "USER: " + " ".join(mask(r["prompt"]).split()[:100]) + "\n\nREPLY: " + " ".join(mask(r["response"]).split()[:150])})
rng.shuffle(items)
for i, it in enumerate(items): it["id"] = f"C{i + 1:03d}"
INSTR = "Read the user's message and the reply. Is the reply's language natural for this user, or exaggerated?"
NOTE = ("Natural: the reply reads as a natural answer to this user, whether it stays in Standard English or takes up some of the user's words or grammar "
        "about as much as the user does. Exaggerated: it uses the user's variety more, or more stereotypically, than the user does (caricature). Unclear: you cannot tell.")
g = EXP / "gold"
with open(g / "HUMAN_AUDIT_CAP_TODO.jsonl", "w") as f:
    for it in items: f.write(json.dumps({"id": it["id"], "instruction": INSTR, "text": it["text"], "options": ["Natural", "Exaggerated", "Unclear"], "note": NOTE}, ensure_ascii=False) + "\n")
json.dump({"seed": 20261006, "n_items": len(items), "purpose": "D24 no-caricature check: CAP vs the one-sided recipe (UNMATCHED), same model, F300 AAE/SGE requests",
           "general_instructions": "Thank you for labelling these 40 short items (about 10 minutes). Each item shows a user's message and a chatbot reply; decide how the reply's language relates to the way the user writes.",
           "item_provenance": {it["id"]: {k: it[k] for k in ("arm", "run", "seed_id", "variety", "user_density", "tertile")} for it in items},
           "tertile_cuts_per100w": cut, "decision_rule": "exaggerated share CAP - UNMATCHED, Newcombe 95% CI; Unclear excluded (primary) / counted as not exaggerated (sensitivity); descriptive diagnostic, never a gate; judge agreement (kappa) reported",
           "annotator": "one author with doctoral-level training, only if the user agrees; never described as community validation"}, open(g / "HUMAN_AUDIT_CAP_meta.json", "w"), indent=1)
print(len(items), {arm: sum(1 for i in items if i["arm"] == arm) for arm in ("CAP", "UNMATCHED")})
