"""Opportunity checker for SAE seeds (plan §2.4): >=2 copula/aux sites, >=1 3sg present verb, >=1 negation, >=1 question.
Env p3mv (spaCy en_core_web_sm). Usage: python seeds_check.py seeds_v0.jsonl"""
import json, sys
import spacy

nlp = spacy.load("en_core_web_sm")


def opportunities(text):
    doc = nlp(text)
    return {"aux_copula": sum(t.pos_ == "AUX" or (t.lemma_ == "be" and t.dep_ in {"ROOT", "cop", "aux", "auxpass"}) for t in doc),
            "third_sg_present": sum(t.tag_ == "VBZ" and t.pos_ == "VERB" for t in doc),
            "negation": sum(t.dep_ == "neg" or t.lower_ in {"never", "nobody", "nothing", "no"} for t in doc),
            "question": text.count("?")}


def passes(o):
    return o["aux_copula"] >= 2 and o["third_sg_present"] >= 1 and o["negation"] >= 1 and o["question"] >= 1


if __name__ == "__main__":
    bad = 0
    for l in open(sys.argv[1]):
        r = json.loads(l); o = opportunities(r["text"])
        if not passes(o):
            bad += 1; print("FAIL", r["seed_id"], o, "|", r["text"][:90])
    print(f"{bad} failing")
