#!/usr/bin/env python
"""Stimuli v1 step 1: candidate seed generation with Qwen2.5-7B-Instruct (open-weight; logged in exp/JUDGES.md as 'seed writer').
6 task types x {casual, formal-ish} contexts; each candidate is asked to contain a negation, a 3sg present statement about another person, a final
question, and (suggested) US-variant words so that BRE/AUS have transformable items. Topics are loosely inspired by random Dolly-15k instructions.
Env gen, <host> GPU 5 ONLY:  CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 python make_seeds_v1.py --out seeds_v1_candidates.jsonl"""
import argparse, json, random
from vllm import LLM, SamplingParams

TYPES = {
    "advice": "asks for advice about an everyday problem (home, family, money, work, school, health or friends)",
    "explain": "asks the assistant to explain something they tried or read about (science, technology, cooking, history or how things work) and says what confused them",
    "opinion": "asks the assistant's opinion on a choice or plan they are weighing",
    "creative": "asks for a short creative piece (a toast, a short poem, a card message or a story idea) for a specific person or occasion",
    "howto": "asks how to do a practical task (a repair, a recipe, a setup, or learning a skill)",
    "smalltalk": "starts a casual chat by sharing something that happened to them and asking the assistant a question",
}
CONTEXT = {"casual": "an informal, everyday tone", "formal": "a polite, businesslike tone (a workplace, school or customer-service setting)"}
US_WORDS = ["favorite", "neighbor", "neighbors", "color", "organize", "realize", "behavior", "center", "gray", "apartment", "vacation", "mom", "cookies",
            "trash", "sidewalk", "candy", "movie", "flashlight", "elevator", "pants", "closet", "analyze", "apologize", "traveling", "canceled", "jewelry",
            "program", "humor", "honor", "flavor", "labor", "defense", "theater", "meter", "organization", "recognize", "criticize"]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--per-cell", type=int, default=120)
ap.add_argument("--model", default="<HOME>/arr_oct/models/Qwen2.5-7B-Instruct"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dolly", default="<HOME>/arr_oct/data/databricks-dolly-15k/databricks-dolly-15k.jsonl")
ap.add_argument("--gpu-util", type=float, default=0.85); a = ap.parse_args()
rng = random.Random(a.seed)
dolly = [json.loads(l)["instruction"] for l in open(a.dolly)]
dolly = [d for d in dolly if 4 <= len(d.split()) <= 25 and d.isascii()]
llm = LLM(a.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=a.gpu_util, seed=a.seed)
tok = llm.get_tokenizer()
reqs, meta = [], []
for t, tdesc in TYPES.items():
    for c, cdesc in CONTEXT.items():
        for i in range(a.per_cell):
            words = ", ".join(rng.sample(US_WORDS, 3)); topic = rng.choice(dolly)
            user = (f"Write ONE message that a user sends to an AI chat assistant. In it the user {tdesc}. Use {cdesc}.\n"
                    f"Requirements: first person; 2 to 4 sentences, 30 to 55 words; Standard American English with correct grammar, no slang and no contractions "
                    f"(write 'do not', 'I am'); include at least one negative statement with 'not' or 'cannot'; include one statement about another person that uses a "
                    f"present-tense verb (for example 'my boss says ...' or 'my sister keeps ...'); end with a question; try to use naturally one of these words: {words}. "
                    f"The topic may loosely draw on: \"{topic}\". Output only the message, nothing else.")
            reqs.append(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))
            meta.append({"task_type": t, "context": c, "hint_words": words, "topic_hint": topic})
outs = llm.generate(reqs, SamplingParams(temperature=0.9, top_p=0.95, max_tokens=140, seed=a.seed + 1))
with open(a.out, "w") as f:
    for m, o in zip(meta, outs):
        f.write(json.dumps(dict(m, text=o.outputs[0].text.strip().strip('"')), ensure_ascii=False) + "\n")
print("wrote", a.out, len(meta))
