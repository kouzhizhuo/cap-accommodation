#!/usr/bin/env python
"""Blind open-weight judges on the V4 human-audit items (lead <agent> spec, 2026-10-03). Input: ONLY the TODO file (id, instruction, note, text, options) + the `general_instructions` field of the meta (system prompt, verbatim);
never labels, provenance or detector scores. vLLM, temperature 0.  Output: <outdir>/<tag>.jsonl rows {"id","answer" (verbatim option),"letter" (A/B/C by option index),"raw" (first 300 chars)} + <outdir>/meta_<tag>.json.
  user = instruction + ("Feature: " + note) + text + "Options: " + options, then: 'Answer with exactly one option as JSON {"answer": "<option>"}.'   One retry (with a reminder turn) if unparseable, else answer=null.
  Qwen3-8B: enable_thinking=False.  gpt-oss-20b: harmony chat template, reasoning_effort low (final channel parsed)."""
import argparse, datetime, json, re
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", required=True); ap.add_argument("--harmony", action="store_true"); ap.add_argument("--gpu-util", type=float, default=0.80); a = ap.parse_args()
import vllm
from vllm import LLM, SamplingParams
items = [json.loads(l) for l in open(a.items)]
SYSTEM = json.load(open(a.meta))["general_instructions"]      # the only field read from the meta
INSTR_TAIL = 'Answer with exactly one option as JSON {"answer": "<option>"}.'


def user_msg(it):
    s = it["instruction"] + "\n\n"
    if it.get("note"): s += "Feature: " + it["note"] + "\n\n"
    return s + it["text"] + "\n\nOptions: " + " | ".join(it["options"]) + "\n\n" + INSTR_TAIL


def parse(raw, options):
    t = raw
    if a.harmony:
        m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", raw, re.S); t = m.group(1) if m else raw
    for m in re.finditer(r"\{[^{}]*\}", t):
        try:
            ans = json.loads(m.group(0)).get("answer")
        except Exception: continue
        if isinstance(ans, str):
            for o in options:
                if ans.strip().lower() == o.lower(): return o
    return None


llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, seed=0, trust_remote_code=True, **({} if a.harmony else {"dtype": "bfloat16"}))
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=1024 if a.harmony else 128, skip_special_tokens=False)


def render(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **({"reasoning_effort": "low"} if a.harmony else {"enable_thinking": False}))


base = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(it)}] for it in items]
outs = llm.generate([render(m) for m in base], sp); raws = [o.outputs[0].text for o in outs]
ans = [parse(r, it["options"]) for r, it in zip(raws, items)]
retry = [i for i, x in enumerate(ans) if x is None]
if retry:   # one retry: same conversation + the model's reply + a reminder
    msgs = [base[i] + [{"role": "assistant", "content": raws[i].split("<|end|>")[0][-300:] if not a.harmony else ""}, {"role": "user", "content": "Reply with only the JSON object " + '{"answer": "<option>"}' + " using one option exactly as written."}] for i in retry]
    outs2 = llm.generate([render(m) for m in msgs], sp)
    for i, o in zip(retry, outs2):
        r2 = o.outputs[0].text; x = parse(r2, items[i]["options"])
        if x is not None: ans[i], raws[i] = x, r2
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
with open(out / f"{a.tag}.jsonl", "w") as f:
    for it, x, r in zip(items, ans, raws):
        f.write(json.dumps({"id": it["id"], "answer": x, "letter": None if x is None else "ABC"[it["options"].index(x)], "raw": r[:300]}, ensure_ascii=False) + "\n")
rev = None
mm = Path(a.model) / ".cache/huggingface/download/config.json.metadata"
if mm.exists(): rev = mm.read_text().splitlines()[0]
json.dump({"tag": a.tag, "model_path": a.model, "hf_revision": rev, "vllm": vllm.__version__, "date": datetime.date.today().isoformat(), "n_items": len(items), "n_unparseable": sum(x is None for x in ans),
           "n_retried": len(retry), "decoding": {"temperature": 0.0, "max_tokens": sp.max_tokens, "seed": 0, "enable_thinking": None if a.harmony else False, "reasoning_effort": "low" if a.harmony else None},
           "system_prompt": "meta.general_instructions verbatim (HUMAN_AUDIT_V4_meta.json)", "user_template": 'instruction + "\\n\\n" + ("Feature: " + note + "\\n\\n" if note) + text + "\\n\\nOptions: " + " | ".join(options) + "\\n\\n" + ' + repr(INSTR_TAIL),
           "retry": "one retry with a reminder turn if the JSON answer is unparseable or not an exact option"}, open(out / f"meta_{a.tag}.json", "w"), indent=1)
print(a.tag, "done", len(items), "unparseable", sum(x is None for x in ans))
