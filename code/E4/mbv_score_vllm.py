#!/usr/bin/env python
"""vLLM variant of E4/mbv_score.py (same semantics, ~50x faster at scale): lp = sum over response tokens (y + EOS) of log pi(token | chat-template prompt + preceding response tokens), n = number of response tokens.
Uses prompt_logprobs of one forward pass over prompt+response (tensor parallel via --tp). Prompt = the model's chat template (--template-from lets a base model borrow it); BOS handling as in mbv_score.py.
  python E4/mbv_score_vllm.py --model-path M [--template-from T] [--tokenizer TOK] --items E4/mbv_items_big.jsonl --lineage L5 --tp 4 --out E4/mbv_lp_big/<email>"""
import argparse, json, time
from pathlib import Path
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model-path", required=True); ap.add_argument("--template-from", default=None); ap.add_argument("--tokenizer", default=None); ap.add_argument("--items", required=True)
ap.add_argument("--lineage", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tp", type=int, default=1); ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--gpu-util", type=float, default=0.88); ap.add_argument("--enforce-eager", action="store_true"); ap.add_argument("--max-num-seqs", type=int, default=64); ap.add_argument("--trust-remote-code", action="store_true"); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model_path, trust_remote_code=a.trust_remote_code)
tpl = AutoTokenizer.from_pretrained(a.template_from, trust_remote_code=a.trust_remote_code) if a.template_from else tok
rows = [json.loads(l) for l in open(a.items)]; rows = [r for r in rows if r["lineage"] == a.lineage][: a.limit]
bos, eos = tok.bos_token_id, tok.eos_token_id; data = []
for r in rows:
    p = tpl.apply_chat_template([{"role": "user", "content": r["x"]}], tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False).input_ids
    if bos is not None and (not pi or pi[0] != bos): pi = [bos] + pi
    yi = tok(r["y"], add_special_tokens=False).input_ids + [eos]
    if len(pi) + len(yi) <= a.max_len: data.append((r["item"], pi, yi))
llm = LLM(a.model_path, tokenizer=a.tokenizer or a.model_path, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, dtype="bfloat16", max_model_len=a.max_len + 8, seed=0,
          trust_remote_code=a.trust_remote_code, enable_prefix_caching=False, max_num_seqs=a.max_num_seqs, enforce_eager=a.enforce_eager)
t0 = time.time(); sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
outs = llm.generate([{"prompt_token_ids": pi + yi} for _, pi, yi in data], sp)
res = []
for (item, pi, yi), o in zip(data, outs):
    pl = o.prompt_logprobs; tot = 0.0
    for k, t in enumerate(yi):
        d = pl[len(pi) + k]; tot += d[t].logprob
    res.append({"item": item, "lp": tot, "n": len(yi)})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f: f.write("".join(json.dumps(o) + "\n" for o in res))
print(f"scored {len(res)}/{len(rows)} items in {time.time() - t0:.0f}s -> {a.out}")
