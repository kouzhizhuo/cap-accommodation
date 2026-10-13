#!/usr/bin/env python
"""Teacher-forced log-likelihood of y given x under a model (env gen, ONE GPU). Prompt = the model's chat template (--template-from lets a base model borrow it);
response = y + EOS. Output: item, lp (sum of response-token log-probs incl. EOS), n (response tokens).
  CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 python E4/mbv_score.py --model-path M --template-from T --items E4/mbv_items.jsonl --lineage L2 --out E4/mbv_lp/<email>"""
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model-path"); ap.add_argument("--template-from", default=None); ap.add_argument("--items"); ap.add_argument("--lineage")
ap.add_argument("--out"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--tokenizer", default=None); ap.add_argument("--multi-gpu", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true"); ap.add_argument("--max-len", type=int, default=1100); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model_path, trust_remote_code=a.trust_remote_code); tpl = AutoTokenizer.from_pretrained(a.template_from, trust_remote_code=a.trust_remote_code) if a.template_from else tok
model = (AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=a.trust_remote_code) if a.multi_gpu else AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, trust_remote_code=a.trust_remote_code).cuda()).eval()
rows = [json.loads(l) for l in open(a.items)]; rows = [r for r in rows if r["lineage"] == a.lineage]
bos = tok.bos_token_id; eos = tok.eos_token_id; data = []
for r in rows:
    p = tpl.apply_chat_template([{"role": "user", "content": r["x"]}], tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False).input_ids
    if bos is not None and (not pi or pi[0] != bos): pi = [bos] + pi
    yi = tok(r["y"], add_special_tokens=False).input_ids + [eos]
    if len(pi) + len(yi) <= a.max_len: data.append((r["item"], pi, yi))
data.sort(key=lambda d: len(d[1]) + len(d[2])); out = []; t0 = time.time()
with torch.no_grad():
    for i in range(0, len(data), a.bs):
        b = data[i:i + a.bs]; L = max(len(p) + len(y) for _, p, y in b)
        ids = torch.full((len(b), L), tok.pad_token_id or 0); att = torch.zeros((len(b), L), dtype=torch.long); lab = torch.full((len(b), L), -100)
        for j, (_, p, y) in enumerate(b):
            s = p + y; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; lab[j, len(p):len(s)] = torch.tensor(y)
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, :-1].float(); tgt = lab[:, 1:].cuda()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * (tgt != -100)
        for j, (item, _, y) in enumerate(b): out.append({"item": item, "lp": float(lp[j].sum()), "n": len(y)})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f: f.write("".join(json.dumps(o) + "\n" for o in out))
print(f"scored {len(out)}/{len(rows)} items in {time.time() - t0:.0f}s -> {a.out}")
