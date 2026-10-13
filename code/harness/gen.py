#!/usr/bin/env python
"""P3 generation (vLLM). One model-stage x condition files x sampling seeds -> JSONL + .meta.json. Env `gen`.
<host>: CUDA_VISIBLE_DEVICES=5 ONLY.  Always: HF_HUB_OFFLINE=1 (local paths) VLLM_USE_FLASHINFER_SAMPLER=0.

  python gen.py --model-path $M/L1/sft --lineage L1 --stage sft --stimuli exp/stimuli/v0 --conds SAE INF AAE SGE \
      --scaffold chat --seeds 0 1 2 --n 2 --out exp/P0/gen/L1@sft

Scaffolds (plan §5):
  chat : the lineage's chat template (aligned stages).  `--template-from` lets a base model borrow it (F1).
  F1   : the lineage's chat template applied to a BASE model (needs --template-from <aligned sibling dir>).
  F2   : "The following is a conversation between a user and a helpful assistant.\n\nUser: {p}\n\nAssistant:" + stop strings.
  raw  : the prompt text itself; the model continues it (Blevins-style continuation).
Stimuli files: <stimuli>/<COND>.jsonl with fields seed_id, condition, text (extra fields are carried into the output).
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import env_info, file_sha1, read_jsonl, request_seed

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True); ap.add_argument("--lineage", required=True); ap.add_argument("--stage", required=True)
ap.add_argument("--revision", default="main"); ap.add_argument("--template-from", default=None)
ap.add_argument("--trust-remote-code", action="store_true"); ap.add_argument("--chat-template-kwargs", default="{}", help="JSON, e.g. {\"enable_thinking\": false} for Qwen3")
ap.add_argument("--tokenizer", default=None, help="tokenizer dir when the model repo ships none (OLMo-2-32B SFT/Instruct)")
ap.add_argument("--stimuli", required=True, help="dir with <COND>.jsonl"); ap.add_argument("--conds", nargs="+", required=True)
ap.add_argument("--scaffold", choices=["chat", "F1", "F2", "raw"], required=True)
ap.add_argument("--system", default=None, help="system prompt (chat/F1 only), used by the persona / competence controls")
ap.add_argument("--prefix-msgs", default=None, help="JSON file: list of {role, content} turns inserted before the user turn (few-shot demonstrations, baseline B4)")
ap.add_argument("--user-suffix", default="", help="text appended to the user turn (competence / style instruction conditions)")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--n", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--max-tokens", type=int, default=400); ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--gpu-util", type=float, default=0.80); ap.add_argument("--max-num-seqs", type=int, default=None); ap.add_argument("--enforce-eager", action="store_true"); ap.add_argument("--keep-special", action="store_true", help="skip_special_tokens=False"); ap.add_argument("--stop-token-ids", type=int, nargs="*", default=None); ap.add_argument("--pangu-parse", action="store_true", help="openPangu: response = answer after [unused17] (official generate.py parsing), thinking kept in `thinking`"); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--out", required=True, help="output prefix (no extension)")
a = ap.parse_args()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if a.lineage == "E5":   # E5 runs are also evaluated on the generalisation varieties whenever their stimulus files exist (read fresh by every gen invocation)
    a.conds = list(a.conds) + [c for c in ("NGP", "APP", "FOR") if c not in a.conds and (Path(a.stimuli) / f"{c}.jsonl").exists()]
rows = []
for c in a.conds:
    rows += read_jsonl(Path(a.stimuli) / f"{c}.jsonl")[: a.limit]
llm = LLM(a.model_path, tokenizer=a.tokenizer, trust_remote_code=a.trust_remote_code, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, dtype="bfloat16", enforce_eager=a.enforce_eager, **({"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}),
          max_model_len=a.max_model_len, seed=0)
tok = llm.get_tokenizer()
tpl_tok = AutoTokenizer.from_pretrained(a.template_from, trust_remote_code=a.trust_remote_code) if a.template_from else tok
CTK = json.loads(a.chat_template_kwargs)
PREFIX = json.load(open(a.prefix_msgs)) if a.prefix_msgs else []
F2_HEAD = "The following is a conversation between a user and a helpful assistant.\n\n"


def render(text):
    text = text + a.user_suffix
    if a.scaffold in ("chat", "F1"):
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + PREFIX + [{"role": "user", "content": text}]
        s = tpl_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CTK)
    elif a.scaffold == "F2":
        s = F2_HEAD + f"User: {text}\n\nAssistant:"
    else:
        s = text + "\n\n"
    ids = tok(s, add_special_tokens=False).input_ids
    if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
    return s, ids


rendered = [render(r["text"]) for r in rows]
stop = ["\nUser:", "\n\nUser", "\nHuman:"] if a.scaffold in ("F2", "raw") else None
reqs, sps, keys = [], [], []
for seed in a.seeds:
    for i, (s, ids) in enumerate(rendered):
        reqs.append({"prompt_token_ids": ids})
        sps.append(SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, stop=stop,
                                  seed=request_seed(seed, i), **({"skip_special_tokens": False} if a.keep_special else {}), **({"stop_token_ids": a.stop_token_ids} if a.stop_token_ids else {})))
        keys.append((seed, i))
t0 = time.time(); outs = llm.generate(reqs, sps); dt = time.time() - t0
ntok = sum(len(c.token_ids) for o in outs for c in o.outputs)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out + ".jsonl", "w") as f:
    for (seed, i), o in zip(keys, outs):
        r = rows[i]
        for j, c in enumerate(o.outputs):
            resp, extra = c.text.strip(), {}
            if a.pangu_parse:     # official openPangu parsing (inference/generate.py)
                think = c.text.split("[unused17]")[0].split("[unused16]")[-1].strip() if "[unused17]" in c.text else None
                resp = c.text.split("[unused17]")[-1].split("[unused10]")[0].replace("[unused16]", "").strip()
                extra = {"thinking": think, "has_think_close": "[unused17]" in c.text}
            f.write(json.dumps({**extra, **{"model": f"{a.lineage}@{a.stage}", "lineage": a.lineage, "stage": a.stage, "revision": a.revision,
                                "condition": r["condition"], "seed_id": r["seed_id"], "task_type": r.get("task_type"),
                                "context": r.get("context"), "sampling_seed": seed, "sample": j, "scaffold": a.scaffold,
                                "prompt": r["text"], "response": resp, "finish": c.finish_reason,
                                "n_tokens": len(c.token_ids), "temperature": a.temperature, "top_p": a.top_p,
                                "max_tokens": a.max_tokens, "system": a.system, "user_suffix": a.user_suffix, "prefix_msgs": PREFIX or None}},
                               ensure_ascii=False) + "\n")
meta = {"args": vars(a), "n_requests": len(reqs), "n_outputs": sum(len(o.outputs) for o in outs), "gen_tokens": ntok,
        "seconds": round(dt, 1), "tok_per_s": round(ntok / dt), "stimuli_sha1": {c: file_sha1(Path(a.stimuli) / f"{c}.jsonl") for c in a.conds},
        "example_rendered_prompt": rendered[0][0], "env": env_info()}
json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
print(f"[done] {a.lineage}@{a.stage}/{a.scaffold}: {meta['n_outputs']} outputs, {ntok} tokens, {dt:.0f}s, {ntok/dt:.0f} tok/s", flush=True)
