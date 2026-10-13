#!/usr/bin/env python
"""Deployed-chatbot row (user approval 2026-10-02: closed models as SUBJECTS only, never judges). LOCAL job, no GPU.
an OpenAI-compatible endpoint ($OPENAI_BASE_URL) (key from $OPENAI_API_KEY, never printed). Prompts = stimuli v1: 100 seeds stratified by task type (seed 20261002) x
{SAE, INF, FOR, BRE, AUS, AAE, SGE, NGP, APP}, 1 sample, NO system prompt, provider-default temperature; plus the explicit-request condition for AAE/SGE
(system "The user writes in <variety>; reply in the same variety."). max_completion_tokens=1024. Output rows follow gen.py's schema so detectors.score applies unchanged
(lineage = model code, stage = deployed / deployed-request). Resumable; every batch is appended to logs/api_spend.md.
  python harness/run_api.py --test            # 2 prompts/model, prints the cost projection
  python harness/run_api.py --run [--models gpt53chat ...]"""
import argparse, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
MODELS = {"gpt53chat": "openai/gpt-5.3-chat", "sonnet46": "anthropic/claude-sonnet-4.6", "gemini25flash": "google/gemini-2.5-flash",
          "dsv4pro": "deepseek/deepseek-v4-pro", "kimik25": "bedrock/moonshotai.kimi-k2.5"}
CONDS = ["SAE", "INF", "FOR", "BRE", "AUS", "AAE", "SGE", "NGP", "APP"]
REQ = {"AAE": "African American English", "SGE": "Colloquial Singapore English (Singlish)"}
MAXTOK = {"gemini25flash": 4096}   # Gemini 2.5 Flash spends the budget on hidden reasoning (981 of 1024 tokens in the test): visible reply needs headroom. Others: 1024.
BASE = os.environ["OPENAI_BASE_URL"]; KEY = os.environ.get("OPENAI_API_KEY", "")
ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--run", action="store_true"); ap.add_argument("--models", nargs="+", default=list(MODELS))
ap.add_argument("--out", default=str(EXP / "E1" / "gen_api")); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()


def load(c): return {json.loads(l)["seed_id"]: json.loads(l) for l in open(EXP / "stimuli" / "v1" / f"{c}.jsonl")}
seeds = [json.loads(l) for l in open(EXP / "stimuli" / "seeds_v1.jsonl")]; by = {}
for s in seeds: by.setdefault(s["task_type"], []).append(s["seed_id"])
rng = random.Random(20261002); types = sorted(by); quota = {t: 100 // len(types) + (1 if i < 100 % len(types) else 0) for i, t in enumerate(types)}
CHOSEN = sorted(x for t in types for x in rng.sample(sorted(by[t]), min(quota[t], len(by[t]))))
STIM = {c: load(c) for c in CONDS}; META = {s["seed_id"]: s for s in seeds}


def call(model_id, messages, max_tok=1024):
    body = json.dumps({"model": model_id, "messages": messages, "max_completion_tokens": max_tok}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read()); h = r.headers
            return d, h.get(os.environ.get("COST_HEADER", "x-response-cost"))
        except Exception as e:
            err = str(e)[:200]; time.sleep(2 ** attempt * 2)
    return {"error": err}, None


def jobs(code, limit=None):
    J = [("deployed", c, sid, None) for c in CONDS for sid in CHOSEN] + [("deployed-request", c, sid, REQ[c]) for c in REQ for sid in CHOSEN]
    return J[:limit] if limit else J


def run(code, limit=None):
    out = Path(a.out) / f"{code}@deployed.jsonl"; out.parent.mkdir(parents=True, exist_ok=True); outr = Path(a.out) / f"{code}@deployed_request.jsonl"
    done = set()
    for p in (out, outr):
        if p.exists(): done |= {(json.loads(l)["stage"], json.loads(l)["condition"], json.loads(l)["seed_id"]) for l in open(p)}
    todo = [j for j in jobs(code, limit) if (j[0], j[1], j[2]) not in done]
    stat = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0, "served": set(), "errors": 0, "t0": time.strftime("%Y-%m-%d %H:%M")}

    def one(j):
        stage, c, sid, variety = j; text = STIM[c][sid]["text"]
        msgs = ([{"role": "system", "content": f"The user writes in {variety}; reply in the same variety."}] if variety else []) + [{"role": "user", "content": text}]
        d, cost = call(MODELS[code], msgs, MAXTOK.get(code, 1024))
        if "error" in d and "choices" not in d: return j, None, None, d["error"]
        return j, d, cost, None
    with ThreadPoolExecutor(a.workers) as ex:
        for j, d, cost, err in ex.map(one, todo):
            stage, c, sid, variety = j
            if err or not d.get("choices"): stat["errors"] += 1; continue
            ch = d["choices"][0]; u = d.get("usage") or {}; txt = (ch.get("message") or {}).get("content") or ""
            stat["calls"] += 1; stat["prompt_tokens"] += u.get("prompt_tokens", 0); stat["completion_tokens"] += u.get("completion_tokens", 0)
            stat["reasoning_tokens"] += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0); stat["served"].add(d.get("model", "?"))
            try: stat["cost"] += float(cost or 0)
            except ValueError: pass
            row = {"model": f"{code}@{stage}", "lineage": code, "stage": stage, "revision": d.get("model", MODELS[code]), "condition": c, "seed_id": sid, "task_type": META[sid]["task_type"],
                   "context": META[sid]["context"], "sampling_seed": 0, "sample": 0, "scaffold": "chat", "prompt": STIM[c][sid]["text"], "response": txt.strip(), "finish": ch.get("finish_reason"),
                   "n_tokens": u.get("completion_tokens"), "temperature": "provider default", "max_tokens": MAXTOK.get(code, 1024), "system": f"request:{variety}" if variety else None,
                   "served_model": d.get("model"), "date": time.strftime("%Y-%m-%d")}
            with open(outr if variety else out, "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stat["served"] = sorted(stat["served"]); return stat


def log_spend(code, st, tag):
    p = EXP.parent / "logs" / "api_spend.md"
    if not p.exists() or "| model |" not in p.read_text(): p.write_text((p.read_text() if p.exists() else "# P3 API spend log\n") + "\n| date | batch | model (requested -> served) | params | calls | prompt tok | completion tok (reasoning) | $ (proxy cost header; 0 = not reported) |\n|---|---|---|---|---|---|---|---|\n")
    with open(p, "a") as f:
        f.write(f"| {st['t0']} | {tag} | {MODELS[code]} -> {','.join(st['served'])} | max_completion_tokens={MAXTOK.get(code, 1024)}, temperature=provider default, no system prompt (request cond: system) | {st['calls']} (+{st['errors']} errors) | {st['prompt_tokens']} | {st['completion_tokens']} ({st['reasoning_tokens']}) | {st['cost']:.4f} |\n")


if a.test:
    for code in a.models:
        st = run(code, limit=1); st2 = None
        print(code, {k: v for k, v in st.items() if k != "t0"}); log_spend(code, st, "test(1 call)")
elif a.run:
    for code in a.models:
        st = run(code); print(code, {k: v for k, v in st.items() if k != "t0"}); log_spend(code, st, "full run")
        json.dump({"served": st["served"]}, open(Path(a.out) / f"{code}_served.json", "w"))
