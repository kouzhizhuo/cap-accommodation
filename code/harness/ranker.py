#!/usr/bin/env python
"""s-nlp/roberta-base-formality-ranker (trained on GYAFC + Pavlick-Tetreault; ungated, cc-by-nc-sa-4.0; label 1 = formal). Score = P(formal); texts over 512 tokens: mean of the chunk scores over 510-token chunks weighted by chunk length.
  partA :  python harness/ranker.py partA --out results/audit/ranker_partA.json         (40 V4 pairs: shown A/B (trimmed) and the full vernacular / INF replies)
  texts :  python harness/ranker.py score --gen 'E1/gen/*@*.jsonl' --out E1/ranker_scores.parquet [--device cuda]   (every reply: adds the formality_ranker column keyed like E1/scores.parquet)"""
import argparse, glob, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
EXP = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["partA", "score"]); ap.add_argument("--out", required=True); ap.add_argument("--model", default="s-nlp/roberta-base-formality-ranker")
ap.add_argument("--gen", nargs="+", default=None); ap.add_argument("--threads", type=int, default=None); ap.add_argument("--device", default="cpu"); ap.add_argument("--bs", type=int, default=64); a = ap.parse_args()
if a.threads: torch.set_num_threads(a.threads)
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
assert model.config.id2label[1] == "formal"


@torch.no_grad()
def p_formal(texts):
    chunks, owner = [], []
    for i, t in enumerate(texts):
        ids = tok(t or " ", add_special_tokens=False, truncation=False).input_ids or [tok.unk_token_id]
        for s in range(0, len(ids), 510): chunks.append(ids[s:s + 510]); owner.append(i)
    order = sorted(range(len(chunks)), key=lambda k: len(chunks[k])); score = [0.0] * len(chunks)
    for b in range(0, len(order), a.bs):
        idx = order[b:b + a.bs]; L = max(len(chunks[k]) for k in idx) + 2
        ii = torch.full((len(idx), L), tok.pad_token_id); am = torch.zeros((len(idx), L), dtype=torch.long)
        for j, k in enumerate(idx):
            s = [tok.cls_token_id] + chunks[k] + [tok.sep_token_id]; ii[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
        p = torch.softmax(model(input_ids=ii.to(a.device), attention_mask=am.to(a.device)).logits, -1)[:, 1].float().cpu().tolist()
        for j, k in enumerate(idx): score[k] = p[j]
    num, den = [0.0] * len(texts), [0] * len(texts)
    for k, o in enumerate(owner): num[o] += score[k] * len(chunks[k]); den[o] += len(chunks[k])
    return [num[i] / max(1, den[i]) for i in range(len(texts))]


if a.mode == "partA":
    G = EXP / "gold"; todo = {json.loads(l)["id"]: json.loads(l) for l in open(G / "HUMAN_AUDIT_V4_TODO.jsonl")}; prov = json.load(open(G / "HUMAN_AUDIT_V4_meta.json"))["item_provenance"]
    gens = {}
    def full(lin, cond, sid):
        for fn in (f"{lin}@sft.jsonl", f"{lin}@sft__NGP.jsonl"):
            if (fn, "loaded") not in gens:
                p = EXP.parent / "E1" / "gen" / fn if (EXP.parent / "E1" / "gen" / fn).exists() else EXP / "E1" / "gen" / fn
                gens[(fn, "loaded")] = 1
                if p.exists():
                    for l in open(p):
                        r = json.loads(l)
                        if r["sampling_seed"] == 0 and r["sample"] == 0: gens[(r["lineage"], r["condition"], r["seed_id"])] = r["response"]
        return gens[(lin, cond, sid)]
    rows, texts = {}, []
    for iid, p in prov.items():
        if "tercile" not in p: continue
        t = todo[iid]["text"]; A = t.split("REPLY B:")[0].replace("REPLY A:", "").strip(); B = t.split("REPLY B:")[1].strip()
        rows[iid] = {"A_shown": A, "B_shown": B, "vern_full": full(p["lineage"], p["variety"], p["seed_id"]), "inf_full": full(p["lineage"], "INF", p["seed_id"])}
    ids = list(rows); keys = ("A_shown", "B_shown", "vern_full", "inf_full")
    flat = [rows[i][k] for i in ids for k in keys]; sc = p_formal(flat)
    out = {i: {k: round(sc[n * 4 + j], 5) for j, k in enumerate(keys)} for n, i in enumerate(ids)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("partA items", len(out))
else:
    import pandas as pd
    rows = []
    for fn in sorted({f for g in a.gen for f in glob.glob(g)}):
        for l in open(fn):
            r = json.loads(l); rows.append((r["model"], r["condition"], r["seed_id"], r["sampling_seed"], r["sample"], r["response"]))
    sc = p_formal([r[5] for r in rows])
    df = pd.DataFrame([r[:5] for r in rows], columns=["model", "condition", "seed_id", "sampling_seed", "sample"]); df["formality_ranker"] = sc
    df.to_parquet(a.out, index=False); print("scored", len(df))
