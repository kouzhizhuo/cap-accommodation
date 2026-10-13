#!/usr/bin/env python
"""Score generations (and prompts) with the detectors. Env p3mv, CPU. One row per response (or prompt) with per-feature columns
`<variety>.<feature>.{real,opp,real_n,opp_n}` so any F_train/F_heldout split can be evaluated later (lead, plan v1.1), plus
n_words, copy_rate (share of response words in shared >=4-grams), HD formality F.

  MV_COREF=0 python -m detectors.score --gen 'P0/gen/*.jsonl' --stimuli stimuli/v0 --out P0/pilot_scores.parquet --workers 32
"""
import argparse, glob, json, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detectors  # noqa: F401  (path shim)

_W = {}


def hd_formality(doc):
    """Heylighen & Dewaele F-score on spaCy UPOS: (noun+adj+prep+art - pron - verb - adv - intj + 100)/2, percentages of words."""
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    if not toks:
        return None
    n = len(toks)
    c = lambda f: 100.0 * sum(f(t) for t in toks) / n
    noun = c(lambda t: t.pos_ in {"NOUN", "PROPN"}); adj = c(lambda t: t.pos_ == "ADJ"); prep = c(lambda t: t.pos_ == "ADP")
    art = c(lambda t: t.lower_ in {"a", "an", "the"}); pron = c(lambda t: t.pos_ == "PRON")
    verb = c(lambda t: t.pos_ in {"VERB", "AUX"}); adv = c(lambda t: t.pos_ == "ADV"); intj = c(lambda t: t.pos_ == "INTJ")
    return (noun + adj + prep + art - pron - verb - adv - intj + 100.0) / 2.0


def score_one(job):
    from detectors.base import score_text
    from detectors.shared import nlp
    kind, meta, text, prompt = job
    res = score_text(text, prompt=prompt, use_filter=bool(prompt))
    row = dict(meta); row["kind"] = kind; row["n_words"] = res["n_words"]; row["copy_rate"] = res["copy_rate"]
    row["formality"] = hd_formality(nlp()(text))
    for k, v in res["feat"].items():
        for c in ("real", "opp", "real_n", "opp_n"):
            row[f"{k}.{c}"] = v[c]
    from detectors import split_items
    from detectors.base import in_ranges
    for g in split_items.GROUPS:     # per-item counts for the F_train / F_heldout split
        sp = split_items.spans_of(g, text)
        row[f"SPLIT.{g}.real"] = len(sp); row[f"SPLIT.{g}.real_n"] = sum(not in_ranges(x, res["_ranges"]) for x in sp)
        for nm in split_items.ITEMS[g]:       # per-item counts: lets the FP rule drop single items (ion, whip, mane, steady, ...)
            si = split_items.item_spans(g, nm, text); row[f"ITEM.{g}.{nm}.real"] = len(si); row[f"ITEM.{g}.{nm}.real_n"] = sum(not in_ranges(x, res["_ranges"]) for x in si)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True); ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = []
    for cond_file in sorted(Path(a.stimuli).glob("*.jsonl")):          # prompts: raw densities, no novelty filter
        for l in open(cond_file):
            r = json.loads(l)
            jobs.append(("prompt", {"model": "PROMPT", "condition": r["condition"], "seed_id": r["seed_id"]}, r["text"], ""))
    for pat in a.gen:
        for fn in sorted(glob.glob(pat)):
            for i, l in enumerate(open(fn)):
                if a.limit and i >= a.limit:
                    break
                r = json.loads(l)
                jobs.append(("response", {"model": r["model"], "lineage": r["lineage"], "stage": r["stage"], "scaffold": r["scaffold"],
                                          "condition": r["condition"], "seed_id": r["seed_id"], "context": r.get("context"),
                                          "task_type": r.get("task_type"), "sampling_seed": r["sampling_seed"], "sample": r["sample"],
                                          "finish": r["finish"], "n_tokens": r["n_tokens"]}, r["response"], r["prompt"]))
    print(f"{len(jobs)} texts to score", flush=True)
    with Pool(a.workers) as pool:
        rows = pool.map(score_one, jobs, chunksize=8)
    import pandas as pd
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print("wrote", a.out, df.shape, flush=True)


if __name__ == "__main__":
    main()
