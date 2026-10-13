#!/usr/bin/env python
"""E4 Day-0 regime check, stage 1 (bulk, CPU, regex recognisers only).

For every row of a preference dataset: prompt / chosen / rejected -> per-feature hit counts of the RETAINED morphosyntactic features
(validation gate) and the lexical/rate features.  Responses are counted raw and 'novel' (spans inside >=4-gram copies of the prompt or
inside quotation marks are dropped, detectors.base.copied_ranges).  No Multi-VALUE, no spaCy.

  cd <HOME>/arr_oct/runs/P3/exp && source ../../../envs/p3mv/bin/activate
  MV_COREF=0 python E4/prefdata_bulk.py --name olmo-2-0425-1b-preference-mix --workers 48
"""
import argparse, glob, json, os, re, sys, time
from multiprocessing import Pool
import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detectors  # noqa: F401
from detectors.base import FEATURES, copied_ranges, in_ranges, _tokens
import detectors.aae, detectors.sge, detectors.inf  # noqa: F401

RETAINED_MORPH = {"AAE": ["ain_t", "completive_done", "finna_gon", "negative_concord", "yall"],
                  "SGE": ["can_cannot", "finish_completive"]}
LEXRATE = {"AAE": ["lexicon", "habitual_be", "g_drop"], "SGE": ["particles", "lexicon"]}
FEATS = {}  # "AAE.ain_t" -> recogniser
for v in ("AAE", "SGE"):
    byid = {f.id: f for f in FEATURES[v]}
    for fid in RETAINED_MORPH[v] + LEXRATE[v]:
        assert not byid[fid].needs_spacy, fid
        FEATS[f"{v}.{fid}"] = byid[fid].recogniser
import detectors.bre  # noqa: F401  (BRE/AUS spelling+lexis, regex only)
# BRE/AUS (coordinator update): reported separately, NOT part of the AAE/SGE dialect-bearing definition. AUS.spelling == BRE.spelling (same recogniser).
_byid = {f"{f.variety}.{f.id}": f for v in ("BRE", "AUS") for f in FEATURES[v]}
for _n in ("BRE.spelling", "BRE.lexis", "BRE.have_got", "AUS.aus_lexis"):
    assert not _byid[_n].needs_spacy
    FEATS[_n] = _byid[_n].recogniser
FNAMES = list(FEATS)
MAXCH = 30000
DATA = "<HOME>/arr_oct/data/P3_prefdata"

_CODE = re.compile(r"```")


def msgs_to_pr(msgs):
    """(prompt_text = all user turns joined, response = final assistant message)."""
    msgs = list(msgs)
    resp = ""
    if msgs and msgs[-1]["role"] == "assistant":
        resp = msgs[-1]["content"]; msgs = msgs[:-1]
    prompt = "\n".join(m["content"] for m in msgs if m["role"] == "user")
    return prompt, resp


def count_text(text, ranges=None):
    text = text[:MAXCH]
    out = []
    for n in FNAMES:
        sp = FEATS[n](text)
        out.append(len(sp) if ranges is None else sum(1 for s in sp if not in_ranges(s, ranges)))
    return out


def process(job):
    idx, pid, source, prompt, chosen, rejected = job
    prompt = prompt[:MAXCH]
    row = {"idx": idx, "id": pid, "source": source, "p_words": len(_tokens(prompt)),
           "p_code": int(bool(_CODE.search(prompt))),
           "p_nonascii": (sum(ord(c) > 127 for c in prompt) / max(1, len(prompt)))}
    for n, c in zip(FNAMES, count_text(prompt)):
        row[f"p.{n}"] = c
    for tag, resp in (("c", chosen), ("r", rejected)):
        if resp is None:
            continue
        resp = resp[:MAXCH]
        rng, _ = copied_ranges(resp, prompt)
        row[f"{tag}_words"] = len(_tokens(resp))
        raw = count_text(resp)
        nov = count_text(resp, rng)
        for n, a, b in zip(FNAMES, raw, nov):
            row[f"{tag}raw.{n}"] = a
            row[f"{tag}.{n}"] = b
    return row


def jobs_pref(name):
    i = 0
    for f in sorted(glob.glob(f"{DATA}/{name}/data/*.parquet")):
        t = pq.ParquetFile(f)
        for rg in range(t.num_row_groups):
            for r in t.read_row_group(rg, columns=["id", "source", "chosen", "rejected"]).to_pylist():
                p, c = msgs_to_pr(r["chosen"]); _, rj = msgs_to_pr(r["rejected"])
                yield (i, r["id"], r["source"], p, c, rj)
                i += 1


def jobs_sft(name):
    i = 0
    for f in sorted(glob.glob(f"{DATA}/{name}/data/*.parquet")):
        t = pq.ParquetFile(f)
        cols = [c for c in ("id", "source", "messages") if c in t.schema_arrow.names]
        for rg in range(t.num_row_groups):
            for r in t.read_row_group(rg, columns=cols).to_pylist():
                p, _ = msgs_to_pr(r["messages"])
                yield (i, r.get("id"), r.get("source"), p, None, None)
                i += 1


def jobs_dolly():
    for i, l in enumerate(open("<HOME>/arr_oct/data/databricks-dolly-15k/databricks-dolly-15k.jsonl")):
        r = json.loads(l)
        yield (i, str(i), "dolly:" + r.get("category", ""), r["instruction"], None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.name == "dolly-15k":
        gen = jobs_dolly()
    elif "sft" in a.name:
        gen = jobs_sft(a.name)
    else:
        gen = jobs_pref(a.name)
    if a.limit:
        import itertools
        gen = itertools.islice(gen, a.limit)
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = []
        for k, r in enumerate(pool.imap(process, gen, chunksize=64)):
            rows.append(r)
            if k % 50000 == 0:
                print(a.name, k, f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bulk", f"{a.name}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    json.dump({"name": a.name, "n_rows": len(df), "seconds": time.time() - t0, "workers": a.workers, "features": FNAMES},
              open(out.replace(".parquet", ".meta.json"), "w"))
    print("wrote", out, df.shape, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
