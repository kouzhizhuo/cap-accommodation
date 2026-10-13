#!/usr/bin/env python
"""E4 Day-0 regime check, stage 2: full detectors.base.score_text (recognisers + Multi-VALUE opportunities, spaCy features included)
on a fixed random sample (numpy default_rng(0)) of N prompts per dataset.  Prompts only (no novelty filter: prompt='' ).

  MV_COREF=0 python E4/prefdata_full_sample.py --name olmo-2-0425-1b-preference-mix --n 20000 --workers 48
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors  # noqa: F401
from prefdata_bulk import jobs_pref

MAXCH = 3000


def work(job):
    idx, pid, source, prompt = job
    from detectors.base import score_text
    prompt = prompt[:MAXCH]
    try:
        res = score_text(prompt, prompt="", varieties=("AAE", "SGE"), use_filter=False)
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "id": pid, "source": source, "error": repr(e)[:120]}
    row = {"idx": idx, "id": pid, "source": source, "n_words": res["n_words"]}
    for k, v in res["feat"].items():
        for c in ("real", "opp", "real_n", "opp_n"):
            row[f"{k}.{c}"] = v[c]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    n_rows = json.load(open(f"{here}/bulk/{a.name}.meta.json"))["n_rows"]
    sel = set(np.random.default_rng(0).choice(n_rows, size=min(a.n, n_rows), replace=False).tolist())

    def gen():
        for (i, pid, src, p, _, _) in jobs_pref(a.name):
            if i in sel:
                yield (i, pid, src, p)

    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(work, gen(), chunksize=16)):
            rows.append(r)
            if k % 2000 == 0:
                print(a.name, k, f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = f"{here}/bulk/{a.name}.fullsample.parquet"
    df.to_parquet(out, index=False)
    json.dump({"name": a.name, "n": len(df), "seconds": time.time() - t0, "workers": a.workers, "seed": 0, "maxchars": MAXCH},
              open(out.replace(".parquet", ".meta.json"), "w"))
    print("wrote", out, df.shape, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
