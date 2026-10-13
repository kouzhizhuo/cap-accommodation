#!/bin/bash
# Fixed 120k-pair subset (seeded) of the 1B preference mix, saved as a parquet dataset dir; run once (idempotent).
W=<REMOTE_ROOT>/arr_oct; SRC=$W/data/P3_E5/olmo-2-0425-1b-preference-mix; DST=$W/data/P3_E5/subset120k
[ -f $DST/data/train-00000-of-00001.parquet ] && exit 0; mkdir -p $DST/data
$W/envs/gen/bin/python - <<PY
from datasets import load_dataset
ds = load_dataset("parquet", data_files="$SRC/data/train-*.parquet", split="train")
sub = ds.shuffle(seed=0).select(range(120000))
import pyarrow.parquet as pq   # plain arrow schema: the newer datasets' HF feature metadata is unreadable by the older datasets in env oi
pq.write_table(sub.data.table.replace_schema_metadata(None), "$DST/data/train-00000-of-00001.parquet")
import json; json.dump({"source": "allenai/olmo-2-0425-1b-preference-mix", "n": len(sub), "rows_in_source": len(ds), "shuffle_seed": 0}, open("$DST/SUBSET.json", "w"))
print("subset", len(sub), "of", len(ds))
PY
