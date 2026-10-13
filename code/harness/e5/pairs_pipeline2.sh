#!/bin/bash
# E5 pair pipeline, ROUND 2 (round 1 pool was 18,110 < the 20k pairs needed): new candidates (seed 8), filtered against stimuli v1 AND the round-1 pool,
# replies for the new seeds, then pairs_build.py on the MERGED pool/replies and the run-list appends (same as round 1 step 5). Waits for round 1 to finish.
W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; PP=$P/E5/pairs; SUB=$W/data/P3_E5/subset120k; OUTP=$W/data/P3_E5/pairs
GEN=$W/envs/gen/bin/python; MV=$W/envs/p3mv/bin/python; mkdir -p $PP/stim2 $PP/replies2
export NLTK_DATA=$W/nltk_data MV_COREF=0 TMPDIR=$W/tmp HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date '+%m-%d %H:%M')] R2 $*" >> $PP/pipeline.log; }
until grep -q "step4 build" $PP/pipeline.log 2>/dev/null; do sleep 30; done
[ -f $PP/PAIRS_DONE ] && { log "round 1 already produced 20k pairs - nothing to do"; exit 0; }
log "start"
[ -s $PP/cands2.jsonl ] || CUDA_VISIBLE_DEVICES=6 $GEN $E/stimuli/make_seeds_v1.py --out $PP/cands2.jsonl --per-cell 3700 --seed 8 --model $W/models/Qwen2.5-7B-Instruct \
    --dolly $W/data/databricks-dolly-15k/databricks-dolly-15k.jsonl --gpu-util 0.80 > $PP/step1b.log 2>&1
cat $E/stimuli/seeds_v1.jsonl $PP/pool.jsonl > $PP/exclude2.jsonl
cd $E/stimuli
$MV seeds_select_v1.py --cands $PP/cands2.jsonl --out $PP/pool2.jsonl --no-ukcheck --per-cell 1000000 --exclude $PP/exclude2.jsonl --id-prefix e5t > $PP/step2b.log 2>&1
log "pool2: $(wc -l < $PP/pool2.jsonl)"
$MV - <<PY
import json
with open("$PP/stim2/SAE.jsonl", "w") as f:
    for l in open("$PP/pool2.jsonl"):
        r = json.loads(l); f.write(json.dumps({"seed_id": r["seed_id"], "condition": "SAE", "text": r["text"], "task_type": r["task_type"], "context": r["context"]}) + "\n")
PY
[ -s $PP/replies2/<email> ] || CUDA_VISIBLE_DEVICES=6 $GEN $E/harness/gen.py --model-path $W/models/P3/E5/sft --lineage E5 --stage sftpairs --stimuli $PP/stim2 --conds SAE \
    --scaffold chat --seeds 0 1 --n 1 --gpu-util 0.85 --out $PP/replies2/E5@sftpairs > $PP/step3b.log 2>&1
cat $PP/pool.jsonl $PP/pool2.jsonl > $PP/pool_all.jsonl; cat $PP/replies/<email> $PP/replies2/<email> > $PP/replies_all.jsonl
cd $E/harness/e5; $MV pairs_build.py --seeds $PP/pool_all.jsonl --replies $PP/replies_all.jsonl --base $SUB/data/train-00000-of-00001.parquet --out-root $OUTP --workers 48 > $PP/step4b.log 2>&1
log "merged build: $(tail -1 $PP/step4b.log | cut -c1-400)"
$MV - <<PY > $PP/check20k.txt 2>&1
import json, sys
st = json.load(open("$OUTP/PAIRS_STATS.json")); print(st["sizes"].get("T20k"))
PY
[ "$(tail -1 $PP/check20k.txt)" = "20000" ] || { log "T20k has $(tail -1 $PP/check20k.txt) pairs, need 20000 - STOP (see step4b.log)"; exit 1; }
# one mixed dataset dir per run = 120k base subset + the added pairs, shuffled (single-path data field for the lane script)
$MV - <<PY
import pyarrow as pa, pyarrow.parquet as pq, numpy as np
from pathlib import Path
base = pq.read_table("$SUB/data/train-00000-of-00001.parquet")
for name in ["T20k", "T5k", "T1k", "C20k", "C5k", "A5k", "A1k"]:
    add = pq.read_table("$OUTP/%s/data/train-00000-of-00001.parquet" % name).cast(base.schema)
    t = pa.concat_tables([base, add]); perm = np.random.default_rng(0).permutation(t.num_rows); t = t.take(pa.array(perm))
    d = Path("$OUTP/mix_%s/data" % name); d.mkdir(parents=True, exist_ok=True); pq.write_table(t.replace_schema_metadata(None), d / "train-00000-of-00001.parquet")
    print(name, t.num_rows)
PY
d() { echo "$OUTP/mix_$1"; }
printf "T20k_s0|$(d T20k)|1.0|0|0\nT5k_s0|$(d T5k)|1.0|0|0\nT20k_s1|$(d T20k)|1.0|1|0\nT5k_s1|$(d T5k)|1.0|1|0\nA5k_s0|$(d A5k)|1.0|0|0\n" >> $P/E5/lane_6.list
printf "C20k_s0|$(d C20k)|1.0|0|0\nC5k_s0|$(d C5k)|1.0|0|0\nT20k_s2|$(d T20k)|1.0|2|0\nT5k_s2|$(d T5k)|1.0|2|0\n" >> $P/E5/lane_4.list
printf "C20k_s1|$(d C20k)|1.0|1|0\nC5k_s1|$(d C5k)|1.0|1|0\nd0_s2|$SUB|1.0|2|0\nT1k_s0|$(d T1k)|1.0|0|0\nA1k_s0|$(d A1k)|1.0|0|0\n" >> $P/E5/lane_5.list
echo done > $PP/PAIRS_DONE; log "mixed datasets built; runs appended to lane lists; PAIRS_DONE"
