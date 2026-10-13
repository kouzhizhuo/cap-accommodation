#!/bin/bash
# E5 60k dose (user decision 2026-10-03; F_train uptake shown at 20k): more candidate seeds (rounds 3..), replies, pairs_build into a SEPARATE root (pairs60/; the 1k/5k/20k files stay untouched),
# mixed datasets mix_T60k / mix_C60k (120k base + added pairs), then append runs. Takes the first <host> GPU in {4,5,6} that becomes free (lanes are drained first by drain_lanes step).
W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; PP=$P/E5/pairs; SUB=$W/data/P3_E5/subset120k; OUTP=$W/data/P3_E5/pairs60
GEN=$W/envs/gen/bin/python; MV=$W/envs/p3mv/bin/python; mkdir -p $OUTP $PP/r3
export NLTK_DATA=$W/nltk_data MV_COREF=0 TMPDIR=$W/tmp HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date '+%m-%d %H:%M')] R3 $*" >> $PP/pipeline.log; }
pickgpu() { while true; do for g in 4 5 6; do m=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1==g{print $2}'); [ "$m" -lt 3000 ] && { echo $g; return; }; done; sleep 10; done; }
cat $PP/pool.jsonl $PP/pool2.jsonl > $PP/r3/pool_all.jsonl; cat $PP/replies/<email> $PP/replies2/<email> > $PP/r3/replies_all.jsonl
log "start; base pool $(wc -l < $PP/r3/pool_all.jsonl)"
for rd in 3 4 5 6 7 8 9; do
  have=$(wc -l < $PP/r3/pool_all.jsonl); [ $have -ge 98000 ] && break
  D=$PP/r3/round$rd; mkdir -p $D/stim $D/replies; G=$(pickgpu); log "round $rd on GPU $G (pool $have)"
  [ -s $D/cands.jsonl ] || CUDA_VISIBLE_DEVICES=$G $GEN $E/stimuli/make_seeds_v1.py --out $D/cands.jsonl --per-cell 3700 --seed $((6+rd)) --model $W/models/Qwen2.5-7B-Instruct \
      --dolly $W/data/databricks-dolly-15k/databricks-dolly-15k.jsonl --gpu-util 0.80 > $D/step1.log 2>&1
  cat $E/stimuli/seeds_v1.jsonl $PP/r3/pool_all.jsonl > $D/exclude.jsonl; cd $E/stimuli
  $MV seeds_select_v1.py --cands $D/cands.jsonl --out $D/pool.jsonl --no-ukcheck --per-cell 1000000 --exclude $D/exclude.jsonl --id-prefix e5r$rd > $D/step2.log 2>&1
  $MV - <<PY
import json
with open("$D/stim/SAE.jsonl", "w") as f:
    for l in open("$D/pool.jsonl"):
        r = json.loads(l); f.write(json.dumps({"seed_id": r["seed_id"], "condition": "SAE", "text": r["text"], "task_type": r["task_type"], "context": r["context"]}) + "\n")
PY
  G=$(pickgpu)
  [ -s $D/replies/<email> ] || CUDA_VISIBLE_DEVICES=$G $GEN $E/harness/gen.py --model-path $W/models/P3/E5/sft --lineage E5 --stage sftpairs --stimuli $D/stim --conds SAE \
      --scaffold chat --seeds 0 1 --n 1 --gpu-util 0.85 --out $D/replies/E5@sftpairs > $D/step3.log 2>&1
  cat $D/pool.jsonl >> $PP/r3/pool_all.jsonl; cat $D/replies/<email> >> $PP/r3/replies_all.jsonl
  log "round $rd done: +$(wc -l < $D/pool.jsonl) seeds, replies $(wc -l < $D/replies/<email>)"
done
cd $E/harness/e5; $MV pairs_build.py --seeds $PP/r3/pool_all.jsonl --replies $PP/r3/replies_all.jsonl --base $SUB/data/train-00000-of-00001.parquet --out-root $OUTP --workers 64 --doses 20000 60000 > $PP/r3/build.log 2>&1
log "build: $(tail -1 $PP/r3/build.log | cut -c1-300)"
[ -f $OUTP/T60k/data/train-00000-of-00001.parquet ] || { log "NO 60k pairs - STOP (see r3/build.log); pairs_valid may be < 60k"; exit 1; }
$MV - <<PY
import pyarrow as pa, pyarrow.parquet as pq, numpy as np
from pathlib import Path
base = pq.read_table("$SUB/data/train-00000-of-00001.parquet")
for name in ["T60k", "C60k"]:
    add = pq.read_table("$OUTP/%s/data/train-00000-of-00001.parquet" % name).cast(base.schema)
    t = pa.concat_tables([base, add]); perm = np.random.default_rng(0).permutation(t.num_rows); t = t.take(pa.array(perm))
    d = Path("$OUTP/mix_%s/data" % name); d.mkdir(parents=True, exist_ok=True); pq.write_table(t.replace_schema_metadata(None), d / "train-00000-of-00001.parquet"); print(name, t.num_rows)
PY
echo done > $PP/r3/PAIRS60_DONE; log "mixed datasets built (PAIRS60_DONE)"
