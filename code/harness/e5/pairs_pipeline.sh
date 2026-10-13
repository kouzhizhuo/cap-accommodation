#!/bin/bash
# E5 pair-file pipeline on <host> (GPU 6 for the two generation steps, CPU for filtering/transforms), unattended; marker $PP/PAIRS_DONE.
#  1 Qwen2.5-7B-Instruct writes ~44k candidate seeds (disjoint random Dolly topics, seed 7)   2 filter (opportunity checker, dedupe, disjoint from stimuli v1)
#  3 OLMo-2-0425-1B-SFT writes two replies per seed (T 0.7, max 400)   4 pairs_build.py (F_train-only transforms; T/C/A datasets)   5 append runs to the lane lists
W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; PP=$P/E5/pairs; SUB=$W/data/P3_E5/subset120k; OUTP=$W/data/P3_E5/pairs
GEN=$W/envs/gen/bin/python; MV=$W/envs/p3mv/bin/python; mkdir -p $PP/stim $PP/replies $OUTP
export NLTK_DATA=$W/nltk_data MV_COREF=0 TMPDIR=$W/tmp HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $PP/pipeline.log; }
log "start"
[ -s $PP/cands.jsonl ] || CUDA_VISIBLE_DEVICES=6 $GEN $E/stimuli/make_seeds_v1.py --out $PP/cands.jsonl --per-cell 3700 --seed 7 --model $W/models/Qwen2.5-7B-Instruct \
    --dolly $W/data/databricks-dolly-15k/databricks-dolly-15k.jsonl --gpu-util 0.80 > $PP/step1.log 2>&1
log "step1 candidates: $(wc -l < $PP/cands.jsonl)"
cd $E/stimuli
[ -s $PP/pool.jsonl ] || $MV seeds_select_v1.py --cands $PP/cands.jsonl --out $PP/pool.jsonl --no-ukcheck --per-cell 1000000 --exclude $E/stimuli/seeds_v1.jsonl --id-prefix e5s > $PP/step2.log 2>&1
log "step2 pool: $(wc -l < $PP/pool.jsonl) ($(tail -3 $PP/step2.log | head -1 | cut -c1-200))"
$MV - <<PY
import json
with open("$PP/stim/SAE.jsonl", "w") as f:
    for l in open("$PP/pool.jsonl"):
        r = json.loads(l); f.write(json.dumps({"seed_id": r["seed_id"], "condition": "SAE", "text": r["text"], "task_type": r["task_type"], "context": r["context"]}) + "\n")
PY
[ -s $PP/replies/<email> ] || CUDA_VISIBLE_DEVICES=6 $GEN $E/harness/gen.py --model-path $W/models/P3/E5/sft --lineage E5 --stage sftpairs --stimuli $PP/stim --conds SAE \
    --scaffold chat --seeds 0 1 --n 1 --gpu-util 0.85 --out $PP/replies/E5@sftpairs > $PP/step3.log 2>&1
log "step3 replies: $(wc -l < $PP/replies/<email>)"
cd $E/harness/e5; $MV pairs_build.py --seeds $PP/pool.jsonl --replies $PP/replies/<email> --base $SUB/data/train-00000-of-00001.parquet --out-root $OUTP --workers 48 > $PP/step4.log 2>&1
log "step4 build: $(tail -1 $PP/step4.log | cut -c1-400)"
[ -f $OUTP/T20k/data/train-00000-of-00001.parquet ] || { log "NO 20k pairs - stop (see step4.log)"; exit 1; }
d() { echo "$SUB 1.0 $OUTP/$1"; }        # mixer list = 120k base subset + the added pairs
printf "T20k_s0|$(d T20k)|1.0|0|0\nT5k_s0|$(d T5k)|1.0|0|0\nT20k_s1|$(d T20k)|1.0|1|0\nT5k_s1|$(d T5k)|1.0|1|0\nA5k_s0|$(d A5k)|1.0|0|0\n" >> $P/E5/lane_6.list
printf "C20k_s0|$(d C20k)|1.0|0|0\nC5k_s0|$(d C5k)|1.0|0|0\nT20k_s2|$(d T20k)|1.0|2|0\nT5k_s2|$(d T5k)|1.0|2|0\n" >> $P/E5/lane_4.list
printf "d0_s2|$SUB|1.0|2|0\nC20k_s1|$(d C20k)|1.0|1|0\nC5k_s1|$(d C5k)|1.0|1|0\nT1k_s0|$(d T1k)|1.0|0|0\nA1k_s0|$(d A1k)|1.0|0|0\n" >> $P/E5/lane_5.list
echo done > $PP/PAIRS_DONE; log "runs appended to lane lists; PAIRS_DONE"
