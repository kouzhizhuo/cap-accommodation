#!/bin/bash
# B7 self-rewrite baseline on ONE GPU (lock-wrapped by the caller): Qwen3-8B INSTRUCT, 2 passes, F300.  Usage: bash b7_run.sh <gpu>
GPU=$1; W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; B=$P/Q3/b7; F=$P/E5/evalsets/F300; MD=$W/models/Qwen3-8B; GEN=$W/envs/gen/bin/python; MV=$W/envs/p3mv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0; mkdir -p $B
C="SAE AAE SGE NGP APP"
$MV $E/harness/e5/b7_make.py stim 1 --draft $P/Q3/gen_f300/<email> --f300 $F --out $B/stim1 > $B/stim1.log 2>&1
[ -f $B/p1.meta.json ] || $GEN $E/harness/gen.py --model-path $MD --template-from $MD --lineage Q3 --stage B7p1 --stimuli $B/stim1 --conds $C --scaffold chat --chat-template-kwargs '{"enable_thinking": false}' --seeds 0 --n 1 --gpu-util 0.88 --out $B/p1 > $B/p1.log 2>&1
$MV $E/harness/e5/b7_make.py stim 2 --draft $B/p1.jsonl --f300 $F --out $B/stim2 > $B/stim2.log 2>&1
[ -f $B/p2.meta.json ] || $GEN $E/harness/gen.py --model-path $MD --template-from $MD --lineage Q3 --stage B7p2 --stimuli $B/stim2 --conds $C --scaffold chat --chat-template-kwargs '{"enable_thinking": false}' --seeds 0 --n 1 --gpu-util 0.88 --out $B/p2 > $B/p2.log 2>&1
$MV $E/harness/e5/b7_make.py collect --p2 $B/p2.jsonl --out $P/Q3/gen_f300/<email> > $B/collect.log 2>&1 && cp $B/p2.meta.json $P/Q3/gen_f300/<email> && echo "B7 done $(date)" >> $B/done.log
