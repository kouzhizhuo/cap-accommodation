#!/bin/bash
# METHOD post-run evaluations for one trained checkpoint (called by e5_lane.sh [1B] / q3_lane.sh [Qwen3-8B, KIND=q3] after training).  Usage: post_extra.sh <runid> <modeldir> <gpu>
#   F300 fresh test (SAE AAE SGE NGP APP; seeds 0 1 2 x n 2, E1 protocol, 400 tokens), V300 validation (SAE AAE SGE), D density intervention, O overrides, IFEval (greedy 1,280), GSM8K (0-shot CoT, greedy 768)
# outputs: $ROOT/gen_<set>/<TAG>@<id>.jsonl  (ROOT = E5 for the 1B runs, Q3 for Qwen3-8B)
id=$1; MD=$2; GPU=$3; KIND=${KIND:-e5}; W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; GEN=$W/envs/gen/bin/python; ES=$P/E5/evalsets
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ "$KIND" = q3 ]; then ROOT=$P/Q3; TAG=Q3; LIN=Q3; TPL=$MD; CK='--chat-template-kwargs {"enable_thinking":false}'; L=$P/E5/qlane_$GPU.log; else ROOT=$P/E5; TAG=E5; LIN=E5; TPL=$W/models/P3/E5/dpo_released; CK=""; L=$P/E5/lane_$GPU.log; fi
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
g() { local out=$1 stim=$2 conds=$3; shift 3; mkdir -p $(dirname $out); [ -f $out.meta.json ] && return 0
  $GEN $E/harness/gen.py --model-path $MD --template-from $TPL --lineage $LIN --stage $id --stimuli $stim --conds $conds --scaffold chat --out $out --gpu-util 0.85 $CK "$@" > $out.log 2>&1 && log "gen ok $out" || log "gen FAILED $out"; }
[ -d $ES/F300 ] && g $ROOT/gen_f300/$TAG@$id $ES/F300 "SAE AAE SGE NGP APP" --seeds 0 1 2 --n 2
V=$W/data/P3_E5/pairs_v3/val; [ -d $V ] && g $ROOT/gen_val/${TAG}v@$id $V "SAE AAE SGE" --seeds 0 1 2 --n 2
[ -d $ES/D ] && g $ROOT/gen_d/${TAG}d@$id $ES/D "AAE_lo AAE_hi SGE_lo SGE_hi" --seeds 0 1 2 --n 1
[ -d $ES/O ] && g $ROOT/gen_o/${TAG}o@$id $ES/O "O_STD O_FORMAL O_NOMIMIC O_AAE O_SGE" --seeds 0 --n 2
[ -d $E/stimuli/ifeval ] && g $ROOT/ifeval/$TAG@ifeval-$id $E/stimuli/ifeval IFEVAL --seeds 0 --n 1 --temperature 0 --max-tokens 1280
[ -d $E/stimuli/gsm8k ] && g $ROOT/gsm8k/$TAG@gsm-$id $E/stimuli/gsm8k GSM8K --seeds 0 --n 1 --temperature 0 --max-tokens 768
exit 0
