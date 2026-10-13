#!/bin/bash
# P3 eval queue on ONE <host> GPU (lock-wrapped by the caller).  Usage: queue_p3e.sh <gpu> <taskfile>   Resumable (skips tasks whose .meta.json exists); re-reads the taskfile so tasks can be appended.
# task line:  <outdir>/<name>|<modelkey>|<stimuli_dir>|<conds>|<seeds>|<n>|<extra gen args>      modelkey = E5 run id (E5/runs/<id>) | dpo_released | sft
GPU=$1; TASKS=$2; W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; TPL=$W/models/P3/E5/dpo_released; GEN=$W/envs/gen/bin/python; L=$P/E5/queue_e_$GPU.log
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
mdir() { case $1 in q3_instruct) echo $W/models/Qwen3-8B;; q3_*) echo $P/Q3/runs/${1#q3_}/merged;; dpo_released) echo $W/models/P3/E5/dpo_released;; sft) echo $W/models/P3/E5/sft;; *) dirname $(find $P/E5/runs/$1 -name config.json | head -1);; esac; }
B3="Reply in the same variety and register the user writes in, matching how much they use it; never exaggerate it."
done_n=0; idle=0
while [ $idle -lt 90 ]; do
  next=""; while IFS='|' read -r out key stim conds seeds n extra; do
    [ -z "$out" ] && continue; [ -f $out.meta.json ] || [ -f $out.FAILED ] && continue; next="$out|$key|$stim|$conds|$seeds|$n|$extra"; break; done < $TASKS
  if [ -z "$next" ]; then idle=$((idle+1)); sleep 60; continue; fi
  idle=0; IFS='|' read -r out key stim conds seeds n extra <<< "$next"; mkdir -p $(dirname $out)
  md=$(mdir $key); [ -z "$md" ] || [ ! -d "$md" ] && { log "no model for $key"; touch $out.FAILED; continue; }
  tpl=$TPL; ck=""; lin=E5; case $key in q3_*) tpl=$md; ck="--chat-template-kwargs '{\"enable_thinking\": false}'"; lin=Q3;; esac
  extra=${extra//@B3@/--system \"$B3\"}; extra=${extra//@DEMO@/--prefix-msgs $P/E5/fewshot_demo.json}
  log "start $out ($key)"
  eval $GEN $E/harness/gen.py --model-path $md --template-from $tpl --lineage $lin $ck --stage $(basename $out) --stimuli $stim --conds $conds --scaffold chat --seeds $seeds --n $n --gpu-util 0.85 --out $out $extra > $out.log 2>&1 && log "done $out" || { log "FAILED $out"; touch $out.FAILED; }
done
log "queue end"
