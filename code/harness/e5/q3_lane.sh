#!/bin/bash
# Q3 LoRA-DPO lane on ONE <host> GPU (env oi_q3 = oi + transformers 4.51.3, open-instruct a996c34b; METHOD v4: Qwen3-8B non-thinking, LoRA r64/alpha128/dropout .05, dpo_norm beta 5, lr 1e-5 (arg), eff. batch 128). Usage: bash e5_lane.sh <gpu>   (gpu in {4,5}); reads $P/E5/lane_<gpu>.list, one run per line:
#   runid|dataset_dir|n_or_fraction|train_seed|max_steps(0=full epoch)
# Resumable (skips runs with a DONE marker), re-reads the list so runs can be APPENDED while it is alive, yields to other users (waits while its GPU
# shows >3 GB in use), and after each run generates E1-core replies on stimuli v1 (env gen) into $P/E5/gen/E5@<runid>*.jsonl for scoring on <host>.
# Recipe: SFT init allenai/OLMo-2-0425-1B-SFT, dpo_norm beta 5, lr 2.5e-6, 1 epoch, 128 pairs/step (8 x 16 accum on 1 GPU), seq 2048, no flash-attn.
GPU=$1; W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; LIST=$P/E5/qlane_$GPU.list; L=$P/E5/qlane_$GPU.log
SFT=${Q3_BASE:-$W/models/Qwen3-8B}; TPL=$SFT; LR=${Q3_LR:-1e-5}
mkdir -p $P/Q3/runs $P/Q3/gen $P/Q3/flags $W/tmp
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
gpu_free() { [ "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}')" -lt 3000 ]; }
modeldir() { local c; c=$(find $1 -name adapter_config.json -not -path "*/merged/*" 2>/dev/null | head -1); [ -n "$c" ] && dirname $c; }   # open-instruct saves into $OUT/<exp>__<seed>__<ts>/
has_weights() { [ -n "$(find $1 \( -name 'adapter_model*' -o -name 'pytorch_model*.bin' -o -name '*.safetensors' \) 2>/dev/null | head -1)" ]; }
killtree() { local c; for c in $(pgrep -P $1); do killtree $c; done; kill -9 $1 2>/dev/null; }
run_one() { local id=$1 data=$2 frac=$3 seed=$4 steps=$5; local OUT=$P/Q3/runs/$id
  until gpu_free; do log "GPU $GPU busy (other user/infra?) - yielding 10 min"; sleep 600; done
  log "start $id ($data $frac seed $seed steps $steps)"; rm -rf $OUT $P/Q3/cache_$id; mkdir -p $OUT
  ( cd $W/code/open-instruct_q3; source $W/envs/oi_q3/bin/activate
    export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$W/.hf_cache TMPDIR=$W/tmp TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD
    export CACHE_ROOT=/tmp/arr_cache_e5_$GPU; mkdir -p $CACHE_ROOT; export TRITON_CACHE_DIR=$CACHE_ROOT/triton TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_ext
    local ms=""; [ "$steps" != "0" ] && ms="--max_train_steps $steps"
    exec accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port $((29500 + GPU)) --use_deepspeed \
      --deepspeed_config_file $W/code/stage2_no_offloading_accelerate.conf --deepspeed_multinode_launcher standard \
      open_instruct/dpo_tune_cache.py --exp_name q3_$id --learning_rate $LR --seed $seed \
      --model_name_or_path $SFT --tokenizer_name_or_path $SFT --max_seq_length 2048 --per_device_train_batch_size 1 --gradient_accumulation_steps 128 \
      --lr_scheduler_type linear --warmup_ratio 0.1 --weight_decay 0.0 --num_train_epochs 1 $ms \
      --output_dir $OUT --logging_steps 10 --gradient_checkpointing --use_flash_attn False --dataset_mixer_list $data $frac \
      --use_slow_tokenizer False --chat_template_name qwen3_native --use_lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 --dpo_loss_type dpo_norm --dpo_beta 5 \
      --push_to_hub False --try_launch_beaker_eval_jobs False --dataset_cache_mode local --dataset_local_cache_dir $P/Q3/cache_$id ) > $P/Q3/runs/$id.train.log 2>&1 &
  local tp=$!
  # teardown watcher (infra <agent>): after saving, open-instruct hangs holding ~77 GB. When "Remaining files" is logged and the model file exists,
  # wait 30 s, then kill the process TREE by PID (never by name pattern).
  while kill -0 $tp 2>/dev/null; do
    if grep -q "Remaining files" $P/Q3/runs/$id.train.log 2>/dev/null && has_weights $OUT; then
      sleep 30; log "teardown watcher: killing process tree of $id (pid $tp)"; killtree $tp; break; fi
    sleep 20
  done
  wait $tp 2>/dev/null
  local rc=0   # exit code is meaningless after the watcher kill; success = model files present
  if ! has_weights $OUT; then log "FAILED $id (rc $rc)"; touch $P/Q3/flags/$id.FAILED; return 1; fi
  rm -rf $P/Q3/cache_$id
  MD=$(modeldir $OUT); MG=$OUT/merged
  if [ ! -f $MG/config.json ]; then CUDA_VISIBLE_DEVICES=$GPU $W/envs/oi_q3/bin/python $E/harness/e5/merge_lora.py --base $SFT --adapter $MD --out $MG > $P/Q3/runs/$id.merge.log 2>&1 || { log "MERGE FAILED $id"; touch $P/Q3/flags/$id.FAILED; return 1; }; fi
  if [ "$id" != smoke ]; then   # replies on the E1-core conditions (v1 stimuli), scored later on <host> (env p3mv)
    CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 $W/envs/gen/bin/python $E/harness/gen.py --model-path $MG --chat-template-kwargs '{"enable_thinking": false}' --template-from $TPL \
       --lineage Q3 --stage $id --stimuli $E/stimuli/v1 --conds SAE INF AAE SGE BRE AUS --scaffold chat --seeds 0 1 2 --n 2 --out $P/Q3/gen/Q3@$id > $P/Q3/runs/$id.gen.log 2>&1 \
       && log "gen done $id" || log "gen FAILED $id"
  fi
  [ "$id" != smoke ] && [ -x $E/harness/e5/post_extra.sh ] && KIND=q3 bash $E/harness/e5/post_extra.sh $id $MG $GPU
  touch $P/Q3/flags/$id.DONE; log "done $id"; return 0; }
log "lane $GPU start"
idle=0
while [ $idle -lt 360 ]; do   # up to 12 h of waiting for appended runs
  next=""; while IFS='|' read -r id data frac seed steps; do
    [ -z "$id" ] && continue; [ "$id" = END ] && continue; [ -f $P/Q3/flags/$id.DONE ] || [ -f $P/Q3/flags/$id.FAILED ] && continue
    [ "$id" != smoke ] && [ ! -f $P/Q3/flags/smoke.DONE ] && { [ -f $P/Q3/flags/smoke.FAILED ] && { log "smoke failed - lane $GPU stops"; exit 1; }; continue; }   # every lane waits for the smoke test
    next="$id|$data|$frac|$seed|$steps"; break
  done < $LIST
  if [ -n "$next" ]; then idle=0; IFS='|' read -r id data frac seed steps <<< "$next"; run_one $id $data $frac $seed $steps; else idle=$((idle+1)); sleep 120; fi
done
log "lane $GPU idle-timeout end"
