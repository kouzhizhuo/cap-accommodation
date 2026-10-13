#!/bin/bash
# E5 post-processing on <host> (CPU, unattended): whenever a new E5 generation file appears, score all E5 gens (+ released L1 sft/dpo refs) and write E5/summary.json
# + exp/results/E5/summary.json (keys e5.<row>.<col>, RES_KEYS). Marker: $P/E5/post_w1.DONE after the first d0 + treatment + control have been exported.
W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; E=$P/exp; MV=$W/envs/p3mv/bin/python; L=$P/E5/post.log
export NLTK_DATA=$W/nltk_data MV_COREF=0 TMPDIR=$W/tmp; last=""
for i in $(seq 1 1440); do
  cur=$(ls -la $P/E5/gen/E5@*.jsonl 2>/dev/null | awk '{print $5, $9}' | md5sum)
  if [ -n "$(ls $P/E5/gen/E5@*.jsonl 2>/dev/null)" ] && [ "$cur" != "$last" ]; then last=$cur; sleep 120; cd $E
    echo "[$(date '+%m-%d %H:%M')] scoring" >> $L
    $MV -m detectors.score --gen "$P/E5/gen/E5@*.jsonl" "$P/E5/gen_ref/L1@*.jsonl" --stimuli stimuli/v1 --out $P/E5/scores.parquet --workers 48 >> $L 2>&1
    $MV analysis/e5_export.py --scores $P/E5/scores.parquet --out $E/results/E5/summary.json >> $L 2>&1
    echo "[$(date '+%m-%d %H:%M')] export done: $(ls $P/E5/gen/E5@*.jsonl | wc -l) gen files" >> $L
  fi; sleep 300
done
