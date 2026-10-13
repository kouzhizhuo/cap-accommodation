#!/bin/bash
# After PAIRS60_DONE: put the 60k runs at the head of the lanes (T60k_s0, C60k_s0, T60k_s1, C60k_s1), then give the deferred runs back. Lanes were drained by drain_lanes.sh.
W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3; PP=$P/E5/pairs; OUTP=$W/data/P3_E5/pairs60
until [ -f $PP/r3/PAIRS60_DONE ]; do sleep 60; done
if [ ! -f $OUTP/mix_T60k/data/train-00000-of-00001.parquet ]; then echo "[$(date '+%m-%d %H:%M')] append60: no mix_T60k - giving deferred runs back" >> $PP/pipeline.log; fi
ln() { echo "$1|$OUTP/mix_$2|1.0|$3|0"; }
[ -f $OUTP/mix_T60k/data/train-00000-of-00001.parquet ] && { ln T60k_s0 T60k 0 >> $P/E5/lane_4.list; ln C60k_s0 C60k 0 >> $P/E5/lane_5.list; ln T60k_s1 T60k 1 >> $P/E5/lane_6.list; ln C60k_s1 C60k 1 >> $P/E5/lane_4.list; }
for g in 4 5 6; do [ -f $P/E5/deferred_$g.list ] && cat $P/E5/deferred_$g.list >> $P/E5/lane_$g.list; done
echo "[$(date '+%m-%d %H:%M')] append60: 60k runs appended, deferred runs returned" >> $PP/pipeline.log
