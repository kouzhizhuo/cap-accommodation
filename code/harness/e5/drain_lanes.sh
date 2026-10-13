#!/bin/bash
# keeps, per lane list, the DONE entries and the run currently training; moves every not-yet-started entry to E5/deferred_<gpu>.list (restored by append60.sh)
W=<REMOTE_ROOT>/arr_oct; P=$W/runs/P3
for g in 4 5 6; do L=$P/E5/lane_$g.list; cur=$(grep "^\[.*\] start " $P/E5/lane_$g.log | tail -n 1 | sed 's/.*start \([^ ]*\) .*/\1/')
  : > $P/E5/deferred_$g.list; : > $L.new
  while IFS= read -r line; do id=${line%%|*}; [ -z "$id" ] && continue
    if [ -f $P/E5/flags/$id.DONE ] || [ "$id" = "$cur" ] || [ "$id" = smoke ]; then echo "$line" >> $L.new; else echo "$line" >> $P/E5/deferred_$g.list; fi
  done < $L; mv $L.new $L; echo "lane $g keeps: $(cut -d'|' -f1 $L | tr '\n' ' ') | deferred: $(cut -d'|' -f1 $P/E5/deferred_$g.list | tr '\n' ' ')"
done
