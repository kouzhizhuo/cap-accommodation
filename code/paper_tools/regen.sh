#!/bin/bash
# Regenerate every \res key and the generated tables from exp/results (lead <agent>, 2026-10-04). Run from paper/.
set -e
( cd .. && python3 exp/analysis/lead_derived.py > /dev/null )
python3 tools/json2tex.py \
  ../exp/results/design/design_measured.json:design ../exp/results/E1/summary.json:e1 ../exp/results/derived/newlineages_summary.json:e1 ../exp/results/derived/newlineages_extras.json:e1 ../exp/results/E4/mbv_new.json:e4 ../exp/results/breadth/summary.json:e1b ../exp/results/derived/breadth_extras.json:e1b ../exp/results/newgen/summary.json:e1n ../exp/results/derived/newgen_extras.json:e1n ../exp/results/derived/newgen_q38_summary.json:e1n ../exp/results/derived/newgen_q38_extras.json:e1n ../exp/results/derived/openpangu_summary.json:e1o ../exp/results/derived/openpangu_extras.json:e1o ../exp/results/E1b/summary.json:e1r ../exp/results/E1/deployed/summary_all.json:e1d ../exp/results/derived/deployed_extras.json:e1d ../exp/results/E1/g0.json:g0 ../exp/results/E1/g2.json:g2 \
  ../exp/results/E1/comp.json:comp ../exp/results/E1/g2_ranker.json:g2r ../exp/results/E1/authentic.json:au ../exp/results/E4/mbv.json:e4 ../exp/results/E4/mbv_32b.json:e4 ../exp/results/E4/mbv_breadth.json:e4b ../exp/results/E4/mbv_newgen.json:e4n ../exp/results/E4/interaction.json:e4i \
  ../exp/results/E4/mixed.json:e4m ../exp/results/derived/lead_derived.json:dv ../exp/results/derived/e2_paper.json:e2 \
  ../exp/results/derived/e3_paper.json:e3 ../exp/results/derived/fp_paper.json:fp ../exp/results/derived/e4_adjusted.json:e4a ../exp/results/derived/e5_calibration.json:e5c ../exp/results/audit/audit_cap.json:audit.cap ../exp/results/E5/tests.json:e5t \
  ../exp/results/E5/summary_paper.json:e5 ../exp/results/audit/audit_v4.json:audit.v4 ../exp/results/audit/judges_v4.json:audit.j ../exp/results/audit/ranker_v4.json:audit.r ../exp/results/audit/auth_precision.json:audit.auth ../exp/results/derived/reply_hits.json:rh \
  -o numbers/results.tex --digits 3
python3 tools/make_tables.py
python3 tools/style_tables.py   # 2026-10-13: P1-style colours for every table
