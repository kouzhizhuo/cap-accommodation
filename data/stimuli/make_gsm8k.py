#!/usr/bin/env python
"""GSM8K test (1,319 problems, official openai/gsm8k test split as staged on <host>) -> stimuli/gsm8k/GSM8K.jsonl, 0-shot CoT.  Gold answer kept in `gold`."""
import json, sys, re
import pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]
t = pq.read_table(src).to_pylist()
with open(out, "w") as f:
    for i, r in enumerate(t):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        f.write(json.dumps({"seed_id": f"gsm{i:04d}", "condition": "GSM8K", "arm": "GSM", "task_type": "gsm8k", "context": "gsm8k", "gold": gold,
                            "text": r["question"].strip() + "\n\nThink step by step, then end with the line \"The answer is N\" where N is the final number."}) + "\n")
print(len(t))
