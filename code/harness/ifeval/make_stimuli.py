import json, sys
from pathlib import Path
here = Path(__file__).resolve().parent; out = here.parents[1] / "stimuli" / "ifeval"; out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(here / "ifeval_input_data.jsonl") if l.strip()]
with open(out / "IFEVAL.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"seed_id": f"if{r['key']}", "condition": "IFEVAL", "arm": "IF", "text": r["prompt"], "task_type": "ifeval", "context": "ifeval"}) + "\n")
(out / "IFEVAL.READY").write_text("ok"); print(len(rows), "prompts ->", out)
