#!/usr/bin/env python
"""Score gen.py outputs on IFEval.  python ifeval_score.py --gen E5/ifeval/E5@d0_s0.jsonl [...] --out results/E5/ifeval.json
Output per gen file: prompt-level strict/loose accuracy and instruction-level strict/loose accuracy (lm-eval-harness definitions)."""
import argparse, json, sys
from pathlib import Path
here = Path(__file__).resolve().parent; sys.path.insert(0, str(here / "pylibs"))
from lm_eval.tasks.ifeval import utils as U
ap = argparse.ArgumentParser(); ap.add_argument("--gen", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
data = {f"if{r['key']}": r for r in (json.loads(l) for l in open(here / "ifeval_input_data.jsonl") if l.strip())}
res = {}
for fn in a.gen:
    ps = pl = isl = il = ni = n = 0; ntrunc = 0
    for l in open(fn):
        g = json.loads(l); d = data[g["seed_id"]]; inp = U.InputExample(key=d["key"], instruction_id_list=d["instruction_id_list"], prompt=d["prompt"], kwargs=d["kwargs"])
        s, lo = U.test_instruction_following_strict(inp, g["response"]), U.test_instruction_following_loose(inp, g["response"])
        n += 1; ps += s.follow_all_instructions; pl += lo.follow_all_instructions; isl += sum(s.follow_instruction_list); il += sum(lo.follow_instruction_list); ni += len(s.follow_instruction_list)
        ntrunc += g.get("finish") == "length"
    res[Path(fn).stem] = {"n_prompts": n, "prompt_strict": ps / n, "prompt_loose": pl / n, "inst_strict": isl / ni, "inst_loose": il / ni, "truncated_share": ntrunc / n}
    print(Path(fn).stem, {k: round(v, 3) for k, v in res[Path(fn).stem].items() if k != "n_prompts"})
json.dump(res, open(a.out, "w"), indent=1)
