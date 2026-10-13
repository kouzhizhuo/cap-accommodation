#!/usr/bin/env python3
"""D24 M4: capability non-inferiority of CAP (IFEval prompt-level strict, primary; loose and GSM8K secondary), lead <agent>.
Delta = mean accuracy over the treatment arm's runs - over the comparison arm's runs, in points (x100); 95% CI from a two-level
bootstrap (runs within arm with replacement, then prompts, paired across arms), B 2000, seed 20261012. M4 holds if CI-lo > -2.
Input: results/METHOD/perprompt/{ifeval,gsm8k}_perprompt.jsonl ({run, id, correct[, strict, loose]}). Output: results/METHOD/cap_m4.json."""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1]; D = EXP / "results/METHOD/perprompt"
B, SEED, MARGIN = 2000, 20261012, -2.0
CMP = {   # name: (treatment regex, comparison regex)
    "1b_cap_vs_control": (r"E5@CAP60k_s\d+", r"E5@C60k_s\d+"), "1b_onesided_vs_control": (r"E5@T60k_s\d+", r"E5@C60k_s\d+"),
    "q3_cap_vs_control": (r"Q3@CAP_s\d+", r"Q3@CTL_s\d+"), "q3_onesided_vs_control": (r"Q3@UNM_s\d+", r"Q3@CTL_s\d+"),
    "q3_cap_vs_instruct": (r"Q3@CAP_s\d+", r"Q3@INSTRUCT"), "q3_sysprompt_vs_instruct": (r"Q3@INSTRUCT-B3", r"Q3@INSTRUCT"),
    "1b_noO_vs_control": (r"E5@MC60k_s\d+", r"E5@C60k_s\d+"), "1b_noA_vs_control": (r"E5@MUO60k_s\d+", r"E5@C60k_s\d+"), "1b_msft_vs_control": (r"E5@MSFT_s\d+", r"E5@C60k_s\d+"),
    "1b_x15_vs_control": (r"E5@CAPx15_s\d+", r"E5@C60k_s\d+"), "1b_mx_vs_control": (r"E5@MX60k_s\d+", r"E5@C60k_s\d+")}   # ablations (2026-10-06)


def load(f, key):
    M = defaultdict(dict)
    for l in open(D / f):
        r = json.loads(l); M[r["run"]][r["id"]] = float(r.get(key, r["correct"]))
    return M


def contrast(M, rt, rc):
    T = sorted(r for r in M if re.fullmatch(rt, r)); C = sorted(r for r in M if re.fullmatch(rc, r))
    if not T or not C: return None
    ids = sorted(set.intersection(*[set(M[r]) for r in T + C]))
    t = np.array([[M[r][i] for i in ids] for r in T]); c = np.array([[M[r][i] for i in ids] for r in C])
    rng = np.random.default_rng(SEED); n = len(ids); bs = []
    for _ in range(B):
        tt = t[rng.integers(0, len(T), len(T))].mean(0); cc = c[rng.integers(0, len(C), len(C))].mean(0); i = rng.integers(0, n, n)
        bs.append(100 * (tt[i].mean() - cc[i].mean()))
    d = 100 * (t.mean() - c.mean())
    return {"treat_runs": T, "control_runs": C, "n_prompts": n, "treat_acc": 100 * t.mean(), "control_acc": 100 * c.mean(), "delta": d,
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "noninferior": bool(np.percentile(bs, 2.5) > MARGIN),
            "per_run_treat": [100 * x for x in t.mean(1)], "per_run_control": [100 * x for x in c.mean(1)],
            **{k + "__digits": 1 for k in ("treat_acc", "control_acc", "delta", "lo", "hi")}}


out = {"margin_points": MARGIN, "spec": "D24 M4, two-level bootstrap (runs, prompts), B 2000, seed 20261012"}
for bench, f, key in (("ifeval_strict", "ifeval_perprompt.jsonl", "strict"), ("ifeval_loose", "ifeval_perprompt.jsonl", "loose"), ("gsm8k", "gsm8k_perprompt.jsonl", "correct")):
    M = load(f, key); out[bench] = {k: v for k, (rt, rc) in CMP.items() if (v := contrast(M, rt, rc))}
(EXP / "results/METHOD/cap_m4.json").write_text(json.dumps(out, indent=1))
for bench in ("ifeval_strict", "gsm8k"):
    for k, v in out[bench].items():
        print(f"{bench:13s} {k:26s} {v['treat_acc']:.1f} vs {v['control_acc']:.1f}  d={v['delta']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}] runs {len(v['treat_runs'])}/{len(v['control_runs'])} NI={v['noninferior']}")
