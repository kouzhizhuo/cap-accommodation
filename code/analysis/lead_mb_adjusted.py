#!/usr/bin/env python3
"""Per-lineage adjusted paired contrast of the update diagnostic (lead <agent>, 2026-10-04; post hoc breakdown of the
gate-specified test in E4/mbv_interaction.py, which pools lineages with common covariate slopes).
Per lineage, DPO vs SFT: OLS  MB ~ prestige + 1 + z(baseline ll) + z(log tokens), one row per (request, cue), cues AAE/SGE (0)
vs BRE/AUS (1); baseline ll = SFT log-likelihood per token of the matched reply y_v given x_v (as in mbv_interaction.py).
CI: bootstrap over requests (B = 2000, seed 20261012). Also the slope of MB on z(baseline ll), overall and within cue.
Inputs: the synced log-prob files (_incoming/<host>/E4/mbv_lp, _incoming/slurm/E4/mbv_lp_big). Output: results/derived/e4_adjusted.json"""
import json
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1]
DIRS = [EXP / '_incoming/<host>/E4/mbv_lp', EXP / '_incoming/slurm/E4/mbv_lp_big', EXP / '_incoming/<host>/E4/mbv_lp_breadth', EXP / '_incoming/<host>/E4/mbv_lp_newgen',
        EXP / '_incoming/<host>/E4/mbv_lp_new']   # D18 lineages (Zephyr-7B, Tulu-2 7B/13B), synced 2026-10-04 ~22:00
STIG, PRES = ('AAE', 'SGE'), ('BRE', 'AUS')
LIN = {'L1': 'l1', 'L2': 'l2', 'L2B': 'olmo2_13b', 'L5': 'olmo2_32b', 'L3': 'l3', 'L4': 'l4',
       'zephyr_7b': 'zephyr_7b', 'tulu2_7b': 'tulu2_7b', 'tulu2_13b': 'tulu2_13b', 'O31': 'olmo31_32b'}   # D18: new lineages enter the per-lineage test unchanged
PENDING = {}   # OLMo-3.1-32B integrated 2026-10-05 ~03:20 once its E1 summary and MB_v landed

def load(name):
    for d in DIRS:
        p = d / f'{name}.jsonl'
        if p.exists() and p.stat().st_size > 0:
            return {json.loads(l)['item']: json.loads(l) for l in open(p)}

def load_pair(lin, pol, ref):
    """Both files of a pair from the SAME directory (newest directory first), so a rerun is never mixed with an older run."""
    for d in [DIRS[-1]] + DIRS[:-1]:
        pp, rp = d / f'{lin}@{pol}.jsonl', d / f'{lin}@{ref}.jsonl'
        if pp.exists() and rp.exists() and pp.stat().st_size > 0 and rp.stat().st_size > 0:
            rd = lambda q: {json.loads(l)['item']: json.loads(l) for l in open(q)}
            return rd(pp), rd(rp)
    return None, None

def rows_for(lin, pol='dpo', ref='sft'):
    P, R = load_pair(lin, pol, ref); cell = {}
    for k in P:
        if k in R:
            _, sid, v, pk, rk = k.split('|'); cell.setdefault((sid, v), {})[(pk, rk)] = (P[k]['lp'] - R[k]['lp'], P[k]['n'], R[k]['lp'])
    out = []
    for (sid, v), d in cell.items():
        if v not in STIG + PRES or len(d) < 4: continue
        n = lambda pk, rk: d[(pk, rk)][0] / d[(pk, rk)][1]
        mb = (n('v', 'v') - n('v', 'std')) - (n('SAE', 'v') - n('SAE', 'std'))
        out.append((sid, 1.0 if v in PRES else 0.0, mb, d[('v', 'v')][2] / d[('v', 'v')][1], np.log(d[('v', 'v')][1]), v))
    return out

z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
def coefs(rows):
    a = np.array([r[1:5] for r in rows]); G, Y, LL, LN = a.T
    b = np.linalg.lstsq(np.column_stack([G, np.ones_like(G), z(LL), z(LN)]), Y, rcond=None)[0]
    return b[0], b[2]

out = {}
# each estimate gets its own bootstrap stream seeded 20261012 (fixed 2026-10-04: one shared stream made CIs depend on loop order)
for lin, key in LIN.items():
    rng = np.random.default_rng(20261012)
    rows = rows_for(lin); sids = sorted({r[0] for r in rows}); by = {s: [r for r in rows if r[0] == s] for s in sids}
    adj, slope = coefs(rows)
    bs = [coefs([r for s in rng.choice(sids, len(sids)) for r in by[s]])[0] for _ in range(2000)]
    cues = sorted({r[5] for r in rows}); a = np.array([r[1:5] for r in rows])
    C = np.column_stack([np.array([1.0 if r[5] == c else 0.0 for r in rows]) for c in cues])
    within = np.linalg.lstsq(np.column_stack([z(a[:, 2]), z(a[:, 3]), C]), a[:, 1], rcond=None)[0][0]
    raw = float(np.linalg.lstsq(np.column_stack([a[:, 0], np.ones(len(a))]), a[:, 1], rcond=None)[0][0])
    out[key] = {'adj': adj, 'adj_lo': float(np.percentile(bs, 2.5)), 'adj_hi': float(np.percentile(bs, 97.5)), 'raw_ols': raw, 'll_slope': slope,
                'll_slope_within': within, 'n_rows': len(rows),
                'll_prestige': float(a[a[:, 0] == 1, 2].mean()), 'll_stig': float(a[a[:, 0] == 0, 2].mean())}
# D18 lineages pending integration (same estimator; not counted in n_adj_pos / n_lin until integrated)
pend = {}
for lin, key in PENDING.items():
    if load_pair(lin, 'dpo', 'sft')[0] is None: continue
    rng = np.random.default_rng(20261012)
    rows = rows_for(lin); sids = sorted({r[0] for r in rows}); by = {s_: [r for r in rows if r[0] == s_] for s_ in sids}
    adj, slope = coefs(rows); bs = [coefs([r for s_ in rng.choice(sids, len(sids)) for r in by[s_]])[0] for _ in range(2000)]
    a_ = np.array([r[1:5] for r in rows])
    raw = float(np.linalg.lstsq(np.column_stack([a_[:, 0], np.ones(len(a_))]), a_[:, 1], rcond=None)[0][0])
    braw = [float(np.linalg.lstsq(np.column_stack([np.array([r[1] for s_ in sel for r in by[s_]]), np.ones(sum(len(by[s_]) for s_ in sel))]),
                                  np.array([r[2] for s_ in sel for r in by[s_]]), rcond=None)[0][0]) for sel in [rng.choice(sids, len(sids)) for _ in range(2000)]]
    pend[key] = {'adj': adj, 'adj_lo': float(np.percentile(bs, 2.5)), 'adj_hi': float(np.percentile(bs, 97.5)), 'raw_ols': raw,
                 'raw_lo': float(np.percentile(braw, 2.5)), 'raw_hi': float(np.percentile(braw, 97.5)), 'll_slope': slope, 'n_rows': len(rows),
                 'd18': 'REPLICATES' if np.percentile(bs, 2.5) > 0 else ('REVERSED' if np.percentile(bs, 97.5) < 0 else 'INCONCLUSIVE'),
                 'slope_32b_rule_met': bool(slope <= -0.020)}
out['d18_pending'] = pend
# breadth / newest-generation updates (descriptive generalisation; same estimator), keyed out['breadth'][<family>_<pol>_vs_<ref>]
BREADTH = [('mistral_7b', 'instruct', 'base'), ('mistral_24b', 'instruct', 'base'), ('qwen3_8b', 'instruct', 'base'), ('falcon3_7b', 'instruct', 'base'),
           ('granite33_8b', 'instruct', 'base'), ('yi15_9b', 'chat', 'base'), ('mimo_7b', 'rl', 'base'), ('smollm3_3b', 'final', 'base'),
           ('smollm3_3b', 'apo', 'sft'), ('qwen35_4b', 'instruct', 'base'), ('qwen35_9b', 'instruct', 'base'), ('dsv2lite', 'chat', 'base'),
           ('qwen25_7b', 'instruct', 'base'), ('dsllm_7b', 'chat', 'base')]   # popular-models rows (2026-10-05); skipped until their log-probs land
br = {}
for fam, pol, ref in BREADTH:
    if load_pair(fam, pol, ref)[0] is None: print('skip (no log-probs):', fam, pol, ref); continue
    rng = np.random.default_rng(20261012)
    rows = rows_for(fam, pol, ref); sids = sorted({r[0] for r in rows}); by = {s_: [r for r in rows if r[0] == s_] for s_ in sids}
    adj, slope = coefs(rows); bs = [coefs([r for s_ in rng.choice(sids, len(sids)) for r in by[s_]])[0] for _ in range(2000)]
    a_ = np.array([r[1:5] for r in rows]); G_ = a_[:, 0]
    raw = float(np.linalg.lstsq(np.column_stack([G_, np.ones_like(G_)]), a_[:, 1], rcond=None)[0][0])
    rr = np.random.default_rng(20261012)   # raw-contrast CI (main-table popular rows, 2026-10-05): its own stream, as every estimate
    braw = [float(np.linalg.lstsq(np.column_stack([np.array([r[1] for s_ in sel for r in by[s_]]), np.ones(sum(len(by[s_]) for s_ in sel))]),
                                  np.array([r[2] for s_ in sel for r in by[s_]]), rcond=None)[0][0]) for sel in [rr.choice(sids, len(sids)) for _ in range(2000)]]
    br[f'{fam}_{pol}_vs_{ref}'] = {'adj': adj, 'adj_lo': float(np.percentile(bs, 2.5)), 'adj_hi': float(np.percentile(bs, 97.5)), 'raw': raw,
                                   'raw_lo': float(np.percentile(braw, 2.5)), 'raw_hi': float(np.percentile(braw, 97.5)), 'll_slope': slope, 'n_rows': len(rows)}
out['breadth'] = br
# SFT vs base of the stage-resolved D18 lineages with a released base (Zephyr-7B); same estimator, kept OUT of the breadth count
sb = {}
for fam, pol, ref in [('zephyr_7b', 'sft', 'base'), ('O31', 'sft', 'base'), ('L1', 'sft', 'base'), ('L2', 'sft', 'base')]:   # L1/L2 added 2026-10-06 (main update table, no empty cells)
    if load_pair(fam, pol, ref)[0] is None: continue
    rng = np.random.default_rng(20261012)
    rows = rows_for(fam, pol, ref); sids = sorted({r[0] for r in rows}); by = {s_: [r for r in rows if r[0] == s_] for s_ in sids}
    adj, slope = coefs(rows); bs = [coefs([r for s_ in rng.choice(sids, len(sids)) for r in by[s_]])[0] for _ in range(2000)]
    a_ = np.array([r[1:5] for r in rows]); G_ = a_[:, 0]
    raw = float(np.linalg.lstsq(np.column_stack([G_, np.ones_like(G_)]), a_[:, 1], rcond=None)[0][0])
    sb[{'O31': 'olmo31_32b', 'L1': 'l1', 'L2': 'l2'}.get(fam, fam)] = {'adj': adj, 'adj_lo': float(np.percentile(bs, 2.5)), 'adj_hi': float(np.percentile(bs, 97.5)), 'raw': raw, 'll_slope': slope}
out['sftbase'] = sb
# per-variety MB_v for updates whose mbv file lacks the pair (SmolLM3 final vs base; main update table): mean over requests of the
# rows_for() matching bonus, request bootstrap B 2000, own stream per estimate (same estimator as E4/mbv_analyze.py's length-normalised MB)
mx = {}
for fam, pol, ref in [('smollm3_3b', 'final', 'base')]:
    if load_pair(fam, pol, ref)[0] is None: continue
    rows = rows_for(fam, pol, ref); d = {}
    for v in STIG + PRES:
        x = np.array([r[2] for r in rows if r[5] == v]); rng = np.random.default_rng(20261012)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        d[v.lower()] = {'mb': float(x.mean()), 'mb_lo': float(np.percentile(bs, 2.5)), 'mb_hi': float(np.percentile(bs, 97.5)), 'n': int(len(x))}
    mx[f'{fam}_{pol}_vs_{ref}'] = d
out['mbv_extra'] = mx
out['breadth_n'] = len(br); out['breadth_adj_pos'] = sum(v['adj_lo'] > 0 for v in br.values()); out['breadth_adj_neg'] = sum(v['adj_hi'] < 0 for v in br.values())
out['breadth_raw_pos'] = sum(v['raw'] > 0 for v in br.values())
NAMES = {'mistral_7b_instruct_vs_base': 'Mistral-7B', 'mistral_24b_instruct_vs_base': 'Mistral-Small-24B', 'qwen3_8b_instruct_vs_base': 'Qwen3-8B',
         'falcon3_7b_instruct_vs_base': 'Falcon3-7B', 'granite33_8b_instruct_vs_base': 'Granite-3.3-8B', 'yi15_9b_chat_vs_base': 'Yi-1.5-9B',
         'smollm3_3b_final_vs_base': 'SmolLM3-3B', 'smollm3_3b_apo_vs_sft': "SmolLM3's preference stage", 'qwen35_4b_instruct_vs_base': 'Qwen3.5-4B',
         'qwen35_9b_instruct_vs_base': 'Qwen3.5-9B', 'dsv2lite_chat_vs_base': 'DeepSeek-V2-Lite', 'mimo_7b_rl_vs_base': 'MiMo-7B',
         'qwen25_7b_instruct_vs_base': 'Qwen2.5-7B', 'dsllm_7b_chat_vs_base': 'DeepSeek-LLM-7B'}
_inc = [NAMES[k] for k, v in br.items() if v['adj_lo'] <= 0 <= v['adj_hi']]
out['breadth_inconcl_names'] = (', '.join(_inc[:-1]) + ' and ' + _inc[-1]) if len(_inc) > 1 else (_inc[0] if _inc else 'none')
out['n_adj_pos'] = sum(v['adj_lo'] > 0 for k, v in out.items() if isinstance(v, dict) and 'adj_lo' in v)
out['n_lin'] = len(LIN)
out['max_abs_ll_slope_others'] = max(abs(v['ll_slope']) for k, v in out.items() if isinstance(v, dict) and 'll_slope' in v and k != 'olmo2_32b')
# sensitivity (integrity sweep 2026-10-05): the pre-registration names a bootstrap stratified by task type, but every CI above
# resamples requests unstratified; re-run the per-lineage adjusted contrast with requests resampled within their task type
TT = {json.loads(l)['seed_id']: json.loads(l)['task_type'] for l in open(EXP / 'stimuli/seeds_v1.jsonl')}
st = {}
for lin, key in LIN.items():
    rng = np.random.default_rng(20261012)
    rows = rows_for(lin); sids = sorted({r[0] for r in rows}); by = {s_: [r for r in rows if r[0] == s_] for s_ in sids}
    strata = {}
    for s_ in sids: strata.setdefault(TT[s_], []).append(s_)
    bs = [coefs([r for g in strata.values() for s_ in rng.choice(g, len(g)) for r in by[s_]])[0] for _ in range(2000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    st[key] = {'adj_lo': lo, 'adj_hi': hi, 'width_ratio': (hi - lo) / (out[key]['adj_hi'] - out[key]['adj_lo']), 'n_strata': len(strata)}
out['strat'] = {'by_lineage': st, 'n_pos': sum(v['adj_lo'] > 0 for v in st.values()), 'n_lin': len(st),
                'min_width_ratio': min(v['width_ratio'] for v in st.values()), 'max_width_ratio': max(v['width_ratio'] for v in st.values()),
                'min_width_ratio__digits': 2, 'max_width_ratio__digits': 2}
json.dump(out, open(EXP / 'results/derived/e4_adjusted.json', 'w'), indent=1)
print({k: (round(v['adj'], 4), round(v['adj_lo'], 4), round(v['ll_slope'], 4)) if isinstance(v, dict) and 'adj' in v else v for k, v in out.items() if k != 'breadth'})
for k, v in br.items(): print(k, 'raw %+.4f adj %+.4f [%+.4f, %+.4f] slope %+.4f' % (v['raw'], v['adj'], v['adj_lo'], v['adj_hi'], v['ll_slope']))
