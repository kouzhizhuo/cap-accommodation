"""Derived counts for the paper text (lead <agent>, 2026-10-02). Reads only synced local files; no new data.
Writes exp/results/derived/lead_derived.json. Counts use the 95% bootstrap CIs already in the inputs."""
import json
from pathlib import Path
R = 'exp/results/'
S = json.load(open(R + 'E1/summary.json')); M = json.load(open(R + 'E4/mbv.json')); G = json.load(open(R + 'E1/g2.json'))
M.update(json.load(open(R + 'E4/mbv_32b.json')))   # OLMo-2-32B, the stage-resolved scale lineage (p3-mbv-big, synced 2026-10-04)
full = ['l1', 'l2', 'l3', 'olmo2_13b', 'olmo2_32b']; allm = full + ['l4']
first = {m: ('sft' if m == 'l4' else 'base') for m in allm}
# D18 lineages (results/E1/newlineages/summary.json, E4/mbv_new.json; synced 2026-10-04 ~22:00) enter every lineage count.
# Their summary's `split` is the v1.4 split, so split (pre-registered, gated), split_rate and rho are recomputed from the point
# estimates with e1_extras.py's definitions (checked against the stored split_prereg of an existing lineage below).
from scipy.stats import spearmanr as _spr
def _ext(nd):
    g = lambda v, k: (nd.get(v) or {}).get(k)
    def S_(k):
        P = [g(v, k) for v in ('bre', 'inf', 'for') if g(v, k) is not None]; Q = [g(v, k) for v in ('aae', 'sge', 'ngp', 'app') if g(v, k) is not None]
        return sum(P) / len(P) - sum(Q) / len(Q)
    xs = [(g(v, 'cr_rate'), t) for v, t in {'bre': 3, 'aus': 2, 'sge': 1, 'aae': 1, 'app': 1, 'ngp': 1}.items() if g(v, 'cr_rate') is not None]
    return {'split_prereg': S_('cr'), 'split_rate': S_('cr_rate'), 'rho': float(_spr([x for x, _ in xs], [t for _, t in xs])[0])}
assert abs(_ext(S['l2']['dpo'])['split_prereg'] - S['l2']['dpo']['split_prereg']) < 1e-4
NL = json.load(open(R + 'E1/newlineages/summary.json')); NEW = ['zephyr_7b', 'tulu2_7b', 'tulu2_13b', 'olmo31_32b']
_ex = {m: {st: _ext(nd) for st, nd in NL[m].items() if isinstance(nd, dict) and 'bre' in nd} for m in NEW}
json.dump(_ex, open(R + 'derived/newlineages_extras.json', 'w'), indent=1)
json.dump({m: NL[m] for m in NEW}, open(R + 'derived/newlineages_summary.json', 'w'), indent=1)   # lineage entries only (json2tex: no duplicate meta keys)
for m in NEW:
    S[m] = NL[m]
    for st, e in _ex[m].items():
        S[m][st]['split_v14'] = S[m][st].get('split'); S[m][st].update(e); S[m][st]['split'] = e['split_prereg']
_MN = json.load(open(R + 'E4/mbv_new.json')); M.update({m: _MN[m] for m in NEW})
allm = allm + NEW; first.update({'zephyr_7b': 'base', 'tulu2_7b': 'sft', 'tulu2_13b': 'sft', 'olmo31_32b': 'base'})   # O31 base run 2026-10-05
def get(d, *ks):
    for k in ks: d = d[k]
    return d
out = {}
# behaviour: S and spelling CR fall from the first released stage to DPO
out['n_lineages'] = len(allm)
out['n_split_falls'] = sum(get(S, m, 'dpo', 'split') < get(S, m, first[m], 'split') for m in allm)
out['n_bre_falls_ci'] = sum(get(S, m, 'dpo', 'bre', 'cr_rate_hi') < get(S, m, first[m], 'bre', 'cr_rate_lo') for m in allm)   # rate-based (G0: spelling)
out['n_inf_falls'] = sum(get(S, m, 'dpo', 'inf', 'cr') < get(S, m, first[m], 'inf', 'cr') for m in allm)
# update diagnostic MB_v (DPO vs SFT)
mb = lambda m, v, f='mb': get(M, m, 'dpo_vs_sft', v, f)
out['n_mb_aae_neg'] = sum(mb(m, 'aae', 'mb_hi') < 0 for m in allm)
out['n_mb_sge_neg'] = sum(mb(m, 'sge', 'mb_hi') < 0 for m in allm)
out['n_mb_inf_neg'] = sum(mb(m, 'inf', 'mb_hi') < 0 for m in allm)
out['n_mb_bre_nonneg'] = sum(mb(m, 'bre', 'mb_hi') >= 0 for m in allm)   # CI reaches 0 or above = not penalised
out['n_mb_aus_nonneg'] = sum(mb(m, 'aus', 'mb_hi') >= 0 for m in allm)
out['n_mb_prestige_spared'] = sum(mb(m, 'bre', 'mb_hi') >= 0 and mb(m, 'aus', 'mb_hi') >= 0 for m in allm)
# register divergence (Holm) at SFT and DPO over stigmatized varieties
for st in ['sft', 'dpo']:
    cells = [(m, v) for m in allm for v in ['aae', 'sge', 'ngp', 'app'] if st in G.get(m, {}) and v in G[m][st]]
    out[f'n_df_div_{st}'] = sum(G[m][st][v]['p_holm'] < 0.05 and G[m][st][v]['dF_v_minus_dF_inf'] > 0 for m, v in cells)
    out[f'n_df_cells_{st}'] = len(cells)
out['n_sge_neg_bre_nonneg'] = sum(mb(m, 'sge') < 0 and mb(m, 'bre', 'mb_hi') >= 0 for m in allm)
base_cells = [(m, v) for m in full for v in ['aae', 'sge', 'ngp', 'app', 'bre', 'aus', 'inf'] if 'base' in G.get(m, {}) and v in G[m]['base']]
out['n_base_df_neg'] = sum(G[m]['base'][v]['dF_v'] < 0 for m, v in base_cells); out['n_base_df_cells'] = len(base_cells)
# E5: key-safe copy of the per-arm summary for json2tex (the raw summary has '@' run ids)
E5 = json.load(open(R + 'E5/summary.json'))
arms = {k: {f: v for f, v in d.items() if isinstance(v, (int, float)) or v is None} for k, d in E5.items() if isinstance(d, dict) and k not in ('runs', 'heldout_item_fp_sae_replies')}
json.dump(arms, open(R + 'E5/summary_paper.json', 'w'), indent=1)
# COMP at the human-corrected rate: rule B (ANNOTATION_PROTOCOL §7) failed on the V4 audit, so each feature's request rate is
# multiplied by the audited precision before the COMP formula of analysis/comp.py (mean over features of min(1, rate/authentic)).
CJ = json.load(open(R + 'E1/comp.json')); AU = json.load(open(R + 'audit/audit_v4.json'))
pr = AU['b']['primary']['precision']; out['comp_corr'] = {'precision': pr['rate']}
for m in ('l2', 'l3', 'l4'):
    for v in ('aae', 'sge'):
        rat = CJ[m]['dpo'][v]['by_feature_ratio']
        cc = lambda q: sum(min(1.0, r * q) for r in rat.values()) / len(rat)
        out['comp_corr'][f'{m}_{v}'] = {'comp': cc(pr['rate']), 'lo': cc(pr['lo']), 'hi': cc(pr['hi']),
                                        'share_ge_50': sum(r * pr['rate'] >= 0.5 for r in rat.values()) / len(rat)}
# PREREG P3 estimand (H3): share of retained features (PREREG §9) at >= 50% of the authentic rate, detector and human-corrected
RET = {'aae': ['finna_gon', 'completive_done', 'ain_t', 'negative_concord', 'yall'], 'sge': ['particles', 'lexicon']}
out['h3'] = {}
for m in ('l2', 'l3', 'l4'):
    for v, fs in RET.items():
        rat = CJ[m]['dpo'][v]['by_feature_ratio']
        out['h3'][f'{m}_{v}'] = {'n_ret': len(fs), 'k_ret': sum(rat[f] >= 0.5 for f in fs), 'k_ret_corr': sum(rat[f] * pr['rate'] >= 0.5 for f in fs)}
# split robustness: pre-registered S (opportunity-based) vs. rate-based S; which fine-tuning step carries the larger decline
out['n_split_rate_falls'] = sum(get(S, m, 'dpo', 'split_rate') < get(S, m, first[m], 'split_rate') for m in allm)
withbase = [m for m in allm if first[m] == 'base']; out['n_lineages_base'] = len(withbase)
for key, tag in (('split', 'prereg'), ('split_rate', 'rate')):
    out[f'n_sft_larger_{tag}'] = sum((get(S, m, 'base', key) - get(S, m, 'sft', key)) > (get(S, m, 'sft', key) - get(S, m, 'dpo', key)) for m in withbase)
# E4 paired between-cue contrast (prestige minus stigmatized MB_v within prompt; exp/results/E4/interaction.json)
IJ = json.load(open(R + 'E4/interaction.json'))
out['n_int_pos'] = sum(v['raw_lo'] > 0 for v in IJ['dpo_vs_sft']['per_lineage'].values()); out['n_int'] = IJ['dpo_vs_sft']['n_lineages']
out['n_int_sftbase_pos'] = sum(v['raw_lo'] > 0 for v in IJ['sft_vs_base']['per_lineage'].values()); out['n_int_sftbase'] = IJ['sft_vs_base']['n_lineages']
out['n_int_loo_pos'] = sum(x > 0 for x in IJ['dpo_vs_sft']['adjusted']['leave_one_lineage_out'].values())
BR = IJ['breadth_instruct_vs_base']
out['n_int_breadth'] = len(BR); out['n_int_breadth_pos'] = sum(v['raw_lo'] > 0 for v in BR.values()); out['n_int_breadth_pt_pos'] = sum(v['raw'] > 0 for v in BR.values())
# preference data: share of prompts with >=1 retained dialect-feature hit (union of AAE/SGE morph; recogniser hits = upper bound)
PC = json.load(open('exp/E4/prefdata_counts.json'))['datasets']
out['pref'] = {tag: 100 * PC[name]['prompt']['union.morph']['share'] for name, tag in
               (('olmo-2-0425-1b-preference-mix', 'olmo2_1b'), ('olmo-2-1124-7b-preference-mix', 'olmo2_7b'), ('llama-3.1-tulu-3-8b-preference-mixture', 'tulu3'))}
out['pref']['max'] = max(out['pref'].values())
out['pref']['floor'] = 100 * json.load(open('exp/E4/prefdata_counts.json'))['net_of_floor']['floor']['union.morph']
out['pref'].update({f'{k}__digits': 2 for k in list(out['pref'])})
# significance marks for the compact update table (CI over requests excludes 0)
MBR = json.load(open(R + 'E4/mbv_breadth.json'))
def mark(lo, hi): return '\\sig{*}' if (lo > 0 or hi < 0) else ''
out['e4sig'] = {}
for m, pair in [(m, 'dpo_vs_sft') for m in allm] + [('l1', 'sft_vs_base'), ('l2', 'sft_vs_base')]:
    for v in ('aae', 'sge', 'bre', 'aus', 'inf'):
        d = M[m][pair][v]; out['e4sig'][f'{m}_{pair}_{v}'] = mark(d['mb_lo'], d['mb_hi'])
for m in ('mistral_7b', 'qwen3_8b'):
    for v in ('aae', 'sge', 'bre', 'aus', 'inf'):
        d = MBR[m]['instruct_vs_base'][v]; out['e4sig'][f'{m}_instruct_vs_base_{v}'] = mark(d['mb_lo'], d['mb_hi'])
for pair, blk in (('dpo_vs_sft', IJ['dpo_vs_sft']['per_lineage']), ('sft_vs_base', IJ['sft_vs_base']['per_lineage']), ('instruct_vs_base', IJ['breadth_instruct_vs_base'])):
    for m, d in blk.items(): out['e4sig'][f'int_{m}_{pair}'] = mark(d['raw_lo'], d['raw_hi'])
# E5 doses as a share of the re-run's training pairs (fixed 120k-pair subset of the 1B mix + n added pairs; notes/E5_MANIFEST.md)
out['e5dose'] = {f'share_{k}': 100 * n / (120000 + n) for k, n in (('1k', 1000), ('5k', 5000), ('20k', 20000), ('60k', 60000))}
out['e5dose'].update({'share_1k__digits': 1, 'share_5k__digits': 0, 'share_20k__digits': 0, 'share_60k__digits': 0})
T5 = json.load(open(R + 'E5/tests.json')); out['e5dose']['ftrain60'] = T5['ftrain']['effect_60k_T_minus_C']['mean']; out['e5dose']['ftrain60__digits'] = 1
# PREREG P4 mixed model (exp/results/E4/mixed.json): tier effect on MB_v, DPO vs SFT, per lineage
MX = json.load(open(R + 'E4/mixed.json'))['dpo_vs_sft']
out['n_mixed_tier_pos'] = sum(MX[m]['tier'] > 0 for m in allm); out['n_mixed_tier_sig'] = sum(MX[m]['tier_lo'] > 0 for m in allm)
# stage attribution robust to framing: SFT->DPO keeps the chat template (all lineages); base->SFT crosses templates, so
# count a decline only if it holds under all three base framings (raw continuation, chat template, transcript scaffold)
EST = {'bre': 'cr_rate', 'inf': 'cr'}
FRAMES = ('base_raw', 'base_f1', 'base'); allframes = [m for m in withbase if all(fr in S[m] for fr in FRAMES)]
out['n_lineages_allframes'] = len(allframes)
for v, st in EST.items():
    out[f'n_{v}_dpo_falls'] = sum(get(S, m, 'dpo', v, st) < get(S, m, 'sft', v, st) for m in allm)
    out[f'n_{v}_sft_falls_allframes'] = sum(all(get(S, m, fr, v, st) > get(S, m, 'sft', v, st) for fr in FRAMES) for m in allframes)
# 32B preference mix (same scan; exp/E4/prefdata_counts_32b.json) vs the 7B (1124) mix: prompt share and chosen-minus-rejected
# AAE density on dialect-bearing prompts (novel spans). Correlational only.
P32 = json.load(open('exp/E4/prefdata_counts_32b.json'))['datasets']['olmo-2-0325-32b-preference-mix']
P7 = json.load(open('exp/E4/prefdata_counts.json'))['datasets']['olmo-2-1124-7b-preference-mix']
out['pref']['olmo2_32b'] = 100 * P32['prompt']['union.morph']['share']; out['pref']['olmo2_32b__digits'] = 2
for tag, d in (('7b', P7), ('32b', P32)):
    x = d['dialect_bearing_prompts']['AAE']['novel']
    out['prefwin'] = out.get('prefwin', {})
    out['prefwin'][tag] = {'diff': x['D_chosen_minus_rejected']['mean'], 'lo': x['D_chosen_minus_rejected']['ci95'][0], 'hi': x['D_chosen_minus_rejected']['ci95'][1], 'tie': 100 * x['share_tie'], 'n': x['D_chosen_minus_rejected']['n']}
    out['prefwin'][tag].update({'tie__digits': 0})
# IFEval on the E5 re-runs (results/E5/ifeval.json; keys have '@') -> key-safe copy, prompt-level strict/loose accuracy x100.
# Several runs per arm (T60k_s0..s2, C60k_s0..s2): the arm value is the MEAN over runs (fixed 2026-10-04; before, the last run
# read silently overwrote the others), with n and per-run strict values kept.
IF = json.load(open(R + 'E5/ifeval.json'))
from collections import defaultdict as _dd
_runs = _dd(list)
for k, v in IF.items():
    arm, seed = k.split('ifeval-')[1].split('_s')                # d0, t20k, t60k, c20k, c60k ; seed
    _runs[arm.lower()].append((int(seed), v))
out['ifeval'] = {}
for arm, rv in _runs.items():
    rv.sort()
    out['ifeval'][arm] = {'strict': 100 * sum(v['prompt_strict'] for _, v in rv) / len(rv), 'loose': 100 * sum(v['prompt_loose'] for _, v in rv) / len(rv),
                          'n': len(rv), 'strict__digits': 1, 'loose__digits': 1}
    for sd, v in rv:
        out['ifeval'][arm][f'strict_s{sd}'] = 100 * v['prompt_strict']; out['ifeval'][arm][f'strict_s{sd}__digits'] = 1
out['ifeval']['drop60'] = out['ifeval']['d0']['strict'] - out['ifeval']['t60k']['strict']; out['ifeval']['drop60__digits'] = 1
out['ifeval']['drop60_vs_c'] = out['ifeval']['c60k']['strict'] - out['ifeval']['t60k']['strict']; out['ifeval']['drop60_vs_c__digits'] = 1
out['ifeval']['maxdrop60_run'] = out['ifeval']['d0']['strict'] - min(v for k, v in out['ifeval']['t60k'].items() if k.startswith('strict_s') and not k.endswith('digits')); out['ifeval']['maxdrop60_run__digits'] = 1
# widely used + newest-generation families (results/breadth, results/newgen): base -> final release, behaviour only
BR = json.load(open(R + 'breadth/summary.json')); NG = json.load(open(R + 'newgen/summary.json'))
FAM = [('Mistral-7B', BR, 'mistral_7b', 'instruct'), ('Mistral-24B', BR, 'mistral_24b', 'instruct'), ('Qwen3-8B', BR, 'qwen3_8b', 'instruct'),
       ('Falcon3-7B', BR, 'falcon3_7b', 'instruct'), ('Granite-3.3-8B', BR, 'granite33_8b', 'instruct'), ('Yi-1.5-9B', BR, 'yi15_9b', 'chat'),
       ('DeepSeek-V2-Lite', BR, 'dsv2lite', 'chat'), ('DeepSeek-LLM-7B', BR, 'dsllm_7b', 'chat'), ('MiMo-7B', BR, 'mimo_7b', 'rl'), ('SmolLM3-3B', BR, 'smollm3_3b', 'rlvr'),
       ('Qwen3.5-4B', NG, 'qwen35_4b', 'instruct'), ('Qwen3.5-9B', NG, 'qwen35_9b', 'instruct')]
fall = lambda D, m, f, v, st: D[m][f][v][st] < D[m]['base'][v][st]
out['fam'] = {'n': len(FAM), 'n_newgen': 2,
              'inf_falls': sum(fall(D, m, f, 'inf', 'cr') for _, D, m, f in FAM),
              'bre_falls': sum(fall(D, m, f, 'bre', 'cr_rate') for _, D, m, f in FAM),
              'split_falls': sum(D[m][f]['split'] < D[m]['base']['split'] for _, D, m, f in FAM),
              'aae_final_max': max(abs(D[m][f]['aae']['cr']) for _, D, m, f in FAM)}
rise = [n for n, D, m, f in FAM if not fall(D, m, f, 'bre', 'cr_rate')]
out['fam']['bre_rise_names'] = ' and '.join(rise) if len(rise) <= 2 else ', '.join(rise[:-1]) + ' and ' + rise[-1]
out['fam']['newgen_bre_falls'] = sum(fall(D, m, f, 'bre', 'cr_rate') for n, D, m, f in FAM if m.startswith('qwen35'))
# real-chat prevalence (WildChat-1M first turns, exp/stimuli/armN/mining_stats.json) vs the AAE share of a preference mix
WS = json.load(open('exp/stimuli/armN/mining_stats.json'))
out['wild'] = {'n': WS['eligible'], 'aae': 100 * WS['prevalence_share']['AAE_bearing(>=1 retained AAE morph hit)'],
               'sge': 100 * WS['prevalence_share']['SGE_bearing(>=1 particle/lexis hit)'], 'aae__digits': 2, 'sge__digits': 2}
out['pref']['aae_morph_7b'] = 100 * json.load(open('exp/E4/prefdata_counts.json'))['datasets']['olmo-2-1124-7b-preference-mix']['prompt']['AAE.morph']['share']
out['pref']['aae_morph_7b__digits'] = 2
# synthetic-gold validation of the Multi-VALUE-ported features (results/detectors/validation_synthetic.json, detectors/validate.py).
# Precision at equal class sizes is reported analytically as R / (R + FPR) (stable); the gate used a sampled estimate, same pass/fail.
VS = json.load(open('exp/results/detectors/validation_synthetic.json'))
vf = VS['features']
cnt = {k: v for k, v in vf.items() if v['gate_pass']}
pb = lambda v: v['recall'] / (v['recall'] + v['FPR'])
out['syn'] = {'n_pool': VS['n_pool_sentences'], 'n_counted': len(cnt), 'n_tested': len(vf),
              'r_min': min(v['recall'] for v in cnt.values()), 'r_max': max(v['recall'] for v in cnt.values()),
              'p_min': min(pb(v) for v in cnt.values()), 'fpr_max': 100 * max(v['FPR'] for v in cnt.values()),
              'aae_copula_r': vf['AAE.copula_absence']['recall'], 'aae_uninflect_r': vf['AAE.uninflect']['recall'],
              'r_min__digits': 2, 'r_max__digits': 2, 'p_min__digits': 2, 'fpr_max__digits': 1,
              'aae_copula_r__digits': 2, 'aae_uninflect_r__digits': 2,
              'gate_consistent': all((pb(v) >= .85) == (v['precision_balanced'] >= .85) for v in vf.values() if v['precision_balanced'] == v['precision_balanced'])}
# compute (results/design/gpu_hours.json, coder; a lower bound that excludes runs still training)
GH = json.load(open('exp/results/design/gpu_hours.json'))
out['gpu'] = {'total': int(round(GH['total_gpu_h'])),
              'dpo': int(round(sum(v for k, v in GH['by_experiment_gpu_h'].items() if k.startswith('E5_DPO'))))}
# deployed chatbots (results/E1/deployed/summary_all.json, coder; 1 sample per request). split_rate and rho recomputed from the
# per-variety point estimates with e1_extras.py's definitions (P = {BRE, INF, FOR}, G = {AAE, SGE, NGP, APP}; rho = Spearman of
# cr_rate with the frozen tiers), which reproduce the stored open-model values exactly; the coder's `split` is the v1.4 split.
from scipy.stats import spearmanr as _sp
DEP = json.load(open('exp/results/E1/deployed/summary_all.json'))
BOTS = ['gpt53chat', 'sonnet46', 'gemini25flash', 'dsv4pro', 'kimik25']
TIER = {'bre': 3, 'aus': 2, 'sge': 1, 'aae': 1, 'app': 1, 'ngp': 1}
dx, cr = {}, lambda b, v, k: (DEP[b]['deployed'].get(v) or {}).get(k)
for b in BOTS:
    P = [cr(b, v, 'cr_rate') for v in ('bre', 'inf', 'for') if cr(b, v, 'cr_rate') is not None]
    G = [cr(b, v, 'cr_rate') for v in ('aae', 'sge', 'ngp', 'app') if cr(b, v, 'cr_rate') is not None]
    xs = [(cr(b, v, 'cr_rate'), t) for v, t in TIER.items() if cr(b, v, 'cr_rate') is not None]
    dx[b] = {'deployed': {'split_rate': sum(P) / len(P) - sum(G) / len(G), 'rho': float(_sp([x for x, _ in xs], [t for _, t in xs])[0])}}
json.dump(dx, open('exp/results/derived/deployed_extras.json', 'w'), indent=1)
openng = []
for f in ('exp/results/E1/summary.json', 'exp/results/breadth/summary.json', 'exp/results/newgen/summary.json'):
    for lin, sts in json.load(open(f)).items():
        if isinstance(sts, dict) and lin != 'meta':
            for st, nd in sts.items():
                if isinstance(nd, dict) and not st.startswith('base') and (nd.get('ngp') or {}).get('cr_rate') is not None:
                    openng.append(nd['ngp']['cr_rate'])
vals = lambda v, k: [cr(b, v, k) for b in BOTS if cr(b, v, k) is not None]
out['dep'] = {'query_date': json.load(open(R + 'E1/deployed_meta.json'))['query_date'], 'n': len(BOTS), 'aae_max': max(vals('aae', 'cr')), 'inf_max': max(vals('inf', 'cr')),
              'bre_min': min(vals('bre', 'cr_rate')), 'bre_max': max(vals('bre', 'cr_rate')),
              'ngp_min': min(vals('ngp', 'cr_rate')), 'ngp_max': max(vals('ngp', 'cr_rate')),
              'ngp_open_max': max(openng), 'n_open_aligned': len(openng)}
# the same split_rate / rho for the breadth and newest-generation rows (their summaries lack them); recomputation agrees with the
# stored values of all 39 open E1 rows to < 1e-5 (per-variety values are stored rounded) and exactly for rho
def _sr_rho(nd):
    f = lambda v: (nd.get(v) or {}).get('cr_rate')
    P = [f(v) for v in ('bre', 'inf', 'for') if f(v) is not None]; G = [f(v) for v in ('aae', 'sge', 'ngp', 'app') if f(v) is not None]
    xs = [(f(v), t) for v, t in TIER.items() if f(v) is not None]
    return {'split_rate': sum(P) / len(P) - sum(G) / len(G), 'rho': float(_sp([x for x, _ in xs], [t for _, t in xs])[0])}
for src, dst in (('breadth', 'breadth_extras'), ('newgen', 'newgen_extras')):
    J = json.load(open(f'exp/results/{src}/summary.json'))
    json.dump({lin: {st: _sr_rho(nd) for st, nd in sts.items() if isinstance(nd, dict) and 'bre' in nd}
               for lin, sts in J.items() if isinstance(sts, dict) and lin not in ('meta', 'prompts', 'validation')},
              open(f'exp/results/derived/{dst}.json', 'w'), indent=1)
# E5 60k per run (D19): run effect = run crheld minus the mean of the 60k controls (summary.json runs); s1's own test from
# results/E5/tests_T60k_s1.json (e5_tests.py unchanged, T60k_s0 excluded)
E5S = json.load(open(R + 'E5/summary.json'))['runs']
c60 = [v['crheld'] for v in E5S['ctl60k'].values()]; cm = sum(c60) / len(c60)
t60 = E5S['aa60k']
out['e560'] = {'n': len(t60)}
for rid, rv in t60.items():                                  # rid = T60k_s<k>
    k = rid.split('_s')[-1]
    out['e560'][f'eff_s{k}'] = rv['crheld'] - cm
    out['e560'][f'ftrain_s{k}'] = rv['ftrain']; out['e560'][f'ftrain_s{k}__digits'] = 1
    tf = Path(R + f'E5/tests_T60k_s{k}.json')
    if tf.exists():
        T1 = json.load(open(tf))['crheld']['effect_60k_T_minus_C']
        out['e560'].update({f'lo_s{k}': T1['lo'], f'hi_s{k}': T1['hi'], f'test_s{k}': T1['mean']})
# register divergence with the validated GYAFC ranker (results/E1/g2_ranker.json); D20 (pre-declared 15:16): HOLDS iff after SFT
# >= ceil(n/4) cells have dF_v - dF_inf > 0 at Holm p < .05 AND no cell has a CI entirely below 0
import math as _m
GR = json.load(open(R + 'E1/g2_ranker.json'))
out['dfr'] = {}
for st in ('sft', 'dpo'):
    cells = [(m, v) for m in allm for v in ['aae', 'sge', 'ngp', 'app'] if st in GR.get(m, {}) and v in GR[m][st]]
    out['dfr'][f'n_cells_{st}'] = len(cells)
    out['dfr'][f'n_div_{st}'] = sum(GR[m][st][v]['p_holm'] < 0.05 and GR[m][st][v]['dF_v_minus_dF_inf'] > 0 for m, v in cells)
    out['dfr'][f'n_opp_{st}'] = sum(GR[m][st][v]['hi'] < 0 for m, v in cells)
bc = [(m, v) for m in full for v in ['aae', 'sge', 'ngp', 'app'] if 'base' in GR.get(m, {}) and v in GR[m]['base']]
out['dfr']['n_base_cells'] = len(bc); out['dfr']['n_base_div_pos'] = sum(GR[m]['base'][v]['dF_v_minus_dF_inf'] > 0 for m, v in bc)
out['dfr']['holds'] = int(out['dfr']['n_div_sft'] >= _m.ceil(out['dfr']['n_cells_sft'] / 4) and out['dfr']['n_opp_sft'] == 0)
# authentic arm (results/E1/authentic.json, coder): D21 (pre-declared 19:06, applied UNCHANGED per lineage; coordinator 2026-10-04):
# YES iff (i) Groenwold AAE cr after DPO <= .02 with CI hi <= .05 AND (ii) cr after DPO <= cr at base under every base framing run
# (the paper's all-three-framings convention); NO iff CI lo > .02; else INCONCLUSIVE.
AU = json.load(open(R + 'E1/authentic.json'))
_lins = [m for m in AU if m not in ('refusal_rule', 'sets', 'refusal') and isinstance(AU[m], dict)
         and AU[m].get('dpo', {}).get('auth_gro', {}).get('aae', {}).get('cr') is not None]   # complete Groenwold results only (file is written incrementally)
_both, _res = 0, {}
for m in _lins:
    d = AU[m]['dpo']['auth_gro']['aae']
    c1 = d['cr'] <= 0.02 and d['cr_hi'] <= 0.05
    bases = [b for b in ('base', 'base_raw', 'base_f1') if b in AU[m] and AU[m][b].get('auth_gro', {}).get('aae', {}).get('cr') is not None]
    c2 = bool(bases) and all(d['cr'] <= AU[m][b]['auth_gro']['aae']['cr'] for b in bases)
    # D21 (ii) applies 'where the same lineage's base model was run'; lineages without a run base are reported apart, never as meeting both
    _res[m] = {'c1': int(c1), 'c2': int(c2), 'n_bases': len(bases), 'verdict': ('YES' if (c1 and c2) else ('NO' if d['cr_lo'] > 0.02 else 'INCONCLUSIVE')) if bases else ('C1 ONLY (no base run)' if c1 else 'INCONCLUSIVE')}
    _both += int(c1 and c2 and bool(bases))
out['auth'] = {'n_lin': sum(v['n_bases'] > 0 for v in _res.values()), 'n_lin_all': len(_lins), 'k_both': _both, 'k_c1': sum(v['c1'] for v in _res.values()), 'by_lineage': _res}
_AS = json.load(open(R + 'E1/authentic_stats.json')); out['auth'].update({f'n_{k}': _AS[k]['n_pairs'] for k in ('auth_gro', 'auth_endive_aae', 'auth_endive_sge')})
out['auth']['post_max'] = max(AU[m][st]['auth_gro']['aae']['cr'] for m in _lins for st in ('sft', 'dpo', 'instruct', 'rlvr') if AU[m].get(st, {}).get('auth_gro', {}).get('aae', {}).get('cr') is not None)
_rf = AU['refusal']['l2']['dpo']['auth_gro']
out['auth'].update({'ref_pair_dpo': 100 * _rf['pair_excluded_share'], 'ref_dia_dpo': 100 * _rf['dialect'], 'ref_sae_dpo': 100 * _rf['sae'],
                    'ref_pair_dpo__digits': 0, 'ref_dia_dpo__digits': 0, 'ref_sae_dpo__digits': 0, 'n_pairs_gro': AU['refusal']['l2']['dpo']['auth_gro']['n_pairs']})
# Qwen3.8-27B (results/E1/newgen_q38/summary.json; chat release only, no base): full-table row; split_rate/rho as for breadth rows
_Q = json.load(open(R + 'E1/newgen_q38/summary.json'))
json.dump({m: {st: _sr_rho(nd) for st, nd in v.items() if isinstance(nd, dict) and 'bre' in nd} for m, v in _Q.items() if m not in ('meta', 'prompts', 'validation')},
          open(R + 'derived/newgen_q38_extras.json', 'w'), indent=1)
json.dump({m: v for m, v in _Q.items() if m not in ('meta', 'prompts', 'validation')}, open(R + 'derived/newgen_q38_summary.json', 'w'), indent=1)
# count phrases for the text ("all n" if every lineage, else "k of n"), so a counterexample can never sit under the word "all"
_ph = lambda k, n: f'all {n}' if k == n else f'{k} of {n}'
_EA = json.load(open(R + 'derived/e4_adjusted.json'))
out['txt'] = {'bre_dpo': _ph(out['n_bre_dpo_falls'], out['n_lineages']), 'inf_dpo': _ph(out['n_inf_dpo_falls'], out['n_lineages']),
              'adj': _ph(_EA['n_adj_pos'], _EA['n_lin']), 'split': _ph(out['n_split_falls'], out['n_lineages']),
              'strat': _ph(_EA['strat']['n_pos'], _EA['strat']['n_lin'])}   # task-type-stratified bootstrap sensitivity (app:prereg)
# openPangu-Embedded-7B-V1.1 (results/E1/openpangu/{summary,check}.json; D23 + addendum 3): three runs (chat = vendor Chinese system
# prompt, chat_enprompt = same prompt in English (primary), chat_nosys); lineage-only copy for json2tex, split_rate/rho, check shares
_O = json.load(open(R + 'E1/openpangu/summary.json')); _OC = json.load(open(R + 'E1/openpangu/check.json'))
json.dump({m: v for m, v in _O.items() if m not in ('meta', 'prompts', 'validation')}, open(R + 'derived/openpangu_summary.json', 'w'), indent=1)
json.dump({m: {st: _sr_rho(nd) for st, nd in v.items() if isinstance(nd, dict) and 'bre' in nd} for m, v in _O.items() if m not in ('meta', 'prompts', 'validation')},
          open(R + 'derived/openpangu_extras.json', 'w'), indent=1)
out['opg'] = {k: {'zh': 100 * c['mostly_chinese_share'], 'trunc': 100 * c['finish_length_share'], 'zh__digits': 0, 'trunc__digits': 0} for k, c in _OC.items()}
# ROBUSTNESS (coder, 2026-10-05 17:07): reply cap 400 vs 1,024 tokens (E1/cap1024.json) and an independent set of sampling seeds
# (E1_s5/summary.json, 4 lineages). Appendix one-liners: largest |change| of any variety's CR, and the replication of the DPO declines.
_C = json.load(open(R + 'E1/cap1024.json'))['cells']; _V = ['aae', 'sge', 'inf', 'bre', 'aus', 'ngp', 'app']
_al = [k for k in _C if k.split('.')[1] in ('dpo', 'instruct', 'rlvr', 'final')]
out['cap'] = {'n_cells': len(_C), 'trunc400_min': min(_C[k]['len_share_400'] for k in _al), 'trunc400_max': max(_C[k]['len_share_400'] for k in _al),
              'trunc1024_max': max(_C[k]['len_share_1024'] for k in _C),
              # the estimator the paper reports per cue (Sec. 3): opportunity-based for INF, AAE, APP; rate-based for BRE, SGE, AUS, NGP
              'max_abs_d_used': max(abs(_C[k][v][e + '_1024'] - _C[k][v][e + '_400']) for k in _C for v, e in (('inf', 'cr'), ('aae', 'cr'), ('app', 'cr'), ('bre', 'cr_rate'), ('sge', 'cr_rate'), ('aus', 'cr_rate'), ('ngp', 'cr_rate'))
                                    if v in _C[k] and _C[k][v].get(e + '_400') is not None and _C[k][v].get(e + '_1024') is not None)}
out['cap'].update({k + '__digits': 2 for k in ('trunc400_min', 'trunc400_max', 'trunc1024_max')}); out['cap']['max_abs_d_used__digits'] = 3
_S5 = json.load(open(R + 'E1_s5/summary.json')); _S = json.load(open(R + 'E1/summary.json'))
_l5 = [l for l in ('l2', 'olmo2_13b', 'l3', 'l4') if l in _S5]
_d = [abs(_S5[l][st][v][e] - _S[l][st][v][e]) for l in _l5 for st in _S5[l] if st in _S.get(l, {}) for v, e in (('bre', 'cr_rate'), ('inf', 'cr'), ('aae', 'cr'))
      if v in _S5[l][st] and v in _S[l][st] and _S5[l][st][v].get(e) is not None and _S[l][st][v].get(e) is not None]
_first5 = {l: ('base' if 'base' in _S5[l] else 'sft') for l in _l5}
_bre = sum(_S5[l]['dpo']['bre']['cr_rate'] < _S5[l][_first5[l]]['bre']['cr_rate'] for l in _l5); _inf = sum(_S5[l]['dpo']['inf']['cr'] < _S5[l][_first5[l]]['inf']['cr'] for l in _l5)
# METHOD (D24): pilot gate on V300 (results/METHOD/gate_pilot.json) and the pre-training pair audit (_incoming/<host>/METHOD/pairs_audit_1b.json)
_G = json.load(open(R + 'METHOD/gate_pilot.json')); _gt = _G['gate']
out['gate'] = {'cr_diff': _G['trained_feature_cr']['diff'], 'cr_lo': _G['trained_feature_cr']['diff_ci'][0], 'cr_hi': _G['trained_feature_cr']['diff_ci'][1],
               'ratio_aae': _G['calibration_ratio_of_means']['treat']['AAE'], 'ratio_sge': _G['calibration_ratio_of_means']['treat']['SGE'],
               'ratio_aae_lo': _G['calibration_ratio_of_means']['treat_ci_AAE'][0], 'ratio_aae_hi': _G['calibration_ratio_of_means']['treat_ci_AAE'][1],
               'ratio_sge_lo': _G['calibration_ratio_of_means']['treat_ci_SGE'][0], 'ratio_sge_hi': _G['calibration_ratio_of_means']['treat_ci_SGE'][1],
               'ins_diff': _G['insertion_per100w']['diff'], 'n_pass': sum(bool(_gt[k]) for k in _gt if k != 'PASS'), 'n_checks': len([k for k in _gt if k != 'PASS'])}
out['gate'].update({k + '__digits': 2 for k in ('cr_diff', 'cr_lo', 'cr_hi', 'ratio_aae', 'ratio_sge', 'ratio_aae_lo', 'ratio_aae_hi', 'ratio_sge_lo', 'ratio_sge_hi')}); out['gate']['ins_diff__digits'] = 3
_PA = json.load(open(R + '../_incoming/<host>/METHOD/pairs_audit_1b.json'))
out['pa'] = {'within25': _PA['mirror_target_vs_achieved']['share_within_25pct'], 'undershoot': -_PA['mirror_target_vs_achieved']['mean_signed_hits'],
             'fluency_o': _PA['sft_logprob_per_token']['O']['mirror_minus_full'], 'within25__digits': 2, 'undershoot__digits': 1, 'fluency_o__digits': 2}
# CAP RESULTS (D24; frozen exp/analysis/cap_tests.py outputs + cap_m4.json): keys dv.capres.* and the main CAP table data
_CT = {k: json.load(open(R + f'METHOD/cap_tests_{k}.json')) for k in ('1b', '1b_unm', '1b_ctl', 'q3', 'q3_unm', 'q3_ctl', 'q3_ins')}
_M4 = json.load(open(R + 'METHOD/cap_m4.json'))
def _dl(t, m): x = t[m]; return {'v': x['delta'], 'lo': x['lo'], 'hi': x['hi']}
cr = {}
for mdl, cap, unm, m4c, m4u in (('b1', '1b', '1b_unm', '1b_cap_vs_control', '1b_onesided_vs_control'), ('q3', 'q3', 'q3_unm', 'q3_cap_vs_control', 'q3_onesided_vs_control')):
    C, U = _CT[cap], _CT[unm]; e = {'n_cap': len(C['arms']['treat']), 'n_ctl': len(C['arms']['control']), 'n_unm': len(U['arms']['treat'])}
    for arm, T in (('cap', C), ('unm', U)):
        for met in ('crheld', 'crg', 'insertion'):
            dd = _dl(T, met); e[f'{arm}_{met}'] = dd['v']; e[f'{arm}_{met}_lo'] = dd['lo']; e[f'{arm}_{met}_hi'] = dd['hi']
        e[f'{arm}_ostd'] = T['overrides']['O_STD']['delta']; e[f'{arm}_ostd_lo'] = T['overrides']['O_STD']['lo']; e[f'{arm}_ostd_hi'] = T['overrides']['O_STD']['hi']
        e[f'{arm}_ratio_aae'] = T['calibration']['AAE']['treat_ratio']; e[f'{arm}_ratio_sge'] = T['calibration']['SGE']['treat_ratio']
        e[f'{arm}_track'] = T['tracking']['SGE']['beta']; e[f'{arm}_track_lo'] = T['tracking']['SGE']['lo']; e[f'{arm}_track_hi'] = T['tracking']['SGE']['hi']
        for b, k4 in (('ifeval', m4c if arm == 'cap' else m4u),):
            x = _M4['ifeval_strict'][k4]; e[f'{arm}_ifeval'] = x['delta']; e[f'{arm}_ifeval_lo'] = x['lo']; e[f'{arm}_ifeval_hi'] = x['hi']
        x = _M4['gsm8k'].get(m4c if arm == 'cap' else m4u)
        if x: e[f'{arm}_gsm'] = x['delta']; e[f'{arm}_gsm_lo'] = x['lo']; e[f'{arm}_gsm_hi'] = x['hi']
    e['fold'] = e['cap_crheld'] / e['unm_crheld'] if e['unm_crheld'] else None
    e['m'] = {**C['criteria'], 'M4': bool(_M4['ifeval_strict'][m4c]['noninferior'])}
    for k in list(e):
        if isinstance(e[k], float): e[k + '__digits'] = 3 if any(t in k for t in ('crheld', 'crg', 'insertion', 'ostd', 'track')) else (1 if any(t in k for t in ('ifeval', 'gsm', 'fold')) else 2)
    e['n_pass'] = sum(e['m'].values()); cr[mdl] = e
out['capres'] = cr
out['txt']['capcrit_b1'] = ', '.join(f"{k} {'passes' if v else 'fails'}" for k, v in sorted(cr['b1']['m'].items()))
out['txt']['capcrit_q3'] = ', '.join(f"{k} {'passes' if v else 'fails'}" for k, v in sorted(cr['q3']['m'].items()))
# table rows: arm levels (CR x100 at table time; ratios; per-100-word rates; IFEval/GSM8K accuracy in %)
def _lv(T, arm_is_treat=True):
    pick = (lambda x: sum(x['treat_runs']) / len(x['treat_runs'])) if arm_is_treat else (lambda x: sum(x['control_runs']) / len(x['control_runs']))
    return {'crheld': pick(T['crheld']), 'crg': pick(T['crg']), 'ins': pick(T['insertion']), 'ostd': T['overrides']['O_STD']['treat' if arm_is_treat else 'control'],
            'r_aae': T['calibration']['AAE']['treat_ratio'], 'r_sge': T['calibration']['SGE']['treat_ratio'], 'track': T['tracking']['SGE']['beta']}
acc = lambda b, k, side: _M4[b][k]['treat_acc' if side == 't' else 'control_acc']
rows = [('OLMo-2-1B', 'Control', len(_CT['1b']['arms']['control']), {**_lv(_CT['1b_ctl']), 'ifeval': acc('ifeval_strict', '1b_cap_vs_control', 'c'), 'gsm': acc('gsm8k', '1b_cap_vs_control', 'c')}),
        ('OLMo-2-1B', 'One-sided', len(_CT['1b_unm']['arms']['treat']), {**_lv(_CT['1b_unm']), 'ifeval': acc('ifeval_strict', '1b_onesided_vs_control', 't'), 'gsm': acc('gsm8k', '1b_onesided_vs_control', 't')}),
        ('OLMo-2-1B', 'CAP', len(_CT['1b']['arms']['treat']), {**_lv(_CT['1b']), 'ifeval': acc('ifeval_strict', '1b_cap_vs_control', 't'), 'gsm': acc('gsm8k', '1b_cap_vs_control', 't')}),
        ('Qwen3-8B', 'Unchanged', 1, {**_lv(_CT['q3_ins']), 'ifeval': acc('ifeval_strict', 'q3_cap_vs_instruct', 'c'), 'gsm': acc('gsm8k', 'q3_cap_vs_instruct', 'c')}),
        ('Qwen3-8B', 'Control', len(_CT['q3']['arms']['control']), {**_lv(_CT['q3_ctl']), 'ifeval': acc('ifeval_strict', 'q3_cap_vs_control', 'c'), 'gsm': acc('gsm8k', 'q3_cap_vs_control', 'c')}),
        ('Qwen3-8B', 'One-sided', len(_CT['q3_unm']['arms']['treat']), {**_lv(_CT['q3_unm']), 'ifeval': acc('ifeval_strict', 'q3_onesided_vs_control', 't'), 'gsm': acc('gsm8k', 'q3_onesided_vs_control', 't')}),
        ('Qwen3-8B', 'CAP', len(_CT['q3']['arms']['treat']), {**_lv(_CT['q3']), 'ifeval': acc('ifeval_strict', 'q3_cap_vs_control', 't'), 'gsm': acc('gsm8k', 'q3_cap_vs_control', 't')})]
json.dump({'rows': rows, 'criteria': {k: v['m'] for k, v in cr.items()}}, open(R + 'derived/cap_table.json', 'w'), indent=1)
# ABLATION (1B, vs the matched control; arms enter as their frozen cap_tests outputs exist): CAP, -O, -A, mirrored SFT, 1.5x target, one-sided (old/new builder)
ABL = [('CAP', '1b', '1b_cap_vs_control'), ('w/o O pairs', '1b_noO_s01', '1b_noO_vs_control'), ('w/o A pairs', '1b_noA_s01', '1b_noA_vs_control'), ('mirrored SFT', '1b_msft', '1b_msft_vs_control'),
       ('target 1.5x (expl.)', '1b_x15', '1b_x15_vs_control'), ('one-sided (rebuilt)', '1b_mx_s01', '1b_mx_vs_control'), ('one-sided', '1b_unm', '1b_onesided_vs_control')]
arows, ab = [], {}
for name, ck, mk in ABL:
    fp = Path(R + f'METHOD/cap_tests_{ck}.json')
    if not fp.exists() or mk not in _M4['ifeval_strict']: continue
    T = json.load(open(fp)); o = T['overrides']['O_STD']
    r_ = {'n': len(T['arms']['treat']), 'crheld': T['crheld']['delta'], 'crg': T['crg']['delta'], 'ins': T['insertion']['delta'], 'ostd': o['delta'], 'ifeval': _M4['ifeval_strict'][mk]['delta']}
    arows.append((name, r_)); key = ck.replace('1b_', '').replace('_s01', '') if ck != '1b' else 'cap'   # noO from the 2-run file (s0+s1)
    ab.update({f'{key}_{k}': v for k, v in r_.items()}); ab.update({f'{key}_{k}__digits': (3 if k in ('crheld', 'crg', 'ins') else 1 if k == 'ifeval' else 2) for k in r_ if k != 'n'})
    ab.update({f'{key}_crheld_lo': T['crheld']['lo'], f'{key}_crheld_hi': T['crheld']['hi'], f'{key}_crheld_lo__digits': 3, f'{key}_crheld_hi__digits': 3})   # 2026-10-07: calibration keys for the confound control
    for v_ in ('AAE', 'SGE'):
        c_ = T['calibration'].get(v_)
        if c_: ab.update({f'{key}_{v_.lower()}_ratio': c_['treat_ratio'], f'{key}_{v_.lower()}_ratio__digits': 2, f'{key}_{v_.lower()}_tail': 100 * c_['treat_tail'], f'{key}_{v_.lower()}_tail__digits': 0})
if 'x15_crheld' in ab and 'cap_crheld' in ab: ab['x15_fold'] = ab['x15_crheld'] / ab['cap_crheld']; ab['x15_fold__digits'] = 1
out['capabl'] = ab
# METHOD SENSITIVITY (2026-10-07): CAP 1B held-out gain under a request-only bootstrap, without the 2% FP exclusion, on the v1 requests
_m = json.load(open(R + 'METHOD/cap_tests_1b.json'))['crheld']; _e = json.load(open(R + 'METHOD/sens_1b_noexcl.json'))['crheld']; _v = json.load(open(R + 'METHOD/sens_1b_v1.json'))['crheld']
out['capsens'] = {'req_lo': _m['reqonly_lo'], 'req_hi': _m['reqonly_hi'], 'noexcl': _e['delta'], 'noexcl_lo': _e['lo'], 'noexcl_hi': _e['hi'], 'v1': _v['delta'], 'v1_lo': _v['lo'], 'v1_hi': _v['hi']}
out['capsens'].update({k + '__digits': 3 for k in list(out['capsens'])})
json.dump({'rows': arows}, open(R + 'derived/cap_abl.json', 'w'), indent=1)
out['s5'] = {'n_lin': len(_l5), 'max_abs_d': max(_d), 'max_abs_d__digits': 3, 'bre_falls': _bre, 'inf_falls': _inf}
out['txt']['s5_bre'] = _ph(_bre, len(_l5)); out['txt']['s5_inf'] = _ph(_inf, len(_l5))
out['src'] = 'E1/summary.json, E4/mbv.json, E1/g2.json, E1/comp.json (local sync 2026-10-01), audit/audit_v4.json (2026-10-03), E4/interaction.json; first stage = base, or SFT for l4'
# E2 interventions ("match my style" instruction, community persona): key-safe copy for json2tex (raw keys contain '@')
import math
e2 = {}
for lin in ('L2', 'L3', 'L4'):
    for name, mdl in json.load(open(R + f'E2/summary_{lin}.json'))['models'].items():
        stage, cond = name.split('@')[1].split('-', 1)
        cond = 'style' if cond == 'style' else 'persona'
        for v, x in mdl['varieties'].items():
            nv = x['novel']
            e2.setdefault(lin.lower(), {}).setdefault(stage, {}).setdefault(cond, {})[v.lower()] = {
                k: nv[k] for k in ('cr', 'cr_lo', 'cr_hi', 'cr_rate', 'cr_rate_lo', 'cr_rate_hi', 'cr_lex', 'cr_lex_lo', 'cr_lex_hi', 'retention_cond')
                if isinstance(nv.get(k), (int, float)) and not math.isnan(nv[k])}
json.dump(e2, open(R + 'derived/e2_paper.json', 'w'), indent=1)
# prior accommodation measures re-run on our replies (E1/style_summary.json; delta = condition minus SAE twin), key-safe for json2tex
ST = json.load(open(R + 'E1/style_summary.json'))['style_ai']
e3 = {}
for g, x in ST.items():
    lin, stage = g.split('@'); stage = stage.replace('-', '_')
    for v, ms in x.items():
        for meas in ('lsm_novel', 'blevins_conv_novel', 'copy_rate'):
            if meas in ms:
                d = ms[meas]; e3.setdefault(lin.lower(), {}).setdefault(stage, {}).setdefault(v.lower(), {})[meas] = {
                    'delta': d['delta'], 'lo': d['ci'][0], 'hi': d['ci'][1], 'mark': '\\sig{*}' if (d['ci'][0] > 0 or d['ci'][1] < 0) else ''}
json.dump(e3, open(R + 'derived/e3_paper.json', 'w'), indent=1)
json.dump(out, open(R + 'derived/lead_derived.json', 'w'), indent=1); print(out)
