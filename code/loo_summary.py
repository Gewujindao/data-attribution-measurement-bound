"""Leave-one-out study: run every analysis (loo_analyze.py) and check each predicted interval or sign (hit / MISS).
  python loo_summary.py <cache/loo> <cache/arch> <results/loo>
"""
import json, os, subprocess, sys
import numpy as np
SCR, ARCH, RES = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = [('b', 'arch', 'all'), ('w16', 'arch', 'all'), ('w128', 'arch', 'all'), ('b_s1', 'arch', 'all'), ('w128_s1', 'arch', 'all'),
        ('b', 'a10', 'all'), ('b_s1', 'a10', 'all'), ('w128', 'a10', 'all'), ('w128_s1', 'a10', 'all'),
        ('bc', 'a01', 'all'), ('bc_s1', 'a01', 'all'),
        ('rare0', 'a01', 'A'), ('rare0', 'a01', 'B'), ('rare0_s1', 'a01', 'A'), ('rare0_s1', 'a01', 'B')]
R = {}
for m, sp, rw in JOBS:
    r = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(HERE, 'loo_analyze.py'), '--dir', SCR, '--model', m, '--spec', sp,
                        '--rows', rw, '--out', RES, '--arch_dir', ARCH], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[{m} {sp} {rw}] FAILED\n{r.stderr[-1500:]}', flush=True); continue
    R[(m, sp, rw)] = json.load(open(os.path.join(RES, f'{m}_{sp}_{rw}.json')))
out = []
def say(s):
    print(s, flush=True); out.append(s)
say('readings (kappa = mean of the three tiers B=0.05k/0.1k/0.2k; betas = row medians at B=0.1k)')
say('model    spec rows  L*_S  L*_P  slope | kappa  kappa_hat | k05    k10    k20   | bW    bP    bW_hat bP_hat | R2_M  R2_row m<1E  Sp(m,mem) med m  med mem')
for key, r in R.items():
    t = r['tiers']; t1 = t[1]; lo = r.get('loo')
    s = (f'{key[0]:8s} {key[1]:4s} {key[2]:4s} {r["Ls"]:.3f} {r["Lp"]:.3f} {r["slope"]:.2f}  | {r["kappa_true"]:+.3f} '
         f'{r.get("kappa_hat_mean", float("nan")):+.3f}    | {t[0]["kappa"]:+.3f} {t[1]["kappa"]:+.3f} {t[2]["kappa"]:+.3f} | '
         f'{t1["betaW"]:.2f}  {t1["betaP"]:.2f}  ')
    if lo:
        s += (f'{t1["betaW_hat"]:.2f}   {t1["betaP_hat"]:.2f}   | {lo["r2M"]:.2f}  {lo["r2row"]:.2f}   {lo["frac_energy_m_below1"]:.2f}  '
              f'{lo["spearman_m_mem_top30"]:+.2f}     {lo["median_m_top30"]:.2f}   {lo["median_mem_top30"]:.2f}')
    say(s)
say('')
hits = {}
def judge(name, ok, note=''):
    hits.setdefault(name.split(':')[0], []).append(bool(ok))
    say(f'  {name:42s} {"hit" if ok else "MISS"}  {note}')
loo1 = [k for k in R if R[k].get('loo') and k[1] in ('arch', 'a01')]
say('mechanism predictions (LOO decomposition, 1% truth)')
for k in loo1:
    r, lo = R[k], R[k]['loo']
    if r['Ls'] >= 0.7:
        judge(f'M1: {k}', lo['frac_energy_m_below1'] <= 0.25 and lo['median_m_top30'] >= 1.10,
              f'm<1 energy {lo["frac_energy_m_below1"]:.2f}, median m {lo["median_m_top30"]:.2f}')
    judge(f'M2: {k}', lo['r2M'] >= 0.5 and lo['r2M'] - lo['r2row'] >= 0.15, f'R2_M {lo["r2M"]:.2f} R2_row {lo["r2row"]:.2f}')
    if abs(r['kappa_true']) >= 0.03:
        judge(f'M3: {k}', np.sign(r['kappa_hat_mean']) == np.sign(r['kappa_true']) and abs(r['kappa_hat_mean'] - r['kappa_true']) <= 0.06,
              f'kappa {r["kappa_true"]:+.3f} kappa_hat {r["kappa_hat_mean"]:+.3f}')
    judge(f'M4: {k}', lo['spearman_m_mem_top30'] >= 0.4, f'Spearman {lo["spearman_m_mem_top30"]:+.2f}')
    for t in r['tiers']:
        if t['kappa'] < -0.03:
            judge(f'M5: {k} B={t["B"]}', t['EP_m'] > t['EW_m'], f'kappa {t["kappa"]:+.3f} E_piW[m] {t["EW_m"]:.2f} E_piP[m] {t["EP_m"]:.2f}')
        elif t['kappa'] > 0.03:
            judge(f'M5: {k} B={t["B"]}', t['EW_m'] > t['EP_m'], f'kappa {t["kappa"]:+.3f} E_piW[m] {t["EW_m"]:.2f} E_piP[m] {t["EP_m"]:.2f}')
    judge(f'M7: {k}', r['tiers'][1]['betaW_hat'] >= 0.95, f'beta_W_hat {r["tiers"][1]["betaW_hat"]:.2f}')
if all(('%s' % m, 'arch', 'all') in R and R[(m, 'arch', 'all')].get('loo') for m in ('w16', 'b', 'w128')):
    c = {m: (R[(m, 'arch', 'all')]['tiers'][1]['betaP_hat'] - R[(m, 'arch', 'all')]['tiers'][1]['betaW_hat']) / R[(m, 'arch', 'all')]['loo']['beta_loo_overall']
         for m in ('w16', 'b', 'w128')}
    dW = R[('w128', 'arch', 'all')]['tiers'][1]['betaW_hat'] - R[('w16', 'arch', 'all')]['tiers'][1]['betaW_hat']
    dP = R[('w128', 'arch', 'all')]['tiers'][1]['betaP_hat'] - R[('w16', 'arch', 'all')]['tiers'][1]['betaP_hat']
    judge('M6: width order of LOO contrast', c['w16'] > c['b'] > c['w128'] and dW > dP,
          f'contrast w16 {c["w16"]:+.3f} b {c["b"]:+.3f} w128 {c["w128"]:+.3f}; rise of bW_hat {dW:+.2f} vs bP_hat {dP:+.2f}')
say('')
say('new settings')
def kt(m, sp, rw='all'):
    return R[(m, sp, rw)]['kappa_true']
for m in ('bc', 'bc_s1'):
    if (m, 'a01', 'all') in R:
        r = R[(m, 'a01', 'all')]; t1 = r['tiers'][1]
        judge(f'T1: {m}', -0.35 <= r['kappa_true'] <= -0.03 and t1['betaP'] > t1['betaW'] and t1['betaP'] > 1 and 0.6 <= r['Ls'] <= 0.97,
              f'kappa {r["kappa_true"]:+.3f} bW {t1["betaW"]:.2f} bP {t1["betaP"]:.2f} L*_S {r["Ls"]:.3f}')
delta_ok = []
for grp, kin, kout in (('T2', 'w128', 'w128_s1'), ('T4', 'b', 'b_s1')):
    ks = []
    for m in (kin, kout):
        if (m, 'a10', 'all') not in R:
            continue
        r = R[(m, 'a10', 'all')]; t1 = r['tiers'][1]; ks.append(r['kappa_true'])
        if grp == 'T2':
            judge(f'T2: {m} interval/sign', -0.02 <= r['kappa_true'] <= 0.15 and r['kappa_true'] > 0, f'kappa(10%) {r["kappa_true"]:+.3f}')
            judge(f'T2: {m} bW>bP', t1['betaW'] > t1['betaP'], f'bW {t1["betaW"]:.2f} bP {t1["betaP"]:.2f}')
            judge(f'T2: {m} L*_S(10%)', 0.55 <= r['Ls'] <= 0.85, f'L*_S {r["Ls"]:.3f}')
        else:
            k1 = {'b': -0.122, 'b_s1': -0.209}[m]
            judge(f'T4: {m}', -0.18 <= r['kappa_true'] <= 0.02 and 0.03 <= r['kappa_true'] - k1 <= 0.15,
                  f'kappa(10%) {r["kappa_true"]:+.3f}, minus kappa(1%) {r["kappa_true"] - k1:+.3f}')
        if 'betaW_hat' in t1:
            dW, dP = t1['betaW'] - t1['betaW_hat'], t1['betaP'] - t1['betaP_hat']
            ok = dW > 0 and dW > dP; delta_ok.append(ok)
            judge(f'{grp}: {m} pair term Delta_W>Delta_perp', ok, f'Delta_W {dW:+.2f} Delta_perp {dP:+.2f}')
            say(f'      contrast c=(bP-bW)/beta at 10%: {(t1["betaP"] - t1["betaW"]) / r["slope"]:+.3f}')
    if grp == 'T2' and len(ks) == 2:
        band = max(0.02, abs(ks[0] - ks[1]))
        judge('T2: both positive beyond zero band', min(ks) > 0 and np.mean(ks) > band, f'kappas {ks[0]:+.3f} {ks[1]:+.3f} band {band:.3f}')
for m in ('rare0', 'rare0_s1'):
    if (m, 'a01', 'A') in R and (m, 'a01', 'B') in R:
        A, B = R[(m, 'a01', 'A')], R[(m, 'a01', 'B')]
        judge(f'T3: {m} A kappa(0.2k)>0', A['tiers'][2]['kappa'] > 0, f'A tiers {[round(t["kappa"], 3) for t in A["tiers"]]}')
        judge(f'T3: {m} A three-tier in [-0.05,0.20]', -0.05 <= A['kappa_true'] <= 0.20, f'{A["kappa_true"]:+.3f}')
        judge(f'T3: {m} kappa_A-kappa_B>=0.10 every tier', all(a['kappa'] - b['kappa'] >= 0.10 for a, b in zip(A['tiers'], B['tiers'])),
              f'diffs {[round(a["kappa"] - b["kappa"], 3) for a, b in zip(A["tiers"], B["tiers"])]}')
        if 'EW_m' in A['tiers'][2]:
            dA = A['tiers'][2]['EW_m'] - A['tiers'][2]['EP_m']; dB = B['tiers'][2]['EW_m'] - B['tiers'][2]['EP_m']
            judge(f'T3: {m} (E_piW-E_piP)[m] A > B', dA > dB, f'A {dA:+.3f} B {dB:+.3f}')
        judge(f'T3: {m} B control', -0.30 <= B['kappa_true'] <= -0.03, f'{B["kappa_true"]:+.3f}')
        judge(f'T3: {m} L*_S(A)>=0.6', A['Ls'] >= 0.6, f'{A["Ls"]:.3f}')
say('')
m3 = hits.get('M3', [])
say(f'rule 3 (prop 4): M3 {sum(m3)}/{len(m3)}, M2 {sum(hits.get("M2", []))}/{len(hits.get("M2", []))}, M5 {sum(hits.get("M5", []))}/{len(hits.get("M5", []))}')
say(f'rule 4 (prop 6): Delta_W>Delta_perp in {sum(delta_ok)}/{len(delta_ok)} model x seed')
pos = []
for m in ('w128', 'w128_s1'):
    if (m, 'a10', 'all') in R: pos.append(('T2', m, R[(m, 'a10', 'all')]['kappa_true']))
for m in ('rare0', 'rare0_s1'):
    if (m, 'a01', 'A') in R: pos.append(('T3A', m, R[(m, 'a01', 'A')]['tiers'][2]['kappa']))
say('rule 2 (positive construction) inputs: ' + ', '.join(f'{g} {m} {v:+.3f}' for g, m, v in pos))
say('hit counts: ' + ', '.join(f'{k} {sum(v)}/{len(v)}' for k, v in hits.items()))
open(os.path.join(RES, 'loo_summary.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
