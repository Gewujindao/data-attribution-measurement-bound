"""Architecture study: table of kappa against structure, hits of the predicted kappa intervals (PRED), and candidate predictors of its sign.
  python arch_summary.py <results/arch> <cache/arch, with Y and truth>
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import rankdata
RES, SCR = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_points import VARIANTS

PRED = {'w16': (-0.15, -0.25, -0.08), 'b': (-0.12, -0.20, -0.05), 'w64': (-0.06, -0.14, 0.0), 'w128': (0.0, -0.04, 0.04),
        'w256': (0.02, -0.02, 0.07), 'd1': (-0.08, -0.16, 0.0), 'd3': (-0.13, -0.22, -0.05), 'd4': (-0.15, -0.25, -0.05),
        'bn': (0.05, 0.0, 0.12), 'gn': (0.0, -0.08, 0.06), 'cnn': (0.0, -0.08, 0.06), 'e5': (-0.05, -0.12, 0.0),
        'e60': (-0.05, -0.12, 0.0), 'lr0.02': (-0.08, -0.15, -0.02), 'lr0.1': (-0.14, -0.25, -0.05), 'wd1e-3': (-0.11, -0.20, -0.04),
        'relu': (-0.16, -0.25, -0.08)}
RATIO_GT1 = {'w128': 'maybe', 'w256': True, 'bn': True, 'gn': 'maybe', 'cnn': 'maybe'}
def cc(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0))

rows = []
for v in VARIANTS:
    for s, nm in ((0, v), (1, f'{v}_s1')):
        f = os.path.join(RES, f'{nm}_readings.json')
        if not os.path.exists(f):
            continue
        r = json.load(open(f)); m = json.load(open(os.path.join(RES, f'{nm}_meta.json')))
        k = r['k']; kt = r['kappa01']
        t01 = [x for x in kt if x['B'] == round(0.1 * k)][0]; t02 = [x for x in kt if x['B'] == round(0.2 * k)][0]
        e5 = r['e3'][str(round(k / 5))]['med']
        # score minus "correlation x constant" at B=0.2k, Pearson and Spearman, from the raw arrays
        Y = np.concatenate([np.load(x) for x in sorted(glob.glob(os.path.join(SCR, f'{nm}_Y_*.npy')))])
        tr = np.load(os.path.join(SCR, f'{nm}_truth.npz')); D1 = 1 - tr['m01']; d = tr['d01']
        Z = Y - Y.mean(1, keepdims=True); R = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(R, full_matrices=False); B = round(0.2 * k)
        Yh = (U[:, :B] * S[:B]) @ Vt[:B]; rho = np.linalg.norm(Yh, axis=1)
        p, q = -D1 @ Y.T, -D1 @ Yh.T
        rk = lambda A: rankdata(A, axis=0)
        exP = np.mean(cc(q, d)) - np.mean(rho * cc(p, d)); exS = np.mean(cc(rk(q), rk(d))) - np.mean(rho * cc(rk(p), rk(d)))
        ky = all(x['rho2'] <= x['F'] + 1e-9 for x in r['methods'])
        rows.append(dict(v=v, s=s, kappa=r['kappa_true'], k05=kt[0]['kappa'], k10=t01['kappa'], k20=t02['kappa'],
                         ratioS=t02['ratioS'], ratioP=t02['ratioP'], exP=exP, exS=exS, rank90=r['rank90'], Ls=r['Ls'],
                         slope=r['slope'], PN=m['P_over_N'], trl=m['train_loss'], tracc=m['train_acc'], teacc=m['test_acc'],
                         erat=(1 - r['F']['0.1']) / r['F']['0.1'], kloo=e5['kloo'], bW=t01['betaW'], bP=t01['betaP'],
                         e3err=e5['err'], e3r90=e5['r90err'], ky=ky))
out = []
out.append('setting s  P/N   trloss  tracc teacc  L*_S  slope rank90 | k05    k10    k20    kappa  | ratioS ratioP | LDS_P-rhoL*_P LDS_S-rhoL*_S | '
           'beta_W beta_perp  E(1-F)/F kLOO   | E3 s=k/5 err r90 | KyFan')
for x in rows:
    out.append(f'{x["v"]:7s} {x["s"]}  {x["PN"]:5.2f} {x["trl"]:.4f}  {x["tracc"]:.3f} {x["teacc"]:.3f}  {x["Ls"]:.3f} {x["slope"]:.2f}  {x["rank90"]:4d}   | '
               f'{x["k05"]:+.3f} {x["k10"]:+.3f} {x["k20"]:+.3f} {x["kappa"]:+.3f} | {x["ratioS"]:.3f}  {x["ratioP"]:.3f}  | {x["exP"]:+.4f}        {x["exS"]:+.4f}        | '
               f'{x["bW"]:.2f}   {x["bP"]:.2f}      {x["erat"]:.2f}    {x["kloo"]:+.3f} | {x["e3err"]:.3f} {x["e3r90"]:.2f}      | {x["ky"]}')
# seed difference and the "zero" band
by = {}
for x in rows:
    by.setdefault(x['v'], {})[x['s']] = x
out.append('')
out.append('per setting (mean of seeds; zero band = max(0.02, seed difference)); prediction hit = point in interval')
sign_set = []
hits = 0; nh = 0
for v in VARIANTS:
    if v not in by: continue
    ks = [by[v][s]['kappa'] for s in by[v]]
    kd = abs(ks[0] - ks[1]) if len(ks) == 2 else 0.0
    km = float(np.mean(ks)); band = max(0.02, kd)
    sg = 0 if abs(km) < band else int(np.sign(km))
    pr = PRED[v]; hit = pr[1] <= km <= pr[2]; hits += hit; nh += 1
    rs = np.mean([by[v][s]['ratioS'] for s in by[v]])
    out.append(f'  {v:7s} kappa {km:+.3f} (seed diff {kd:.3f}) sign {sg:+d} | predicted {pr[0]:+.2f} [{pr[1]:+.2f},{pr[2]:+.2f}] {"hit" if hit else "MISS"} | '
               f'ratioS(0.2k) {rs:.3f} (>1 predicted: {RATIO_GT1.get(v, False)})')
    for s in by[v]:
        if sg != 0: sign_set.append((v, s, sg, by[v][s]))
out.append(f'  prediction hits {hits}/{nh}')
# candidate predictors: best single threshold accuracy on the nonzero settings x seeds
def best_thr(vals, signs):
    vals, signs = np.array(vals), np.array(signs)
    best = 0
    for t in np.unique(vals):
        for direction in (1, -1):
            pred = np.where(direction * (vals - t) > 0, 1, -1)
            best = max(best, np.mean(pred == signs))
    return best
cands = {'(a) P/N': 'PN', '(b) train loss': 'trl', '(c) slope': 'slope', '(d) L*_S': 'Ls', '(e) (1-F)/F at 0.1k': 'erat',
         '(f) kappa_LOO': 'kloo', '(g1) beta_perp-1': 'bP', '(g2) beta_W': 'bW'}
out.append('')
out.append(f'candidate predictors over {len(sign_set)} nonzero setting x seed cases ({sum(1 for c in sign_set if c[2] > 0)} positive):')
for lab, key in cands.items():
    vals = [c[3][key] for c in sign_set]; sg = [c[2] for c in sign_set]
    if key == 'kloo':
        acc = np.mean(np.sign(vals) == np.array(sg)); out.append(f'  {lab:22s} sign agreement {acc:.0%} (no threshold fitted)')
    out.append(f'  {lab:22s} best single-threshold accuracy {best_thr(vals, sg):.0%}')
# magnitude: rank correlation of kappa with each candidate over all setting x seed cases (with and without lr0.1,
# the non-meta-smooth case L*~0.45)
from scipy.stats import spearmanr
out.append('')
out.append(f'Spearman correlation of kappa with each candidate over {len(rows)} cases (all | without lr0.1):')
for lab, key in cands.items():
    a1 = spearmanr([x[key] for x in rows], [x['kappa'] for x in rows])[0]
    r2 = [x for x in rows if x['v'] != 'lr0.1']
    a2 = spearmanr([x[key] for x in r2], [x['kappa'] for x in r2])[0]
    out.append(f'  {lab:22s} {a1:+.2f} | {a2:+.2f}')
out.append(f'  max over cases: kappa {max(x["kappa"] for x in rows):+.3f}, single-tier kappa {max(max(x["k05"], x["k10"], x["k20"]) for x in rows):+.3f}, '
           f'ratioP {max(x["ratioP"] for x in rows):.3f}, ratioS {max(x["ratioS"] for x in rows):.3f}, LDS_P-rhoL*_P {max(x["exP"] for x in rows):+.4f}, '
           f'LDS_S-rhoL*_S {max(x["exS"] for x in rows):+.4f}; Ky-Fan check all {all(x["ky"] for x in rows)}')
print('\n'.join(out))
open(os.path.join(RES, 'arch_summary.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
