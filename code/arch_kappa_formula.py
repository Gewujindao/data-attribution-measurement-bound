"""Architecture study: the exact kappa formula on the 34 cases (check of an identity on existing runs).
  python arch_kappa_formula.py <cache/arch> <results/arch>
kappa_f = (beta_W - beta_perp) * rho*sqrt(1-rho^2) * sd(p)/(sd(a)*sqrt(1-L_P^2)); the population identity is exact,
in-sample it differs only through the sample covariance of q_W and q_perp.  Also splits kappa into its three factors
(contrast c = (beta_perp - beta_W)/beta, geometry rho*sqrt(1-rho^2), noise L_P/sqrt(1-L_P^2)) and ranks them against kappa.
"""
import glob, os, sys
import numpy as np
from scipy.stats import spearmanr
SCR, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_points import VARIANTS
def cc(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0))
def slope(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return (x * y).sum(0) / (x * x).sum(0)
rows, out = [], []
out.append('case        kappa(3 tiers) kappa_f   mean|row diff| | c=(bP-bW)/beta(median) rho*sqrt(1-rho^2)  L_P/sqrt(1-L_P^2)  (B=10, row means)')
for v in VARIANTS:
    for nm in (v, f'{v}_s1'):
        Y = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(SCR, f'{nm}_Y_*.npy')))])
        tr = np.load(os.path.join(SCR, f'{nm}_truth.npz')); D1 = 1 - tr['m01']; d = tr['d01']
        k = Y.shape[0]; Yc = Y - Y.mean(1, keepdims=True); nrm = np.linalg.norm(Yc, axis=1); R = Yc / nrm[:, None]
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        P = -D1 @ Y.T; Lp = cc(P, d); beta = slope(P, d)
        ks, kfs, dif = [], [], 0.0
        for B in (5, 10, 20):
            Uw = (U[:, :B] * S[:B]) @ Vt[:B] * nrm[:, None]; Vp = Yc - Uw
            qW, qP = -D1 @ Uw.T, -D1 @ Vp.T
            rho = np.linalg.norm(Uw, axis=1) / nrm
            rqa, rqp = cc(qW, d), cc(qW, P)
            kap = (rqa - rqp * Lp) / np.sqrt((1 - rqp ** 2) * (1 - Lp ** 2))
            bW, bP = slope(qW, d), slope(qP, d)
            kf = (bW - bP) * rho * np.sqrt(1 - rho ** 2) * P.std(0) / (d.std(0) * np.sqrt(1 - Lp ** 2))
            ks.append(kap.mean()); kfs.append(kf.mean()); dif = max(dif, np.abs(kap - kf).mean())
            if B == 10:
                c = np.median((bP - bW) / beta); geo = np.mean(rho * np.sqrt(1 - rho ** 2)); noi = np.mean(Lp / np.sqrt(1 - Lp ** 2))
        rows.append(dict(nm=nm, v=v, kappa=np.mean(ks), kf=np.mean(kfs), c=c, geo=geo, noi=noi))
        out.append(f'{nm:11s} {np.mean(ks):+.3f}         {np.mean(kfs):+.3f}    {dif:.3f}     | {c:+.3f}          {geo:.3f}              {noi:.2f}')
K = [r['kappa'] for r in rows]
out.append('')
out.append('Spearman over 34 cases (all | without lr0.1):')
for lab, key in (('kappa_f', 'kf'), ('contrast c', 'c'), ('geometry', 'geo'), ('noise factor', 'noi')):
    a1 = spearmanr([r[key] for r in rows], K)[0]
    r2 = [r for r in rows if r['v'] != 'lr0.1']
    a2 = spearmanr([r[key] for r in r2], [r['kappa'] for r in r2])[0]
    out.append(f'  kappa vs {lab:13s} {a1:+.2f} | {a2:+.2f}')
print('\n'.join(out))
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'kappa_formula.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
