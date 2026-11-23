"""Baselines: (1) B80 vs k on fine grid (spherical SVD oracle); (2) cheap prior (endpoint IF) + rank-B oracle on residual.
Usage: python baseline_kscale_prior.py <base.npz> <k300.npz>
"""
import sys
import numpy as np
from scipy.stats import rankdata

def load(p):
    D = np.load(p); Y = D['Yep'].sum(1); return D, Y
def lds_per(Yh, m, d):
    pred = -(1 - m) @ Yh.T
    rp, rd = rankdata(pred, axis=0), rankdata(d, axis=0)
    rp -= rp.mean(0); rd -= rd.mean(0)
    return (rp * rd).sum(0) / np.sqrt((rp ** 2).sum(0) * (rd ** 2).sum(0) + 1e-300)

D, Y = load(sys.argv[2])
m, d = D['m01'], D['d01']
k = Y.shape[0]
Rn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
print('B80 (abs 0.8) and B90rel (0.9 x exact) vs k, spherical SVD oracle, 3 random subsets each (k300 run)')
for kk in [x for x in (25, 50, 100, 150, 200, 297, 300) if x <= k]:
    out = []
    for rep in range(3 if kk < k else 1):
        sub = np.sort(np.random.RandomState(10 + rep).choice(k, kk, replace=False))
        u, s, vt = np.linalg.svd(Rn[sub], full_matrices=False)
        ex = lds_per(Y[sub], m, d[:, sub]).mean()
        b80 = b90 = None
        for B in range(1, kk + 1):
            v = lds_per((u[:, :B] * s[:B]) @ vt[:B], m, d[:, sub]).mean()
            if b90 is None and v >= 0.9 * ex: b90 = B
            if b80 is None and v >= 0.8: b80 = B
            if b80 and b90: break
        out.append((b80, b90))
    b80s = [o[0] for o in out]; b90s = [o[1] for o in out]
    r80 = f'{np.mean(b80s)/kk:.2f}' if None not in b80s else 'not reached'
    print(f'  k={kk}: exact {ex:.3f}; B80 {b80s} (B80/k {r80});  B90rel {b90s} (/k {np.mean(b90s)/kk:.2f})')

if sys.argv[1] == 'none':
    sys.exit()
D, Y = load(sys.argv[1])
m, d = D['m01'], D['d01']
GV, GtV, lam = D['GV'], D['GtV'], D['lam']
ev = np.abs(lam) + 0.01 * np.abs(lam).max()
IF = -(GtV / ev) @ GV.T
print(f'\nbase: endpoint IF LDS {lds_per(IF, m, d).mean():.3f}')
c = (IF * Y).sum(1) / (IF * IF).sum(1)            # oracle per-row scale of the prior
R = Y - c[:, None] * IF
print(f'  fraction of ||Y||^2 left after oracle-scaled IF prior: {(R**2).sum()/(Y**2).sum():.3f}')
Rr = R / np.linalg.norm(R, axis=1, keepdims=True)
u, s, vt = np.linalg.svd(Rr, full_matrices=False)
rnR = np.linalg.norm(R, axis=1)
print('  prior + rank-B oracle on normalized residual (rescaled by true residual norm):',
      '  '.join(f'B={B}:{lds_per(c[:,None]*IF + rnR[:,None]*((u[:,:B]*s[:B])@vt[:B]), m, d).mean():.3f}' for B in (0, 5, 10, 20, 30, 50)))
u2, s2, vt2 = np.linalg.svd(R, full_matrices=False)
print('  prior + rank-B raw truncated SVD of residual:',
      '  '.join(f'B={B}:{lds_per(c[:,None]*IF + (u2[:,:B]*s2[:B])@vt2[:B], m, d).mean():.3f}' for B in (0, 5, 10, 20, 30, 50)))
