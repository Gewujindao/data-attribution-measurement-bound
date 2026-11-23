"""Ceiling: (1) counterexamples for 'LDS <= c rho L*'; (2) arithmetic on the published SPELL table (k=500, B=100).
Runs in a few seconds. Usage: python ceiling_counterexamples.py
"""
import numpy as np
from scipy.stats import rankdata

rng = np.random.RandomState(0)
N, n, M = 1000, 10, 4000
masks = np.ones((M, N))
for m in range(M):
    masks[m, rng.choice(N, n, replace=False)] = 0
D = 1 - masks

def corrs(yhat, y, a):
    q, p = -D @ yhat, -D @ y
    pe = lambda u, v: np.corrcoef(u, v)[0, 1]
    sp = lambda u, v: pe(rankdata(u), rankdata(v))
    yc, hc = y - y.mean(), yhat - yhat.mean()
    rho = yc @ hc / np.linalg.norm(yc) / np.linalg.norm(hc)
    return rho, pe(q, a), sp(q, a), pe(p, a), sp(p, a)

print('(1a) Pearson: the error carries the linear part of the response that the exact row misses (kappa > 0)')
y = rng.standard_normal(N); h = rng.standard_normal(N); h -= (h @ y) / (y @ y) * y   # h orthogonal to y
g_dm = y + 0.8 * h                                  # true linear datamodel; actual = linear in the mask + noise
a = -D @ g_dm + 0.3 * np.std(D @ g_dm) * rng.standard_normal(M)
for name, yhat in (('exact row', y), ('row + h (rho<1)', y + 0.8 * h), ('h only (rho=0)', h)):
    r, lp, ls, Lp, Ls = corrs(yhat, y, a)
    print(f'   {name:18s} rho {r:+.3f}  LDS_P {lp:.3f}  rho*L*_P {r*Lp:+.3f}  | LDS_S {ls:.3f}  rho*L*_S {r*Ls:+.3f}')

print('(1b) Spearman with heavy tails, actual exactly linear (L* = 1, kappa = 0)')
for H in (30, 100, 300):
    y = rng.standard_normal(N); y[0] = H
    a = -D @ y
    y_nobig = y.copy(); y_nobig[0] = 0
    y_big = np.zeros(N); y_big[0] = H
    for name, yhat in (('drop the big entry', y_nobig), ('keep only the big entry', y_big)):
        r, lp, ls, Lp, Ls = corrs(yhat, y, a)
        print(f'   H={H:3d} {name:24s} rho {r:.3f}  LDS_S {ls:.3f}  rho*L*_S {r*Ls:.3f}  ratio {ls/(r*Ls):.2f}   (LDS_P {lp:.3f} = rho {r:.3f})')

print('(1c) Gaussian rows, actual = beta*p + independent noise: LDS_S vs rho*L*_S and the link g(r)=6/pi asin(r/2)')
y = rng.standard_normal(N)
a = -D @ y + 0.6 * np.std(D @ y) * rng.standard_normal(M)
for c in (0.3, 1.0, 2.0):
    e = rng.standard_normal(N)
    r, lp, ls, Lp, Ls = corrs(y + c * e, y, a)
    print(f'   noise {c}: rho {r:.3f}  LDS_S {ls:.3f}  rho*L*_S {r*Ls:.3f}  ratio {ls/(r*Ls):.3f}  g(LDS_P) {6/np.pi*np.arcsin(lp/2):.3f}')

print('\n(2) published table (k=500, B=100=0.2k): Spell / spherical-SVD oracle / rank-B SVD of raw Y')
tab = {'CIFAR-10 1%': (0.4538, 0.5189, 0.3420), 'CIFAR-10 5%': (0.3708, 0.4007, 0.2545),
       'TinyStories 1%': (0.5451, 0.6130, 0.4622), 'TinyStories 5%': (0.5488, 0.6217, 0.4635)}
for kk, (sp, orc, raw) in tab.items():
    print(f'   {kk:15s} Spell<oracle {sp < orc}  Spell>raw-SVD {sp > raw}  headroom oracle-Spell {orc-sp:.4f}  Spell/oracle {sp/orc:.3f}')
