"""Ceiling: how the saving factor k/B scales with the number of test points k.
Measured: rank90 and oracle B at LDS targets on random subsets of k' test points (S3 has k=900, S1 k=300).
Model: the E1 surrogate spectrum (power law + floor) fitted on s rows, simulated for k' up to 10 N.
Usage: python ceiling_k_scaling.py <point> <s_fit>
"""
import os, sys, time
import numpy as np
from ceiling_common import SCR, POINTS, load, cnorm, ranks, corr_cols, oracle_curve, energy_curve, rank_at

t0 = time.time()
name, sfit = sys.argv[1], int(sys.argv[2])
Y, tr, meta = load(POINTS[name])
k, N = Y.shape
masks, d = tr['m01'], tr['d01']
R = cnorm(Y)
Lstar = corr_cols(ranks(-(1 - masks) @ Y.T), ranks(d))
print(f'[{name}] k={k} N={N} L* {Lstar.mean():.3f}')
print("  k'   rank90 (3 subsets)        B at LDS 0.8 | B at 0.9 L*   k/rank90")
rs = np.random.RandomState(3)
meas = []
for kk in [x for x in (25, 50, 100, 150, 200, 300, 450, 600, 900) if x <= k]:
    r90s, b80s, b9s = [], [], []
    for rep in range(3 if kk < k else 1):
        sub = np.sort(rs.choice(k, kk, replace=False))
        Ls, F = oracle_curve(R[sub], masks, d[:, sub], np.arange(1, kk + 1))
        r90s.append(rank_at(F, 0.9))
        b80s.append(int(np.argmax(Ls >= 0.8)) + 1 if Ls.max() >= 0.8 else None)
        b9s.append(int(np.argmax(Ls >= 0.9 * Lstar[sub].mean())) + 1)
    meas.append((kk, np.mean(r90s)))
    print(f'  {kk:4d}  {r90s}  {b80s} | {b9s}   {kk/np.mean(r90s):.2f}', flush=True)
ks = np.array([m[0] for m in meas]); rr = np.array([m[1] for m in meas])
loc = np.diff(np.log(rr)) / np.diff(np.log(ks))
print('  local exponent d log rank90 / d log k between consecutive k\':', ' '.join(f'{x:.2f}' for x in loc))
from scipy.optimize import curve_fit
(a_, D_), _ = curve_fit(lambda n, a, D: a * n / (1 + n / D), ks.astype(float), rr, p0=(0.9, k))
print(f'  saturating fit rank90(k\') = a k\'/(1+k\'/D): a {a_:.3f} D {D_:.0f}; max rel. misfit {np.max(np.abs(a_*ks/(1+ks/D_)/rr-1)):.3f}; '
      f'limit a*D {a_*D_:.0f}; implied local exponent 1/(1+k/D) at k=100/300/900: ' +
      ' '.join(f'{1/(1+x/D_):.2f}' for x in (100, 300, 900)))
Kc = R @ R.T
tau2 = (Kc[~np.eye(k, dtype=bool)] ** 2).mean()
print(f'  tau2 = mean squared off-diagonal cosine {tau2:.4f}: participation-ratio form gives D = (1-tau2)/tau2 = {(1-tau2)/tau2:.0f}')

# surrogate fitted on sfit random rows (same grid and objective as E1 in ceiling_estimators.py)
ALPHAS = np.round(np.arange(0, 3.01, 0.15), 2)
FS = np.array([0, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9])
COMBOS = [(al, f) for al in ALPHAS for f in FS]
Ls_, Lh_ = np.load(os.path.join(SCR, f'lib_N{N}_n{sfit}.npy')), np.load(os.path.join(SCR, f'lib_N{N}_n{sfit//2}.npy'))
rng = np.random.RandomState(4)
fits = []
for rep in range(5):
    idx = np.sort(rng.choice(k, sfit, replace=False))
    Fs = energy_curve(R[idx])
    Fh = np.mean([energy_curve(R[idx][np.sort(rng.choice(sfit, sfit // 2, replace=False))]) for _ in range(10)], 0)
    sse = ((Ls_[:, :sfit - 1] - Fs[:sfit - 1]) ** 2).sum(1) + ((Lh_[:, :sfit // 2 - 1] - Fh[:sfit // 2 - 1]) ** 2).sum(1)
    fits.append(COMBOS[int(np.argmin(sse))])
print(f'  E1 fits on {sfit} rows (5 draws): {fits}')
al, f = fits[0]
p = np.arange(1, N + 1) ** (-al); p /= p.sum(); lmb = (1 - f) * p + f / N
print(f'  surrogate (alpha {al}, f {f}): population rank90 = {rank_at(np.cumsum(np.sort(lmb)[::-1]), 0.9)}')
sim = []
kmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
for kk in [x for x in (25, 50, 100, 200, 300, 600, 900, 1500, 3000, 6000, 10000) if x <= kmax]:
    X = np.random.RandomState(kk).standard_normal((kk, N)) * np.sqrt(lmb)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = rank_at(energy_curve(X), 0.9)
    sim.append((kk, r))
    if time.time() - t0 > 200: break
print('  surrogate rank90(k\'):', ' '.join(f'{a}:{b}' for a, b in sim))
s_k = np.array([x[0] for x in sim]); s_r = np.array([x[1] for x in sim])
print('  surrogate local exponents:', ' '.join(f'{s_k[i]}-{s_k[i+1]}:{np.log(s_r[i+1]/s_r[i])/np.log(s_k[i+1]/s_k[i]):.2f}' for i in range(len(sim) - 1)))
print(f'total {time.time()-t0:.0f}s')
