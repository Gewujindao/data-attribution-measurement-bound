"""Ceiling: pre-estimate the ceiling curve (spherical SVD oracle LDS vs B, and rank90) from s << k replays.
Estimators of the k-row energy curve F_k(B): E0 proportional, E1 Gaussian surrogate (power law + floor spectrum, grid fit
on the s-row and s/2-row curves), E2 sub-sample power-law extrapolation of the rank needed for each energy level.
Conversion to LDS: G_cal (leave-one-out projections of the s rows, with their retrain truth) or G_th (L* sqrt(F)).
Usage: python ceiling_estimators.py <point> <k_eval> <s> [draws]      results appended to <scratch>/t2_<point>_<k_eval>_<s>.npz
"""
import os, sys, time
import numpy as np
from scipy.optimize import curve_fit
from ceiling_common import SCR, POINTS, load, cnorm, ranks, corr_cols, oracle_curve, energy_curve, rank_at

t0 = time.time()
name, keval, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
draws = int(sys.argv[4]) if len(sys.argv) > 4 else 20
ALPHAS = np.round(np.arange(0, 3.01, 0.15), 2)
FS = np.array([0, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9])
COMBOS = [(al, f) for al in ALPHAS for f in FS]

Y, tr, meta = load(POINTS[name])
k, N = Y.shape
masks, dall = tr['m01'], tr['d01']
sel = np.arange(k) if keval == k else np.sort(np.random.RandomState(99).choice(k, keval, replace=False))
Ysel, dsel = Y[sel], dall[:, sel]
R = cnorm(Ysel)


def lam(al, f):
    p = np.arange(1, N + 1) ** (-al); p /= p.sum()
    return (1 - f) * p + f / N


def sim_curve(lmb, n, reps, rng):
    sq = np.sqrt(lmb); out = np.zeros(n)
    for _ in range(reps):
        X = rng.standard_normal((n, N)) * sq
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        out += energy_curve(X)
    return out / reps


def library(n):
    fn = os.path.join(SCR, f'lib_N{N}_n{n}.npy')
    if os.path.exists(fn):
        return np.load(fn)
    rng = np.random.RandomState(1000 + n)
    L = np.array([sim_curve(lam(al, f), n, 16 if n <= 60 else 8, rng) for al, f in COMBOS])
    np.save(fn, L)
    print(f'  library N={N} n={n} built {time.time()-t0:.0f}s', flush=True)
    return L


pred_cache_fn = os.path.join(SCR, f'pred_N{N}_k{keval}.npz')
pred_cache = dict(np.load(pred_cache_fn)) if os.path.exists(pred_cache_fn) else {}
def pred_curve(ci):
    key = str(ci)
    if key not in pred_cache:
        pred_cache[key] = sim_curve(lam(*COMBOS[ci]), keval, 3, np.random.RandomState(7 + ci))
        np.savez(pred_cache_fn, **pred_cache)
    return pred_cache[key]


def frac_rank(F, e):
    """continuous b with F(b)=e, F(0)=0, linear between integers."""
    Fz = np.concatenate([[0], F])
    i = np.searchsorted(Fz, e)
    if i >= len(Fz): return float(len(F))
    return i - 1 + (e - Fz[i - 1]) / (Fz[i] - Fz[i - 1])


def curve_from_levels(levels, b):
    b = np.maximum.accumulate(np.minimum(b, keval))
    xs = np.concatenate([[0], b, [keval]]); ys = np.concatenate([[0], levels, [1]])
    return np.interp(np.arange(1, keval + 1), xs, ys)


def calib(idx):
    """leave-one-out projections of the s sampled rows: (mean rho^2, mean LDS, mean rho, mean L*_j rho_j) per dimension b."""
    Rs = R[idx]; m = len(idx)
    rho2 = np.zeros((m, m - 1)); L = np.zeros((m, m - 1))
    for j in range(m):
        vt = np.linalg.svd(np.delete(Rs, j, 0), full_matrices=False)[2]
        c = vt @ Rs[j]
        pred = -np.cumsum(((1 - masks) @ vt.T) * c, axis=1)
        rho2[j] = np.cumsum(c ** 2)
        L[j] = corr_cols(ranks(pred), ranks(dsel[:, idx[j]])[:, None].repeat(m - 1, 1))
    rho = np.sqrt(rho2)
    return rho2.mean(0), L.mean(0), rho.mean(0), (rho * Lstar[idx][:, None]).mean(0)


# ---------- exact reference ----------
Bs = np.arange(1, keval + 1)
ex_fn = os.path.join(SCR, f'exact_{name}_{keval}.npz')
if os.path.exists(ex_fn):
    ex = np.load(ex_fn); LDSk, Fk, Lstar = ex['LDSk'], ex['Fk'], ex['Lstar']
else:
    LDSk, Fk = oracle_curve(R, masks, dsel, Bs)
    Lstar = corr_cols(ranks(-(1 - masks) @ Ysel.T), ranks(dsel))
    np.savez(ex_fn, LDSk=LDSk, Fk=Fk, Lstar=Lstar)
r90 = rank_at(Fk, 0.9)
print(f'[{name} k_eval={keval} s={s}] exact: L* {Lstar.mean():.3f} rank90 {r90}  oracle LDS at 0.1k/0.2k/0.5k '
      f'{LDSk[keval//10-1]:.3f}/{LDSk[keval//5-1]:.3f}/{LDSk[keval//2-1]:.3f}  ({time.time()-t0:.0f}s)', flush=True)

h, q4 = s // 2, max(4, s // 4)
Ls, Lh = library(s), library(h)
out_fn = os.path.join(SCR, f't2_{name}_{keval}_{s}.npz')
res = {kk: list(v) for kk, v in np.load(out_fn, allow_pickle=True).items()} if os.path.exists(out_fn) else {}
done = len(res.get('draw', []))
for dr in range(done, draws):
    if time.time() - t0 > 230:
        print('budget reached; resume later', flush=True); break
    rng = np.random.RandomState(500 + dr)
    idx = np.sort(rng.choice(keval, s, replace=False))
    Rs = R[idx]
    Fs = energy_curve(Rs)
    halves = [np.sort(rng.choice(s, h, replace=False)) for _ in range(10)]
    quart = [np.sort(rng.choice(s, q4, replace=False)) for _ in range(10)]
    Fh = np.mean([energy_curve(Rs[x]) for x in halves], 0)
    Fq = np.mean([energy_curve(Rs[x]) for x in quart], 0)
    # E0
    F0 = np.interp(Bs * s / keval, np.arange(0, s + 1), np.concatenate([[0], Fs]))
    # E1
    sse = ((Ls[:, :s - 1] - Fs[:s - 1]) ** 2).sum(1) + ((Lh[:, :h - 1] - Fh[:h - 1]) ** 2).sum(1)
    ci = int(np.argmin(sse))
    F1 = pred_curve(ci)
    # E2
    levels = np.arange(0.30, 0.951, 0.05)
    lb = []
    for e in levels:
        bb = np.array([frac_rank(Fq, e), frac_rank(Fh, e), frac_rank(Fs, e)])
        gam, a0 = np.polyfit(np.log([q4, h, s]), np.log(np.maximum(bb, 1e-3)), 1)
        lb.append(np.exp(a0) * keval ** gam)
    F2 = curve_from_levels(levels, np.array(lb))
    # E3 (designed on S1 after E0-E2): saturating (Michaelis-Menten) extrapolation
    #     b_e(n) = a_e n / (1 + n / D_e), the exact form of the participation ratio n / (1 + (n-1) tau2)
    ns = sorted(set([max(4, s // 4), s // 3, s // 2, (2 * s) // 3, s]))
    cur = {n: (np.mean([energy_curve(Rs[np.sort(rng.choice(s, n, replace=False))]) for _ in range(10)], 0) if n < s else Fs) for n in ns}
    lev3 = np.arange(0.10, 0.951, 0.05)
    b3 = []
    for e in lev3:
        bb = np.array([frac_rank(cur[n], e) for n in ns])
        try:
            (a3, D3), _ = curve_fit(lambda n, a_, D_: a_ * n / (1 + n / D_), np.array(ns, float), bb, p0=(e, 3 * s), bounds=([0, 1], [1.5, 1e6]))
        except RuntimeError:
            a3, D3 = bb[-1] / s, 1e6
        b3.append(a3 * keval / (1 + keval / D3))
    F3 = curve_from_levels(lev3, np.array(b3))
    # conversion
    Lbar = Lstar[idx].mean()
    x, yv, m1, m1L = calib(idx)
    xs = np.concatenate([[0], x, [1]]); ys = np.maximum.accumulate(np.concatenate([[0], yv, [Lbar]]))
    o = np.argsort(xs); xs, ys = xs[o], np.maximum.accumulate(ys[o])
    Gcal = lambda F: np.interp(F, xs, ys)            # G_cal: interpolated leave-one-out calibration
    Gth = lambda F: Lbar * np.sqrt(F)                # no calibration: LDS = L* sqrt(F)
    # alternative conversion: ratio r = LDS / (mean L*rho) as a function of mean rho, r(1)=1
    rx = np.concatenate([m1, [1.0]]); ry = np.concatenate([yv / m1L, [1.0]])
    o = np.argsort(rx); rx, ry = rx[o], ry[o]
    Grat = lambda F: Lbar * np.sqrt(F) * np.interp(np.sqrt(F), rx, ry)
    row = dict(draw=dr, ci=ci, alpha=COMBOS[ci][0], f=COMBOS[ci][1], Lbar=Lbar, conv_only=np.abs(Gcal(Fk) - LDSk).max(),
               conv_only_th=np.abs(Gth(Fk) - LDSk).max(), conv_only_rat=np.abs(Grat(Fk) - LDSk).max())
    for nm, Fh_ in (('E0', F0), ('E1', F1), ('E2', F2), ('E3', F3)):
        rh = rank_at(Fh_, 0.9)
        row[nm + '_r90'] = rh
        row[nm + '_Ferr'] = np.abs(Fh_ - Fk).max()
        for gn, G in (('cal', Gcal), ('th', Gth), ('rat', Grat)):
            Lh_ = G(Fh_)
            row[f'{nm}_{gn}_max'] = np.abs(Lh_ - LDSk).max()
            row[f'{nm}_{gn}_at10'] = Lh_[keval // 10 - 1] - LDSk[keval // 10 - 1]
            row[f'{nm}_{gn}_at20'] = Lh_[keval // 5 - 1] - LDSk[keval // 5 - 1]
    for kk, v in row.items():
        res.setdefault(kk, []).append(v)
    np.savez(out_fn, **{kk: np.array(v) for kk, v in res.items()})
    print(f'  draw {dr}: E1 (alpha {COMBOS[ci][0]}, f {COMBOS[ci][1]}) r90 {row["E1_r90"]} maxerr {row["E1_cal_max"]:.3f} | '
          f'E0 r90 {row["E0_r90"]} | E2 r90 {row["E2_r90"]} {row["E2_cal_max"]:.3f} | E3 r90 {row["E3_r90"]} {row["E3_rat_max"]:.3f} | conv-only cal/th/rat {row["conv_only"]:.3f}/'
          f'{row["conv_only_th"]:.3f}/{row["conv_only_rat"]:.3f}  ({time.time()-t0:.0f}s)', flush=True)
print(f'total {time.time()-t0:.0f}s', flush=True)
