"""Ceiling: from centred cosine rho to LDS.
(a) exact sample identity  LDS_P = r_qp L*_P + sqrt(1-r_qp^2) sqrt(1-L*_P^2) kappa  (kappa = partial corr of q and a given p)
(b) Spearman link g(r) = 6/pi asin(r/2)
(c) class ceiling for mean rho (and mean L*rho): Frank-Wolfe on the Fantope gives a certified upper bound and feasible subspaces.
Usage: python ceiling_cosine_to_lds.py S1 [S2 ...]
"""
import sys
import numpy as np
from ceiling_common import POINTS, load, cnorm, preds, lds_s, lds_p, corr_cols, g_sp


def fw_ceiling(Z, B, w, iters=300):
    """max over Fantope {0<=P<=I, tr P=B} of mean_j w_j sqrt(z_j' P z_j). Returns (best feasible projection value,
    its basis, certified upper bound)."""
    k, r = Z.shape
    P = np.zeros((r, r)); P[np.arange(B), np.arange(B)] = 1.0   # start at SVD oracle (Z columns are SVD directions)
    best_lo, best_V, best_up = -1, None, np.inf
    for t in range(iters):
        x = np.clip(np.sum((Z @ P) * Z, 1), 1e-12, None)
        f = np.mean(w * np.sqrt(x))
        G = (Z * (w / (2 * np.sqrt(x)))[:, None]).T @ Z / k
        ev, V = np.linalg.eigh(G)
        S = V[:, -B:]
        Sm = S @ S.T
        gap = np.sum(G * (Sm - P))
        best_up = min(best_up, f + gap)
        xs = np.sum((Z @ S) ** 2, 1)
        fs = np.mean(w * np.sqrt(np.clip(xs, 0, None)))
        if fs > best_lo:
            best_lo, best_V = fs, S
        P = P + 2 / (t + 2) * (Sm - P)
    return best_lo, best_V, best_up


opts = [x for x in sys.argv[1:] if x not in POINTS]          # optional: noratio | fw=0.1,0.2 | nofw
fw_arg = [x[3:] for x in opts if x.startswith('fw=')]
for name in [x for x in sys.argv[1:] if x in POINTS]:
    Y, tr, meta = load(POINTS[name])
    k, N = Y.shape
    R = cnorm(Y)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    Z = U * S                                   # coordinates of rows in the SVD basis (unit rows)
    F = np.cumsum(S ** 2) / k
    print(f'===== {name} {POINTS[name]}: P={int(meta["P"])} test acc {float(meta["acc"]):.3f} k={k} N={N} rank90={np.searchsorted(F, 0.9)+1}')
    for key in ('01', '05'):
        masks, d = tr['m' + key], tr['d' + key]
        p = preds(masks, Y)
        LxS, LxP = lds_s(masks, d, Y), lds_p(masks, d, Y)
        print(f'-- {key}% deletion, M={masks.shape[0]}: exact L*_S {LxS.mean():.3f} (sd over test points {LxS.std():.3f})  L*_P {LxP.mean():.3f}'
              f'  mean g(L*_P) {g_sp(LxP).mean():.3f}  slope(actual/pred) median {np.median((p*d).sum(0)/(p*p).sum(0)):.2f}')
        if 'noratio' not in opts: print('   B     LDS_S  rhoL*_S ratioS | LDS_P  rhoL*_P ratioP | kappa  k-term samp-term idmax | gLDS_P-LDS_S')
        for frac in ((0.05, 0.1, 0.2, 0.3, 0.5) if 'noratio' not in opts else ()):
            B = max(1, int(round(frac * k)))
            Yh = Z[:, :B] @ Vt[:B]
            rho = np.linalg.norm(Yh, axis=1)
            q = preds(masks, Yh)
            LS, LP = lds_s(masks, d, Yh), lds_p(masks, d, Yh)
            rqp = corr_cols(q, p)
            kap = (LP - rqp * LxP) / np.sqrt((1 - rqp ** 2) * (1 - LxP ** 2))
            kterm = np.sqrt(1 - rqp ** 2) * np.sqrt(1 - LxP ** 2) * kap
            idm = np.abs(LP - (rqp * LxP + kterm)).max()
            print(f'  {B:4d}  {LS.mean():.3f}  {np.mean(rho*LxS):.3f}  {LS.mean()/np.mean(rho*LxS):.3f} | {LP.mean():.3f}  {np.mean(rho*LxP):.3f}  {LP.mean()/np.mean(rho*LxP):.3f}'
                  f' | {kap.mean():+.3f} {kterm.mean():+.3f} {np.mean((rqp-rho)*LxP):+.3f} {idm:.1e} | {np.mean(g_sp(LP))-LS.mean():+.3f}')
        if key == '01' and 'nofw' not in opts:
            print('   class ceiling for mean rho and mean L*rho (Frank-Wolfe on the Fantope; lo = best feasible B-dim subspace, up = certified):')
            fr = [float(x) for x in fw_arg[0].split(',')] if fw_arg else ((0.1, 0.2, 0.3) if k <= 300 else (0.1, 0.2))
            for frac in fr:
                B = max(1, int(round(frac * k)))
                rho_svd = np.linalg.norm(Z[:, :B], axis=1)
                it = int([x[3:] for x in opts if x.startswith('it=')][0]) if any(x.startswith('it=') for x in opts) else (300 if k <= 300 else 100)
                lo, V, up = fw_ceiling(Z, B, np.ones(k), it)
                lo2, V2, up2 = fw_ceiling(Z, B, LxS, it)
                Yh2 = (Z @ V2) @ (V2.T @ Vt)
                print(f'  B={B}: mean rho  SVD {rho_svd.mean():.4f}  feasible-best {lo:.4f}  certified-up {up:.4f}  sqrt(F) {np.sqrt(F[B-1]):.4f}'
                      f' | mean L*rho  SVD {np.mean(LxS*rho_svd):.4f} feasible-best {lo2:.4f} up {up2:.4f}'
                      f' | measured LDS_S  SVD {lds_s(masks, d, Z[:, :B] @ Vt[:B]).mean():.4f}  L*rho-optimal subspace {lds_s(masks, d, Yh2).mean():.4f}', flush=True)
