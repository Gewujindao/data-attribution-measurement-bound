"""Estimator test: test of the estimator E3 on new settings and of the sign condition kappa.
  python estimator_test_eval.py <tag> exact          oracle curve, rank90, L*, and the kappa / Spearman-link table on the SVD oracle
  python estimator_test_eval.py <tag> <s> [draws]    the estimator E3 + G_cal for budget s (20 draws, resumable), plus the leave-one-out kappa estimate
Results go to ../cache/estimator_test/eval_<tag>_*.npz; printed lines are the record.
"""
import glob, os, sys, time
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import rankdata

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'estimator_test')
t0 = time.time()
tag = sys.argv[1]
Y = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(SCR, f'{tag}_Y_*.npy')))])
tr = np.load(os.path.join(SCR, f'{tag}_truth.npz'))
masks, d = tr['m01'], tr['d01']
k, N = Y.shape
D1 = 1 - masks


def cnorm(A):
    Z = A - A.mean(1, keepdims=True)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)
def ranks(A):
    r = rankdata(A, axis=0); return r - r.mean(0)
def corr_cols(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0) + 1e-300)
def energy_curve(Rm):
    n = Rm.shape[0]
    ev = np.linalg.eigvalsh(Rm @ Rm.T if n <= Rm.shape[1] else Rm.T @ Rm)[::-1]
    return np.cumsum(np.clip(ev, 0, None)[:n]) / n
def rank_at(F, level):
    i = np.searchsorted(F, level); return int(i + 1) if i < len(F) else len(F) + 1
def g_sp(r):
    return 6 / np.pi * np.arcsin(np.clip(r, -1, 1) / 2)
def partial(q, p, a):
    """partial correlation of columns of q with a given p (a, p: (M,), q: (M, m))."""
    rqa, rqp, rpa = corr_cols(q, a[:, None]), corr_cols(q, p[:, None]), corr_cols(p[:, None], a[:, None])[0]
    return (rqa - rqp * rpa) / np.sqrt(np.clip((1 - rqp ** 2) * (1 - rpa ** 2), 1e-300, None))


R = cnorm(Y)
P_ex = -D1 @ Y.T                                   # exact linear predictions (M, k)
Ls = corr_cols(ranks(P_ex), ranks(d))
ex_fn = os.path.join(SCR, f'eval_{tag}_exact.npz')

if sys.argv[2] == 'exact':
    Lp = corr_cols(P_ex, d)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    Zc = U * S
    A = D1 @ Vt.T
    rd = ranks(d)
    pred = np.zeros((masks.shape[0], k)); LDSk = np.zeros(k)
    for B in range(1, k + 1):
        if B <= Zc.shape[1]:            # beyond rank(R) (k > N) the projection is the exact row; the curve stays flat
            pred -= np.outer(A[:, B - 1], Zc[:, B - 1])
            last = corr_cols(ranks(pred), rd).mean()
        LDSk[B - 1] = last
    Fk = np.cumsum(S ** 2) / k
    Fk = np.concatenate([Fk, np.ones(max(0, k - len(Fk)))])
    r90 = rank_at(Fk, 0.9)
    print(f'[{tag}] k={k} N={N} rank90={r90} (rank90/k {r90/k:.2f})  L*_S {Ls.mean():.3f} (sd {Ls.std():.3f})  L*_P {Lp.mean():.3f}  '
          f'L*_S-g(L*_P) {np.mean(Ls - g_sp(Lp)):+.3f}  oracle LDS at 0.1k/0.2k {LDSk[max(1,round(k/10))-1]:.3f}/{LDSk[max(1,round(k/5))-1]:.3f}  ({time.time()-t0:.0f}s)')
    kap_rows = []
    print('   B   LDS_S  rhoL*_S ratioS | LDS_P  rhoL*_P ratioP | mean kappa | g(LDS_P)-LDS_S')
    for frac in (0.05, 0.1, 0.2):
        B = max(1, int(round(frac * k)))
        Yh = Zc[:, :B] @ Vt[:B]
        rho = np.linalg.norm(Yh, axis=1)
        q = -D1 @ Yh.T
        LS, LP = corr_cols(ranks(q), rd), corr_cols(q, d)
        rqa, rqp, rpa = corr_cols(q, d), corr_cols(q, P_ex), corr_cols(P_ex, d)
        kap = (rqa - rqp * rpa) / np.sqrt((1 - rqp ** 2) * (1 - rpa ** 2))
        kap_rows.append(kap.mean())
        # amplification along the kept (W) and dropped parts of each row: slopes of a on q_W and on q_perp
        nrm = np.linalg.norm(Y - Y.mean(1, keepdims=True), axis=1)
        qW = q * nrm[None]; qP = P_ex - qW
        cW, cP, cA = qW - qW.mean(0), qP - qP.mean(0), d - d.mean(0)
        bW, bP = (cW * cA).sum(0) / (cW * cW).sum(0), (cP * cA).sum(0) / (cP * cP).sum(0)
        agree = np.mean(np.sign(kap) == np.sign(bW - bP))
        print(f'  {B:4d}  {LS.mean():.3f}  {np.mean(rho*Ls):.3f}  {LS.mean()/np.mean(rho*Ls):.3f} | {LP.mean():.3f}  {np.mean(rho*Lp):.3f}  '
              f'{LP.mean()/np.mean(rho*Lp):.3f} | {kap.mean():+.3f} | {np.mean(g_sp(LP))-LS.mean():+.3f} | '
              f'median beta_W {np.median(bW):.2f} beta_perp {np.median(bP):.2f}; sign(kappa)=sign(bW-bP) for {agree:.0%} of rows')
    print(f'  kappa_true (mean over B=0.05k,0.1k,0.2k) = {np.mean(kap_rows):+.3f}')
    np.savez(ex_fn, LDSk=LDSk, Fk=Fk, r90=r90, Ls=Ls, Lp=Lp, kap=np.array(kap_rows))
    sys.exit()

s = int(sys.argv[2]); draws = int(sys.argv[3]) if len(sys.argv) > 3 else 20
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 200
ex = np.load(ex_fn); LDSk, Fk, r90 = ex['LDSk'], ex['Fk'], int(ex['r90'])


def e3_curve(Rs, rng):
    """the estimator E3: saturating-hyperbola extrapolation of the rank needed for each energy level."""
    s_ = Rs.shape[0]
    ns = sorted(set([max(4, s_ // 4), s_ // 3, s_ // 2, (2 * s_) // 3, s_]))
    cur = {}
    for n in ns:
        if n < s_:
            cur[n] = np.mean([energy_curve(Rs[np.sort(rng.choice(s_, n, replace=False))]) for _ in range(10)], 0)
        else:
            cur[n] = energy_curve(Rs)
    lev = np.arange(0.10, 0.951, 0.05)
    bk = []
    for e in lev:
        bb = []
        for n in ns:
            Fz = np.concatenate([[0], cur[n]]); i = np.searchsorted(Fz, e)
            bb.append(float(n) if i >= len(Fz) else i - 1 + (e - Fz[i - 1]) / (Fz[i] - Fz[i - 1]))
        bb = np.array(bb)
        try:
            (a3, D3), _ = curve_fit(lambda n, a_, D_: a_ * n / (1 + n / D_), np.array(ns, float), bb, p0=(e, 3 * s_),
                                    bounds=([0, 1], [1.5, 1e6]))
        except RuntimeError:
            a3, D3 = bb[-1] / s_, 1e6
        bk.append(a3 * k / (1 + k / D3))
    bk = np.maximum.accumulate(np.minimum(np.array(bk), k))
    return np.interp(np.arange(1, k + 1), np.concatenate([[0], bk, [k]]), np.concatenate([[0], lev, [1]]))


def g_cal(idx):
    """G_cal of the estimator E3 (leave-one-out projections) and the leave-one-out kappa estimate."""
    Rs = R[idx]; m = len(idx)
    rho2 = np.zeros((m, m - 1)); L = np.zeros((m, m - 1)); K = np.zeros((m, m - 1))
    for j in range(m):
        vt = np.linalg.svd(np.delete(Rs, j, 0), full_matrices=False)[2]
        c = vt @ Rs[j]
        pred = -np.cumsum((D1 @ vt.T) * c, axis=1)
        rho2[j] = np.cumsum(c ** 2)
        L[j] = corr_cols(ranks(pred), ranks(d[:, idx[j]])[:, None].repeat(m - 1, 1))
        K[j] = partial(pred, P_ex[:, idx[j]], d[:, idx[j]])
    x, y = rho2.mean(0), L.mean(0)
    xs = np.concatenate([[0], x, [1]]); ys = np.maximum.accumulate(np.concatenate([[0], y, [Ls[idx].mean()]]))
    o = np.argsort(xs); xs, ys = xs[o], np.maximum.accumulate(ys[o])
    return (lambda F: np.interp(F, xs, ys)), K.mean()


out_fn = os.path.join(SCR, f'eval_{tag}_{s}.npz')
res = {kk: list(v) for kk, v in np.load(out_fn).items()} if os.path.exists(out_fn) else {}
for dr in range(len(res.get('draw', [])), draws):
    if time.time() - t0 > budget:
        print('budget reached; resume later', flush=True); break
    rng = np.random.RandomState(500 + dr)
    idx = np.sort(rng.choice(k, s, replace=False))
    F3 = e3_curve(R[idx], rng)
    G, kl = g_cal(idx)
    r3 = rank_at(F3, 0.9)
    row = dict(draw=dr, err=np.abs(G(F3) - LDSk).max(), conv=np.abs(G(Fk) - LDSk).max(), r90hat=r3, r90err=abs(r3 / r90 - 1),
               kloo=kl, Ferr=np.abs(F3 - Fk).max())
    for kk, v in row.items():
        res.setdefault(kk, []).append(v)
    np.savez(out_fn, **{kk: np.array(v) for kk, v in res.items()})
    print(f'  [{tag} s={s}] draw {dr}: LDS err {row["err"]:.3f} (conv-only {row["conv"]:.3f})  r90 {r3} vs {r90} ({row["r90err"]:.2f})  '
          f'kappa_LOO {kl:+.3f}  ({time.time()-t0:.0f}s)', flush=True)
print(f'total {time.time()-t0:.0f}s')
