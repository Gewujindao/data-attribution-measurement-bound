"""Scale ladder: all readings of one ladder point from its exact rows and retrain truth (CPU, numpy).
  python ladder_analyze.py --dir D --point L2 [--out results/ladder_L2] [--draws 20] [--fw_iters 100] [--parts all|exact|e3]
Readings (1% deletion unless noted):
  spectrum of the row-normalised matrix R: singular values, F(B), rank50/80/90, participation ratio, tau2;
  L*_S, L*_P, Spearman offset; SVD-oracle LDS curve; oracle/(rho L*) ratio, kappa, beta_W / beta_perp, identity check
  (also at 5% if present); Frank-Wolfe certificate of the class ceiling for mean rho and mean L*rho;
  two simulated linear-measurement methods (random row probes; randomised subspace iteration) and their distance to the
  oracle; the estimator E3 + G_cal of the estimator test (code identical) and the proportional estimator E0 at s=k/10 and k/5, plus the
  leave-one-out kappa estimate.
"""
import argparse, glob, json, os, time
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--point', required=True)
ap.add_argument('--out', default=None); ap.add_argument('--draws', type=int, default=20)
ap.add_argument('--fw_iters', type=int, default=100); ap.add_argument('--parts', default='all')
ap.add_argument('--budget', type=float, default=1e9)
a = ap.parse_args()
t0 = time.time()
tag = os.path.join(a.dir, a.point)
out_dir = a.out or a.dir
os.makedirs(out_dir, exist_ok=True)
Y = np.concatenate([np.load(f) for f in sorted(glob.glob(f'{tag}_Y_*.npy'))])
tr = np.load(f'{tag}_truth.npz')
masks, d = tr['m01'], tr['d01']
k, N = Y.shape
D1 = 1 - masks
LOG = []
def say(s):
    print(s, flush=True); LOG.append(s)


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
def partial(q, p, a_):
    rqa, rqp, rpa = corr_cols(q, a_[:, None]), corr_cols(q, p[:, None]), corr_cols(p[:, None], a_[:, None])[0]
    return (rqa - rqp * rpa) / np.sqrt(np.clip((1 - rqp ** 2) * (1 - rpa ** 2), 1e-300, None))


R = cnorm(Y)
nrm = np.linalg.norm(Y - Y.mean(1, keepdims=True), axis=1)
P_ex = -D1 @ Y.T
Ls = corr_cols(ranks(P_ex), ranks(d))
Lp = corr_cols(P_ex, d)
U, S, Vt = np.linalg.svd(R, full_matrices=False)
Zc = U * S
Fk = np.cumsum(S ** 2) / k
Fk = np.concatenate([Fk, np.ones(max(0, k - len(Fk)))])
r90 = rank_at(Fk, 0.9)
res = dict(point=a.point, k=k, N=N)
B_of = lambda frac: max(1, int(round(frac * k)))

# ---------------------------------------------------------------- spectrum and exact quantities
G = R @ R.T
tau2 = (np.sum(G ** 2) - k) / (k * (k - 1))
PR = k ** 2 / np.sum(G ** 2)
res.update(rank50=rank_at(Fk, 0.5), rank80=rank_at(Fk, 0.8), rank90=r90, PR=float(PR), tau2=float(tau2),
           F={f: float(Fk[B_of(f) - 1]) for f in (0.05, 0.1, 0.2, 0.3, 0.5)},
           sv_top=[float(x) for x in S[:10]], Ls=float(Ls.mean()), Lp=float(Lp.mean()), Ls_sd=float(Ls.std()),
           offset=float(np.mean(Ls - g_sp(Lp))), slope=float(np.median((P_ex * d).sum(0) / (P_ex * P_ex).sum(0))))
say(f'[{a.point}] k={k} N={N}  L*_S {Ls.mean():.3f} (sd {Ls.std():.3f})  L*_P {Lp.mean():.3f}  offset L*_S-g(L*_P) {res["offset"]:+.3f}  '
    f'actual/pred slope median {res["slope"]:.2f}')
say(f'  spectrum: rank50/80/90 = {res["rank50"]}/{res["rank80"]}/{r90} (rank90/k {r90/k:.2f}); PR {PR:.1f}; tau2 {tau2:.4f}; '
    f'F(0.05k,0.1k,0.2k,0.3k,0.5k) = ' + ' '.join(f'{v:.3f}' for v in res['F'].values()))

rd = ranks(d)
pred = np.zeros((masks.shape[0], k)); LDSk = np.zeros(k); A_ = D1 @ Vt.T; last = 0.0
for B in range(1, k + 1):
    if B <= Zc.shape[1]:
        pred -= np.outer(A_[:, B - 1], Zc[:, B - 1])
        last = corr_cols(ranks(pred), rd).mean()
    LDSk[B - 1] = last
res['oracle'] = {f: float(LDSk[B_of(f) - 1]) for f in (0.05, 0.1, 0.2, 0.3, 0.5)}
say('  SVD oracle LDS at 0.05k/0.1k/0.2k/0.3k/0.5k: ' + ' '.join(f'{v:.3f}' for v in res['oracle'].values()))


def kappa_table(mk, dk, Lsk, Lpk, label):
    rdk = ranks(dk); Pk = -(1 - mk) @ Y.T
    rows = []
    say(f'  [{label}]  B   LDS_S  rhoL*_S ratioS | LDS_P  rhoL*_P ratioP | mean kappa | identity max | beta_W beta_perp sign-agree')
    for frac in (0.05, 0.1, 0.2, 0.3, 0.5):
        B = B_of(frac)
        Yh = Zc[:, :B] @ Vt[:B]
        rho = np.linalg.norm(Yh, axis=1)
        q = -(1 - mk) @ Yh.T
        LS, LP = corr_cols(ranks(q), rdk), corr_cols(q, dk)
        rqa, rqp, rpa = corr_cols(q, dk), corr_cols(q, Pk), corr_cols(Pk, dk)
        kap = (rqa - rqp * rpa) / np.sqrt((1 - rqp ** 2) * (1 - rpa ** 2))
        idm = np.abs(LP - (rqp * Lpk + np.sqrt(1 - rqp ** 2) * np.sqrt(1 - Lpk ** 2) * kap)).max()
        qW = q * nrm[None]; qP = Pk - qW
        cW, cP, cA = qW - qW.mean(0), qP - qP.mean(0), dk - dk.mean(0)
        bW, bP = (cW * cA).sum(0) / (cW * cW).sum(0), (cP * cA).sum(0) / (cP * cP).sum(0)
        agree = np.mean(np.sign(kap) == np.sign(bW - bP))
        rows.append(dict(B=B, LDS_S=float(LS.mean()), ratioS=float(LS.mean() / np.mean(rho * Lsk)),
                         ratioP=float(LP.mean() / np.mean(rho * Lpk)), kappa=float(kap.mean()), idmax=float(idm),
                         betaW=float(np.median(bW)), betaP=float(np.median(bP)), agree=float(agree)))
        say(f'   {B:5d}  {LS.mean():.3f}  {np.mean(rho*Lsk):.3f}  {LS.mean()/np.mean(rho*Lsk):.3f} | {LP.mean():.3f}  {np.mean(rho*Lpk):.3f}  '
            f'{LP.mean()/np.mean(rho*Lpk):.3f} | {kap.mean():+.3f} | {idm:.1e} | {np.median(bW):.2f} {np.median(bP):.2f} {agree:.0%}')
    return rows


if a.parts in ('all', 'exact'):
    res['kappa01'] = kappa_table(masks, d, Ls, Lp, '1%')
    res['kappa_true'] = float(np.mean([r['kappa'] for r in res['kappa01'][:3]]))
    say(f'  kappa_true (mean over B=0.05k,0.1k,0.2k) = {res["kappa_true"]:+.3f}')
    if 'm05' in tr.files:
        P5 = -(1 - tr['m05']) @ Y.T
        res['kappa05'] = kappa_table(tr['m05'], tr['d05'], corr_cols(ranks(P5), ranks(tr['d05'])), corr_cols(P5, tr['d05']), '5%')

    # ---------------- Frank-Wolfe certificate (same algorithm as ceiling_cosine_to_lds.fw_ceiling)
    def fw_ceiling(Z, B, w, iters):
        kk, r = Z.shape
        P = np.zeros((r, r)); P[np.arange(B), np.arange(B)] = 1.0
        best_lo, best_up = -1, np.inf
        for t in range(iters):
            x = np.clip(np.sum((Z @ P) * Z, 1), 1e-12, None)
            f = np.mean(w * np.sqrt(x))
            Gm = (Z * (w / (2 * np.sqrt(x)))[:, None]).T @ Z / kk
            ev, V = np.linalg.eigh(Gm); Sx = V[:, -B:]; Sm = Sx @ Sx.T
            best_up = min(best_up, f + np.sum(Gm * (Sm - P)))
            best_lo = max(best_lo, np.mean(w * np.sqrt(np.clip(np.sum((Z @ Sx) ** 2, 1), 0, None))))
            P = P + 2 / (t + 2) * (Sm - P)
        return best_lo, best_up
    res['fw'] = []
    for frac in (0.1, 0.2):
        B = B_of(frac)
        rho_svd = np.linalg.norm(Zc[:, :B], axis=1)
        lo, up = fw_ceiling(Zc, B, np.ones(k), a.fw_iters)
        lo2, up2 = fw_ceiling(Zc, B, Ls, a.fw_iters)
        res['fw'].append(dict(B=B, rho_svd=float(rho_svd.mean()), rho_up=float(up), Lrho_svd=float(np.mean(Ls * rho_svd)), Lrho_up=float(up2)))
        say(f'  class ceiling B={B}: mean rho SVD {rho_svd.mean():.4f} certified-up {up:.4f} (gap {up-rho_svd.mean():.4f}) | '
            f'mean L*rho SVD {np.mean(Ls*rho_svd):.4f} up {up2:.4f} (gap {up2-np.mean(Ls*rho_svd):.4f})  ({time.time()-t0:.0f}s)')

    # ---------------- simulated linear-measurement methods (answers computed from the exact Y)
    def lds_of_subspace(W):                         # W: (N, m) orthonormal, inside 1-perp
        Yh = (R @ W) @ W.T
        rho2 = np.sum(Yh ** 2, 1)
        return float(rho2.mean()), float(corr_cols(ranks(-D1 @ Yh.T), rd).mean())
    def orth_perp1(M):                              # rows of M span; project out the ones direction; orthonormal basis (N, m)
        M = M - M.mean(1, keepdims=True)
        return np.linalg.qr(M.T)[0]
    res['methods'] = []
    for frac in (0.1, 0.2):
        B = B_of(frac)
        vals = {'random': [], 'subspace_iter': []}
        for sd in range(3):
            rg = np.random.RandomState(900 + sd)
            Z0 = rg.standard_normal((k, B))
            vals['random'].append(lds_of_subspace(orth_perp1((Z0 * nrm[:, None]).T @ R)))   # B row replays z^T Y
            b3 = max(1, B // 3)
            A1 = (rg.standard_normal((k, b3)) * nrm[:, None]).T @ R                         # b3 row replays
            Q = np.linalg.qr(R @ A1.T)[0]                                                   # b3 column replays Y u
            A2 = (Q * nrm[:, None]).T @ R                                                   # b3 row replays
            vals['subspace_iter'].append(lds_of_subspace(orth_perp1(np.vstack([A1, A2]))))
        for nm, v in vals.items():
            r2, l = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
            res['methods'].append(dict(method=nm, B=B, rho2=float(r2), lds=float(l), F=float(Fk[B - 1]), oracle=float(LDSk[B - 1])))
            say(f'  {nm:14s} B={B}: mean rho^2 {r2:.3f} (<= F(B) {Fk[B-1]:.3f})  LDS {l:.3f}  oracle {LDSk[B-1]:.3f}  ratio {l/LDSk[B-1]:.3f}')


# ---------------------------------------------------------------- the estimator E3 + G_cal (identical to estimator_test_eval.py) and E0
def e3_curve(Rs, rng):
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
    Rs = R[idx]; m = len(idx)
    rho2 = np.zeros((m, m - 1)); L = np.zeros((m, m - 1)); K = np.zeros((m, m - 1))
    for j in range(m):
        vt = np.linalg.svd(np.delete(Rs, j, 0), full_matrices=False)[2]
        c = vt @ Rs[j]
        pr = -np.cumsum((D1 @ vt.T) * c, axis=1)
        rho2[j] = np.cumsum(c ** 2)
        L[j] = corr_cols(ranks(pr), ranks(d[:, idx[j]])[:, None].repeat(m - 1, 1))
        K[j] = partial(pr, P_ex[:, idx[j]], d[:, idx[j]])
    x, y = rho2.mean(0), L.mean(0)
    xs = np.concatenate([[0], x, [1]]); ys = np.maximum.accumulate(np.concatenate([[0], y, [Ls[idx].mean()]]))
    o = np.argsort(xs); xs, ys = xs[o], np.maximum.accumulate(ys[o])
    return (lambda Fv: np.interp(Fv, xs, ys)), K.mean()


def e0_curve(Rs):
    s_ = Rs.shape[0]
    Fs = np.concatenate([[0], energy_curve(Rs)])
    return np.interp(np.arange(1, k + 1) * s_ / k, np.arange(s_ + 1), Fs)


if a.parts in ('all', 'e3'):
    res['e3'] = {}
    for s in (max(2, round(k / 10)), round(k / 5)):
        rows = []
        for dr in range(a.draws):
            if time.time() - t0 > a.budget:
                say('  budget reached in E3'); break
            rng = np.random.RandomState(500 + dr)
            idx = np.sort(rng.choice(k, s, replace=False))
            F3 = e3_curve(R[idx], rng)
            Gf, kl = g_cal(idx)
            F0 = e0_curve(R[idx])
            r3, r0 = rank_at(F3, 0.9), rank_at(F0, 0.9)
            rows.append(dict(err=np.abs(Gf(F3) - LDSk).max(), conv=np.abs(Gf(Fk) - LDSk).max(), r90err=abs(r3 / r90 - 1),
                             r90bias=r3 / r90 - 1, e0err=np.abs(Gf(F0) - LDSk).max(), e0r90=r0 / r90 - 1, kloo=kl))
        med = {kk: float(np.median([r[kk] for r in rows])) for kk in rows[0]}
        q90 = {kk: float(np.quantile([r[kk] for r in rows], 0.9)) for kk in ('err', 'r90err')}
        res['e3'][s] = dict(med=med, q90=q90, n=len(rows))
        say(f'  E3 s={s} ({len(rows)} draws): LDS err {med["err"]:.3f}[{q90["err"]:.3f}]  r90 err {med["r90err"]:.2f}[{q90["r90err"]:.2f}] '
            f'(bias {med["r90bias"]:+.2f})  conv-only {med["conv"]:.3f}  | E0: LDS err {med["e0err"]:.3f}  r90 bias {med["e0r90"]:+.2f} '
            f'| kappa_LOO {med["kloo"]:+.3f}  ({time.time()-t0:.0f}s)')

json.dump(res, open(os.path.join(out_dir, f'{a.point}_readings.json'), 'w'), indent=1, default=float)
open(os.path.join(out_dir, f'{a.point}_readings.txt'), 'w', encoding='utf-8').write('\n'.join(LOG) + '\n')
print(f'total {time.time()-t0:.0f}s')
