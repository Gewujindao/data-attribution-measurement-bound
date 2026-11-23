"""Leave-one-out study: readings of one model x deletion spec x row subset, plus the leave-one-out (LOO) decomposition.
  python loo_analyze.py --dir <cache/loo> --model b --spec a10|a01|arch [--rows all|A|B] --out <results/loo> [--arch_dir <cache/arch>]
spec 'arch' = the 1% truth of the architecture study (<arch_dir>/<model>_truth.npz).  rows A = 0..99, B = 100..199 (rare0 models).
Measured (subset retrains), per B in {0.05k, 0.1k, 0.2k} of the row subset, SVD oracle W from that subset only:
  kappa, beta_W, beta_perp (per-row slopes of a on q_W, q_perp), overall slope beta, rho, L_P, and the exact formula
  kappa_f = (beta_W - beta_perp) * rho*sqrt(1-rho^2) * sd(p) / (sd(a) * sqrt(1-L_P^2))   (Proposition 1 in PROOFS.md).
LOO decomposition (all N single deletions; g = -LOO response, so that a(S) ~ -sum_{i in S} g_i):
  beta_hat_W/perp = <u,g>/|u|^2, <v,g>/|v|^2 per row (self part of the data model, no pair terms);
  kappa_hat = same formula with the LOO betas and the measured noise factor;
  pooled multipliers m_i = sum_j g_ji y_ji / sum_j y_ji^2; R2 of g - y explained by (m_i - 1) y_ji vs by a per-row scalar;
  betas of the test-independent multiplier model g = M y; Spearman(m_i, mem_i) on the top 30% columns by energy;
  pi-weighted mean multiplier and memorisation on the kept / discarded parts.
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import rankdata, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--model', required=True); ap.add_argument('--spec', required=True)
ap.add_argument('--rows', default='all'); ap.add_argument('--out', required=True); ap.add_argument('--arch_dir', default=None)
a = ap.parse_args()
tag = os.path.join(a.dir, a.model)
Y = np.concatenate([np.load(f) for f in sorted(glob.glob(f'{tag}_Y_*.npy'))])
if a.spec == 'arch':
    tr = np.load(os.path.join(a.arch_dir, f'{a.model}_truth.npz')); masks, d = tr['m01'], tr['d01']
else:
    masks = np.load(f'{tag}_{a.spec}_masks.npy')
    d = np.concatenate([np.load(f) for f in sorted(glob.glob(f'{tag}_{a.spec}_[0-9][0-9][0-9].npy'))])
rows = {'all': np.arange(Y.shape[0]), 'A': np.arange(0, 100), 'B': np.arange(100, 200)}[a.rows]
Y, d = Y[rows], d[:, rows]
k, N = Y.shape
D1 = 1 - masks
loo_files = sorted(glob.glob(f'{tag}_loo_*.npz'))
have_loo = len(loo_files) == int(np.ceil(N / 50))
if have_loo:
    L = [np.load(f) for f in loo_files]
    G = -np.concatenate([x['d'] for x in L])[:, rows].T          # k x N
    mem = np.concatenate([x['mem'] for x in L])
LOG = []
def say(s):
    print(s, flush=True); LOG.append(s)
def cc(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0) + 1e-300)
def slope(x, yv):
    x = x - x.mean(0); yv = yv - yv.mean(0)
    return (x * yv).sum(0) / (x * x).sum(0)
rk = lambda A: rankdata(A, axis=0)

mu = Y.mean(1, keepdims=True); Yc = Y - mu
nrm = np.linalg.norm(Yc, axis=1); R = Yc / nrm[:, None]
U, S, Vt = np.linalg.svd(R, full_matrices=False)
P = -D1 @ Y.T
Lp, Ls = cc(P, d), cc(rk(P), rk(d))
beta_all = slope(P, d)
F = np.cumsum(S ** 2) / k
res = dict(model=a.model, spec=a.spec, rows=a.rows, k=k, N=N, M=masks.shape[0], n_del=int(D1[0].sum()),
           Lp=float(Lp.mean()), Ls=float(Ls.mean()), slope=float(np.median(beta_all)))
say(f'[{a.model} {a.spec} rows={a.rows}] k={k} N={N} M={masks.shape[0]} deleted={int(D1[0].sum())}  L*_P {Lp.mean():.3f}  L*_S {Ls.mean():.3f}  '
    f'slope(median) {np.median(beta_all):.2f}')
if have_loo:
    g_dev = G - Y
    mhat = (G * Y).sum(0) / (Y * Y).sum(0)
    colE = (Y * Y).sum(0)
    top = colE >= np.quantile(colE, 0.7)
    r2M = 1 - np.sum((G - mhat[None] * Y) ** 2) / np.sum(g_dev ** 2)
    bj = (G * Y).sum(1) / (Y * Y).sum(1)
    r2row = 1 - np.sum((G - bj[:, None] * Y) ** 2) / np.sum(g_dev ** 2)
    wfrac = np.sum(colE[mhat < 1]) / colE.sum()
    sp = spearmanr(mhat[top], mem[top])[0]
    psi = lambda x: np.where(np.abs(x) < 1e-8, 1.0, x * np.exp(np.clip(x, -50, 50)) / np.expm1(np.clip(x, -50, 50)))
    ratio_psi = float(np.median(mhat[top] / psi(np.clip(mem[top], 0, None))))
    lin_lds = cc(P, -D1 @ G.T)                                   # how well the LOO linear model tracks the linear prediction
    res['loo'] = dict(r2M=float(r2M), r2row=float(r2row), frac_energy_m_below1=float(wfrac), spearman_m_mem_top30=float(sp),
                      median_m_top30=float(np.median(mhat[top])), median_mem_top30=float(np.median(mem[top])),
                      median_m_over_psi_mem=ratio_psi, beta_loo_overall=float(np.median((Yc * G).sum(1) / (Yc * Yc).sum(1))),
                      corr_p_looLinear=float(lin_lds.mean()), lds_looLinear=float(cc(-D1 @ G.T, d).mean()))
    say(f'  LOO: R2(g-y | test-independent m) {r2M:.3f}  vs per-row scalar {r2row:.3f};  energy share of columns with m<1 {wfrac:.3f};  '
        f'Spearman(m, mem) top30% {sp:+.2f};  median m {np.median(mhat[top]):.2f}, mem {np.median(mem[top]):.2f}, m/Psi(mem) {ratio_psi:.2f}')
    say(f'      overall LOO slope (median) {res["loo"]["beta_loo_overall"]:.2f};  corr(p, LOO-linear) {lin_lds.mean():.3f};  '
        f'Pearson(LOO-linear, actual) {res["loo"]["lds_looLinear"]:.3f}')
say('   B  |  kappa  kappa_f | beta_W beta_perp beta | rho   | LOO: bW_hat bP_hat kappa_hat | M-model: bW bP | E_piW[m] E_piP[m] | E_piW[mem] E_piP[mem] | sign agree(row) kappa vs kappa_hat')
res['tiers'] = []
for frac in (0.05, 0.1, 0.2):
    B = max(1, int(round(frac * k)))
    Uw = (U[:, :B] * S[:B]) @ Vt[:B] * nrm[:, None]                    # P_W y_c, original scale (k x N)
    Vp = Yc - Uw                                                       # P_perp y_c
    qW, qP = -D1 @ Uw.T, -D1 @ Vp.T                                    # P = qW + qP + const
    rho = np.linalg.norm(Uw, axis=1) / nrm
    rqa, rqp = cc(qW, d), cc(qW, P)
    kap = (rqa - rqp * Lp) / np.sqrt((1 - rqp ** 2) * (1 - Lp ** 2))
    bW, bP = slope(qW, d), slope(qP, d)
    noise = P.std(0) / (d.std(0) * np.sqrt(1 - Lp ** 2))
    kf = (bW - bP) * rho * np.sqrt(1 - rho ** 2) * noise
    t = dict(B=B, kappa=float(kap.mean()), kappa_f=float(kf.mean()), betaW=float(np.median(bW)), betaP=float(np.median(bP)),
             beta=float(np.median(beta_all)), rho=float(rho.mean()), F=float(F[B - 1]), contrast=float(np.median((bP - bW) / beta_all)))
    line = f'  {B:3d} | {kap.mean():+.3f}  {kf.mean():+.3f} | {np.median(bW):.2f}   {np.median(bP):.2f}     {np.median(beta_all):.2f} | {rho.mean():.3f} |'
    if have_loo:
        hW = (Uw * G).sum(1) / (Uw * Uw).sum(1); hP = (Vp * G).sum(1) / (Vp * Vp).sum(1)
        kh = (hW - hP) * rho * np.sqrt(1 - rho ** 2) * noise
        GM = mhat[None] * Y
        mW = (Uw * GM).sum(1) / (Uw * Uw).sum(1); mP = (Vp * GM).sum(1) / (Vp * Vp).sum(1)
        piW = Uw * Yc / (Uw * Uw).sum(1, keepdims=True); piP = Vp * Yc / (Vp * Vp).sum(1, keepdims=True)
        eWm, ePm = (piW * mhat).sum(1), (piP * mhat).sum(1)
        eWz, ePz = (piW * mem).sum(1), (piP * mem).sum(1)
        agree = np.mean(np.sign(kap) == np.sign(kh))
        t.update(betaW_hat=float(np.median(hW)), betaP_hat=float(np.median(hP)), kappa_hat=float(kh.mean()),
                 betaW_M=float(np.median(mW)), betaP_M=float(np.median(mP)), EW_m=float(np.median(eWm)), EP_m=float(np.median(ePm)),
                 EW_mem=float(np.median(eWz)), EP_mem=float(np.median(ePz)), row_sign_agree=float(agree),
                 corr_kappa_khat_rows=float(np.corrcoef(kap, kh)[0, 1]))
        line += (f' {np.median(hW):.2f}   {np.median(hP):.2f}   {kh.mean():+.3f}   | {np.median(mW):.2f} {np.median(mP):.2f} | '
                 f'{np.median(eWm):.2f}  {np.median(ePm):.2f} | {np.median(eWz):.2f}  {np.median(ePz):.2f} | {agree:.0%} (row corr {t["corr_kappa_khat_rows"]:+.2f})')
    res['tiers'].append(t); say(line)
res['kappa_true'] = float(np.mean([t['kappa'] for t in res['tiers']]))
if have_loo:
    res['kappa_hat_mean'] = float(np.mean([t['kappa_hat'] for t in res['tiers']]))
say(f'  kappa (mean of three tiers) {res["kappa_true"]:+.3f}' + (f'   kappa_hat (LOO self part) {res["kappa_hat_mean"]:+.3f}' if have_loo else ''))
os.makedirs(a.out, exist_ok=True)
nm = f'{a.model}_{a.spec}_{a.rows}'
json.dump(res, open(os.path.join(a.out, f'{nm}.json'), 'w'), indent=1)
open(os.path.join(a.out, f'{nm}.txt'), 'w', encoding='utf-8').write('\n'.join(LOG) + '\n')
