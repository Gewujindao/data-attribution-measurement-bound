"""Ceiling scripts: shared helpers (loading a scale point, LDS, oracle curves)."""
import glob, os
import numpy as np
from scipy.stats import rankdata

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'ceiling')
POINTS = {'S1': 'digits_h32_N1000_k300', 'S2': 'digits_h128_N1000_k300',
          'S3': 'synth_h32_N1000_k900', 'S4': 'synth_h32_N4000_k300', 'S4b': 'synth_h32_N4000_k300_e5'}


def load(tag):
    fs = sorted(glob.glob(os.path.join(SCR, f'{tag}_Y_*.npy')))
    Y = np.concatenate([np.load(f) for f in fs])
    tr = np.load(os.path.join(SCR, f'{tag}_truth.npz'))
    meta = np.load(os.path.join(SCR, f'{tag}_meta.npz'))
    return Y, tr, meta


def cnorm(Y):
    Z = Y - Y.mean(1, keepdims=True)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def ranks(A):
    r = rankdata(A, axis=0)
    return r - r.mean(0)


def corr_cols(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0) + 1e-300)


def preds(masks, Yh):
    return -(1 - masks) @ Yh.T          # (M, rows)


def lds_s(masks, d, Yh):
    """Spearman LDS per row; d is (M, rows) actual change for those rows."""
    return corr_cols(ranks(preds(masks, Yh)), ranks(d))


def lds_p(masks, d, Yh):
    return corr_cols(preds(masks, Yh), d)


def oracle_curve(R, masks, d, Bs):
    """Spherical SVD oracle: mean Spearman LDS and mean rho^2 for each B in Bs (sorted). Incremental in B."""
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    C = R @ Vt.T                       # k x r coefficients
    A = (1 - masks) @ Vt.T             # M x r
    rd = ranks(d)
    out_l, out_f = [], []
    pred = np.zeros((masks.shape[0], R.shape[0]))
    cur = 0
    e = np.cumsum(S ** 2) / R.shape[0]
    for B in Bs:
        while cur < B:
            pred -= np.outer(A[:, cur], C[:, cur]); cur += 1
        out_l.append(corr_cols(ranks(pred), rd).mean() if B > 0 else 0.0)
        out_f.append(e[B - 1] if B > 0 else 0.0)
    return np.array(out_l), np.array(out_f)


def energy_curve(R):
    """F(b), b=1..n (n = rows), from the Gram eigenvalues."""
    n = R.shape[0]
    if n <= R.shape[1]:
        ev = np.linalg.eigvalsh(R @ R.T)[::-1]
    else:
        ev = np.linalg.eigvalsh(R.T @ R)[::-1]
    ev = np.clip(ev, 0, None)
    return np.cumsum(ev[:n]) / n


def rank_at(F, level):
    """smallest b (1-based) with F(b) >= level."""
    i = np.searchsorted(F, level)
    return int(i + 1) if i < len(F) else len(F) + 1


def g_sp(r):
    return 6 / np.pi * np.arcsin(np.clip(r, -1, 1) / 2)
