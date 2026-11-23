"""Multiplier measurements: per-model quantities for question Q1 of multiplier_analyze.py; resumable, stops after --budget s.
  python multiplier_self_influence.py --dir <cache/multiplier> --models w16,b,... [--budget 240]
(1) <m>_ggn.npz: at theta_T, training margins z, margin Jacobian Gram K0 = J J^T and curvature-weighted Gram
    Kd = diag(sqrt l'') J J^T diag(sqrt l''), l''=sigma(z)sigma(-z) (cross-entropy = log(1+e^-z) in the margin z).
    Terminal leverage for ridge lambda: h = diag(Kd (Kd + lambda I)^-1) (dual form of l''_i dz_i^T (J^T L J + lambda I)^-1 dz_i).
(2) <m>_si_<c0>.npy: exact self-influence SI_i = d z_i(T) / d w_i (adjoint replay with the training point's own margin
    as test function) for the training points in the top 30% of column energy of Y (index saved in <m>_sicols.npy).
(3) <m>_split_<c0>.npz: exact test rows split into the contributions of steps t < T/2 and t >= T/2.
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
from torch.func import vmap, grad
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiplier_points import build

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--models', required=True); ap.add_argument('--budget', type=float, default=240)
a = ap.parse_args()
t0 = time.time()
left = lambda: time.time() - t0 < a.budget


def replay_split(pt, rows):
    a_ = pt.test_grads(rows).detach(); b_ = torch.zeros_like(a_)
    Ye = torch.zeros(len(rows), pt.N, dtype=torch.float64); Yl = torch.zeros_like(Ye)
    half = pt.T // 2
    for s0 in reversed(sorted(pt._ck)):
        s1 = min(s0 + pt._seg, pt.T)
        th, v = pt._ck[s0][0], pt._ck[s0][1]
        ths, ones = [], torch.ones(pt.bs, dtype=pt.dt)
        for t in range(s0, s1):
            ths.append(th)
            v = pt.mom * v + pt._wgrad(th, pt.bidx[t], ones) + pt.wd * th
            th = th - pt.eta[t] * v
        for t in range(s1 - 1, s0 - 1, -1):
            c = b_ - pt.eta[t] * a_
            D, Hc = pt._step_jvp(ths[t - s0], pt.bidx[t], c)
            (Yl if t >= half else Ye)[:, pt.bidx[t]] += D.to(torch.float64) / pt.bs
            a_ = a_ + Hc + pt.wd * c
            b_ = pt.mom * c
    return Ye.numpy(), Yl.numpy()


for nm in a.models.split(','):
    if not left():
        break
    tag = os.path.join(a.dir, nm)
    Y = np.concatenate([np.load(f) for f in sorted(glob.glob(f'{tag}_Y_*.npy'))])
    colE = (Y * Y).sum(0)
    cols = np.where(colE >= np.quantile(colE, 0.7))[0]
    np.save(f'{tag}_sicols.npy', cols)
    need = (not os.path.exists(f'{tag}_ggn.npz') or
            any(not os.path.exists(f'{tag}_si_{c0:04d}.npy') for c0 in range(0, len(cols), 50)) or
            any(not os.path.exists(f'{tag}_split_{c0:04d}.npz') for c0 in range(0, Y.shape[0], 50)))
    if not need:
        continue
    pt = build(nm); pt.fit_reference(seg=pt.T); th = pt._thT
    if not os.path.exists(f'{tag}_ggn.npz'):
        z = pt.margin_vec(th, pt.Xtr, pt.ytr).detach()
        J = vmap(grad(lambda p, x, yy: pt.margin_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))(th, pt.Xtr, pt.ytr).detach()
        K0 = (J @ J.T).numpy()
        l2 = torch.sigmoid(z) * torch.sigmoid(-z)
        J.mul_(torch.sqrt(l2)[:, None])
        Kd = (J @ J.T).numpy(); del J
        np.savez(f'{tag}_ggn.npz', z=z.numpy(), K0=K0, Kd=Kd)
        print(f'[{nm}] ggn ({time.time()-t0:.0f}s)', flush=True)
    Xte, yte = pt.Xte, pt.yte
    for c0 in range(0, len(cols), 50):
        fb = f'{tag}_si_{c0:04d}.npy'
        if os.path.exists(fb) or not left():
            continue
        idx = cols[c0:c0 + 50]
        pt.Xte, pt.yte = pt.Xtr[idx], pt.ytr[idx]
        rows = pt.replay_rows(list(range(len(idx))))
        np.save(fb, rows[np.arange(len(idx)), idx])
        pt.Xte, pt.yte = Xte, yte
        print(f'  [{nm}] si {c0}+ ({time.time()-t0:.0f}s)', flush=True)
    pt.Xte, pt.yte = Xte, yte
    for c0 in range(0, Y.shape[0], 50):
        fb = f'{tag}_split_{c0:04d}.npz'
        if os.path.exists(fb) or not left():
            continue
        rows = list(range(c0, min(c0 + 50, Y.shape[0])))
        Ye, Yl = replay_split(pt, rows)
        err = np.abs(Ye + Yl - Y[rows]).max() / np.abs(Y[rows]).max()
        np.savez(fb, early=Ye, late=Yl)
        print(f'  [{nm}] split {c0}+ max rel |early+late-Y| {err:.1e} ({time.time()-t0:.0f}s)', flush=True)
ok = []
for nm in a.models.split(','):
    tag = os.path.join(a.dir, nm)
    if not os.path.exists(f'{tag}_sicols.npy'):
        continue
    nc = len(np.load(f'{tag}_sicols.npy')); k = sum(np.load(f).shape[0] for f in glob.glob(f'{tag}_Y_*.npy'))
    if (os.path.exists(f'{tag}_ggn.npz') and all(os.path.exists(f'{tag}_si_{c0:04d}.npy') for c0 in range(0, nc, 50))
            and all(os.path.exists(f'{tag}_split_{c0:04d}.npz') for c0 in range(0, k, 50))):
        ok.append(nm)
print(f'self done {len(ok)}/{len(a.models.split(","))}: {",".join(ok)}  total {time.time()-t0:.0f}s', flush=True)
