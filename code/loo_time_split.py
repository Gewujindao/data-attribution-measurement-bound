"""Leave-one-out study: time-split exact rows.  Same adjoint replay as ladder_core.Point.replay_rows,
but the contribution of step t goes to Y_early if t < T/2 and to Y_late otherwise (Y_early + Y_late = Y exactly).
  python loo_time_split.py --dir <cache/loo> --models b,rare0,rare0_s1 [--budget 240]
Writes <dir>/<model>_Ysplit_<c0>.npz (early, late) per chunk of 50 rows and prints max |early+late - Y|.
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loo_points import build

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--models', required=True); ap.add_argument('--budget', type=float, default=240)
a = ap.parse_args()
t0 = time.time()


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
    pt = None
    Y = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(a.dir, f'{nm}_Y_*.npy')))])
    for c0 in range(0, Y.shape[0], 50):
        fb = os.path.join(a.dir, f'{nm}_Ysplit_{c0:04d}.npz')
        if os.path.exists(fb):
            continue
        if time.time() - t0 > a.budget:
            break
        if pt is None:
            pt = build(nm); pt.fit_reference(seg=pt.T)
        rows = list(range(c0, min(c0 + 50, Y.shape[0])))
        Ye, Yl = replay_split(pt, rows)
        err = np.abs(Ye + Yl - Y[rows]).max() / np.abs(Y[rows]).max()
        np.savez(fb, early=Ye, late=Yl)
        print(f'[{nm}] rows {c0}+  max rel |early+late-Y| {err:.1e}  ({time.time()-t0:.0f}s)', flush=True)
done = all(os.path.exists(os.path.join(a.dir, f'{nm}_Ysplit_{c0:04d}.npz'))
           for nm in a.models.split(',') for c0 in range(0, len(glob.glob(os.path.join(a.dir, f'{nm}_Y_*.npy'))) * 50, 50))
print(f'split done: {done}  total {time.time()-t0:.0f}s', flush=True)
