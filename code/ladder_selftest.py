"""Scale ladder: correctness checks of the exact replay (ladder_core.py) on the tiny CNN points.
  python ladder_selftest.py [--device cpu|cuda] [--dtype float64|float32]
(1) adjoint rows vs central finite differences in the data weights (GroupNorm and batch-norm-with-reference variants,
    weight decay, one-cycle schedule, momentum);  (2) checkpoint segment length, micro-batching and row chunking do
    not change the rows.  On a GPU run it once in float64 (correctness) and once in float32 (precision of the ladder).
"""
import argparse, os, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu'); ap.add_argument('--dtype', default='float64'); ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()
if a.device.startswith('cuda'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
torch.set_num_threads(a.threads)
if a.device.startswith('cuda'):
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_core import Point

t0 = time.time()
dt = getattr(torch, a.dtype)
# Max-pooling makes training piecewise smooth: a finite-difference step that is too large crosses a kink
# (eps=1e-4 did on tiny_gn), so the float64 check uses eps=1e-6.  In float32 a finite difference is useless;
# there the float32 rows are compared with float64 adjoint rows of the same point.
eps, tol = 1e-6, 1e-5
ok = True
for name in ('tiny_gn', 'tiny_bn'):
    pt = Point(name, a.device, dt)
    pt.fit_reference(seg=pt.T)
    rows = [0, 7, 33]
    Y = pt.replay_rows(rows)
    if a.dtype == 'float64':
        rng = np.random.RandomState(3)
        cols = rng.choice(pt.N, 6, replace=False)
        ref = np.zeros((len(rows), len(cols)))
        for jj, i in enumerate(cols):
            w = torch.ones(pt.N, dtype=dt, device=pt.dev)
            w[i] += eps; fp = pt.margins(pt.train(w))[rows]
            w[i] -= 2 * eps; fm = pt.margins(pt.train(w))[rows]
            ref[:, jj] = ((fp - fm) / (2 * eps)).detach().cpu().numpy()
        what = 'finite difference (eps 1e-6)'
    else:
        p64 = Point(name, a.device, torch.float64); p64.fit_reference(seg=p64.T)
        cols = np.arange(pt.N); ref = p64.replay_rows(rows); tol = 1e-3
        what = 'float64 adjoint'
    ad = Y[:, cols]
    rel = np.abs(ad - ref).max() / np.abs(ref).max()
    cos = np.sum(ad * ref) / np.sqrt(np.sum(ad ** 2) * np.sum(ref ** 2))
    print(f'[{name}] P={pt.P} T={pt.T}: adjoint vs {what}: max rel err {rel:.2e}, cosine {cos:.8f} (|Y| max {np.abs(ref).max():.2e})', flush=True)
    ok &= rel < tol
    # segment length / chunking / micro-batching
    pt2 = Point(name, a.device, dt, micro=(10 if name == 'tiny_gn' else None))
    pt2.fit_reference(seg=4)
    Y2 = np.concatenate([pt2.replay_rows([r]) for r in rows])
    rel2 = np.abs(Y2 - Y).max() / np.abs(Y).max()
    print(f'[{name}] seg=4{" micro=10" if name == "tiny_gn" else ""} chunk=1 vs seg=T chunk=3: max rel diff {rel2:.2e}', flush=True)
    ok &= rel2 < (1e-10 if a.dtype == 'float64' else 1e-4)
print(f'selftest {"PASSED" if ok else "FAILED"}  ({time.time()-t0:.1f}s)')
