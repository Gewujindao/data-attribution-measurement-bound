"""Baselines: exact metagradient influence matrix, LDS ground truth, endpoint baselines.
Small deterministic setting: sklearn digits, MLP 64-h-h-10 (tanh), SGD+momentum, fixed order, float64.
Usage: python baseline_generate.py --hidden 32 --N 1000 --k 100 --tag base
"""
import argparse, time, os
import numpy as np
import torch
from torch.func import functional_call, vmap, grad, hessian
from sklearn.datasets import load_digits

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--hidden', type=int, default=32)
ap.add_argument('--N', type=int, default=1000)
ap.add_argument('--k', type=int, default=100)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--bs', type=int, default=50)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--mom', type=float, default=0.9)
ap.add_argument('--M', type=int, default=100)
ap.add_argument('--tag', default='base')
ap.add_argument('--nohess', action='store_true')
ap.add_argument('--out', required=True)
a = ap.parse_args()
t0 = time.time()

# ---------------- data ----------------
X, y = load_digits(return_X_y=True)
y = y.astype(np.int64)
rng = np.random.RandomState(0)
perm = rng.permutation(len(y))
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
Xtr, ytr = torch.tensor(X[perm[:a.N]]), torch.tensor(y[perm[:a.N]])
Xte, yte = torch.tensor(X[perm[-a.k:]]), torch.tensor(y[perm[-a.k:]])
N, k, E, bs, H = a.N, a.k, a.epochs, a.bs, a.hidden
spe = N // bs
orders = [torch.tensor(rng.permutation(N)) for _ in range(E)]

g = torch.Generator().manual_seed(1)
def init():
    d = [64, H, H, 10]
    ps = []
    for i in range(3):
        ps.append(torch.randn(d[i], d[i + 1], generator=g) / np.sqrt(d[i]))
        ps.append(torch.zeros(d[i + 1]))
    return ps
P0 = init()
shapes = [p.shape for p in P0]
sizes = [p.numel() for p in P0]
Pdim = sum(sizes)

def fwd(ps, x):  # ps: list with leading model dim M; x: (M,B,64)
    h = torch.tanh(x @ ps[0] + ps[1][:, None, :])
    h = torch.tanh(h @ ps[2] + ps[3][:, None, :])
    return h @ ps[4] + ps[5][:, None, :]

def margin(logits, yy):  # logits (...,10)
    yy = yy.expand(logits.shape[:-1])
    ly = logits.gather(-1, yy[..., None])[..., 0]
    other = logits.masked_fill(yy[..., None] == torch.arange(10), -1e30)
    return ly - torch.logsumexp(other, -1)

def ce(logits, yy):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, 10), yy.expand(logits.shape[:-1]).reshape(-1), reduction='none').reshape(logits.shape[:-1])

def train(W, create_graph=False, hook=None):
    """W: (M,E,N) data weights. Returns final params list (M,...)."""
    M = W.shape[0]
    ps = [p[None].expand(M, *p.shape).clone().requires_grad_(True) for p in P0]
    vs = [torch.zeros_like(p) for p in ps]
    for e in range(E):
        for s in range(spe):
            idx = orders[e][s * bs:(s + 1) * bs]
            xb = Xtr[idx][None].expand(M, -1, -1)
            lo = fwd(ps, xb)
            L = (W[:, e, idx] * ce(lo, ytr[idx])).sum() / bs
            gs = torch.autograd.grad(L, ps, create_graph=create_graph)
            if hook is not None:
                hook(e, s, idx, ps)
            vs = [a.mom * v + gg for v, gg in zip(vs, gs)]
            ps = [p - a.lr * v for p, v in zip(ps, vs)]
            if not create_graph:
                ps = [p.detach().requires_grad_(True) for p in ps]
                vs = [v.detach() for v in vs]
    return ps

def test_margin(ps):
    return margin(fwd(ps, Xte[None].expand(ps[0].shape[0], -1, -1)), yte)  # (M,k)

# ---------------- exact metagradient (per-epoch split) ----------------
W = torch.ones(1, E, N, requires_grad=True)
ps = train(W, create_graph=True)
f0 = test_margin(ps)[0]
acc = (fwd(ps, Xte[None]).argmax(-1)[0] == yte).double().mean().item()
tr_acc = (fwd(ps, Xtr[None]).argmax(-1)[0] == ytr).double().mean().item()
print(f'[{a.tag}] P={Pdim} train acc {tr_acc:.3f} test acc {acc:.3f} train {time.time()-t0:.1f}s', flush=True)
Yep = np.zeros((k, E, N))
for t in range(k):
    gW, = torch.autograd.grad(f0[t], W, retain_graph=(t < k - 1))
    Yep[t] = gW[0].numpy()
del ps
f0 = f0.detach()
print(f'exact Y done {time.time()-t0:.1f}s', flush=True)
pT = [p[0].detach() for p in train(torch.ones(1, E, N))]

# ---------------- ground truth: random subset retrains ----------------
truth = {}
for alpha in (0.01, 0.05):
    r = np.random.RandomState(int(alpha * 1000) + 7)
    masks = np.ones((a.M, N))
    for m in range(a.M):
        masks[m, r.choice(N, int(round(alpha * N)), replace=False)] = 0
    Wm = torch.tensor(masks)[:, None, :].expand(a.M, E, N).contiguous()
    with torch.no_grad():
        pass
    pm = train(Wm)
    fm = test_margin([p.detach() for p in pm]).detach().numpy()
    truth[alpha] = (masks, fm - f0.numpy()[None])
print(f'truth done {time.time()-t0:.1f}s', flush=True)

# ---------------- endpoint linearization ----------------
def unflat(v):
    out, o = [], 0
    for sh, sz in zip(shapes, sizes):
        out.append(v[o:o + sz].reshape(sh)); o += sz
    return out
def f_single(v, x):
    ps = [p[None] for p in unflat(v)]
    return fwd(ps, x[None, None])[0, 0]
def loss_i(v, x, yy):
    return torch.nn.functional.cross_entropy(f_single(v, x)[None], yy[None])
def marg_i(v, x, yy):
    return margin(f_single(v, x)[None], yy[None])[0]
thT = torch.cat([p.reshape(-1) for p in pT])
G = vmap(grad(loss_i), in_dims=(None, 0, 0))(thT, Xtr, ytr).numpy()    # N x P
Gt = vmap(grad(marg_i), in_dims=(None, 0, 0))(thT, Xte, yte).numpy()   # k x P
if a.nohess:
    GV = GtV = lam = np.zeros(1)
else:
  vv = thT.clone().requires_grad_(True)
  Lm = ce(fwd([p[None] for p in unflat(vv)], Xtr[None]), ytr).mean()
  gL, = torch.autograd.grad(Lm, vv, create_graph=True)
  Hs = np.zeros((Pdim, Pdim))
  for c in range(0, Pdim, 25):
      Ic = torch.zeros(min(25, Pdim - c), Pdim)
      Ic[torch.arange(Ic.shape[0]), torch.arange(c, c + Ic.shape[0])] = 1
      Hs[c:c + 25] = torch.autograd.grad(gL, vv, grad_outputs=Ic, is_grads_batched=True, retain_graph=True)[0].numpy()
  del gL, Lm
  Hs += Hs.T
  Hs *= 0.5
  lam, V = np.linalg.eigh(Hs)
  del Hs
  GV, GtV = G @ V, Gt @ V
  del V
print(f'hessian done {time.time()-t0:.1f}s  eig min {lam.min():.3e} max {lam.max():.3e}', flush=True)

# ---------------- TracIn-style trajectory (checkpoint per step) ----------------
Ytr_end = np.zeros((k, N))   # sum_s lr * grad f_t(theta_T) . grad l_i(theta_s)
Ytr_ck = np.zeros((k, N))    # sum_s lr * grad f_t(theta_s) . grad l_i(theta_s)
def hook(e, s, idx, ps):
    v = torch.cat([p[0].detach().reshape(-1) for p in ps])
    gi = vmap(grad(loss_i), in_dims=(None, 0, 0))(v, Xtr[idx], ytr[idx]).numpy()
    gt = vmap(grad(marg_i), in_dims=(None, 0, 0))(v, Xte, yte).numpy()
    Ytr_end[:, idx.numpy()] -= a.lr / bs * Gt @ gi.T
    Ytr_ck[:, idx.numpy()] -= a.lr / bs * gt @ gi.T
train(torch.ones(1, E, N), hook=hook)
print(f'tracin done {time.time()-t0:.1f}s', flush=True)

np.savez(a.out, Yep=Yep, f0=f0.numpy(), GV=GV, GtV=GtV, lam=lam,Ytr_end=Ytr_end, Ytr_ck=Ytr_ck,
         ytr=ytr.numpy(), yte=yte.numpy(), m01=truth[0.01][0], d01=truth[0.01][1],
         m05=truth[0.05][0], d05=truth[0.05][1], acc=acc, tr_acc=tr_acc, P=Pdim)
print(f'saved; total {time.time()-t0:.1f}s', flush=True)
