"""Ceiling: generate one scale point (exact attribution matrix Y + retrain truth).
Exact Y by the hand-written adjoint recursion of adjoint_surrogates.py, batched over test points, written chunk by chunk
(resumable: existing chunk files are skipped; the run stops itself after --budget seconds).
Usage: python ceiling_generate.py --data digits|synth --hidden 32 --N 1000 --k 300 --dir <scratch dir> [--stage exact|truth]
"""
import argparse, os, time
import numpy as np
import torch
from torch.func import vmap, grad, jvp
from sklearn.datasets import load_digits, make_classification

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--hidden', type=int, default=32)
ap.add_argument('--N', type=int, default=1000)
ap.add_argument('--k', type=int, default=300)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--stage', default='exact')
ap.add_argument('--budget', type=float, default=240)
ap.add_argument('--dir', required=True)
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.dir, exist_ok=True)
N, k, E, bs, Hd, lr, mom = a.N, a.k, a.epochs, 50, a.hidden, 0.05, 0.9

if a.data == 'digits':
    X, y = load_digits(return_X_y=True)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    tr, te = perm[:N], perm[-k:]
else:
    X, y = make_classification(n_samples=N + k, n_features=64, n_informative=24, n_redundant=8, n_classes=10,
                               n_clusters_per_class=2, class_sep=1.5, flip_y=0.01, random_state=0)
    rng = np.random.RandomState(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    tr, te = np.arange(N), np.arange(N, N + k)
y = y.astype(np.int64)
Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
Xte, yte = torch.tensor(X[te]), torch.tensor(y[te])
spe = N // bs
T = E * spe
orders = [torch.tensor(rng.permutation(N)) for _ in range(E)]
bidx = [orders[t // spe][(t % spe) * bs:(t % spe + 1) * bs] for t in range(T)]
g = torch.Generator().manual_seed(1)
d = [64, Hd, Hd, 10]
P0 = []
for i in range(3):
    P0.append(torch.randn(d[i], d[i + 1], generator=g) / np.sqrt(d[i]))
    P0.append(torch.zeros(d[i + 1]))
shapes = [p.shape for p in P0]
sizes = [p.numel() for p in P0]
P = sum(sizes)
th0 = torch.cat([p.reshape(-1) for p in P0])

def logits_fn(v, x):
    ps, o = [], 0
    for sh, sz in zip(shapes, sizes):
        ps.append(v[o:o + sz].reshape(sh)); o += sz
    h = torch.tanh(x @ ps[0] + ps[1])
    h = torch.tanh(h @ ps[2] + ps[3])
    return h @ ps[4] + ps[5]
def ce_vec(v, x, yy):
    lo = logits_fn(v, x)
    return torch.logsumexp(lo, -1) - lo.gather(-1, yy[:, None])[:, 0]
def margin_vec(v, x, yy):
    lo = logits_fn(v, x)
    ly = lo.gather(-1, yy[:, None])[:, 0]
    other = lo.masked_fill(yy[:, None] == torch.arange(10), -1e30)
    return ly - torch.logsumexp(other, -1)

wgrad = vmap(grad(lambda p, xb, yb, w: (w * ce_vec(p, xb, yb)).sum() / bs), in_dims=(0, None, None, 0))
def trainM(Wm, record=False):
    M = Wm.shape[0]
    th = th0[None].repeat(M, 1); v = torch.zeros_like(th)
    TH = torch.zeros(T + 1, P) if record else None
    for t in range(T):
        idx = bidx[t]
        if record: TH[t] = th[0]
        v = mom * v + wgrad(th, Xtr[idx], ytr[idx], Wm[:, idx])
        th = th - lr * v
    if record:
        TH[T] = th[0]; return th, TH
    return th
def margins(th):
    return vmap(lambda p: margin_vec(p, Xte, yte))(th)

tag = f'{a.data}_h{Hd}_N{N}_k{k}' + (f'_e{E}' if E != 20 else '')
if a.stage == 'exact':
    thT, TH = trainM(torch.ones(1, N), record=True)
    acc = (logits_fn(TH[T], Xte).argmax(-1) == yte).double().mean().item()
    tracc = (logits_fn(TH[T], Xtr).argmax(-1) == ytr).double().mean().item()
    print(f'[{tag}] P={P} T={T} train acc {tracc:.3f} test acc {acc:.3f}  {time.time()-t0:.1f}s', flush=True)
    Gf = vmap(grad(lambda p, x, yy: margin_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))(TH[T], Xte, yte)
    pergrad = vmap(grad(lambda p, x, yy: ce_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))
    def hvpB(th, xb, yb, C):
        gfn = grad(lambda p: ce_vec(p, xb, yb).mean())
        th = th.clone()
        return vmap(lambda c: jvp(gfn, (th,), (c,))[1])(C)
    for c0 in range(0, k, a.chunk):
        fn = os.path.join(a.dir, f'{tag}_Y_{c0:04d}.npy')
        if os.path.exists(fn):
            continue
        if time.time() - t0 > a.budget:
            print('budget reached; resume later', flush=True); break
        aa = Gf[c0:c0 + a.chunk].clone(); bb = torch.zeros_like(aa)
        Yc = np.zeros((aa.shape[0], N))
        for t in range(T - 1, -1, -1):
            idx = bidx[t]
            c = bb - lr * aa
            Yc[:, idx.numpy()] += (c @ pergrad(TH[t], Xtr[idx], ytr[idx]).T / bs).numpy()
            aa = aa + hvpB(TH[t], Xtr[idx], ytr[idx], c)
            bb = mom * c
        np.save(fn, Yc)
        print(f'  chunk {c0}: done {time.time()-t0:.1f}s', flush=True)
    np.savez(os.path.join(a.dir, f'{tag}_meta.npz'), acc=acc, tracc=tracc, P=P, T=T, ytr=ytr.numpy(), yte=yte.numpy())
elif a.stage == 'truth':
    f0 = margins(trainM(torch.ones(1, N)))[0].numpy()
    out, complete = {}, True
    for alpha, M, seed in ((0.01, 200, 17), (0.05, 100, 23)):
        r = np.random.RandomState(seed)
        masks = np.ones((M, N))
        for m in range(M):
            masks[m, r.choice(N, int(round(alpha * N)), replace=False)] = 0
        dl = np.zeros((M, k))
        for m0 in range(0, M, 50):           # batches of 50 retrains, each saved (resumable)
            fb = os.path.join(a.dir, f'{tag}_truth_{int(alpha*100):02d}_{m0:03d}.npy')
            if not os.path.exists(fb):
                if time.time() - t0 > a.budget:
                    complete = False; continue
                np.save(fb, margins(trainM(torch.tensor(masks[m0:m0 + 50]))).numpy() - f0[None])
            dl[m0:m0 + 50] = np.load(fb)
        out[f'm{int(alpha*100):02d}'] = masks; out[f'd{int(alpha*100):02d}'] = dl
        print(f'[{tag}] truth {alpha}: {time.time()-t0:.1f}s', flush=True)
    if complete:
        np.savez(os.path.join(a.dir, f'{tag}_truth.npz'), f0=f0, **out)
    else:
        print('budget reached; resume later', flush=True)
print(f'total {time.time()-t0:.1f}s', flush=True)
