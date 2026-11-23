"""Estimator test: generate one new setting (exact attribution matrix Y + 1%-deletion retrain truth).
Same hand-written adjoint recursion as ceiling_generate.py, generalised to: ReLU/tanh, depth 2/3, several sklearn datasets,
regression (MSE loss, test function = prediction), out-of-distribution test inputs. Resumable chunks and retrain batches.
Usage: python estimator_test_generate.py --setting A --stage exact|truth --dir <scratch dir>
"""
import argparse, os, time
import numpy as np
import torch
from torch.func import vmap, grad, jvp
from sklearn.datasets import load_digits, make_classification, make_blobs, make_friedman1

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
SETTINGS = {   # data, act, depth, hidden, N, k, epochs, lr, ood
    'A':  dict(data='digits', act='relu', depth=2, hidden=64, N=1000, k=300, epochs=20, lr=0.05, ood=0),
    'B':  dict(data='digits', act='tanh', depth=3, hidden=32, N=1200, k=400, epochs=17, lr=0.05, ood=0),
    'C':  dict(data='synth_cls', act='tanh', depth=2, hidden=32, N=2000, k=500, epochs=10, lr=0.05, ood=0),
    'D':  dict(data='blobs', act='tanh', depth=2, hidden=96, N=800, k=600, epochs=25, lr=0.05, ood=0),
    'E':  dict(data='digits', act='tanh', depth=2, hidden=32, N=500, k=800, epochs=40, lr=0.05, ood=0),
    'F':  dict(data='friedman', act='tanh', depth=2, hidden=32, N=1000, k=300, epochs=20, lr=0.05, ood=0),
    'H1': dict(data='digits', act='relu', depth=2, hidden=48, N=1000, k=100, epochs=20, lr=0.05, ood=0),
    'H2': dict(data='digits', act='tanh', depth=2, hidden=32, N=1000, k=300, epochs=20, lr=0.05, ood=60),
    'H3': dict(data='synth_hard', act='tanh', depth=2, hidden=32, N=3000, k=300, epochs=25, lr=0.05, ood=0),
}
ap = argparse.ArgumentParser()
ap.add_argument('--setting', required=True)
ap.add_argument('--stage', default='exact')
ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--budget', type=float, default=200)
ap.add_argument('--tbatch', type=int, default=50)      # retrains per saved batch (25 for the larger models)
ap.add_argument('--dir', required=True)
a = ap.parse_args()
cfg = SETTINGS[a.setting]
t0 = time.time()
os.makedirs(a.dir, exist_ok=True)
N, k, E, bs, lr, mom, ood = cfg['N'], cfg['k'], cfg['epochs'], 50, cfg['lr'], 0.9, cfg['ood']
reg = cfg['data'] == 'friedman'

if cfg['data'] == 'digits':
    X, y = load_digits(return_X_y=True)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    tr, te = perm[:N], perm[-(k - ood):]
    Xte_np = X[te]
    if ood:
        Xte_np = np.concatenate([Xte_np, np.random.RandomState(11).standard_normal((ood, X.shape[1]))])
    yte_np = np.concatenate([y[te], np.zeros(ood)])
else:
    if cfg['data'] == 'synth_cls':
        X, y = make_classification(n_samples=N + k, n_features=40, n_informative=12, n_redundant=4, n_classes=5,
                                   n_clusters_per_class=1, class_sep=2.0, flip_y=0.0, random_state=1)
    elif cfg['data'] == 'blobs':
        X, y = make_blobs(n_samples=N + k, centers=8, n_features=32, cluster_std=2.5, random_state=2)
    elif cfg['data'] == 'synth_hard':
        X, y = make_classification(n_samples=N + k, n_features=64, n_informative=24, n_redundant=8, n_classes=10,
                                   n_clusters_per_class=2, class_sep=1.5, flip_y=0.01, random_state=5)
    else:
        X, y = make_friedman1(n_samples=N + k, n_features=10, noise=0.5, random_state=3)
        y = (y - y.mean()) / y.std()
    rng = np.random.RandomState(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    tr, te = np.arange(N), np.arange(N, N + k)
    Xte_np, yte_np = X[te], y[te]
ydt = torch.float64 if reg else torch.int64
Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr]).to(ydt)
Xte, yte = torch.tensor(Xte_np), torch.tensor(yte_np).to(ydt)
din, dout = X.shape[1], (1 if reg else int(y.max()) + 1)
spe = N // bs
T = E * spe
orders = [torch.tensor(rng.permutation(N)) for _ in range(E)]
bidx = [orders[t // spe][(t % spe) * bs:(t % spe + 1) * bs] for t in range(T)]
g = torch.Generator().manual_seed(1)
dims = [din] + [cfg['hidden']] * cfg['depth'] + [dout]
P0 = []
for i in range(len(dims) - 1):
    P0.append(torch.randn(dims[i], dims[i + 1], generator=g) / np.sqrt(dims[i]))
    P0.append(torch.zeros(dims[i + 1]))
shapes = [p.shape for p in P0]
sizes = [p.numel() for p in P0]
P = sum(sizes)
th0 = torch.cat([p.reshape(-1) for p in P0])
act = torch.tanh if cfg['act'] == 'tanh' else torch.relu

def out_fn(v, x):
    ps, o = [], 0
    for sh, sz in zip(shapes, sizes):
        ps.append(v[o:o + sz].reshape(sh)); o += sz
    h = x
    for i in range(0, len(ps) - 2, 2):
        h = act(h @ ps[i] + ps[i + 1])
    return h @ ps[-2] + ps[-1]
def loss_vec(v, x, yy):
    lo = out_fn(v, x)
    if reg:
        return 0.5 * (lo[:, 0] - yy) ** 2
    return torch.logsumexp(lo, -1) - lo.gather(-1, yy[:, None])[:, 0]
def test_vec(v, x, yy):
    lo = out_fn(v, x)
    if reg:
        return lo[:, 0]
    ly = lo.gather(-1, yy[:, None])[:, 0]
    other = lo.masked_fill(yy[:, None] == torch.arange(dout), -1e30)
    return ly - torch.logsumexp(other, -1)

wgrad = vmap(grad(lambda p, xb, yb, w: (w * loss_vec(p, xb, yb)).sum() / bs), in_dims=(0, None, None, 0))
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

tag = f'setting_{a.setting}'
thT = trainM(torch.ones(1, N))[0] if a.stage == 'truth' else None
if a.stage == 'exact':
    thT_, TH = trainM(torch.ones(1, N), record=True)
    thT = TH[T]
if ood and not reg:      # OOD test inputs are labelled by the fully trained model's prediction
    yte[k - ood:] = out_fn(thT, Xte[k - ood:]).argmax(-1)
def margins(th):
    return vmap(lambda p: test_vec(p, Xte, yte))(th)

if a.stage == 'exact':
    lo = out_fn(TH[T], Xte)
    acc = float(((lo[:, 0] - yte) ** 2).mean()) if reg else float((lo.argmax(-1)[:k - ood] == yte[:k - ood]).double().mean())
    print(f'[{tag}] P={P} T={T} test {"mse" if reg else "acc"} {acc:.3f}  {time.time()-t0:.1f}s', flush=True)
    Gf = vmap(grad(lambda p, x, yy: test_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))(TH[T], Xte, yte)
    pergrad = vmap(grad(lambda p, x, yy: loss_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))
    def hvpB(th, xb, yb, C):
        gfn = grad(lambda p: loss_vec(p, xb, yb).mean())
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
    np.savez(os.path.join(a.dir, f'{tag}_meta.npz'), acc=acc, P=P, T=T)
elif a.stage == 'truth':
    f0 = margins(thT[None])[0].detach().numpy()
    r = np.random.RandomState(17)
    M = 200
    masks = np.ones((M, N))
    for m in range(M):
        masks[m, r.choice(N, int(round(0.01 * N)), replace=False)] = 0
    dl, complete = np.zeros((M, k)), True
    tb = a.tbatch
    for m0 in range(0, M, tb):
        fb = os.path.join(a.dir, f'{tag}_truth_{m0:03d}.npy' if tb == 50 else f'{tag}_truth_b{tb}_{m0:03d}.npy')
        if not os.path.exists(fb):
            if time.time() - t0 > a.budget:
                complete = False; continue
            np.save(fb, margins(trainM(torch.tensor(masks[m0:m0 + tb]))).detach().numpy() - f0[None])
            print(f'  retrain batch {m0}: {time.time()-t0:.1f}s', flush=True)
        dl[m0:m0 + tb] = np.load(fb)
    if complete:
        np.savez(os.path.join(a.dir, f'{tag}_truth.npz'), f0=f0, m01=masks, d01=dl)
    else:
        print('budget reached; resume later', flush=True)
print(f'total {time.time()-t0:.1f}s', flush=True)
