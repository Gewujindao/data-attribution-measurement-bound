"""Adjoint surrogates: hand-written adjoint recursion (per-step Hessian-vector products) on the base setting of baseline_generate.py.
Modes:
  exact   exact Y split by epoch, finite-difference check, LDS truth, truncation + low-rank-early oracle (test 2)
  share   replace the per-step minibatch Hessian by shared ones (test 1)
  secant  surrogate Hessian action from the stored adjoint trajectories of B exact replays (test 3)
Usage: python adjoint_surrogates.py --mode exact --tmp <npz path>
"""
import argparse, time
import numpy as np
import torch
from torch.func import vmap, grad, jvp
from scipy.stats import rankdata
from sklearn.datasets import load_digits

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True)
ap.add_argument('--tmp', required=True)
ap.add_argument('--B', type=int, default=20)
ap.add_argument('--meval', type=int, default=40)
ap.add_argument('--nH', type=int, default=300)
ap.add_argument('--rmax', type=int, default=220)
ap.add_argument('--q', type=int, default=1)
a = ap.parse_args()
t0 = time.time()
N, k, E, bs, Hd, lr, mom, Msub = 1000, 100, 20, 50, 32, 0.05, 0.9, 100

# ---------------- data / model (same seeds and order as baseline_generate.py) ----------------
X, y = load_digits(return_X_y=True)
y = y.astype(np.int64)
rng = np.random.RandomState(0)
perm = rng.permutation(len(y))
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
Xtr, ytr = torch.tensor(X[perm[:N]]), torch.tensor(y[perm[:N]])
Xte, yte = torch.tensor(X[perm[-k:]]), torch.tensor(y[perm[-k:]])
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
        gg = wgrad(th, Xtr[idx], ytr[idx], Wm[:, idx])
        v = mom * v + gg
        th = th - lr * v
    if record:
        TH[T] = th[0]; return th, TH
    return th
def margins(th):
    return vmap(lambda p: margin_vec(p, Xte, yte))(th)

thT, TH = trainM(torch.ones(1, N), record=True)
f0 = margins(thT)[0]
Gf = vmap(grad(lambda p, x, yy: margin_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))(TH[T], Xte, yte)  # k x P
pergrad = vmap(grad(lambda p, x, yy: ce_vec(p, x[None], yy[None])[0]), in_dims=(None, 0, 0))
print(f'[{a.mode}] P={P} T={T} test acc {(logits_fn(TH[T], Xte).argmax(-1) == yte).double().mean():.3f}  {time.time()-t0:.1f}s', flush=True)

def hvpB(th, xb, yb, C):
    gfn = grad(lambda p: ce_vec(p, xb, yb).mean())
    th = th.clone()                            # a view of TH would make forward AD allocate for the whole base
    return vmap(lambda c: jvp(gfn, (th,), (c,))[1])(C)
def h_exact(t, C):
    return hvpB(TH[t], Xtr[bidx[t]], ytr[bidx[t]], C)

def backward(X0, hfun, split=False, store=None):
    """Adjoint recursion for theta_{t+1}=theta_t-lr*v_{t+1}, v_{t+1}=mom*v_t+g_t. Rows of X0 are grad f(theta_T)."""
    m = X0.shape[0]
    aa = X0.clone(); bb = torch.zeros_like(aa)
    Y = np.zeros((m, E, N)) if split else np.zeros((m, N))
    for t in range(T - 1, -1, -1):
        idx = bidx[t]
        c = bb - lr * aa                       # df/dg_t
        contrib = (c @ pergrad(TH[t], Xtr[idx], ytr[idx]).T / bs).numpy()
        if split: Y[:, t // spe, idx.numpy()] = contrib
        else: Y[:, idx.numpy()] += contrib
        Hc = hfun(t, c)
        if store is not None: store(t, c, Hc)
        aa = aa + Hc
        bb = mom * c
    return Y

def lds_per(Yh, masks, dlt):
    pred = -(1 - masks) @ Yh.T
    rp, rd = rankdata(pred, axis=0), rankdata(dlt, axis=0)
    rp -= rp.mean(0); rd -= rd.mean(0)
    return (rp * rd).sum(0) / np.sqrt((rp ** 2).sum(0) * (rd ** 2).sum(0) + 1e-300)
def ccos(A, Bm):
    A = A - A.mean(1, keepdims=True); Bm = Bm - Bm.mean(1, keepdims=True)
    return (A * Bm).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1) + 1e-300)

if a.mode == 'exact':
    Yep = backward(Gf, h_exact, split=True)
    Y = Yep.sum(1)
    print(f'exact Y by adjoint recursion {time.time()-t0:.1f}s', flush=True)
    ii = [3, 500, 997]; eps = 1e-3
    Wfd = torch.ones(6, N)
    for j, i in enumerate(ii):
        Wfd[2 * j, i] += eps; Wfd[2 * j + 1, i] -= eps
    ff = margins(trainM(Wfd)).numpy()
    for j, i in enumerate(ii):
        fd = (ff[2 * j] - ff[2 * j + 1]) / (2 * eps)
        print(f'  finite-difference check train sample {i}: max|fd-Y| {np.abs(fd - Y[:, i]).max():.2e}  max|Y| {np.abs(Y[:, i]).max():.2e}')
    r = np.random.RandomState(17)
    masks = np.ones((Msub, N))
    for m in range(Msub):
        masks[m, r.choice(N, 10, replace=False)] = 0
    dlt = margins(trainM(torch.tensor(masks))).numpy() - f0.numpy()[None]
    np.savez(a.tmp, Yep=Yep, masks=masks, dlt=dlt)
    Ls = lds_per(Y, masks, dlt).mean()
    print(f'truth done {time.time()-t0:.1f}s; exact LDS 1% = {Ls:.3f}', flush=True)

    def sph(A, B):
        if B == 0: return np.zeros_like(A)
        rn = np.linalg.norm(A, axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(A / rn, full_matrices=False)
        return rn * ((U[:, :B] * S[:B]) @ Vt[:B])
    def spec(A):
        s = np.linalg.svd(A / np.linalg.norm(A, axis=1, keepdims=True), compute_uv=False) ** 2
        return s.sum() ** 2 / (s ** 2).sum(), int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9) + 1), int(np.searchsorted(np.cumsum(s) / s.sum(), 0.85) + 1)
    print('\n== LDS vs centred cosine rho (spherical SVD oracle on full Y) ==')
    for B in (10, 20, 30, 50, 60, 80):
        Yh = sph(Y, B); rho = ccos(Yh, Y)
        print(f'  B={B}: LDS {lds_per(Yh, masks, dlt).mean():.3f}  mean rho {rho.mean():.3f}  rho*L* {rho.mean()*Ls:.3f}  mean rho^2 {np.mean(rho**2):.3f}')
    print('  full Y row-normalised: PR %.1f rank90 %d rank85 %d' % spec(Y))
    print('\n== test 2: exact late part (truncated replays) + rank-B spherical oracle on the early part ==')
    for late in (1, 2, 3, 4, 6, 10):
        Yl = Yep[:, E - late:].sum(1); Ye = Y - Yl
        pr, r90, r85 = spec(Ye)
        line = f'  late {late} ep (cost {late/E:.2f}k): late-only LDS {lds_per(Yl, masks, dlt).mean():.3f}; early part energy {np.sum(Ye**2)/np.sum(Y**2):.2f} PR {pr:.1f} rank90 {r90};'
        for B in (5, 10, 15, 20, 30, 50):
            line += f' B={B}:{lds_per(Yl + sph(Ye, B), masks, dlt).mean():.3f}'
        print(line)
    print('  per-epoch row-normalised rank90 of single-epoch parts:', ' '.join(str(spec(Yep[:, e])[1]) for e in range(E)))
    print('  cumulative early part (epochs < e) rank90:', ' '.join(f'{e}:{spec(Yep[:, :e].sum(1))[1]}' for e in (1, 2, 5, 10, 15)))

elif a.mode == 'share':
    D = np.load(a.tmp); Y = D['Yep'].sum(1); masks, dlt = D['masks'], D['dlt']
    ev = np.arange(a.meval)
    rs = np.random.RandomState(5)
    sub = torch.tensor(np.sort(rs.choice(N, a.nH, replace=False)))
    XH, yH = Xtr[sub], ytr[sub]
    def rep(name, Yh):
        print(f'  {name}: LDS {lds_per(Yh, masks, dlt[:, ev]).mean():.3f}  mean centred cos with exact {ccos(Yh, Y[ev]).mean():.3f}  ({time.time()-t0:.1f}s)', flush=True)
    print(f'== test 1 on {a.meval} test points; exact rows LDS {lds_per(Y[ev], masks, dlt[:, ev]).mean():.3f} ==')
    rep('A0 zero Hessian', backward(Gf[ev], lambda t, C: torch.zeros_like(C)))
    rs2 = np.random.RandomState(6)
    other = [torch.tensor(rs2.choice(N, bs, replace=False)) for _ in range(T)]
    rep('A1b another random minibatch at theta_t', backward(Gf[ev], lambda t, C: hvpB(TH[t], Xtr[other[t]], ytr[other[t]], C)))
    rep(f'A1 fixed {a.nH}-sample Hessian at theta_t', backward(Gf[ev], lambda t, C: hvpB(TH[t], XH, yH, C)))
    for W in (20, 100):
        rep(f'A2 fixed {a.nH}-sample Hessian at window midpoint, W={W}', backward(Gf[ev], lambda t, C: hvpB(TH[(t // W) * W + W // 2], XH, yH, C)))

elif a.mode == 'secant':
    D = np.load(a.tmp); Y = D['Yep'].sum(1); masks, dlt = D['masks'], D['dlt']
    pm = np.random.RandomState(11).permutation(k)
    pr, ev = pm[:a.B], pm[k - a.meval:]
    Cs = np.zeros((T, a.B, P), dtype=np.float32); HCs = np.zeros((T, a.B, P), dtype=np.float32)
    def store(t, c, Hc):
        Cs[t] = c.numpy(); HCs[t] = Hc.numpy()
    Ypr = backward(Gf[pr], h_exact, store=store)
    print(f'== test 3: B={a.B} exact replays stored ({time.time()-t0:.1f}s); held-out {a.meval} test points, exact LDS {lds_per(Y[ev], masks, dlt[:, ev]).mean():.3f} ==', flush=True)
    def rep(name, Yh):
        print(f'  {name}: LDS {lds_per(Yh, masks, dlt[:, ev]).mean():.3f}  mean centred cos with exact {ccos(Yh, Y[ev]).mean():.3f}  ({time.time()-t0:.1f}s)', flush=True)
    al = np.linalg.lstsq(Gf[pr].numpy().T, Gf[ev].numpy().T, rcond=None)[0]       # B x m
    res = np.linalg.norm(Gf[ev].numpy().T - Gf[pr].numpy().T @ al, axis=0) / np.linalg.norm(Gf[ev].numpy(), axis=1)
    rep(f'S1 parameter-space regression on probes (unexplained grad norm {res.mean():.2f})', al.T @ Ypr)
    rep('A0 zero Hessian', backward(Gf[ev], lambda t, C: torch.zeros_like(C)))
    def pinv_sym(G, rel):
        w, V = np.linalg.eigh(G)
        keep = w > rel * w[-1]
        return (V[:, keep] / w[keep]) @ V[:, keep].T
    def surrogate(W):
        cache = {}
        def h(t, C):
            c = C.numpy()
            St, Ht = Cs[t].astype(np.float64), HCs[t].astype(np.float64)
            a1 = pinv_sym(St @ St.T, 1e-10) @ (St @ c.T)            # B x m
            out = (Ht.T @ a1).T
            if W > 1:
                w = t // W
                if w not in cache:
                    cache.clear()
                    S = Cs[w * W:(w + 1) * W].reshape(-1, P).astype(np.float64)
                    Yh = HCs[w * W:(w + 1) * W].reshape(-1, P).astype(np.float64)
                    cache[w] = (S, Yh, pinv_sym(S @ S.T, 1e-8))
                S, Yh, Gi = cache[w]
                r = c - (St.T @ a1).T
                out = out + (Yh.T @ (Gi @ (S @ r.T))).T
            return torch.tensor(out)
        return h
    rep('S2 same-step secant pairs only', backward(Gf[ev], surrogate(1)))
    for W in (5, 20):
        rep(f'S3 same-step + within-window pairs, W={W}', backward(Gf[ev], surrogate(W)))
elif a.mode == 'lowrank':
    # test 4: per-window (W=20 steps = 1 epoch) rank-r Ritz approximation of a subsampled Hessian at the window midpoint,
    # learned by randomized subspace iteration (test-point independent); complement of the subspace gets H=0.
    D = np.load(a.tmp); Y = D['Yep'].sum(1); masks, dlt = D['masks'], D['dlt']
    W, rmax, q = 20, a.rmax, a.q
    rs = np.random.RandomState(5)
    Us, Ls = [], []
    for w in range(T // W):
        sub = torch.tensor(np.sort(rs.choice(N, a.nH, replace=False)))
        th = TH[w * W + W // 2]
        hv = lambda V: torch.cat([hvpB(th, Xtr[sub], ytr[sub], V[i:i + 44]) for i in range(0, V.shape[0], 44)])
        Q = torch.linalg.qr(torch.randn(P, rmax, generator=torch.Generator().manual_seed(100 + w)))[0].T
        for _ in range(q):
            Q = torch.linalg.qr(hv(Q).T)[0].T
        lam, V = torch.linalg.eigh(Q @ hv(Q).T)
        o = torch.argsort(-lam.abs())
        Us.append((V[:, o].T @ Q)); Ls.append(lam[o])
        if w in (0, 1, 5, 19):
            l = Ls[-1]
            print(f'  window {w}: Ritz |lam| at index 1/5/10/20/50/100/{rmax}: ' + ' '.join(f'{l[i-1].abs():.3g}' for i in (1, 5, 10, 20, 50, 100, rmax)) +
                  f'; negative among top 20: {(l[:20] < 0).sum().item()}', flush=True)
    print(f'== test 4: nH={a.nH}, q={q}, rmax={rmax}; subspaces learned in {time.time()-t0:.1f}s; all {k} test points; exact LDS {lds_per(Y, masks, dlt).mean():.3f} ==', flush=True)
    def sph(A, B):
        rn = np.linalg.norm(A, axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(A / rn, full_matrices=False)
        return rn * ((U[:, :B] * S[:B]) @ Vt[:B])
    for r in [x for x in (5, 10, 20, 50, 100, 200, 400) if x <= rmax]:
        Yh = backward(Gf, lambda t, C: ((C @ Us[t // W][:r].T) * Ls[t // W][:r]) @ Us[t // W][:r])
        cst = (q + 1) * (r + 10) * a.nH / N
        cc = (Yh * Y).sum(1) / (Yh * Yh).sum(1)
        R = Y - cc[:, None] * Yh
        print(f'  rank {r} (learning cost about {cst:.0f} replay-equivalents if run with r+10 probes): LDS {lds_per(Yh, masks, dlt).mean():.3f}  mean centred cos {ccos(Yh, Y).mean():.3f}'
              f' | + rank-B oracle on residual: B=10 {lds_per(cc[:, None] * Yh + sph(R, 10), masks, dlt).mean():.3f}  B=20 {lds_per(cc[:, None] * Yh + sph(R, 20), masks, dlt).mean():.3f}  ({time.time()-t0:.1f}s)', flush=True)
print(f'total {time.time()-t0:.1f}s')
