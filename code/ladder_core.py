"""Scale ladder: device-agnostic exact attribution core for the scale ladder.

One "point" = data + model + training recipe + retrain-truth spec.  Everything is deterministic given the point.
  * legacy MLP points S1 (ceiling scripts) and H1 (estimator test) rebuild the old data, init and batch order exactly;
  * CNN points use a functional ResNet-9 (SPELL's architecture, width multiplier m) on a class-balanced CIFAR-10
    subset read from the local binary release (cifar-10-batches-bin), or tiny synthetic images for CPU tests.

Training (all points):   v <- mom*v + grad(sum_i w_i l_i / bs) + wd*theta ;  theta <- theta - eta_t * v
Exact row of Y for a test function f (adjoint through the whole trajectory):
   a_T = grad f(theta_T), b_T = 0;  for t = T-1..0:  c = b - eta_t a;  Y[:, batch_t] += (c . grad l_i(theta_t)) / bs;
                                                     a <- a + (H_t + wd) c;  b <- mom * c
   Both "c . grad l_i" (all i in the batch) and "H_t c" come from ONE forward-over-reverse jvp per (micro-)batch.
Normalisation layers: 'gn' (GroupNorm, per-sample, exact micro-batching) or 'bn' (batch statistics while training;
the test function uses statistics of a fixed reference batch of training images at theta_T, so the state stays
(theta, v)).  A deleted point keeps its image in the batch and only gets loss weight 0 (same definition as the
metagradient); the retrain truth uses exactly that definition.
"""
import math, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import vmap, grad, jvp

MOM_CORR = 1 + 1 / (1 - 0.85)          # SPELL divides lr and wd by 1 + 1/(1-momentum)
CNN_TRUTH = ((0.01, 200, 17), (0.05, 100, 23))
POINTS = {
    # ---- legacy CPU MLP points (exact rebuilds of ceiling_generate.py / estimator_test_generate.py) ----
    'S1': dict(kind='mlp', legacy='w3', data='digits', act='tanh', depth=2, hidden=32, N=1000, k=300, epochs=20, bs=50,
               lr=0.05, mom=0.9, wd=0.0, sched='const', truth=((0.01, 200, 17), (0.05, 100, 23)), tb=50),
    'H1': dict(kind='mlp', legacy='w4', data='digits', act='relu', depth=2, hidden=48, N=1000, k=100, epochs=20, bs=50,
               lr=0.05, mom=0.9, wd=0.0, sched='const', truth=((0.01, 200, 17),), tb=50),
    # ---- tiny CPU test points of the CNN path (synthetic 16x16 images) ----
    'tiny_gn': dict(kind='cnn', data='synth', width=1 / 16, norm='gn', N=120, k=60, epochs=3, bs=20, lr=0.05, mom=0.85,
                     wd=5e-4, sched='onecycle', truth=((0.01, 40, 17), (0.05, 20, 23)), tb=10),
    'tiny_bn': dict(kind='cnn', data='synth', width=1 / 16, norm='bn', N=120, k=60, epochs=3, bs=20, lr=0.05, mom=0.85,
                     wd=5e-4, sched='onecycle', truth=((0.01, 40, 17), (0.05, 20, 23)), tb=10),
    # ---- GPU ladder (CIFAR-10, batch-norm as in SPELL; no micro-batching); recipe follows SPELL's text (lr, wd divided by 1+1/(1-mom); one-cycle) ----
    'L2':  dict(kind='cnn', data='cifar', width=0.25, norm='bn', N=5000, k=500, epochs=20, bs=100,
                lr=0.085 / MOM_CORR, mom=0.85, wd=5e-4 / MOM_CORR, sched='onecycle', truth=CNN_TRUTH, tb=10),
    'L2n': dict(kind='cnn', data='cifar', width=0.25, norm='bn', N=10000, k=500, epochs=20, bs=200,
                lr=0.085 / MOM_CORR, mom=0.85, wd=5e-4 / MOM_CORR, sched='onecycle', truth=CNN_TRUTH, tb=10),
    'L3':  dict(kind='cnn', data='cifar', width=0.5, norm='bn', N=10000, k=500, epochs=20, bs=200,
                lr=0.085 / MOM_CORR, mom=0.85, wd=5e-4 / MOM_CORR, sched='onecycle', truth=CNN_TRUTH, tb=10),
    'L3w': dict(kind='cnn', data='cifar', width=1.0, norm='bn', N=10000, k=500, epochs=20, bs=200,
                lr=0.085 / MOM_CORR, mom=0.85, wd=5e-4 / MOM_CORR, sched='onecycle', truth=CNN_TRUTH, tb=10),
    # ---- rented GPU: SPELL's CIFAR-10 setting (n=50,000, batch 500, T=2,000) ----
    'R4gn': dict(kind='cnn', data='cifar', width=1.0, norm='gn', N=50000, k=500, epochs=20, bs=500,
                 lr=0.085 / MOM_CORR, mom=0.85, wd=5e-4 / MOM_CORR, sched='onecycle', truth=CNN_TRUTH, tb=10),
    'R4':   dict(kind='cnn', data='cifar', width=1.0, norm='bn', N=50000, k=500, epochs=20, bs=500,
                 lr=0.085 / MOM_CORR, mom=0.85, wd=5e-4 / MOM_CORR, sched='onecycle', truth=CNN_TRUTH, tb=10),
}


# ------------------------------------------------------------------ data
def load_cifar_bin(root):
    """CIFAR-10 binary release: <root>/cifar-10-batches-bin/{data_batch_1..5,test_batch}.bin (raw bytes, no pickle)."""
    d = os.path.join(root, 'cifar-10-batches-bin')
    names = [f'data_batch_{i}.bin' for i in range(1, 6)] + ['test_batch.bin']
    missing = [n for n in names if not os.path.isfile(os.path.join(d, n))]
    if missing:
        raise FileNotFoundError(f'CIFAR-10 binary files not found in {d} (missing: {", ".join(missing)}). '
                                f'Put the official "CIFAR-10 binary version" there, or pass --data_root.')
    def read(n):
        raw = np.fromfile(os.path.join(d, n), dtype=np.uint8).reshape(-1, 3073)
        return raw[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0, raw[:, 0].astype(np.int64)
    tr = [read(n) for n in names[:5]]
    Xtr, ytr = np.concatenate([t[0] for t in tr]), np.concatenate([t[1] for t in tr])
    Xte, yte = read(names[5])
    return Xtr, ytr, Xte, yte


def balanced_subset(y, n, rng):
    """class-balanced, prefix-nested subset (first n of a fixed class-interleaved order)."""
    per = [rng.permutation(np.where(y == c)[0]) for c in range(10)]
    order = np.stack([p[:min(len(q) for q in per)] for p in per], 1).reshape(-1)   # c0,c1,...,c9,c0,...
    return order[:n]


def synth_images(n, rng):
    tmpl = rng.standard_normal((10, 3, 16, 16)).astype(np.float32)
    y = rng.randint(0, 10, n)
    X = tmpl[y] + 1.5 * rng.standard_normal((n, 3, 16, 16)).astype(np.float32)
    return X, y.astype(np.int64)


# ------------------------------------------------------------------ the point
class Point:
    def __init__(self, name, device='cpu', dtype=torch.float64, data_root=None, micro=None):
        cfg = dict(POINTS[name]); self.name, self.cfg = name, cfg
        self.dev, self.dt = torch.device(device), dtype
        self.N, self.k, self.bs, E = cfg['N'], cfg['k'], cfg['bs'], cfg['epochs']
        self.mom, self.wd = cfg['mom'], cfg['wd']
        self.micro = micro or self.bs
        assert self.bs % self.micro == 0
        if cfg['kind'] == 'cnn' and cfg['norm'] == 'bn':
            assert self.micro == self.bs, 'batch-norm couples the batch: micro-batching would not be exact'
        if cfg['kind'] == 'mlp':
            self._build_mlp(cfg, E)
        else:
            self._build_cnn(cfg, E, data_root)
        T = self.T
        if cfg['sched'] == 'const':
            eta = np.full(T, cfg['lr'])
        else:                       # one-cycle linear: 0.1*lr at t=0, lr at 0.3T, 0 at T
            t = np.arange(T) / T
            eta = cfg['lr'] * np.where(t < 0.3, 0.1 + 0.9 * t / 0.3, (1 - t) / 0.7)
        self.eta = [float(e) for e in eta]
        self.P = int(self.th0.numel())

    # ---- legacy MLP (bitwise rebuild of the old generators) ----
    def _build_mlp(self, cfg, E):
        from sklearn.datasets import load_digits
        N, k = self.N, self.k
        X, y = load_digits(return_X_y=True)
        y = y.astype(np.int64)
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        tr, te = perm[:N], perm[-k:]
        spe = N // self.bs; self.T = E * spe
        orders = [rng.permutation(N) for _ in range(E)]
        self.bidx = [torch.tensor(orders[t // spe][(t % spe) * self.bs:(t % spe + 1) * self.bs], device=self.dev)
                     for t in range(self.T)]
        self.Xtr = torch.tensor(X[tr], dtype=self.dt, device=self.dev); self.ytr = torch.tensor(y[tr], device=self.dev)
        self.Xte = torch.tensor(X[te], dtype=self.dt, device=self.dev); self.yte = torch.tensor(y[te], device=self.dev)
        self.Xref = None
        g = torch.Generator().manual_seed(1)
        dims = [64] + [cfg['hidden']] * cfg['depth'] + [10]
        P0 = []
        for i in range(len(dims) - 1):
            P0.append(torch.randn(dims[i], dims[i + 1], generator=g, dtype=torch.float64) / np.sqrt(dims[i]))
            P0.append(torch.zeros(dims[i + 1], dtype=torch.float64))
        self.shapes = [p.shape for p in P0]; self.sizes = [p.numel() for p in P0]
        self.th0 = torch.cat([p.reshape(-1) for p in P0]).to(self.dt).to(self.dev)
        act = torch.tanh if cfg['act'] == 'tanh' else torch.relu
        def logits(v, x, xref=None):
            ps = [q.reshape(s) for q, s in zip(torch.split(v, self.sizes), self.shapes)]
            h = x
            for i in range(0, len(ps) - 2, 2):
                h = act(h @ ps[i] + ps[i + 1])
            return h @ ps[-2] + ps[-1]
        self.logits = logits

    # ---- functional ResNet-9 ----
    def _build_cnn(self, cfg, E, data_root):
        N, k, m = self.N, self.k, cfg['width']
        rng = np.random.RandomState(0)
        if cfg['data'] == 'cifar':
            Xa, ya, Xb, yb = load_cifar_bin(data_root)
            tr, te = balanced_subset(ya, N, rng), balanced_subset(yb, k, np.random.RandomState(1))
            mu, sd = Xa[tr].mean((0, 2, 3), keepdims=True), Xa[tr].std((0, 2, 3), keepdims=True)
            Xtr, ytr, Xte, yte = (Xa[tr] - mu) / sd, ya[tr], (Xb[te] - mu) / sd, yb[te]
        else:
            Xtr, ytr = synth_images(N, rng); Xte, yte = synth_images(k, np.random.RandomState(1))
        spe = N // self.bs; self.T = E * spe
        orders = [rng.permutation(N) for _ in range(E)]
        self.bidx = [torch.tensor(orders[t // spe][(t % spe) * self.bs:(t % spe + 1) * self.bs], device=self.dev)
                     for t in range(self.T)]
        self.Xtr = torch.tensor(Xtr, dtype=self.dt, device=self.dev); self.ytr = torch.tensor(ytr, device=self.dev)
        self.Xte = torch.tensor(Xte, dtype=self.dt, device=self.dev); self.yte = torch.tensor(yte, device=self.dev)
        self.Xref = self.Xtr[:min(500, N)] if cfg['norm'] == 'bn' else None
        c = [max(4, int(round(80 * m))), int(round(160 * m)), int(round(320 * m)), int(round(640 * m))]
        convs = [(3, c[0]), (c[0], c[1]), (c[1], c[1]), (c[1], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[3]), (c[3], c[3])]
        g = torch.Generator().manual_seed(1)
        P0 = []
        for ci, co in convs:
            P0 += [torch.randn(co, ci, 3, 3, generator=g, dtype=torch.float64) * math.sqrt(2 / (9 * ci)),
                   torch.ones(co, dtype=torch.float64), torch.zeros(co, dtype=torch.float64)]
        P0 += [torch.randn(c[3], 10, generator=g, dtype=torch.float64) / math.sqrt(c[3]), torch.zeros(10, dtype=torch.float64)]
        self.shapes = [p.shape for p in P0]; self.sizes = [p.numel() for p in P0]
        self.th0 = torch.cat([p.reshape(-1) for p in P0]).to(self.dt).to(self.dev)
        norm = cfg['norm']

        def nrm(h, hr, gm, bt):
            C = h.shape[1]
            if norm == 'gn':
                gg = math.gcd(C, 8)
                return F.group_norm(h, gg, gm, bt), None
            src = h if hr is None else hr
            mean = src.mean((0, 2, 3), keepdim=True); var = src.var((0, 2, 3), unbiased=False, keepdim=True)
            f = lambda z: (z - mean) / torch.sqrt(var + 1e-5) * gm[None, :, None, None] + bt[None, :, None, None]
            return f(h), (None if hr is None else f(hr))

        def block(h, hr, ps, i, pool):
            W, gm, bt = ps[3 * i], ps[3 * i + 1], ps[3 * i + 2]
            h = F.conv2d(h, W, padding=1); hr = None if hr is None else F.conv2d(hr, W, padding=1)
            h, hr = nrm(h, hr, gm, bt)
            h = F.gelu(h); hr = None if hr is None else F.gelu(hr)
            if pool:
                h = F.max_pool2d(h, 2); hr = None if hr is None else F.max_pool2d(hr, 2)
            return h, hr

        def logits(v, x, xref=None):
            ps = [q.reshape(s) for q, s in zip(torch.split(v, self.sizes), self.shapes)]
            h, hr = block(x, xref, ps, 0, False)
            h, hr = block(h, hr, ps, 1, True)
            r, rr = block(h, hr, ps, 2, False); r, rr = block(r, rr, ps, 3, False)
            h = h + r; hr = None if hr is None else hr + rr
            h, hr = block(h, hr, ps, 4, True)
            h, hr = block(h, hr, ps, 5, True)
            r, rr = block(h, hr, ps, 6, False); r, rr = block(r, rr, ps, 7, False)
            h = h + r
            h = torch.amax(h, dim=(2, 3))
            return 0.125 * (h @ ps[-2] + ps[-1])
        self.logits = logits

    # ------------------------------------------------------------------ losses and test functions
    def loss_vec(self, v, xb, yb):
        lo = self.logits(v, xb)
        return torch.logsumexp(lo, -1) - lo.gather(-1, yb[:, None])[:, 0]

    def margin_vec(self, v, X, Y):
        lo = self.logits(v, X, self.Xref)
        ly = lo.gather(-1, Y[:, None])[:, 0]
        other = lo.masked_fill(Y[:, None] == torch.arange(lo.shape[-1], device=lo.device), -1e30)
        return ly - torch.logsumexp(other, -1)

    def margins(self, th):
        return self.margin_vec(th, self.Xte, self.yte)

    def test_grads(self, rows):
        """gradient of the margin of each test point in `rows` at theta (c x P)."""
        th = self._thT
        X, Y = self.Xte[rows], self.yte[rows]
        f = lambda p, x, yy: self.margin_vec(p, x[None], yy[None])[0]
        return vmap(grad(f), in_dims=(None, 0, 0))(th, X, Y)

    # ------------------------------------------------------------------ training
    def _wgrad(self, th, idx, w):
        """grad of sum_i w_i l_i / bs over the batch idx (micro-batched; exact for per-sample losses)."""
        gtot = None
        for m0 in range(0, self.bs, self.micro):
            ii = idx[m0:m0 + self.micro]
            gg = grad(lambda p: (w[m0:m0 + self.micro] * self.loss_vec(p, self.Xtr[ii], self.ytr[ii])).sum() / self.bs)(th)
            gtot = gg if gtot is None else gtot + gg
        return gtot

    def train(self, w, ckpt_every=None, t_stop=None):
        """one model with data weights w (N,). Returns theta_T and, if asked, checkpoints {t: (theta_t, v_t)} on CPU."""
        th, v = self.th0.clone(), torch.zeros_like(self.th0)
        ck = {}
        T = self.T if t_stop is None else t_stop
        for t in range(T):
            if ckpt_every and t % ckpt_every == 0:
                ck[t] = (th.cpu(), v.cpu())
            idx = self.bidx[t]
            v = self.mom * v + self._wgrad(th, idx, w[idx]) + self.wd * th
            th = th - self.eta[t] * v
        return (th, ck) if ckpt_every else th

    def train_many_mlp(self, Wm):
        """legacy vmapped retrains (MLP points): identical arithmetic to ceiling_generate/estimator_test_generate trainM."""
        bs = self.bs
        wg = vmap(grad(lambda p, xb, yb, w: (w * self.loss_vec(p, xb, yb)).sum() / bs), in_dims=(0, None, None, 0))
        M = Wm.shape[0]
        th = self.th0[None].repeat(M, 1); v = torch.zeros_like(th)
        for t in range(self.T):
            idx = self.bidx[t]
            v = self.mom * v + wg(th, self.Xtr[idx], self.ytr[idx], Wm[:, idx])
            if self.wd: v = v + self.wd * th
            th = th - self.eta[t] * v
        return th

    def fit_reference(self, seg):
        """train with all weights 1, keep checkpoints every `seg` steps; sets theta_T."""
        thT, ck = self.train(torch.ones(self.N, dtype=self.dt, device=self.dev), ckpt_every=seg)
        self._thT, self._ck, self._seg = thT, ck, seg
        return thT

    # ------------------------------------------------------------------ exact replay
    def _step_jvp(self, th, idx, C):
        """for each row c of C: (c . grad l_i(th)) for all i in the batch, and (H_batch c). One jvp per micro-batch."""
        Ds, Hc = [], None
        for m0 in range(0, self.bs, self.micro):
            ii = idx[m0:m0 + self.micro]
            xb, yb = self.Xtr[ii], self.ytr[ii]
            fn = lambda p: (self.loss_vec(p, xb, yb), grad(lambda q: self.loss_vec(q, xb, yb).sum() / self.bs)(p))
            D, H = vmap(lambda u: jvp(fn, (th,), (u,))[1])(C)
            Ds.append(D); Hc = H if Hc is None else Hc + H
        return torch.cat(Ds, 1), Hc

    def replay_rows(self, rows, t_limit=None):
        """exact rows of Y for the test points `rows` (needs fit_reference first). Returns (c x N) float64 numpy.
        t_limit: only replay the last t_limit steps (used for timing)."""
        a = self.test_grads(rows).detach(); b = torch.zeros_like(a)
        Yc = torch.zeros(len(rows), self.N, dtype=torch.float64, device=self.dev)
        starts = sorted(self._ck)
        done = 0
        for s0 in reversed(starts):
            s1 = min(s0 + self._seg, self.T)
            th, v = self._ck[s0][0].to(self.dev), self._ck[s0][1].to(self.dev)
            ths = []
            ones = torch.ones(self.bs, dtype=self.dt, device=self.dev)
            for t in range(s0, s1):                     # recompute the segment exactly as in training
                ths.append(th)
                v = self.mom * v + self._wgrad(th, self.bidx[t], ones) + self.wd * th
                th = th - self.eta[t] * v
            for t in range(s1 - 1, s0 - 1, -1):
                c = b - self.eta[t] * a
                D, Hc = self._step_jvp(ths[t - s0], self.bidx[t], c)
                Yc[:, self.bidx[t]] += D.to(torch.float64) / self.bs
                a = a + Hc + self.wd * c
                b = self.mom * c
                done += 1
                if t_limit and done >= t_limit:
                    return Yc.cpu().numpy()
        return Yc.cpu().numpy()


def make_masks(N, alpha, M, seed):
    r = np.random.RandomState(seed)
    masks = np.ones((M, N))
    for m in range(M):
        masks[m, r.choice(N, int(round(alpha * N)), replace=False)] = 0
    return masks
