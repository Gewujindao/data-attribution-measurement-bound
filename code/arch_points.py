"""Architecture study: one-factor-at-a-time digits settings for the sign of kappa.
Reuses the exact replay of ladder_core.Point; only the data/model builder is new.
Setting names: b, w16, w64, w128, w256, d1, d3, d4, bn, gn, cnn, e5, e60, lr0.02, lr0.1, wd1e-3, relu; suffix _s1 = seed 1.
Seed 0 of the base equals the base setting of baseline_generate.py (same split, batch order and init); seed s>0 changes init and batch order only.
"""
import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_core import Point, POINTS

BASE = dict(kind='w6', arch='mlp', act='tanh', depth=2, hidden=32, norm='none', N=1000, k=100, epochs=20, bs=50,
            lr=0.05, mom=0.9, wd=0.0, sched='const', truth=((0.01, 200, 17),), tb=50, seed=0)
VARIANTS = {'b': {}, 'w16': dict(hidden=16), 'w64': dict(hidden=64), 'w128': dict(hidden=128), 'w256': dict(hidden=256),
            'd1': dict(depth=1), 'd3': dict(depth=3), 'd4': dict(depth=4), 'bn': dict(norm='bn'), 'gn': dict(norm='gn'),
            'cnn': dict(arch='cnn'), 'e5': dict(epochs=5), 'e60': dict(epochs=60), 'lr0.02': dict(lr=0.02),
            'lr0.1': dict(lr=0.1), 'wd1e-3': dict(wd=1e-3), 'relu': dict(act='relu')}
for v, ch in VARIANTS.items():
    for s in (0, 1):
        POINTS[v if s == 0 else f'{v}_s1'] = dict(BASE, **ch, seed=s)


class P6(Point):
    def _build_cnn(self, cfg, E, data_root):          # called by Point.__init__ for every kind other than 'mlp'
        from sklearn.datasets import load_digits
        N, k, s = self.N, self.k, cfg['seed']
        X, y = load_digits(return_X_y=True)
        y = y.astype(np.int64)
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        tr, te = perm[:N], perm[-k:]
        spe = N // self.bs; self.T = E * spe
        orng = rng if s == 0 else np.random.RandomState(1000 + s)
        orders = [orng.permutation(N) for _ in range(E)]
        self.bidx = [torch.tensor(orders[t // spe][(t % spe) * self.bs:(t % spe + 1) * self.bs], device=self.dev)
                     for t in range(self.T)]
        self.Xtr = torch.tensor(X[tr], dtype=self.dt, device=self.dev); self.ytr = torch.tensor(y[tr], device=self.dev)
        self.Xte = torch.tensor(X[te], dtype=self.dt, device=self.dev); self.yte = torch.tensor(y[te], device=self.dev)
        self.Xref = self.Xtr[:500] if cfg['norm'] == 'bn' else None
        g = torch.Generator().manual_seed(1 + 100 * s)
        act = torch.tanh if cfg['act'] == 'tanh' else torch.relu
        norm = cfg['norm']
        P0 = []
        if cfg['arch'] == 'mlp':
            dims = [64] + [cfg['hidden']] * cfg['depth'] + [10]
            for i in range(len(dims) - 1):
                P0.append(torch.randn(dims[i], dims[i + 1], generator=g, dtype=torch.float64) / np.sqrt(dims[i]))
                P0.append(torch.zeros(dims[i + 1], dtype=torch.float64))
                if norm != 'none' and i < len(dims) - 2:
                    P0 += [torch.ones(dims[i + 1], dtype=torch.float64), torch.zeros(dims[i + 1], dtype=torch.float64)]
        else:
            for ci, co in ((1, 8), (8, 16), (16, 16)):
                P0 += [torch.randn(co, ci, 3, 3, generator=g, dtype=torch.float64) * math.sqrt(1 / (9 * ci)),
                       torch.zeros(co, dtype=torch.float64)]
            P0 += [torch.randn(256, 10, generator=g, dtype=torch.float64) / 16.0, torch.zeros(10, dtype=torch.float64)]
        self.shapes = [p.shape for p in P0]; self.sizes = [p.numel() for p in P0]
        self.th0 = torch.cat([p.reshape(-1) for p in P0]).to(self.dt).to(self.dev)

        def nrm(h, hr, gm, bt):          # h: (n, C)
            if norm == 'gn':
                n, C = h.shape
                z = h.reshape(n, 4, C // 4)
                z = (z - z.mean(-1, keepdim=True)) / torch.sqrt(z.var(-1, unbiased=False, keepdim=True) + 1e-5)
                return z.reshape(n, C) * gm + bt, None
            src = h if hr is None else hr
            mu, var = src.mean(0, keepdim=True), src.var(0, unbiased=False, keepdim=True)
            f = lambda z: (z - mu) / torch.sqrt(var + 1e-5) * gm + bt
            return f(h), (None if hr is None else f(hr))

        def logits(v, x, xref=None):
            ps = [q.reshape(sh) for q, sh in zip(torch.split(v, self.sizes), self.shapes)]
            if cfg['arch'] == 'cnn':
                h = x.reshape(-1, 1, 8, 8)
                h = act(F.conv2d(h, ps[0], ps[1], padding=1))
                h = act(F.conv2d(h, ps[2], ps[3], padding=1))
                h = F.avg_pool2d(h, 2)
                h = act(F.conv2d(h, ps[4], ps[5], padding=1))
                return h.reshape(h.shape[0], -1) @ ps[6] + ps[7]
            h, hr, i = x, xref, 0
            step = 4 if norm != 'none' else 2
            for _ in range(cfg['depth']):
                W, b = ps[i], ps[i + 1]
                h = h @ W + b; hr = None if hr is None else hr @ W + b
                if norm != 'none':
                    h, hr = nrm(h, hr, ps[i + 2], ps[i + 3])
                h = act(h); hr = None if hr is None else act(hr)
                i += step
            return h @ ps[i] + ps[i + 1]
        self.logits = logits
