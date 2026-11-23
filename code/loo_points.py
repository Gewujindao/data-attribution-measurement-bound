"""Leave-one-out study: models for the sign-of-kappa derivation checks.

Reused architecture-study models (built by arch_points.P6, exact rows already on disk): b, b_s1, w16, w128, w128_s1.
New models (built here by P7, same training loop / exact replay of ladder_core.Point):
  bc, bc_s1        breast_cancer (569 x 30, 2 classes), tanh MLP 30-32-32-2, N=400, k=100, 50 epochs (T=400)
  rare0, rare0_s1  digits, tanh MLP 64-32-32-10, N=1000 with at most 10 training examples of class 0;
                   test rows 0..99 = class-0 points (covariate shift), rows 100..199 = ordinary test points (control)
Deletion specs (alpha, M, mask seed) are listed in SPECS; the same trained model serves every spec and the LOO set.
Seed s>0 changes init (generator 1+100s) and batch order (RandomState(1000+s)) only, as in the architecture study.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_core import Point, POINTS
from arch_points import P6          # registers the architecture-study names in POINTS

NEW = dict(kind='w7', act='tanh', depth=2, hidden=32, bs=50, lr=0.05, mom=0.9, wd=0.0, sched='const', tb=50)
for s in (0, 1):
    sfx = '' if s == 0 else '_s1'
    POINTS['bc' + sfx] = dict(NEW, data='breast', N=400, k=100, epochs=50, seed=s, truth=((0.01, 200, 17),))
    POINTS['rare0' + sfx] = dict(NEW, data='digits_rare', rare_class=0, n_rare=10, kA=100, kB=100,
                                 N=1000, k=200, epochs=20, seed=s, truth=((0.01, 200, 17),))

# model name -> list of (spec tag, alpha, M, mask seed); 'loo' = all N single deletions
SPECS = {'b': [('a10', 0.10, 200, 29)], 'b_s1': [('a10', 0.10, 200, 29)],
         'w128': [('a10', 0.10, 200, 29)], 'w128_s1': [('a10', 0.10, 200, 29)], 'w16': [],
         'bc': [('a01', 0.01, 200, 17)], 'bc_s1': [('a01', 0.01, 200, 17)],
         'rare0': [('a01', 0.01, 200, 17)], 'rare0_s1': [('a01', 0.01, 200, 17)]}
REUSED = ('b', 'b_s1', 'w16', 'w128', 'w128_s1')


class P7(Point):
    def _build_cnn(self, cfg, E, data_root):          # Point.__init__ calls this for every kind other than 'mlp'
        N, k, s = self.N, self.k, cfg['seed']
        rng = np.random.RandomState(0)
        if cfg['data'] == 'breast':
            from sklearn.datasets import load_breast_cancer
            X, y = load_breast_cancer(return_X_y=True)
            perm = rng.permutation(len(y))
            tr, te = perm[:N], perm[-k:]
        else:
            from sklearn.datasets import load_digits
            X, y = load_digits(return_X_y=True)
            perm = rng.permutation(len(y))
            c, nr = cfg['rare_class'], cfg['n_rare']
            tr, nc = [], 0
            for i in perm:
                if len(tr) == N:
                    break
                if y[i] == c:
                    if nc == nr:
                        continue
                    nc += 1
                tr.append(i)
            trs = set(tr)
            rest = [i for i in perm if i not in trs]
            A = [i for i in rest if y[i] == c][:cfg['kA']]
            As = set(A)
            Bs = [i for i in rest if i not in As][:cfg['kB']]
            tr, te = np.array(tr), np.array(A + Bs)
            assert len(te) == k
        y = y.astype(np.int64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        spe = N // self.bs; self.T = E * spe
        orng = rng if s == 0 else np.random.RandomState(1000 + s)
        orders = [orng.permutation(N) for _ in range(E)]
        self.bidx = [torch.tensor(orders[t // spe][(t % spe) * self.bs:(t % spe + 1) * self.bs], device=self.dev)
                     for t in range(self.T)]
        self.Xtr = torch.tensor(X[tr], dtype=self.dt, device=self.dev); self.ytr = torch.tensor(y[tr], device=self.dev)
        self.Xte = torch.tensor(X[te], dtype=self.dt, device=self.dev); self.yte = torch.tensor(y[te], device=self.dev)
        self.Xref = None
        self.tr_idx, self.te_idx = tr, te
        g = torch.Generator().manual_seed(1 + 100 * s)
        dims = [X.shape[1]] + [cfg['hidden']] * cfg['depth'] + [int(y.max()) + 1]
        P0 = []
        for i in range(len(dims) - 1):
            P0.append(torch.randn(dims[i], dims[i + 1], generator=g, dtype=torch.float64) / np.sqrt(dims[i]))
            P0.append(torch.zeros(dims[i + 1], dtype=torch.float64))
        self.shapes = [p.shape for p in P0]; self.sizes = [p.numel() for p in P0]
        self.th0 = torch.cat([p.reshape(-1) for p in P0]).to(self.dt).to(self.dev)

        def logits(v, x, xref=None):
            ps = [q.reshape(sh) for q, sh in zip(torch.split(v, self.sizes), self.shapes)]
            h = x
            for i in range(0, len(ps) - 2, 2):
                h = torch.tanh(h @ ps[i] + ps[i + 1])
            return h @ ps[-2] + ps[-1]
        self.logits = logits


def build(name):
    return (P6 if name in REUSED else P7)(name, 'cpu', torch.float64)
