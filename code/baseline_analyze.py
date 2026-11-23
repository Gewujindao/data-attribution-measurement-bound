"""Baselines: LDS, structure quantities and oracle curves from baseline_generate.py output.
Usage: python baseline_analyze.py <npz>
"""
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans

D = np.load(sys.argv[1])
Yep, ytr, yte = D['Yep'], D['ytr'], D['yte']
Y = Yep.sum(1)
k, E, N = Yep.shape
T = {'1%': (D['m01'], D['d01']), '5%': (D['m05'], D['d05'])}
print(f'k={k} N={N} E={E} P={int(D["P"])} train acc {float(D["tr_acc"]):.3f} test acc {float(D["acc"]):.3f}')

def lds_per(Yh, key):
    m, d = T[key]
    pred = -(1 - m) @ Yh.T          # (M,k)
    rp, rd = rankdata(pred, axis=0), rankdata(d, axis=0)
    rp -= rp.mean(0); rd -= rd.mean(0)
    return (rp * rd).sum(0) / np.sqrt((rp ** 2).sum(0) * (rd ** 2).sum(0) + 1e-300)
def lds(Yh, key='1%'):
    return np.nanmean(lds_per(Yh, key))

rn = np.linalg.norm(Y, axis=1)
q = rn <= np.quantile(rn, 0.25)
print('\n== exact metagradient Y ==')
for key in T:
    p = lds_per(Y, key)
    print(f'LDS {key}: mean {p.mean():.3f}  min {p.min():.3f}  bottom-quartile-norm {p[q].mean():.3f}')
# linearity check: actual change vs first-order prediction magnitude
for key in T:
    m, d = T[key]
    pred = -(1 - m) @ Y.T
    print(f'  {key}: Pearson(pred,actual) pooled-per-point mean {np.mean([np.corrcoef(pred[:,t], d[:,t])[0,1] for t in range(k)]):.3f}; '
          f'|actual| median {np.median(np.abs(d)):.2e}, slope median {np.median((pred*d).sum(0)/(pred**2).sum(0)):.3f}')

print('\n== cheap baselines ==')
if D['GV'].size > 1:
    GV, GtV, lam = D['GV'], D['GtV'], D['lam']
    print(f'grad dot (endpoint, identity Hessian): 1% {lds(-GtV @ GV.T):.3f}  5% {lds(-GtV @ GV.T, "5%"):.3f}')
    lmax = np.abs(lam).max()
    best = None
    for kind in ('abs', 'clip'):
        for rel in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            ev = (np.abs(lam) if kind == 'abs' else np.maximum(lam, 0)) + rel * lmax
            Yh = -(GtV / ev) @ GV.T
            v1, v5 = lds(Yh), lds(Yh, '5%')
            print(f'  IF exact Hessian [{kind}] damping {rel:g}*lmax: 1% {v1:.3f}  5% {v5:.3f}')
            if best is None or v1 > best[0]:
                best = (v1, v5, kind, rel)
    print(f'best endpoint IF (oracle-tuned damping): 1% {best[0]:.3f} 5% {best[1]:.3f} ({best[2]}, {best[3]:g})')
print(f'TracIn (test grad at end, train grads along path): 1% {lds(D["Ytr_end"]):.3f} 5% {lds(D["Ytr_end"], "5%"):.3f}')
print(f'TracIn (both grads at each step): 1% {lds(D["Ytr_ck"]):.3f} 5% {lds(D["Ytr_ck"], "5%"):.3f}')

print('\n== b1/b2 spectra ==')
def spec(A, name):
    s = np.linalg.svd(A, compute_uv=False)
    e = s ** 2
    pr = e.sum() ** 2 / (e ** 2).sum()
    p = e / e.sum(); er = np.exp(-(p * np.log(p + 1e-300)).sum())
    r90 = int(np.searchsorted(np.cumsum(p), 0.9) + 1)
    r99 = int(np.searchsorted(np.cumsum(p), 0.99) + 1)
    print(f'{name}: PR {pr:.1f}  entropy-erank {er:.1f}  rank90 {r90}  rank99 {r99}  (k={k})  s1/s10 {s[0]/s[9]:.1f}')
cn = np.linalg.norm(Y, axis=0)
Rn = Y / rn[:, None]
Cn = Y / cn[None]
spec(Y, 'raw'); spec(Rn, 'row-normalized'); spec(Cn, 'col-normalized')
print(f'row norm span log10(max/min) {np.log10(rn.max()/rn.min()):.2f}; col norm span {np.log10(cn.max()/cn.min()):.2f}; '
      f'col norm top10% share of ||Y||^2 {np.sort(cn**2)[::-1][:N//10].sum()/(cn**2).sum():.2f}')

print('\n== b3 correlations ==')
C = Rn @ Rn.T
same = yte[:, None] == yte[None]
off = ~np.eye(k, dtype=bool)
print(f'test-test |cos|: same class median {np.median(np.abs(C[same & off])):.3f}, diff class median {np.median(np.abs(C[~same])):.3f}; '
      f'same-class mean cos {C[same & off].mean():.3f}')
Cc = Cn.T @ Cn
samec = ytr[:, None] == ytr[None]
offc = ~np.eye(N, dtype=bool)
print(f'train-train |cos| (over test points): same class median {np.median(np.abs(Cc[samec & offc])):.3f}, diff {np.median(np.abs(Cc[~samec])):.3f}')

print('\n== b4 concentration ==')
Y2 = np.sort(Y ** 2, axis=1)[:, ::-1]
tot = Y2.sum(1)
for f in (0.01, 0.05, 0.10, 0.25):
    print(f'  top {f:.0%} train points hold {np.mean(Y2[:, :max(1,int(f*N))].sum(1)/tot):.2f} of row L2^2 (mean over test points)')
cm = np.array([(Y[t, ytr == yte[t]] ** 2).sum() / (Y[t] ** 2).sum() for t in range(k)])
print(f'  same-class train points hold {cm.mean():.2f} of row L2^2 (class share {np.mean(ytr[:,None]==np.arange(10)):.2f})')
top = np.argsort(-np.abs(Y), axis=1)[:, :20]
jac = [len(set(top[i]) & set(top[j])) / len(set(top[i]) | set(top[j])) for i in range(k) for j in range(i + 1, k) if yte[i] == yte[j]]
print(f'  top-20 train set Jaccard between same-class test points: mean {np.mean(jac):.3f}')

print('\n== b5 training stage ==')
en = (Yep ** 2).sum((0, 2)); en /= en.sum()
print('  energy share by epoch:', ' '.join(f'{x:.3f}' for x in en))
for j in (1, 2, 4, 6, 10):
    print(f'  first {j} epochs only: LDS 1% {lds(Yep[:, :j].sum(1)):.3f} | last {j} epochs only: {lds(Yep[:, -j:].sum(1)):.3f}')
print('  single-epoch LDS 1%:', ' '.join(f'{lds(Yep[:, e]):.2f}' for e in range(E)))
print('  cos(Y^(e), Y) per epoch:', ' '.join(f'{np.sum(Yep[:,e]*Y)/np.linalg.norm(Yep[:,e])/np.linalg.norm(Y):.2f}' for e in range(E)))

print('\n== a) oracle curves: LDS 1% (5%) vs B ==')
Bs = [1, 2, 5, 10, 20, 30, 50, 80, k]
U, S, Vt = np.linalg.svd(Rn, full_matrices=False)
Uc, Sc, Vtc = np.linalg.svd(Cn, full_matrices=False)
Ur, Sr, Vtr = np.linalg.svd(Y, full_matrices=False)
def curve(name, f):
    out = []
    for B in Bs:
        Yh = f(B); out.append((B, lds(Yh), lds(Yh, '5%')))
    print(f'{name}: ' + '  '.join(f'B={b}:{v1:.3f}({v5:.3f})' for b, v1, v5 in out))
    b80 = next((b for b, v1, _ in out if v1 >= 0.8), None)
    print(f'   B80 = {b80}  (B/k = {b80/k if b80 else float("nan"):.2f})')
curve('a1 test-side spherical SVD oracle', lambda B: (U[:, :B] * S[:B]) @ Vt[:B])
curve('   raw truncated SVD', lambda B: (Ur[:, :B] * Sr[:B]) @ Vtr[:B])
curve('a2 train-side col-normalized SVD oracle', lambda B: ((Uc[:, :B] * Sc[:B]) @ Vtc[:B]) * cn[None])
def sparse(s):
    idx = np.argsort(-np.abs(Y), axis=1)[:, :s]
    Z = np.zeros_like(Y); np.put_along_axis(Z, idx, np.take_along_axis(Y, idx, 1), 1); return Z
print('a3 sparse oracle (top-s per row): ' + '  '.join(f's={s}:{lds(sparse(s)):.3f}' for s in (5, 10, 20, 50, 100, 200, 500, N)))

cord = np.argsort(-cn)
def colsub(B):
    Z = np.zeros_like(Y); Z[:, cord[:B]] = Y[:, cord[:B]]; return Z
print('a6 column-subset oracle (B forward replays on the B largest-norm training points): ' +
      '  '.join(f'B={B}:{lds(colsub(B)):.3f}' for B in (10, 20, 50, 100, 200, 500)))

print('a7 scaling with k: spherical-SVD oracle on random subsets of k\' test points')
rs = np.random.RandomState(3)
for kk in [x for x in (25, 50, 100, 200, 300) if x <= k]:
    sub = np.sort(rs.choice(k, kk, replace=False))
    Rs = Rn[sub]
    us, ss, vts = np.linalg.svd(Rs, full_matrices=False)
    e = ss ** 2; pr = e.sum() ** 2 / (e ** 2).sum()
    def lds_sub(Yh):
        full = np.zeros_like(Y); full[sub] = Yh
        return np.nanmean(lds_per(full, '1%')[sub])
    ex = lds_sub(Y[sub])
    res = []
    for frac in (0.1, 0.2, 0.3, 0.5, 0.8):
        B = max(1, int(round(frac * kk)))
        res.append((frac, B, lds_sub((us[:, :B] * ss[:B]) @ vts[:B])))
    print(f'   k\'={kk}: PR(row-norm) {pr:.1f} (PR/k\' {pr/kk:.2f}); exact {ex:.3f}; ' +
          '  '.join(f'B={B}({fr:.0%}):{v:.3f}' for fr, B, v in res))

def cluster_pred(lab, weighted):
    Yh = np.zeros_like(Y)
    for c in np.unique(lab):
        ii = lab == c
        row = (Rn[ii] if weighted else Y[ii]).sum(0)
        Yh[ii] = row
    return Yh
print('a4 cluster oracle (k-means on true normalized rows; centroid = one replay of the cluster-sum test function):')
for B in (5, 10, 20, 30, 50):
    lab = KMeans(B, n_init=5, random_state=0).fit_predict(Rn)
    print(f'   B={B}: norm-weighted {lds(cluster_pred(lab, True)):.3f}  raw-sum {lds(cluster_pred(lab, False)):.3f}')
print(f'a4 cheap clustering by test label (B=10): raw-sum {lds(cluster_pred(yte, False)):.3f}')
if D['GtV'].size > 1:
    F = D['GtV']; F = F / np.linalg.norm(F, axis=1, keepdims=True)
    for B in (10, 20, 30, 50):
        lab = KMeans(B, n_init=5, random_state=0).fit_predict(F)
        print(f'   cheap k-means on endpoint test gradients B={B}: raw-sum {lds(cluster_pred(lab, False)):.3f}')
# hybrid oracle: low-rank test side + per-row residual sparsity
print('a5 combined oracle (rank-B spherical SVD + top-s residual per row) LDS 1%:')
for B in (5, 10, 20):
    L = (U[:, :B] * S[:B]) @ Vt[:B]
    R = Rn - L
    for s in (10, 50):
        idx = np.argsort(-np.abs(R), axis=1)[:, :s]
        Z = np.zeros_like(R); np.put_along_axis(Z, idx, np.take_along_axis(R, idx, 1), 1)
        print(f'   B={B} s={s}: {lds(L + Z):.3f}')
