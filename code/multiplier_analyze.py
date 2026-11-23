"""Multiplier measurements: all readings and the hit / MISS check of each prediction.
  python multiplier_analyze.py <cache/multiplier> <cache/arch> <results/multiplier>
Q1 (w16, b, w128 x 2 seeds): weighted decomposition of the element-wise multiplier m_ji = G_ji / y_ji with weights y_ji^2 over the
   top-30% columns C: SS_T = sum (G - mbar y)^2, between-points SS_B = sum_i W_i (mhat_i - mbar)^2, within SS_W = sum (G - mhat_i y)^2;
   point-level predictors (terminal RIF factor 1/(1-h) for lambda grid, mem/SI, mem, Psi(mem)) by weighted affine fit of mhat on x;
   late share: fit G - mhat y ~ c (y_late - lambda_i y) on the within part.
Q2: column concentration n80 / participation ratio by mhat and mem quartiles (iid models; rare0 groups A and B).
Q3: m<1 energy share, small-denominator / sign checks, half-weight convexity, width / init sweeps, 10% sub-additivity S_add.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
SCR, ARCH, RES = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(RES, exist_ok=True)
out, J = [], {}
def say(s=''):
    print(s, flush=True); out.append(s)
def ld(nm, pat):
    return sorted(glob.glob(os.path.join(SCR, f'{nm}_{pat}')))
def load(nm):
    d = dict(Y=np.concatenate([np.load(f) for f in ld(nm, 'Y_*.npy')]))
    L = [np.load(f) for f in ld(nm, 'loo_*.npz')]
    d['G'] = -np.concatenate([x['d'] for x in L]).T; d['mem'] = np.concatenate([x['mem'] for x in L])
    H = [np.load(f) for f in ld(nm, 'half_*.npz')]
    if len(H) == 20:
        d['Gh'] = -np.concatenate([x['d'] for x in H]).T
    if os.path.exists(os.path.join(SCR, f'{nm}_ggn.npz')) and len(ld(nm, 'si_*.npy')) == 6 and len(ld(nm, 'split_*.npz')) == 2:
        g = np.load(os.path.join(SCR, f'{nm}_ggn.npz')); d['z'], d['Kd'] = g['z'], g['Kd']
        d['sicols'] = np.load(os.path.join(SCR, f'{nm}_sicols.npy'))
        d['SI'] = np.concatenate([np.load(f) for f in ld(nm, 'si_*.npy')])
        d['YL'] = np.concatenate([np.load(f)['late'] for f in ld(nm, 'split_*.npz')])
    if len(ld(nm, 'a10_[0-9][0-9][0-9].npy')) == 4:
        d['m10'] = np.load(os.path.join(SCR, f'{nm}_a10_masks.npy'))
        d['d10'] = np.concatenate([np.load(f) for f in ld(nm, 'a10_[0-9][0-9][0-9].npy')])
    if os.path.exists(os.path.join(ARCH, f'{nm}_truth.npz')):
        t = np.load(os.path.join(ARCH, f'{nm}_truth.npz')); d['m01'], d['d01'] = t['m01'], t['d01']
    return d
def wfit(x, yv, w):
    """weighted affine fit; returns R^2 (weighted) of yv on x."""
    X = np.stack([np.ones_like(x), x], 1); Wd = w[:, None]
    beta = np.linalg.lstsq(X * np.sqrt(Wd), yv * np.sqrt(w), rcond=None)[0]
    r = yv - X @ beta; mu = np.sum(w * yv) / w.sum()
    return 1 - np.sum(w * r ** 2) / np.sum(w * (yv - mu) ** 2)
def cc(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A * A).sum(0) * (B * B).sum(0))
def slope(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return (x * y).sum(0) / (x * x).sum(0)
psi = lambda x: np.where(np.abs(x) < 1e-6, 1.0, x / (-np.expm1(-np.clip(x, -50, 50))))
def mfrac(d):
    Y, G = d['Y'], d['G']; colE = (Y * Y).sum(0); mh = (G * Y).sum(0) / colE
    return float(colE[mh < 1].sum() / colE.sum()), mh, colE

MODELS = [f'{b}{s}' for b in ('w16', 'b', 'w64', 'w128', 'b_i05', 'b_i2') for s in ('', '_s1')]
D = {nm: load(nm) for nm in MODELS + ['rare0', 'rare0_s1'] if len(ld(nm, 'loo_*.npz')) == 20}
LAM_ES = 1000 / (0.05 / (1 - 0.9) * 400)
GRID = 10.0 ** np.arange(-3, 3.01, 0.5)

# ---------------------------------------------------------------- Q1
say('Q1  weighted decomposition of m_ji = G/y over top-30% columns (weights y^2); shares of SS_T unless noted')
say('model    SS_W/SS_T | point-level share of SS_T (share of SS_B): RIF(lam_ES)  RIF(best lam)  mem/SI  mem  Psi(mem) | late: of SS_W  of SS_T | Spearman(mhat, mem/SI) RIF(lam_ES)')
Q1 = {}
def h_of(Kd, lam):
    ev, Q = np.linalg.eigh(Kd); ev = np.clip(ev, 0, None)
    return (Q ** 2) @ (ev / (ev + lam))
for nm in [m for m in MODELS if m in D and 'Kd' in D[m]]:
    d = D[nm]; C = d['sicols']; y, g, yl = d['Y'][:, C], d['G'][:, C], d['YL'][:, C]
    W = (y * y).sum(0); mh = (g * y).sum(0) / W; mbar = (g * y).sum() / (y * y).sum()
    SST = np.sum((g - mbar * y) ** 2); SSB = np.sum(W * (mh - mbar) ** 2); SSW = np.sum((g - mh * y) ** 2)
    ev, Q = np.linalg.eigh(d['Kd']); ev = np.clip(ev, 0, None)
    rif = {lam: 1 / (1 - ((Q ** 2) @ (ev / (ev + lam)))[C]) for lam in list(GRID) + [LAM_ES]}
    r2rif = {lam: wfit(rif[lam], mh, W) for lam in rif}
    mem, SI = d['mem'][C], d['SI']
    ratio = mem / SI
    ok = np.isfinite(ratio) & (np.abs(SI) > 1e-12)
    r2ms = wfit(np.clip(ratio, -10, 10), mh, W)
    r2mem = wfit(mem, mh, W); r2psi = wfit(psi(mem), mh, W)
    lamr = (yl * y).sum(0) / W; rr = g - mh * y; e = yl - lamr * y
    SSL = np.sum(rr * e) ** 2 / np.sum(e * e)
    raw = {k: 1 - np.sum(W * (mh - x) ** 2) / SSB for k, x in (('rif', rif[LAM_ES]), ('mem/SI', np.clip(ratio, -10, 10)), ('psi', psi(mem)))}
    # descriptive: every column scaled to unit energy, so no column dominates
    nrmc = np.sqrt(W); yn, gn, yln = y / nrmc, g / nrmc, yl / nrmc
    mbn = (gn * yn).sum() / (yn * yn).sum()
    SSTn = np.sum((gn - mbn * yn) ** 2); SSBn = np.sum((mh - mbn) ** 2); SSWn = np.sum((gn - mh * yn) ** 2)
    one = np.ones_like(W)
    r2n = dict(rif=wfit(rif[LAM_ES], mh, one), ms=wfit(np.clip(ratio, -10, 10), mh, one), mem=wfit(mem, mh, one))
    en = yln - ((yln * yn).sum(0)) * yn; rn = gn - mh * yn
    SSLn = np.sum(rn * en) ** 2 / np.sum(en * en)
    low = mh < 1
    Q1[nm] = dict(SSWn_T=SSWn / SSTn, r2n=r2n, late_Wn=SSLn / SSWn, SSW_T=SSW / SST, SSB_T=SSB / SST, r2rif=r2rif, r2ms=r2ms, r2mem=r2mem, r2psi=r2psi, late_W=SSL / SSW, late_T=SSL / SST,
                  sp_ms=spearmanr(mh, ratio)[0], sp_rif=spearmanr(mh, rif[LAM_ES])[0], raw=raw, n_low=int(low.sum()),
                  rif_low_med=float(np.median(rif[LAM_ES][low])) if low.any() else np.nan,
                  rif_low_below=float(np.mean(rif[LAM_ES][low] < np.median(rif[LAM_ES]))) if low.any() else np.nan,
                  ms_low_below1=float(np.mean(ratio[low] < 1)) if low.any() else np.nan,
                  rif_med=float(np.median(rif[LAM_ES])), rif_q90=float(np.quantile(rif[LAM_ES], 0.9)),
                  med_mh_over_ms=float(np.median(mh[ok] / ratio[ok])), rifs={str(l): rif[l] for l in rif}, mh=mh, ratio=ratio)
# best lambda chosen on the other seed
for nm in Q1:
    fam = nm.replace('_s1', ''); other = fam if nm.endswith('_s1') else fam + '_s1'
    src = Q1.get(other, Q1[nm])
    lam_best = max(GRID, key=lambda l: src['r2rif'][l])
    q = Q1[nm]; q['lam_best'] = float(lam_best); q['r2rif_best'] = q['r2rif'][lam_best]
    q['rif_low_med_best'] = float(np.median(q['rifs'][str(lam_best)][q['mh'] < 1])) if q['n_low'] else np.nan
    say(f'{nm:8s} {q["SSW_T"]:.3f}     | {q["r2rif"][LAM_ES]*q["SSB_T"]:.3f} ({q["r2rif"][LAM_ES]:.2f})  {q["r2rif_best"]*q["SSB_T"]:.3f} ({q["r2rif_best"]:.2f}, lam {lam_best:g})  '
        f'{q["r2ms"]*q["SSB_T"]:.3f} ({q["r2ms"]:.2f})  {q["r2mem"]*q["SSB_T"]:.3f} ({q["r2mem"]:.2f})  {q["r2psi"]*q["SSB_T"]:.3f} ({q["r2psi"]:.2f}) | '
        f'{q["late_W"]:.3f}  {q["late_T"]:.3f} | {q["sp_ms"]:+.2f} {q["sp_rif"]:+.2f}')
say('  equal column weight (descriptive): ' + '; '.join(f'{nm} SS_W/SS_T {Q1[nm]["SSWn_T"]:.3f}, R2_B RIF {Q1[nm]["r2n"]["rif"]:.2f} mem/SI {Q1[nm]["r2n"]["ms"]:.2f} mem {Q1[nm]["r2n"]["mem"]:.2f}, late of SS_W {Q1[nm]["late_Wn"]:.3f}' for nm in Q1))
say('  absolute (no fit) R^2 on SS_B: ' + '; '.join(f'{nm} RIF {Q1[nm]["raw"]["rif"]:+.2f} mem/SI {Q1[nm]["raw"]["mem/SI"]:+.2f} Psi {Q1[nm]["raw"]["psi"]:+.2f}' for nm in Q1))
say('  RIF(lam_ES) median / 90% quantile over C; on mhat<1 columns: n, median RIF (lam_ES | best), share below the C median; share with mem/SI<1; median mhat/(mem/SI):')
for nm, q in Q1.items():
    say(f'   {nm:8s} RIF med {q["rif_med"]:.3f} q90 {q["rif_q90"]:.3f} | mhat<1: n={q["n_low"]:3d}  RIF {q["rif_low_med"]:.3f} | {q["rif_low_med_best"]:.3f}  '
        f'below-median {q["rif_low_below"]:.2f}  mem/SI<1 {q["ms_low_below1"]:.2f} | median mhat/(mem/SI) {q["med_mh_over_ms"]:.2f}')

# ---------------------------------------------------------------- Q2
say(); say('Q2  column concentration over test rows (top-30% columns of the row group): median n80 and PR, high vs low quartile of mhat and of mem')
say('model    rows | n80 hi-m / lo-m (ratio) | PR hi-m / lo-m (ratio) | n80 hi-mem / lo-mem (ratio) | [rare class-0 training points in C: n, median n80]')
Q2 = {}
def conc(Yg, Gg, mem, extra=None):
    colE = (Yg * Yg).sum(0); C = np.where(colE >= np.quantile(colE, 0.7))[0]
    y, g = Yg[:, C], Gg[:, C]; mh = (g * y).sum(0) / (y * y).sum(0)
    e = np.sort(y * y, axis=0)[::-1] / (y * y).sum(0)
    n80 = (np.cumsum(e, 0) < 0.8).sum(0) + 1
    pr = (y * y).sum(0) ** 2 / (y ** 4).sum(0)
    def ratio(key):
        lo, hi = np.quantile(key, [0.25, 0.75])
        return (np.median(n80[key >= hi]), np.median(n80[key <= lo]), np.median(pr[key >= hi]), np.median(pr[key <= lo]))
    rm, rz = ratio(mh), ratio(mem[C])
    # descriptive: share of each column's energy kept by the SVD oracle W (B = 0.1 x rows)
    Yc = Yg - Yg.mean(1, keepdims=True); nr = np.linalg.norm(Yc, axis=1)
    U, S, Vt = np.linalg.svd(Yc / nr[:, None], full_matrices=False); B = max(1, int(round(0.1 * Yg.shape[0])))
    Uw = (U[:, :B] * S[:B]) @ Vt[:B] * nr[:, None]
    kept = (Uw[:, C] ** 2).sum(0) / (Yc[:, C] ** 2).sum(0)
    lo, hi = np.quantile(mh, [0.25, 0.75])
    rk = (np.median(kept[mh >= hi]), np.median(kept[mh <= lo]))
    ex = None
    if extra is not None:
        sel = np.isin(C, extra)
        ex = (int(sel.sum()), float(np.median(n80[sel])) if sel.any() else np.nan)
    return dict(m=rm, mem=rz, extra=ex, kept=rk)
iid = [m for m in ('w16', 'w16_s1', 'b', 'b_s1', 'w128', 'w128_s1') if m in D]
for nm in iid:
    Q2[nm] = conc(D[nm]['Y'], D[nm]['G'], D[nm]['mem'])
rare_idx = {}
for nm in ('rare0', 'rare0_s1'):
    if nm in D:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from multiplier_points import build
        ytr = build(nm).ytr.numpy(); rare_idx[nm] = np.where(ytr == 0)[0]
        for grp, rows in (('A', slice(0, 100)), ('B', slice(100, 200))):
            Q2[f'{nm}_{grp}'] = conc(D[nm]['Y'][rows], D[nm]['G'][rows], D[nm]['mem'], rare_idx[nm])
for key, q in Q2.items():
    a, b = q['m'], q['mem']
    s = (f'{key:10s} | {a[0]:5.1f} / {a[1]:5.1f} ({a[0]/a[1]:.2f}) | {a[2]:5.1f} / {a[3]:5.1f} ({a[2]/a[3]:.2f}) | {b[0]:5.1f} / {b[1]:5.1f} ({b[0]/b[1]:.2f})')
    s += f' | kept by W hi-m / lo-m {q["kept"][0]:.2f} / {q["kept"][1]:.2f}'
    if q['extra']:
        s += f' | class-0 in C: {q["extra"][0]}, n80 {q["extra"][1]:.1f}'
    say(s)

# ---------------------------------------------------------------- Q3
say(); say('Q3  m<1 energy share (all columns, mhat pooled over all rows), measurement checks, half weight, sweeps, sub-additivity at 10%')
Q3 = {}
say('model    m<1 share | Q3a col-norm ratio (m<1 / m>=1) | Q3b sign agree, cols with median ratio<1 | Q3d half: convex ok / n, concave ok / n | S_add 10% | S_add 1% | L*_S 10%')
for nm in [m for m in MODELS if m in D]:
    d = D[nm]; fr, mh, colE = mfrac(d)
    Y, G = d['Y'], d['G']; C = np.where(colE >= np.quantile(colE, 0.7))[0]
    low, high = C[mh[C] < 1], C[mh[C] >= 1]
    nr = (np.median(np.sqrt(colE[low])) / np.median(np.sqrt(colE[high]))) if len(low) and len(high) else np.nan
    sg, rb = [], []
    for i in low:
        sel = np.abs(Y[:, i]) > np.median(np.abs(Y[:, i]))
        sg.append(np.mean(np.sign(G[sel, i]) == np.sign(Y[sel, i]))); rb.append(np.median(G[sel, i] / Y[sel, i]) < 1)
    q = dict(frac=fr, nr=float(nr), sign=float(np.mean(sg)) if sg else np.nan, rb=float(np.mean(rb)) if rb else np.nan, n_low=len(low))
    if 'Gh' in d:
        mhh = (d['Gh'] * Y).sum(0) / (0.5 * colE)
        cv = C[mh[C] < 0.95]; ccv = C[mh[C] > 1.2]
        q['cvx_ok'] = int(np.sum((mhh[cv] >= mh[cv]) & (mhh[cv] <= 1))); q['cvx_n'] = len(cv)
        q['ccv_ok'] = int(np.sum((mhh[ccv] >= 1) & (mhh[ccv] <= mh[ccv]))); q['ccv_n'] = len(ccv)
        q['mhh_low_med'] = float(np.median(mhh[cv])) if len(cv) else np.nan
    for tag, mk, dk in (('10', 'm10', 'd10'), ('01', 'm01', 'd01')):
        if mk in d:
            D1 = 1 - d[mk]; aL = -D1 @ G.T
            q[f'S{tag}'] = float(np.median(slope(aL, d[dk])))
            P = -D1 @ Y.T
            from scipy.stats import rankdata
            q[f'Ls{tag}'] = float(cc(rankdata(P, axis=0), rankdata(d[dk], axis=0)).mean())
    Q3[nm] = q
    say(f'{nm:8s} {fr:.3f}     | {q["nr"]:.2f} (n_low {len(low)})                | {q["sign"]:.2f}  {q["rb"]:.2f}                          | '
        + (f'{q["cvx_ok"]}/{q["cvx_n"]} (med half {q["mhh_low_med"]:.2f}), {q["ccv_ok"]}/{q["ccv_n"]}' if 'cvx_n' in q else '   -   ')
        + f'  | {q.get("S10", np.nan):.3f}  | {q.get("S01", np.nan):.3f} | {q.get("Ls10", np.nan):.3f}')
fam = {b: [Q3[m]['frac'] for m in (b, b + '_s1') if m in Q3] for b in ('w16', 'b', 'w64', 'w128', 'b_i05', 'b_i2')}
say('  seed-mean m<1 share: ' + ', '.join(f'{b} {np.mean(v):.3f}' for b, v in fam.items() if v))

# ---------------------------------------------------------------- prediction checks
say(); say('prediction checks')
hits = {}
def judge(name, ok, note=''):
    hits.setdefault(name.split()[0], []).append(bool(ok)); say(f'  {name:44s} {"hit" if ok else "MISS"}  {note}')
for nm, q in Q1.items():
    judge(f'Q1a {nm}', 0.25 <= q['SSW_T'] <= 0.75, f'SS_W/SS_T {q["SSW_T"]:.3f}')
    judge(f'Q1b(ms>=0.30 of SS_B) {nm}', q['r2ms'] >= 0.30, f'{q["r2ms"]:.2f}')
    judge(f'Q1b(RIF_ES<ms) {nm}', q['r2rif'][LAM_ES] < q['r2ms'], f'RIF {q["r2rif"][LAM_ES]:.2f} vs {q["r2ms"]:.2f}')
    judge(f'Q1b(RIFbest<ms) {nm}', q['r2rif_best'] < q['r2ms'], f'RIF {q["r2rif_best"]:.2f} vs {q["r2ms"]:.2f}')
    judge(f'Q1c {nm}', q['late_W'] >= 0.15, f'{q["late_W"]:.3f}')
    if q['n_low']:
        judge(f'Q1d {nm}', 1.0 <= q['rif_low_med'] <= 1.10, f'{q["rif_low_med"]:.3f}')
    judge(f'Q1e(sp) {nm}', q['sp_ms'] >= 0.5, f'{q["sp_ms"]:+.2f}')
    if q['n_low'] >= 20:
        judge(f'Q1e(low) {nm}', q['ms_low_below1'] >= 0.5, f'{q["ms_low_below1"]:.2f}')
for nm in iid:
    a, b = Q2[nm]['m'], Q2[nm]['mem']
    judge(f'Q2a(m) {nm}', a[0] / a[1] <= 0.7, f'{a[0]/a[1]:.2f}'); judge(f'Q2a(mem) {nm}', b[0] / b[1] <= 0.7, f'{b[0]/b[1]:.2f}')
for nm in ('rare0', 'rare0_s1'):
    if f'{nm}_A' in Q2:
        ra = Q2[f'{nm}_A']['m'][0] / Q2[f'{nm}_A']['m'][1]; rb = Q2[f'{nm}_B']['m'][0] / Q2[f'{nm}_B']['m'][1]
        judge(f'Q2b {nm}', ra > rb and ra >= 0.9 and rb <= 0.8, f'A {ra:.2f} B {rb:.2f}')
for nm in ('w16', 'w16_s1', 'b', 'b_s1'):
    if nm in Q3:
        q = Q3[nm]
        if q['n_low']:
            judge(f'Q3a {nm}', q['nr'] >= 0.5, f'{q["nr"]:.2f}'); judge(f'Q3b {nm}', q['sign'] >= 0.9 and q['rb'] >= 0.8, f'{q["sign"]:.2f} {q["rb"]:.2f}')
if 'w16_s1' in Q3:
    judge('Q3c w16_s1', Q3['w16_s1']['frac'] >= 0.2, f'{Q3["w16_s1"]["frac"]:.3f}')
cv = [Q3[m] for m in ('w16', 'w16_s1', 'b', 'b_s1') if m in Q3 and 'cvx_n' in Q3[m]]
if cv:
    a_ok, a_n = sum(q['cvx_ok'] for q in cv), sum(q['cvx_n'] for q in cv); b_ok, b_n = sum(q['ccv_ok'] for q in cv), sum(q['ccv_n'] for q in cv)
    judge('Q3d convex', a_n > 0 and a_ok / a_n >= 0.7, f'{a_ok}/{a_n}'); judge('Q3d concave', b_n > 0 and b_ok / b_n >= 0.7, f'{b_ok}/{b_n}')
mf = {b: np.mean(v) for b, v in fam.items() if len(v) == 2}
if all(k in mf for k in ('w16', 'b', 'w64', 'w128')):
    judge('Q3e width', mf['w16'] > mf['b'] > mf['w64'] > mf['w128'], ' > '.join(f'{mf[k]:.3f}' for k in ('w16', 'b', 'w64', 'w128')))
if all(k in mf for k in ('b_i05', 'b', 'b_i2')):
    judge('Q3f init', mf['b_i05'] > mf['b'] > mf['b_i2'], ' > '.join(f'{mf[k]:.3f}' for k in ('b_i05', 'b', 'b_i2')))
cases = [m for m in MODELS if m in Q3 and 'S10' in Q3[m]]
if len(cases) >= 6:
    sp = spearmanr([Q3[m]['frac'] for m in cases], [1 - Q3[m]['S10'] for m in cases])[0]
    c1 = all(Q3[m]['S10'] < 1 for m in cases if Q3[m]['frac'] >= 0.2); c2 = all(Q3[m]['S10'] >= 0.97 for m in cases if Q3[m]['frac'] <= 0.05)
    judge('Q3g', sp >= 0.5 and c1 and c2, f'Spearman {sp:+.2f} over {len(cases)}; S_add<1 where share>=0.2: {c1}; >=0.97 where share<=0.05: {c2}')
say('hit counts: ' + ', '.join(f'{k} {sum(v)}/{len(v)}' for k, v in hits.items()))
open(os.path.join(RES, 'multiplier_summary.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
json.dump(dict(Q1={k: {kk: (vv if not isinstance(vv, (np.ndarray, dict)) else None) for kk, vv in v.items()} for k, v in Q1.items()},
               Q2={k: v for k, v in Q2.items()}, Q3=Q3), open(os.path.join(RES, 'multiplier_readings.json'), 'w'), indent=1, default=float)
