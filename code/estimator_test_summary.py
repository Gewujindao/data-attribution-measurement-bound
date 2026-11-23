"""Estimator test: summary of the test on new settings (pass / fail of each new setting at s=k/10 and s=k/5) and of kappa.  Usage: python estimator_test_summary.py"""
import os
import numpy as np

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'estimator_test')
NEW = [('A', 300), ('B', 400), ('C', 500), ('D', 600), ('E', 800), ('F', 300), ('H1', 100), ('H2', 300), ('H3', 300)]
OLD = [('S1', 'digits_h32_N1000_k300', 300), ('S4b', 'synth_h32_N4000_k300_e5', 300), ('S2', 'digits_h128_N1000_k300', 300)]
passes = {10: 0, 5: 0}
print('setting  k  rank90  L*_S | s=k/10: LDS err med[90%]  r90 err med[90%]  verdict | s=k/5: same | kappa_true  kappa_LOO(s=k/10)')
rows_k = []
for nm, k in NEW + [(o[0], o[2]) for o in OLD]:
    tag = f'setting_{nm}' if nm in dict(NEW) else dict((o[0], o[1]) for o in OLD)[nm]
    fe = os.path.join(SCR, f'eval_{tag}_exact.npz')
    if not os.path.exists(fe):
        print(f'{nm:4s} (not evaluated)'); continue
    ex = np.load(fe)
    Lstar = ex['Ls'].mean()
    line = f'{nm:4s} {k:4d} {int(ex["r90"]):4d}  {Lstar:.3f} |'
    kl = np.nan
    for div in (10, 5):
        s = int(round(k / div))
        f = os.path.join(SCR, f'eval_{tag}_{s}.npz')
        if not os.path.exists(f):
            line += f' s={s}: (missing) |'; continue
        D = np.load(f)
        e, r = D['err'], D['r90err']
        if div == 10: kl = np.median(D['kloo'])
        degenerate = Lstar < 0.3
        ok = (np.median(r) <= 0.25) and (degenerate or np.median(e) <= 0.05)
        if nm in dict(NEW) and len(e) == 20: passes[div] += int(ok)
        line += (f' s={s} (n={len(e)}): {np.median(e):.3f}[{np.quantile(e,.9):.3f}] {np.median(r):.2f}[{np.quantile(r,.9):.2f}] '
                 f'conv {np.median(D["conv"]):.3f} {"PASS" if ok else "fail"}{" (LDS undecidable)" if degenerate else ""} |')
    kt = ex['kap'].mean()
    rows_k.append((nm, kt, kl))
    print(line + f' {kt:+.3f}  {kl:+.3f}   (kappa at 0.05k/0.1k/0.2k: ' + ' '.join(f'{x:+.3f}' for x in ex['kap']) + ')')
print(f'new settings passing: s=k/10 {passes[10]}/9, s=k/5 {passes[5]}/9  (usable >=6, unusable <=4)')
kt = np.array([r[1] for r in rows_k]); kl = np.array([r[2] for r in rows_k])
m = (np.abs(kt) >= 0.02) & np.isfinite(kl)
if m.sum():
    print(f'kappa sign agreement where |kappa_true|>=0.02: {(np.sign(kt[m]) == np.sign(kl[m])).sum()}/{m.sum()}; '
          f'correlation(kappa_true, kappa_LOO) over {np.isfinite(kl).sum()} settings: {np.corrcoef(kt[np.isfinite(kl)], kl[np.isfinite(kl)])[0,1]:.2f}')
