"""Ceiling: summarise the ceiling-curve pre-estimates of ceiling_estimators.py (t2_<point>_<k_eval>_<s>.npz).  Usage: python ceiling_estimators_summary.py"""
import glob, os
import numpy as np
from ceiling_common import SCR

print('point k_eval s  draws | exact r90 | estimator: median[90%] max_B|dLDS| for conversions cal/th/rat ; median |r90 err| [90%] ; '
      'sd(r90_hat) ; median signed dLDS at 0.2k (rat)')
for fn in sorted(glob.glob(os.path.join(SCR, 't2_*.npz')), key=lambda f: (f.split('_')[1], int(f.split('_')[2]), int(f.split('_')[3][:-4]))):
    _, pt, ke, s = os.path.basename(fn)[:-4].split('_')
    D = np.load(fn)
    ex = np.load(os.path.join(SCR, f'exact_{pt}_{ke}.npz'))
    Fk = ex['Fk']; r90 = int(np.searchsorted(Fk, 0.9) + 1)
    n = len(D['draw'])
    print(f'{pt} {ke} {s} n={n} | r90 {r90} | conv-only cal/th/rat median {np.median(D["conv_only"]):.3f}/{np.median(D["conv_only_th"]):.3f}/'
          f'{np.median(D["conv_only_rat"]):.3f} | E1 fits alpha {np.median(D["alpha"]):.2f} f {np.median(D["f"]):.2f}')
    for e in ('E0', 'E1', 'E2', 'E3'):
        r = D[e + '_r90']; re = np.abs(r / r90 - 1)
        m = ' '.join(f'{np.median(D[f"{e}_{g}_max"]):.3f}[{np.quantile(D[f"{e}_{g}_max"], .9):.3f}]' for g in ('cal', 'th', 'rat'))
        print(f'    {e}: {m} ; r90 med {np.median(r):.0f} err {np.median(re):.2f}[{np.quantile(re, .9):.2f}] ; sd {r.std():.1f} ; '
              f'dLDS@0.2k {np.median(D[e + "_rat_at20"]):+.3f} ; maxFerr {np.median(D[e + "_Ferr"]):.3f}')
