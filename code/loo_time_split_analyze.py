"""Leave-one-out study: checks S1-S3 on the time-split rows and the LOO responses.
  python loo_time_split_analyze.py <cache/loo> <results/loo>
zeta_W = <u, y_late>/|u|^2, zeta_perp = <v, y_late>/|v|^2 (row medians; W = SVD oracle of the row group, B = 0.05k/0.1k/0.2k);
S3: inside each top-30% column (by energy), Spearman over test points with |y_ji| above the column median of
G_ji/y_ji (LOO multiplier) against y_late,ji/y_ji (late share); mean over columns.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
SCR, OUT = sys.argv[1], sys.argv[2]
JOBS = [('b', np.arange(100), 'all'), ('rare0', np.arange(100), 'A'), ('rare0', np.arange(100, 200), 'B'),
        ('rare0_s1', np.arange(100), 'A'), ('rare0_s1', np.arange(100, 200), 'B')]
out, res = [], {}
def say(s):
    print(s, flush=True); out.append(s)
say('model     rows | B: zeta_W zeta_perp (perp-W) | S3 mean column Spearman(m_ji, late share) [columns]')
for m, rows, lab in JOBS:
    Y = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(SCR, f'{m}_Y_*.npy')))])[rows]
    sp = [np.load(f) for f in sorted(glob.glob(os.path.join(SCR, f'{m}_Ysplit_*.npz')))]
    Yl = np.concatenate([x['late'] for x in sp])[rows]
    G = -np.concatenate([np.load(f)['d'] for f in sorted(glob.glob(os.path.join(SCR, f'{m}_loo_*.npz')))])[:, rows].T
    k = Y.shape[0]
    Yc = Y - Y.mean(1, keepdims=True); nrm = np.linalg.norm(Yc, axis=1)
    U, S, Vt = np.linalg.svd(Yc / nrm[:, None], full_matrices=False)
    Ylc = Yl - Yl.mean(1, keepdims=True)
    line = f'{m:9s} {lab:3s} |'
    tiers = []
    for frac in (0.05, 0.1, 0.2):
        B = int(round(frac * k))
        Uw = (U[:, :B] * S[:B]) @ Vt[:B] * nrm[:, None]; Vp = Yc - Uw
        zW = np.median((Uw * Ylc).sum(1) / (Uw * Uw).sum(1)); zP = np.median((Vp * Ylc).sum(1) / (Vp * Vp).sum(1))
        tiers.append(dict(B=B, zW=float(zW), zP=float(zP)))
        line += f' {B}: {zW:.2f} {zP:.2f} ({zP - zW:+.2f}) |'
    colE = (Y * Y).sum(0); top = np.where(colE >= np.quantile(colE, 0.7))[0]
    rs = []
    for i in top:
        sel = np.abs(Y[:, i]) > np.median(np.abs(Y[:, i]))
        if sel.sum() >= 10:
            r = spearmanr(G[sel, i] / Y[sel, i], Yl[sel, i] / Y[sel, i])[0]
            if np.isfinite(r):
                rs.append(r)
    line += f' S3 {np.mean(rs):+.3f} [{len(rs)}]'
    res[f'{m}_{lab}'] = dict(tiers=tiers, S3=float(np.mean(rs)), late_share_total=float((Yl * Y).sum() / (Y * Y).sum()))
    say(line)
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'split_checks.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
json.dump(res, open(os.path.join(OUT, 'split_checks.json'), 'w'), indent=1)
