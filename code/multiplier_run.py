"""Multiplier measurements: exact rows, subset retrains, all-N leave-one-out and half-weight retrains; resumable, stops after --budget s.
  python multiplier_run.py --dir <cache/multiplier> --arch_dir <cache/arch> --out <results/multiplier> --only m1,m2,... [--budget 240]
Per model (multiplier_points.JOBS): meta; exact rows (copied from --arch_dir for reused models, else replayed in chunks of 50);
f0 (vmapped full training); subset retrains per spec (chunks of 50); LOO: N single deletions (weight 0) -> test margins
minus f0 and own-margin drop (memorisation, margin units); HALF: N single half-weightings (weight 0.5), same outputs.
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiplier_points import build, JOBS, REUSED_Y
from ladder_core import POINTS, make_masks

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--arch_dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--only', required=True); ap.add_argument('--budget', type=float, default=240)
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.dir, exist_ok=True); os.makedirs(a.out, exist_ok=True)
names = a.only.split(',')
left = lambda: time.time() - t0 < a.budget
CH = 50


def files(nm):
    k, N, J = POINTS[nm]['k'], POINTS[nm]['N'], JOBS[nm]
    fs = [f'{nm}_Y_{c0:04d}.npy' for c0 in range(0, k, CH)]
    for tag, al, M, sd in J['specs']:
        fs += [f'{nm}_{tag}_{m0:03d}.npy' for m0 in range(0, M, CH)]
    for kind in ('loo', 'half'):
        if J[kind]:
            fs += [f'{nm}_{kind}_{c0:04d}.npz' for c0 in range(0, N, CH)]
    return [os.path.join(a.dir, f) for f in fs]


for nm in names:
    if all(os.path.exists(f) for f in files(nm)):
        continue
    if not left():
        break
    tag0 = os.path.join(a.dir, nm); J = JOBS[nm]
    pt = build(nm)
    if nm in REUSED_Y:
        for f in glob.glob(os.path.join(a.arch_dir, f'{nm}_Y_*.npy')) + [os.path.join(a.arch_dir, f'{nm}_f0.npy')]:
            dst = os.path.join(a.dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    if not os.path.exists(os.path.join(a.out, f'{nm}_meta.json')):
        th = pt.train_many_mlp(torch.ones(1, pt.N, dtype=torch.float64))[0]
        lo = pt.logits(th, pt.Xtr)
        meta = dict(name=nm, P=pt.P, N=pt.N, k=pt.k, T=pt.T, train_loss=float(pt.loss_vec(th, pt.Xtr, pt.ytr).mean()),
                    train_acc=float((lo.argmax(-1) == pt.ytr).double().mean()),
                    test_acc=float((pt.logits(th, pt.Xte).argmax(-1) == pt.yte).double().mean()))
        json.dump(meta, open(os.path.join(a.out, f'{nm}_meta.json'), 'w'))
        print(f'[{nm}] P={pt.P} train loss {meta["train_loss"]:.4f} acc {meta["train_acc"]:.3f} test {meta["test_acc"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    todo = [c0 for c0 in range(0, pt.k, CH) if not os.path.exists(f'{tag0}_Y_{c0:04d}.npy')]
    if todo:
        pt.fit_reference(seg=pt.T)
        for c0 in todo:
            if not left():
                break
            np.save(f'{tag0}_Y_{c0:04d}.npy', pt.replay_rows(list(range(c0, min(c0 + CH, pt.k)))))
            print(f'  [{nm}] rows {c0}+ ({time.time()-t0:.0f}s)', flush=True)
    if not left():
        break
    if not os.path.exists(f'{tag0}_f0.npy'):
        np.save(f'{tag0}_f0.npy', pt.margins(pt.train_many_mlp(torch.ones(1, pt.N, dtype=torch.float64))[0]).detach().numpy())
    f0 = np.load(f'{tag0}_f0.npy')
    for tag, al, M, sd in J['specs']:
        masks = make_masks(pt.N, al, M, sd)
        np.save(f'{tag0}_{tag}_masks.npy', masks)
        for m0 in range(0, M, CH):
            fb = f'{tag0}_{tag}_{m0:03d}.npy'
            if os.path.exists(fb) or not left():
                continue
            out = torch.func.vmap(pt.margins)(pt.train_many_mlp(torch.tensor(masks[m0:m0 + CH])))
            np.save(fb, out.detach().numpy() - f0[None])
            print(f'  [{nm}] {tag} retrains {m0}+ ({time.time()-t0:.0f}s)', flush=True)
    if not os.path.exists(f'{tag0}_ztrain.npy'):
        thf = pt.train_many_mlp(torch.ones(1, pt.N, dtype=torch.float64))[0]
        np.save(f'{tag0}_ztrain.npy', pt.margin_vec(thf, pt.Xtr, pt.ytr).detach().numpy())
    zt = np.load(f'{tag0}_ztrain.npy')
    selfm = lambda th, x, yy: pt.margin_vec(th, x[None], yy[None])[0]
    for kind, wv in (('loo', 0.0), ('half', 0.5)):
        if not J[kind]:
            continue
        for c0 in range(0, pt.N, CH):
            fb = f'{tag0}_{kind}_{c0:04d}.npz'
            if os.path.exists(fb) or not left():
                continue
            ii = np.arange(c0, min(c0 + CH, pt.N))
            Wm = torch.ones(len(ii), pt.N, dtype=torch.float64); Wm[np.arange(len(ii)), ii] = wv
            th = pt.train_many_mlp(Wm)
            d = torch.func.vmap(pt.margins)(th).detach().numpy() - f0[None]
            zs = torch.func.vmap(selfm)(th, pt.Xtr[ii], pt.ytr[ii]).detach().numpy()
            np.savez(fb, d=d, mem=zt[ii] - zs)
            print(f'  [{nm}] {kind} {c0}+ ({time.time()-t0:.0f}s)', flush=True)
done = [n for n in names if all(os.path.exists(f) for f in files(n))]
print(f'models done {len(done)}/{len(names)}: {",".join(done)}  total {time.time()-t0:.0f}s', flush=True)
