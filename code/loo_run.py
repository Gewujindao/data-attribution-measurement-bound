"""Leave-one-out study: exact rows, subset retrains and all-N leave-one-out retrains; resumable, stops after --budget s.
  python loo_run.py --dir <cache/loo> --arch_dir <cache/arch> --out <results/loo> [--only bc,rare0,...] [--budget 240] [--noloo w128_s1,...]
Per model: meta; exact rows (chunks of 50; reused architecture-study models read their rows from --arch_dir); f0 (vmapped full training,
same arithmetic as the retrains); for each spec of loo_points.SPECS: M subset retrains (chunks of 50);
LOO: N single-deletion retrains (chunks of 50) -> test margins minus f0 and the deleted point's own margin minus its
margin in the full model (memorisation in margin units).  Analysis is a separate step (loo_analyze.py).
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loo_points import build, SPECS, REUSED
from ladder_core import POINTS, make_masks

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--arch_dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--only', default=None); ap.add_argument('--budget', type=float, default=240)
ap.add_argument('--noloo', default='')
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.dir, exist_ok=True); os.makedirs(a.out, exist_ok=True)
names = a.only.split(',') if a.only else list(SPECS)
noloo = set(a.noloo.split(',')) if a.noloo else set()
left = lambda: time.time() - t0 < a.budget
CH = 50


def done_model(nm):
    pt_k = POINTS[nm]['k']; N = POINTS[nm]['N']
    ok = all(os.path.exists(os.path.join(a.dir, f'{nm}_Y_{c0:04d}.npy')) for c0 in range(0, pt_k, CH))
    for tag, al, M, sd in SPECS[nm]:
        ok &= all(os.path.exists(os.path.join(a.dir, f'{nm}_{tag}_{m0:03d}.npy')) for m0 in range(0, M, CH))
    if nm not in noloo:
        ok &= all(os.path.exists(os.path.join(a.dir, f'{nm}_loo_{c0:04d}.npz')) for c0 in range(0, N, CH))
    return ok


for nm in names:
    if done_model(nm):
        continue
    if not left():
        break
    tag0 = os.path.join(a.dir, nm)
    pt = build(nm)
    if nm in REUSED:                                  # exact rows and f0 from the architecture study (same model, same arithmetic)
        for f in glob.glob(os.path.join(a.arch_dir, f'{nm}_Y_*.npy')) + [os.path.join(a.arch_dir, f'{nm}_f0.npy')]:
            dst = os.path.join(a.dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    elif not os.path.exists(os.path.join(a.out, f'{nm}_meta.json')):
        th = pt.train(torch.ones(pt.N, dtype=torch.float64))
        lo = pt.logits(th, pt.Xtr)
        trl = float(pt.loss_vec(th, pt.Xtr, pt.ytr).mean())
        meta = dict(name=nm, P=pt.P, N=pt.N, k=pt.k, T=pt.T, P_over_N=pt.P / pt.N, train_loss=trl,
                    train_acc=float((lo.argmax(-1) == pt.ytr).double().mean()),
                    test_acc=float((pt.logits(th, pt.Xte).argmax(-1) == pt.yte).double().mean()),
                    test_labels=[int(v) for v in pt.yte], n_train_per_class=np.bincount(pt.ytr.numpy()).tolist())
        json.dump(meta, open(os.path.join(a.out, f'{nm}_meta.json'), 'w'))
        print(f'[{nm}] P={pt.P} T={pt.T} train loss {trl:.4f} acc {meta["train_acc"]:.3f} test acc {meta["test_acc"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
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
    for tag, al, M, sd in SPECS[nm]:
        masks = make_masks(pt.N, al, M, sd)
        np.save(f'{tag0}_{tag}_masks.npy', masks)
        for m0 in range(0, M, CH):
            fb = f'{tag0}_{tag}_{m0:03d}.npy'
            if os.path.exists(fb):
                continue
            if not left():
                break
            out = torch.func.vmap(pt.margins)(pt.train_many_mlp(torch.tensor(masks[m0:m0 + CH])))
            np.save(fb, out.detach().numpy() - f0[None])
            print(f'  [{nm}] {tag} retrains {m0}+ ({time.time()-t0:.0f}s)', flush=True)
    if nm in noloo or not left():
        continue
    if not os.path.exists(f'{tag0}_ztrain.npy'):
        thf = pt.train_many_mlp(torch.ones(1, pt.N, dtype=torch.float64))[0]
        np.save(f'{tag0}_ztrain.npy', pt.margin_vec(thf, pt.Xtr, pt.ytr).detach().numpy())
    zt = np.load(f'{tag0}_ztrain.npy')
    selfm = lambda th, x, yy: pt.margin_vec(th, x[None], yy[None])[0]
    for c0 in range(0, pt.N, CH):
        fb = f'{tag0}_loo_{c0:04d}.npz'
        if os.path.exists(fb):
            continue
        if not left():
            break
        ii = np.arange(c0, min(c0 + CH, pt.N))
        Wm = torch.ones(len(ii), pt.N, dtype=torch.float64); Wm[np.arange(len(ii)), ii] = 0
        th = pt.train_many_mlp(Wm)
        d = torch.func.vmap(pt.margins)(th).detach().numpy() - f0[None]
        zs = torch.func.vmap(selfm)(th, pt.Xtr[ii], pt.ytr[ii]).detach().numpy()
        np.savez(fb, d=d, mem=zt[ii] - zs)
        print(f'  [{nm}] loo {c0}+ ({time.time()-t0:.0f}s)', flush=True)
done = [n for n in names if done_model(n)]
print(f'models done {len(done)}/{len(names)}: {",".join(done)}  total {time.time()-t0:.0f}s', flush=True)
