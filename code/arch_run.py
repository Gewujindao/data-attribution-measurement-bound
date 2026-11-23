"""Architecture study: generate and analyse the one-factor settings, resumable, stops itself after --budget seconds.
  python arch_run.py --dir <cache/arch> --out <results/arch> [--only b,w16,...] [--budget 250]
Per setting: meta (P, train loss/acc, test acc), exact rows (chunks of 50), 200 retrains at 1% (batches of 50, vmapped),
then ladder_analyze.py on it (readings json/txt into --out).
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_points import P6, VARIANTS
from ladder_core import POINTS, make_masks

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--only', default=None); ap.add_argument('--budget', type=float, default=250)
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.dir, exist_ok=True); os.makedirs(a.out, exist_ok=True)
names = a.only.split(',') if a.only else [v for v in VARIANTS] + [f'{v}_s1' for v in VARIANTS]
left = lambda: time.time() - t0 < a.budget
for nm in names:
    if os.path.exists(os.path.join(a.out, f'{nm}_readings.json')):
        continue
    if not left():
        break
    tag = os.path.join(a.dir, nm); cfg = POINTS[nm]
    pt = P6(nm, 'cpu', torch.float64)
    if not os.path.exists(os.path.join(a.out, f'{nm}_meta.json')):
        th = pt.train(torch.ones(pt.N, dtype=torch.float64))
        lo = pt.logits(th, pt.Xtr)
        trl = float(pt.loss_vec(th, pt.Xtr, pt.ytr).mean())
        meta = dict(name=nm, P=pt.P, N=pt.N, P_over_N=pt.P / pt.N, train_loss=trl,
                    train_acc=float((lo.argmax(-1) == pt.ytr).double().mean()),
                    test_acc=float((pt.logits(th, pt.Xte, pt.Xref).argmax(-1) == pt.yte).double().mean()))
        json.dump(meta, open(os.path.join(a.out, f'{nm}_meta.json'), 'w'))
        print(f'[{nm}] P={pt.P} train loss {trl:.4f} train acc {meta["train_acc"]:.3f} test acc {meta["test_acc"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    ch = 10 if cfg['arch'] == 'cnn' else 50          # conv rows are slower, so they are written in smaller chunks
    todo = [c0 for c0 in range(0, pt.k, ch) if not os.path.exists(f'{tag}_Y_{c0:04d}.npy')]
    if todo:
        pt.fit_reference(seg=pt.T)
        for c0 in todo:
            if not left():
                break
            np.save(f'{tag}_Y_{c0:04d}.npy', pt.replay_rows(list(range(c0, min(c0 + ch, pt.k)))))
            print(f'  [{nm}] rows {c0}+ ({time.time()-t0:.0f}s)', flush=True)
    if not left():
        break
    alpha, M, seed = cfg['truth'][0]
    masks = make_masks(pt.N, alpha, M, seed)
    if not os.path.exists(f'{tag}_f0.npy'):
        np.save(f'{tag}_f0.npy', pt.margins(pt.train_many_mlp(torch.ones(1, pt.N, dtype=torch.float64))[0]).detach().numpy())
    f0 = np.load(f'{tag}_f0.npy')
    for m0 in range(0, M, ch):
        fb = f'{tag}_truth_01_{m0:03d}.npy'
        if os.path.exists(fb):
            continue
        if not left():
            break
        out = torch.func.vmap(pt.margins)(pt.train_many_mlp(torch.tensor(masks[m0:m0 + ch])))
        np.save(fb, out.detach().numpy() - f0[None])
        print(f'  [{nm}] retrains {m0}+ ({time.time()-t0:.0f}s)', flush=True)
    fbs = [f'{tag}_truth_01_{m0:03d}.npy' for m0 in range(0, M, ch)]
    if not all(os.path.exists(f) for f in fbs) or not all(os.path.exists(f'{tag}_Y_{c0:04d}.npy') for c0 in range(0, pt.k, ch)):
        break
    np.savez(f'{tag}_truth.npz', f0=f0, m01=masks, d01=np.concatenate([np.load(f) for f in fbs]))
    if not left():
        break
    r = subprocess.run([sys.executable, '-X', 'utf8', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ladder_analyze.py'),
                        '--dir', a.dir, '--point', nm, '--out', a.out, '--draws', '20'], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
done = [n for n in names if os.path.exists(os.path.join(a.out, f'{n}_readings.json'))]
print(f'settings done {len(done)}/{len(names)}  total {time.time()-t0:.0f}s', flush=True)
