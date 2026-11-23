"""Scale ladder: generate one ladder point (exact attribution rows + retrain truth), resumable, any device.
  python ladder_generate.py --point L2 --stage calib  --dir D [--device cuda]   time one training / one replay chunk, project hours
  python ladder_generate.py --point L2 --stage exact  --dir D [--rows 20]       exact rows (chunk files; --rows = only the first rows)
  python ladder_generate.py --point L2 --stage truth  --dir D [--mmax 50]       retrain truth (batch files; --mmax = first masks only)
  python ladder_generate.py --point L2 --stage pilot  --dir D                   L* on the rows / masks generated so far (go / no-go)
Every stage stops itself after --budget seconds and continues where it stopped on the next call.
"""
import argparse, glob, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--point', required=True)
ap.add_argument('--stage', required=True, choices=['calib', 'exact', 'truth', 'pilot'])
ap.add_argument('--dir', required=True)
ap.add_argument('--device', default='cpu')
ap.add_argument('--dtype', default=None, help='float64 (CPU default) or float32 (GPU default)')
ap.add_argument('--data_root', default=os.environ.get('CIFAR_ROOT', ''))
ap.add_argument('--chunk', type=int, default=None, help='test rows replayed together (vmapped)')
ap.add_argument('--micro', type=int, default=None, help='micro-batch for gradients/HVPs (gn only)')
ap.add_argument('--seg', type=int, default=None, help='checkpoint every seg steps (default: one epoch)')
ap.add_argument('--rows', type=int, default=None)
ap.add_argument('--mmax', type=int, default=None)
ap.add_argument('--budget', type=float, default=1e9)
ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--tf32', action='store_true')
a = ap.parse_args()

if a.device.startswith('cuda'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
torch.set_num_threads(a.threads)
if a.device.startswith('cuda'):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_core import Point, POINTS, make_masks

t0 = time.time()
os.makedirs(a.dir, exist_ok=True)
dtype = getattr(torch, a.dtype) if a.dtype else (torch.float32 if a.device.startswith('cuda') else torch.float64)
cfg = POINTS[a.point]
pt = Point(a.point, a.device, dtype, a.data_root or None, a.micro)
seg = a.seg or (pt.N // pt.bs)
chunk = a.chunk or (50 if cfg['kind'] == 'mlp' else 8)
tag = os.path.join(a.dir, a.point)
sync = (lambda: torch.cuda.synchronize()) if a.device.startswith('cuda') else (lambda: None)
print(f'[{a.point}] P={pt.P} N={pt.N} k={pt.k} T={pt.T} bs={pt.bs} micro={pt.micro} seg={seg} chunk={chunk} '
      f'{a.device} {str(dtype).split(".")[1]}', flush=True)


def truth_batches():
    """yields (key, alpha, masks, m0, file) for every saved batch of retrains, in order."""
    for alpha, M, seed in cfg['truth']:
        masks = make_masks(pt.N, alpha, M, seed)
        key = f'{int(round(alpha * 100)):02d}'
        for m0 in range(0, M, cfg['tb']):
            yield key, alpha, masks, m0, f'{tag}_truth_{key}_{m0:03d}.npy'


if a.stage == 'calib':
    ones = torch.ones(pt.N, dtype=dtype, device=pt.dev)
    sync(); t1 = time.time()
    pt.fit_reference(seg); sync(); t_tr = time.time() - t1
    acc = float((pt.logits(pt._thT, pt.Xte, pt.Xref).argmax(-1) == pt.yte).double().mean())
    steps = min(pt.T, max(5, seg // 2))
    sync(); t1 = time.time()
    pt.replay_rows(list(range(chunk)), t_limit=steps); sync(); t_rep = time.time() - t1
    # t_rep = test grads + one segment recompute (~ seg steps of training) + `steps` reverse steps
    t_step_tr = t_tr / pt.T
    t_rev = max(t_rep - seg * t_step_tr, 1e-9) / steps
    per_chunk = pt.T * t_step_tr + pt.T * t_rev
    n_chunks = -(-pt.k // chunk)
    exact_h = n_chunks * per_chunk / 3600
    n_retrain = sum(M for _, M, _ in cfg['truth'])
    truth_h = n_retrain * t_tr / 3600
    ck_gb = len(pt._ck) * 2 * pt.P * pt.th0.element_size() / 2 ** 30
    seg_gb = seg * pt.P * pt.th0.element_size() / 2 ** 30
    out = dict(point=a.point, P=pt.P, T=pt.T, test_acc=acc, t_train_s=t_tr, t_reverse_step_chunk_s=t_rev, chunk=chunk,
               micro=pt.micro, exact_hours=exact_h, truth_hours=truth_h, replay_over_train=per_chunk / chunk / t_tr,
               checkpoints_cpu_GiB=ck_gb, segment_device_GiB=seg_gb)
    json.dump(out, open(f'{tag}_calib.json', 'w'), indent=1)
    print(f'  one training {t_tr:.1f}s (test acc {acc:.3f}); one reverse step for {chunk} rows {t_rev*1e3:.1f} ms; '
          f'one row = {per_chunk/chunk/t_tr:.2f} trainings', flush=True)
    print(f'  projected: exact {pt.k} rows {exact_h:.2f} h, truth {n_retrain} retrains {truth_h:.2f} h, total {exact_h+truth_h:.2f} h; '
          f'checkpoints {ck_gb:.2f} GiB (CPU), segment {seg_gb:.2f} GiB (device)', flush=True)

elif a.stage == 'exact':
    nrow = a.rows or pt.k
    todo = [c0 for c0 in range(0, nrow, chunk) if not os.path.exists(f'{tag}_Y_{c0:04d}.npy')]
    if todo:
        pt.fit_reference(seg)
        acc = float((pt.logits(pt._thT, pt.Xte, pt.Xref).argmax(-1) == pt.yte).double().mean())
        print(f'  reference training done, test acc {acc:.3f}  ({time.time()-t0:.1f}s)', flush=True)
        json.dump(dict(point=a.point, P=pt.P, T=pt.T, N=pt.N, k=pt.k, test_acc=acc, chunk=chunk), open(f'{tag}_meta.json', 'w'))
    for c0 in todo:
        if time.time() - t0 > a.budget:
            print('budget reached; resume later', flush=True); break
        rows = list(range(c0, min(c0 + chunk, nrow)))
        np.save(f'{tag}_Y_{c0:04d}.npy', pt.replay_rows(rows))
        print(f'  rows {rows[0]}-{rows[-1]} done ({time.time()-t0:.1f}s)', flush=True)
    left = [c0 for c0 in range(0, nrow, chunk) if not os.path.exists(f'{tag}_Y_{c0:04d}.npy')]
    print(f'exact: {len(left)} chunks left', flush=True)

elif a.stage == 'truth':
    f0_fn = f'{tag}_f0.npy'
    if not os.path.exists(f0_fn):
        th = pt.train_many_mlp(torch.ones(1, pt.N, dtype=dtype, device=pt.dev))[0] if cfg['kind'] == 'mlp' \
            else pt.train(torch.ones(pt.N, dtype=dtype, device=pt.dev))
        np.save(f0_fn, pt.margins(th).detach().cpu().numpy().astype(np.float64))
    f0 = np.load(f0_fn)
    complete = True
    for key, alpha, masks, m0, fb in truth_batches():
        if a.mmax and m0 >= a.mmax:
            complete = False; continue
        if os.path.exists(fb):
            continue
        if time.time() - t0 > a.budget:
            complete = False; continue
        Wm = torch.tensor(masks[m0:m0 + cfg['tb']], dtype=dtype, device=pt.dev)
        if cfg['kind'] == 'mlp':
            out = torch.func.vmap(pt.margins)(pt.train_many_mlp(Wm))
        else:
            out = torch.stack([pt.margins(pt.train(w)) for w in Wm])
        np.save(fb, out.detach().cpu().numpy().astype(np.float64) - f0[None])
        print(f'  retrains {key}% {m0}-{m0+len(Wm)-1} done ({time.time()-t0:.1f}s)', flush=True)
    if complete:
        res = dict(f0=f0)
        for alpha, M, seed in cfg['truth']:
            key = f'{int(round(alpha * 100)):02d}'
            res['m' + key] = make_masks(pt.N, alpha, M, seed)
            res['d' + key] = np.concatenate([np.load(f'{tag}_truth_{key}_{m0:03d}.npy') for m0 in range(0, M, cfg['tb'])])
        np.savez(f'{tag}_truth.npz', **res)
        print('truth complete', flush=True)
    else:
        print('truth: not complete yet', flush=True)

elif a.stage == 'pilot':
    from scipy.stats import rankdata
    fs = sorted(glob.glob(f'{tag}_Y_*.npy'))
    Y = np.concatenate([np.load(f) for f in fs])
    key, alpha, M, seed = '01', *cfg['truth'][0]
    masks = make_masks(pt.N, alpha, M, seed)
    bs_ = sorted(glob.glob(f'{tag}_truth_01_*.npy'))
    d = np.concatenate([np.load(f) for f in bs_])
    masks = masks[:len(d)]; d = d[:, :len(Y)]
    P_ = -(1 - masks) @ Y.T
    rk = lambda A: rankdata(A, axis=0) - (A.shape[0] + 1) / 2
    cc = lambda A, B: ((A - A.mean(0)) * (B - B.mean(0))).sum(0) / np.sqrt(((A - A.mean(0)) ** 2).sum(0) * ((B - B.mean(0)) ** 2).sum(0))
    Ls = cc(rk(P_), rk(d)); slope = np.median((P_ * d).sum(0) / (P_ * P_).sum(0))
    go = Ls.mean() >= 0.6
    print(f'  pilot: {len(Y)} rows x {len(d)} retrains: L*_S mean {Ls.mean():.3f} (sd {Ls.std():.3f}), actual/predicted slope median {slope:.2f}'
          f'  ->  {"GO" if go else "NO-GO (L*<0.6: not meta-smooth enough)"}', flush=True)
    json.dump(dict(rows=len(Y), retrains=len(d), Ls=float(Ls.mean()), slope=float(slope), go=bool(go)), open(f'{tag}_pilot.json', 'w'))
print(f'total {time.time()-t0:.1f}s', flush=True)
