"""Multiplier measurements: model registry.
Reused architecture-study models (exact rows on disk in the architecture-study cache dir): w16, b, w64, w128 and their _s1 seeds.
Reused leave-one-out-study model builders: rare0, rare0_s1 (loo_points.P7; rows recomputed).
New: b_i05, b_i2 (+_s1) = the architecture-study base b with every initial parameter multiplied by 0.5 or 2 (biases start at 0,
so this scales the initial weights); data, batch order and training recipe unchanged.
JOBS: model -> dict(specs=[(tag, alpha, M, mask seed)], loo=bool, half=bool)
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_core import POINTS
from arch_points import P6
from loo_points import P7

SCALE = {}
for s in ('', '_s1'):
    for tag, sc in (('i05', 0.5), ('i2', 2.0)):
        POINTS[f'b_{tag}{s}'] = dict(POINTS['b' + s]); SCALE[f'b_{tag}{s}'] = sc
A10 = [('a10', 0.10, 200, 29)]
JOBS = {}
for base in ('w16', 'b', 'w128', 'w64', 'b_i05', 'b_i2'):
    for s in ('', '_s1'):
        JOBS[base + s] = dict(specs=A10, loo=True, half=base in ('w16', 'b'))
for s in ('', '_s1'):
    JOBS['rare0' + s] = dict(specs=[], loo=True, half=False)
REUSED_Y = tuple(f'{b}{s}' for b in ('w16', 'b', 'w64', 'w128') for s in ('', '_s1'))


def build(name):
    if name.startswith('rare0'):
        return P7(name, 'cpu', torch.float64)
    pt = P6(name, 'cpu', torch.float64)
    if name in SCALE:
        pt.th0 = pt.th0 * SCALE[name]
    return pt
