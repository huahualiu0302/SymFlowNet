"""Plane Poiseuille benchmark of the solver (Online Resource 1, Table S1).

    python lbm/channel_test.py

Pressure-driven flow between two walls in a channel of H fluid nodes and length 40, simulated with BGK and TRT.
The mean and nodal velocities in the middle section are compared with the analytical profile
u(y) = G / (2 rho nu) y (H - y), y = 1/2, 3/2, ..., with G the pressure gradient measured in the middle of the channel.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lbm_d2q9 as L  # noqa: E402


def channel(scheme, H):
    b = L.Batch(1, scheme, 1, ny0=H, nx0=40)
    b.max_steps = 400_000
    b.fill(0, f'H{H}', np.zeros((H, 40), np.uint8))
    while True:
        b.step(b.check)
        g = b.diagnose()[0]
        if g['done']:
            break
    res = b.result(0)
    ux, rho = res['ux200'], res['rho200']
    p = (rho[:, 10:30].mean(axis=0) - 1) / 3.0
    G = -np.polyfit(np.arange(p.size), p, 1)[0]
    r0 = rho[:, 20].mean()
    exact_mean = G * H ** 2 / (12 * r0 * L.NU)
    y = np.arange(H) + 0.5
    exact = G / (2 * r0 * L.NU) * y * (H - y)
    return dict(scheme=scheme, H=H, mean_err=float(ux[:, 20].mean() / exact_mean - 1),
                node_err=float(np.abs(ux[:, 20] / exact - 1).max()), converged=g['converged'], steps=g['steps'])


if __name__ == '__main__':
    print('scheme  H  mean velocity error  max nodal error')
    for scheme in ('bgk', 'trt'):
        for H in (1, 2, 3, 4, 8, 16):
            r = channel(scheme, H)
            print(f'{scheme.upper():4s} {H:3d}  {100 * r["mean_err"]:+9.3f}%  {100 * r["node_err"]:9.3f}%')
