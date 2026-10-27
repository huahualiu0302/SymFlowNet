"""Compute the LBM velocity labels of a stack of slices in the x- or y-direction (Sect. 3.2).

    python lbm/run_labels.py --slices slices/Parker.npz --direction x --out labels
    python lbm/run_labels.py --slices slices/Parker.npz --direction y --out labels

Input: an npz file with 'solid' (N, H, W) uint8, 1 in the grains, and 'filenames' (N,).
The x-direction label is driven from the left to the right column. The y-direction label is computed on the
transposed slice and transposed back (Eq. (1)), which gives the downward flow from the first to the last row.
Each finished slice is stored in <out>/work/<name>_<direction>/<k>.npz, so an interrupted run resumes where it stopped.
At the end <out>/<name>_<direction>.npz is written with solid, u (velocity component along the driving direction,
lattice units of the original grid, block-averaged over the r x r nodes of each pixel), passed, K, steps and filenames,
and a log.csv with the convergence statistics of every simulation.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lbm_d2q9 as L  # noqa: E402

NODES_PER_BATCH = 6e6
FIELDS = ['k', 'converged', 'passed', 'steps', 'K_pixel2', 'vel_change', 'est_remaining', 'link_flux_spread',
          'max_mach', 'seconds']


def save_slice(path, res):
    tmp = path.with_name(path.stem + '.tmp.npz')
    meta = {k: res[k] for k in FIELDS}
    np.savez_compressed(tmp, meta=json.dumps(meta), ux=res['ux200'], uy=res['uy200'], rho=res['rho200'])
    os.replace(tmp, path)


def simulate(masks, work, r=4, scheme='trt', log=print):
    """Simulate every mask (1 = solid) driven along x; finished slices are skipped."""
    work.mkdir(parents=True, exist_ok=True)
    todo = [k for k in range(len(masks)) if not (work / f'{k}.npz').exists()]
    log(f'{len(masks)} slices, {len(todo)} to simulate, {scheme.upper()} r = {r}')
    if not todo:
        return
    ny0, nx0 = masks.shape[1:]
    nodes = (ny0 + 2) * r * (nx0 + 2) * r
    slots = max(1, min(len(todo), 128, int(NODES_PER_BATCH // nodes)))
    b = L.Batch(r, scheme, slots, ny0=ny0, nx0=nx0)
    queue = list(todo)
    t_start, t_last, n_done = time.time(), 0.0, 0
    for s in range(slots):
        if queue:
            k = queue.pop(0)
            b.fill(s, k, masks[k])
    while any(st is not None for st in b.slot):
        b.step(b.check)
        for s, g in b.diagnose().items():
            if not g['done']:
                continue
            res = b.result(s)
            res.update(k=int(res['case_id']), converged=g['converged'], steps=g['steps'], vel_change=g['ch'],
                       est_remaining=g['est'], link_flux_spread=g['spread'])
            res['passed'] = bool(res['converged'] and res['finite'] and res['solid_nonzero'] == 0
                                 and res['max_mach'] < 0.1)
            save_slice(work / f'{res["k"]}.npz', res)
            n_done += 1
            b.release(s)
            if queue:
                k = queue.pop(0)
                b.fill(s, k, masks[k])
        if time.time() - t_last > 60:
            t_last = time.time()
            el = t_last - t_start
            eta = el / n_done * (len(todo) - n_done) / 3600 if n_done else float('nan')
            log(f'finished {n_done}/{len(todo)}, {el / 60:.1f} min, about {eta:.2f} h left')


def export(solid, filenames, work, direction, path):
    rows, u, passed, K, steps = [], [], [], [], []
    for k in range(len(solid)):
        z = np.load(work / f'{k}.npz')
        meta = json.loads(str(z['meta']))
        ux = z["ux"]
        u.append((ux.T if direction == 'y' else ux).astype(np.float32))   # driving-direction component
        passed.append(meta['passed'])
        K.append(meta['K_pixel2'])
        steps.append(meta['steps'])
        rows.append(dict(filename=str(filenames[k]), **meta))
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, solid=solid.astype(np.uint8), u=np.stack(u), passed=np.array(passed),
                        K=np.array(K), steps=np.array(steps), filenames=np.array(filenames))
    os.replace(tmp, path)
    with open(path.with_suffix('.log.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{path}: {sum(passed)}/{len(passed)} simulations passed')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slices', required=True)
    ap.add_argument('--direction', choices=['x', 'y'], required=True)
    ap.add_argument('--out', default='labels')
    ap.add_argument('--name', help='output name, default: file name of --slices')
    ap.add_argument('--r', type=int, default=4, help='lattice nodes per pixel along each axis')
    ap.add_argument('--scheme', choices=['trt', 'bgk'], default='trt')
    a = ap.parse_args()
    d = np.load(a.slices)
    solid, filenames = d['solid'].astype(np.uint8), [str(x) for x in d['filenames']]
    name = a.name or Path(a.slices).stem
    out = Path(a.out)
    work = out / 'work' / f'{name}_{a.direction}'
    masks = np.ascontiguousarray(solid.transpose(0, 2, 1) if a.direction == 'y' else solid)
    simulate(masks, work, a.r, a.scheme)
    export(solid, filenames, work, a.direction, out / f'{name}_{a.direction}.npz')


if __name__ == '__main__':
    main()
