"""Evaluate trained models on the test sets in both directions (Sect. 2.6, Tables 6 and 7).

    python evaluate.py --data data --runs runs

For every run in <runs> with a best.pt, and for both directions of the test sets of Bandera Gray, Parker and
Bentheimer, the script writes
    <runs>/<run>/per_slice.csv   per-slice R^2, relative error and permeabilities (lattice units)
    <runs>/summary.csv           pooled R^2, median per-slice R^2, median relative error and median permeability error
The predictions are averaged over the flipped slices (Eq. (7)); 'plain' rows use the original slice only.
The row 'simulated_direction' takes the permeability of the simulated direction as that of the other direction.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from symflownet.data import Data, permeability
from symflownet.symmetry import predict
from train import build, config

TEST_SETS = ['test_BanderaGray', 'test_Parker', 'test_Bentheimer']


def per_slice(u, q, pore):
    e, t = (u - q)[pore], q[pore]
    sst = float(((t - t.mean()) ** 2).sum())
    nt = float(np.linalg.norm(t))
    return dict(r2=1 - float((e ** 2).sum()) / sst if sst > 0 else np.nan,
                rel_err=float(np.linalg.norm(e)) / nt if nt > 0 else np.nan)


def evaluate_run(rd, data, rows):
    name = rd.name
    model, seed = name.rsplit('_s', 1)
    nets, _ = build(config(model), data.device)
    ck = torch.load(rd / 'best.pt', map_location=data.device, weights_only=False)
    for k, v in ck['nets'].items():
        nets[k].load_state_dict(v)
    for n in nets.values():
        if hasattr(n, 'eval'):
            n.eval()
    arr = data.arr
    out = []
    for ts in TEST_SETS:
        idx = data.split[ts]
        for target in ('x', 'y'):
            for mode in ('average', 'plain'):
                tn = predict(nets, data, idx, target, average=mode == 'average')
                errs, trues, sl = [], [], []
                for j, i in enumerate(idx):
                    pore = arr['solid'][i] == 0
                    u = data.denorm(tn[j], i) * pore
                    q = arr['u' + target][i]
                    m = per_slice(u, q, pore)
                    k_true, k_pred = permeability(q), permeability(u)
                    m.update(test_set=ts, direction=target, mode=mode, slice=int(arr['num'][i]),
                             K_lbm=k_true, K_pred=k_pred,
                             K_err=abs(k_pred / k_true - 1) if k_true > 1e-6 else np.nan)
                    sl.append(m)
                    errs.append((u - q)[pore])
                    trues.append(q[pore])
                e, t = np.concatenate(errs), np.concatenate(trues)
                rows.append(dict(run=name, model=model, seed=int(seed), test_set=ts, direction=target, mode=mode,
                                 pooled_r2=1 - float(np.mean(e ** 2)) / float(np.var(t)),
                                 median_slice_r2=float(np.nanmedian([s['r2'] for s in sl])),
                                 median_rel_err=float(np.nanmedian([s['rel_err'] for s in sl])),
                                 median_K_err=float(np.nanmedian([s['K_err'] for s in sl]))))
                out += sl
                print(f'{name} {ts} {target} {mode}: pooled R2 {rows[-1]["pooled_r2"]:.3f}, '
                      f'median K error {rows[-1]["median_K_err"]:.3f}')
    write_csv(rd / 'per_slice.csv', out)


def simulated_direction(data, rows):
    arr = data.arr
    for ts in TEST_SETS:
        for target, src in (('x', 'y'), ('y', 'x')):
            k = [(permeability(arr['u' + src][i]), permeability(arr['u' + target][i])) for i in data.split[ts]]
            rows.append(dict(run='simulated_direction', model='simulated_direction', seed=0, test_set=ts,
                             direction=target, mode='plain',
                             median_K_err=float(np.median([abs(a / b - 1) for a, b in k if b > 1e-6]))))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    data = Data(a.data, dev)
    rows = []
    for rd in sorted(p.parent for p in Path(a.runs).glob('*/best.pt')):
        evaluate_run(rd, data, rows)
    simulated_direction(data, rows)
    write_csv(Path(a.runs) / 'summary.csv', rows)


if __name__ == '__main__':
    main()
