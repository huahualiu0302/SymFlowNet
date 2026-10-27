"""Velocity labels, normalization (Sect. 2.4) and data split (Sect. 3.3).

Label files (written by lbm/run_labels.py), one per rock and driving direction:
    <label_dir>/<rock>_x.npz and <label_dir>/<rock>_y.npz with
        solid      (N, 200, 200) uint8, 1 in the grains
        u          (N, 200, 200) float32, velocity component along the driving direction, lattice units
        passed     (N,) bool, convergence and quality checks of the simulation
        filenames  (N,) str, slice names; the number in the name orders the slices
The y-direction field is the downward flow from the first to the last row.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

NU, GRADP = 0.2, 3.94e-6                  # lattice viscosity and pressure gradient of the labels (Table 2)
BG = 'BanderaGray'
TRAIN_ROCKS = ['Berea', 'Castlegate']     # trained on all of their slices
TEST_ROCKS = ['Parker', 'Bentheimer']     # used only for testing
ROCKS = [BG] + TRAIN_ROCKS + TEST_ROCKS

# Bandera Gray: 1000 slices = 40 horizontal planes x 5 x 5 tiles. Slices 25l+1 ... 25l+25 come from plane l.
# Depth of each plane in the 1000^3 image (voxel index, l = 0 ... 39).
BG_PLANE_DEPTH = [12, 174, 799, 844, 342, 784, 583, 303, 892, 308, 65, 280, 264, 734, 677, 314, 792, 803, 670, 93,
                  195, 877, 445, 46, 113, 158, 546, 223, 355, 3, 372, 957, 170, 641, 362, 201, 801, 570, 298, 447]
TILES_PER_PLANE = 25


def geometry_scale(solid):
    """Velocity scale s of Eq. (4), computed from the pore image only.

    s = |grad p| / nu * mean over the pore pixels of (d - 1/2)^2, where d is the distance to the nearest grain pixel.
    The halfway bounce-back wall lies half a pixel from the pore pixel centres.
    """
    from scipy.ndimage import distance_transform_edt
    pore = ~solid.astype(bool)
    if not pore.any():
        return 1.0
    d = distance_transform_edt(np.pad(pore, 1))[1:-1, 1:-1]
    d = np.clip(d - 0.5, 0, None)
    return float(GRADP / NU * np.mean(d[pore] ** 2))


def permeability(u):
    """Slice permeability in lattice units (pixel^2), Eq. (9): K = nu * mean(u over the whole slice) / |grad p|."""
    return NU * float(np.mean(u, dtype=np.float64)) / GRADP


def depth_split(depth, sizes=(('train', 20), ('val', 8), ('test', 12))):
    """Divide the planes into contiguous depth intervals and assign whole intervals to train, validation and test.

    The largest gap G for which the intervals fill the three subsets exactly is used, so planes of different subsets
    are at least G voxels apart. Within each subset the planes are assigned alternately, in depth order, to the sets
    X (simulated in x) and Y (simulated in y). Returns {plane: (set, subset)} and the minimum gap.
    """
    n = len(depth)
    order = sorted(range(n), key=lambda p: depth[p])
    for gap in range(100, 0, -1):
        segs = [[order[0]]]
        for i in range(1, n):
            if depth[order[i]] - depth[order[i - 1]] >= gap:
                segs.append([order[i]])
            else:
                segs[-1].append(order[i])
        if len(segs) > 13:
            continue
        for put in itertools.product(range(len(sizes)), repeat=len(segs)):
            if all(sum(len(s) for s, q in zip(segs, put) if q == k) == size for k, (_, size) in enumerate(sizes)):
                assign = {}
                for k, (part, _) in enumerate(sizes):
                    planes = sorted((p for s, q in zip(segs, put) if q == k for p in s), key=lambda p: depth[p])
                    for j, p in enumerate(planes):
                        assign[p] = ('X' if j % 2 == 0 else 'Y', part)
                return assign, gap
    raise RuntimeError('no depth split found')


def slice_number(name):
    return int(Path(str(name)).stem)


def build_dataset(label_dir, out_dir, log=print):
    """Read the label files, check them, and write data.npz, split.json and norm.json to out_dir."""
    label_dir, out_dir = Path(label_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    D = {k: [] for k in ('solid', 'ux', 'uy', 'ok', 'rock', 'num', 'scale')}
    for rock in ROCKS:
        X = dict(np.load(label_dir / f'{rock}_x.npz'))
        Y = dict(np.load(label_dir / f'{rock}_y.npz'))
        if list(X['filenames']) != list(Y['filenames']) or not np.array_equal(X['solid'], Y['solid']):
            raise ValueError(f'{rock}: x and y label files do not match')
        for i in range(len(X['filenames'])):
            D['solid'].append(X['solid'][i].astype(np.uint8))
            D['ux'].append(X['u'][i].astype(np.float32))
            D['uy'].append(Y['u'][i].astype(np.float32))
            D['ok'].append(bool(X['passed'][i] and Y['passed'][i]))      # converged in both directions
            D['rock'].append(rock)
            D['num'].append(slice_number(X['filenames'][i]))
            D['scale'].append(geometry_scale(X['solid'][i]))
        log(f'{rock}: {len(X["filenames"])} slices')
    arr = {k: np.array(v) for k, v in D.items()}
    split = make_split(arr, log)

    # Eq. (5): t = asinh(z / z0) / lambda with z = u / s. z0 is the median |z| of the Bandera Gray training slices,
    # and lambda maps their largest |z| to 0.8.
    zs = []
    for key, part in (('ux', 'X'), ('uy', 'Y')):
        for i in split[f'BG_{part}']:
            pore = arr['solid'][i] == 0
            zs.append(np.abs(arr[key][i][pore] / arr['scale'][i]))
    zs = np.concatenate(zs)
    z0 = float(np.median(zs))
    lam = 1.25 * math.asinh(float(zs.max()) / z0)
    norm = dict(z0=z0, lam=lam)
    log(f'normalization: z0 = {z0:.4e}, lambda = {lam:.4f}')
    np.savez(out_dir / 'data.npz', **arr)
    (out_dir / 'split.json').write_text(json.dumps(split, indent=1), encoding='utf-8')
    (out_dir / 'norm.json').write_text(json.dumps(norm, indent=1), encoding='utf-8')


def make_split(arr, log=print):
    """Index lists into data.npz (Sect. 3.3, Table 3).

    X, Y: training slices simulated only in x or only in y. Bandera Gray training planes alternate between the two
          sets, Berea and Castlegate slices are assigned in groups of four consecutive slices.
    val, test_BanderaGray: Bandera Gray planes of the other depth intervals, both directions.
    test_Parker, test_Bentheimer: all slices of the two rocks not used in training.
    Slices whose simulations did not converge are excluded.
    """
    assign, gap = depth_split(BG_PLANE_DEPTH)
    sp = {k: [] for k in ('X', 'Y', 'val', 'test_BanderaGray')}
    bg = sorted((int(i) for i in np.where((arr['rock'] == BG) & arr['ok'])[0]), key=lambda i: arr['num'][i])
    for i in bg:
        s, part = assign[(arr['num'][i] - 1) // TILES_PER_PLANE]
        key = s if part == 'train' else ('val' if part == 'val' else 'test_BanderaGray')
        sp[key].append(i)
    sp['BG_X'] = list(sp['X'])
    sp['BG_Y'] = list(sp['Y'])
    for rock in TRAIN_ROCKS:
        idx = sorted((int(i) for i in np.where((arr['rock'] == rock) & arr['ok'])[0]), key=lambda i: arr['num'][i])
        for i in idx:
            sp['X' if ((arr['num'][i] - 1) // 4) % 2 == 0 else 'Y'].append(i)
    for rock in TEST_ROCKS:
        sp[f'test_{rock}'] = [int(i) for i in np.where((arr['rock'] == rock) & arr['ok'])[0]]
    log(f'split: planes of different subsets are at least {gap} voxels apart; '
        + ', '.join(f'{k} {len(v)}' for k, v in sp.items() if not k.startswith('BG_')))
    return sp


class Data:
    """All slices on the device as tensors: p (pore image), x and y (normalized x- and y-direction fields)."""

    def __init__(self, data_dir, device):
        import torch
        data_dir = Path(data_dir)
        z = np.load(data_dir / 'data.npz')
        self.arr = {k: z[k] for k in z.files}
        self.split = json.loads((data_dir / 'split.json').read_text(encoding='utf-8'))
        nm = json.loads((data_dir / 'norm.json').read_text(encoding='utf-8'))
        self.z0, self.lam = nm['z0'], nm['lam']
        self.device = device
        a = self.arr
        pore = (a['solid'] == 0).astype(np.float32)
        sc = (a['scale'] * self.z0)[:, None, None]

        def t(v):
            return torch.from_numpy(np.ascontiguousarray(v, np.float32))[:, None].to(device)

        self.P = t(pore)
        self.X = t(np.arcsinh(a['ux'] / sc) / self.lam * pore)
        self.Y = t(np.arcsinh(a['uy'] / sc) / self.lam * pore)

    def denorm(self, t, i):
        """Network output of slice i back to velocity in lattice units (inverse of Eqs. (4) and (5))."""
        return np.sinh(t * self.lam) * self.arr['scale'][i] * self.z0

    def get(self, idx):
        import torch
        idx = torch.as_tensor(idx, device=self.device)
        return dict(p=self.P[idx], x=self.X[idx], y=self.Y[idx])
