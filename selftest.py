"""Quick test on synthetic data, on the CPU, without the rock images (a few minutes).

    python selftest.py

1. Checks the symmetry relations: transforming a sample and transforming it back gives the sample, and the
   transposed x-direction network gives the same y-direction field as Eq. (1).
2. Builds synthetic label files with an exact transpose symmetry, prepares the data, trains every model for a few
   iterations, resumes an interrupted run and evaluates the models.
3. Runs the LBM solver on two small slices in both directions if NVIDIA Warp is installed.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import train
from symflownet import data as D
from symflownet.models import FlowNet, Transposed
from symflownet.symmetry import OPS, op_apply, op_inverse

SIZE = 40


def check_symmetry():
    torch.manual_seed(0)
    d = dict(p=torch.rand(2, 1, SIZE, SIZE), x=torch.rand(2, 1, SIZE, SIZE), y=torch.rand(2, 1, SIZE, SIZE))
    for op in OPS:
        back = op_inverse(op_apply(d, op), op)
        assert all(torch.equal(back[k], d[k]) for k in d), op
    net = FlowNet(1, n_down=3).eval()
    p = (torch.rand(1, 1, SIZE, SIZE) > 0.4).float()
    with torch.no_grad():
        uy = Transposed(net)(None, p)
        ref = net(None, p.transpose(-1, -2)).transpose(-1, -2)
    assert torch.allclose(uy, ref)
    print('symmetry relations: ok')


def fake_labels(root):
    """Synthetic slices and fields: u ~ d^2 in the pores, y-direction field from the transposed slice."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    rng = np.random.default_rng(0)

    def flow(solid):
        pore = solid == 0
        q = distance_transform_edt(pore) ** 2 * pore
        return (q / max(q.mean(), 1e-9) * 1e-5).astype(np.float32)

    counts = dict(BanderaGray=1000, Berea=24, Castlegate=24, Parker=12, Bentheimer=12)
    root.mkdir(parents=True)
    for rock, n in counts.items():
        solid = np.stack([(gaussian_filter(rng.random((SIZE, SIZE)), 2) > 0.55).astype(np.uint8) for _ in range(n)])
        names = [f'{k + 1}.png' for k in range(n)]
        for direction in ('x', 'y'):
            u = np.stack([flow(s) if direction == 'x' else flow(s.T).T for s in solid])
            np.savez(root / f'{rock}_{direction}.npz', solid=solid, u=u, passed=np.ones(n, bool), filenames=names)


def check_training(tmp):
    import evaluate
    fake_labels(tmp / 'labels')
    D.build_dataset(tmp / 'labels', tmp / 'data')
    data = D.Data(tmp / 'data', torch.device('cpu'))
    for k in ('X', 'Y', 'val'):                       # keep the test short
        data.split[k] = data.split[k][:40]
    train.HP.update(iters=4, val_every=2, batch=2)
    for model in train.MODELS:
        train.train(model, 0, data, tmp / 'runs')
    # interrupted run: stop after two iterations, resume, and compare with the uninterrupted run
    train.train('symflownet', 0, data, tmp / 'resumed', stop_at=2)
    assert not (tmp / 'resumed' / 'symflownet_s0' / 'done.json').exists()
    train.train('symflownet', 0, data, tmp / 'resumed')
    a = torch.load(tmp / 'runs' / 'symflownet_s0' / 'best.pt', weights_only=False)['nets']['x']
    b = torch.load(tmp / 'resumed' / 'symflownet_s0' / 'best.pt', weights_only=False)['nets']['x']
    assert all(torch.equal(a[k], b[k]) for k in a), 'resumed run differs from the uninterrupted run'
    rows = []
    for rd in sorted(p.parent for p in (tmp / 'runs').glob('*/best.pt')):
        evaluate.evaluate_run(rd, data, rows)
    evaluate.simulated_direction(data, rows)
    evaluate.write_csv(tmp / 'runs' / 'summary.csv', rows)
    print('training, resuming and evaluation: ok')


def check_lbm(tmp):
    try:
        import warp  # noqa: F401
    except ImportError:
        print('NVIDIA Warp not installed, LBM test skipped')
        return
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'lbm'))
    import lbm_d2q9 as L
    import run_labels as R
    solid = np.zeros((2, 24, 24), np.uint8)
    solid[:, 8:14, 6:12] = 1
    solid[1, 2:6, 15:20] = 1
    for direction in ('x', 'y'):
        masks = np.ascontiguousarray(solid.transpose(0, 2, 1) if direction == 'y' else solid)
        L.MAX_STEPS = 50_000
        R.simulate(masks, tmp / 'work' / direction, r=1)
        R.export(solid, ['1.png', '2.png'], tmp / 'work' / direction, direction, tmp / f'lbm_{direction}.npz')
    ux, uy = np.load(tmp / 'lbm_x.npz')['u'], np.load(tmp / 'lbm_y.npz')['u']
    assert (ux[:, solid[0] == 1] == 0).all() and ux.mean() > 0 and uy.mean() > 0
    print('LBM labels in both directions: ok')


if __name__ == '__main__':
    check_symmetry()
    tmp = Path(tempfile.mkdtemp(prefix='symflownet_test_'))
    try:
        check_training(tmp)
        check_lbm(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('all tests passed')
