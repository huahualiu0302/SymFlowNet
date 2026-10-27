"""Train SymFlowNet and the reference models of Table 5.

    python train.py --data data --model symflownet --seed 0

Models (the number of training simulations is given in Table 5 of the paper):
    symflownet           pore image -> x-direction field; set Y enters through the transpose (Eq. (3)), random flips
    separate             one network per direction, each trained on its own set without transpose or flips
    no_flip              SymFlowNet without the flips
    no_cbam              SymFlowNet without CBAM in the decoder
    symflownet_both      SymFlowNet trained on every training slice in both directions
    symflownet_half      SymFlowNet trained on every second slice of X and Y
    symflownet_quarter   SymFlowNet trained on every fourth slice of X and Y
    translation          supervised translation between the two directions, field and pore image as input,
                         trained on every second slice of X and Y simulated in both directions
    translation_all      supervised translation on every training slice in both directions
    cyclegan             unpaired CycleGAN between the fields of X and Y
    cyclegan_semi        CycleGAN with a supervised loss on every tenth slice of X, also simulated in y

The weights with the lowest validation loss (mean absolute error of Eq. (6) on the validation slices in both
directions) are saved to <out>/<model>_s<seed>/best.pt. A checkpoint is written at every validation, and running
the same command again resumes from it.
"""
import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from symflownet.data import Data
from symflownet.models import FlowNet, ImagePool, SNDiscriminator, Transposed
from symflownet.symmetry import FLIPS, OPS, op_apply

HP = dict(ngf=32, ndf=64, n_down=3, n_resnet=6, lr_g=2e-4, lr_d=1e-4, beta1=0.5, beta2=0.999, batch=8,
          iters=25000, val_every=500, lam_cyc=10.0, lam_flux=0.2, lam_sup=10.0, noise=0.02, pool=50)

MODELS = {
    'symflownet': dict(kind='geo'),
    'separate': dict(kind='geo', sym=False, flip=False),
    'no_flip': dict(kind='geo', flip=False),
    'no_cbam': dict(kind='geo', att=False),
    'symflownet_both': dict(kind='geo', pool='both'),
    'symflownet_half': dict(kind='geo', step=2),
    'symflownet_quarter': dict(kind='geo', step=4),
    'translation': dict(kind='sup', pairs='half'),
    'translation_all': dict(kind='sup', pairs='all'),
    'cyclegan': dict(kind='cyc'),
    'cyclegan_semi': dict(kind='cyc', pairs='semi'),
}


def config(model):
    c = dict(kind='geo', att=True, sym=True, flip=True, step=1, pool=None, pairs=None)
    c.update(MODELS[model])
    return c


def subsets(split):
    """Training pools of all models, built from the sets X and Y of the split."""
    S = dict(split)
    S['all'] = S['X'] + S['Y']
    S['half'] = S['X'][::2] + S['Y'][::2]
    S['semi'] = S['X'][::10]
    return S


def build(cfg, device):
    """nets['x'] predicts the x-direction field and nets['y'] the y-direction field."""
    mk = dict(use_attention=cfg['att'], ngf=HP['ngf'], n_resnet=HP['n_resnet'], n_down=HP['n_down'])
    betas = (HP['beta1'], HP['beta2'])
    if cfg['kind'] == 'geo' and cfg['sym']:
        gx = FlowNet(1, **mk).to(device)
        nets = dict(x=gx, y=Transposed(gx))           # Eq. (1)
        params = gx.parameters()
    else:
        in_ch = 1 if cfg['kind'] == 'geo' else 2
        gy = FlowNet(in_ch, **mk).to(device)          # created before the x-direction network, as in the paper runs
        nets = dict(x=FlowNet(in_ch, **mk).to(device), y=gy)
        params = itertools.chain(nets['x'].parameters(), nets['y'].parameters())
    opts = dict(G=torch.optim.Adam(params, lr=HP['lr_g'], betas=betas))
    if cfg['kind'] == 'cyc':
        for k in ('Dx', 'Dy'):
            nets[k] = SNDiscriminator(2, HP['ndf']).to(device)
            opts[k] = torch.optim.Adam(nets[k].parameters(), lr=HP['lr_d'], betas=betas)
    return nets, opts


def trainable(nets):
    return {k: v for k, v in nets.items() if not isinstance(v, Transposed)}


def flux_penalty(t, pore, lam, direction):
    """Variation of the flux through cross-sections normal to the flow (CycleGAN models only).

    Standard deviation of the section fluxes divided by their mean, computed from the scaled velocity z.
    """
    z = torch.sinh(t.clamp(-1, 1) * lam) * pore
    q = z.sum(-1) if direction == 'y' else z.sum(-2)
    return (q.std(-1) / (q.mean(-1).abs() + 0.02 * t.shape[-1])).mean()


def validate(nets, data):
    """Mean absolute error of Eq. (6) over the pore pixels of the validation slices in both directions."""
    for n in trainable(nets).values():
        n.eval()
    idx = data.split['val']
    err = []
    with torch.no_grad():
        d = data.get(idx)
        for tgt, src in (('y', 'x'), ('x', 'y')):
            pr = torch.cat([nets[tgt](d[src][k:k + 25], d['p'][k:k + 25]) for k in range(0, len(idx), 25)])
            err.append(float((pr - d[tgt]).abs()[d['p'] > 0].mean()))
    for n in trainable(nets).values():
        n.train()
    return float(np.mean(err))


def save(path, obj):
    tmp = path.with_name(path.stem + '.tmp.pt')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(model, seed, data, out, log=print, stop_at=None):
    cfg = config(model)
    rd = Path(out) / f'{model}_s{seed}'
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / 'done.json').exists():
        log(f'{rd.name} already trained')
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = data.device
    nets, opts = build(cfg, dev)
    pools = dict(x=ImagePool(HP['pool']), y=ImagePool(HP['pool']))
    start, best, best_it, hist = 0, float('inf'), -1, []
    ck = rd / 'ckpt.pt'
    if ck.exists():
        c = torch.load(ck, map_location=dev, weights_only=False)
        for k, v in trainable(nets).items():
            v.load_state_dict(c['nets'][k])
        for k, v in opts.items():
            v.load_state_dict(c['opts'][k])
        for k, p in pools.items():
            p.images = [t.to(dev) for t in c['pools'][k]]
        start, best, best_it, hist = c['it'], c['best'], c['best_it'], c['hist']
        torch.set_rng_state(c['rng']['torch'])
        random.setstate(c['rng']['py'])
        if dev.type == 'cuda':
            torch.cuda.set_rng_state(c['rng']['cuda'])
        log(f'{rd.name}: resumed at iteration {start}')
    rng = np.random.default_rng([seed, start])
    S, B, lam = subsets(data.split), HP['batch'], data.lam
    pool_x, pool_y = S['X'][::cfg['step']], S['Y'][::cfg['step']]
    if cfg['pool'] == 'both':
        pool_x = pool_y = S['all']
    flips = FLIPS if cfg['flip'] else [(0, 0, 0)]
    l1, mse = F.l1_loss, F.mse_loss

    def draw(lst):
        return data.get(rng.choice(lst, B, replace=len(lst) < B))

    def x_samples(own, other):
        """A batch of x-direction samples: from `own` as simulated, or with probability 1/2 from `other`
        (y-direction simulations) through the transpose, then one of the flips (Eq. (3))."""
        use_other = cfg['sym'] and rng.random() < 0.5
        d = draw(other if use_other else own)
        fv, fh, _ = flips[rng.integers(len(flips))]
        e = op_apply(d, (fv, fh, int(use_other)))
        return e['p'], e['x']

    def y_samples(own, other):
        use_other = cfg['sym'] and rng.random() < 0.5
        d = draw(other if use_other else own)
        fv, fh, _ = flips[rng.integers(len(flips))]
        e = op_apply(d, (fv, fh, int(use_other)))
        return e['p'], e['y']

    def pairs(lst):
        return op_apply(draw(lst), OPS[rng.integers(len(OPS))])

    t_run, times = time.time(), []
    for it in range(start, HP['iters']):
        t0 = time.time()
        if cfg['kind'] == 'geo' and cfg['sym']:
            p, ux = x_samples(pool_x, pool_y)
            loss = l1(nets['x'](None, p), ux)                          # Eq. (6)
            opts['G'].zero_grad(set_to_none=True)
            loss.backward()
            opts['G'].step()
        elif cfg['kind'] == 'geo':                                     # separate networks
            dx, dy = draw(pool_x), draw(pool_y)
            loss = l1(nets['x'](None, dx['p']), dx['x']) + l1(nets['y'](None, dy['p']), dy['y'])
            opts['G'].zero_grad(set_to_none=True)
            loss.backward()
            opts['G'].step()
        elif cfg['kind'] == 'sup':
            d = pairs(S[cfg['pairs']])
            loss = l1(nets['y'](d['x'], d['p']), d['y']) + l1(nets['x'](d['y'], d['p']), d['x'])
            opts['G'].zero_grad(set_to_none=True)
            loss.backward()
            opts['G'].step()
        else:                                                          # CycleGAN
            px, ux = x_samples(S['X'], S['Y'])
            py, uy = y_samples(S['Y'], S['X'])
            fake_y, fake_x = nets['y'](ux, px), nets['x'](uy, py)
            o1, o2 = nets['Dy'](torch.cat([fake_y, px], 1)), nets['Dx'](torch.cat([fake_x, py], 1))
            g_adv = mse(o1, torch.ones_like(o1)) + mse(o2, torch.ones_like(o2))
            nz = HP['noise']
            cyc = (l1(nets['x'](fake_y + nz * torch.randn_like(fake_y) * px, px), ux)
                   + l1(nets['y'](fake_x + nz * torch.randn_like(fake_x) * py, py), uy)) * HP['lam_cyc']
            flux = (flux_penalty(fake_y, px, lam, 'y') + flux_penalty(fake_x, py, lam, 'x')) * HP['lam_flux']
            loss = g_adv + cyc + flux
            if cfg['pairs'] == 'semi':
                d = pairs(S['semi'])
                loss = loss + (l1(nets['y'](d['x'], d['p']), d['y'])
                               + l1(nets['x'](d['y'], d['p']), d['x'])) * HP['lam_sup']
            opts['G'].zero_grad(set_to_none=True)
            loss.backward()
            opts['G'].step()
            for D, real, fake, pool in (('Dx', torch.cat([ux, px], 1), torch.cat([fake_x, py], 1), pools['x']),
                                        ('Dy', torch.cat([uy, py], 1), torch.cat([fake_y, px], 1), pools['y'])):
                r, f = nets[D](real), nets[D](pool.query(fake.detach()))
                ld = (mse(r, torch.ones_like(r)) + mse(f, torch.zeros_like(f))) * 0.5
                opts[D].zero_grad(set_to_none=True)
                ld.backward()
                opts[D].step()
        times.append(time.time() - t0)
        if (it + 1) % HP['val_every'] == 0 or it + 1 == HP['iters']:
            v = validate(nets, data)
            hist.append(dict(it=it + 1, train_loss=float(loss.detach()), val_mae=v))
            if v < best:
                best, best_it = v, it + 1
                save(rd / 'best.pt', dict(nets={k: n.state_dict() for k, n in trainable(nets).items()
                                                if not k.startswith('D')}, model=model, seed=seed, it=it + 1, hp=HP))
            rngs = dict(torch=torch.get_rng_state(), py=random.getstate(),
                        cuda=torch.cuda.get_rng_state() if dev.type == 'cuda' else None)
            save(ck, dict(it=it + 1, best=best, best_it=best_it, hist=hist, rng=rngs,
                          nets={k: n.state_dict() for k, n in trainable(nets).items()},
                          opts={k: o.state_dict() for k, o in opts.items()},
                          pools={k: [t.cpu() for t in p.images] for k, p in pools.items()}))
            rng = np.random.default_rng([seed, it + 1])            # same random sequence with or without resuming
            sec = float(np.median(times[-200:]))
            log(f'{rd.name}: iteration {it + 1}/{HP["iters"]}, validation MAE {v:.5f} (best at {best_it}), '
                f'{sec * 1000:.0f} ms per iteration, {sec * (HP["iters"] - it - 1) / 60:.0f} min left')
            if stop_at and it + 1 >= stop_at and it + 1 < HP['iters']:
                return                                                 # used by selftest.py to test resuming
    (rd / 'done.json').write_text(json.dumps(dict(model=model, seed=seed, best_it=best_it, best_val_mae=best,
                                                  minutes=round((time.time() - t_run) / 60, 1), hist=hist,
                                                  hp=HP), indent=1), encoding='utf-8')
    ck.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data', help='folder with data.npz, split.json and norm.json')
    ap.add_argument('--out', default='runs')
    ap.add_argument('--model', default='symflownet', choices=list(MODELS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=HP['iters'])
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    HP['iters'] = a.iters
    HP['val_every'] = min(HP['val_every'], a.iters)
    dev = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    train(a.model, a.seed, Data(a.data, dev), a.out)


if __name__ == '__main__':
    main()
