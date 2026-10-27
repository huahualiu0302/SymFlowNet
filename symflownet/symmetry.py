"""Transpose and reflection symmetry of creeping flow in a slice (Sect. 2.2, Eqs. (1)-(3), (7)).

A transform is op = (up-down flip, left-right flip, transpose). It acts on a dict with the pore image 'p' and the
x- and y-direction fields 'x' and 'y'. Transposing exchanges the two axes, so the transposed y-direction field is the
x-direction field of the transposed slice (Eq. (1)) and the two fields swap. The flips keep each field in its own
direction (Eq. (2)).
"""
import numpy as np
import torch

OPS = [(fv, fh, tr) for tr in (0, 1) for fv in (0, 1) for fh in (0, 1)]
FLIPS = [op for op in OPS if not op[2]]          # the four operations Gamma of Eq. (2)


def op_apply(d, op):
    fv, fh, tr = op
    out = {}
    for k, v in d.items():
        if v is not None:
            if fv:
                v = torch.flip(v, [-2])
            if fh:
                v = torch.flip(v, [-1])
            if tr:
                v = v.transpose(-1, -2)
        out[k] = v
    if tr:
        out['x'], out['y'] = out.get('y'), out.get('x')
    return out


def op_inverse(d, op):
    fv, fh, tr = op
    out = dict(d)
    if tr:
        out['x'], out['y'] = out.get('y'), out.get('x')
    for k, v in out.items():
        if v is None:
            continue
        if tr:
            v = v.transpose(-1, -2)
        if fh:
            v = torch.flip(v, [-1])
        if fv:
            v = torch.flip(v, [-2])
        out[k] = v
    return out


def predict(nets, data, idx, target, average=True, chunk=25):
    """Normalized prediction of the target field ('x' or 'y') for the slices idx, as a numpy array.

    nets['x'] predicts the x-direction field and nets['y'] the y-direction field, both called as net(field, pore),
    where field is the simulated field of the other direction (used only by the translation models).
    With average=True the prediction is averaged over the transforms in OPS (Eq. (7)): each output is transformed back
    and converted to velocity before averaging. For SymFlowNet nets['y'] is the transposed x-direction network, so
    the transposed transforms repeat the four flips exactly and the average equals the average over the four flips.
    """
    src = 'y' if target == 'x' else 'x'
    ops = OPS if average else [(0, 0, 0)]
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), chunk):
            d = data.get(idx[k:k + chunk])
            acc = 0
            for op in ops:
                e = op_apply(dict(p=d['p'], x=d['x'] if src == 'x' else None, y=d['y'] if src == 'y' else None), op)
                tgt = 'y' if e['x'] is not None else 'x'
                e[tgt] = nets[tgt](e['x' if tgt == 'y' else 'y'], e['p'])
                acc = acc + torch.sinh(op_inverse(e, op)[target] * data.lam)
            out.append((torch.asinh(acc / len(ops)) / data.lam).cpu().numpy()[:, 0])
    return np.concatenate(out)
