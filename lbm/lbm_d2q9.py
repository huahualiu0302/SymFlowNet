"""D2Q9 lattice Boltzmann solver for the velocity labels (Sect. 3.2, Table 2), on the GPU with NVIDIA Warp.

A batch of slices is simulated at once. The boundaries and parameters follow Zhao et al., J. Hydrol. 639, 131501 (2024):
  * D2Q9, kinematic viscosity 0.2 (tau = 1.1), pressure gradient 3.94e-6 on the original grid, rho_out = 1, p = rho/3;
  * one layer around the slice: solid walls in the top and bottom rows, constant-density (Zou-He) boundaries in the
    left and right columns, corners solid;
  * halfway bounce-back; the simulation starts from rest and the inlet density is raised over 2000 r^2 steps.
Lattice refinement r: each pixel becomes r x r nodes and the boundary layer is r nodes thick. The pressure gradient is
3.94e-6 / r^3 (diffusive scaling), so the refined lattice solves the same problem as the original grid. The results are
returned in the lattice units of the original grid: u = r u_fine, rho - 1 = r^2 (rho_fine - 1).
Collision: TRT with magic parameter 3/16 (default) or BGK, which is TRT with tau- = tau+. With TRT the halfway
bounce-back wall lies exactly midway between solid and fluid nodes, independent of the viscosity.
Convergence (each slice separately): checks every 200 r^2 + 1 steps (an odd interval, so that an odd-even oscillation
is not missed); converged after three consecutive checks with relative velocity change < 1e-7, geometrically
extrapolated remaining error < 1e-5 and relative spread of the fluid-link mass flux across vertical faces < 1e-3, after
the ramp. A slice that carries no flow (rms velocity < 1e-12) is also converged.
"""
import math
import time

import numpy as np
import warp as wp

wp.config.quiet = True

REAL = wp.float64
vec9 = wp.types.vector(length=9, dtype=REAL)

# D2Q9: 0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE; y is the row index
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], np.int32)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, np.float64)

NU = 0.2
TAU = 3 * NU + 0.5
GRADP = 3.94e-6
RHO_OUT = 1.0
RAMP = 2000          # ramp steps on the original grid, multiplied by r^2
CHECK = 200          # check interval on the original grid, 200 r^2 + 1 on the refined lattice
VEL_TOL = 1e-7
EST_TOL = 1e-5
FLUX_TOL = 1e-3
REST_TOL = 1e-12
MAX_STEPS = 2_000_000  # iteration limit on the original grid, multiplied by r


def omegas(scheme):
    om_p = 1.0 / TAU
    if scheme == 'bgk':
        return om_p, om_p
    if scheme == 'trt':
        tau_m = 0.5 + (3.0 / 16.0) / (TAU - 0.5)
        return om_p, 1.0 / tau_m
    raise ValueError(scheme)


@wp.kernel
def k_step(gin: wp.array4d(dtype=REAL), gout: wp.array4d(dtype=REAL),
           solid: wp.array3d(dtype=wp.uint8), active: wp.array(dtype=wp.int32),
           t0: wp.array(dtype=wp.int32), ramp: wp.array(dtype=wp.int32),
           rin_t: wp.array(dtype=REAL), rout: REAL, t: wp.int32,
           om_p: REAL, om_m: REAL,
           cx: wp.array(dtype=wp.int32), cy: wp.array(dtype=wp.int32),
           opp: wp.array(dtype=wp.int32), w: wp.array(dtype=REAL),
           write: wp.int32,
           rho_o: wp.array3d(dtype=REAL), ux_o: wp.array3d(dtype=REAL), uy_o: wp.array3d(dtype=REAL)):
    b, y, x = wp.tid()
    if active[b] == 0:
        return
    ny = solid.shape[1]
    nx = solid.shape[2]
    zero = REAL(0.0)
    one = REAL(1.0)
    if solid[b, y, x] != wp.uint8(0):
        for i in range(9):
            gout[b, i, y, x] = w[i]
        if write != 0:
            rho_o[b, y, x] = one
            ux_o[b, y, x] = zero
            uy_o[b, y, x] = zero
        return

    # pull streaming; a population coming from a solid node is the post-collision opposite one (halfway bounce-back)
    f = vec9()
    for i in range(9):
        sx = x - cx[i]
        sy = y - cy[i]
        if sx < 0:
            sx = sx + nx
        if sx >= nx:
            sx = sx - nx
        if sy < 0:
            sy = sy + ny
        if sy >= ny:
            sy = sy - ny
        if solid[b, sy, sx] != wp.uint8(0):
            f[i] = gin[b, opp[i], y, x]
        else:
            f[i] = gin[b, i, sy, sx]

    r = zero
    for i in range(9):
        r = r + f[i]
    u = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8]) / r
    v = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]) / r

    half = REAL(0.5)
    two3 = REAL(2.0 / 3.0)
    one6 = REAL(1.0 / 6.0)
    if x == 0:
        k = REAL(t - t0[b])
        frac = wp.min(one, k / REAL(wp.max(ramp[b], 1)))
        r = rout + (rin_t[b] - rout) * (half - half * wp.cos(REAL(3.141592653589793) * frac))
        u = one - (f[0] + f[2] + f[4] + REAL(2.0) * (f[3] + f[6] + f[7])) / r
        v = zero
        f[1] = f[3] + two3 * u * r
        f[5] = f[7] - half * (f[2] - f[4]) + one6 * u * r
        f[8] = f[6] + half * (f[2] - f[4]) + one6 * u * r
    elif x == nx - 1:
        r = rout
        u = -one + (f[0] + f[2] + f[4] + REAL(2.0) * (f[1] + f[5] + f[8])) / r
        v = zero
        f[3] = f[1] - two3 * u * r
        f[6] = f[8] - half * (f[2] - f[4]) - one6 * u * r
        f[7] = f[5] + half * (f[2] - f[4]) - one6 * u * r

    if write != 0:
        rho_o[b, y, x] = r
        ux_o[b, y, x] = u
        uy_o[b, y, x] = v

    usq = u * u + v * v
    for i in range(9):
        j = opp[i]
        cu = REAL(cx[i]) * u + REAL(cy[i]) * v
        e_p = w[i] * r * (one + REAL(4.5) * cu * cu - REAL(1.5) * usq)
        e_m = w[i] * r * REAL(3.0) * cu
        g_p = half * (f[i] + f[j])
        g_m = half * (f[i] - f[j])
        gout[b, i, y, x] = f[i] - om_p * (g_p - e_p) - om_m * (g_m - e_m)


@wp.kernel
def k_check(ux: wp.array3d(dtype=REAL), uy: wp.array3d(dtype=REAL),
            uxp: wp.array3d(dtype=REAL), uyp: wp.array3d(dtype=REAL),
            solid: wp.array3d(dtype=wp.uint8), active: wp.array(dtype=wp.int32),
            acc: wp.array2d(dtype=REAL)):
    b, y, x = wp.tid()
    if active[b] == 0 or solid[b, y, x] != wp.uint8(0):
        return
    du = ux[b, y, x] - uxp[b, y, x]
    dv = uy[b, y, x] - uyp[b, y, x]
    wp.atomic_add(acc, b, 0, du * du + dv * dv)
    wp.atomic_add(acc, b, 1, ux[b, y, x] * ux[b, y, x] + uy[b, y, x] * uy[b, y, x])
    uxp[b, y, x] = ux[b, y, x]
    uyp[b, y, x] = uy[b, y, x]


@wp.kernel
def k_flux(g: wp.array4d(dtype=REAL), solid: wp.array3d(dtype=wp.uint8),
           active: wp.array(dtype=wp.int32), flux: wp.array2d(dtype=REAL)):
    """Net mass carried by fluid-fluid links across every vertical lattice face, after collision."""
    b, y, x = wp.tid()
    if active[b] == 0 or solid[b, y, x] != wp.uint8(0):
        return
    ny = solid.shape[1]
    nx = solid.shape[2]
    if x < nx - 1:
        if solid[b, y, x + 1] == wp.uint8(0):
            wp.atomic_add(flux, b, x, g[b, 1, y, x])
        if y + 1 < ny:
            if solid[b, y + 1, x + 1] == wp.uint8(0):
                wp.atomic_add(flux, b, x, g[b, 5, y, x])
        if y - 1 >= 0:
            if solid[b, y - 1, x + 1] == wp.uint8(0):
                wp.atomic_add(flux, b, x, g[b, 8, y, x])
    if x > 0:
        if solid[b, y, x - 1] == wp.uint8(0):
            wp.atomic_add(flux, b, x - 1, -g[b, 3, y, x])
        if y + 1 < ny:
            if solid[b, y + 1, x - 1] == wp.uint8(0):
                wp.atomic_add(flux, b, x - 1, -g[b, 6, y, x])
        if y - 1 >= 0:
            if solid[b, y - 1, x - 1] == wp.uint8(0):
                wp.atomic_add(flux, b, x - 1, -g[b, 7, y, x])


@wp.kernel
def k_fill(g: wp.array4d(dtype=REAL), src: wp.array3d(dtype=REAL), slot: wp.int32):
    i, y, x = wp.tid()
    g[slot, i, y, x] = src[i, y, x]


def domain(rock, r):
    """rock: slice with 1 in the grains. Returns the refined domain with the boundary layer, 1 = solid."""
    rock = np.asarray(rock).astype(bool)
    core = np.kron(rock, np.ones((r, r), bool))
    ny, nx = core.shape
    m = np.zeros((ny + 2 * r, nx + 2 * r), bool)
    m[r:-r, r:-r] = core
    m[:r, :] = True
    m[-r:, :] = True
    return m


def equilibrium(rho, ux, uy):
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux ** 2 + uy ** 2
    return W[:, None, None] * rho[None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usq[None])


def initial_populations(mask):
    """Start from rest (rho = 1, u = 0); the inlet density is then raised with a cosine ramp.

    A coarse-grid solution is not used as the initial field: it excites a checkerboard mode along x that does not decay.
    """
    f = np.repeat(W[:, None, None], mask.shape[0], axis=1)
    return np.repeat(f[:, :, :1], mask.shape[1], axis=2).copy()


def block_mean(a, r):
    return a.reshape(a.shape[0] // r, r, a.shape[1] // r, r).mean(axis=(1, 3))


class Batch:
    """A batch of slices with the same refinement and collision operator. A finished slot is refilled at once."""

    def __init__(self, r, scheme, slots, device=None, ny0=200, nx0=200):
        self.r, self.scheme = r, scheme
        self.device = device or ('cuda:0' if wp.is_cuda_available() else 'cpu')
        self.ny, self.nx = (ny0 + 2) * r, (nx0 + 2) * r
        self.B = slots
        d, B, ny, nx = self.device, slots, self.ny, self.nx
        self.gA = wp.zeros((B, 9, ny, nx), dtype=REAL, device=d)
        self.gB = wp.zeros((B, 9, ny, nx), dtype=REAL, device=d)
        self.rho = wp.zeros((B, ny, nx), dtype=REAL, device=d)
        self.ux = wp.zeros((B, ny, nx), dtype=REAL, device=d)
        self.uy = wp.zeros((B, ny, nx), dtype=REAL, device=d)
        self.uxp = wp.zeros((B, ny, nx), dtype=REAL, device=d)
        self.uyp = wp.zeros((B, ny, nx), dtype=REAL, device=d)
        self.solid_np = np.ones((B, ny, nx), np.uint8)
        self.solid = wp.array(self.solid_np, dtype=wp.uint8, device=d)
        self.acc = wp.zeros((B, 2), dtype=REAL, device=d)
        self.flux = wp.zeros((B, nx - 1), dtype=REAL, device=d)
        self.active_np = np.zeros(B, np.int32)
        self.active = wp.array(self.active_np, dtype=wp.int32, device=d)
        self.t0_np = np.zeros(B, np.int32)
        self.t0 = wp.array(self.t0_np, dtype=wp.int32, device=d)
        self.ramp_np = np.zeros(B, np.int32)
        self.ramp = wp.array(self.ramp_np, dtype=wp.int32, device=d)
        g_r = GRADP / r ** 3
        self.rin_target = RHO_OUT + 3.0 * (nx - 1) * g_r
        self.rin = wp.array(np.full(B, self.rin_target), dtype=REAL, device=d)
        self.cx = wp.array(CX, dtype=wp.int32, device=d)
        self.cy = wp.array(CY, dtype=wp.int32, device=d)
        self.opp = wp.array(OPP, dtype=wp.int32, device=d)
        self.w = wp.array(W, dtype=REAL, device=d)
        self.om_p, self.om_m = omegas(scheme)
        self.t = 0
        self.check = CHECK * r * r + 1
        self.max_steps = MAX_STEPS * r
        self.slot = [None] * B

    def fill(self, s, case_id, rock, meta=None):
        mask = domain(rock, self.r)
        f = initial_populations(mask)
        src = wp.array(f, dtype=REAL, device=self.device)
        wp.launch(k_fill, dim=(9, self.ny, self.nx), inputs=[self.gA, src, s], device=self.device)
        self.solid_np[s] = mask
        self.active_np[s] = 1
        self.t0_np[s] = self.t
        self.ramp_np[s] = RAMP * self.r ** 2
        wp.copy(self.solid[s], wp.array(self.solid_np[s], dtype=wp.uint8, device=self.device))
        self._sync_small()
        self.slot[s] = dict(case_id=case_id, rock=np.asarray(rock).astype(np.uint8), mask=mask,
                            meta=meta or {}, hist=[], stable=0, ch_prev=None,
                            start=time.time(), init='rest',
                            n_fluid=int((~mask).sum()))

    def release(self, s):
        self.active_np[s] = 0
        self.slot[s] = None
        self._sync_small()

    def _sync_small(self):
        d = self.device
        wp.copy(self.active, wp.array(self.active_np, dtype=wp.int32, device=d))
        wp.copy(self.t0, wp.array(self.t0_np, dtype=wp.int32, device=d))
        wp.copy(self.ramp, wp.array(self.ramp_np, dtype=wp.int32, device=d))

    def step(self, n):
        """Advance n steps; the macroscopic fields are written in the last step."""
        for k in range(n):
            self.t += 1
            wp.launch(k_step, dim=(self.B, self.ny, self.nx), device=self.device,
                      inputs=[self.gA, self.gB, self.solid, self.active, self.t0, self.ramp,
                              self.rin, RHO_OUT, self.t, self.om_p, self.om_m,
                              self.cx, self.cy, self.opp, self.w, int(k == n - 1),
                              self.rho, self.ux, self.uy])
            self.gA, self.gB = self.gB, self.gA

    def diagnose(self):
        """Convergence quantities of every active slot."""
        self.acc.zero_()
        self.flux.zero_()
        dim = (self.B, self.ny, self.nx)
        wp.launch(k_check, dim=dim, device=self.device,
                  inputs=[self.ux, self.uy, self.uxp, self.uyp, self.solid, self.active, self.acc])
        wp.launch(k_flux, dim=dim, device=self.device,
                  inputs=[self.gA, self.solid, self.active, self.flux])
        acc = self.acc.numpy()
        flux = self.flux.numpy()
        out = {}
        for s in range(self.B):
            st = self.slot[s]
            if st is None:
                continue
            steps = self.t - int(self.t0_np[s])
            du2, u2 = acc[s]
            ch = math.sqrt(du2 / u2) if u2 > 0 else 0.0
            if steps <= self.check:
                ch = math.inf     # nothing to compare with at the first check
            q = flux[s]
            qmax = np.abs(q).max()
            spread = float((q.max() - q.min()) / qmax) if qmax > 0 else 0.0
            rms = math.sqrt(u2 / max(st['n_fluid'], 1))
            est = math.inf
            if st['ch_prev'] is not None and math.isfinite(ch) and st['ch_prev'] > 0:
                ratio = ch / st['ch_prev']
                if ratio < 0.999:
                    est = ch * ratio / (1 - ratio)
            st['ch_prev'] = ch if math.isfinite(ch) else None
            near_rest = rms < REST_TOL
            ok = near_rest or (ch < VEL_TOL and spread < FLUX_TOL and (est < EST_TOL or ch < 1e-11))
            st['stable'] = st['stable'] + 1 if ok else 0
            st['hist'].append((steps, ch, est, spread, float(q.mean()), rms))
            done = (st['stable'] >= 3 and steps >= self.ramp_np[s]) or steps >= self.max_steps
            out[s] = dict(steps=steps, ch=ch, est=est, spread=spread, rms=rms,
                          converged=bool(st['stable'] >= 3 and steps >= self.ramp_np[s]), done=done)
        return out

    def result(self, s):
        """Fields of one slot in the lattice units of the original grid, averaged over the r x r nodes of each pixel."""
        r = self.r
        st = self.slot[s]
        rho = self.rho[s].numpy().astype(np.float64)
        ux = self.ux[s].numpy().astype(np.float64)
        uy = self.uy[s].numpy().astype(np.float64)
        mask = st['mask']
        ux[mask] = 0.0
        uy[mask] = 0.0
        rho[mask] = 1.0
        ux_c, uy_c = ux * r, uy * r
        drho_c = (rho - 1.0) * r * r
        core = (slice(r, -r), slice(r, -r))
        ux200 = block_mean(ux_c[core], r)
        uy200 = block_mean(uy_c[core], r)
        drho200 = block_mean(drho_c[core], r)
        K = NU * float(np.mean((1.0 + drho_c[core]) * ux_c[core])) / GRADP
        speed = np.hypot(ux, uy)
        return dict(
            case_id=st['case_id'], r=r, scheme=self.scheme,
            ux200=ux200, uy200=uy200, rho200=1.0 + drho200,
            fine_ux=ux_c[core].astype(np.float32), fine_uy=uy_c[core].astype(np.float32),
            fine_drho=drho_c[core].astype(np.float32),
            K_pixel2=K, max_mach=float(speed.max() * math.sqrt(3.0)),
            solid_nonzero=int(np.count_nonzero(speed[mask])),
            finite=bool(np.isfinite(rho).all() and np.isfinite(ux).all() and np.isfinite(uy).all()
                        and (rho > 0).all()),
            seconds=time.time() - st['start'], init=st['init'],
            hist=np.array(st['hist'], np.float64))
