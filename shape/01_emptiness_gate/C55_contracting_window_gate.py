# PROVENANCE: 04_calculations/code/20261009_claude_R5/C55_contracting_window_gate.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Reheating coupling in the time-symmetric past (D48 Sec. 2): visible-photon energy produced on the remote contracting passage of a gauge-kinetic window, R = rho_gamma/|rho_C| = 10^93.9 (j = 1, free Casimir; x 1/(1-xi) -> 10^94.0 at xi = 0.30). Units h = 1, M_P = 1. Paper: PRD Sec. VI B (sec:s-reheating, Eq. eq:s-windowR, Fig. fig:s-gate a), App. app:gate. Pool size --procs (default env DNEC_WORKERS or 10).

C55: remote contracting passage of the visible gauge window (direction (a), round R5).

Independent of C49: own background integrator, own mode variables (A, Pi_A) with
action S = 1/2 int deta I^2 (A'^2 - k^2 A^2), integration on the CONTRACTING branch
forward in time, readout after the window has closed (I = 1 exactly, conformal
Maxwell), where the occupation is unambiguous.

Model (D42 weak-visible branch): V/(Mp^2 h^2) = 3 (1 - exp(-sqrt(2/3) phi))^2,
Casimir rho_C = -A Mp^2 h^2 exp(-4n), A = V(phi_b), n = ln(L/L_b), tau = h t,
time-symmetric central bounce H = phidot = 0 at tau = 0.
Gauge kinetic function K = I^2 = 1 + c phi f(|phi|), c = 0.01, flat-bump window.

Output: comoving visible-photon energy produced in the contracting passage,
compared with the comoving Casimir energy A that must support the bounce.
R = e^{4n} rho_gamma / |rho_C| is constant after the window closes (both scale
as a^-4), so R >= 1 means the produced radiation forbids H = 0 at the bounce.

Usage:  python C55_contracting_window_gate.py [--quick]
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, hashlib, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C55_window_gate')
PHIB = 5.475227427784655          # C43 weak_visible_c0.01 calibration
HMP = 6.746747288599453e-06       # h / Mp
C = 0.01
S23 = np.sqrt(2.0 / 3.0)


def V(phi):
    return 3.0 * (1.0 - np.exp(-S23 * phi)) ** 2


def Vp(phi):
    e = np.exp(-S23 * phi)
    return 6.0 * S23 * e * (1.0 - e)


A_CAS = V(PHIB)


def window_f(phi):
    """f(|phi|): 1 for |phi|<=0.5, 0 for |phi|>=1.5, flat-bump interpolation.
    Written independently of C49: f = g(1-x)/(g(x)+g(1-x)) with g(x)=exp(-1/x)."""
    x = abs(phi) - 0.5
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    # f = 1 - s, s = e^{-1/x}/(e^{-1/x}+e^{-1/(1-x)}) = 1/(1+exp(1/x-1/(1-x)))
    d = 1.0 / x - 1.0 / (1.0 - x)
    return 1.0 - 1.0 / (1.0 + np.exp(d)) if d < 700 else 1.0


def I2(phi):
    return 1.0 + C * phi * window_f(phi)


# ---------------------------------------------------------------- background
class Background:
    """Expanding branch tau >= 0 from the bounce; contracting branch by symmetry."""

    def __init__(self, tmax=262.0, rtol=1e-12):
        def rhs(t, y):
            n, H, phi, pd = y
            return [H, -0.5 * pd * pd + (2.0 / 3.0) * A_CAS * np.exp(-4.0 * n),
                    pd, -3.0 * H * pd - Vp(phi)]

        def ev_end(t, y):          # epsilon_H = 1  <=>  phidot^2/2 = H^2 (units Mp=h=1)
            return 0.5 * y[3] ** 2 - y[1] ** 2 if y[1] > 0.05 else -1.0
        ev_end.direction = 1

        def ev_turn(t, y):
            return y[3]

        self.sol = solve_ivp(rhs, (0.0, tmax), [0.0, 0.0, PHIB, 0.0], method='DOP853',
                             rtol=rtol, atol=rtol * 1e-2, max_step=0.05,
                             events=[ev_end, ev_turn], dense_output=True)
        assert self.sol.success
        self.t_end = float(self.sol.t_events[0][0])
        self.n_end = float(self.sol.sol(self.t_end)[0])
        self.t_on = brentq(lambda t: self.sol.sol(t)[2] - 1.5, 1.0, self.t_end, xtol=1e-13)
        self.turns = [float(t) for t in self.sol.t_events[1] if t > self.t_end]
        # independent cross-check with a different integrator
        sol2 = solve_ivp(rhs, (0.0, 120.0), [0.0, 0.0, PHIB, 0.0], method='Radau',
                         rtol=1e-11, atol=1e-13, dense_output=True)
        tt = np.linspace(1.0, 120.0, 400)
        self.crosscheck_maxdiff = float(np.max(np.abs(sol2.sol(tt) - self.sol.sol(tt))))
        grid = np.linspace(0.0, tmax, 20001)
        y = self.sol.sol(grid)
        self.constraint_max = float(np.max(np.abs(3 * y[1] ** 2 - 0.5 * y[3] ** 2 - V(y[2])
                                                   + A_CAS * np.exp(-4 * y[0]))))

    def state(self, tau):
        """Contracting branch for tau < 0 by time reflection."""
        if tau >= 0:
            return self.sol.sol(tau)
        n, H, phi, pd = self.sol.sol(-tau)
        return np.array([n, -H, phi, -pd])


# ---------------------------------------------------------------- modes
_BG = None


def _bg():
    global _BG
    if _BG is None:
        _BG = Background()
    return _BG


def run_mode(args):
    """Vacuum declared at contracting time tau_i (a scalar turning point, I'=0);
    integrate forward to tau_f (window closed, I = 1). Return occupation n_q.
    Units: comoving q = k/(a_end h); atil = a/a_end = exp(n - n_end);
    d eta~ = d tau / atil;  dA/deta~ = Pi / I^2,  dPi/deta~ = -q^2 I^2 A."""
    q, tau_i, tau_f, rtol, state = args
    bg = _bg()

    def rhs(t, y):
        n, H, phi, pd = bg.state(t)
        at = np.exp(n - bg.n_end)
        i2 = I2(phi)
        Ar, Ai, Pr, Pi_ = y
        return [Pr / (at * i2), Pi_ / (at * i2), -q * q * i2 * Ar / at, -q * q * i2 * Ai / at]

    phi_i = bg.state(tau_i)[2]
    i2 = I2(phi_i)
    # positive-frequency instantaneous vacuum at a turning point: A = 1/(I sqrt(2q)), Pi = -i q I^2 A
    A0 = 1.0 / np.sqrt(i2 * 2.0 * q)
    if state == 'instantaneous':
        P0 = -1j * q * i2 * A0
    else:
        raise ValueError(state)
    y0 = [A0, 0.0, 0.0, (P0).imag]
    # natural max step: resolve the mode period and the scalar oscillation
    n_i = bg.state(tau_i)[0]
    at_min = np.exp(min(n_i, bg.state(tau_f)[0]) - bg.n_end)
    period = 2 * np.pi * at_min / q
    ms = min(0.05, period / 12.0)
    sol = solve_ivp(rhs, (tau_i, tau_f), y0, method='DOP853', rtol=rtol, atol=rtol * 1e-3, max_step=ms)
    if not sol.success:
        return {'q': q, 'tau_i': tau_i, 'ok': False, 'msg': sol.message}
    Ar, Ai, Pr, Pi_ = sol.y[:, -1]
    Aq = Ar + 1j * Ai
    Pq = Pr + 1j * Pi_
    i2f = I2(bg.state(tau_f)[2])
    wron = (Aq * np.conj(Pq) - np.conj(Aq) * Pq)   # should equal i (conserved)
    E = 0.5 * (abs(Pq) ** 2 / i2f + q * q * i2f * abs(Aq) ** 2)
    nq = E / q - 0.5
    return {'q': q, 'tau_i': tau_i, 'ok': True, 'n': float(nq), 'wronskian_err': float(abs(wron - 1j)),
            'nfev': int(sol.nfev)}


def band_energy(qs, ns):
    """int q^4 n d ln q with trapezoid in ln q."""
    lq = np.log(qs)
    f = qs ** 4 * np.maximum(ns, 0.0)
    return float(np.trapezoid(f, lq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--procs', type=int, default=int(os.environ.get('DNEC_WORKERS', '10')))  # package: --workers
    ap.add_argument('--nq', type=int, default=97)
    ap.add_argument('--qmin', type=float, default=1e-3)
    ap.add_argument('--qmax', type=float, default=100.0)
    ap.add_argument('--rtol', type=float, default=1e-10)
    ap.add_argument('--slices', type=str, default='1,2,5,10,20,40,85')
    ap.add_argument('--label', type=str, default='')
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bg = _bg()
    tau_f = -bg.t_on + 0.0  # window closes exactly at |phi| = 1.5 on the climb
    # sanity: at tau_f the field is at 1.5 and climbing toward the bounce
    meta = {'phi_b': PHIB, 'h_over_Mp': HMP, 'c': C, 'A_Casimir_hat': A_CAS,
            't_end': bg.t_end, 'n_end': bg.n_end, 't_on': bg.t_on, 'tau_f_contracting': tau_f,
            'n_turns_stored': len(bg.turns), 'first_turns': bg.turns[:5],
            'background_constraint_max': bg.constraint_max,
            'Radau_vs_DOP853_maxdiff_0_120': bg.crosscheck_maxdiff}
    # prefactor: R = (12/pi^2)(h/Mp)^2 e^{4 n_end} int q^4 n dlnq / A
    pref = (12.0 / np.pi ** 2) * HMP ** 2 * np.exp(4.0 * bg.n_end) / A_CAS
    meta['log10_prefactor'] = float(np.log10(12.0 / np.pi ** 2) + 2 * np.log10(HMP)
                                    + 4.0 * bg.n_end / np.log(10.0) - np.log10(A_CAS))
    # declared-vacuum slices: contracting-branch turning points -t_j
    js = [int(s) for s in args.slices.split(',')] if not args.quick else [1, 5]
    js = [j for j in js if j <= len(bg.turns)]
    nq = 25 if args.quick else args.nq
    qs = np.geomspace(args.qmin, args.qmax, nq)
    meta.update(nq=nq, qmin=args.qmin, qmax=args.qmax, rtol=args.rtol)
    jobs = []
    for j in js:
        tau_i = -bg.turns[j - 1]
        for q in qs:
            jobs.append((float(q), tau_i, tau_f, args.rtol, 'instantaneous'))
    with Pool(args.procs) as pool:
        rows = pool.map(run_mode, jobs, chunksize=1)
    out = {'meta': meta, 'slices': []}
    for j in js:
        tau_i = -bg.turns[j - 1]
        rr = [r for r in rows if r['tau_i'] == tau_i]
        ok = all(r['ok'] for r in rr)
        qv = np.array([r['q'] for r in rr])
        nv = np.array([r.get('n', np.nan) for r in rr])
        Iq = band_energy(qv, nv)
        R = pref * Iq
        out['slices'].append({'turning_point_index_before_climb': j, 'tau_i': tau_i,
                              'all_ok': ok, 'max_wronskian_err': float(max(r.get('wronskian_err', np.nan) for r in rr)),
                              'int_q4_n_dlnq': Iq, 'R_gamma_over_Casimir': R,
                              'log10_R': float(np.log10(R)) if R > 0 else None,
                              'q': qv.tolist(), 'n_q': nv.tolist()})
        print(f'j={j:3d} tau_i={tau_i:9.3f}  int q^4 n = {Iq:.6e}  log10 R = {np.log10(R):.3f}  ok={ok}', flush=True)
    out['runtime_s'] = time.time() - t0
    out['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out['scope'] = ('free transverse visible gauge modes, 12 vectors x 2 polarizations, on the prescribed '
                    'time-symmetric central scalar history; instantaneous oscillator vacuum declared at a '
                    'contracting-branch scalar turning point; readout after the window closes (I=1), where '
                    'Maxwell is conformal and the occupation is exact; no backreaction')
    fn = OUT / ('C55_quick.json' if args.quick else f'C55_window_gate{args.label}.json')
    fn.write_text(json.dumps(out, indent=1))
    print(json.dumps(meta, indent=1))
    print('runtime', out['runtime_s'])


if __name__ == '__main__':
    main()
