# PROVENANCE: 04_calculations/code/20261009_claude_R5/C62_SM_pivot_calibration.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Pivot calibration of the plateau turning point for the AAA SM (D50): phi_b, N_post, hL_b, delta. Paper: PRD Sec. IV D (sec:turning, eq:pa-benchmark), Sec. VIII D.

C62: pivot calibration of the plateau turning point for the SM all-antiperiodic content (D50).

Background (units t -> h t, phi in M_P, n = ln(L/L_b)):
  V/(M_P^2 h^2) = 3 (1 - exp(-sqrt(2/3) phi))^2,  rho_C = -A M_P^2 h^2 e^{-4n},  A = Vhat(phi_b),
  time-symmetric turning point H = phidot = 0 at phi_b.
Pivot: N_* fixed by instant reheating (55.72234537, D30; N_C cancels from the pivot relation, g_* = 106.75),
h/M_P = 6.1159966044e-6 (A_s at N_*). The torus size today L_0 = 6c/H_0 = 26366.87353598 Mpc gives
k_* L_0 = 1318.3436768 (k_* = 0.05/Mpc). Pivot exit: e^{N_exit} = k_* L_0 / (Hhat_exit h L_b),
h L_b = (h/M_P)^{1/2} (N_C gamma / Vhat_b)^{1/4}, gamma = Z2(cube)/pi^2.
Solve N_post(phi_b) - N_* = N_exit(phi_b). Check: N_C = 100 must reproduce phi_b = 5.57348625, N_post = 67.91072486.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C62_SM_pivot')
OUT.mkdir(parents=True, exist_ok=True)
S23 = np.sqrt(2 / 3)
GAMMA = 16.532315959761668 / np.pi ** 2
NSTAR = 55.72234537
HMP = 6.1159966044e-6
KL0 = 0.05 * 26366.87353598


def V(p):
    return 3 * (1 - np.exp(-S23 * p)) ** 2


def Vp(p):
    e = np.exp(-S23 * p)
    return 6 * S23 * e * (1 - e)


def history(phib):
    A = V(phib)

    def rhs(t, y):
        n, H, p, pd = y
        return [H, -0.5 * pd * pd + (2 / 3) * A * np.exp(-4 * n), pd, -3 * H * pd - Vp(p)]

    def end(t, y):
        return 0.5 * y[3] ** 2 - y[1] ** 2 if y[1] > 0.05 else -1.0
    end.terminal = True; end.direction = 1
    s = solve_ivp(rhs, (0, 400), [0, 0, phib, 0], method='DOP853', rtol=1e-12, atol=1e-14, events=[end], dense_output=True)
    tend = s.t_events[0][0]
    nend = s.sol(tend)[0]
    return s, tend, nend, A


def mismatch(phib, NC):
    s, tend, nend, A = history(phib)
    hLb = np.sqrt(HMP) * (NC * GAMMA / A) ** 0.25
    nexit_target = nend - NSTAR
    if nexit_target <= 0.5:            # not enough inflation to contain the pivot after the bounce
        return -50.0 + nexit_target, dict(phi_b=phib, N_post=nend)
    # Hhat at n = nexit_target
    tt = brentq(lambda t: s.sol(t)[0] - nexit_target, 1e-6, tend)
    Hx = s.sol(tt)[1]
    nexit_pivot = np.log(KL0 / (Hx * hLb))
    return nexit_target - nexit_pivot, dict(phi_b=phib, N_post=nend, N_exit=nexit_target, Hhat_exit=Hx, hLb=hLb, A=A)


def solve(NC):
    f = lambda p: mismatch(p, NC)[0]
    pb = brentq(f, 5.2, 6.0, xtol=1e-12)
    return mismatch(pb, NC)[1]


def main():
    out = {}
    for label, NC in [('NC100_check', 100.0), ('SM_AAA', 25.515)]:
        r = solve(NC)
        r['NC'] = NC
        r['delta'] = float(np.sqrt(2) / (3 * r['hLb'] ** 3 / HMP ** 2))
        out[label] = r
        print(label, {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    (OUT / 'C62_results.json').write_text(json.dumps({k: {kk: float(vv) for kk, vv in v.items()} for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    main()
