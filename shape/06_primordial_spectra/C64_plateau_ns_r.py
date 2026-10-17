# PROVENANCE: 04_calculations/code/20261009_claude_R5/C64_plateau_ns_r.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Scalar and tensor spectra of the plateau after the turn (D51, D30, D35): n_s = 0.96534 and r = 0.00346 for instant reheating (N_* = 55.72), n_s = 0.96174, r = 0.00420 for the weak visible decay (N_* = 50.33); initial-state tests. Paper: PRD Sec. IX (s09_spectra.tex), Fig. (n_s, r vs N_*).

C64: exact scalar and tensor spectra of the plateau V = 3 M_P^2 h^2 (1 - exp(-sqrt(2/3) phi))^2 as a function of
N_* (e-folds from pivot exit to the end of inflation), independent of C37 (D30).

Background on the slow-roll attractor (the turning point and Casimir energy have diluted by e^{-4 N_exit} < 1e-21
long before the pivot exits, D51). Modes: Mukhanov-Sasaki v'' + (k^2 - z''/z) v = 0 with z = a phidot/H,
tensors with z = a; Bunch-Davies state set 6 e-folds before exit (kinetic adiabatic order 2; changing to
7 e-folds tests the initial state), frozen 12 e-folds after exit. Tilt from ln k +- 0.01 (central difference).
Integration in e-fold time N with u = v sqrt(2k): standard real-time form.
Checks against D30: N_* = 55.72234537 -> n_s = 0.96534071, r = 0.00346242.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C64_plateau_ns_r')
OUT.mkdir(parents=True, exist_ok=True)
S23 = np.sqrt(2 / 3)


def V(p): return 3 * (1 - np.exp(-S23 * p)) ** 2
def Vp(p): e = np.exp(-S23 * p); return 6 * S23 * e * (1 - e)
def Vpp(p): e = np.exp(-S23 * p); return 6 * (2 / 3) * e * (2 * e - 1)


# background in e-fold time N (M_P = h = 1): phi_N = y, y_N = -(3 - eps)(y + Vp/V*... ) standard:
# with eps = y^2/2:  y' = -(3 - eps) * (y + Vp/(V) * (3 - eps)/ (3 - eps))  -> use exact form below
def bg_rhs(N, s):
    p, y = s                         # y = dphi/dN
    eps = 0.5 * y * y
    H2 = V(p) / (3 - eps)
    return [y, -(3 - eps) * y - Vp(p) / H2]


def background(phi0=7.5):
    y0 = -Vp(phi0) / V(phi0)        # slow-roll attractor start
    def end(N, s): return 0.5 * s[1] ** 2 - 1.0
    end.terminal = True; end.direction = 1
    sol = solve_ivp(bg_rhs, (0, 400), [phi0, y0], method='DOP853', rtol=1e-12, atol=1e-14,
                    events=[end], dense_output=True)
    return sol, sol.t_events[0][0]


SOL, NEND = background()


def state(N):
    p, y = SOL.sol(N)
    eps = 0.5 * y * y
    H = np.sqrt(V(p) / (3 - eps))
    return p, y, eps, H


PREP = {'efolds': 8.0, 'order': 2}


def mode_power(Nx_target, dlnk, tensor):
    """Power of mode k = e^{dlnk} * a(N_x) H(N_x), with N_x = NEND - Nx_target; returns P (dimensionless)."""
    Nx = NEND - Nx_target
    pX, yX, eX, HX = state(Nx)
    k = HX * np.exp(dlnk)                 # a normalized to 1 at N = Nx (spectra are independent of this choice)
    Ni, Nf = Nx - PREP['efolds'], Nx + 12.0
    NREF = Nx

    # variables: v (complex) as function of N; v_NN + (1 - eps) v_N + (k^2/(a^2H^2) - z''/(z a^2 H^2)) v = 0
    def zpp_over_z(N):
        p, y, eps, H = state(N)
        if tensor:
            return (2 - eps)                 # a''/a / (a H)^2 = 2 - eps
        # z''/z/(aH)^2 = 2 - eps + (3/2)... exact: 2 - eps + 2 eta-like terms; use exact expression
        # z = a y (y = dphi/dN); z''/z/(aH)^2 = 2 - eps + (y_NN/y)(1) + ... computed numerically below
        return None

    def rhs(N, s):
        p, y, eps, H = state(N)
        a = np.exp(N - NREF)
        w2 = (k / (a * H)) ** 2
        if tensor:
            m2 = 2 - eps
        else:
            # z = a y: z_N/z = 1 + y_N/y ; z''/z (conformal) /(aH)^2 = (zNN + (1-eps) zN)/z with z=a y
            yN = -(3 - eps) * y - Vp(p) / (V(p) / (3 - eps))
            epsN = y * yN
            # y_NN from differentiating the background equation
            H2 = V(p) / (3 - eps)
            dH2 = (Vp(p) * y) / (3 - eps) + V(p) * epsN / (3 - eps) ** 2
            yNN = epsN * y - (3 - eps) * yN - (Vpp(p) * y) / H2 + Vp(p) * dH2 / H2 ** 2
            zN_z = 1 + yN / y
            zNN_z = 1 + 2 * yN / y + yNN / y
            m2 = zNN_z + (1 - eps) * zN_z
        vr, vi, ur, ui = s
        return [ur, ui, -(1 - eps) * ur - (w2 - m2) * vr, -(1 - eps) * ui - (w2 - m2) * vi]

    # Bunch-Davies-like initial data at Ni (adiabatic, deep inside the Hubble radius)
    def m2_of(N):
        sN = rhs(N, [1.0, 0.0, 0.0, 0.0])        # with v=1, v_N=0: v_NN = -(w2 - m2)
        p, y, eps, H = state(N)
        a = np.exp(N - NREF)
        w2 = (k / (a * H)) ** 2
        return w2 + sN[2]                         # = m2

    def Om2(N):                                   # conformal-time frequency^2: k^2 - (aH)^2 m2
        p, y, eps, H = state(N)
        a = np.exp(N - NREF)
        return k * k - (a * H) ** 2 * m2_of(N)

    p, y, eps, H = state(Ni)
    a = np.exp(Ni - NREF)
    if PREP['order'] == 0:
        v0 = 1.0 / np.sqrt(2 * k)
        vN0 = -1j * (k / (a * H)) * v0
    else:
        O2 = Om2(Ni)
        dN = 1e-4
        dO2_dN = (Om2(Ni + dN) - Om2(Ni - dN)) / (2 * dN)
        Om = np.sqrt(O2)
        dOm_deta = (a * H) * dO2_dN / (2 * Om)    # d/deta = aH d/dN
        v0 = 1.0 / np.sqrt(2 * Om)
        vN0 = (-1j * Om - dOm_deta / (2 * Om)) * v0 / (a * H)
    s = solve_ivp(rhs, (Ni, Nf), [v0, 0.0, vN0.real, vN0.imag], method='DOP853', rtol=1e-10, atol=1e-14)
    vr, vi = s.y[0, -1], s.y[1, -1]
    p, y, eps, H = state(Nf)
    a = np.exp(Nf - NREF)
    if tensor:
        z = a / 2.0                     # h = 2 v/(a M_P) per polarization (M_P=1): P_t = 2 * k^3/(2 pi^2) |2v/a|^2
        Pk = 2 * k ** 3 / (2 * np.pi ** 2) * abs(vr + 1j * vi) ** 2 / z ** 2
    else:
        z = a * abs(y)                  # R = v/z
        Pk = k ** 3 / (2 * np.pi ** 2) * abs(vr + 1j * vi) ** 2 / z ** 2
    return Pk, k


def ns_r(Nstar, step=0.01):
    Pp, _ = mode_power(Nstar, +step, False)
    Pm, _ = mode_power(Nstar, -step, False)
    P0, _ = mode_power(Nstar, 0.0, False)
    T0, _ = mode_power(Nstar, 0.0, True)
    ns = 1 + (np.log(Pp) - np.log(Pm)) / (2 * step)
    return ns, T0 / P0, P0


def main():
    out = {'N_end_from_phi0_7.5': NEND, 'rows': [], 'initial_state_tests': []}
    for ef, order in [(6.0, 0), (6.0, 2), (8.0, 2), (10.0, 2)]:
        PREP['efolds'], PREP['order'] = ef, order
        ns, r, P0 = ns_r(55.72234537)
        out['initial_state_tests'].append({'prep_efolds': ef, 'adiabatic_order': order, 'ns': ns, 'r': r})
        print(f"test prep {ef} order {order}: ns={ns:.8f} r={r:.8f}")
    PREP['efolds'], PREP['order'] = 8.0, 2
    for Ns in [40, 45, 50, 50.32813124, 52, 54, 55.72234537, 57, 60, 63, 66, 70]:
        ns, r, P0 = ns_r(Ns)
        out['rows'].append({'Nstar': Ns, 'ns': ns, 'r': r})
        print(f"N*={Ns:9.4f}  ns={ns:.7f}  r={r:.7f}")
    (OUT / 'C64_results.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

