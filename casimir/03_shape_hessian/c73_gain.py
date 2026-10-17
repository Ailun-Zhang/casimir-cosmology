# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_gain.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Zero-incident-momentum gain G(kappa) = sqrt2 cosh(pi sqrt(4 kappa - 1)/4) (closed form, Gamma, 2F1, ODE) and the guaranteed gain >= 4.500465 (sum kappa = 8/(1-xi), some kappa_A >= 8/(5(1-xi))). Paper: PRD Sec. VII C (Eq. eq:s-gain; 4.500465), App. app:shapelaplacian.

C73 (ii): zero-incident-momentum gain of beta'' + 3H beta' - kappa h^2 sech^2(2ht) beta = 0 on
L = L_b sqrt(cosh 2ht) (H = h tanh 2ht), and the r-dependent lower bound of claim (3).

Exact result (derived in D55): with tau = 2ht, z = (1+tanh tau)/2 the equation is hypergeometric,
  z(1-z) y'' + (1/4 - z/2) y' - (kappa/4) y = 0,  a+b = -1/2, ab = kappa/4, c = 1/4,
and the solution regular at z=0 (beta -> 1, beta' -> 0 in the far past) has
  gain = 2F1(a,b;1/4;1) = Gamma(1/4)Gamma(3/4)/(Gamma(1/4-a)Gamma(1/4-b))
       = sqrt2 cosh[(pi/4) sqrt(4 kappa - 1)]   (kappa > 1/4),  sqrt2 cos[(pi/4) sqrt(1-4kappa)] (kappa < 1/4).
Checks: (1) mpmath hyp2f1 at z=1, (2) direct ODE integration (scipy DOP853) in cosmic time.
Output: results/.../C73/C73_gain.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from c73_common import RES

mp.mp.dps = 30


def gain_closed(kappa):
    k = mp.mpf(kappa)
    if k > mp.mpf(1) / 4:
        return mp.sqrt(2) * mp.cosh(mp.pi / 4 * mp.sqrt(4 * k - 1))
    return mp.sqrt(2) * mp.cos(mp.pi / 4 * mp.sqrt(1 - 4 * k))


def gain_hyp(kappa):
    k = mp.mpf(kappa)
    disc = mp.sqrt(mp.mpc(mp.mpf(1) / 4 - k))
    a = -mp.mpf(1) / 4 + disc / 2
    b = -mp.mpf(1) / 4 - disc / 2
    via_gamma = mp.gamma(mp.mpf(1) / 4) * mp.gamma(mp.mpf(3) / 4) / (mp.gamma(mp.mpf(1) / 4 - a) * mp.gamma(mp.mpf(1) / 4 - b))
    via_2f1 = mp.hyp2f1(a, b, mp.mpf(1) / 4, 1)
    # also check the ODE in z at an interior point by finite differences of 2F1
    z0 = mp.mpf("0.37")
    y = lambda z: mp.hyp2f1(a, b, mp.mpf(1) / 4, z)
    resid = z0 * (1 - z0) * mp.diff(y, z0, 2) + (mp.mpf(1) / 4 - z0 / 2) * mp.diff(y, z0) - k / 4 * y(z0)
    return mp.re(via_gamma), mp.re(via_2f1), abs(resid)


def gain_ode(kappa, t0=20.0):
    h = 1.0

    def rhs(t, y):
        H = h * np.tanh(2 * h * t)
        return [y[1], -3 * H * y[1] + kappa * h ** 2 / np.cosh(2 * h * t) ** 2 * y[0]]

    # max_step prevents the adaptive stepper from striding over the turning region from the flat far past
    sol = solve_ivp(rhs, (-t0, t0), [1.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-16, max_step=0.01)
    return sol.y[0, -1], sol.y[1, -1]


def main():
    out = {"description": __doc__, "rows": []}
    kE, kT2 = mp.mpf("13.5286"), mp.mpf("-6.3524")
    out["paper_kappas"] = {"kappa_E": float(kE), "kappa_T2": float(kT2),
                           "2 kappa_E + 3 kappa_T2": float(2 * kE + 3 * kT2)}
    for label, kap in (("kappa_E", kE), ("kappa_T2", kT2), ("kappa=1/4+0", mp.mpf("0.25000001")), ("kappa=0", mp.mpf(0))):
        gc = gain_closed(kap)
        gg, g2, res = gain_hyp(kap)
        go, dgo = gain_ode(float(kap))
        out["rows"].append({"case": label, "kappa": float(kap), "gain_closed": mp.nstr(gc, 15),
                            "gain_gamma": mp.nstr(gg, 15), "gain_2F1(z=1)": mp.nstr(g2, 15),
                            "ODE_residual_2F1": mp.nstr(res, 3), "gain_ODE": go, "beta'_end": dgo})
    bounds = []
    for r in ("0", "0.1", "0.25", "0.5"):
        rr = mp.mpf(r)
        kbar = 8 / (5 * (1 - rr))
        gc = gain_closed(kbar)
        gg, g2, res = gain_hyp(kbar)
        go, _ = gain_ode(float(kbar))
        bounds.append({"r": float(rr), "sum_kappa = 8/(1-r)": mp.nstr(8 / (1 - rr), 12),
                       "kappa_mean = 8/(5(1-r))": mp.nstr(kbar, 12),
                       "G_lower_bound closed": mp.nstr(gc, 12), "G via Gamma": mp.nstr(gg, 12),
                       "G via 2F1": mp.nstr(g2, 12), "G via ODE": go,
                       "SM aspect gain G(kappa_E/(1-r)) (illustration)": mp.nstr(gain_closed(kE / (1 - rr)), 8)})
    out["lower_bounds"] = bounds
    # monotonicity of G(kappa) on a grid (kappa > -2)
    ks = np.linspace(-2, 40, 4201)
    gs = np.array([float(gain_closed(k)) for k in ks])
    out["G_monotone_increasing_on_[-2,40]"] = bool(np.all(np.diff(gs) > 0))
    out["identities"] = {"Gamma(1/4)Gamma(3/4) - sqrt2 pi": mp.nstr(mp.gamma(0.25) * mp.gamma(0.75) - mp.sqrt(2) * mp.pi, 3),
                         "Gamma(1/2+iy)Gamma(1/2-iy) - pi/cosh(pi y), y=0.9": mp.nstr(abs(mp.gamma(0.5 + 0.9j) * mp.gamma(0.5 - 0.9j) - mp.pi / mp.cosh(mp.pi * 0.9)), 3)}
    (RES / "C73_gain.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["rows"], indent=0, default=float))
    print(json.dumps(bounds, indent=0, default=float))
    print(out["paper_kappas"], out["G_monotone_increasing_on_[-2,40]"], out["identities"])


if __name__ == "__main__":
    main()
