# PROVENANCE: 04_calculations/code/20261013_claude_v12/state_selection/C137c_uncorrelated_optimum.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] G(|t_x|) = min over uncorrelated Gaussians at t_x of the outgoing aspect variance / floor: 2.395, 8.263, 45.10, 313.8, 961.7 at |t_x| = 0.1, 0.25, 0.5, 1, 2 (kappa = 19.32); adiabatic/floor = 1.000485 at t_x = 0. Paper: PRD Eq. eq:s-Gtx, App. app:theorem (tab:s-G).

C137c (Claude v12, state_selection): minimum outgoing aspect variance over UNCORRELATED Gaussians
specified at an earlier time t_x (any width, not the adiabatic one) on v11's background.
min_s [M11^2 s + M12^2/(4 s)] = |M11 M12| for the transfer M(t_x -> +6).  Compared with v11's floor
(t_x = 0, time-symmetric optimum) and with the adiabatic ('natural at t_x') choice.
Also: the width that achieves the optimum, in units of the adiabatic width at t_x (shows that the
optimum is NOT set by local adiabaticity once t_x lies in the frozen era).
"""
import json, hashlib, os
import numpy as np
from scipy.integrate import solve_ivp

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shape_paths as WP  # noqa: E402
OUT = str(WP.wdir("r1/state_selection/run03"))
                   # (package: the work-tree path above replaces the original results path)


def M_of(kappa, t0, t1):
    m = lambda t: np.cosh(2 * t) ** 1.5
    def rhs(t, y):
        H = np.tanh(2 * t)
        f = kappa / np.cosh(2 * t) ** 2
        return [y[1], -3 * H * y[1] + f * y[0], y[3], -3 * H * y[3] + f * y[2]]
    s = solve_ivp(rhs, (t0, t1), [1.0, 0.0, 0.0, 1.0 / m(t0)], rtol=1e-12, atol=1e-16, method="DOP853")
    b1, bd1, b2, bd2 = s.y[:, -1]
    return np.array([[b1, b2], [m(t1) * bd1, m(t1) * bd2]])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(__file__, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    res = dict(code=os.path.basename(__file__), code_sha256=sha, rows=[])
    for kappa in (19.32, -9.07):
        M0 = M_of(kappa, 0.0, 6.0)
        floor = abs(M0[0, 0] * M0[0, 1])
        for tx in (0.0, -0.1, -0.25, -0.5, -1.0, -2.0, -4.0, -8.0):
            M = M_of(kappa, tx, 6.0) if tx < 0 else M0
            opt = abs(M[0, 0] * M[0, 1])
            s_opt = abs(M[0, 1] / M[0, 0]) / 2.0
            mx = np.cosh(2 * tx) ** 1.5
            wloc = np.sqrt(abs(kappa)) / np.cosh(2 * tx)          # local Casimir frequency
            wad = max(wloc, abs(np.tanh(2 * tx)))                   # adiabatic scale max(|omega|, |H|)
            s_ad = 1.0 / (2 * mx * wad)
            var_ad = M[0, 0] ** 2 * s_ad + M[0, 1] ** 2 / (4 * s_ad)
            res["rows"].append(dict(kappa=kappa, t_x=tx, opt_over_floor=opt / floor,
                                    sqrt_opt_over_floor=np.sqrt(opt / floor),
                                    s_opt_over_s_adiabatic=s_opt / s_ad,
                                    adiabatic_over_floor=var_ad / floor))
            print(f"kappa={kappa:6.2f} t_x={tx:5.2f}  opt/floor={opt/floor:.4f}  "
                  f"s_opt/s_ad={s_opt/s_ad:.3e}  adiabatic/floor={var_ad/floor:.3e}")
    with open(os.path.join(OUT, "C137c_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
