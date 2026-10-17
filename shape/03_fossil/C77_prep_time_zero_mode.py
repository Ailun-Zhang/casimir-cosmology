# PROVENANCE: 04_calculations/code/20261011_claude_followup/C77_prep_time_zero_mode.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Preparation-time sensitivity and mean-offset tolerance of the SM aspect fossil with the zero-mode energy (D58 Sec. 3). Paper: PRD Sec. VIII D (error budget).

C77: preparation-time sensitivity and mean-offset tolerance of the SM aspect fossil with the zero-mode energy.

Same construction as C68 part (b) (leading background L = L_b sqrt(cosh 2ht); uncorrelated Gaussian prepared at
h t_p <= 0 with the local natural width; outgoing variance in units of the floor delta |T12|), re-implemented here
without importing C68 (whose module-level code rewrites its own results file). Stiffness kappa_E -> kappa_E/(1-r).

    python C77_prep_time_zero_mode.py
Output: results/20261011_claude_followup/C77/C77_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("fu/C77")
C32 = 3 / 2 ** 1.5
R_ZM = 0.299897
KAPPA_E = 13.528581


def M_from(kappa, t0, t1=12.0):
    """Transfer of (beta, Pi) from h t0 to h t1 on the leading background (C68 shape_rhs)."""
    def f(t, y):
        l3 = np.cosh(2 * t) ** 1.5
        s2 = 1.0 / np.cosh(2 * t) ** 2
        return [C32 * y[1] / l3, (kappa / C32) * l3 * s2 * y[0], C32 * y[3] / l3, (kappa / C32) * l3 * s2 * y[2]]
    s = solve_ivp(f, (t0, t1), [1, 0, 0, 1], method="DOP853", rtol=1e-12, atol=1e-14)
    y = s.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def row(kappa, tp):
    M = M_from(kappa, tp)
    floor = abs(2 * M_from(kappa, 0.0)[0, 0] * M_from(kappa, 0.0)[0, 1])
    l3 = np.cosh(2 * tp) ** 1.5
    s_nat = 3 / (2 * np.sqrt(2) * np.sqrt(kappa)) / l3 * np.cosh(2 * tp)       # in units of delta
    var = M[0, 0] ** 2 * s_nat + M[0, 1] ** 2 / s_nat                             # in units of delta
    return {"ht_prep": tp, "s_nat_over_delta": float(s_nat), "var_over_floor_natural": float(var / floor),
            "rms_ratio_natural": float(np.sqrt(var / floor))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"r": R_ZM}
    for label, kap in (("free", KAPPA_E), ("zero_mode", KAPPA_E / (1 - R_ZM))):
        M0 = M_from(kap, 0.0)
        m11, m12 = float(M0[0, 0]), float(M0[0, 1])
        s_star = abs(m12 / m11)
        s_nat = 3 / (2 * np.sqrt(2) * np.sqrt(kap))
        # mean-offset tolerance: offset at t=0 whose deterministic image m11*offset equals the quantum rms
        # sqrt(delta |T12| (x + 1/x)/2); quoted in units of the turning-point quantum width sqrt(s_nat delta)
        x = s_nat / s_star
        rms_over_sqrt_delta = np.sqrt(abs(2 * m11 * m12) * (x + 1 / x) / 2)
        offset_over_sqrt_delta = rms_over_sqrt_delta / m11
        res[label] = {"kappa_E": kap, "m11": m11, "m12": m12, "x_nat": float(x),
                      "variance_excess_over_floor": float((x + 1 / x) / 2 - 1),
                      "offset_tolerance_in_quantum_widths": float(offset_over_sqrt_delta / np.sqrt(s_nat)),
                      "rows": [row(kap, tp) for tp in (0.0, -0.1, -0.25, -0.5)]}
    (OUT / "C77_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
