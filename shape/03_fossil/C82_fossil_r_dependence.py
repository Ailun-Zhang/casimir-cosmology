# PROVENANCE: 04_calculations/code/20261011_claude_followup/C82_fossil_r_dependence.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Aspect fossil vs zero-mode fraction r = xi (sigma_E 0.0589 free ... 0.1041 at 0.2999), excited zero-mode level, inflaton fluctuation, median ln(Lmax/Lmin)/sigma_E = 1.588, typical side difference (18% at xi = 0.30, 10% free). Paper: PRD Sec. VIII D (tab:s-fossil, eq:s-scaling).

C82: dependence of the aspect fossil on the zero-mode fraction and related R10 review checks.

Leading background L = L_b sqrt(cosh 2ht); transfer from the turning point as in C68/C77 (M_from); the natural-state
variance is delta |T12| (x + 1/x)/2 with T12 = 2 m11 m12, x = s_nat/s_*, s_nat = (3/(2 sqrt2)) delta/sqrt(kappa),
s_* = |m12/m11| delta; delta = delta0 (1 - r)^(-3/4) at fixed h, kappa_E = kappa_E^free/(1 - r).

Checks:
1. sigma_E(r) for r = 0, 0.258, 0.28, 0.2826 (LO + first order, C81), 0.2999, 0.308, 0.32;
2. one excited zero-mode level (lowest SU(2) and SU(3) gaps, (6.0145 - 4.116720) g2^{2/3} and
   (14.800 - 12.5868) g3^{2/3} from van Baal 1988 Table 2 and Pavel 2021 Table 2 as recorded in L25_followup.md)
   raises the effective r by gap/C_free;
3. the inflaton fluctuation (+1/2 to N_C, periodic scalar: lambda_E += (991.56/30)/2, Z_eff += Z_2(1)/2) with the zero-mode
   energy Q held fixed;
4. median of ln(Lmax/Lmin) for an isotropic Gaussian in the E doublet (plus angle modes of width sigma_T2): coefficient
   c = median/sigma_E by Monte Carlo, and the "typical side difference" exp(c sigma_E) - 1 on the plateau history
   (plateau/leading width ratio 0.0587/0.058891 from Table I and C72).

    python -B C82_fossil_r_dependence.py
Output: results/20261011_claude_followup/C82/C82_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("fu/C82")
C32 = 3 / 2 ** 1.5
KE, KT = 13.528581, -6.352387
DELTA0 = 1.5897991e-4
C_FREE, NC, GAMMA, Z2CUBE = 42.7400221506, 25.515, 1.6750738, 16.5323159598
LAMBDA_E_AAA, LAMBDA_E_PPP = 1902.24, -991.56
PLATEAU_RATIO = 0.0587 / 0.058891
G2, G3 = 0.526764, 0.539605


def M_from(kappa, t0=0.0, t1=12.0):
    def f(t, y):
        l3 = math.cosh(2 * t) ** 1.5
        s2 = 1.0 / math.cosh(2 * t) ** 2
        return [C32 * y[1] / l3, (kappa / C32) * l3 * s2 * y[0], C32 * y[3] / l3, (kappa / C32) * l3 * s2 * y[2]]
    y = solve_ivp(f, (t0, t1), [1, 0, 0, 1], method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
    return y[0], y[2]                     # m11, m12


def sigma(kappa, delta):
    m11, m12 = M_from(kappa)
    s_star = abs(m12 / m11) * delta
    s_nat = 3 / (2 * math.sqrt(2)) * delta / math.sqrt(abs(kappa))
    x = s_nat / s_star
    return math.sqrt(abs(2 * m11 * m12) * delta * (x + 1 / x) / 2)


def sigma_r(r):
    return sigma(KE / (1 - r), DELTA0 * (1 - r) ** (-0.75))


def median_coeff(n=2_000_000, seed=7):
    rng = np.random.default_rng(seed)
    S2, S6 = math.sqrt(2), math.sqrt(6)
    X = np.array([np.diag([1, -1, 0]) / S2, np.diag([1, 1, -2]) / S6])
    b = rng.standard_normal((n, 2))
    # diagonal G: ln sides = b_d X_d + b_d2 X_d2 (angle modes negligible at sigma_T2/sigma_E = 0.06: checked below)
    ln = b[:, :1] * np.diag(X[0])[None, :] + b[:, 1:2] * np.diag(X[1])[None, :]
    return float(np.median(ln.max(1) - ln.min(1)))


def median_with_angles(sig_e, sig_t, n=200_000, seed=11):
    rng = np.random.default_rng(seed)
    S2, S6 = math.sqrt(2), math.sqrt(6)
    B = [np.diag([1, -1, 0]) / S2, np.diag([1, 1, -2]) / S6]
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        E = np.zeros((3, 3)); E[i, j] = E[j, i] = 1 / S2; B.append(E)
    B = np.array(B)
    z = rng.standard_normal((n, 5)) * np.array([sig_e] * 2 + [sig_t] * 3)
    out = np.empty(n)
    for k in range(n):
        G = expm(2 * np.tensordot(z[k], B, axes=1))
        s = np.sqrt(np.diag(G))
        out[k] = math.log(s.max() / s.min())
    return float(np.median(out))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    s0 = sigma_r(0.0)
    s30 = sigma_r(0.299897)
    res["sigma_E_leading"] = {str(r): sigma_r(r) for r in (0.0, 0.258, 0.28, 0.2826, 0.299897, 0.308, 0.32)}
    res["ratio_to_free"] = {k: v / s0 for k, v in res["sigma_E_leading"].items()}
    res["dlnsigma_dr_at_0.30"] = (math.log(sigma_r(0.309897)) - math.log(sigma_r(0.289897))) / 0.02
    # 2. one excited zero-mode level
    gaps = {"SU2_lowest": (6.0145 - 4.116720) * G2 ** (2 / 3), "SU3_lowest": (14.800 - 12.5868) * G3 ** (2 / 3)}
    res["zero_mode_excitation"] = {}
    for k, g in gaps.items():
        dr = g / C_FREE
        res["zero_mode_excitation"][k] = dict(gap=g, frac_of_C=dr, sigma_ratio=sigma_r(0.299897 + dr) / s30)
    # 3. inflaton fluctuation with Q fixed
    Q = 0.299897 * C_FREE
    zq = math.pi ** 2 * Q
    lamE_inf = LAMBDA_E_PPP / (-30.0) * 0.5          # per unit periodic weight, times 1/2
    for label, q in (("free", 0.0), ("zero_mode", zq)):
        ze0, ze1 = NC * Z2CUBE - q, (NC + 0.5) * Z2CUBE - q
        k0, k1 = 3 * LAMBDA_E_AAA / ze0, 3 * (LAMBDA_E_AAA + lamE_inf) / ze1
        nce0, nce1 = ze0 / Z2CUBE, ze1 / Z2CUBE
        d0, d1 = DELTA0 * (NC / nce0) ** 0.75, DELTA0 * (NC / nce1) ** 0.75
        res.setdefault("inflaton", {})[label] = dict(kappa=[k0, k1], NCeff=[nce0, nce1],
                                                     sigma_ratio=sigma(k1, d1) / sigma(k0, d0))
    # 4. median coefficient and typical side difference
    c = median_coeff()
    res["median_coeff"] = c
    res["median_with_angle_modes"] = {"zero_mode": median_with_angles(s30, 0.0064896) / s30,
                                      "free": median_with_angles(s0, 0.0066157) / s0}
    res["typical_side_difference_plateau"] = {
        str(r): math.exp(c * res["sigma_E_leading"][str(r)] * PLATEAU_RATIO) - 1
        for r in (0.0, 0.258, 0.28, 0.2826, 0.299897, 0.308, 0.32)}
    (OUT / "C82_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
