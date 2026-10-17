# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_tier2/C160_shear_wdw.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Shear criterion with a quantum volume (D131 Sec. 3): reflection across rho = 3^-3/2 smoothed by Delta rho = 7.8e-5 (relative 4e-4) at X = 2282.49. Paper: PRD App. app:theorem / Sec. VIII (NL-15, NL-17).

C160 part 3 (shape-tier2, D131 Sec. 3): the shear turning criterion s0 <= 12 (zeta/3)^{3/2} (ratio < 3^{-3/2} = 0.19245)
with a quantum volume.

Minisuperspace (C60 units, t in 1/H_eff, L = L_b e^x): S_vol = -3X int e^{3x} x'^2, X = M_P^2 L_b^3 H_eff, shear energy
(X/2) e^{-3x} s0 with s0 = tr((G^-1 M)^2) = sum P^2 / X^2 (P canonical shape momentum, hbar = 1; conserved for free shape
motion through the narrow turning layer).  Hamiltonian constraint -> WDW at fixed shape momentum:
    psi''(x) + K^2(x) psi = 0,   K^2 = 12 X^2 [3 e^{6x} - 3 zeta e^{2x} + s0/2]
(operator-ordering terms are O(1) against O(X^2) and are dropped).  Contracting wave e^{-iS} comes in from large x; the
reflected e^{+iS} is the bounce, the transmitted e^{-iS} at x -> -inf is the shear-dominated crunch.  Classically
R = 1 for s0 < s_c, 0 for s0 > s_c.  Parabolic-cylinder (uniform) prediction near s_c:
    T_crunch = 1/(1 + exp(pi A/sqrt(B))),  A = 6X^2 (s_c - s0),  B = 6 X^2 f''(x*),  f'' = 24 zeta (zeta/3)^{1/2}.
    python -B C160_shear_wdw.py  -> results/.../run01/shear/C160_shear_wdw.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_tier2/run01/shear")
X_PHYS = 2282.49


def K2(x, X, zeta, s0):
    return 12 * X * X * (3 * np.exp(6 * x) - 3 * zeta * np.exp(2 * x) + s0 / 2)


def dK2(x, X, zeta):
    return 12 * X * X * (18 * np.exp(6 * x) - 6 * zeta * np.exp(2 * x))


def scatter(X, zeta, s0, dl=0.8, dr=0.6, rtol=1e-11):
    xs = 0.25 * np.log(zeta / 3)
    xl, xr = xs - dl, xs + dr
    kl = np.sqrt(K2(xl, X, zeta, s0))
    kpl = dK2(xl, X, zeta) / (2 * kl)
    # pure transmitted wave e^{-iS} at xl (flux -1): psi = k^{-1/2}, psi' = (-ik - k'/(2k)) psi
    psi = 1 / np.sqrt(kl)
    dpsi = (-1j * kl - kpl / (2 * kl)) * psi

    def rhs(x, y):
        k2 = K2(x, X, zeta, s0)
        return [y[1], -k2 * y[0], y[3], -k2 * y[2]]
    sol = solve_ivp(rhs, (xl, xr), [psi.real, dpsi.real, psi.imag, dpsi.imag], method="DOP853", rtol=rtol,
                    atol=1e-14 / np.sqrt(kl))
    p = sol.y[0, -1] + 1j * sol.y[2, -1]
    dp = sol.y[1, -1] + 1j * sol.y[3, -1]
    kr = np.sqrt(K2(xr, X, zeta, s0))
    kpr = dK2(xr, X, zeta) / (2 * kr)
    up, um = 1 / np.sqrt(kr), 1 / np.sqrt(kr)
    dup = (1j * kr - kpr / (2 * kr)) * up
    dum = (-1j * kr - kpr / (2 * kr)) * um
    Mx = np.array([[up, um], [dup, dum]])
    a, b = np.linalg.solve(Mx, np.array([p, dp]))
    flux = float(np.imag(np.conj(p) * dp))
    return {"R": float(abs(a / b) ** 2), "T": float(1 / abs(b) ** 2), "flux": flux, "unitarity": float(abs(a) ** 2 - abs(b) ** 2 + 1),
            "nfev": int(sol.nfev)}


def uniform(X, zeta, s0):
    sc = 12 * (zeta / 3) ** 1.5
    fpp = 24 * zeta * np.sqrt(zeta / 3)
    A = 6 * X * X * (sc - s0)
    B = 6 * X * X * fpp
    return 1.0 / (1.0 + np.exp(np.pi * A / np.sqrt(B))), float(np.pi * 6 * X * X / np.sqrt(B))


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"rows": [], "fits": []}
    for zeta in (1.0, 1.2):
        sc = 12 * (zeta / 3) ** 1.5
        for X in (30.0, 100.0, 300.0, 1000.0, X_PHYS):
            _, slope = uniform(X, zeta, sc)          # exponent per unit (s0 - s_c)
            ys = np.linspace(-6, 6, 13)
            Ts, Tu = [], []
            for y in ys:
                s0 = sc + y / slope
                r = scatter(X, zeta, s0)
                tu, _ = uniform(X, zeta, s0)
                Ts.append(r["T"])
                Tu.append(tu)
                res["rows"].append({"zeta": zeta, "X": X, "y": float(y), "s0": float(s0), "ratio_s0_over_12": float(s0 / 12),
                                    "T_num": r["T"], "R_num": r["R"], "T_uniform": float(tu), "unitarity": r["unitarity"],
                                    "nfev": r["nfev"]})
            Ts = np.array(Ts)
            # logit fit T = 1/(1+exp(-(y - y0)/w)): y0 = shift of the 50% point, w = width (uniform: y0 = 0, w = 1)
            m = (Ts > 1e-6) & (Ts < 1 - 1e-6)
            lg = np.log(Ts[m] / (1 - Ts[m]))
            w_inv, c0 = np.polyfit(ys[m], lg, 1)
            y0 = -c0 / w_inv
            fit = {"zeta": zeta, "X": X, "s_c": sc, "ratio_c": sc / 12, "slope_per_s0": slope,
                   "y0_shift": float(y0), "width_w": float(1 / w_inv),
                   "s0_50pct": float(sc + y0 / slope), "ratio_50pct": float((sc + y0 / slope) / 12),
                   "ratio_10_90_width": float(2 * np.log(9) / (w_inv * slope) / 12),
                   "rel_shift_of_ratio_c": float(y0 / slope / sc), "max_abs_T_minus_uniform": float(np.max(np.abs(Ts - np.array(Tu)))),
                   "deep_check": {}}
            for yy in (-30.0, 30.0):                 # far from threshold: classical limit recovered?
                rr = scatter(X, zeta, sc + yy / slope)
                fit["deep_check"]["y=%g" % yy] = {"T_num": rr["T"], "T_uniform": float(uniform(X, zeta, sc + yy / slope)[0])}
            res["fits"].append(fit)
            print(json.dumps({k: fit[k] for k in ("zeta", "X", "ratio_c", "y0_shift", "width_w", "ratio_10_90_width",
                                                   "rel_shift_of_ratio_c", "max_abs_T_minus_uniform")}), flush=True)
    # domain convergence at the physical X
    sc = 12 / 3 ** 1.5
    _, slope = uniform(X_PHYS, 1.0, sc)
    res["domain_check"] = [dict(dl=dl, dr=dr, **scatter(X_PHYS, 1.0, sc + 1.0 / slope, dl=dl, dr=dr)) for dl, dr in
                           ((0.8, 0.6), (1.2, 0.9), (0.5, 0.4))]
    res["physical"] = {"X": X_PHYS, "ratio_c": sc / 12, "ratio_10_90_width": 2 * np.log(9) / slope / 12,
                       "relative_width": 2 * np.log(9) / slope / sc,
                       "airy_dlnL": float((12 * X_PHYS ** 2 * 12) ** (-1 / 3)),
                       "note": "Airy length of the volume turning layer at zeta=1, s0=0: (12 X^2 |dK^2/dx|/(12X^2))^{-1/3}"}
    res["meta"] = {"runtime_s": time.perf_counter() - t0, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "C160_shear_wdw.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["physical"]), json.dumps(res["domain_check"]), flush=True)


if __name__ == "__main__":
    main()
