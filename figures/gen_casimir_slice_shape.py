# PROVENANCE: functions gamma_m12, Z2 and Zeff copied unchanged from
# 04_calculations/code/20261009_claude_R5/C57_spin_structure_Casimir.py (2026-10-15, fig-shape); the slice loop is
# the one of 06_paper_draft/figures/v11/fig2_casimir_shape.py panel (b).
"""Generate data/shape/casimir_slice_AAA.json: fractional change of the Standard-Model Casimir energy density with all
fermions antiperiodic (AAA), Delta rho_C/|rho_C|, on the shape slice G = exp(2 b_d X_d + 2 b_a X_12) through the cube
(free massless fields at trivial holonomy; 15 bosonic and 45 fermionic Weyl units, PRD Sec. IV).

Image form: E_C = -(1/(pi^2 L)) sum_{n != 0} w_n (n^T G n)^-2 with twisted Epstein sums Z_2(G, theta) evaluated by an
Ewald split (s = 2); rho_C = -Z_eff/(pi^2 L^4), so Delta rho_C/|rho_C| = -(Z_eff(G) - Z_eff(1))/Z_eff(1).
Run time about 20 s. Used by fig_prd_casimir_shape.py panel (b).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import erfc

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "shape" / "casimir_slice_AAA.json"
R_EW = 7
_grid = np.array(list(itertools.product(range(-R_EW, R_EW + 1), repeat=3)), dtype=float)
_nz = _grid[np.any(_grid != 0, axis=1)]


def gamma_m12(x):
    """Upper incomplete Gamma(-1/2, x) = 2 (x^{-1/2} e^{-x} - sqrt(pi) erfc(sqrt x))."""
    return 2.0 * (np.exp(-x) / np.sqrt(x) - np.sqrt(np.pi) * erfc(np.sqrt(x)))


def Z2(G, theta):
    theta = np.asarray(theta, float)
    Q = np.einsum('ni,ij,nj->n', _nz, G, _nz)
    c = np.cos(2 * np.pi * _nz @ theta)
    direct = np.sum(c * (1 + np.pi * Q) * np.exp(-np.pi * Q) / Q ** 2)
    Gi = np.linalg.inv(G)
    mm = _grid - theta
    P = np.einsum('ni,ij,nj->n', mm, Gi, mm)
    mask = P > 1e-14
    recip = np.sum(np.sqrt(np.pi * P[mask]) * gamma_m12(np.pi * P[mask]))
    periodic = 2.0 if np.all(np.abs(theta - np.round(theta)) < 1e-12) else 0.0
    return direct + np.pi ** 2 / np.sqrt(np.linalg.det(G)) * (recip + periodic) - np.pi ** 2 / 2.0


def Zeff(G, content):
    """content: list of (weight, theta)."""
    return sum(w * Z2(G, th) for w, th in content)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    Xd = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2)
    Xa = np.zeros((3, 3))
    Xa[0, 1] = Xa[1, 0] = 1 / np.sqrt(2)
    content = [(15.0, np.zeros(3)), (-45.0, np.full(3, 0.5))]   # bosons periodic, fermions AAA
    Z0 = Zeff(np.eye(3), content)
    assert abs(Z2(np.eye(3), np.zeros(3)) - 16.5323159598) < 1e-8      # periodic-cube check of C57
    grid = np.linspace(-0.25, 0.25, 61)
    drho = np.empty((grid.size, grid.size))                     # [i_a, i_d]
    for i, ba in enumerate(grid):
        for j, bd in enumerate(grid):
            drho[i, j] = -(Zeff(expm(2 * (bd * Xd + ba * Xa)), content) - Z0) / Z0
    rec = {"description": "Delta rho_C/|rho_C| of the AAA Standard Model on G = exp(2 b_d X_d + 2 b_a X_12); "
                          "rows: b_a (angle), columns: b_d (aspect)",
           "generator": "figures/gen_casimir_slice_shape.py",
           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "content": "15 periodic bosonic units, 45 antiperiodic (AAA) Weyl units; N_C = 15 + 45 * Z2(AAA)/Z2(P)",
           "Z_eff_cube": Z0, "N_C_signed_check": Z0 / Z2(np.eye(3), np.zeros(3)),
           "b_d": grid.tolist(), "b_a": grid.tolist(), "drho": drho.tolist()}
    OUT.write_text(json.dumps(rec), encoding="utf-8")
    print(f"wrote {OUT}  Z_eff(cube) = {Z0:.6f}  N_C = {rec['N_C_signed_check']:.4f}  "
          f"range {drho.min():.4f} .. {drho.max():.4f}")


if __name__ == "__main__":
    main()
