# PROVENANCE: 04_calculations/code/20261011_claude_followup/C80_photon_branch_r.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Photon-pair branch (App. I of v11, D59 Sec. 4): axisymmetric one-pair branch at several zero-mode fractions r = xi; side ratio 4.96 at xi = 0.30 (1.74 free; 3.56/5.43 at 0.258/0.308). Uses the shapelib C60 Casimir. Paper: PRD Sec. VI E (sec:s-excited), App. app:photonpair (app:photonpair).

C80: photon-pair branch of App. I at the adopted zero-mode fraction (R10 review items AUDIT-5, DA-4, EIC-3).

C74 computed the axisymmetric one-pair branch for Q = 0 and for Q = 0.246 C_free (the g = 0.5 pure-gauge value). v1.1
uses r = Q/C_free = 0.30. This script recomputes the branch for several r with an independent implementation:

- Casimir coefficient from the project's twisted Ewald sums (C60_twisted_ewald.Casimir, imported read-only):
  C_free(beta) = Z_eff(G)/pi^2 on G = exp(2 beta X), X = diag(2,-1,-1)/sqrt6, tabulated on a fine beta grid and
  interpolated by a quintic spline (values) whose derivative gives C'(beta);
- C(beta) = C_free(beta) - Q - 4 pi exp(-beta X11), X11 = 2/sqrt6; C_b = C(0);
- minisuperspace equations of App. I, Eq. (I2), in tau = h t, rho = L/L_b:
  (ln rho)' = u, u' = 2C/(C_b rho^4) - v^2/2, beta' = v, v' = -3uv + 3C'(beta)/(C_b rho^4),
  constraint F = 3u^2 - 3 + 3C/(C_b rho^4) - v^2/2 = 0; start at the cube at rest (rho = 1, u = 0);
- adiabaticity parameter eps = (L_b h) rho |v| X11 exp(beta X11)/(2 pi), L_b h = (C_b/3)^{1/4} (h/M_P)^{1/2}.
Validation targets (C74, D56): beta_inf = 0.45108921219 (Q = 0) and 0.97837766 (Q = 10.514045449).

    python -B C80_photon_branch_r.py
Output: results/20261011_claude_followup/C80/C80_results.json
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package: chain root (shape/)
import _shape_paths as WP  # noqa: E402
sys.path.insert(0, str(HERE.parent / "shapelib"))  # package: C60 modules
from C60_twisted_ewald import Casimir, CONTENT_SM  # noqa: E402  (read-only import)

OUT = WP.wdir("fu/C80")
X11 = 2 / math.sqrt(6)
C_FREE_CUBE = 42.7400221506
H_OVER_MP = 6.116e-6


def table(cas, lo=-0.3, hi=2.2, n=1001):
    b = np.linspace(lo, hi, n)
    c = np.array([cas(np.diag([math.exp(2 * x * X11), math.exp(-x * X11), math.exp(-x * X11)]))[0] / math.pi ** 2
                  for x in b])
    return b, c


def branch(spl, dspl, Q, tau_end=12.0):
    Cf = lambda x: float(spl(x)) - Q - 4 * math.pi * math.exp(-x * X11)
    dC = lambda x: float(dspl(x)) + 4 * math.pi * X11 * math.exp(-x * X11)
    Cb = Cf(0.0)

    def rhs(t, y):
        lr, u, b, v = y
        r4 = math.exp(4 * lr)
        return [u, 2 * Cf(b) / (Cb * r4) - v * v / 2, v, -3 * u * v + 3 * dC(b) / (Cb * r4)]
    sol = solve_ivp(rhs, (0.0, tau_end), [0.0, 0.0, 0.0, 0.0], method="DOP853", rtol=1e-11, atol=1e-13,
                    dense_output=True)
    tt = np.linspace(0.0, tau_end, 6001)
    Y = sol.sol(tt)
    lr, u, b, v = Y
    F = 3 * u ** 2 - 3 + 3 * np.array([Cf(x) for x in b]) / (Cb * np.exp(4 * lr)) - v ** 2 / 2
    Lbh = (Cb / 3) ** 0.25 * math.sqrt(H_OVER_MP)
    eps = Lbh * np.exp(lr) * np.abs(v) * X11 * np.exp(b * X11) / (2 * math.pi)
    k10 = np.searchsorted(tt, 10.0)
    return dict(Q=Q, C_b=Cb, beta_inf=float(b[-1]), side_ratio=float(math.exp(math.sqrt(1.5) * b[-1])),
                u_end=float(u[-1]), v_end=float(v[-1]), freeze=float(b[-1] - b[k10]), max_abs_F=float(np.max(np.abs(F))),
                eps_max=float(eps.max()), tau_eps_max=float(tt[int(eps.argmax())]), pair_share=float(4 * math.pi / Cb))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cas = Casimir(CONTENT_SM)
    b, c = table(cas)
    spl = make_interp_spline(b, c, k=5)
    dspl = spl.derivative()
    # spline accuracy at off-grid points
    test = np.linspace(-0.25, 2.1, 9) + 0.00123
    exact = np.array([cas(np.diag([math.exp(2 * x * X11), math.exp(-x * X11), math.exp(-x * X11)]))[0] / math.pi ** 2
                      for x in test])
    err = float(np.max(np.abs(spl(test) - exact) / np.abs(exact)))
    res = {"C_free_cube": float(spl(0.0)), "C_free_cube_ref": C_FREE_CUBE, "spline_max_rel_err": err, "runs": {}}
    print("C_free(0) =", float(spl(0.0)), "spline rel err", err, flush=True)
    for label, r in (("r0", 0.0), ("C74_Q0.246", 10.514045449 / C_FREE_CUBE), ("r0.258", 0.2580), ("r0.286", 0.2860),
                     ("r0.2999", 0.299897), ("r0.308", 0.3083), ("r0.32", 0.32)):
        out = branch(spl, dspl, r * C_FREE_CUBE)
        out["r"] = r
        res["runs"][label] = out
        print(label, {k: (round(v, 10) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    (OUT / "C80_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
