# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_top.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Exact top chiral (axial) bubble in the constant gauge background: G_T1, G_T3, G_Y, H3Y(m_t^2) per colour; verifies the axial log B = q_Phi^2 m_t^2 (= top part of 2 gamma_phi^C x tree) -> r3/holonomy_cross/run04/C149_top_chiral.json.  Paper: PRD App. H2' (top chiral term). Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 run04: exact chiral top bubble in a constant background vs the CCA (holonomy-cross, D121 run04 appendix).

Per AAA mode k = 2 pi (n + 1/2), chiral basis, background c along e_x (c-units, unit coupling):
  Cartan direction (charges q_L of t_L, q_R of t_R):  H = [[-s.(k - q_L c e), m], [m, s.(k - q_R c e)]]  (4x4)
  T1 (non-Cartan, t_L <-> b_L, b massless):           H = [[-s.k + c (s.e) T1, M], [M^dag, s.k]] on (t_L, b_L, t_R) (6x6)
  E_k(c) = -1/2 sum |eigenvalues|;  exact_k = d2E_k/dc2 (central difference);  CCA_k = -sum_chi q_chi^2 (1/w - k_x^2/w^3).
Delta_k = exact_k - CCA_k -> B/(2 k^3) at large k (axial log); renormalised (MS-bar, mu = 1/L):
  G = sum_k [Delta_k - B/(2 w_k^3)] + B S_A(m^2),  S_A = -4 d2 E_std,A/d(m^2)^2,  E_std,A = D_A + V_CW(3/2)  (C138_core).
RG check: B must equal q_Phi^2 m^2 per colour (top part of 2 gamma_phi^C x tree, q_Phi = charge of the vev component).
Output (per colour, c-units): G_T3 (q_L=-1/2, q_R=0), G_Y (1/6, 2/3), G_T3Y (-1/3, 2/3), G_T1 on an m^2 grid.
Usage: python -B C149_top.py
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_core as X  # noqa: E402
import C138_core as C  # noqa: E402

PI = math.pi
RES4 = os.path.join(X.ROOT, _zp.w("r3/holonomy_cross/run04"))
RMAX = 22
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)


def lattice():
    v = np.arange(-RMAX - 1, RMAX + 1) + 0.5
    n = np.stack(np.meshgrid(v, v, v, indexing="ij"), -1).reshape(-1, 3)
    q = (n * n).sum(1)
    n = n[q <= RMAX * RMAX]
    return 2 * PI * n


K3 = lattice()
KN = np.sqrt((K3 * K3).sum(1))


def sdot(p):
    return p[:, 0, None, None] * sx + p[:, 1, None, None] * sy + p[:, 2, None, None] * sz


def E_cartan(c, m, qL, qR):
    e = np.array([1.0, 0, 0])
    pL, pR = K3 - qL * c * e, K3 - qR * c * e
    H = np.zeros((len(K3), 4, 4), complex)
    H[:, :2, :2] = -sdot(pL)
    H[:, 2:, 2:] = sdot(pR)
    H[:, :2, 2:] = m * np.eye(2)
    H[:, 2:, :2] = m * np.eye(2)
    return -0.5 * np.abs(np.linalg.eigvalsh(H)).sum(1)


def E_T1(c, m):
    H = np.zeros((len(K3), 6, 6), complex)
    sk = sdot(K3)
    H[:, 0:2, 0:2] = -sk
    H[:, 2:4, 2:4] = -sk
    H[:, 0:2, 2:4] = 0.5 * c * sx
    H[:, 2:4, 0:2] = 0.5 * c * sx
    H[:, 4:6, 4:6] = sk
    H[:, 0:2, 4:6] = m * np.eye(2)
    H[:, 4:6, 0:2] = m * np.eye(2)
    return -0.5 * np.abs(np.linalg.eigvalsh(H)).sum(1)


def S_A(m2):
    h = 1e-3 * max(1.0, m2)
    f = lambda x: float(C.DA(x)[0] + C.vcw(x, 1.5)[0])  # noqa: E731
    return -4 * (f(m2 + h) - 2 * f(m2) + f(m2 - h)) / h ** 2


def G_dir(m2, kind, qL=None, qR=None, hc=2e-3):
    m = math.sqrt(m2)
    if kind == "T1":
        E = lambda c: E_T1(c, m)  # noqa: E731
    else:
        E = lambda c: E_cartan(c, m, qL, qR)  # noqa: E731
    ex = (E(hc) - 2 * E(0.0) + E(-hc)) / hc ** 2
    w = np.sqrt(KN ** 2 + m2)
    kx2 = K3[:, 0] ** 2
    if kind == "T1":
        cca = -0.25 * (1 / w - kx2 / w ** 3) - 0.25 * (1 / KN - kx2 / KN ** 3)
    else:
        cca = -(qL ** 2 + qR ** 2) * (1 / w - kx2 / w ** 3)
    d = ex - cca
    # B from shells in the outer half: <2 k^3 Delta_k>
    sel = KN > PI * RMAX
    B = float(np.mean(2 * KN[sel] ** 3 * d[sel])) if m2 > 0 else 0.0
    G = float(np.sum(d - B / (2 * w ** 3)) + B * S_A(m2))
    return G, B


def main():
    os.makedirs(RES4, exist_ok=True)
    m2g = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    dirs = {"T3": (-0.5, 0.0, 0.25), "Y": (1 / 6., 2 / 3., 0.25), "T3Y": (-1 / 3., 2 / 3., 1.0), "T1": (None, None, 0.25)}
    out = {"m2": m2g, "RMAX": RMAX}
    for nm, (qL, qR, qphi2) in dirs.items():
        Gs, Bs = [], []
        for m2 in m2g:
            G, B = G_dir(m2, "T1" if nm == "T1" else "C", qL, qR)
            Gs.append(G)
            Bs.append(B)
        out[nm] = dict(G=Gs, B=Bs, B_over_qphi2_m2=[b / (qphi2 * m2) if m2 > 0 else None for b, m2 in zip(Bs, m2g)],
                       qphi2=qphi2)
        print(nm, "G:", ["%.3e" % g for g in Gs], "B/(qphi^2 m^2):",
              ["%.3f" % x if x is not None else "-" for x in out[nm]["B_over_qphi2_m2"]], flush=True)
    # mixed T3-Y and universality
    out["H3Y"] = [0.5 * (a - b - c) for a, b, c in zip(out["T3Y"]["G"], out["T3"]["G"], out["Y"]["G"])]
    out["universality_T3_over_Y"] = [a / b if abs(b) > 1e-14 else None for a, b in zip(out["T3"]["G"], out["Y"]["G"])]
    out["S_A"] = [S_A(m2) if m2 > 0 else None for m2 in m2g]
    with open(os.path.join(RES4, "C149_top_chiral.json"), "w", encoding="utf-8") as f:
        json.dump(X.clean(out), f, ensure_ascii=False, indent=1)
    print("H3Y:", ["%.3e" % x for x in out["H3Y"]], "T3/Y:", out["universality_T3_over_Y"], "S_A:", out["S_A"], flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
