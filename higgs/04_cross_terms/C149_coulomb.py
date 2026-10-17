# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_coulomb.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] D121 run05: background-covariant Coulomb-gauge vev-mixing vertex and Coulomb-mode (Gauss-law) mismatch; RG-closure check of the one-loop r^2 c^2 logs against 2 gamma_phi^C x tree (-> C149_coulomb_check.json).  Paper: PRD App. H2'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 run05: the missing gauge one-loop r^2 c^2 pieces (holonomy-cross, D121 appendix run05).

(a) transverse vev mixing with the background-covariant Coulomb-gauge vertex 2 (c Phi0)^dag V phi  (c_pol = 2/3,
    dH = -(kappa^2/3) S, S = MS-bar FV bubble at mu = 1/L).  Channels (x-units):
      W1 bg: W1->h, W2->pi3, W3->pi_ch (kappa = g2^2 r/2; W3 = (g2 Z + gY A)/gZ), B->pi_ch (kappa = g2 gY r/2; B = (gY Z - g2 A)/gZ)
      Z bg : Z->h (kappa = gZ^2 r/2), W1,2->pi_ch (kappa = gZ g2 r/2)
(b) charged Coulomb mode in a Z background, exact  w_C^2 = (|k - q_pi c|^2 + b)(|k - q_A c|^2 + a_W)/|k - q_A c|^2
    (q_pi = (gY^2 - g2^2)/(2 gZ), q_A = -g2^2/gZ) minus the CCA (w^2 = |k - q_pi c|^2 + a_W + b);
(c) neutral (h, pi3, Z0) in a Z background: kinetic diag(1, Z_k), Z_k = k^2/(k^2 + a_Z), potential with the twist
    t = (gZ/2) c: [[k^2 + m_h^2 + t^2, -2 i t k_x], [2 i t k_x, k^2 + b + t^2]], minus the CCA (h: m_h^2, pi3: a_Z + b).
(b), (c): per mode second difference in c, large-k coefficient C from <2 k^3 Delta_k>, renormalised
    F = sum'[Delta_k - C/(2 w_M^3)] + C S0(M^2),  S0 = -4 d2E_std/d(m^2)^2 (C138_core, MS-bar, mu = 1/L).
RG closure: d/dln mu of (a)+(b)+(c) [+ top, exact] must equal 2 gamma_phi^C x tree, tree_ZZ = r^2 gZ^2/4.
Usage: python -B C149_coulomb.py
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
RES5 = os.path.join(X.ROOT, _zp.w("r3/holonomy_cross/run05"))
k2s = 4 * PI ** 2 * X.QP
NQ = X.NP_
RL = 20
_n = np.arange(-RL, RL + 1)
_N = np.stack(np.meshgrid(_n, _n, _n, indexing="ij"), -1).reshape(-1, 3)
_q = (_N * _N).sum(1)
KV = 2 * PI * _N[(_q > 0) & (_q <= RL * RL)].astype(float)
KN = np.sqrt((KV * KV).sum(1))


def S0(M2):
    h = 1e-3 * max(1.0, abs(M2))
    f = lambda x: float(C.estd(x)[0])  # noqa: E731
    return -4 * (f(M2 + h) - 2 * f(M2) + f(M2 - h)) / h ** 2


def S_ren(m1, m2):
    M = 0.5 * (m1 + m2)
    w1, w2, wM = np.sqrt(k2s + m1), np.sqrt(k2s + m2), np.sqrt(k2s + M)
    return float(S0(M) + np.sum(NQ * (1 / (w1 * w2 * (w1 + w2)) - 1 / (2 * wM ** 3))))


def vev_new(r, g2, gY, lam):
    gZ2 = g2 * g2 + gY * gY
    aW, aZ, b, mh = g2 * g2 / 4 * r * r, gZ2 / 4 * r * r, lam * r * r, 3 * lam * r * r
    mpi, mpi3 = aW + b, aZ + b
    k2W, k2B = (g2 * g2 * r / 2) ** 2, (g2 * gY * r / 2) ** 2
    dWW = -(k2W * (S_ren(mh, aW) + S_ren(mpi3, aW) + g2 * g2 / gZ2 * S_ren(mpi, aZ) + gY * gY / gZ2 * S_ren(mpi, 0.0))
            + k2B * (gY * gY / gZ2 * S_ren(mpi, aZ) + g2 * g2 / gZ2 * S_ren(mpi, 0.0))) / 3
    dZZ = -((gZ2 * r / 2) ** 2 * S_ren(mh, aZ) + 2 * gZ2 * (g2 * r / 2) ** 2 * S_ren(mpi, aW)) / 3
    logWW = -(k2W * (3 + 0.0) + k2B) / (3 * 4 * PI ** 2)     # 4 channels: 3 x k2W (each split sums to 1) + k2B
    logZZ = -((gZ2 * r / 2) ** 2 + 2 * gZ2 * (g2 * r / 2) ** 2) / (3 * 4 * PI ** 2)
    return dWW, dZZ, logWW, logZZ


def renorm(dk, M2):
    sel = KN > PI * RL
    Cc = float(np.mean(2 * KN[sel] ** 3 * dk[sel]))
    wM = np.sqrt(KN ** 2 + M2)
    return float(np.sum(dk - Cc / (2 * wM ** 3)) + Cc * S0(M2)), Cc


def coul_charged(r, g2, gY, lam, hc=1e-3):
    gZ = math.sqrt(g2 * g2 + gY * gY)
    aW, b = g2 * g2 / 4 * r * r, lam * r * r
    qp, qa = (gY * gY - g2 * g2) / (2 * gZ), -g2 * g2 / gZ
    ex = KV[:, 0]

    def E(c):
        kp2 = ((KV - qp * c * ex[:, None] * 0 - np.outer(np.ones(len(KV)), [qp * c, 0, 0])) ** 2).sum(1)
        ka2 = ((KV - np.outer(np.ones(len(KV)), [qa * c, 0, 0])) ** 2).sum(1)
        kp2m = ((KV + np.outer(np.ones(len(KV)), [qp * c, 0, 0])) ** 2).sum(1)
        ka2m = ((KV + np.outer(np.ones(len(KV)), [qa * c, 0, 0])) ** 2).sum(1)
        wp = np.sqrt((kp2 + b) * (ka2 + aW) / ka2)
        wm = np.sqrt((kp2m + b) * (ka2m + aW) / ka2m)
        return 0.5 * (wp + wm)
    exact = (E(hc) - 2 * E(0.0) + E(-hc)) / hc ** 2
    w = np.sqrt(KN ** 2 + aW + b)
    cca = qp * qp * (1 / w - KV[:, 0] ** 2 / w ** 3)
    F, Cc = renorm(exact - cca, aW)
    return F, Cc, 2 * aW * (gZ / 2) ** 2 / 3


def coul_neutral(r, g2, gY, lam, hc=1e-3):
    gZ2 = g2 * g2 + gY * gY
    aZ, b, mh = gZ2 / 4 * r * r, lam * r * r, 3 * lam * r * r
    qn = math.sqrt(gZ2) / 2
    k2 = KN ** 2
    Zk = k2 / (k2 + aZ)

    def E(c):
        t = qn * c
        A, Cc = k2 + mh + t * t, (k2 + b + t * t) / Zk
        Bc2 = (2 * t * KV[:, 0]) ** 2 / Zk
        disc = np.sqrt(0.25 * (A - Cc) ** 2 + Bc2)
        return 0.5 * (np.sqrt(0.5 * (A + Cc) + disc) + np.sqrt(0.5 * (A + Cc) - disc))
    exact = (E(hc) - 2 * E(0.0) + E(-hc)) / hc ** 2
    wh, wp = np.sqrt(k2 + mh), np.sqrt(k2 + aZ + b)
    cca = qn * qn * 0.5 * ((1 / wh - KV[:, 0] ** 2 / wh ** 3) + (1 / wp - KV[:, 0] ** 2 / wp ** 3))
    F, Cc = renorm(exact - cca, aZ)
    return F, Cc, aZ * qn * qn / 3


def main():
    os.makedirs(RES5, exist_ok=True)
    g2, gY, lam = 0.5268, 0.4415, -0.010
    gZ2 = g2 * g2 + gY * gY
    out = {}
    for r in (1.0, 2.0, 3.0, 4.0, 5.0):
        dWW, dZZ, lWW, lZZ = vev_new(r, g2, gY, lam)
        Fb, Cb, Cb_th = coul_charged(r, g2, gY, lam)
        Fc, Ccn, Cc_th = coul_neutral(r, g2, gY, lam)
        req_Z = -(3 * g2 * g2 + gY * gY) / (16 * PI ** 2) * (r * r / 4) * gZ2
        req_W = -(3 * g2 * g2 + gY * gY) / (16 * PI ** 2) * (r * r / 4) * g2 * g2
        logZ = lZZ + (Cb + Ccn) / (4 * PI ** 2)
        out[f"{r}"] = dict(dH_WW_vev=dWW, dH_ZZ_vev=dZZ, dH_ZZ_coul_charged=Fb, dH_ZZ_coul_neutral=Fc,
                           C_charged_fit=Cb, C_charged_theory=Cb_th, C_neutral_fit=Ccn, C_neutral_theory=Cc_th,
                           log_vev_Z_over_required=lZZ / req_Z, log_vev_W_over_required=lWW / req_W,
                           log_coul_Z_over_required=(Cb + Ccn) / (4 * PI ** 2) / req_Z,
                           closure_Z=logZ / req_Z, coul_over_vev_Z_finite=(Fb + Fc) / dZZ)
        print(r, json.dumps(X.clean(out[f"{r}"])), flush=True)
    with open(os.path.join(RES5, "C149_coulomb_check.json"), "w", encoding="utf-8") as f:
        json.dump(X.clean(out), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
