# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/two_branch/tb_static.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Static-spectrum propagation cross-check of the two-branch quantities (independent of the TDSE grid).  Paper: PRD H3.

C159 / D130 independent check: frozen (u = 0) zero-mode Hamiltonian, propagation by dense eigen-decomposition
(no Crank-Nicolson).  Static H is trivially time-symmetric, U(T,-T) = exp(-2iHT).
    A = P e^{-iHT} psi0,  X = P e^{-2iHT} conj(A),  K = <A|X>,  G = |X|^2,  eps = |K-G|/sqrt(G |A-X|^2)
Resonance limit: K = P1^2/w0, G = P1^3/w0^2, eps = 1.  Also a CN run of the same frozen problem (same grid) at two T.
Usage: python -B tb_static.py
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
sys.dont_write_bytecode = True
import json
import math
import time
import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hr_core as C  # noqa: E402
from hr_potential import PotentialTable  # noqa: E402

ROOT = _zp.WORK
TAB = os.path.join(ROOT, _zp.w("r2/higgs_realtime/run01/tables"))
OUT = os.path.join(ROOT, _zp.w("r3/two_branch/run01"))


def one(mt, dr=0.02, Ts=(25, 50, 100, 200, 400, 800), cn_T=(100, 400)):
    tab = PotentialTable(os.path.join(TAB, "U_table_MU_mt%.1f.npz" % mt))
    g = C.Grid(dr, 16.0)
    fr = C.RFrame(tab, g, 1.0 / 6, r_c=10.0, cap0=20.0)
    x, dx = g.x, g.dx
    V = fr.V(0.0)
    lm = C.landmarks(x, fr.U_real(0.0), 4.4)
    E, v = C.resonance(g, V, 4.4 - 0.0005j, k=8, inner=lm["i_top"])
    lm = C.landmarks(x, fr.U_real(0.0), E.real)
    itop = int(np.searchsorted(x, lm["r_top"]))
    H = C.h_sparse(g, 0.5, V).toarray()
    w, VR = sla.eig(H)
    LU = sla.lu_factor(VR)
    a = np.real(v).copy()
    a[x > lm["r_exit"]] = 0
    i = lm["i_min"]
    U0 = fr.U_real(0.0)
    om = math.sqrt((U0[i + 1] - 2 * U0[i] + U0[i - 1]) / dx ** 2)
    gs = np.exp(-(x - lm["r_min"]) ** 2 / (4 * (1.0 / math.sqrt(2 * om)) ** 2)) * (1 - np.exp(-(x / 0.3) ** 2))
    res = dict(mt=mt, dr=dr, E_res=[E.real, E.imag], gamma0=-2 * E.imag, r_top=lm["r_top"], states={})
    for nm, s in (("a_gamow_real", a), ("c_gauss1.0", gs)):
        psi = s.astype(complex) / math.sqrt(np.sum(s * s) * dx)
        w0 = float(abs(np.sum(v * psi) * dx) ** 2)
        c0 = sla.lu_solve(LU, psi)
        recon = float(np.max(np.abs(VR @ c0 - psi)))
        recs = []
        for T in Ts:
            pT = VR @ (np.exp(-1j * w * T) * c0)
            A = np.where(np.arange(g.N) < itop, pT, 0)
            cX = sla.lu_solve(LU, np.conj(A))
            X = np.where(np.arange(g.N) < itop, VR @ (np.exp(-2j * w * T) * cX), 0)
            P1 = float(np.sum(np.abs(A) ** 2) * dx)
            G = float(np.sum(np.abs(X) ** 2) * dx)
            K = complex(np.sum(np.conj(A) * X) * dx)
            Y2 = float(np.sum(np.abs(A - X) ** 2) * dx)
            recs.append(dict(T=T, P1=P1, K_re=K.real, K_im=K.imag, G=G, eps=abs(K - G) / math.sqrt(G * Y2),
                             R_K=K.real * w0 / P1 ** 2, R_G=G * w0 ** 2 / P1 ** 3,
                             Phaz=math.exp(-res["gamma0"] * T)))
        cn = []
        for T in cn_T:
            # same frozen problem with Crank-Nicolson (deta = 0.025), cross-check of the stepping used in tb_tdse.py
            n = int(round(T / 0.025))
            hh = T / n
            p = psi.copy()
            for _ in range(n):
                p = C.cn_step(p, g, 0.5, V - E.real, hh)
            A = np.where(np.arange(g.N) < itop, p, 0)
            q = np.conj(A)
            for _ in range(2 * n):
                q = C.cn_step(q, g, 0.5, V - E.real, hh)
            X = np.where(np.arange(g.N) < itop, q, 0)
            P1 = float(np.sum(np.abs(A) ** 2) * dx)
            G = float(np.sum(np.abs(X) ** 2) * dx)
            K = complex(np.sum(np.conj(A) * X) * dx)
            cn.append(dict(T=T, P1=P1, K_re=K.real, K_im=K.imag, G=G))
        res["states"][nm] = dict(w0=w0, recon_err=recon, recs=recs, cn=cn)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    out = [one(172.6), one(172.2)]
    for r in out:
        print("mt", r["mt"], "gamma0 %.5e" % r["gamma0"])
        for nm, s in r["states"].items():
            print(" ", nm, "w0=%.5f recon=%.1e" % (s["w0"], s["recon_err"]))
            for q in s["recs"]:
                print("   T=%4d P1=%.5f K=%.6f%+.1ej G=%.6f eps=%.4f R_K=%.5f R_G=%.5f" %
                      (q["T"], q["P1"], q["K_re"], q["K_im"], q["G"], q["eps"], q["R_K"], q["R_G"]))
            for q in s["cn"]:
                print("   CN T=%4d P1=%.5f K=%.6f G=%.6f" % (q["T"], q["P1"], q["K_re"], q["G"]))
    with open(os.path.join(OUT, "static_check.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(code=os.path.abspath(__file__), time_s=time.time() - t0), runs=out), f, indent=1)
    print("time %.1f s" % (time.time() - t0))
