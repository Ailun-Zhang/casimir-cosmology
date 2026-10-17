# -*- coding: utf-8 -*-
"""R14 check 3b: the end of the turn (u = ht in [3,5]) with root continuation in u.
Conformal-time zero-mode Hamiltonian gets -(1/2)(1-6 xi_H)(L''/L) r^2, L''/L = 2 (hL_b)^2 cosh 2u.
xi_H = 0 (D108's minimal coupling), 1/6 (conformal), 1/2 (stabilising).  MU model, SM couplings run to
mu = mu_b/sqrt(cosh 2u), 2-loop + 3-loop calibration (R14_rge).  Continuation: the converged pole at node u_i is
the Muller start at u_{i+1} (the plain fixed start in check3 jumped to a spurious pole at u=4 for m_t>=172.5).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import math
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import R14_res as R  # noqa: E402
import R14_rge  # noqa: E402


def main():
    t0 = time.time()
    cal, _ = R14_rge.calibration()
    un = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0001, 0.125)]), 6)
    mus = R14_rge.MU_B / np.sqrt(np.cosh(2 * un))
    Yc = R14_rge.solve(172.6, 0.1180, 125.13).sol(np.log(mus))
    profs = [R.gauss_profile(float(Yc[1, i]), float(Yc[0, i])) for i in range(len(un))]
    combos = [(172.0, 0.1180), (172.2, 0.1180), (172.6, 0.1180)]
    Y = {cb: R14_rge.solve(cb[0], cb[1], 125.13).sol(np.log(mus)) for cb in combos}
    r, m = R.grid(16.0)
    sp_ = [R.spar(p) for p in profs]
    out = dict(cal=cal, u=un.tolist(), HL=(R.HLB * np.sqrt(np.cosh(2 * un))).tolist())
    uf = np.linspace(0, 5.0, 10001)
    wf = 1.0 / (R.HLB * np.sqrt(np.cosh(2 * uf)))
    for tag, fac in (("xi=0", 1.0), ("xi=1/6", 0.0), ("xi=1/2", -2.0)):
        G = np.zeros((len(combos), len(un)))
        Q = np.zeros_like(G)
        guess = None
        prev = None
        diag = []
        for i, u in enumerate(un):
            at = fac * R.HLB ** 2 * math.cosh(2 * u)
            cols = np.array([R.U_radial(profs[i], "MU", float(Y[cb][4, i] + cal), float(Yc[1, i]),
                                        float(Yc[0, i]), float(Y[cb][3, i]), r, at) for cb in combos])
            s = np.full(len(combos), sp_[i][0])
            al = np.full(len(combos), sp_[i][1])
            E, it, rs = R.solve_set(list(cols), r, m, s, al, guess=guess)

            def ok(E_, g_):
                if not np.all(np.isfinite(E_)):
                    return False
                if g_ is None:
                    return True
                return bool(np.all(np.abs(E_.real - g_.real) < 0.25) and np.all(-2 * E_.imag > -1e-9)
                            and np.all(np.abs(np.log(np.maximum(-2 * E_.imag, 1e-300) /
                                                     np.maximum(-2 * g_.imag, 1e-300))) < 3.5))
            if not ok(E, guess):   # homotopy in the potential from the previous node
                Eh = guess.copy()
                for t in np.linspace(0.125, 1.0, 8):
                    colsh = (1 - t) * prev + t * cols
                    Eh, _, _ = R.solve_set(list(colsh), r, m, s, al, guess=Eh)
                diag.append(dict(u=float(u), first=[str(x) for x in E], homotopy=[str(x) for x in Eh]))
                E = Eh
            guess = E.copy()
            prev = cols
            G[:, i] = -2 * E.imag
            Q[:, i] = E.real
        res = {"diag": diag}
        for j, cb in enumerate(combos):
            spl = CubicSpline(un, np.log(np.maximum(G[j], 1e-300)))
            gf = np.exp(spl(uf))
            cum = {}
            for umax in (3.0, 3.5, 4.0, 4.5, 5.0):
                sel = uf <= umax + 1e-12
                I = float(simpson(gf[sel] * wf[sel], x=uf[sel]))
                tail = float(G[j, list(un).index(umax)] * R.eta_rem(umax))
                cum[str(umax)] = dict(I=I, P1=math.exp(-I), I_plus_frozen_tail=I + tail,
                                      P1_frozen_tail=math.exp(-I - tail))
            res[f"Mt={cb[0]}"] = dict(gamma=G[j].tolist(), Q=Q[j].tolist(), cumulative=cum,
                                      lam_nodes=[float(Y[cb][4, i] + cal) for i in range(len(un))])
        out[tag] = res
    out["eta_rem"] = {str(u): R.eta_rem(u) for u in (3.0, 3.5, 4.0, 4.5, 5.0)}
    out["time_s"] = time.time() - t0
    R.merge_json("results.json", "check3b_tail", out)
    k = out["xi=0"]["Mt=172.6"]["cumulative"]
    R.merge_json("STATUS.json", "check3b", dict(done=True, P1_xi0_172p6={u: k[u]["P1"] for u in k},
                                                P1_xi16_172p6_u3=out["xi=1/6"]["Mt=172.6"]["cumulative"]["3.0"]
                                                ["P1_frozen_tail"]))
    # compact print
    for tag in ("xi=0", "xi=1/6", "xi=1/2"):
        print(tag, "homotopy fallbacks:", [(d["u"], d["first"][:1], d["homotopy"][:1]) for d in out[tag]["diag"]])
        for key, v in out[tag].items():
            if key == "diag":
                continue
            print(tag, key, "gamma(u=0,1,2,3,3.5,4,4.5,5)=",
                  " ".join("%.3e" % v["gamma"][list(un).index(x)] for x in (0, 1, 2, 3, 3.5, 4, 4.5, 5)),
                  "| Q(3,4,5)=", " ".join("%.3f" % v["Q"][list(un).index(x)] for x in (3, 4, 5)))
            print("   P1 cumulative to u=3,3.5,4,4.5,5:",
                  " ".join("%.3f" % v["cumulative"][str(x)]["P1"] for x in (3.0, 3.5, 4.0, 4.5, 5.0)),
                  "| with frozen tail:", " ".join("%.3f" % v["cumulative"][str(x)]["P1_frozen_tail"]
                                                   for x in (3.0, 3.5, 4.0, 4.5, 5.0)))
    print("eta_rem", out["eta_rem"], "time", out["time_s"])


if __name__ == "__main__":
    main()
