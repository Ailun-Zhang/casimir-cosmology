# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_centre.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent universality (C156/D128, v12 round 3) on 2026-10-14; imports U156_twist (same dir).
r"""[v12 package, 2026-10-15] SU(3) centre-holonomy vacuum of the PPA/PAA spin structures (N_C 13.124/20.871 -> 21.068/23.669).  Paper: PRD Sec. IV D.

C156 T2c: PPA / PAA at their one-loop holonomy vacuum (SU(3) centre k/3 on every periodic cycle; U156_hol2 result).
Quarks: twist th_F + k_i/3 on the periodic cycles (antitriplets -k/3, same sums); leptons th_F; bosons untwisted.
  Z_eff = 15 Z_2(G,0) - 3 [12 Z_2(G, th_q) + 3 Z_2(G, th_l)]  -> N_C, tetragonal force F, kappa_tet, stationary points;
  per-direction holonomy curvatures (D121 normalisation, nonzero modes):
    kappa_Y,i = [Z_P - 11 Y_i(th_q) - 9 Y_i(th_l)]/(6 pi^2), kappa_2,i = [5 Z_P - 9 Y_i(th_q) - 3 Y_i(th_l)]/(6 pi^2),
    kappa_3,i = [Z_P - 2 Y_i(th_q)]/pi^2;  Omega_X,i^2 = kappa_Y,i g_Y^2; small-r inversion radius r_inv = 2 sqrt(-kappa_Y,i);
  top tadpole Z_1(th_q) (AAA: -2.519356); L_b, hL_b, eta_inf, mu_b with Q = 12.8 (conditional).
Usage: python -B U156_centre.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import U156_twist as TW  # noqa: E402

PI = math.pi
ONE = (1.0, 1.0, 1.0)
ZP = TW.Zs(1, ONE, (0, 0, 0))
Z20 = TW.Zs(2, ONE, (0, 0, 0))


def zeff(a, thq, thl):
    return 15 * TW.Zs(2, a, (0, 0, 0)) - 3 * (12 * TW.Zs(2, a, thq) + 3 * TW.Zs(2, a, thl))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cases = {"AAA_trivial": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 2),
             "PPA_trivial": ((0, 0, 0.5), (0, 0, 0.5), 2),
             "PPA_centre(1,1)": ((1 / 3, 1 / 3, 0.5), (0, 0, 0.5), 2),
             "PPA_centre(1,2)": ((1 / 3, 2 / 3, 0.5), (0, 0, 0.5), 2),
             "PPA_centre(1,0)": ((1 / 3, 0, 0.5), (0, 0, 0.5), 2),
             "PAA_trivial": ((0, 0.5, 0.5), (0, 0.5, 0.5), 0),
             "PAA_centre(1)": ((1 / 3, 0.5, 0.5), (0, 0.5, 0.5), 0)}
    out = {}
    ref = None
    for name, (thq, thl, axis) in cases.items():
        z0 = zeff(ONE, thq, thl)
        NC = z0 / Z20
        Yq = [TW.Ydir(thq, i) for i in range(3)]
        Yl = [TW.Ydir(thl, i) for i in range(3)]
        kY = [(ZP - 11 * Yq[i] - 9 * Yl[i]) / (6 * PI ** 2) for i in range(3)]
        k2 = [(5 * ZP - 9 * Yq[i] - 3 * Yl[i]) / (6 * PI ** 2) for i in range(3)]
        k3 = [(ZP - 2 * Yq[i]) / PI ** 2 for i in range(3)]
        h = 1e-3
        zp = (zeff(TW.tet(h, axis), thq, thl) - zeff(TW.tet(-h, axis), thq, thl)) / (2 * h)
        zpp = (zeff(TW.tet(h, axis), thq, thl) - 2 * z0 + zeff(TW.tet(-h, axis), thq, thl)) / h ** 2
        bs = np.round(np.arange(-1.5, 1.5001, 0.25), 3)
        zz = [zeff(TW.tet(b, axis), thq, thl) for b in bs]
        dz = np.gradient(np.array(zz), bs)
        stat = [float(0.5 * (bs[j] + bs[j + 1])) for j in range(len(bs) - 1) if dz[j] * dz[j + 1] < 0]
        NCe = NC - 12.8 / TW.GAM
        r = dict(N_C=NC, Zeff=z0, Yq=Yq, Yl=Yl, kappa_Y=kY, kappa_2=k2, kappa_3=k3,
                 OmegaX2=[k * 0.4427 ** 2 for k in kY], r_inv=[2 * math.sqrt(-k) if k < 0 else 0.0 for k in kY],
                 top_tadpole_Z1=TW.Zs(1, ONE, thq), Zp=zp, Zpp=zpp, F=3 * zp / z0, kappa_tet=3 * zpp / z0,
                 F_eff=3 * zp / z0 / (1 - 12.8 / (NC * TW.GAM)) if NCe > 0 else None, stationary=stat,
                 beta_scan=bs.tolist(), Zeff_scan=zz, N_C_eff=NCe)
        if name == "AAA_trivial":
            ref = NCe
        out[name] = r
        print(name, json.dumps({k: r[k] for k in ("N_C", "kappa_Y", "kappa_2", "kappa_3", "r_inv", "top_tadpole_Z1", "F",
                                                  "kappa_tet", "stationary", "N_C_eff")}), flush=True)
    for name, r in out.items():
        if r["N_C_eff"] > 0:
            fac = (r["N_C_eff"] / ref) ** 0.25
            r.update(LbMP=722.0 * fac, hLb=0.00437243 * fac, eta_inf=1.31102877714606 / (0.00437243 * fac),
                     mu_b_GeV=2.435e18 / (722.0 * fac), dln_eta_vs_AAA=-math.log(fac))
            sym, full = TW.shape_ode(r["kappa_tet"] / (1 - 12.8 / (r["N_C"] * TW.GAM)), r["F_eff"])
            r.update(dbeta_half_sym=sym, dbeta_from_rest=full)
    with open(os.path.join(TW.OUT, "C156_centre.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {x: v.get(x) for x in ("N_C_eff", "hLb", "eta_inf", "mu_b_GeV", "dln_eta_vs_AAA", "dbeta_half_sym",
                                                  "dbeta_from_rest")} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
