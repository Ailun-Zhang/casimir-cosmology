# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_profile.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Adiabatic 2D_ex barrier, wall positions (r_top, r2(Q)) and guide checks for lam = -0.010 / -0.013.  Paper: PRD H2.

T14a step 1: adiabatic barrier profile, wall choice, radial-guide checks.  python -B T14a_profile.py"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import sys
import numpy as np
import T14a_common as T

sys.stdout.reconfigure(encoding="utf-8")


def main():
    old = T.jload(T.OLDSUM)
    out = {}
    r = np.arange(0.2, 9.0, 0.001)
    for lam in (-0.010, -0.013):
        Q = old["Q"]["2D_ex"][f"{lam}"]
        rec = {"Q_2D_ex_siegert": Q, "gamma_2D_ex": old["gamma"]["2D_ex"][f"{lam}"],
               "Q_2D_G_siegert": old["Q"]["2D_G"][f"{lam}"], "gamma_2D_G": old["gamma"]["2D_G"][f"{lam}"]}
        for var in ("2D_ex", "2D_G"):
            U = T.u_adiabatic(var, lam, r)
            i0 = np.argmin(np.abs(r - 2.0))
            it = i0 + int(np.argmax(U[i0:]))
            Qv = old["Q"][var][f"{lam}"]
            above = np.where(U[it:] < Qv)[0]
            r2 = float(r[it + above[0]]) if len(above) else None
            below = np.where(U[:it] < Qv)[0]
            r1 = float(r[below[-1]]) if len(below) else None
            rec[var] = dict(r_top=float(r[it]), U_top=float(U[it]), r1=r1, r2=r2,
                            U_at={f"{x}": float(U[np.argmin(np.abs(r - x))]) for x in (1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0,
                                                                                         5.5, 6.0, 6.5, 7.0, 7.5)})
        out[f"{lam}"] = rec
        print(lam, rec, flush=True)
    # radial guide checks at lam = 0 (rw = 9) and lam = -0.010 (rw = 5.6)
    for lam, rw in ((0.0, 9.0), (-0.010, 5.6), (-0.013, 5.6)):
        g = T.radial_guide(lam, rw)
        out[f"guide_{lam}_{rw}"] = dict(eps=g["eps"], consistency=T.guide_consistency(g),
                                        f_at={f"{x}": float(np.interp(x, g["r"], g["f"])) for x in (0.0, 0.5, 1, 2, 3, 4, 5)})
        print("guide", lam, rw, out[f"guide_{lam}_{rw}"], flush=True)
    dvs = T.dv_table(-0.010, 7.0)
    out["dV_MC_m0p010"] = {f"{x}": float(dvs(x)) for x in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)}
    print("dV", out["dV_MC_m0p010"], flush=True)
    T.wjson(T.RES + r"\T14a_profile.json", out)


if __name__ == "__main__":
    main()
