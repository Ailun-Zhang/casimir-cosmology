# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_hol2.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent universality (C156/D128, v12 round 3) on 2026-10-14; imports U156_twist.Zs (same dir).
r"""[v12 package, 2026-10-15] Holonomy vacua of the PPA/PAA sectors (second pass).  Paper: PRD Sec. IV D.

C156 T2b: electroweak x SU(3)-centre holonomy vacuum along ONE periodic cycle for PPA / PAA (AAA, PPA-A as controls).
Background on cycle i: U(1)_Y twist delta_Y (twist of a Y = 1 field), SU(2) Cartan exp(2 pi i delta_3 T3), SU(3) centre k3.
Field twist on cycle i: th_F,i + Y delta_Y + T3 delta_3 + k3 * triality/3.  One-loop energy (L = 1) = full free image sum
E = -(1/pi^2) sum_f d_f s_f Z_2(th_f) (harmonic n = 0 modes of twisted bosons included).  Untwisted periodic bosons carry
zero modes whose leading-order energy replaces their (vanishing) harmonic value: + Q_W = eps_2 g2^(2/3) = 2.70 when the W
are untwisted (delta_3 in Z), + Q_HY = 1.07 gamma = 1.79 when the whole Higgs doublet is untwisted (v11 App. SM).
Identification (Z_6): (dY, d3, k3) ~ (dY+1, d3+1, k3+1) -> scan dY in [0,1], d3 in [0,2), k3 = 0,1,2.
Reports per case: global minimum (with / without the zero-mode corrections), the minimum with a Higgs component untwisted
(d3 = +-dY mod 2; the large-r constraint), Delta V_untwist, the Higgs gap at the global minimum, and Delta N_C.
Usage: python -B U156_hol2.py
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
import time
import numpy as np
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import U156_twist as TW  # noqa: E402

OUT = TW.OUT
PI = math.pi
QW, QHY = 4.1167 * 0.5277 ** (2 / 3), 1.07 * TW.GAM
# Weyl components (left-handed): (mult per generation incl. colour, Y, T3, triality)
COMP = [(3, 1 / 6, +0.5, 1), (3, 1 / 6, -0.5, 1), (3, -2 / 3, 0.0, -1), (3, 1 / 3, 0.0, -1), (1, -0.5, +0.5, 0),
        (1, -0.5, -0.5, 0), (1, 1.0, 0.0, 0)]


def w(x):
    return x - math.floor(x + 0.5)


@lru_cache(maxsize=None)
def Z2c(t0, t1, t2):
    return TW.Zs(2, (1.0, 1.0, 1.0), (t0, t1, t2))


def Z2(th):
    return Z2c(*(round(w(x), 10) for x in th))


def energy(i, thF, dY, d3, k3):
    e = np.zeros(3)
    e[i] = 1.0
    E = 0.0
    tw = {}
    for t3 in (+0.5, -0.5):                                     # Higgs components, complex scalar d = 1, Y = 1/2
        x = 0.5 * dY + t3 * d3
        tw["H%+.1f" % t3] = abs(w(x))
        E += -(1.0 / PI ** 2) * Z2(x * e)
    E += -(2.0 / PI ** 2) * Z2(d3 * e)                         # W+-: complex vector, d = 2
    for mult, Y, T3, tri in COMP:                              # fermions s = -1, d = 1, 3 generations
        x = Y * dY + T3 * d3 + tri * k3 / 3.0
        E += (3.0 * mult / PI ** 2) * Z2(np.array(thF) + x * e)
    zW = abs(w(d3)) < 1e-9
    zH = tw["H+0.5"] < 1e-9 and tw["H-0.5"] < 1e-9
    return E, E + (QW if zW else 0.0) + (QHY if zH else 0.0), tw


def scan(case, i, thF, step=0.05):
    rows = []
    for k3 in (0, 1, 2):
        for dY in np.round(np.arange(0.0, 1.0001, step), 6):
            for d3 in np.round(np.arange(0.0, 2.0, step), 6):
                E0, E1, tw = energy(i, thF, dY, d3, k3)
                rows.append((k3, float(dY), float(d3), E0, E1, min(tw.values())))
    A = np.array([r[3] for r in rows])
    B = np.array([r[4] for r in rows])
    unt = np.array([r[5] < 1e-9 for r in rows])
    E_triv0, E_triv1, _ = energy(i, thF, 0.0, 0.0, 0)
    out = dict(case=case, dir=i, E_trivial=[E_triv0, E_triv1])
    for lab, V in (("harmonic", A), ("with_zero_modes", B)):
        j = int(np.argmin(V))
        ju = int(np.argmin(np.where(unt, V, np.inf)))
        out[lab] = dict(glob=dict(k3=rows[j][0], dY=rows[j][1], d3=rows[j][2], E=float(V[j]), higgs_min_twist=rows[j][5],
                                  higgs_gap=2 * PI * rows[j][5]),
                        untw=dict(k3=rows[ju][0], dY=rows[ju][1], d3=rows[ju][2], E=float(V[ju])),
                        dV_untwist=float(V[ju] - V[j]), dE_vs_trivial=float(V[j] - (E_triv0 if lab == "harmonic" else E_triv1)),
                        dNC_vs_trivial=float(-(V[j] - (E_triv0 if lab == "harmonic" else E_triv1)) / TW.GAM))
    # named points
    pts = {"trivial": (0, 0.0, 0.0), "SU3centre": (1, 0.0, 0.0), "Y=1 (=SU3xSU2 centre)": (0, 1.0, 0.0),
           "SU2centre": (0, 0.0, 1.0)}
    out["points"] = {k: dict(zip(("harmonic", "with_zero_modes"), energy(i, thF, v[1], v[2], v[0])[:2])) for k, v in pts.items()}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    res = {}
    for case, i in (("AAA", 0), ("PPA", 0), ("PPA", 2), ("PAA", 0), ("PAA", 1)):
        r = scan(case, i, TW.SPIN[case])
        res["%s_dir%d" % (case, i)] = r
        print(case, i, "%.0fs" % (time.time() - t0), json.dumps({k: r[k] for k in ("E_trivial", "harmonic", "with_zero_modes",
                                                                                   "points")}), flush=True)
    # PPA: both periodic cycles carry the minimising background (check of additivity at the named points)
    thF = TW.SPIN["PPA"]
    two = {}
    for lab, (k3, dY, d3) in {"trivial": (0, 0.0, 0.0), "Y=1 both": (0, 1.0, 0.0), "SU3centre both": (1, 0.0, 0.0),
                              "glob both": (res["PPA_dir0"]["with_zero_modes"]["glob"]["k3"],
                                            res["PPA_dir0"]["with_zero_modes"]["glob"]["dY"],
                                            res["PPA_dir0"]["with_zero_modes"]["glob"]["d3"])}.items():
        E = 0.0
        e = np.array([1.0, 1.0, 0.0])
        for t3 in (+0.5, -0.5):
            E += -(1.0 / PI ** 2) * Z2((0.5 * dY + t3 * d3) * e)
        E += -(2.0 / PI ** 2) * Z2(d3 * e)
        for mult, Y, T3, tri in COMP:
            E += (3.0 * mult / PI ** 2) * Z2(np.array(thF) + (Y * dY + T3 * d3 + tri * k3 / 3.0) * e)
        two[lab] = E
    res["PPA_two_cycles"] = two
    res["Q_W"], res["Q_HY"], res["time_s"] = QW, QHY, time.time() - t0
    with open(os.path.join(OUT, "C156_hol2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(two), "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
