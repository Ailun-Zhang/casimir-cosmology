# PROVENANCE: new script written 2026-10-16 by rev-code (R18 revision, DA-04) for the v12 code/data package.
"""[v12 package, R18] What the one-loop finite-volume matching does to the zero-mode potential (D112 2D_ex model,
lambda = -0.010 and -0.013): the fraction by which it lowers the local slope of the gauge zero-point confinement at
r = 1-5, and the barrier height above the resonance level with and without it.

Inputs (package files, read directly):
  figures/data/higgs/generated/potential_2Dex_lam010_013.csv   columns per lambda: Uad (adiabatic radial potential,
      incl. the 4D centrifugal term 3/(8 r^2)), Ug = Uad - 3/(8 r^2), quartic = lam r^4/4, dV = one-loop MC
      finite-volume matching, zeropoint = Ug - quartic - dV (gauge zero-point energy); written by
      figures/prepare_data_higgs.py (gen_potential) with the production code, grid r = 0.02 ... 9 (step 0.01, units 1/L)
  higgs/05_fulldim_validation/expected/r3/fulldim_dmc/run01/T14a_profile.json   Q_2D_ex_siegert (resonance level)
Output (work tree): w/matching_potential/matching_potential.json (no time stamp).
  slope_reduction[r]   = -(d dV/dr) / (d zeropoint/dr) at r = 1, 2, 3, 4, 5 (central differences on the grid)
  barrier_above_Q      = max_{r>1.5} U - Q with the matching (U) and without it (U - dV), for U = Uad (the potential of
                         the radial problem whose top T14a_profile records as U_top; headline) and for U = Ug (the
                         field-space potential without 3/(8 r^2); the R18 referee's probe)
Paper: PRD Sec. V (effective slope sigma_eff ~ sigma_0/3), report Secs. 6-7 ([num:] pointers; no macro).
Run: python -B matching_potential.py
"""
import os as _os  # path bootstrap: zmlib + package root (as in the other packaged scripts)
import sys as _sys
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")
if _ZM not in _sys.path:
    _sys.path.insert(1, _ZM)
import _zpaths as _zp  # noqa: E402
import csv
import json
import os
import sys

import numpy as np

CSV = os.path.join(_zp.PKG, "figures", "data", "higgs", "generated", "potential_2Dex_lam010_013.csv")
PROF = os.path.join(_zp.HIGGS, "05_fulldim_validation", "expected", "r3", "fulldim_dmc", "run01", "T14a_profile.json")
OUT = _zp.w("w/matching_potential/matching_potential.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    col = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    r = col["r"]
    prof = json.load(open(PROF, encoding="utf-8"))
    out = {"task": "R18 DA-04: one-loop finite-volume matching vs the gauge zero-point confinement (D112 2D_ex model)",
           "inputs": {"potential": "figures/data/higgs/generated/potential_2Dex_lam010_013.csv",
                      "resonance": "higgs/05_fulldim_validation/expected/r3/fulldim_dmc/run01/T14a_profile.json "
                                   "#/<lambda>/Q_2D_ex_siegert"},
           "lambda": {}}
    for tag, key in (("m0p010", "-0.01"), ("m0p013", "-0.013")):
        zp, dV = col["zeropoint_" + tag], col["dV_" + tag]
        dzp, ddV = np.gradient(zp, r), np.gradient(dV, r)
        sl = {}
        for rr in (1.0, 2.0, 3.0, 4.0, 5.0):
            k = int(np.argmin(np.abs(r - rr)))
            sl["%.1f" % rr] = {"r": float(r[k]), "dzeropoint_dr": float(dzp[k]), "ddV_dr": float(ddV[k]),
                               "slope_reduction": float(-ddV[k] / dzp[k])}
        Q = prof[key]["Q_2D_ex_siegert"]
        bar = {}
        for name in ("Uad", "Ug"):
            U = col["%s_%s" % (name, tag)]
            U0 = U - dV
            m = r > 1.5
            i, i0 = int(np.argmax(np.where(m, U, -1e300))), int(np.argmax(np.where(m, U0, -1e300)))
            bar[name] = {"with_matching": {"r_top": float(r[i]), "U_top": float(U[i]), "height_above_Q": float(U[i] - Q)},
                         "without_matching": {"r_top": float(r[i0]), "U_top": float(U0[i0]),
                                              "height_above_Q": float(U0[i0] - Q)},
                         "ratio_with_to_without": float((U[i] - Q) / (U0[i0] - Q))}
        red = [v["slope_reduction"] for v in sl.values()]
        out["lambda"][key] = {"Q_2D_ex_siegert": Q, "slope": sl, "slope_reduction_range_r1_5": [min(red), max(red)],
                              "barrier_above_Q": bar,
                              "U_top_archived_T14a": prof[key]["2D_ex"]["U_top"], "r_top_archived_T14a": prof[key]["2D_ex"]["r_top"]}
    allred = [x for v in out["lambda"].values() for x in v["slope_reduction_range_r1_5"]]
    out["headline"] = {"slope_reduction_range_r1_5": [min(allred), max(allred)],
                       "barrier_above_Q_Uad": {k: [v["barrier_above_Q"]["Uad"]["without_matching"]["height_above_Q"],
                                                   v["barrier_above_Q"]["Uad"]["with_matching"]["height_above_Q"]]
                                               for k, v in out["lambda"].items()},
                       "barrier_above_Q_Ug": {k: [v["barrier_above_Q"]["Ug"]["without_matching"]["height_above_Q"],
                                                  v["barrier_above_Q"]["Ug"]["with_matching"]["height_above_Q"]]
                                              for k, v in out["lambda"].items()},
                       "note": "barrier pairs are [without, with] the one-loop matching, in units 1/L above the "
                               "resonance level Q_2D_ex_siegert; Uad = potential of the radial problem (incl. 3/(8 r^2)), "
                               "Ug = field-space potential (without it)"}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out["headline"], indent=1))
    for k, v in out["lambda"].items():
        print(k, {rr: round(s["slope_reduction"], 3) for rr, s in v["slope"].items()})


if __name__ == "__main__":
    main()
