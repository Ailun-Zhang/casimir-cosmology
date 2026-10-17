# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/R3_mbar_newcentral.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] MS-bar top mass mbar(mbar) (four-loop QCD) of the critical pole masses; env R3_RUN/R3_SUMMARY/R3_OUT select the input summary (run05: R3_RUN=run05 R3_SUMMARY=C149_run05_summary.json R3_OUT=R3_mbar_run05.json).  Paper: PRL Table 1 (mbar_t form), PRD App. H5.

Main-session helper (2026-10-15 W3): MS-bar top mass mbar(mbar) (4-loop QCD, T1's qcd.py) at the
round-3 NEW central critical pole masses (C149 run02: 2D direct + NNLO + cross term).
Regression first: T1 run01 old central 172.580/173.074/173.575 -> 162.80/163.27/163.75 (D120)."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qcd  # noqa: E402

ROOT = _zp.WORK
crit = json.load(open(os.path.join(ROOT, "r3",
                                   "holonomy_cross", os.environ.get("R3_RUN", "run02"), os.environ.get("R3_SUMMARY", "C149_run02_summary.json")), encoding="utf-8"))["crit"]
out = {"regression_T1_run01": {}, "new_central": {}}
for Mt in (172.580, 173.074, 173.575):
    out["regression_T1_run01"][f"{Mt:.3f}"] = round(qcd.mbar_mbar(Mt, qcd.as6_at(Mt, 0.1180), N=4), 4)
for als in ("0.1171", "0.1180", "0.1189"):
    row = {}
    blk = crit[als]["with_cross"] if "with_cross" in crit[als] else crit[als]
    for lev, Mt in blk.items():
        row[lev] = {"pole": round(Mt, 4), "mbar_mbar_QCD4L": round(qcd.mbar_mbar(Mt, qcd.as6_at(Mt, float(als)), N=4), 4)}
    out["new_central"][als] = row
print(json.dumps(out, indent=1, ensure_ascii=False))
dst = os.path.join(ROOT, "r3", "sm_mapping", "run02",
                   os.environ.get("R3_OUT", "R3_mbar_newcentral.json"))
json.dump(out, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("written", dst)
