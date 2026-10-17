# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/xi_regress.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new script by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.
"""[v12 package, 2026-10-15] Regression of the node evaluation: with C161_CROSS=run02 the D129 run01 node (172.60, u = 0) is reproduced; the same node with the run05 cross terms (difference printed).

Regression of the nm_model copy: with C161_CROSS=run02 the copied node evaluation must reproduce the D129 run01 node
file (realtime_2d/run01/nodes/node_mt172.60_u0.0000.npz); then the same node with the run05 cross terms (difference
printed).  Run: python -B xi_regress.py"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json

os.environ["C161_CROSS"] = "run02"
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import nm_model as NM  # noqa: E402

ROOT = NM.ROOT
ref = dict(np.load(_zp.w("r3/realtime_2d/run01/nodes/node_mt172.60_u0.0000.npz")))
r = ref["r"]
out = {}
for tag in ("run02", "run05"):
    os.environ["C161_CROSS"] = tag
    d = NM.node_eval(172.6, NM.ALS, 0.0, r)
    Wl = NM.V_light_c(r, NM.SG48.s, d["beta"], d["c"]["g2"], d["c"]["gY"])
    ev = NM.eps_levels(d["OM"], Wl, NM.SG48, k=2)
    m = (r > 0.3) & (r < 12.0)
    out[tag] = dict(max_abs_dU1=float(np.max(np.abs(d["U1"] - ref["U1"])[m])), max_abs_dOM=float(np.max(np.abs(d["OM"] - ref["OM"])[m])),
                    max_abs_deps0=float(np.max(np.abs(ev[:, 0] - ref["eps"][:, 0])[m])),
                    dUBO_at_r=dict((("%.1f" % x), float((d["U1"] + ev[:, 0] - ref["U1"] - ref["eps"][:, 0])[int(round(x / NM.DR_TAB)) - 1]))
                                   for x in (1.0, 2.0, 2.3, 3.0, 4.0, 5.0)))
    print(tag, json.dumps(out[tag]), flush=True)
os.makedirs(NM.RES, exist_ok=True)
json.dump(out, open(os.path.join(NM.RES, "xi_regress.json"), "w", encoding="utf-8"), indent=1)
