# PROVENANCE: new check written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""End-to-end quick check of the torus KL pipeline (transfer table -> covariance builder -> exact Gaussian KL).

'C75_produce.py validate' builds, in this work tree and from scratch, the T+E covariance (l <= 30, x_max = 160) of the
free-field fossil (sigma_E = 0.0589, beta = sqrt2 sigma_E X_d) and of the cube of equal volume at L0 = 1.07 d_LSS.
This script evaluates the exact KL divergence D (T+E and T only) with the production routines C75_common.load/find
and C75_analyze.D and writes checks/kl_quick.json; run.py compares it with the archived
C75_results.json -> validation/fresh_vs_cached/s1.07_free_fossil/D_*_fresh_fresh.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C75_common as CC   # noqa: E402  (production)
import C75_analyze as CA  # noqa: E402  (production: D = exact KL, eigenvalue method)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    Cf, mf = CC.load(CC.find("defm", 1.07, sig=CC.SIGMAS["free"], own_only=True))
    Cc, mc = CC.load(CC.find("base", 1.07, own_only=True))
    out = {"s": 1.07, "sigma_E": CC.SIGMAS["free"], "amp": CC.amp_of(mf), "modes_half_used_fossil": CC.modes_of(mf),
           "modes_half_used_cube": CC.modes_of(mc)}
    for fl in ("TE", "T"):
        out["D_%s_fresh_fresh" % fl] = CA.D(Cf, Cc, fl)["D"]
    p = CC.WP.wfile("checks/kl_quick.json")
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
