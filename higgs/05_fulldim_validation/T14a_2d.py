# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_2d.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Same-wall 2D_ex / 2D_G finite-difference ground states (Richardson) for the DMC comparison.  Paper: PRD H2.

T14a step 2: same-wall 2D (D112 operator) ground energies Q_2D(r_w) by sparse FD + Richardson (h=0.08/0.04/0.02).
    python -B T14a_2d.py            -> run01/T14a_2d_wall.json   (incremental; resumable per (var, lam, rw))
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import time
import numpy as np
import T14a_common as T

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(T.RES, "T14a_2d_wall.json")
HS = (0.08, 0.04, 0.02)


def cases():
    cs = [(0.0, 8.96, 12.0)]
    cs += [(-0.010, rw, 12.0) for rw in (4.80, 5.12, 5.44, 4.32, 4.08, 5.76, 6.08)]
    cs += [(-0.013, rw, 12.0) for rw in (4.40, 4.64, 4.96, 4.08, 3.84, 5.28)]
    return cs


def one(var, lam, rw, Sw, tilt=0.0):
    old = T.jload(T.OLDSUM)
    q0 = T.Q2D_LAM0 if lam == 0.0 else old["Q"]["2D_ex" if var != "2D_G" else "2D_G"][f"{lam}"]
    e, extra = [], {}
    for h in HS:
        t0 = time.time()
        vals, vecs, r, s = T.fd2d_wall(var, lam, rw, h, Sw=Sw, sigma=q0 - 0.03, k=3, tilt=tilt)
        p2 = vecs[:, 0].reshape(len(r), len(s)) ** 2
        rho = p2.sum(1)
        e.append(float(vals[0]))
        extra[f"{h}"] = dict(levels=vals.tolist(), P_r_gt_3=float(rho[r > 3.0].sum() / rho.sum()),
                             P_s_gt_0p75Sw=float(p2[:, s > 0.75 * Sw].sum() / p2.sum()),
                             mean_r=float((rho * r).sum() / rho.sum()), time_s=round(time.time() - t0, 1))
    rich, rich2 = T.richardson(e)
    return dict(var=var, lam=lam, rw=rw, Sw=Sw, tilt=tilt, e_h=e, Q=rich, Q_rich_h2=rich2, diag=extra, ref_siegert_or_numerov=q0)


def main():
    res = T.jload(OUT) if os.path.exists(OUT) else {}
    for lam, rw, Sw in cases():
        for var in ("2D_ex", "2D_G"):
            key = f"{var}|{lam}|{rw:.2f}"
            if key in res:
                continue
            o = one(var, lam, rw, Sw)
            res[key] = o
            T.wjson(OUT, res)
            print(key, "Q=%.6f (h2: %.6f) ref=%.6f" % (o["Q"], o["Q_rich_h2"], o["ref_siegert_or_numerov"]),
                  "e_h", ["%.6f" % x for x in o["e_h"]], "Pr>3=%.2e" % o["diag"]["0.02"]["P_r_gt_3"], flush=True)
    for lam, rw in ((0.0, 8.96), (-0.010, 5.12), (-0.013, 4.64)):
        for tl in (1.0, -1.0):
            key = f"2D_ex|{lam}|{rw:.2f}|t{tl:g}"
            if key in res:
                continue
            o = one("2D_ex", lam, rw, 12.0, tilt=tl)
            res[key] = o
            T.wjson(OUT, res)
            print(key, "Q=%.6f" % o["Q"], flush=True)


if __name__ == "__main__":
    main()
