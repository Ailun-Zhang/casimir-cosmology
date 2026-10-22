# -*- coding: utf-8 -*-
"""C170 S3: coupling scans of the static box A threshold (run05 final model, N = 300, P = 0.9/0.5/0.1).

One of g2, gY, y_t is multiplied by s = 0.7, 0.85, 1.15, 1.3 in ALL sectors (zero-mode Gaussian profile, one-loop
matching, cross terms incl. the light-fermion twist and the SU(3)-top variational term through m_t = y_t r/sqrt 2);
g3 and lambda are not scaled (lambda is the scanned variable).  lambda grid x = |lambda|^(-1/2) = 7.0 ... 12.5;
two continuation chains per variant from x = 9.0.
Usage: python -B c170_s3.py run [nw] [order]   (order 'first': s = 0.85, 1.15 for all three, then 0.7, 1.3)
"""
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c170_core as CC  # noqa: E402

SCALES_FIRST = [0.85, 1.15]
SCALES_SECOND = [0.7, 1.3]
COUPS = ["g2", "gY", "yt"]
XS = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
X0 = 9.0
CHAINS = os.path.join(CC.OUT, "S3", "chains")


def specs(scales):
    c = CC.couplings_at(172.6, 0.1180, CC.mu_node(0.0))
    out = []
    lam = [CC.lam_of_x(x) for x in XS]
    l0 = CC.lam_of_x(X0)
    up = [l0] + sorted([l for l in lam if l > l0 + 1e-15])
    dn = [l0] + sorted([l for l in lam if l < l0 - 1e-15], reverse=True)
    for s in scales:
        for g in COUPS:
            base = dict(model="final", g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"])
            base[g] = c[g] * s
            base["scan"] = dict(coupling=g, s=s)
            nm = "S3_%s_s%.2f" % (g, s)
            for tag, ls in (("up", up), ("down", dn)):
                out.append(dict(name="%s_%s" % (nm, tag), base=base, lams=ls, stop_after_fail=2,
                                prev0=dict(dQ=0.0041, rho=1.012), outdir=CHAINS))
    return out


def run(nw=12, which="all"):
    t0 = time.time()
    sc = SCALES_FIRST + SCALES_SECOND if which == "all" else (SCALES_FIRST if which == "first" else SCALES_SECOND)
    sp = specs(sc)
    CC.status_update({"stage": "S3 running (%d chains)" % len(sp),
                      "S3": {"state": "running", "scales": sc,
                             "resume": "python -B c170_s3.py run %d %s  (chains resume from S3/chains)" % (nw, which)}})
    with ProcessPoolExecutor(max_workers=max(1, min(nw, 14, len(sp)))) as ex:
        res = list(ex.map(CC.run_chain, sp))
    summ = {r["name"]: [(round(p["pt"]["lam"], 6), p.get("status"), p.get("gamma")) for p in r["points"]] for r in res}
    CC.status_update({"S3": {"state": "chains done (%s)" % which, "time_s": time.time() - t0}})
    print(json.dumps(CC.clean(summ), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "run"
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    if md == "run":
        run(nw, which)
