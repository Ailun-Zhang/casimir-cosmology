# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/run02_fits.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Compiled-SMDR refits (driver c148_fit.c) of the MS-bar couplings at Q = M_t with scaled evaluation scales (xi = 1/4 ... 4, per scale) -> r3/sm_mapping/run02/fits/fits.json.  Needs third_party/smdr built.  Paper: PRD App. H5.

C148 run02: compiled-SMDR refits with scaled evaluation scales (c148_fit.exe), <=4 parallel; -> run02/fits/fits.json"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r02  # noqa: E402

EXE = os.path.join(r02.SRCDIR, "c148_fit" + (".exe" if os.name == "nt" else ""))
OUT = os.path.join(r02.RES2, "fits")
CONFIGS = {"xi0.5": (0.5,) * 5, "xi2": (2,) * 5, "xi1": (1,) * 5,
           "t0.5": (0.5, 1, 1, 1, 1), "t2": (2, 1, 1, 1, 1), "h0.5": (1, 0.5, 1, 1, 1), "h2": (1, 2, 1, 1, 1),
           "Z0.5": (1, 1, 0.5, 1, 1), "Z2": (1, 1, 2, 1, 1), "G0.5": (1, 1, 1, 0.5, 1), "G2": (1, 1, 1, 2, 1),
           "d0.5": (1, 1, 1, 1, 0.5), "d2": (1, 1, 1, 1, 2), "xi0.25": (0.25,) * 5, "xi4": (4,) * 5}
MTS = [172.1, 172.6, 173.1, 173.6]
MH, AS, TOL = 125.13, 0.1180, 1e-10
ENV = dict(os.environ, PATH=os.environ.get("DNEC_MINGW_BIN", "") + os.pathsep + os.environ.get("PATH", ""))


def parse(fn):
    d = {}
    for ln in open(fn, encoding="utf-8"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "RG_mub":
            d["RG_mub"] = dict(mu=float(p[1]), **{p[i]: float(p[i + 1]) for i in range(2, len(p) - 1, 2)})
        else:
            d[p[0]] = float(p[1])
    return d


def one(job):
    cfg, Mt = job
    fn = os.path.join(OUT, "fit_%s_Mt%.2f.txt" % (cfg, Mt))
    if not os.path.exists(fn):
        x = CONFIGS[cfg]
        args = [EXE, "fit", "%.4f" % Mt, "%.4f" % MH, "%.5f" % AS] + ["%g" % v for v in x] + ["%.4f" % Mt, "%g" % TOL, fn]
        t0 = time.time()
        p = subprocess.run(args, cwd=r02.SRCDIR, env=ENV, capture_output=True, text=True, timeout=1200)
        open(fn + ".log", "w", encoding="utf-8").write(p.stdout + p.stderr + "\nrc=%d t=%.1f\n" % (p.returncode, time.time() - t0))
    return cfg, Mt, (parse(fn) if os.path.exists(fn) else None)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(c, m) for c in CONFIGS for m in (MTS if c.startswith("xi") else [172.6, 173.1])]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(os.environ.get("C148_FITW", "4"))) as ex:
        res = list(ex.map(one, jobs))
    out = dict(meta=dict(Mh=MH, alphaS=AS, tol=TOL, Mts=MTS, configs=CONFIGS,
                         scales="Qt=xit*Mt, Qh=xih*160, QZ=xiZ*160 (MZ,MW), QG=xiG*Mt, Qdec=xid*MZ; Qout=Mt"),
               fits={})
    for cfg, Mt, d in res:
        out["fits"].setdefault(cfg, {})["%.2f" % Mt] = d
    r02.U.wjson(os.path.join(OUT, "fits.json"), out)
    bad = [(c, m) for c, m, d in res if d is None]
    r02.status2({"stage": "fits done", "fits": dict(n=len(res), failed=bad, s=time.time() - t0),
                 "next_command": "python -B run02_post.py hazards  (provider variants sfit_*; hazards; post)"},
                done=["compiled-SMDR refits (%d)" % len(res)], files=["fits/fits.json"])
    print("fits done", len(res), "failed", bad, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
