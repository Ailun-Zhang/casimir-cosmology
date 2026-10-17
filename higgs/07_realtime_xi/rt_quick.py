# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/rt_quick.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations\code\20261014_claude_v12_r3\realtime_2d\rt_quick.py on 2026-10-15 by claims-run05 (C161/D132); 4 edits (see make_xi_copies.py)
# PROVENANCE: new driver by agent realtime-2d (C157/D129) on 2026-10-14, modelled on
# 04_calculations/code/20261013_claude_v12_r2/higgs_final/C144_rt.py (read-only); hr_core/hr_part2/hr_post/hr_potential
# are the unchanged copies in this directory (provenance line added).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Real-time survival through the turn on the node tables (D111 lens transform, u_sw = 6 Wigner hand-over): P_surv(xi_H) for m_t = 172.2/172.6/172.95/172.96/173.0; 'post' -> xi_c with P_surv(xi_c) = 0.99 P_surv(1/6) (C157_xi_quick.json).  Paper: PRD App. H3; PRL xi condition.

C157 step 1 (critical path): xi_c in the round-3 central model with the D111/D116 real-time pipeline (unchanged).
Only differences to C144_rt: (i) tables = nm_model U_eff (new central model, node-calibrated), (ii) file names carry m_t
with two decimals (172.95), (iii) output to realtime_2d/run01/part2.  Same u = 0 pole seed rule as C144_rt.
Definition (D111/D116): xi_c from P_surv(xi) = 0.99 P_surv(1/6); central u_sw = 6.0; lambda(phi) [P_surv_rg] and
lambda(1/L) [P_surv; lambda frozen for u > 7, D111/R15 addendum] late-time prescriptions; merged = midpoint +- half diff.
Usage: python -B rt_quick.py run [nw] [set] | post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import time
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _zp.ZMLIB)
sys.path.insert(0, HERE)
import hr_core as HC  # noqa: E402
import hr_part2 as P2M  # noqa: E402
import hr_post as HP  # noqa: E402

ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("w/claims_run05/run01/xi_run05"))
TDIR = os.path.join(RES, "tables")
PDIR = os.path.join(RES, "part2")
X16 = round(1.0 / 6, 6)
XIS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 1.0 / 6]
REFINE = {172.0: [0.01, 0.03], 172.2: [0.03, 0.05], 172.6: [0.07, 0.09], 172.95: [0.11, 0.13], 172.96: [0.11, 0.13], 173.0: [0.11, 0.13]}
MAIN = [172.2, 172.6, 172.95, 172.96, 173.0]
R2 = {"172.2": dict(xi_c=0.036, err=0.003, P_surv_16=0.9842), "172.6": dict(xi_c=0.072, err=0.003, P_surv_16=0.8567),
      "173.0": dict(xi_c=0.120, err=0.001, P_surv_16=0.5137)}
TABLE_KIND = os.environ.get("C157_TABLE", "eff")


def tpath(mt):
    return os.path.join(TDIR, "U_table_%s_mt%.2f.npz" % (TABLE_KIND, mt))


def fname(mt, xi, tag):
    return os.path.join(PDIR, "p2_%s_mt%.2f_xi%.4f.json" % (tag, mt, xi))


P2M.RES, P2M.P2, P2M.tpath, P2M.fname = RES, PDIR, tpath, fname
_orig_resonance = HC.resonance


def _resonance(g, V, sigma, k=1, inner=None):          # C144_rt seed rule (ground level of the well walled at the top)
    if inner is not None:
        n = int(inner)
        sub = HC.Grid(g.dx, n * g.dx)
        Hs = HC.h_sparse(sub, 0.5, np.ascontiguousarray(np.real(V[:n]))).toarray().real
        sigma = complex(float(np.linalg.eigvalsh(Hs)[0]), -0.0005)
    return _orig_resonance(g, V, sigma, k=k, inner=inner)


HC.resonance = _resonance


def _run(args):
    return P2M.run_one(args)


def jobs(which):
    mts = MAIN if which == "main" else ([172.0] if which == "mt172" else MAIN + [172.0])
    J = []
    for xi in XIS + [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]:
        for mt in mts:
            if xi in XIS or xi in REFINE[mt]:
                J.append((mt, xi, {}, "main"))
    return J


def run(nw=4, which="main"):
    os.makedirs(PDIR, exist_ok=True)
    t0 = time.time()
    J = jobs(which)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for o in ex.map(_run, J):
            sw = {"%.1f" % h["u"]: round(h.get("P_surv", float("nan")), 5) for h in o["hist"] if "P_surv" in h}
            print("rt mt=%.2f xi=%.4f E0=%.5f%+.3e P_surv=%s t=%.0fs" % (o["mt"], o["xi"], o["E0"][0], o["E0"][1], sw,
                                                                      o["time_s"]), flush=True)
    print("rt runs done %.0fs (%d jobs)" % (time.time() - t0, len(J)), flush=True)


def post(out_name="C157_xi_quick.json", tag="main"):
    D = {}
    for fn in glob.glob(os.path.join(PDIR, "p2_%s_*.json" % tag)):
        with open(fn, encoding="utf-8") as f:
            d = json.load(f)
        D[(round(d["mt"], 2), round(d["xi"], 6))] = d
    try:
        with open(os.path.join(RES, "C157_calib.json"), encoding="utf-8") as f:
            cal = json.load(f)
    except Exception:  # noqa: BLE001
        cal = None
    xs = sorted(set(k[1] for k in D))
    out = dict(definition="xi_c: P_surv(xi) = 0.99 P_surv(1/6) (D111/D116); u_sw = 6.0 central; lambda(phi) = P_surv_rg, "
                          "lambda(1/L) = P_surv (lambda frozen at u > 7 as in D111/D116, R15 addendum); merged = midpoint "
                          "+- half difference; spread_all = u_sw in {5.5, 6.0, 6.5} x both prescriptions x lin/loglog",
               model="round-3 central: C149 run02 (2D_ex direct + T1 NNLO central + cross terms), 1D U_BO + node-"
                     "calibrated bump (nm_model.py)", table_kind=TABLE_KIND, P={}, xi_c={}, central={}, round2_D116=R2)
    for mt in sorted(set(k[0] for k in D)):
        for key in ("P_surv", "P_surv_rg", "P_surv_mad"):
            for u in (5.0, 5.5, 6.0, 6.5):
                P = {x: HP.val(D[(mt, x)], u, key) for x in xs if (mt, x) in D}
                if not P or any(v is None for v in P.values()) or X16 not in P:
                    continue
                kk = "%s|mt=%.2f|u_sw=%.1f" % (key, mt, u)
                out["P"][kk] = {"%.4f" % x: P[x] for x in sorted(P)}
                out["xi_c"][kk] = HP.xi_c(P)
        tag_ = "mt=%.2f" % mt
        c_phi = out["xi_c"].get("P_surv_rg|%s|u_sw=6.0" % tag_, {}).get("xi_c")
        c_1L = out["xi_c"].get("P_surv|%s|u_sw=6.0" % tag_, {}).get("xi_c")
        alts = [v.get(z) for kk, v in out["xi_c"].items() if tag_ in kk and "u_sw=5.0" not in kk and "mad" not in kk
                for z in ("xi_c", "xi_c_loglog") if v.get(z) is not None]
        cen = dict(lam_phi=c_phi, lam_1L=c_1L, spread_all=[min(alts), max(alts)] if alts else None,
                   u_sw5_lam_phi=out["xi_c"].get("P_surv_rg|%s|u_sw=5.0" % tag_, {}).get("xi_c"),
                   madelung_lam_1L=out["xi_c"].get("P_surv_mad|%s|u_sw=6.0" % tag_, {}).get("xi_c"))
        if c_phi is not None and c_1L is not None:
            cen.update(merged=0.5 * (c_phi + c_1L), half_diff=0.5 * abs(c_phi - c_1L))
        if (mt, X16) in D:
            d16 = D[(mt, X16)]
            cen["E0_u0"] = d16["E0"]
            cen["P_surv_16_u6"] = HP.val(d16, 6.0, "P_surv")
            cen["P_surv_16_u6_rg"] = HP.val(d16, 6.0, "P_surv_rg")
            if cal:
                cen["P1_hazard_2D_C149run02"] = cal["P1_check"].get("%.2f" % mt, {}).get("P1_2D")
                if cen["P1_hazard_2D_C149run02"]:
                    cen["P_surv16_over_P1_minus1"] = cen["P_surv_16_u6"] / cen["P1_hazard_2D_C149run02"] - 1
        if (mt, 0.0) in D:
            cen["P_surv_xi0_u6"] = dict(lam_1L=HP.val(D[(mt, 0.0)], 6.0, "P_surv"),
                                        lam_phi=HP.val(D[(mt, 0.0)], 6.0, "P_surv_rg"),
                                        madelung=HP.val(D[(mt, 0.0)], 6.0, "P_surv_mad"))
        r2 = R2.get("%.1f" % mt) if abs(mt * 10 - round(mt * 10)) < 1e-9 else None
        if r2 and cen.get("merged") is not None:
            cen["delta_vs_round2"] = cen["merged"] - r2["xi_c"]
        out["central"]["%.2f" % mt] = cen
    out["n_runs"] = len(D)
    p = os.path.join(RES, out_name)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)
    for k, v in out["P"].items():
        if "u_sw=6.0" in k:
            print(k, " ".join("%s:%.4f" % (x, y) for x, y in v.items()))
    print(json.dumps(out["central"], indent=1))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "post"
    if md == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 4, sys.argv[3] if len(sys.argv) > 3 else "main")
    elif md == "post":
        post(*(sys.argv[2:4]))
