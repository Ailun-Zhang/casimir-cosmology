# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/realtime_2d/hr_part2.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations/code/20261013_claude_v12_r2/higgs_final/hr_part2.py on 2026-10-14 by realtime-2d (C157/D129); code unchanged unless a later '# R2D:' comment says otherwise.
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_realtime/hr_part2.py on 2026-10-13 by agent higgs-final (C144/D116).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] D111 part 2 drivers (late-turn real-time runs, xi scans) used by realtime_2d.  Paper: PRD App. H3.

C139 Part 2: late-turn fate of the Higgs zero mode for xi_H < 1/6 (replaces the 'two conventions' of run03_xi).
u in [0, u_m]: r-frame CN from the real Gamow state (a);  u in [u_m, max u_sw]: exact lens transform to the y-frame
(y = phi/h, physical frame, time u = ht, mass M = (hL)^3), CN;  at u_sw: p-cell Wigner weights -> cosmic-time
classical evolution (Hubble friction 3 tanh(2u), xi R mass 12 xi h^2, U_nc force with lambda(1/L); variant RG:
lambda(sqrt(phi^2 + h^2 + L^-2))) -> runaway / bounded.  TDSE-absorbed flux = runaway (absorber onset >= 1.5 x
instantaneous barrier top).  P_surv = absolute (initial norm 1 at u = 0).
Usage: python -B hr_part2.py smoke|main|sens|post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hr_core as C  # noqa: E402
from hr_potential import PotentialTable  # noqa: E402

ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r2/higgs_realtime/run01"))
P2 = os.path.join(RES, "part2")
XIS = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 1.0 / 6]
MTS = [172.0, 172.2, 172.6]
USW = [5.0, 5.5, 6.0, 6.5]
BASE = dict(dr=0.01, rmax=16.0, r_c=10.0, cap0_r=20.0, deta=0.025, u_m=4.5, dy=0.003, ymax=64.0, fac=1.5,
            yfloor=10.0, cap0_y=10.0, du_max=0.002, cE=0.25, ny=320, npc=200, u_end=60.0, usw=USW, rg=True, mad=True)


def tpath(mt):
    return os.path.join(RES, "tables", "U_table_MU_mt%.1f.npz" % mt)


def fname(mt, xi, tag):
    return os.path.join(P2, "p2_%s_mt%.1f_xi%.4f.json" % (tag, mt, xi))


def quantiles(y, a2, qs=(0.5, 0.9, 0.99, 0.999)):
    c = np.cumsum(a2)
    c = c / c[-1]
    return {str(q): float(y[min(np.searchsorted(c, q), len(y) - 1)]) for q in qs}


def run_one(args):
    mt, xi, opt, tag = args
    fn = fname(mt, xi, tag)
    if os.path.exists(fn) and tag != "smoke":
        with open(fn, encoding="utf-8") as f:
            return json.load(f)
    t0 = time.time()
    o = dict(BASE, **opt)
    tab = PotentialTable(tpath(mt))
    gr = C.Grid(o["dr"], o["rmax"])
    fr = C.RFrame(tab, gr, xi, r_c=o["r_c"], cap0=o["cap0_r"])
    x, dx = gr.x, gr.dx
    lm = C.landmarks(x, fr.U_real(0.0), 4.4)
    E0, v0 = C.resonance(gr, fr.V(0.0), 4.4 - 0.0005j, k=8, inner=lm["i_top"])
    lm0 = C.landmarks(x, fr.U_real(0.0), E0.real)
    psi = np.real(v0).astype(complex)
    psi[x > lm0["r_exit"]] = 0
    psi /= math.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    hist = []

    def chk(u, p):
        try:
            rt = C.landmarks(x, fr.U_real(u), E0.real)["r_top"]
        except Exception:
            rt = None
        rec = dict(u=u, frame="r", norm=float(np.sum(np.abs(p[:fr.ic]) ** 2) * dx), r_top=rt)
        if rt:
            rec["P_top"] = float(np.sum(np.abs(p[:int(np.searchsorted(x, rt))]) ** 2) * dx)
        hist.append(rec)
    um = o["u_m"]
    psi = C.evolve_r(psi, fr, 0.0, um, o["deta"], E0.real, checks=[c for c in (1.0, 2.0, 3.0, 3.5, um) if c <= um],
                     on_check=chk)
    try:
        rtop_m = C.landmarks(x, fr.U_real(um), E0.real)["r_top"]
    except Exception:
        rtop_m = 3.8
    gy = C.Grid(o["dy"], o["ymax"])
    Phi = C.map_r_to_y(psi, gr, gy, um, r_cut=x[fr.ic])
    mp = dict(u_m=um, norm_r=float(np.sum(np.abs(psi[:fr.ic + 1]) ** 2) * dx),
              norm_y=float(np.sum(np.abs(Phi) ** 2) * gy.dx), r_top=rtop_m)
    yf = C.YFrame(tab, gy, xi, ytop0=rtop_m / float(C.hL(um)), fac=o["fac"], yfloor=o["yfloor"], cap0=o["cap0_y"])

    def chk_y(u, Ph):
        _, M, W = yf.V(u)
        a2 = np.abs(Ph) ** 2
        ic = yf.icap
        rec = dict(u=u, frame="y", M=M, hL=float(C.hL(u)), ytop=yf.ytop, ycap=yf.ycap,
                   norm=float(np.sum(a2[:ic]) * gy.dx), P_top=float(np.sum(a2[:yf.itop]) * gy.dx),
                   yq=quantiles(gy.x[:ic], a2[:ic]))
        if any(abs(u - s) < 1e-9 for s in o["usw"]):
            wc = C.wigner_cells(Ph, gy, yf.ycap, ny=o["ny"], npc=o["npc"])
            P = wc["P"]
            ii, kk = np.nonzero(np.abs(P) > 1e-12 * np.abs(P).max())
            y0, vv, w = wc["y"][ii], wc["p"][kk] / M, P[ii, kk]
            fate, tdec, info = C.classical_fates(y0, vv, u, xi, tab, u_end=o["u_end"])
            rec.update(W_sum=float(w.sum()), W_neg=float(w[w < 0].sum()), norm_in_taper=wc["norm_in"],
                       P_surv=float(w[fate != 2].sum()), P_und=float(w[fate == 0].sum()),
                       P_run_cl=float(w[fate == 2].sum()), cl=info, n_traj=int(len(w)), dyc=wc["dyc"],
                       p_range=[float(wc["p"][0]), float(wc["p"][-1])])
            if o["rg"]:
                f2, _, _ = C.classical_fates(y0, vv, u, xi, tab, u_end=o["u_end"], rg=True)
                rec["P_surv_rg"] = float(w[f2 != 2].sum())
                rec["P_und_rg"] = float(w[f2 == 0].sum())
            if o["mad"]:
                md = wc["mad"]
                f3, _, _ = C.classical_fates(md["y"], md["pm"] / M, u, xi, tab, u_end=o["u_end"])
                rec["P_surv_mad"] = float(md["w"][f3 != 2].sum())
        hist.append(rec)
    C.evolve_y(Phi, yf, um, max(o["usw"]), checks=sorted(set([4.5] + list(o["usw"]))), on_check=chk_y,
               du_max=o["du_max"], cE=o["cE"])
    out = dict(mt=mt, xi=xi, tag=tag, opt=o, E0=[E0.real, E0.imag], lm0=lm0, map=mp, hist=hist,
               time_s=time.time() - t0)
    os.makedirs(P2, exist_ok=True)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def jobs_main():
    return [(mt, xi, {}, "main") for xi in XIS for mt in MTS]


def jobs_sens():
    var = {"capy4": dict(cap0_y=4.0), "capy25fac2": dict(cap0_y=25.0, fac=2.0),
           "fine_y": dict(dy=0.002, du_max=0.001, cE=0.12), "fine_r": dict(dr=0.007, deta=0.0125),
           "wig480": dict(ny=480, npc=300), "um4.0": dict(u_m=4.0)}
    J = [(172.2, xi, dict(v, rg=False, mad=False), k) for k, v in var.items() for xi in (0.04, 0.08, 0.12, 1.0 / 6)]
    J += [(172.2, xi, dict(dy=0.0015, usw=[5.0, 5.5, 6.0, 6.5, 7.0], rg=False, mad=False), "to7")
          for xi in (0.08, 1.0 / 6)]
    return J


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    t0 = time.time()
    if mode == "smoke":
        o = dict(usw=[5.0, 6.0], ny=160, npc=120, rg=True, mad=True)
        res = [run_one((172.6, 0.04, o, "smoke")), run_one((172.6, 0.04, dict(o, u_m=4.0), "smoke"))]
        for r in res:
            print("u_m", r["opt"]["u_m"], "map", r["map"], "t=%.1f" % r["time_s"])
            for h in r["hist"]:
                print(json.dumps({k: h[k] for k in h if k not in ("cl",)}))
        sys.exit(0)
    if mode == "fine":   # refinement of the xi grid near the threshold (added after the main scan)
        jobs = [(172.0, x, {}, "main") for x in (0.01, 0.02, 0.03)] + \
               [(172.2, x, {}, "main") for x in (0.02, 0.03, 0.05)] + [(172.6, x, {}, "main") for x in (0.11, 0.13)]
    else:
        jobs = jobs_main() if mode == "main" else jobs_sens()
    with ProcessPoolExecutor(max_workers=5) as ex:
        for r in ex.map(run_one, jobs):
            sw = {"%.1f" % h["u"]: round(h.get("P_surv", float("nan")), 5) for h in r["hist"] if "P_surv" in h}
            print("%s mt=%.1f xi=%.4f %s t=%.0fs" % (r["tag"], r["mt"], r["xi"], sw, r["time_s"]), flush=True)
    print("done %.0f s" % (time.time() - t0))
