# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/realtime_2d/hr_post.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations/code/20261013_claude_v12_r2/higgs_final/hr_post.py on 2026-10-14 by realtime-2d (C157/D129); code unchanged unless a later '# R2D:' comment says otherwise.
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_realtime/hr_post.py on 2026-10-13 by agent higgs-final (C144/D116).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] D111 post-processing (xi_c extraction).  Paper: PRD App. H3.

C139 post-processing: Part-1 measure table, Part-2 P_surv table, xi_c (P(xi) = 0.99 P(1/6)) with errors,
late-time gauge-binding check, inflationary stochastic numbers.  Writes run01/C139_summary.json."""
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
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hr_core as C  # noqa: E402
from hr_potential import PotentialTable  # noqa: E402

ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r2/higgs_realtime/run01"))
P2 = os.path.join(RES, "part2")
MTS = [172.0, 172.2, 172.6]
USW = [5.0, 5.5, 6.0, 6.5]
X16 = round(1.0 / 6, 6)


def load(tag):
    D = {}
    for fn in glob.glob(os.path.join(P2, "p2_%s_*.json" % tag)):
        with open(fn, encoding="utf-8") as f:
            d = json.load(f)
        D[(d["mt"], round(d["xi"], 6))] = d
    return D


def val(d, u, key="P_surv"):
    for h in d["hist"]:
        if abs(h["u"] - u) < 1e-9 and key in h:
            return h[key]
    return None


def xi_c(P, frac=0.99):
    xs = sorted(P)
    R = [P[x] / P[X16] for x in xs]
    js = [j for j in range(len(xs)) if R[j] < frac and xs[j] < X16]
    if not js:
        return dict(xi_c=None, note="all grid xi satisfy; xi_c <= %.3f" % xs[0])
    j = max(js)
    x1, x2, r1, r2 = xs[j], xs[j + 1], R[j], R[j + 1]
    lin = x1 + (frac - r1) * (x2 - x1) / (r2 - r1)
    d1, d2, dm = -math.log(r1), -math.log(min(r2, 0.999999999)), -math.log(frac)
    try:
        e1, e2 = 1 - 6 * x1, 1 - 6 * x2
        if e2 <= 1e-9 or d2 <= 0:
            raise ValueError
        le = math.log(e1) + (math.log(dm) - math.log(d1)) * (math.log(e2) - math.log(e1)) / (math.log(d2) - math.log(d1))
        lg = (1 - math.exp(le)) / 6
    except (ValueError, ZeroDivisionError):
        lg = lin
    return dict(xi_c=lin, xi_c_loglog=lg, bracket=[x1, x2], R_bracket=[r1, r2])


def part1():
    with open(os.path.join(RES, "part1_results.json"), encoding="utf-8") as f:
        d = json.load(f)
    S = d["summary"]
    tab = {}
    for k, s in S.items():
        tab[k] = dict(w0=s["w0"], P_in0=s["P_in0"], P_in_eta5=s["early_P_in"].get("eta=5"),
                      P_in_eta20=s["early_P_in"].get("eta=20"), ratio_u5=s["nodes"]["u=5"]["ratio"],
                      cond_ratio_u5=s["nodes"]["u=5"]["ratio"] / s["w0"],
                      w_over_haz_max_dev=max(abs(v["w_over_haz"] - 1) for v in s["nodes"].values()),
                      P_inf=s["P_inf"], P_haz_inf=s["P_haz_inf"])
    gc = {}
    for r in d["runs"]:
        if r["tag"] == "base" and r.get("gamma_cmp"):
            rel = [abs(g["gamma_cap"] / g["gamma_siegert"] - 1) for g in r["gamma_cmp"]]
            dq = [abs(g["Q_cap"] - g["Q_siegert"]) for g in r["gamma_cmp"]]
            gc["%.1f" % r["mt"]] = dict(max_rel_gamma=max(rel), max_dQ=max(dq), n=len(rel))
    return dict(states=tab, gamma_cap_vs_siegert=gc)


def part2():
    M = load("main")
    out = dict(P={}, xi_c={}, norms={})
    xis = sorted(set(k[1] for k in M))
    for mt in MTS:
        for key in ("P_surv", "P_surv_rg", "P_surv_mad"):
            for u in USW:
                P = {x: val(M[(mt, x)], u, key) for x in xis if (mt, x) in M}
                if any(v is None for v in P.values()) or X16 not in P:
                    continue
                out["P"]["%s|mt=%.1f|u_sw=%.1f" % (key, mt, u)] = {"%.4f" % x: P[x] for x in sorted(P)}
                out["xi_c"]["%s|mt=%.1f|u_sw=%.1f" % (key, mt, u)] = xi_c(P)
        for x in xis:
            if (mt, x) in M:
                h = M[(mt, x)]["hist"]
                out["norms"]["mt=%.1f|xi=%.4f" % (mt, x)] = {"%.2f" % e["u"]: [round(e["norm"], 5),
                                                             round(e.get("P_top") or -1, 5), e.get("P_run_cl")]
                                                             for e in h}
    cen = {}
    for mt in MTS:
        base = out["xi_c"].get("P_surv|mt=%.1f|u_sw=6.0" % mt, {}).get("xi_c")
        if base is None:
            continue
        alt = []
        for key in ("P_surv", "P_surv_rg", "P_surv_mad"):
            for u in (5.5, 6.0, 6.5):
                c = out["xi_c"].get("%s|mt=%.1f|u_sw=%.1f" % (key, mt, u), {})
                for kk in ("xi_c", "xi_c_loglog"):
                    if c.get(kk) is not None:
                        alt.append((key, u, kk, c[kk]))
        cen["%.1f" % mt] = dict(central=base, spread=[min(a[3] for a in alt), max(a[3] for a in alt)],
                                u_sw5=out["xi_c"].get("P_surv|mt=%.1f|u_sw=5.0" % mt, {}).get("xi_c"), alts=alt)
    out["xi_c_central"] = cen
    sens = {}
    sub = [0.04, 0.08, 0.12, X16]
    for tag in ("capy4", "capy25fac2", "fine_y", "fine_r", "wig480", "um4.0", "to7"):
        V = load(tag)
        if not V:
            continue
        row = {}
        for u in USW + [7.0]:
            P = {x: val(V[(172.2, x)], u) for x in sub if (172.2, x) in V}
            Pm = {x: val(M[(172.2, x)], u) for x in sub if (172.2, x) in M}
            if X16 not in P or any(v is None for v in P.values()):
                continue
            row["u_sw=%.1f" % u] = dict(P={"%.4f" % x: P[x] for x in P},
                                        rel_vs_main={"%.4f" % x: (P[x] / Pm[x] - 1) if Pm.get(x) else None
                                                     for x in P},
                                        xi_c=xi_c(P).get("xi_c"),
                                        xi_c_main_subgrid=xi_c(Pm).get("xi_c") if X16 in Pm and None not in
                                        Pm.values() else None)
        sens[tag] = row
    out["sens_mt172.2"] = sens
    return out


def checks():
    tab = PotentialTable(os.path.join(RES, "tables", "U_table_MU_mt172.2.npz"))
    gb = {}
    for u in (5.0, 5.5, 6.0, 6.5, 7.0):
        hl = float(C.hL(u))
        M, lam = hl ** 3, tab.lam(u)
        row = {}
        for xi in (0.04, 0.10, 1.0 / 6):
            yb = math.sqrt(12 * xi / abs(lam))
            for y in (5.0, yb):
                r = hl * y
                Fnq = float(tab.dU_on(np.array([r]), u)[0] - lam * r ** 3)
                row["xi=%.3f,y=%.2f" % (xi, y)] = dict(r=r, ratio_nonquartic_to_xiR=abs(Fnq) / M / (12 * xi * y),
                                                       ratio_to_quartic=abs(Fnq) / M / (abs(lam) * y ** 3))
        gb["u=%.1f" % u] = dict(hL=hl, M=M, lam=lam, rows=row)
    lamL = tab.lam(7.0)
    st = {}
    for xi in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 1.0 / 6):
        B = 192 * math.pi ** 2 * xi ** 2 / abs(lamL)
        st["%.4f" % xi] = dict(sigma_eq_over_H=1 / math.sqrt(32 * math.pi ** 2 * xi), y_b=math.sqrt(12 * xi / abs(lamL)),
                               B=B, B_minus_3N60=B - 180.0)
    xst = math.sqrt(180 * abs(lamL) / (192 * math.pi ** 2))
    return dict(gauge_binding_late=gb, stochastic=dict(lam_late=lamL, table=st, xi_needed_B_gt_3N_N60=xst,
                                                        H_over_2pi_in_y=1 / (2 * math.pi)))


if __name__ == "__main__":
    out = dict(part1=part1(), part2=part2(), checks=checks())
    with open(os.path.join(RES, "C139_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    p1 = out["part1"]
    print("PART1 gamma CAP vs Siegert:", json.dumps(p1["gamma_cap_vs_siegert"]))
    for k, v in p1["states"].items():
        print("%-42s w0=%.4f Pin0=%.4f Pin(eta5)=%.4f ratio_u5=%.4f cond=%.4f wdev=%.1e Pinf=%.4f Phaz_inf=%.4f" % (
            k, v["w0"], v["P_in0"], v["P_in_eta5"], v["ratio_u5"], v["cond_ratio_u5"], v["w_over_haz_max_dev"],
            v["P_inf"], v["P_haz_inf"]))
    p2 = out["part2"]
    for k, v in p2["P"].items():
        print(k, " ".join("%s:%.4f" % (x, y) for x, y in v.items()))
    for k, v in p2["xi_c"].items():
        print("xi_c", k, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}))
    for k, v in p2["xi_c_central"].items():
        print("CENTRAL mt=%s xi_c=%.4f spread=[%.4f,%.4f] u_sw5=%s" % (k, v["central"], v["spread"][0], v["spread"][1],
                                                                       v["u_sw5"]))
    for k, v in p2["sens_mt172.2"].items():
        print("SENS", k, json.dumps({a: dict(xi_c=b["xi_c"], main=b["xi_c_main_subgrid"],
                                             rel={x: (round(r, 4) if r is not None else None)
                                                  for x, r in b["rel_vs_main"].items()}) for a, b in v.items()}))
    print("CHECKS", json.dumps(out["checks"], indent=0)[:3000])
