# -*- coding: utf-8 -*-
"""C170 S2: gauge zero-mode strength kappa (box A couplings, 2D model without cross terms).

kappa multiplies g2 and gY only in the zero-mode Hamiltonian: the SU(2)xU(1) conditional Gaussian (kinetic, magnetic and
mass terms of the gauge zero modes -> C0, B_x, Om_eff^2, i.e. also the hypercharge oscillator and the Higgs couplings
(g2 x.T + gY Y w) Phi); the one-loop matching dV_MC(r) stays at the physical couplings.
Models: '2D_Gdmcs' (main) = conditional Gaussian + DMC shape transported with the exact zero-mode scaling
        dE_DMC(kappa; r) = kappa^(2/3) dE_DMC(1; kappa^(1/3) r)  (= production 2D_ex at kappa = 1);
        '2D_G' (bracket) = DMC shape dropped (strict variational model at every kappa).
Numerics: FAST for kappa >= 1; for kappa < 1 the s box is scaled with the zero-mode length kappa^(-1/3)
(N = 24 k^-1/3, Smax = 12 k^-1/3; box check 28/14 k^-1/3).
Usage: python -B c170_s2.py run [nw] | refine nw k1,k2,... | refine2 nw k1,k2,... | curves | kzero | scalecheck | reg2dex
"""
import os
import sys
import math
import time
import json
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c170_core as CC  # noqa: E402
import numpy as np  # noqa: E402

W, C = CC.W, CC.C
KAPPAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
MODELS = ["2D_Gdmcs", "2D_G"]
TARGET = math.log(2.0) / 300.0                       # Gamma for P = 1/2 at N = 300
UP = [-0.012, -0.010, -0.008, -0.006, -0.005, -0.004, -0.003, -0.002, -0.0015, -0.001, -0.0005, -0.0002, 0.0]
DOWN = [-0.012, -0.014, -0.016, -0.018, -0.020, -0.023, -0.026, -0.030, -0.035, -0.040]
CHAINS = os.path.join(CC.OUT, "S2", "chains")


def boxA():
    return CC.couplings_at(172.6, 0.1180, CC.mu_node(0.0))


def cfg_for(k):
    if k >= 1.0:
        return dict(CC.FAST), dict(CC.BOXCHK)
    f = k ** (-1.0 / 3.0)
    return (dict(CC.FAST, N=int(round(24 * f)), Smax=12.0 * f),
            dict(N=int(round(28 * f)), Smax=14.0 * f))


def chain_specs(kappas, models, up=UP, down=DOWN, tag=""):
    c = boxA()
    out = []
    for k in kappas:
        cfg, bxc = cfg_for(k)
        for m in models:
            base = dict(model=m, kappa=float(k), g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"])
            nm = "S2_%s_k%.4f%s" % (m, k, tag)
            out.append(dict(name=nm + "_up", base=base, lams=list(up), stop_after_fail=99, stop_gamma_below=TARGET / 30,
                            cfg=cfg, boxchk=bxc, prev0=dict(dQ=0.012, rho=1.07), outdir=CHAINS))
            out.append(dict(name=nm + "_down", base=base, lams=list(down), stop_after_fail=2,
                            stop_gamma_above=TARGET * 30, cfg=cfg, boxchk=bxc, prev0=dict(dQ=0.012, rho=1.07),
                            outdir=CHAINS))
    return out


def _prof(k):
    c = boxA()
    return CC.profile_parts(k * c["g2"], k * c["gY"])[1]


def run(nw=4, kappas=None, tag=""):
    t0 = time.time()
    ks = kappas or KAPPAS
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(ks)))) as ex:     # profiles first (cached on disk)
        pinf = list(ex.map(_prof, ks))
    sp = chain_specs(ks, MODELS, tag=tag)
    CC.status_update({"stage": "S2 running (%d chains)" % len(sp),
                      "S2": {"state": "running", "chains": [s["name"] for s in sp],
                             "resume": "python -B c170_s2.py run %d   (chains resume from S2/chains/*.json)" % nw}})
    with ProcessPoolExecutor(max_workers=max(1, min(nw, 14, len(sp)))) as ex:
        res = list(ex.map(CC.run_chain, sp))
    summ = {r["name"]: [(p["pt"]["lam"], p.get("status"), p.get("gamma")) for p in r["points"]] for r in res}
    CC.status_update({"S2": {"state": "chains done", "time_s": time.time() - t0, "profiles": pinf}})
    print(json.dumps(CC.clean(summ), indent=1))


# ------------------------------------------------------------------------------------------ potentials (figure)
def potential_parts(k, lam, r, model="2D_Gdmcs"):
    """effective radial potential of the 2D model (lowest s channel = 3/2 Om_eff for the quadratic channel) and parts."""
    c = boxA()
    cent = 3.0 / (8 * r * r)
    quart = lam * r ** 4 / 4
    dv = C.dV("MC", r, lam, c["g2"], c["gY"], c["yt"])
    if k == 0.0:
        zp = np.zeros_like(r)
        dmc = np.zeros_like(r)
    else:
        Pp, _ = CC.profile_parts(k * c["g2"], k * c["gY"])
        om = np.sqrt(np.maximum(Pp.om2(r), 0.0))
        zp = Pp.c0bx(r) + 1.5 * om                         # C0 + B_x + (3/2) Om_eff (gauge + hypercharge zero point)
        dmc = k ** (2.0 / 3.0) * CC.DMC(k ** (1.0 / 3.0) * r) if model == "2D_Gdmcs" else np.zeros_like(r)
    return dict(cent=cent, zp=zp, dmc=dmc, quart=quart, dV=dv, U_with=cent + zp + dmc + quart + dv,
                U_without=cent + zp + dmc + quart)


def curves():
    r = np.round(np.arange(0.05, 16.0001, 0.01), 4)
    rows, cols, info = [r], ["r"], {}
    for k in (0.0, 0.5, 1.0):
        d = potential_parts(k, -0.012, r)
        for nm in ("U_with", "U_without", "zp", "dmc", "dV", "quart", "cent"):
            rows.append(d[nm])
            cols.append("%s_k%.2f" % (nm, k))
        info["%.2f" % k] = dict(with_dV=CC.barrier_info(r, d["U_with"]), without_dV=CC.barrier_info(r, d["U_without"]))
    p = os.path.join(CC.OUT, "S2", "C170_S2_potential_lam-0.012.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("# C170 S2: effective radial potential U(r) of the 2D model at box A couplings, lambda = -0.012, units 1/L;"
                " U_with = cent + zp + dmc + quart + dV, U_without = same without the one-loop matching dV; "
                "zp = C0 + B_x + (3/2) Om_eff with zero-mode couplings kappa*(g2, gY); dmc = kappa^(2/3) dE_DMC(kappa^(1/3) r);"
                " kappa = 0: gauge zero modes decoupled (zp = dmc = 0)\n")
        f.write(",".join(cols) + "\n")
        for i in range(len(r)):
            f.write(",".join("%.8e" % c_[i] for c_ in rows) + "\n")
    # local slopes: zero-point (incl. DMC) vs matching, r = 1..5
    sl = {}
    for k in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        d = potential_parts(k, -0.012, r)
        z = d["zp"] + d["dmc"]
        dz, ddv = np.gradient(z, r), np.gradient(d["dV"], r)
        sl["%.2f" % k] = {"%.1f" % x: dict(dzp_dr=float(np.interp(x, r, dz)), ddV_dr=float(np.interp(x, r, ddv)),
                                           reduction=float(-np.interp(x, r, ddv) / np.interp(x, r, dz)))
                          for x in (1.0, 2.0, 3.0, 4.0, 5.0)}
    c = boxA()
    sig0 = 1.5 * c["g2"] + 0.75 * math.hypot(c["g2"], c["gY"])
    out = dict(barriers=info, slopes=sl, sigma0_boxA=sig0, file=os.path.relpath(p, CC.OUT))
    CC.wjson(os.path.join(CC.OUT, "S2", "C170_S2_curves.json"), out)
    print(json.dumps(CC.clean(out), indent=1)[:6000])


def kzero():
    """kappa = 0: gauge zero modes decoupled; Higgs radial potential U = 3/(8 r^2) + lam r^4/4 + dV_MC(r) (1D):
    is there a local maximum (barrier) for lam < 0?  Fine grid r = 0.02 ... 30."""
    c = boxA()
    r0 = np.round(np.arange(0.02, 16.0001, 0.002), 4)
    out = {}
    dv0 = C.dV("MC", r0, -1e-12, c["g2"], c["gY"], c["yt"])
    r = r0
    for lam in (-0.0001, -0.0005, -0.001, -0.002, -0.004, -0.006, -0.008, -0.010, -0.012, -0.016, -0.020, -0.030, -0.040):
        rmax = min(16.0, 0.9 * 2 * math.pi / math.sqrt(3 * abs(lam)))      # m_h^2 = 3 lam r^2 inside the FV series radius
        r = r0[r0 <= rmax]
        dv = C.dV("MC", r, lam, c["g2"], c["gY"], c["yt"])
        U = 3.0 / (8 * r * r) + lam * r ** 4 / 4 + dv
        dU = np.gradient(U, r)
        b = CC.barrier_info(r, U)
        out["%g" % lam] = dict(max_dU_dr=float(dU.max()), r_at_max_dU=float(r[int(np.argmax(dU))]),
                               n_positive_slope=int((dU > 0).sum()), barrier=b, r_max=float(r[-1]),
                               max_dV_slope=float(np.gradient(dv, r).max()))
    r = r0
    dvs = np.gradient(dv0, r)
    out["dV_slope_lam0"] = {"%.1f" % x: float(np.interp(x, r, dvs)) for x in (0.5, 1, 2, 3, 4, 5, 8, 12, 16)}
    out["note"] = ("kappa = 0: C0 = B_x = Om_eff = dE_DMC = 0 (the Gaussian collapses: gauge zero modes are free), the "
                   "s channel is a free particle with threshold 0, so the radial problem is U(r) = 3/(8r^2) + lam r^4/4 "
                   "+ dV_MC(r) at the physical couplings")
    CC.wjson(os.path.join(CC.OUT, "S2", "C170_S2_kappa0.json"), out)
    print(json.dumps(CC.clean({k: (v if not isinstance(v, dict) or "barrier" not in v else
                                   dict(max_dU_dr=v["max_dU_dr"], r=v["r_at_max_dU"], npos=v["n_positive_slope"],
                                        has_barrier=v["barrier"].get("has_barrier")))
                               for k, v in out.items()}), indent=1))


def scalecheck():
    """exact zero-mode scaling without matching: H(kappa, lam) = kappa^(2/3) H(1, lam/kappa^2)  (2D_G, dV off)."""
    c = boxA()
    base = dict(model="2D_G", g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"], dv="none")
    rows = []
    r1 = CC.solve_point(dict(base, kappa=1.0, lam=-0.012), prev=dict(dQ=0.012, rho=1.07))
    for k in (0.5, 1.5):
        cfg, bxc = cfg_for(k)
        rk = CC.solve_point(dict(base, kappa=k, lam=-0.012 * k * k), center=k ** (2 / 3) * r1["eps"],
                            gim=k ** (2 / 3) * r1["gamma"], cfg=cfg, boxchk=bxc)
        rows.append(dict(kappa=k, lam=-0.012 * k * k, eps=rk["eps"], gamma=rk["gamma"], flags=rk["flags"],
                         eps_pred=k ** (2 / 3) * r1["eps"], gamma_pred=k ** (2 / 3) * r1["gamma"],
                         rel_gamma=rk["gamma"] / (k ** (2 / 3) * r1["gamma"]) - 1,
                         rel_eps=rk["eps"] / (k ** (2 / 3) * r1["eps"]) - 1))
    out = dict(ref=dict(kappa=1.0, lam=-0.012, eps=r1["eps"], gamma=r1["gamma"], flags=r1["flags"]), rows=rows)
    CC.wjson(os.path.join(CC.OUT, "S2", "C170_S2_scalecheck.json"), out)
    print(json.dumps(CC.clean(out), indent=1))


def reg2dex():
    """no-cross-term path regression: 2D_ex at box A couplings and lambda_SM(172.6) vs the archived T14b run02 root."""
    c = boxA()
    d = CC.jload(os.path.join(CC.WORK, "archive", "r3", "twod_direct_sm", "run02", "shards2d", "2Dex_main_u0.0000.json"))
    x = [y for y in d["cols"] if round(y["mt"], 2) == 172.6][0]
    pt = dict(model="2D_ex", g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"], lam=c["lam"])
    r = CC.solve_point(pt, center=x["center"], gim=x["gim"], window=tuple(d["window"]))
    out = dict(archived=dict(Q2=x["Q2"], gamma2=x["gamma2"], lam=x["lam"], g2=x["g2"], center=x["center"]),
               mine=dict(eps=r["eps"], gamma=r["gamma"], flags=r["flags"]),
               rel_gamma=r["gamma"] / x["gamma2"] - 1, d_eps=r["eps"] - x["Q2"], d_lam=c["lam"] - x["lam"])
    CC.wjson(os.path.join(CC.OUT, "S2", "C170_S2_reg2dex.json"), out)
    print(json.dumps(CC.clean(out), indent=1))


REF_UP = [-0.008, -0.006, -0.005, -0.004, -0.003, -0.002, -0.0015, -0.001, -0.0005, -0.0002, 0.0]


REF2_UP = [-0.001, -0.0005, -0.0002, 0.0]


def refine2(nw, kappas):
    """second refinement (kappa_min / kappa_thr brackets): lambda = -0.001 ... 0, first seed shift (dQ, rho) = (0.006,
    1.1) (the 2D-1D offset of the no-cross models near the barrier top; the default 0.012 can land the window on a
    spurious narrow state there)."""
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(kappas)))) as ex:
        list(ex.map(_prof, kappas))
    c = boxA()
    sp = []
    for k in kappas:
        cfg, bxc = cfg_for(k)
        for m in MODELS:
            base = dict(model=m, kappa=float(k), g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"])
            sp.append(dict(name="S2_%s_k%.4f_up" % (m, k), base=base, lams=list(REF2_UP), stop_after_fail=99,
                           cfg=cfg, boxchk=bxc, prev0=dict(dQ=0.006, rho=1.1), outdir=CHAINS))
    with ProcessPoolExecutor(max_workers=max(1, min(nw, 14, len(sp)))) as ex:
        res = list(ex.map(CC.run_chain, sp))
    CC.status_update({"S2": {"state": "refinement 2 done", "refine2_kappas": kappas}})
    print(json.dumps(CC.clean({r["name"]: [(p["pt"]["lam"], p.get("status"), p.get("gamma")) for p in r["points"]]
                               for r in res}), indent=1))


def refine(nw, kappas):
    """kappa refinement near the vanishing threshold: up chains only, lambda = -0.008 ... 0."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(kappas)))) as ex:
        list(ex.map(_prof, kappas))
    c = boxA()
    sp = []
    for k in kappas:
        cfg, bxc = cfg_for(k)
        for m in MODELS:
            base = dict(model=m, kappa=float(k), g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"])
            sp.append(dict(name="S2_%s_k%.4f_up" % (m, k), base=base, lams=list(REF_UP), stop_after_fail=99,
                           stop_gamma_below=TARGET / 30, cfg=cfg, boxchk=bxc, prev0=dict(dQ=0.012, rho=1.07),
                           outdir=CHAINS))
    CC.status_update({"S2": {"state": "refinement running", "refine_kappas": kappas,
                             "resume_refine": "python -B c170_s2.py refine %d %s" % (nw, ",".join("%g" % k for k in kappas))}})
    with ProcessPoolExecutor(max_workers=max(1, min(nw, 14, len(sp)))) as ex:
        res = list(ex.map(CC.run_chain, sp))
    CC.status_update({"S2": {"state": "refinement done", "refine_time_s": time.time() - t0}})
    print(json.dumps(CC.clean({r["name"]: [(p["pt"]["lam"], p.get("status"), p.get("gamma")) for p in r["points"]]
                               for r in res}), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "run"
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if md == "run":
        run(nw)
    elif md == "refine":
        refine(nw, [float(z) for z in sys.argv[3].split(",")])
    elif md == "refine2":
        refine2(nw, [float(z) for z in sys.argv[3].split(",")])
    elif md == "curves":
        curves()
    elif md == "kzero":
        kzero()
    elif md == "scalecheck":
        scalecheck()
    elif md == "reg2dex":
        reg2dex()
