# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/nm_model.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations\code\20261014_claude_v12_r3\realtime_2d\nm_model.py on 2026-10-15 by claims-run05 (C161/D132); 8 edits (see make_xi_copies.py)
# PROVENANCE: new module by agent realtime-2d (C157/D129, v12 round 3) on 2026-10-14.  Functions tagged [C149] are copied
# verbatim from 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_direct.py (original read-only, unchanged).
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Real-time xi_c chain, model nodes (D129 nm_model, C161 run05 version): at each (m_t, u) node the 1D BO potential U_BO = U1 + eps0 of the run05 2D model (2D_ex + NNLO + full one-loop cross terms), calibrated to the run05 2D widths (holonomy_cross run05 shards); 'nodes' -> 'calib' -> 'tables' (PotentialTable npz for hr_core).  env C161_CROSS=run02 restores the run02 cross terms (regression).  Paper: PRD App. H3. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C157 / D129 (realtime-2d): the round-3 CENTRAL zero-mode model as node potentials for the real-time pipeline.

Central model (D121 appendix run02 = C149_direct): at node u (mu(u) = mu_b/sqrt(cosh 2u)), column (m_t, alpha_s) with the
T1 NNLO 'central' couplings (sm_provider, M_h = 125.13):
   H_2D = -1/2 d_r^2 + T_s + U1(r) + 1/2 OM(r) s^2 + V_L(r, s)            (s = |w|, 3D s-wave, T_s = sine DVR)
   U1 = [3/(8r^2) + C0 + B_x + lam r^4/4 + dV_MC + dE_DMC]_(T14b 2D_ex)  + dC0 + dBx + dU3      (cross increments)
   OM = Om_eff^2 + dOm2e + Om_X^2(W, H, top)   ;   V_L = exact light-fermion twist energy
1D construction used here (D116 Sec. 0.1 analogue, "node-calibrated"):
   U_BO(r;u)  = U1(r;u) + eps0(r;u)    eps0 = adiabatic w ground energy of T_s + OM s^2/2 + V_L (N = 48 DVR, S_max = 12)
   U_eff(r;u) = U_BO + A(u) exp(-((r - 2.30)/0.35)^2)   (D116 bump shape; A(u) fixed so that the 1D Siegert width equals the
                2D direct width gamma_2D(u) of C149 run02 at the 11 hazard nodes; u > 5: A(5))
Writes only to results/20261014_claude_v12_r3/realtime_2d/run01/.
Usage: python -B nm_model.py check | nodes [nw] | calib | tables
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
HX = _zp.ZMLIB
RT2D = _zp.ZMLIB
os.environ["C150_COUPLINGS"] = "module:" + os.path.join(HX, "c150_nnlo_central.py")
sys.path.insert(0, RT2D)
sys.path.insert(0, HX)
sys.path.insert(0, _zp.step("04_cross_terms"))  # [v12-package] C149_direct5
sys.path.insert(0, HERE)
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cross_core as X  # noqa: E402
ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("w/claims_run05/run01/xi_run05"))
X.RES = RES                                   # any cache of the copied modules goes to MY run directory
import C150_twod as W  # noqa: E402
import cross_kappa as K  # noqa: E402
import cross_2d as T  # noqa: E402
import cross_var as V  # noqa: E402
import sm_provider as SP  # noqa: E402
import C149_direct5 as D5  # noqa: E402  (run05 cross terms; holonomy_cross, read only)
D5.D.SH = D5.D.RES2 = D5.D.MYPROF = RES                  # never write into old runs

W.RES, W.PROFDIR = RES, os.path.join(RES, "profiles")
W.SH1, W.SH2 = os.path.join(RES, "sh1"), os.path.join(RES, "sh2")
T14RES = os.path.join(ROOT, _zp.w("r3/twod_direct_sm/run02"))
C149SH = _zp.w("r3/holonomy_cross/%s/shards" % ("run02" if os.environ.get("C161_CROSS") == "run02" else "run05"))
NODED = os.path.join(RES, "nodes")
TDIR = os.path.join(RES, "tables")
PI = math.pi
PV11 = np.load(K.PROFILE)
ALS = 0.1180
MTS = [172.2, 172.6, 172.95, 172.96, 173.0]
UN11 = np.round(np.arange(0.0, 5.0001, 0.5), 6)
UNODES = np.round(np.arange(0.0, 7.0001, 0.25), 6)
DR_TAB, R_TAB = 0.0025, 17.5
RLIM = 16.6                                   # cross increments (rg = T.RX <= 16.62) frozen beyond this radius
BUMP = dict(r0=2.30, w=0.35)
A_D116 = 0.0937
SG48 = W.SGrid(48, 12.0)


def clean(o):
    return W.clean(o)


def wjson(name, obj):
    p = name if os.path.isabs(name) else os.path.join(RES, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def say(msg):
    print(msg, flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "C157_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


_CS = {}


def CS():
    if "c" not in _CS:
        _CS["c"] = W.Couplings(W.KIND)
    return _CS["c"]


def g3_at(mt, als, mu):                                             # [C149]
    return float(SP.coup_point(mt, als, W.MH, mu, "central")["g3"])


def base_parts(mt, als, u):
    """[C149] T14b run02 profile parts (read only) -> C150 Parts; fallback: compute into MY run01/profiles."""
    fn = os.path.join(T14RES, "profiles", "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))
    if not os.path.exists(fn):
        fn = os.path.join(RES, "profiles", "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))
        if not os.path.exists(fn):
            c = CS()(mt, als, W.mu_node(u))
            d0 = W.compute_profile(c["g2"], c["gY"])
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            tmp = fn[:-4] + "_tmp%d.npz" % os.getpid()
            np.savez(tmp, **{k: np.asarray(d0[k]) for k in ("rf", "C0", "Bx", "Om2e", "EtBr", "g2", "gY", "chk_Et")})
            os.replace(tmp, fn)
    d = dict(np.load(fn))
    return (W.Parts(d["rf"], d["C0"], d["Bx"], d["Om2e"], d["EtBr"]),
            dict(g2=float(d["g2"]), gY=float(d["gY"]), file=fn, rf_max=float(d["rf"][-1]), rf_min=float(d["rf"][0])))


def etot_g(p, r, g2, gY):                                           # [C149]
    a, k, b, c = p
    D = k * b - c * c
    kin = 0.75 * (2 * a + k + b)
    mass = 3 * r * r / 16 * (2 * g2 * g2 / a + (g2 * g2 * b - 2 * g2 * gY * c + gY * gY * k) / D)
    ym = 3 * g2 * g2 / (4 * a * a) + 3 * g2 * g2 * b / (2 * a * D)
    ang = 3.0 / (a * r * r) * (b * (k - a) ** 2 - 2 * c * c * (k - a) + k * c * c) / D
    return kin + mass + ym + ang


def cross_increments(c, g3):
    """[C149] node couplings -> splines on rf of dC0, dBx, dOm2e (variational), OmX2_nonlight(beta'), beta', dU3var."""
    g2, gY, yt, lam = c["g2"], c["gY"], c["yt"], c["lam"]
    rf = PV11["rf"]
    rg = T.RX
    rows = [K.hessian(float(x), lam, g2, gY, yt) for x in rg]
    rr = np.minimum(rf, rg[-1])
    Hs = {}
    for kk in ("H11", "H33", "H3w", "Hww"):
        Hs[kk] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top", "light")) for x in rows])(rr)
        Hs[kk + "_nl"] = CubicSpline(rg, [sum(x["parts"][g][kk] for g in ("W", "H", "top")) for x in rows])(rr)
    P0, _ = V.newton(lambda p: etot_g(p, rf, g2, gY), PV11["P"].astype(float))
    qq = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    omx = Hs["Hww"] - 2 * qq["beta"] * Hs["H3w"] + qq["beta"] ** 2 * Hs["H33"]
    Pst = P0.copy()
    Pst[2] = np.sqrt(np.maximum(qq["Om2"] + omx, 1e-8)) + P0[3] ** 2 / P0[1]
    P1, gres = V.newton(lambda p: etot_g(p, rf, g2, gY) + V.fcross(p, Hs, "SU2+Y"), Pst, iters=12)
    q0 = W.derived(rf, P0, np.gradient(P0, rf, axis=1, edge_order=2), g2, gY)
    q1 = W.derived(rf, P1, np.gradient(P1, rf, axis=1, edge_order=2), g2, gY)
    fsu2 = V.fcross(P1, Hs, "SU2")
    beta = q1["beta"]
    omx_nl = Hs["Hww_nl"] - 2 * beta * Hs["H3w_nl"] + beta ** 2 * Hs["H33_nl"]
    kap = np.array([x["kappa3"] for x in rows])
    k0 = (6 * X.curv(0.0, "P")[0] - 12 * X.curv(0.0, "A")[0]) / (4 * PI ** 2)
    E0, s0 = 12.980 * g3 ** (2 / 3), 3.66 * (g3 / 0.5396) ** (2 / 3)
    Ev, _ = V.su3_var(kap, E0, s0)
    Ev0, _ = V.su3_var(np.array([k0]), E0, s0)
    r2 = rf * rf
    out = dict(dC0=CubicSpline(rf, r2 * (q1["C0"] + fsu2 - q0["C0"])), dBx=CubicSpline(rf, r2 * (q1["Bx"] - q0["Bx"])),
               dOm2e=CubicSpline(rf, q1["Om2e"] - q0["Om2e"]), omx_nl=CubicSpline(rf, omx_nl),
               beta=CubicSpline(rf, beta), dU3=CubicSpline(rg, Ev - Ev0[0]))
    info = dict(newton_grad_max_r_gt_0p05=float(np.max(gres[rf > 0.05])), chk_base_Et=None, E0_su3=E0, s0_su3=s0)
    return out, info


def V_light_c(r, s, beta, g2, gY):                                  # [C149]
    f, E0 = T.ebar_A()
    out = np.zeros((len(r), len(s)))
    for nm, mult, t3, y in K.LIGHT:
        q = np.abs(gY * y - g2 * beta * t3)
        out += -2.0 * mult * (f(np.minimum(q[:, None] * s[None, :] / (2 * PI), 2.5)) - E0)
    return out


def eps_levels(OM, Wl, sg, k=4, chunk=1500):
    """lowest k adiabatic w levels of T_s + OM s^2/2 + Wl(s) for each row (batched eigvalsh)."""
    n = len(OM)
    out = np.empty((n, k))
    idx = np.arange(sg.N)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        H = np.repeat(sg.T[None], b - a, axis=0)
        H[:, idx, idx] += 0.5 * OM[a:b, None] * sg.s2[None, :] + (Wl[a:b] if Wl is not None else 0.0)
        out[a:b] = np.linalg.eigvalsh(H)[:, :k]
    return out


def node_eval(mt, als, u, r):
    """new-model node pieces on the radial grid r: U1, OM, beta (cross increments frozen beyond RLIM)."""
    c = CS()(mt, als, W.mu_node(u))
    g3 = g3_at(mt, als, W.mu_node(u))
    Pp, pinfo = base_parts(mt, als, u)
    dmc = W.DMC()
    U0, OM0 = W.build2d(Pp, dmc, r, c, "2D_ex")
    inc, info = (cross_increments if os.environ.get("C161_CROSS") == "run02" else D5.cross_increments5)(c, g3)
    rc = np.minimum(r, RLIM)
    U1 = U0 + inc["dC0"](rc) / rc ** 2 + inc["dBx"](rc) / rc ** 2 + inc["dU3"](rc)
    OM = OM0 + inc["dOm2e"](rc) + inc["omx_nl"](rc)
    beta = inc["beta"](rc)
    return dict(U1=U1, OM=OM, beta=beta, U0=U0, OM0=OM0, c=c, g3=g3, pinfo=pinfo, info=info)


def node_path(mt, u):
    return os.path.join(NODED, "node_mt%.2f_u%.4f.npz" % (mt, u))


def _node_job(args):
    mt, u = args
    fn = node_path(mt, u)
    if os.path.exists(fn):
        return fn, 0.0
    t0 = time.time()
    nr = int(round(R_TAB / DR_TAB))
    r = np.arange(1, nr + 1) * DR_TAB
    d = node_eval(mt, ALS, u, r)
    Wl = V_light_c(r, SG48.s, d["beta"], d["c"]["g2"], d["c"]["gY"])
    ev = eps_levels(d["OM"], Wl, SG48, k=4)
    evq = 1.5 * np.sqrt(d["OM"])                                     # quadratic (no light twist) reference
    os.makedirs(NODED, exist_ok=True)
    tmp = fn[:-4] + "_tmp%d.npz" % os.getpid()
    np.savez(tmp, r=r, U1=d["U1"], OM=d["OM"], beta=d["beta"], U0=d["U0"], OM0=d["OM0"], eps=ev, eps_quadOM=evq,
             g2=d["c"]["g2"], gY=d["c"]["gY"], yt=d["c"]["yt"], lam=d["c"]["lam"], g3=d["g3"], mu=W.mu_node(u),
             prof=d["pinfo"]["file"], rf_max=d["pinfo"]["rf_max"],
             newton=d["info"]["newton_grad_max_r_gt_0p05"])
    os.replace(tmp, fn)
    return fn, time.time() - t0


def nodes(nw=4, mts=None, us=None):
    jobs = [(mt, float(u)) for mt in (mts or MTS) for u in (UNODES if us is None else us)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for fn, dt in ex.map(_node_job, jobs):
            if dt:
                say("node %s %.1fs" % (os.path.basename(fn), dt))
    say("nodes done: %d jobs %.0fs" % (len(jobs), time.time() - t0))


def load_node(mt, u):
    return dict(np.load(node_path(mt, u)))


def U_BO_on(nd, r):
    """U1 + eps0 interpolated from the node table grid onto r (r inside the table)."""
    UB = nd["U1"] + nd["eps"][:, 0]
    rt = nd["r"]
    return np.interp(r, rt, UB * rt * rt) / (r * r)


# ------------------------------------------------------------------------------------------------ 2D targets
def targets_2d():
    """gamma_2D(u), Q_2D(u) of C149 run02 (alpha_s = 0.1180) at the 11 nodes; m_t = 172.95 by cubic interpolation of
    ln gamma and Q in m_t through 172.6/172.8/173.0/173.2 (leave-one-out check at 172.8 through 172.4/172.6/173.0/173.2)."""
    G, Q = {}, {}
    for u in UN11:
        d = jload(os.path.join(C149SH, "X_main_u%.4f.json" % u))
        for x in d["cols"]:
            if abs(x["als"] - ALS) < 1e-9:
                G[(round(x["mt"], 2), round(u, 4))] = float(x["gamma"])
                Q[(round(x["mt"], 2), round(u, 4))] = float(x["Q"])
    loo = []
    for u in UN11:
        uk = round(u, 4)
        xs = [172.6, 172.8, 173.0, 173.2]
        cg = np.polyfit(xs, [math.log(G[(m, uk)]) for m in xs], 3)
        cq = np.polyfit(xs, [Q[(m, uk)] for m in xs], 3)
        for mx in (172.95, 172.96):
            G[(mx, uk)] = float(math.exp(np.polyval(cg, mx)))
            Q[(mx, uk)] = float(np.polyval(cq, mx))
        xl = [172.4, 172.6, 173.0, 173.2]
        cl = np.polyfit(xl, [math.log(G[(m, uk)]) for m in xl], 3)
        loo.append(float(math.exp(np.polyval(cl, 172.8)) / G[(172.8, uk)] - 1))
    return G, Q, dict(loo_rel_gamma_172p8=loo, max_abs=float(np.max(np.abs(loo))))


# ------------------------------------------------------------------------------------------------ calibration
def _cols(mts, us, A):
    r, m = W.R.grid(16.0)
    cols, keys = [], []
    for mt in mts:
        for u in us:
            nd = load_node(mt, u)
            cols.append(U_BO_on(nd, r) + W.bump(r, A[(mt, u)], BUMP["r0"], BUMP["w"]))
            keys.append((mt, u))
    return cols, keys, r, m


def solve_cols(mts, us, A, guess):
    cols, keys, r, m = _cols(mts, us, A)
    g = np.array([guess[k] for k in keys])
    E, rs, bad, nres = W.solve1d(cols, r, m, g, spread=0.01)
    return {k: E[j] for j, k in enumerate(keys)}, dict(bad=int(np.sum(bad)), max_resid=float(np.max(rs)), nres=nres)


def calib():
    t0 = time.time()
    G, Q, loo = targets_2d()
    us = [round(float(u), 4) for u in UN11]
    keys = [(mt, u) for mt in MTS for u in us]
    out = dict(targets=dict(gamma={"%.2f|%.1f" % k: G[k] for k in keys}, Q={"%.2f|%.1f" % k: Q[k] for k in keys}),
               interp_check=loo, bump_shape=BUMP)
    # A = 0 (pure BO-lo of the new model) and A = A_D116 (D116 bump transferred unchanged)
    guess = {k: complex(Q[k] - 0.02, -0.5 * G[k] / 1.05) for k in keys}
    E0, d0 = solve_cols(MTS, us, {k: 0.0 for k in keys}, guess)
    gD = {k: complex(E0[k].real + 0.012, E0[k].imag * 1.3) for k in keys}
    ED, dD = solve_cols(MTS, us, {k: A_D116 for k in keys}, gD)
    out["A0"] = {"%.2f|%.1f" % k: dict(Q=E0[k].real, gamma=-2 * E0[k].imag, rel=-2 * E0[k].imag / G[k] - 1) for k in keys}
    out["A_D116"] = {"%.2f|%.1f" % k: dict(Q=ED[k].real, gamma=-2 * ED[k].imag, rel=-2 * ED[k].imag / G[k] - 1)
                     for k in keys}
    out["diag"] = dict(A0=d0, AD116=dD)
    say("calib: A=0 rel range %.4f..%.4f ; A=0.0937 rel range %.4f..%.4f" % (
        min(v["rel"] for v in out["A0"].values()), max(v["rel"] for v in out["A0"].values()),
        min(v["rel"] for v in out["A_D116"].values()), max(v["rel"] for v in out["A_D116"].values())))
    # secant on ln gamma(A) per column
    Aa = {k: 0.0 for k in keys}
    Ab = {k: A_D116 for k in keys}
    la = {k: math.log(-2 * E0[k].imag) for k in keys}
    lb = {k: math.log(-2 * ED[k].imag) for k in keys}
    Eb = dict(ED)
    hist = []
    for it in range(6):
        An = {}
        for k in keys:
            lt = math.log(G[k])
            den = lb[k] - la[k]
            An[k] = Ab[k] + (lt - lb[k]) * (Ab[k] - Aa[k]) / den if abs(den) > 1e-14 else Ab[k]
        gn = {k: complex(Eb[k].real + (An[k] - Ab[k]) * 0.12, Eb[k].imag * math.exp(G[k] and
                                                                                    (math.log(G[k]) - lb[k]))) for k in keys}
        En, dn = solve_cols(MTS, us, An, gn)
        mx = max(abs(math.log(-2 * En[k].imag / G[k])) for k in keys)
        hist.append(dict(it=it, max_abs_dln_gamma=mx, diag=dn))
        say("calib it %d: max|dln gamma| = %.2e (bad=%d)" % (it, mx, dn["bad"]))
        for k in keys:
            Aa[k], la[k] = Ab[k], lb[k]
            Ab[k], lb[k], Eb[k] = An[k], math.log(-2 * En[k].imag), En[k]
        if mx < 2e-6:
            break
    out["calibrated"] = {"%.2f|%.1f" % k: dict(A=Ab[k], Q=Eb[k].real, gamma=-2 * Eb[k].imag,
                                               rel=-2 * Eb[k].imag / G[k] - 1, dQ_vs_2D=Eb[k].real - Q[k]) for k in keys}
    out["secant_hist"] = hist
    # hazard check: I = int gamma deta over the 11 nodes (C150 integ), P1 vs C149 run02
    p1 = {}
    for mt in MTS:
        g2d = np.array([G[(mt, u)] for u in us])
        g1d = np.array([-2 * Eb[(mt, u)].imag for u in us])
        g0 = np.array([-2 * E0[(mt, u)].imag for u in us])
        gd = np.array([-2 * ED[(mt, u)].imag for u in us])
        p1["%.2f" % mt] = dict(P1_2D=math.exp(-W.integ(UN11, g2d)), P1_1D_calibrated=math.exp(-W.integ(UN11, g1d)),
                               P1_1D_A0=math.exp(-W.integ(UN11, g0)), P1_1D_AD116=math.exp(-W.integ(UN11, gd)),
                               A_nodes=[Ab[(mt, u)] for u in us])
    out["P1_check"] = p1
    out["time_s"] = time.time() - t0
    wjson("C157_calib.json", out)
    say("calib done %.0fs: %s" % (time.time() - t0, json.dumps(clean({k: (round(v["P1_2D"], 5),
                                                                       round(v["P1_1D_calibrated"], 5),
                                                                       round(v["P1_1D_A0"], 5),
                                                                       round(v["P1_1D_AD116"], 5))
                                                                   for k, v in p1.items()}))))
    return out


# ------------------------------------------------------------------------------------------------ tables
def tpath(mt, kind="eff"):
    return os.path.join(TDIR, "U_table_%s_mt%.2f.npz" % (kind, mt))


def A_of_u(mt, cal):
    us = np.array(UN11)
    A = np.array([cal["calibrated"]["%.2f|%.1f" % (mt, u)]["A"] for u in us])
    sp = CubicSpline(us, A)
    return lambda u: float(sp(min(float(u), 5.0))), A


def tables():
    cal = jload(os.path.join(RES, "C157_calib.json"))
    os.makedirs(TDIR, exist_ok=True)
    logmu = np.linspace(np.log(1e10), np.log(3e18), 600)
    for mt in MTS:
        Af, An = A_of_u(mt, cal)
        rows_eff, rows_bo, rows_e0, info = [], [], [], []
        for u in UNODES:
            nd = load_node(mt, float(u))
            r = nd["r"]
            UB = nd["U1"] + nd["eps"][:, 0]
            rows_bo.append(UB)
            rows_eff.append(UB + W.bump(r, Af(u), BUMP["r0"], BUMP["w"]))
            rows_e0.append(nd["eps"][:, 0])
            info.append(dict(u=float(u), lam=float(nd["lam"]), g2=float(nd["g2"]), gY=float(nd["gY"]), yt=float(nd["yt"]),
                             mu=float(nd["mu"]), A=Af(u)))
        lam = np.array([x["lam"] for x in info])
        lr = SP.couplings(mt, ALS, 125.13, np.exp(logmu), "central")["lam"]
        meta = dict(model="C161 run05 central model 1D: U_BO = U1 + eps0 (C149 run05 2D_ex + NNLO + full one-loop cross terms) "
                          "+ node-calibrated bump A(u) exp(-((r-2.30)/0.35)^2)", m_t=mt, alpha_s=ALS, M_h=125.13,
                    couplings="sm_provider central (T1 NNLO)", A_nodes=list(map(float, An)), node_info=info,
                    built=time.strftime("%Y-%m-%d %H:%M:%S"))
        for kind, rows, lamv in (("eff", rows_eff, lam), ("bo", rows_bo, lam), ("eps0", rows_e0, 0 * lam)):
            fn = tpath(mt, kind)
            tmp = fn[:-4] + "_tmp.npz"
            np.savez(tmp, r=r, u=UNODES, U=np.array(rows), lam=lamv, g2=np.array([x["g2"] for x in info]),
                     gY=np.array([x["gY"] for x in info]), yt=np.array([x["yt"] for x in info]),
                     mu=np.array([x["mu"] for x in info]), logmu=logmu, lam_run=lr, meta=json.dumps(clean(meta)))
            os.replace(tmp, fn)
        say("tables mt=%.2f: lam(u=0)=%.6f lam(u=7)=%.6f A(0..5)=%s" % (mt, lam[0], lam[-1],
                                                                        np.round(An, 4).tolist()))


def check():
    t0 = time.time()
    rf = PV11["rf"]
    Pp, pinfo = base_parts(172.6, ALS, 0.0)
    print("PV11 rf range", float(rf[0]), float(rf[-1]), len(rf), "T14b prof rf", pinfo["rf_min"], pinfo["rf_max"])
    print("T.RX", float(T.RX[0]), float(T.RX[-1]), len(T.RX), "LIGHT n", len(K.LIGHT))
    r = np.arange(1, 7001) * DR_TAB
    t1 = time.time()
    d = node_eval(172.6, ALS, 0.0, r)
    t2 = time.time()
    Wl = V_light_c(r, SG48.s, d["beta"], d["c"]["g2"], d["c"]["gY"])
    ev = eps_levels(d["OM"], Wl, SG48, k=4)
    t3 = time.time()
    sg24 = W.SGrid(24, 12.0)
    Wl24 = V_light_c(r, sg24.s, d["beta"], d["c"]["g2"], d["c"]["gY"])
    ev24 = eps_levels(d["OM"], Wl24, sg24, k=2)
    for x in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 16.0, 17.5):
        i = int(round(x / DR_TAB)) - 1
        print("r=%5.2f U1=%.5f U0=%.5f OM=%.5f OM0=%.5f beta=%.4f eps0=%.5f eps0_N24=%.5f 1.5sqrtOM=%.5f eps1-eps0=%.4f" % (
            r[i], d["U1"][i], d["U0"][i], d["OM"][i], d["OM0"][i], d["beta"][i], ev[i, 0], ev24[i, 0],
            1.5 * math.sqrt(d["OM"][i]), ev[i, 1] - ev[i, 0]))
    print("couplings", d["c"], "g3", d["g3"], "newton", d["info"])
    print("timing: profile/base %.1fs node_eval %.1fs eps %.1fs" % (t1 - t0, t2 - t1, t3 - t2))
    G, Q, loo = targets_2d()
    print("2D target 172.6 u=0: gamma %.5e Q %.6f ; 172.95: %.5e ; loo %s" % (G[(172.6, 0.0)], Q[(172.6, 0.0)],
                                                                             G[(172.95, 0.0)], loo["max_abs"]))
    # quick 1D Siegert of U_BO (A = 0) at u = 0
    rr, m = W.R.grid(16.0)
    UB = np.interp(rr, r, (d["U1"] + ev[:, 0]) * r * r) / (rr * rr)
    E, rs, bad, nres = W.solve1d([UB, UB + W.bump(rr, A_D116, 2.30, 0.35)], rr, m,
                                 np.array([Q[(172.6, 0.0)] - 0.02 - 0.5j * G[(172.6, 0.0)] / 1.05,
                                           Q[(172.6, 0.0)] - 0.005 - 0.5j * G[(172.6, 0.0)] * 1.2]), spread=0.01)
    print("1D BO-lo A=0:", E[0], "gamma ratio 1D/2D = %.4f" % (-2 * E[0].imag / G[(172.6, 0.0)]),
          "| A=0.0937:", E[1], "ratio %.4f" % (-2 * E[1].imag / G[(172.6, 0.0)]), "bad", bad, "resid", rs)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "check"
    if md == "check":
        check()
    elif md == "nodes":
        nodes(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif md == "calib":
        calib()
    elif md == "tables":
        tables()
