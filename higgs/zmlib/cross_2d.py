# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/cross_2d.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] 2D solver pieces with the cross-term channel potential, including the exact light-fermion twist V_L(r, s).  Paper: PRD App. H2'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 stages C/D -- zero-mode/nonzero-mode cross terms in the D112 (r, |w|) 2D model (holonomy-cross, D121 Sec. 3).

H_2D(D112) + cross terms (variant tags after '|', joined by '+'):
  SU3 / SU3v : U1 += DeltaU_3(r) = 1/2 [kappa3(r) - kappa3(0)] <c^2>_3, <c^2> = 3.66 (Gaussian) / 3.77 (virial)
  SU2        : U1 += E_X^x(r) = (3/2) H11/a + (3/4) H33/k   (all SM fields, CCA curvatures, cross_kappa)
  Yq         : om2 += Om_X^2(r) (all SM fields, quadratic)
  Yb         : om2 += Om_X^2 without light fermions (W, Higgs, top quadratic)
  YL         : + exact light-fermion twist energy V_L(r, s), s-wave (direction-averaged) image sum
               V_L = sum_alpha (-2 n_alpha)[Ebar_A(|q_alpha| s/2pi) - Ebar_A(0)],  q_alpha = gY Y - g2 beta(r) T3,
               Ebar_A(t) = -(1/2pi^2) sum'_w (-1)^{|w|^2} j0(2 pi |w| t)/|w|^4   (massless AAA, exact in t)
Solver = local UNCHANGED copy of C140_beyond_bo (provenance line); job2/bc2 below are adapted copies of B.job/B.bc_data
with the extra channel potential W(r, s).  Centers for the window root search from lambda = 0 FD ground-state shifts.
Usage: python -B cross_2d.py stageC | stageD | post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import numpy as np
from scipy.interpolate import CubicSpline
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_core as X  # noqa: E402
import cross_kappa as K  # noqa: E402
import C140_beyond_bo as B  # noqa: E402  (local unchanged copy)

B.RES = X.RES                      # logs of the copied solver go to this run directory
ORIG_BUILD = B.build
RX = np.round(np.arange(0.01, 16.6201, 0.02), 4)
C140SUM = os.path.join(X.ROOT, _zp.w("r2/higgs_beyond_bo/run01/C140_summary.json"))
_XT, _EB = {}, {}
PI = math.pi


def xterms(lam):
    key = round(float(lam), 6)
    if key in _XT:
        return _XT[key]
    path = os.path.join(X.RES, "C149_xterms_%g.npz" % key)
    if os.path.exists(path):
        d = dict(np.load(path))
    else:
        prof = K.derived_profile()
        rows = [K.hessian(float(x), key) for x in RX]
        a, k = np.interp(RX, prof["rf"], prof["a"]), np.interp(RX, prof["rf"], prof["k"])
        beta = np.interp(RX, prof["rf"], prof["beta"])
        d = dict(r=RX, beta=beta, dU3=np.array([x["dU3_gauss"] for x in rows]),
                 dU3v=np.array([x["dU3_virial"] for x in rows]))
        for g in ("W", "H", "top", "light"):
            Hg = {kk: np.array([x["parts"][g][kk] for x in rows]) for kk in ("H33", "Hww", "H3w", "H11")}
            d["EXx_" + g] = 1.5 * Hg["H11"] / a + 0.75 * Hg["H33"] / k
            d["OmX2_" + g] = Hg["Hww"] - 2 * beta * Hg["H3w"] + beta ** 2 * Hg["H33"]
        np.savez(path, **d)
    sp = {kk: CubicSpline(d["r"], d[kk]) for kk in d if kk != "r"}
    _XT[key] = sp
    return sp


def ebar_A():
    if "f" not in _EB:
        t = np.linspace(0.0, 2.5, 2501)
        rw = np.sqrt(X.QW)
        wt = X.SW * X.NW / X.QW ** 2
        E = np.array([-(1 / (2 * PI ** 2)) * np.sum(np.sinc(2 * rw * tt) * wt) for tt in t])
        _EB["f"] = CubicSpline(t, E)
        _EB["E0"] = float(E[0])
    return _EB["f"], _EB["E0"]


def V_light(r, s, lam):
    """exact s-wave light-fermion twist energy on the grid r (n,) x s (N,) -> (n, N)."""
    f, E0 = ebar_A()
    beta = xterms(lam)["beta"](r)
    g2, gY = K.V11["g2"], K.V11["gY"]
    out = np.zeros((len(r), len(s)))
    for nm, mult, t3, y in K.LIGHT:
        q = np.abs(gY * y - g2 * beta * t3)
        out += -2.0 * mult * (f(np.minimum(q[:, None] * s[None, :] / (2 * PI), 2.5)) - E0)
    return out


def tags_of(var):
    return var.split("|")[1].split("+") if "|" in var else []


def build2(M, var, match, lam, r, tilt=0.0):
    if "|" not in var:
        return ORIG_BUILD(M, var, match, lam, r, tilt)
    base = var.split("|")[0]
    U1, om2 = ORIG_BUILD(M, base, match, lam, r, tilt)
    sp = xterms(lam)
    for t in tags_of(var):
        if t == "SU3":
            U1 = U1 + sp["dU3"](r)
        elif t == "SU3v":
            U1 = U1 + sp["dU3v"](r)
        elif t == "SU2":
            U1 = U1 + sum(sp["EXx_" + g](r) for g in ("W", "H", "top", "light"))
        elif t == "Yq":
            om2 = om2 + sum(sp["OmX2_" + g](r) for g in ("W", "H", "top", "light"))
        elif t == "Yb":
            om2 = om2 + sum(sp["OmX2_" + g](r) for g in ("W", "H", "top"))
        elif t == "YL":
            pass                                  # exact light-fermion term added in job2 / fd_E0
        else:
            raise ValueError(t)
    return U1, om2


B.build = build2


def bc2(U1, om2, Wl, sg, h, m, order=2):
    """B.bc_data with the extra channel potential W(r_i, s) (adapted copy)."""
    P = U1.shape[0]
    N = sg.N
    eps = np.empty((P, 5, N))
    Phi = np.empty((P, N, N))
    for p in range(P):
        for j, i in enumerate(range(m - 2, m + 3)):
            Hc = sg.T + np.diag(0.5 * om2[p, i] * sg.s2 + (Wl[p][i] if Wl[p] is not None else 0.0))
            w, V = np.linalg.eigh(Hc)
            eps[p, j] = w
            if i == m:
                Phi[p] = V
    Uc = U1[:, m - 2:m + 3][:, :, None] + eps
    Um_ = Uc[:, 2]
    U1d = (Uc[:, 0] - 8 * Uc[:, 1] + 8 * Uc[:, 3] - Uc[:, 4]) / (12 * h)
    U2d = (-Uc[:, 0] + 16 * Uc[:, 1] - 30 * Uc[:, 2] + 16 * Uc[:, 3] - Uc[:, 4]) / (12 * h * h)

    def Y(E):
        d = E[:, None] - Um_
        kk = np.where(d.real > 0, np.sqrt(2 * d + 0j), 1j * np.sqrt(-2 * d + 0j))
        k1 = -U1d / kk
        k2 = -U2d / kk - U1d ** 2 / kk ** 3
        y = 1j * kk - k1 / (2 * kk) + (1j / (2 * kk)) * (3 * k1 ** 2 / (4 * kk ** 2) - k2 / (2 * kk))
        return np.einsum("pab,pb,pcb->pac", Phi, y, Phi)
    return Y


def job2(spec):
    """adapted copy of B.job (window rule only) with the exact light-fermion channel term."""
    t0 = time.time()
    M = B.Model()
    cfg = spec["cfg"]
    N, h, Rout, Smax = int(cfg["N"]), float(cfg["h"]), float(cfg["Rout"]), float(cfg.get("Smax", 14.0))
    r, m = B.rgrid(Rout, h)
    sg = B.SGrid(N, Smax)
    probs = spec["probs"]
    P = len(probs)
    U1 = np.empty((P, len(r)))
    om2 = np.empty((P, len(r)))
    Wl, ph0, ph1, fr = [], np.empty(P), np.empty(P), []
    for j, p in enumerate(probs):
        u, o = build2(M, p["var"], p["match"], p["lam"], r)
        U1[j], om2[j] = u, o
        Wl.append(V_light(r, sg.s, p["lam"]) if "YL" in tags_of(p["var"]) else None)
        s, al = B.frob(u, r)
        ph0[j] = r[0] ** s * (1 + al / s * r[0])
        ph1[j] = r[1] ** s * (1 + al / s * r[1])
        fr.append([s, al])
    Dd = U1[:, :, None] + 0.5 * om2[:, :, None] * sg.s2[None, None, :]
    for j in range(P):
        if Wl[j] is not None:
            Dd[j] += Wl[j]
    Tn = (-(h * h / 6.0) * sg.T).astype(complex)
    Yf = bc2(U1, om2, Wl, sg, h, m)
    cn = {}

    def fun(E):
        s_, la = B.prop(np.asarray(E, complex), Dd, Tn, h, m, ph0, ph1, Yf)
        if "c" not in cn:
            cn["c"] = la.copy()
        return s_ * np.exp(np.clip(la - cn["c"], -700.0, 700.0))
    half, step = spec["window"]
    offs = np.arange(-half, half + 0.5 * step, step)
    cen = np.array([p["center"] for p in probs], float)
    av = np.array([np.abs(fun(cen + o + 0j)) for o in offs])
    i0 = int(np.argmin(np.abs(offs)))
    jmin = np.empty(P, int)
    for jj in range(P):
        a_ = av[:, jj]
        loc = [i for i in range(1, len(offs) - 1) if a_[i] < a_[i - 1] and a_[i] < a_[i + 1]]
        jmin[jj] = min(loc, key=lambda i: abs(i - i0)) if loc else i0
    g = cen + offs[jmin] - 0.5j * np.array([p["gimag"] for p in probs])
    d = float(spec.get("d", 0.001))
    E, it, act, last = B.muller(fun, g - d, g + d, g + 0j)
    rows = []
    for j, p in enumerate(probs):
        rows.append(dict(var=p["var"], match=p["match"], lam=p["lam"], Q=float(E[j].real), gamma=float(-2 * E[j].imag),
                         converged=bool((not act[j]) and np.isfinite(E[j])), center=float(cen[j]),
                         E_min=float(cen[j] + offs[jmin[j]]), edge=bool(jmin[j] in (0, len(offs) - 1)),
                         depth=float(av[jmin[j], j] / np.median(av[:, j])), frob=fr[j]))
    out = dict(name=spec.get("name"), cfg=cfg, rows=rows, muller_iter=int(it), time_s=time.time() - t0)
    X.wjson("job_%s.json" % spec.get("name"), out)
    return out


def fd_E0(var, lam_x=0.0, h=0.04, Rw=9.0, Sw=11.0):
    """lambda = 0 local (L0) 2D ground state of a variant (sparse FD) -> E0, <s^2>(r), rho(r)."""
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    M = B.Model()
    nr, ns = int(round(Rw / h)) - 1, int(round(Sw / h)) - 1
    r, s = h * np.arange(1, nr + 1), h * np.arange(1, ns + 1)
    base = var.split("|")[0]
    v2 = var if "|" not in var else base + "|" + "+".join(tags_of(var))
    U1, om2 = build2(M, v2, "L0", lam_x, r)
    Vext = U1[:, None] + 0.5 * om2[:, None] * s[None, :] ** 2
    if "YL" in tags_of(var):
        Vext = Vext + V_light(r, s, lam_x)

    def lap(n):
        return sparse.diags([np.full(n - 1, -0.5 / h ** 2), np.full(n, 1.0 / h ** 2), np.full(n - 1, -0.5 / h ** 2)],
                            [-1, 0, 1])
    H = (sparse.kron(lap(nr), sparse.identity(ns)) + sparse.kron(sparse.identity(nr), lap(ns))
         + sparse.diags(Vext.ravel())).tocsc()
    vals, vecs = eigsh(H, k=1, sigma=3.5, which="LM")
    p2 = vecs[:, 0].reshape(nr, ns) ** 2
    rho = p2.sum(1)
    s2c = (p2 * s[None, :] ** 2).sum(1) / np.maximum(rho, 1e-300)
    return dict(E0=float(vals[0]), r=r, rho=rho / rho.sum(), s2c=s2c, s2_mean=float((p2 * s[None, :] ** 2).sum() / p2.sum()))


def ref2d():
    with open(C140SUM, encoding="utf-8") as f:
        S = json.load(f)
    return S


def specs_for(lams, variants, name, S, E0s, extra_ref=None):
    probs = []
    for lam in lams:
        Q0 = S["Q"]["2D_ex"].get(f"{lam}") if extra_ref is None else extra_ref[lam][0]
        g0 = S["gamma"]["2D_ex"].get(f"{lam}") if extra_ref is None else extra_ref[lam][1]
        for v in variants:
            c = Q0 + (E0s[v] - E0s["2D_ex"])
            probs.append(dict(var=v, match="MC", lam=lam, center=c, gimag=g0))
    return dict(name=name, cfg=dict(N=32, Smax=14.0, h=0.002, Rout=16.0, order=2), window=(0.05, 0.0025), probs=probs)


VAR_C = ["2D_ex", "2D_ex|SU3", "2D_ex|SU2", "2D_ex|Yq", "2D_ex|SU3+SU2+Yq"]


def stageC():
    t0 = time.time()
    S = ref2d()
    E0s, s2 = {}, {}
    for v in VAR_C + ["2D_ex|Yb+YL", "2D_ex|SU3+SU2+Yb+YL"]:
        f = fd_E0(v)
        E0s[v], s2[v] = f["E0"], f["s2_mean"]
        print("FD", v, f["E0"], f["s2_mean"], flush=True)
    # first-order (lambda = 0 2D_ex density) shifts of each term
    base = fd_E0("2D_ex")
    sp = xterms(-0.010)
    w = base["rho"]
    fo = dict(SU3=float((w * sp["dU3"](base["r"])).sum()),
              SU2=float((w * sum(sp["EXx_" + g](base["r"]) for g in ("W", "H", "top", "light"))).sum()),
              Yq=float((w * 0.5 * sum(sp["OmX2_" + g](base["r"]) for g in ("W", "H", "top", "light")) * base["s2c"]).sum()))
    X.wjson("C149_stageC_fd.json", dict(E0=E0s, s2_mean=s2, first_order_lam0=fo, time_s=time.time() - t0))
    specs = [specs_for([-0.010], [v], "stageC_%d" % i, S, E0s) for i, v in enumerate(VAR_C)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(job2, specs))
    rows = {x["var"]: x for o in outs for x in o["rows"]}
    g0 = rows["2D_ex"]["gamma"]
    res = dict(E0=E0s, first_order_lam0=fo, rows=rows,
               ratio_vs_2Dex={v: rows[v]["gamma"] / g0 for v in rows},
               ln_ratio={v: math.log(rows[v]["gamma"] / g0) for v in rows},
               dQ={v: rows[v]["Q"] - rows["2D_ex"]["Q"] for v in rows},
               regression_2Dex_vs_D112=dict(Q=rows["2D_ex"]["Q"] - S["Q"]["2D_ex"]["-0.01"],
                                            gamma_rel=g0 / S["gamma"]["2D_ex"]["-0.01"] - 1))
    # indicative hazard translation with D112 slopes (dlnI/dm_t = 5.717 / 3.513 at P1 = 0.9 / 0.5)
    res["dmc_indicative_GeV"] = {v: {"P1=0.9": -res["ln_ratio"][v] / 5.717, "P1=0.5": -res["ln_ratio"][v] / 3.513}
                                 for v in rows}
    res["time_s"] = time.time() - t0
    X.wjson("C149_stageC.json", res)
    print(json.dumps(X.clean(res), indent=1), flush=True)


# ---------------------------------------------------------------------------------------------- stage D
LAM_D = [-0.008, -0.010, -0.012, -0.0143, -0.0165]
FULL = "2D_ex|SU3+SU2+Yb+YL"
VAR_D = ["2D_ex", FULL, "2D_ex|SU3v+SU2+Yb+YL", "2D_ex|SU3+SU2+Yq", "2D_ex|SU3", "2D_ex|SU2", "2D_ex|Yb+YL",
         "2D_ex|SU3+SU2z+Yz"]
VAR_D_SHORT = ["2D_ex", FULL, "2D_ex|SU3+SU2+Yq"]          # lambda = -0.012 / -0.0165 (bracketing only)
# D116 Sec. 0.2 (alpha_s = 0.1180, xi_H = 1/6, u <= 5): P1(m_t) and lambda_b at the critical points
D116_MT = [171.8, 171.9, 172.0, 172.1, 172.2, 172.3, 172.4, 172.5, 172.6, 172.7, 172.8, 172.9, 173.0, 173.1, 173.2,
           173.3, 173.4, 173.5, 173.6, 173.7, 173.8, 173.9, 174.0]
D116_P1 = [0.9997, 0.9990, 0.9970, 0.9927, 0.9842, 0.9691, 0.9446, 0.9078, 0.8565, 0.7895, 0.7075, 0.6136, 0.5127,
           0.4111, 0.3153, 0.2306, 0.1606, 0.1064, 0.0669, 0.040, 0.023, 0.012, 0.006]
D116_MC = {"P1=0.9": 172.517, "P1=0.5": 173.012, "P1=0.1": 173.514}
D116_LAMC = {"P1=0.9": -0.00965, "P1=0.5": -0.01291, "P1=0.1": -0.01624}


def zero_mass_xterms(r, lam):
    """r = 0 curvatures (no decoupling) with the r-dependent conditional widths -- bracket for the CCA mass dependence."""
    h0 = K.hessian(1e-6, lam)["tot"]
    prof = K.derived_profile()
    a, k = np.interp(r, prof["rf"], prof["a"]), np.interp(r, prof["rf"], prof["k"])
    beta = np.interp(r, prof["rf"], prof["beta"])
    return 1.5 * h0["H11"] / a + 0.75 * h0["H33"] / k, h0["Hww"] - 2 * beta * h0["H3w"] + beta ** 2 * h0["H33"]


_ORIG2 = build2


def build3(M, var, match, lam, r, tilt=0.0):
    tg = tags_of(var)
    if "SU2z" not in tg and "Yz" not in tg:
        return _ORIG2(M, var, match, lam, r, tilt)
    rest = [t for t in tg if t not in ("SU2z", "Yz")]
    v = var.split("|")[0] + ("|" + "+".join(rest) if rest else "")
    U1, om2 = _ORIG2(M, v, match, lam, r, tilt)
    ex, oy = zero_mass_xterms(r, lam)
    if "SU2z" in tg:
        U1 = U1 + ex
    if "Yz" in tg:
        om2 = om2 + oy
    return U1, om2


build2 = build3
B.build = build3


def stageD(which="main"):
    t0 = time.time()
    S = ref2d()
    allv = sorted(set(VAR_D + VAR_D_SHORT))
    p = os.path.join(X.RES, "C149_stageD_fd.json")
    E0s = json.load(open(p, encoding="utf-8"))["E0"] if os.path.exists(p) else {}
    for v in allv:
        if v not in E0s:
            E0s[v] = fd_E0(v)["E0"]
    X.wjson("C149_stageD_fd.json", dict(E0=E0s))
    for lam in LAM_D:                      # cache the cross-term splines before the worker pool starts
        xterms(lam)
    ref = {}
    for lam in LAM_D:
        if f"{lam}" in S["Q"]["2D_ex"]:
            ref[lam] = (S["Q"]["2D_ex"][f"{lam}"], S["gamma"]["2D_ex"][f"{lam}"])
        else:                                  # extrapolate (lambda = -0.0165)
            ref[lam] = (4.27375 + 3.19 * (lam + 0.0143), 3.7738e-3 * math.exp(-409.0 * (lam + 0.0143)))
    specs = []
    for lam in LAM_D:
        vs = VAR_D if lam in (-0.008, -0.010, -0.0143) else VAR_D_SHORT
        for v in vs:
            nm = "D_%g_%s" % (lam, v.replace("|", "_").replace("+", "_"))
            if os.path.exists(os.path.join(X.RES, "job_%s.json" % nm)):
                continue
            sp = specs_for([lam], [v], nm, S, E0s, extra_ref=ref)
            specs.append(sp)
    if which == "smax":
        specs = [dict(specs_for([-0.010], [FULL], "D_smax12", S, E0s, extra_ref=ref),
                      cfg=dict(N=28, Smax=12.0, h=0.002, Rout=16.0, order=2))]
    print("stageD jobs:", len(specs), flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for o in ex.map(job2, specs):
            x = o["rows"][0]
            print("%s Q=%.6f g=%.5e conv=%s edge=%s t=%.0fs" % (o["name"], x["Q"], x["gamma"], x["converged"], x["edge"],
                                                               o["time_s"]), flush=True)
    print("stageD done %.0fs" % (time.time() - t0), flush=True)


def post():
    """ratios vs 2D_ex, static lambda_c (eta = 299.84), hazard translation with the D116 table."""
    rows = {}
    for fn in os.listdir(X.RES):
        if fn.startswith("job_D_") and fn.endswith(".json"):
            o = json.load(open(os.path.join(X.RES, fn), encoding="utf-8"))
            for x in o["rows"]:
                rows[(x["var"], round(x["lam"], 6))] = x
    bad = [f"{k}" for k, x in rows.items() if not (x["converged"] and x["gamma"] > 0) or x["edge"]]
    lnI = CubicSpline(D116_MT, np.log(-np.log(np.array(D116_P1))))
    vars_ = sorted(set(k[0] for k in rows))
    out = dict(nonconverged_or_edge=bad, gamma={}, Q={}, ratio={}, lam_c_static={}, dmc_hazard={}, dmc_static={})
    eta = B.ETA
    targets = {"P1=0.9": -math.log(0.9) / eta, "P1=0.5": math.log(2) / eta, "P1=0.1": -math.log(0.1) / eta}
    for v in vars_:
        ls = sorted(l for (vv, l) in rows if vv == v)
        out["gamma"][v] = {f"{l}": rows[(v, l)]["gamma"] for l in ls}
        out["Q"][v] = {f"{l}": rows[(v, l)]["Q"] for l in ls}
        if v != "2D_ex":
            out["ratio"][v] = {f"{l}": rows[(v, l)]["gamma"] / rows[("2D_ex", l)]["gamma"] for l in ls
                               if ("2D_ex", l) in rows}
    # ln ratio interpolation in lambda (quadratic through available points) and hazard translation
    base_ls = sorted(l for (vv, l) in rows if vv == "2D_ex")
    lg0 = CubicSpline(base_ls, np.log([rows[("2D_ex", l)]["gamma"] for l in base_ls]))
    for v in out["ratio"]:
        ls = sorted(float(l) for l in out["ratio"][v])
        lr = np.log([out["ratio"][v][f"{l}"] for l in ls])
        deg = min(2, len(ls) - 1)
        cf = np.polyfit(ls, lr, deg)
        hz, st, lc = {}, {}, {}
        for kk in D116_MC:
            lb = D116_LAMC[kk]
            lnr = float(np.polyval(cf, lb))
            slope = float(lnI(D116_MC[kk], 1))
            hz[kk] = dict(lam_b=lb, ln_ratio=lnr, ratio=math.exp(lnr), dlnI_dmt=slope, dmc=-lnr / slope,
                          mc_new=D116_MC[kk] - lnr / slope, extrapolated=bool(lb < min(ls) or lb > max(ls)))
            # static: solve ln gamma_2Dex(l) + lnratio(l) = ln target
            from scipy.optimize import brentq
            try:
                l0 = brentq(lambda x: lg0(x) - math.log(targets[kk]), base_ls[0], base_ls[-1])
                l1 = brentq(lambda x: lg0(x) + np.polyval(cf, x) - math.log(targets[kk]), base_ls[0], base_ls[-1])
                lc[kk] = dict(lam_c_2Dex=l0, lam_c_new=l1, dlam_c=l1 - l0)
                st[kk] = (l1 - l0) / B.DLAM_DMT
            except ValueError:
                lc[kk], st[kk] = None, None
        out["dmc_hazard"][v], out["dmc_static"][v], out["lam_c_static"][v] = hz, st, lc
    X.wjson("C149_stageD_post.json", out)
    print(json.dumps(X.clean({k: out[k] for k in ("ratio", "dmc_hazard", "dmc_static", "nonconverged_or_edge")}),
                     indent=1), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "stageC":
        stageC()
    elif cmd == "stageD":
        stageD()
    elif cmd == "smax":
        stageD("smax")
    elif cmd == "post":
        post()
