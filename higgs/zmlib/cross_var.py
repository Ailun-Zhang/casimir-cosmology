# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/cross_var.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] SU(2)/U(1) Gaussian functional re-minimised with the cross terms (Newton), SU(3)-top variational energy E(s).  Paper: PRD App. H2'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 stage D(b,c) -- VARIATIONAL treatment of the cross terms (holonomy-cross, D121 Sec. 3.3).

(b) SU(2)/U(1): the one-loop Hessian is added to the D108 Gaussian energy functional (body frame, per direction
    precision K = diag(a, a, [[k, c], [c, b]]), D = k b - c^2, covariance K^-1/2):
      F_SU2 = (3/2) H11/a + (3/4) H33/k                         (conditional x-part)
      F_Y   = (3/4)(Hww k - 2 c H3w + c^2 H33/k)/D = (3/4) Om_X^2/bt   (w-part, Om_X^2 = Hww - 2 beta H3w + beta^2 H33)
    E_tot(a,k,b,c; r) [D108 Sec. 1.2] + F is re-minimised at every r (vectorised Newton from the D108 optimum), and the
    2D model is rebuilt from the new (a', k', beta', bt'): C0' = C0(p') + F_SU2(p'), B_x', Om'^2_eff = Om^2(p') + k' beta''^2/2,
    w-potential 1/2 (Om'^2_eff + Om_X^2(beta')) s^2, light-fermion part replaced by the exact V_L(r, s; beta') ('YL').
(c) SU(3)-top: Gaussian functional E(s) = alpha/s + beta s^2 + kappa3(r) s/2 (s = <c^2>_total; alpha, beta fixed by
    E0, s0 at kappa = 0: Gaussian 8.603/3.66, virial 8.343/3.77); dU3var(r) = E*(kappa3(r)) - E*(kappa3(0)).
Usage: python -B cross_var.py prep | solve | all
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
import cross_2d as T  # noqa: E402

PI = math.pi
g2, gY = K.V11["g2"], K.V11["gY"]
VPATH = os.path.join(X.RES, "C149_var_profiles.npz")
E0_SU3 = {"gauss": (12.980 * 0.5396 ** (2 / 3), 3.66), "virial": (12.5868 * 0.5396 ** (2 / 3), 3.77)}


def etot(p, r):
    a, k, b, c = p
    D = k * b - c * c
    kin = 0.75 * (2 * a + k + b)
    mass = 3 * r * r / 16 * (2 * g2 * g2 / a + (g2 * g2 * b - 2 * g2 * gY * c + gY * gY * k) / D)
    ym = 3 * g2 * g2 / (4 * a * a) + 3 * g2 * g2 * b / (2 * a * D)
    ang = 3.0 / (a * r * r) * (b * (k - a) ** 2 - 2 * c * c * (k - a) + k * c * c) / D
    return kin + mass + ym + ang


def fcross(p, H, which):
    a, k, b, c = p
    D = k * b - c * c
    out = 0.0
    if "SU2" in which:
        out = out + 1.5 * H["H11"] / a + 0.75 * H["H33"] / k
    if "Y" in which:
        out = out + 0.75 * (H["Hww"] * k - 2 * c * H["H3w"] + c * c * H["H33"] / k) / D
    return out


def newton(fun, p0, iters=8):
    p = p0.copy()
    n = p.shape[0]
    for it in range(iters):
        h = 1e-4 * np.maximum(np.abs(p), 1e-3)
        f0 = fun(p)
        g = np.empty_like(p)
        Hs = np.empty((n, n) + p.shape[1:])
        fp, fm = [], []
        for i in range(n):
            e = np.zeros_like(p)
            e[i] = h[i]
            fp.append(fun(p + e))
            fm.append(fun(p - e))
            g[i] = (fp[i] - fm[i]) / (2 * h[i])
            Hs[i, i] = (fp[i] - 2 * f0 + fm[i]) / h[i] ** 2
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros_like(p)
                ej = np.zeros_like(p)
                ei[i] = h[i]
                ej[j] = h[j]
                v = (fun(p + ei + ej) - fun(p + ei - ej) - fun(p - ei + ej) + fun(p - ei - ej)) / (4 * h[i] * h[j])
                Hs[i, j] = Hs[j, i] = v
        dp = np.linalg.solve(np.moveaxis(Hs, (0, 1), (-2, -1)), np.moveaxis(g, 0, -1)[..., None])[..., 0]
        p = p - np.moveaxis(dp, -1, 0)
    return p, np.max(np.abs(g), axis=0)


def derived_from(rf, P):
    a, k, b, c = P
    dP = np.gradient(P, rf, axis=1, edge_order=2)
    q = T.B.derived(rf, P, dP, g2, gY)
    return q


def su3_var(kap, E0, s0):
    beta_ = E0 / (3 * s0 * s0)
    alpha = 2 * E0 * s0 / 3
    s = np.full_like(kap, s0)
    for _ in range(60):                       # Newton on dE/ds = -alpha/s^2 + 2 beta s + kappa/2
        f = -alpha / s ** 2 + 2 * beta_ * s + kap / 2
        fp = 2 * alpha / s ** 3 + 2 * beta_
        s = s - f / fp
    return alpha / s + beta_ * s * s + kap * s / 2, s


def prep():
    t0 = time.time()
    d = np.load(K.PROFILE)
    rf, P0, Et = d["rf"], d["P"].astype(float), d["Et"]
    chk = float(np.max(np.abs(etot(P0, rf) - Et)))
    # raw one-loop Hessians (lambda = -0.010) on a grid, splined to rf
    rg = T.RX
    rows = [K.hessian(float(x), -0.010) for x in rg]
    Hsp = {}
    for g in ("W", "H", "top", "light"):
        for kk in ("H11", "H33", "H3w", "Hww"):
            Hsp[(g, kk)] = CubicSpline(rg, [x["parts"][g][kk] for x in rows])
    rr = np.minimum(rf, rg[-1])
    Hall = {kk: sum(Hsp[(g, kk)](rr) for g in ("W", "H", "top", "light")) for kk in ("H11", "H33", "H3w", "Hww")}
    Hlight = {kk: Hsp[("light", kk)](rr) for kk in ("H11", "H33", "H3w", "Hww")}
    out = dict(rf=rf, check_etot_vs_profile=chk)
    # gradient of E_tot at the D108 optimum (stationarity check of the transcription)
    _, g0 = newton(lambda p: etot(p, rf), P0, iters=1)
    out["grad_at_D108_opt_max"] = float(np.max(g0))
    prof = {}
    for nm, which in (("all", "SU2+Y"), ("SU2", "SU2"), ("Y", "Y")):
        P1, gres = newton(lambda p: etot(p, rf) + fcross(p, Hall, which), P0)
        q = derived_from(rf, P1)
        dE = etot(P1, rf) + fcross(P1, Hall, which) - Et
        # first-order prediction of the BO energy change, for comparison
        dE1 = fcross(P0, Hall, which)
        prof[nm] = dict(P=P1, gres=float(np.max(gres)), dE=dE, dE1=dE1, q=q)
        OmX2 = lambda HH, qq: HH["Hww"] - 2 * qq["beta"] * HH["H3w"] + qq["beta"] ** 2 * HH["H33"]  # noqa: E731
        sav = {f"{nm}|P": P1, f"{nm}|C0": q["C0"] + (fcross(P1, Hall, "SU2") if "SU2" in which else 0.0),
               f"{nm}|Bx": q["Bx"], f"{nm}|Om2e": q["Om2e"], f"{nm}|beta": q["beta"],
               f"{nm}|OmX2_all": OmX2(Hall, q) if "Y" in which else 0 * rf,
               f"{nm}|OmX2_light": OmX2(Hlight, q) if "Y" in which else 0 * rf}
        out.update(sav)
        print(nm, "newton |grad| max", prof[nm]["gres"], "dE(BO) at r=1/3/5:",
              [float(np.interp(x, rf, dE)) for x in (1.0, 3.0, 5.0)], "first-order:",
              [float(np.interp(x, rf, dE1)) for x in (1.0, 3.0, 5.0)], flush=True)
    # SU(3)-top variational
    kap = np.array([x["kappa3"] for x in rows])
    for tag, (E0, s0) in E0_SU3.items():
        Ev, sv = su3_var(kap, E0, s0)
        out[f"dU3var_{tag}"] = np.interp(rr, rg, Ev - Ev[0] + 0 * Ev) - 0.0
        out[f"s3_{tag}"] = np.interp(rr, rg, sv)
        k0 = (6 * X.curv(0.0, "P")[0] - 12 * X.curv(0.0, "A")[0]) / (4 * PI ** 2)
        Ev0, sv0 = su3_var(np.array([k0]), E0, s0)
        out[f"dU3var_{tag}"] = np.interp(rr, rg, Ev - Ev0[0])
        out[f"dU3first_{tag}"] = np.interp(rr, rg, 0.5 * (kap - k0) * s0)
        out[f"s3_at_kappa0_{tag}"] = float(sv0[0])
    np.savez(VPATH, **{k_: v for k_, v in out.items() if isinstance(v, np.ndarray)})
    summ = dict(check_etot_vs_profile=chk, grad_at_D108_opt_max=out["grad_at_D108_opt_max"],
                newton_grad={nm: prof[nm]["gres"] for nm in prof},
                table={nm: {f"{x}": dict(dE_BO_var=float(np.interp(x, rf, prof[nm]["dE"])),
                                         dE_BO_first=float(np.interp(x, rf, prof[nm]["dE1"])),
                                         a=float(np.interp(x, rf, prof[nm]["P"][0])), a0=float(np.interp(x, rf, P0[0])),
                                         beta=float(np.interp(x, rf, prof[nm]["q"]["beta"])),
                                         beta0=float(np.interp(x, rf, P0[3] / P0[1])),
                                         bt=float(np.interp(x, rf, prof[nm]["q"]["bt"])))
                            for x in (0.3, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)} for nm in prof},
                su3={tag: dict(s_at_kappa0=out[f"s3_at_kappa0_{tag}"],
                               dU3var={f"{x}": float(np.interp(x, rf, out[f"dU3var_{tag}"])) for x in (1.0, 2.0, 3.0, 4.0, 5.0)},
                               dU3first={f"{x}": float(np.interp(x, rf, out[f"dU3first_{tag}"])) for x in (1.0, 2.0, 3.0, 4.0, 5.0)})
                     for tag in E0_SU3}, time_s=time.time() - t0)
    X.wjson("C149_var_prep.json", summ)
    print(json.dumps(X.clean(summ), indent=1)[:6000], flush=True)


_VP = {}


def vp():
    if "d" not in _VP:
        d = dict(np.load(VPATH))
        rf = np.load(K.PROFILE)["rf"]
        _VP["sp"] = {k_: CubicSpline(rf, v) for k_, v in d.items() if v.ndim == 1 and len(v) == len(rf)}
        _VP["d"] = d
    return _VP["sp"]


_ORIG_B2 = T.build2
_ORIG_VL = T.V_light


def build_var(M, var, match, lam, r, tilt=0.0):
    if not var.startswith("VAR"):
        tg = T.tags_of(var)
        rest = [t for t in tg if t not in ("SU3var", "SU3varv")]
        v = var.split("|")[0] + ("|" + "+".join(rest) if rest else "")
        U1, om2 = _ORIG_B2(M, v, match, lam, r, tilt)
        if "SU3var" in tg:
            U1 = U1 + vp()["dU3var_gauss"](r)
        if "SU3varv" in tg:
            U1 = U1 + vp()["dU3var_virial"](r)
        return U1, om2
    prof = var.split("|")[0].split("_")[1]           # VAR_all / VAR_SU2 / VAR_Y
    tg = T.tags_of(var)
    sp = vp()
    U1, om2 = T.ORIG_BUILD(M, "2D_ex", match, lam, r, tilt)
    r2 = r * r
    U1 = U1 - M.s_c0(r) / r2 - M.s_bx(r) / r2 + sp[f"{prof}|C0"](r) + sp[f"{prof}|Bx"](r)
    om2 = sp[f"{prof}|Om2e"](r)
    if prof in ("all", "Y"):
        om2 = om2 + sp[f"{prof}|OmX2_all"](r) - (sp[f"{prof}|OmX2_light"](r) if "YL" in tg else 0.0)
    for t in tg:
        if t == "SU3var":
            U1 = U1 + sp["dU3var_gauss"](r)
        elif t == "SU3varv":
            U1 = U1 + sp["dU3var_virial"](r)
        elif t == "SU3":
            U1 = U1 + sp["dU3first_gauss"](r)
    return U1, om2


def vlight_var(r, s, lam):
    if not _CUR.get("var", "").startswith("VAR"):
        return _ORIG_VL(r, s, lam)
    prof = _CUR["var"].split("|")[0].split("_")[1]
    f, E0 = T.ebar_A()
    beta = vp()[f"{prof}|beta"](r)
    out = np.zeros((len(r), len(s)))
    for nm, mult, t3, y in K.LIGHT:
        q = np.abs(gY * y - g2 * beta * t3)
        out += -2.0 * mult * (f(np.minimum(q[:, None] * s[None, :] / (2 * PI), 2.5)) - E0)
    return out


_CUR = {}
T.build2 = build_var
T.V_light = vlight_var


def job_var(spec):
    _CUR["var"] = spec["probs"][0]["var"]            # one problem per spec
    return T.job2(spec)


LAM3 = [-0.008, -0.010, -0.0143]
VARS = [("VAR_all|SU3var+YL", LAM3 + [-0.012, -0.0165]), ("VAR_all|SU3var+Yq", LAM3), ("VAR_SU2", LAM3),
        ("VAR_Y|YL", LAM3), ("2D_ex|SU3var", LAM3), ("VAR_all|SU3varv+YL", [-0.010])]


def solve():
    t0 = time.time()
    S = T.ref2d()
    E0s = {}
    base = None
    for v, _ in VARS:
        _CUR["var"] = v
        E0s[v] = T.fd_E0(v)["E0"]
    _CUR["var"] = "2D_ex"
    base = T.fd_E0("2D_ex")["E0"]
    E0s["2D_ex"] = base
    X.wjson("C149_var_fd.json", dict(E0=E0s, shift={v: E0s[v] - base for v in E0s}))
    print("FD shifts", {v: round(E0s[v] - base, 5) for v in E0s}, flush=True)
    ref = {}
    for fn in os.listdir(X.RES):
        if fn.startswith("job_D_") and fn.endswith(".json"):
            o = json.load(open(os.path.join(X.RES, fn), encoding="utf-8"))
            for x in o["rows"]:
                if x["var"] == "2D_ex":
                    ref[round(x["lam"], 6)] = (x["Q"], x["gamma"])
    specs = []
    for v, ls in VARS:
        for lam in ls:
            nm = "D_%g_%s" % (lam, v.replace("|", "_").replace("+", "_"))
            if os.path.exists(os.path.join(X.RES, "job_%s.json" % nm)):
                continue
            specs.append(T.specs_for([lam], [v], nm, S, E0s, extra_ref={lam: ref[round(lam, 6)]}))
    print("var jobs:", len(specs), flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for o in ex.map(job_var, specs):
            x = o["rows"][0]
            print("%s Q=%.6f g=%.5e conv=%s edge=%s t=%.0fs" % (o["name"], x["Q"], x["gamma"], x["converged"], x["edge"],
                                                               o["time_s"]), flush=True)
    print("var solve done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("prep", "all"):
        prep()
    if cmd in ("solve", "all"):
        solve()
