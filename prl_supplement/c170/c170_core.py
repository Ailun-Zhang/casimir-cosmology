# -*- coding: utf-8 -*-
r"""C170 core: static-box calculations of the Letter (2026-10-19).

Static finite-volume box: flat three-torus of fixed edge L (Minkowski time, no gravity), AAA fermions, couplings at
mu = 1/L, held for T = N L; the Higgs zero mode survives with P = exp(-Gamma N), Gamma in units 1/L.

The model is the run05 production model of the PRL centre (D121 run05 = 2D_ex + DMC shape + one-loop MC matching +
the complete cross-term set), evaluated with EXPLICIT couplings (g2, gY, y_t, g3, lambda) instead of the node lookup
(m_t, alpha_s, u) of the production node job C149_direct.jobX ("lambda-override path").  Every physics function is the
production one, imported unchanged from the copied package modules:
    C150_twod.compute_profile / Parts / build2d / DMC / rgrid / SGrid / phases / window_roots / muller2 / FAST
    C149_direct5.cross_increments5 (installed as C149_direct.cross_increments), C149_direct.V_light_c / make_funW
    C138_core.dV('MC', ...), R14_res.jost / C150_twod.solve_res (1D adiabatic diagnostic only)
New here (not production physics): the kappa switch of the zero-mode couplings (S2), the coupling multipliers (S3),
seeds by continuation / 1D-adiabatic guidance, the effective radial potential (lowest adiabatic s-channel) with its
barrier top and WKB exponent, and the (28, 14) box check for the cross-term model (C150 rule; jobX has no box check).
Units: L = 1 (energies, widths in 1/L).
"""
import os
import sys
import json
import math
import time
import shutil
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
# [supplement] path block adapted for standalone use; physics unchanged
SUPP = os.path.dirname(HERE)                                        # supplement root
PKG = os.path.join(SUPP, "c170_modules")                            # filled by setup_from_release.py
REL = os.environ.get("DNEC_RELEASE", "")                            # code release v1.0.0 (read only)
OUT = os.environ.get("C170_OUT", os.path.join(SUPP, "rerun", "C170"))
WORK = os.path.join(OUT, "work")
ZM = os.path.join(PKG, "higgs", "zmlib")
XT = os.path.join(PKG, "higgs", "04_cross_terms")

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["DNEC_HIGGS_WORK"] = WORK
os.environ["C150_COUPLINGS"] = "module:" + os.path.join(ZM, "c150_nnlo_central.py")
os.environ["C149_V"] = "A"
for _v in ("DNEC_SMDR", "C150_PROVIDER_VARIANT", "C150_RUN"):
    os.environ.pop(_v, None)

# archived inputs: (source below REL, destination below WORK).  Production paths first, then reference-only copies.
STAGE = [
    ("higgs/data/r1/higgs_survival/run02_stage3/S3_prop.json", "r1/higgs_survival/run02_stage3/S3_prop.json"),
    ("higgs/03_zero_mode_2d/expected/r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
     "r2/higgs_beyond_bo/run01/C140_profile_v11.npz"),
    ("higgs/03_zero_mode_2d/expected/r2/higgs_beyond_bo/run01/C140_summary.json",
     "r2/higgs_beyond_bo/run01/C140_summary.json"),
    ("higgs/03_zero_mode_2d/expected/r2/higgs_beyond_bo/run01/C140_prep.json", "r2/higgs_beyond_bo/run01/C140_prep.json"),
    ("higgs/02_finite_volume_matching/expected/r2/higgs_matching/run01/C138_summary.json",
     "r2/higgs_matching/run01/C138_summary.json"),
    ("higgs/data/r2/higgs_final/run01/C144_ueff.json", "r2/higgs_final/run01/C144_ueff.json"),
    ("higgs/data/r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/C144_hazard.json"),
    ("higgs/data/r2/higgs_final/run01/hazard", "r2/higgs_final/run01/hazard"),
    ("higgs/03_zero_mode_2d/expected/r3/twod_direct_sm/run02/C150_summary.json", "r3/twod_direct_sm/run02/C150_summary.json"),
    ("higgs/03_zero_mode_2d/expected/r3/twod_direct_sm/run02/shards2d", "r3/twod_direct_sm/run02/shards2d"),
    ("higgs/04_cross_terms/expected/r3/holonomy_cross/run04/C149_top_chiral.json",
     "r3/holonomy_cross/run04/C149_top_chiral.json"),
    ("higgs/04_cross_terms/expected/r3/holonomy_cross/run04/C149_run04_summary.json",
     "r3/holonomy_cross/run04/C149_run04_summary.json"),
    # reference only (never on a production read/write path)
    ("higgs/04_cross_terms/expected/r3/holonomy_cross/run05", "archive/r3/holonomy_cross/run05"),
    ("higgs/03_zero_mode_2d/expected/r3/twod_direct_sm/run02/shards2d", "archive/r3/twod_direct_sm/run02/shards2d"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage_inputs():
    """copy archived inputs (if absent) and return a record {dest: [source, sha256]} (files) / file counts (dirs)."""
    rec = {}
    if not REL or not os.path.isdir(REL):
        raise SystemExit("set DNEC_RELEASE to the folder of the code release v1.0.0 (Zenodo 10.5281/zenodo.23434068)")
    for src, dst in STAGE:
        s, d = os.path.join(REL, src), os.path.join(WORK, dst)
        if os.path.isdir(s):
            n = 0
            for dd, _, files in os.walk(s):
                for fn in files:
                    if fn.endswith(".sha256"):
                        continue
                    a = os.path.join(dd, fn)
                    b = os.path.join(d, os.path.relpath(a, s))
                    if not os.path.exists(b):
                        os.makedirs(os.path.dirname(b), exist_ok=True)
                        shutil.copy2(a, b)
                    n += 1
            rec[dst] = dict(source="08_release/v12_code_data/" + src, n_files=n)
        else:
            if not os.path.exists(d):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
            rec[dst] = dict(source="08_release/v12_code_data/" + src, sha256=sha256(d), sha256_source=sha256(s))
    return rec


os.makedirs(WORK, exist_ok=True)
if not os.path.exists(os.path.join(WORK, "r3", "holonomy_cross", "run04", "C149_top_chiral.json")):
    stage_inputs()

sys.path.insert(0, XT)
sys.path.insert(1, ZM)
import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
import C149_direct5 as D5  # noqa: E402  (imports C149_direct4 -> C149_direct -> C150_twod, cross_*; installs run05)

D = D5.D
W, K, X, T, V, SP = D.W, D.K, D.X, D.T, D.V, D.SP
R, C = W.R, W.C
assert D.cross_increments is D5.cross_increments5, "run05 cross increments not installed"
PI = math.pi
FAST = dict(W.FAST)
BASE = dict(W.BASE)
WINDOW = tuple(D.WINDOW)            # (0.02, 0.0015) run05 production window
WIN_WIDE = tuple(W.WIN_WIDE)        # (0.05, 0.0025) D112 wide window (retry)
BOXCHK = dict(W.BOXCHK)             # (N, Smax) = (28, 14)
ETA = W.ETA                          # eta_inf = 299.84 (conformal duration of one branch)
MH = 125.13
DMC = W.DMC()
C170 = os.path.join(WORK, "c170")
PROFC = os.path.join(C170, "profiles")


def clean(o):
    return W.clean(o)


def wjson(p, o):
    W.wjson(p, o)


def jload(p):
    return W.jload(p)


# ---------------------------------------------------------------------------------------------- couplings
def couplings_at(mt, als, mu, mh=MH):
    """SM couplings (sm_provider 'central', the production NNLO mapping) at scale mu [GeV]."""
    c = SP.coup_point(mt, als, mh, mu, "central")
    return dict(g2=float(c["g2"]), gY=float(c["gY"]), yt=float(c["yt"]), lam=float(c["lam"]), g3=float(c["g3"]))


def mu_node(u):
    return W.mu_node(u)


# ---------------------------------------------------------------------------------------------- profiles
def profile_parts(g2z, gYz):
    """conditional-Gaussian gauge profile (D108 functional, C150_twod.compute_profile) for the ZERO-MODE couplings
    (g2z, gYz); cached as npz with the production keep set; returns (C150_twod.Parts, info)."""
    key = "%.17g|%.17g" % (g2z, gYz)
    fn = os.path.join(PROFC, "prof_%s.npz" % hashlib.sha1(key.encode()).hexdigest()[:16])
    t0 = time.time()
    if not os.path.exists(fn):
        d0 = W.compute_profile(float(g2z), float(gYz))
        os.makedirs(PROFC, exist_ok=True)
        tmp = fn[:-4] + "_tmp%d.npz" % os.getpid()
        np.savez(tmp, key=np.array(key), **{k: np.asarray(d0[k]) for k in
                                           ("rf", "C0", "Bx", "Om2e", "EtBr", "Bw", "g2", "gY", "chk_Et", "chk_Br",
                                            "maxdec", "min_hess_eig")})
        os.replace(tmp, fn)
    d = dict(np.load(fn))
    return (W.Parts(d["rf"], d["C0"], d["Bx"], d["Om2e"], d["EtBr"]),
            dict(file=os.path.basename(fn), g2z=float(d["g2"]), gYz=float(d["gY"]), chk_Et=float(d["chk_Et"]),
                 maxdec=float(d["maxdec"]), min_hess_eig=float(d["min_hess_eig"]), t_prof=time.time() - t0))


# ---------------------------------------------------------------------------------------------- model
MODELS = ("final", "2D_ex", "2D_G", "2D_Gdmcs")


def newton_safe(fun, p0, iters=8):
    """verbatim copy of 10_universality/U156_coup.newton_safe (D128): cross_var.newton with a Levenberg shift
    1e-10*max|diag| on point-wise singular finite-difference Hessians; regular points are unchanged.  Used only as a
    fallback if the production cross_var.newton returns non-finite values (recorded in info)."""
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
        M = np.moveaxis(Hs, (0, 1), (-2, -1)).copy()
        dt = np.abs(np.linalg.det(M))
        sc = np.max(np.abs(np.diagonal(M, axis1=-2, axis2=-1)), axis=-1)
        bad = ~(dt > 1e-30 * np.maximum(sc, 1e-300) ** n) | ~np.isfinite(dt)
        if bad.any():
            M[bad] += (1e-10 * np.maximum(sc[bad], 1e-12))[:, None, None] * np.eye(n)[None]
        dp = np.linalg.solve(M, np.moveaxis(g, 0, -1)[..., None])[..., 0]
        dp[~np.isfinite(dp)] = 0.0
        p = p - np.moveaxis(dp, -1, 0)
    return p, np.max(np.abs(g), axis=0)


def build_point(pt, r2):
    """pt: dict(model, g2, gY, yt, g3, lam[, kappa]).  Returns U1 (n,), OM (n,), wl(s)->(n, N) or None, info.

    final   : run05 = C149_direct.jobX column (2D_ex base + cross_increments5 + exact light-fermion twist)
    2D_ex   : T14b 2D model with DMC shape, no cross terms (C150_twod.build2d '2D_ex')
    2D_G    : same without the DMC shape (build2d '2D_G')
    2D_Gdmcs: 2D_G + kappa^(2/3) dE_DMC(kappa^(1/3) r)   (S2 sensitivity variant only)
    kappa multiplies (g2, gY) ONLY in the conditional Gaussian (C0, B_x, Om_eff^2: kinetic+magnetic+mass terms of the
    gauge zero modes, hypercharge oscillator); dV_MC (matching) is always evaluated at the physical couplings."""
    model = pt["model"]
    kap = float(pt.get("kappa", 1.0))
    c = dict(g2=float(pt["g2"]), gY=float(pt["gY"]), yt=float(pt["yt"]), lam=float(pt["lam"]))
    Pp, pinfo = profile_parts(kap * c["g2"], kap * c["gY"])
    info = dict(profile=pinfo)
    wlf = None
    if model == "final":
        if kap != 1.0:
            raise ValueError("kappa switch is defined for the models without cross terms only")
        U0, OM0 = W.build2d(Pp, DMC, r2, c, "2D_ex")
        t1 = time.time()

        def _ok(inc_, xi_):
            return (all(np.all(np.isfinite(inc_[k](r2))) for k in inc_)
                    and np.isfinite(xi_["newton_grad_max_r_gt_0p05"]) and xi_["newton_grad_max_r_gt_0p05"] < 1e-6)
        try:
            inc, xinfo = D.cross_increments(c, float(pt["g3"]))
            okx = _ok(inc, xinfo)
        except np.linalg.LinAlgError:
            okx = False
        if not okx:
            # fallback 1: start the Newton re-minimisation of the D108 Gaussian functional at its exact optimum for
            # these couplings (R14_res.gauss_profile, same functional and rf grid) instead of the fixed-coupling v11
            # profile, which is too far away for couplings scaled by >~10% (the production Newton has no line search)
            gp = R.gauss_profile(c["g2"], c["gY"])
            pv = D.PV11
            assert np.array_equal(np.asarray(gp["rf"]), np.asarray(pv["rf"])), "rf grids differ"
            D.PV11 = {"rf": pv["rf"], "P": np.asarray(gp["P"], float)}
            try:
                inc, xinfo = D.cross_increments(c, float(pt["g3"]))
                okx = _ok(inc, xinfo)
            except np.linalg.LinAlgError:
                okx = False
            finally:
                D.PV11 = pv
            info["newton_start"] = "gauss_profile(g2, gY) optimum"
        if not okx:                                   # fallback 2 (D128 precedent): Levenberg-safe Newton
            orig = V.newton
            V.newton = newton_safe
            try:
                inc, xinfo = D.cross_increments(c, float(pt["g3"]))
            finally:
                V.newton = orig
            info["newton_safe_used"] = True
        info.update(xinfo, t_cross=time.time() - t1)
        rr2 = r2 * r2
        U1 = U0 + inc["dC0"](r2) / rr2 + inc["dBx"](r2) / rr2 + inc["dU3"](r2)
        OM = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
        bet = inc["beta"](r2)
        g2, gY = c["g2"], c["gY"]

        def wlf(s):
            return D.V_light_c(r2, s, bet, g2, gY)
        bad = not (np.all(np.isfinite(U1)) and np.all(np.isfinite(OM)) and np.all(np.isfinite(bet)))
        info["nonfinite_model"] = bool(bad)
    elif model in ("2D_ex", "2D_G"):
        U1, OM = W.build2d(Pp, DMC, r2, c, model)
    elif model == "2D_Gdmcs":
        U1, OM = W.build2d(Pp, DMC, r2, c, "2D_G")
        U1 = U1 + kap ** (2.0 / 3.0) * DMC(kap ** (1.0 / 3.0) * r2)
    else:
        raise ValueError(model)
    if pt.get("dv", "phys") == "none":                       # scaling check only: matching switched off
        U1 = U1 - C.dV("MC", r2, c["lam"], c["g2"], c["gY"], c["yt"])
    return U1, OM, wlf, info


def eps0_channel(OM, Wl, sg):
    """lowest eigenvalue of the s-channel Hamiltonian T_s + Om^2 s^2/2 (+ W_L(r, s)) at every r (sine DVR, as bc2)."""
    n, N = len(OM), sg.N
    H = np.broadcast_to(sg.T, (n, N, N)).copy()
    idx = np.arange(N)
    H[:, idx, idx] += 0.5 * OM[:, None] * sg.s2[None, :] + (Wl if Wl is not None else 0.0)
    return np.linalg.eigvalsh(H)[:, 0]


def barrier_info(r, U, E=None):
    """well (first local minimum for r > 0.2) and barrier top (first local maximum after it) of the effective radial
    potential U(r); WKB exponent S = 2 int sqrt(2 (U - E)) dr across the barrier at energy E."""
    sel = np.where(r > 0.2)[0]
    rr, uu = r[sel], U[sel]
    d = np.diff(uu)
    mins = [i for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]]
    out = dict(has_well=bool(mins))
    if not mins:
        out.update(has_barrier=False)
        return out
    iw = mins[0]
    maxs = [i for i in range(iw + 1, len(d)) if d[i - 1] > 0 >= d[i]]
    out.update(r_well=float(rr[iw]), U_well=float(uu[iw]))
    if not maxs:
        out.update(has_barrier=False, U_end=float(uu[-1]))
        return out
    it = maxs[0]
    out.update(has_barrier=True, r_top=float(rr[it]), U_top=float(uu[it]), height=float(uu[it] - uu[iw]))
    if E is not None:
        E = float(E)
        out["E_minus_Utop"] = float(E - uu[it])
        if E >= uu[it]:
            out.update(S_wkb=0.0, r_in=None, r_out=None)
        else:
            i0 = it
            while i0 > 0 and uu[i0 - 1] > E:
                i0 -= 1
            i1 = it
            while i1 < len(uu) - 1 and uu[i1 + 1] > E:
                i1 += 1
            seg = slice(i0, i1 + 1)
            q = np.sqrt(np.maximum(2.0 * (uu[seg] - E), 0.0))
            out.update(S_wkb=float(2.0 * np.trapezoid(q, rr[seg])), r_in=float(rr[i0]), r_out=float(rr[i1]),
                       r_out_at_end=bool(i1 == len(uu) - 1))
    return out


def root1d(U, r, m, h, Elo, Ehi, step=0.002, guess=None):
    """1D adiabatic Siegert root of U(r) (R14 Numerov + 2nd-order WKB outgoing condition, C150_twod.solve_res):
    detrended deepest real-axis |f| minimum in [Elo, Ehi] (or Muller from guess), then Muller.  Diagnostic/seed only."""
    s, al = W.frob(U, r)
    if guess is None:
        Es = np.arange(Elo, Ehi + 0.5 * step, step)
        if len(Es) < 5:
            return None
        Ut = np.repeat(U[:, None], len(Es), axis=1)
        f = np.abs(R.jost(Es + 0j, Ut, r, m, np.full(len(Es), s), np.full(len(Es), al), h)[0])
        lv = np.log(f + 1e-300)
        y = lv - np.polyval(np.polyfit(Es, lv, 1), Es)
        loc = [i for i in range(1, len(Es) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]
        if not loc:
            return None
        i = loc[0]                    # lowest real-axis zero region = ground quasi-bound level (deep wells hold more)
        g = complex(Es[i], -5e-4)
    else:
        g = complex(guess)
    E, rs = W.solve_res([U], r, m, np.array([g]), h=h, spread=0.002)
    E = complex(E[0])
    if not np.isfinite(E):
        return None
    return dict(E=[E.real, E.imag], eps=E.real, gamma=-2 * E.imag, resid=float(rs[0]))


# ---------------------------------------------------------------------------------------------- one point
def solve_point(pt, center=None, gim=None, cfg=None, window=None, boxcheck=True, wide_retry=True, with1d=True,
                prev=None, boxchk=None):
    """Siegert root of the 2D model at one point with explicit couplings (lambda-override path).
    center/gim: explicit production-style seed (window rule around center).  Otherwise the seed is the 1D adiabatic
    root of the same model, shifted by (dQ, rho) = (eps2D - eps1D, Gamma2D/Gamma1D) of the neighbouring solved point
    'prev' (default (0, 1))."""
    t0 = time.time()
    cfg = dict(FAST if cfg is None else cfg)
    window = tuple(WINDOW if window is None else window)
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = W.rgrid(Rout, h)
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    U1, OM, wlf, info = build_point(pt, r2)
    t_build = time.time() - t0
    Wl = wlf(sg.s) if wlf else None
    Ueff = U1 + eps0_channel(OM, Wl, sg)
    bar = barrier_info(r2, Ueff)
    rec = dict(pt=dict(pt), cfg=cfg, barrier=bar, info=info)
    if not bar.get("has_barrier") and center is None:
        rec.update(status="no_barrier", time_s=time.time() - t0, t_build=t_build)   # no quasi-bound state to find
        return rec
    # 1D adiabatic root (seed and diagnostic)
    r1d = None
    if with1d and bar.get("has_barrier"):
        lo = bar["U_well"] + 0.02
        hi = bar["U_top"] + 0.3
        g1, ref = None, None
        if center is not None:
            g1 = complex(center, -0.5 * (gim if gim else 1e-3))
        elif prev is not None and prev.get("eps1d") is not None:      # continuation: previous 1D root
            g1 = complex(prev["eps1d"], -0.5 * max(prev.get("gamma1d", 1e-3), 1e-12))
            ref = prev["eps1d"]
        try:
            r1d = root1d(Ueff, r2, m2, h, lo, hi, guess=g1)
            if r1d is not None and not (0 < r1d["gamma"] < 5 and lo - 0.1 < r1d["eps"] < hi + 0.5
                                        and (ref is None or abs(r1d["eps"] - ref) < 0.1)):
                r1d = root1d(Ueff, r2, m2, h, lo, hi)          # guess led astray: scan (lowest dip)
        except Exception as e:  # noqa: BLE001
            rec["root1d_error"] = repr(e)
            r1d = None
    rec["root1d"] = r1d
    seed_src = "explicit"
    if center is None:
        dq, rho = (prev or {}).get("dQ", 0.0), (prev or {}).get("rho", 1.0)
        if r1d is not None:
            center, gim = r1d["eps"] + dq, max(r1d["gamma"] * rho, 1e-14)
            seed_src = "1D+shift"
        elif prev is not None and prev.get("eps") is not None:
            center, gim = prev["eps"], prev["gamma"]
            seed_src = "prev"
        else:
            # no real-axis zero of the 1D adiabatic Jost function between the well bottom and U_top + 0.3:
            # no quasi-bound level of the adiabatic potential (not a narrow resonance)
            rec.update(status="no_level", time_s=time.time() - t0, t_build=t_build)
            return rec
    rec["seed"] = dict(center=float(center), gim=float(gim), src=seed_src)
    ph0, ph1, s_, al_ = W.phases(U1, r2)
    U1a, OMa = U1[None, :], OM[None, :]
    fun, cnt = D.make_funW(U1a, OMa, [Wl], r2, m2, h, sg, int(cfg["order"]), np.array([ph0]), np.array([ph1]))
    cen, gg = np.array([float(center)]), np.array([float(gim)])
    E, wd = W.window_roots(fun, cen, gg, window)
    nev = cnt["n"]
    E0 = complex(E[0])
    edge = bool(abs(wd["Emin"][0] - cen[0]) >= window[0] - 1e-9)
    nc, noloc = bool(wd["act"][0]), bool(wd["noloc"][0])
    gam = -2 * E0.imag
    flags = [f for f, b in (("edge", edge), ("muller_nc", nc), ("no_local_min", noloc),
                            ("gamma<=0", not gam > 0), ("nan", not np.isfinite(E0))) if b]
    used_win, retried, first = window, False, list(flags)
    if flags and wide_retry:
        fun2, cnt2 = D.make_funW(U1a, OMa, [Wl], r2, m2, h, sg, int(cfg["order"]), np.array([ph0]), np.array([ph1]))
        E2, wd2 = W.window_roots(fun2, cen, gg, WIN_WIDE)
        nev += cnt2["n"]
        E2c = complex(E2[0])
        g2_ = -2 * E2c.imag
        edge2 = bool(abs(wd2["Emin"][0] - cen[0]) >= WIN_WIDE[0] - 1e-9)
        fl2 = [f for f, b in (("edge", edge2), ("muller_nc", bool(wd2["act"][0])),
                              ("no_local_min", bool(wd2["noloc"][0])), ("gamma<=0", not g2_ > 0),
                              ("nan", not np.isfinite(E2c))) if b]
        retried = True
        if len(fl2) <= len(flags):
            E0, flags, wd, used_win = E2c, fl2, wd2, WIN_WIDE
    gam = -2 * E0.imag
    rec.update(E=[E0.real, E0.imag], eps=E0.real, gamma=gam, flags=flags, flags_first=first, retried=retried,
               window=list(used_win), Emin=float(wd["Emin"][0]), muller_it=int(wd["it"]), n_eval=nev,
               frob=[float(s_), float(al_)])
    # box check (C150 rule): (N, Smax) = (28, 14), light-fermion term rebuilt on the new s grid
    if boxcheck and np.isfinite(E0):
        bxc = dict(BOXCHK if boxchk is None else boxchk)
        rec["boxchk"] = bxc
        sgb = W.SGrid(bxc["N"], bxc["Smax"])
        Wlb = wlf(sgb.s) if wlf else None
        funb, cntb = D.make_funW(U1a, OMa, [Wlb], r2, m2, h, sgb, int(cfg["order"]), np.array([ph0]),
                                 np.array([ph1]))
        Eb, itb, actb, _ = W.muller2(funb, np.array([E0 - 1e-4]), np.array([E0 + 1e-4]), np.array([E0 + 0j]), maxit=25)
        Eb = complex(Eb[0])
        dEb = abs(Eb - E0) if np.isfinite(Eb) else float("inf")
        rec.update(box_E=[Eb.real, Eb.imag], box_dE=float(dEb), box_nc=bool(actb[0]),
                   box_rel_gamma=float((-2 * Eb.imag) / gam - 1) if gam > 0 and np.isfinite(Eb) else None)
        if (not np.isfinite(Eb)) or bool(actb[0]) or dEb > max(2e-7, 2e-3 * max(gam, 0.0)):
            rec["flags"] = rec["flags"] + ["box_sensitive"]
    bar2 = barrier_info(r2, Ueff, E0.real if np.isfinite(E0) else None)
    rec["barrier"] = bar2
    if r1d is not None and np.isfinite(E0):
        rec["dQ_2D_1D"] = float(E0.real - r1d["eps"])
        rec["rho_2D_1D"] = float(gam / r1d["gamma"]) if r1d["gamma"] > 0 else None
    rec["status"] = classify(rec)
    rec["time_s"] = time.time() - t0
    rec["t_build"] = t_build
    return rec


def classify(rec):
    """narrow-resonance status of a solved point:
    ok         : Siegert root found by the production window rule without any flag, below the adiabatic barrier top;
    ok_broad   : the only flags are window-rule flags (no interior |f| minimum / minimum at the window edge, which
                 happens when Gamma is comparable to the window), and the root passes the (28, 14) box check and agrees
                 with the 1D adiabatic root of the same model (|ln Gamma2D/Gamma1D| < 0.25, |eps2D - eps1D| < 0.02);
    any root   : must be consistent with the 1D adiabatic root (|ln Gamma2D/Gamma1D| < 0.5, |eps2D - eps1D| < 0.03),
                 otherwise 'flagged' (rejects roots of other states, e.g. a narrow spurious root at kappa = 0.45);
    above_top  : eps >= U_top - Gamma/2 (no narrow resonance: the level reaches the top of the adiabatic barrier);
    no_barrier : the effective radial potential has no local maximum after the well;
    flagged / error / no_seed : otherwise (not used)."""
    if rec.get("status") in ("error", "no_seed", "no_level", "no_barrier") or rec.get("eps") is None:
        return rec.get("status", "error")
    bar = rec.get("barrier") or {}
    E, gam = rec["eps"], rec["gamma"]
    if not bar.get("has_barrier"):
        return "no_barrier"
    fl = rec.get("flags") or []
    r1 = rec.get("root1d") or {}
    verified = (not fl) or (set(fl) <= {"no_local_min", "edge"} and r1.get("gamma") and gam and gam > 0
                            and abs(math.log(gam / r1["gamma"])) < 0.25 and abs(E - r1["eps"]) < 0.02)
    if not verified:
        return "flagged"
    # every root must belong to the resonance of the adiabatic channel: consistent with the 1D adiabatic root
    # (2D/1D: final model 1.01-1.02 and dQ ~ +0.004; no-cross models 1.05-1.2 and dQ ~ +0.01)
    if r1.get("gamma") and (not gam or gam <= 0 or abs(math.log(gam / r1["gamma"])) > 0.5 or abs(E - r1["eps"]) > 0.03):
        return "flagged"
    if E >= bar["U_top"] - 0.5 * max(gam or 0.0, 0.0):
        return "above_top"
    return "ok" if not fl else "ok_broad"


GOOD = ("ok", "ok_broad")


def lam_of_x(x):
    return -1.0 / (x * x)


def x_of_lam(lam):
    return abs(lam) ** -0.5 if lam != 0 else float("inf")


# ---------------------------------------------------------------------------------------------- chains
def run_chain(spec):
    """continuation in lambda: spec = dict(name, base (point without lam), lams (ordered outward from the anchor),
    anchor (optional dict(eps, gamma) at lams[0]), stop_after_fail (int), cfg, outdir).  Resumes from its JSON."""
    out_fn = os.path.join(spec["outdir"], spec["name"] + ".json")
    log_fn = os.path.join(OUT, "logs", spec["name"] + ".log")
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    os.makedirs(os.path.dirname(log_fn), exist_ok=True)
    res = jload(out_fn) if os.path.exists(out_fn) else dict(name=spec["name"], base=spec["base"], points=[])
    done = {round(p["pt"]["lam"], 12): p for p in res["points"]}
    nfail = 0
    prev = None
    t0 = time.time()
    for i, lam in enumerate(spec["lams"]):
        key = round(lam, 12)
        if key in done:
            p = done[key]
            p["status"] = classify(p)
        else:
            pt = dict(spec["base"], lam=float(lam))
            kw = {}
            if i == 0 and spec.get("anchor"):
                kw = dict(center=spec["anchor"]["eps"], gim=spec["anchor"]["gamma"])
            if i == 0 and not spec.get("anchor") and spec.get("prev0"):
                prev = dict(spec["prev0"])
            try:
                p = solve_point(pt, prev=prev, cfg=spec.get("cfg"), boxchk=spec.get("boxchk"), **kw)
            except Exception as e:  # noqa: BLE001
                p = dict(pt=pt, status="error", error=repr(e))
            res["points"].append(p)
            wjson(out_fn, res)
            with open(log_fn, "a", encoding="utf-8") as f:
                f.write("%s lam=%.6f x=%.4f status=%s eps=%s gamma=%s flags=%s 1D=%s Utop=%s S=%s seed=%s t=%.1fs\n" % (
                    time.strftime("%H:%M:%S"), lam, x_of_lam(lam), p.get("status"), p.get("eps"), p.get("gamma"),
                    p.get("flags"), (p.get("root1d") or {}).get("gamma"), (p.get("barrier") or {}).get("U_top"),
                    (p.get("barrier") or {}).get("S_wkb"), (p.get("seed") or {}).get("src"), p.get("time_s", 0.0)))
        if p.get("status") in GOOD:
            nfail = 0
            prev = dict(eps=p["eps"], gamma=p["gamma"], dQ=p.get("dQ_2D_1D", 0.0) or 0.0, rho=p.get("rho_2D_1D") or 1.0)
            if p.get("root1d"):
                prev.update(eps1d=p["root1d"]["eps"], gamma1d=p["root1d"]["gamma"])
            if spec.get("stop_gamma_below") is not None and p["gamma"] < spec["stop_gamma_below"]:
                break
            if spec.get("stop_gamma_above") is not None and p["gamma"] > spec["stop_gamma_above"]:
                break
        else:
            nfail += 1
            if nfail >= spec.get("stop_after_fail", 2):
                break
    res["time_s_total"] = res.get("time_s_total", 0.0) + (time.time() - t0)
    wjson(out_fn, res)
    return res


def prune_chain(fn, drop=("flagged", "error", "no_seed")):
    """remove points whose (re)classified status is in 'drop' (they are recomputed when the chain resumes);
    the removed records are appended to <fn>.pruned.json for the record."""
    d = jload(fn)
    keep, gone = [], []
    for p in d["points"]:
        p["status"] = classify(p)
        (gone if p["status"] in drop else keep).append(p)
    if gone:
        pf = fn[:-5] + ".pruned.json"
        old = jload(pf) if os.path.exists(pf) else []
        wjson(pf, old + gone)
        d["points"] = keep
        wjson(fn, d)
    return [(p["pt"]["lam"], p.get("flags")) for p in gone]


def status_update(upd):
    p = os.path.join(OUT, "STATUS.json")
    try:
        st = jload(p)
    except Exception:  # noqa: BLE001
        st = {}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(st.get(k), dict):
            st[k].update(v)
        else:
            st[k] = v
    st["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wjson(p, st)
    return st
