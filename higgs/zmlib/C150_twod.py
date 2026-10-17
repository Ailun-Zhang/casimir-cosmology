# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C150_twod.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261014_claude_v12_r3/twod_direct_sm/C150_twod.py on 2026-10-14 by agent holonomy-cross (C149 run02).
# PROVENANCE: new module by agent twod-direct (C150/D122, v12 round 3) on 2026-10-14. Functions tagged [C140] are copied verbatim from 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/C140_beyond_bo.py; functions tagged [C144] verbatim from 04_calculations/code/20261013_claude_v12_r2/higgs_final/C144_model.py / C144_hazard.py (originals read-only, unchanged).
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Direct 2D hazard (D122/T14b): the 2D_ex model solved at each hazard node u with SM running couplings at mu(u) = mu_b/sqrt(cosh 2u); hazard I = int_0^5 gamma du/(hL_b sqrt(cosh 2u)), P1 = exp(-I); critical m_t by spline root of ln I.  Coupling source pluggable (env C150_COUPLINGS = 'R14' or 'module:<file>'); env C150_RUN selects the run directory (run01 = R14 mapping, run02 = NNLO mapping).  Paper: PRD App. H2-H3, convergence table. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C150 / D122 (task T14b, agent twod-direct): hazard P1(m_t) from the 2D (r,|w|) model 2D_ex (D112) solved DIRECTLY
at every hazard node with SM running couplings -- no 1D U_eff bump, no v11 "coupling transfer".

At node u (mu(u) = mu_b/sqrt(cosh 2u)) and column (m_t, alpha_s), with (lam, g2, gY, yt) of the coupling source:
    H_2D = -1/2 d2/dr2 + 3/(8r^2) - 1/2 Lap_w + C0(r) + B_x(r) + 1/2 Om_eff(r)^2 |w|^2 + lam r^4/4 + dV_MC(r) + dE_DMC(r)
C0, B_x, Om_eff^2: D112 Sec. 1.1 decomposition of the D108/R14 conditional Gaussian recomputed with (g2, gY) of the column
('pc'), or the C144 convention ('c144': (172.6, 0.1180) profile, read-only C144 files); dV_MC = C138 MC+CW (mu = 1/L);
dE_DMC = Stage-3 DMC shape spline (L units, the only piece still taken from v11).  2D Siegert resonance: sine-DVR in
s = |w|, matrix Numerov in r, 2nd-order WKB outgoing condition in the local adiabatic channels, D112 local-window rule
(|f| scan on a real window around a seed -> interior local minimum -> small-spacing Muller); seed = 1D U_eff root at the
same node and couplings; box-insensitivity check with (N, S_max) = (28, 14); flagged roots are retried with the wide
D112 window (0.05, 0.0025).  Hazard (C138/C144): I = int_0^5 gamma du/(hL_b sqrt(cosh 2u)), P1 = exp(-I).
Coupling source is pluggable: env C150_COUPLINGS = 'R14' (R14_rge 2-loop + 3-loop calibration constant; default) or
'module:<file.py>' (module with couplings(mt, als, mu) -> dict(g2, gY, yt, lam), e.g. the T1 NNLO mapping); env
C150_RUN selects the result sub-directory (default run01).
Usage: python -B C150_twod.py time | two main|G|als|n29|conv|c144 [nw] | oned [nw] | post
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import glob
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C138_core as C  # noqa: E402  (copied unchanged, provenance line added)

R = C.R
RGE = C.R14_rge

KIND = os.environ.get("C150_COUPLINGS", "R14")
RUN = os.environ.get("C150_RUN", "run01")
ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r3/twod_direct_sm"), RUN)
PROFDIR = os.path.join(RES, "profiles")
SH1 = os.path.join(RES, "shards1d")
SH2 = os.path.join(RES, "shards2d")
C144RES = os.path.join(ROOT, _zp.w("r2/higgs_final/run01"))
C144PROF = os.path.join(C144RES, "profiles")
S3DIR = os.path.join(ROOT, _zp.w("r1/higgs_survival/run02_stage3"))
PROF_V11 = os.path.join(ROOT, _zp.w("r2/higgs_beyond_bo/run01/C140_profile_v11.npz"))
C140PREP = os.path.join(ROOT, _zp.w("r2/higgs_beyond_bo/run01/C140_prep.json"))
C138SUM = os.path.join(ROOT, _zp.w("r2/higgs_matching/run01/C138_summary.json"))
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585)
HLB = 0.00437243
ETA = 1.31102877714606 / HLB
AS0, MH = 0.1180, 125.13
R0 = 0.01
UN29 = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0001, 0.125)]), 6)
UN11 = np.round(np.arange(0.0, 5.0001, 0.5), 6)
MT_MAIN = [round(172.0 + 0.2 * i, 2) for i in range(10)]                       # 172.0 ... 173.8
MT_G = [172.2, 172.4, 172.6, 172.8, 173.0, 173.2]
MT_AS = {0.1171: [171.8, 172.2, 172.6, 173.0, 173.4], 0.1189: [172.4, 172.8, 173.2, 173.6, 174.0]}
MT_FINE = [round(float(x), 2) for x in np.arange(171.6, 174.0001, 0.1)]
MT_FINE2 = [round(float(x), 2) for x in np.arange(171.6, 174.0001, 0.2)]
ALS = [0.1171, 0.1180, 0.1189]
TGT = {"P1=0.9": -math.log(0.9), "P1=0.5": math.log(2.0), "P1=0.1": math.log(10.0)}
FAST = dict(N=24, Smax=12.0, h=0.004, Rout=16.0, order=2)
BASE = dict(N=32, Smax=14.0, h=0.002, Rout=16.0, order=2)
BOXCHK = dict(N=28, Smax=14.0)
CONV = {"BASE_N32_S14_h0.002": BASE, "N24_S12_h0.002": dict(FAST, h=0.002), "N32_S16": dict(FAST, N=32, Smax=16.0),
        "Rout14": dict(FAST, Rout=14.0), "wkb0": dict(FAST, order=0)}
WIN = {"2D_ex": (0.012, 0.0012), "2D_G": (0.03, 0.0015)}
WIN_WIDE = (0.05, 0.0025)
DQ2D = {"2D_ex": -0.0035, "2D_G": -0.003}
# 1D seeds at u = 0 relative to the C144 U_eff root (dQ, gamma factor): v11 lam = -0.010 offsets (D112 Sec. 3.3)
OFF1D = {"Ueff": (0.0, 1.0), "UeffG": (0.053, 1.28), "lo": (-0.020, 0.93), "MC": (0.1197, 2.37), "1D_ex": (0.065, 1.88),
         "MU": (0.099, 4.9), "MU_CW": (0.105, 3.6)}
# T1 (C148/D120 Sec. 0 item 3): C144 pipeline (U_eff, C144 profiles, 29 nodes) with sm_provider 'central'
T1_C144PIPE = {"0.1180": {"P1=0.9": 172.580, "P1=0.5": 173.074, "P1=0.1": 173.575},
               "0.1171": {"P1=0.9": 172.233, "P1=0.5": 172.726, "P1=0.1": 173.226},
               "0.1189": {"P1=0.9": 172.928, "P1=0.5": 173.423, "P1=0.1": 173.924}}


# ------------------------------------------------------------------------------------------------ io
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, (complex, np.complexfloating)):
        return [clean(o.real), clean(o.imag)]
    return o


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def say(msg):
    print(msg, flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "C150_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def status(upd):
    p = os.path.join(RES, "STATUS.json")
    try:
        st = jload(p)
    except Exception:  # noqa: BLE001
        st = {}
    for k, v in upd.items():
        if k in ("done", "files") and isinstance(st.get(k), list):
            for x in v:
                if x not in st[k]:
                    st[k].append(x)
        elif isinstance(v, dict) and isinstance(st.get(k), dict):
            st[k].update(v)
        else:
            st[k] = v
    st["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wjson(p, st)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ couplings
class Couplings:
    """pluggable SM mapping.  'R14': R14_rge 2-loop RGE + 3-loop calibration constant on lambda (C138/C144);
    'module:<file.py>': module exposing couplings(mt, als, mu) -> dict(g2, gY, yt, lam) (e.g. T1 NNLO mapping)."""

    def __init__(self, kind="R14", cal=None):
        self.kind, self._sol, self.mod = kind, {}, None
        if kind == "R14":
            self.cal = float(cal) if cal is not None else RGE.calibration()[0]
        elif kind.startswith("module:"):
            import importlib.util
            path = kind.split(":", 1)[1]
            sp = importlib.util.spec_from_file_location("c150_coupling_source", path)
            self.mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(self.mod)
            self.cal = 0.0
        else:
            raise ValueError(kind)

    def __call__(self, mt, als, mu):
        if self.mod is not None:
            d = self.mod.couplings(mt, als, mu)
            return dict(g2=float(d["g2"]), gY=float(d["gY"]), yt=float(d["yt"]), lam=float(d["lam"]))
        k = (round(mt, 4), round(als, 5))
        if k not in self._sol:
            self._sol[k] = RGE.solve(mt, als, MH)
        y = self._sol[k].sol(np.log(mu))
        return dict(gY=float(y[0]), g2=float(y[1]), g3=float(y[2]), yt=float(y[3]), lam=float(y[4] + self.cal))


def mu_node(u):
    return RGE.MU_B / math.sqrt(math.cosh(2 * u))


def load_prm():
    return jload(os.path.join(C144RES, "C144_ueff.json"))["final"]["params"]


_G144 = {}


def c144_E(als, mt, u):
    """C144 U_eff root (stored hazard shard) at node u -- used ONLY as a root-search seed."""
    a = "%.4f" % als
    if a not in _G144:
        d = jload(os.path.join(C144RES, "hazard", "haz_as%s.json" % a))
        tab = {round(p[0], 2): np.array(d["E_re"][j]) + 1j * np.array(d["E_im"][j]) for j, p in enumerate(d["pairs"])}
        _G144[a] = (np.array(d["u"], float), tab)
    uu, tab = _G144[a]
    return complex(tab[round(mt, 2)][int(np.argmin(np.abs(uu - u)))])


# ------------------------------------------------------------------------------------------------ [C144] model pieces
def derived(rf, P, dP, g2, gY):
    """verbatim from C140_beyond_bo.derived (D112 Sec. 1.1)."""
    a, k, b, c = P
    ap, kp, bp, cp = dP
    r = rf
    beta = c / k
    betap = (cp * k - c * kp) / k ** 2
    D = k * b - c * c
    bt = D / k
    Dp = kp * b + k * bp - 2 * c * cp
    btp = (Dp * k - D * kp) / k ** 2
    Om2 = r * r / 4 * (gY - g2 * beta) ** 2 + 2 * g2 * g2 * beta ** 2 / a + 4 * a * beta ** 2 / (r * r)
    C0 = (0.75 * (2 * a + k) + 0.75 * k * beta ** 2 + 3.0 / 16 * r * r * (2 * g2 * g2 / a + g2 * g2 / k)
          + 0.75 * g2 * g2 / a ** 2 + 1.5 * g2 * g2 / (a * k) + 3 * (k - a) ** 2 / (a * k * r * r))
    Bx = 3.0 / 16 * (2 * (ap / a) ** 2 + (kp / k) ** 2)
    Bw = 3.0 / 16 * (btp / bt) ** 2
    Bb = 3 * k * betap ** 2 / (8 * bt)
    Om2e = Om2 + 0.5 * k * betap ** 2
    return dict(a=a, k=k, beta=beta, betap=betap, bt=bt, Om2=Om2, Om2e=Om2e, C0=C0, Bx=Bx, Bw=Bw, Bb=Bb)


def parts_from_P(rf, P, dP, Et, Br, g2, gY):
    q = derived(rf, P, dP, g2, gY)
    return dict(rf=rf, C0=q["C0"], Bx=q["Bx"], Om2e=q["Om2e"], Om2=q["Om2"], Bw=q["Bw"], Bb=q["Bb"], EtBr=Et + Br,
                Et=Et, Br=Br)


def compute_profile(g2, gY):
    gp = R.gauss_profile(g2, gY)
    rf, P = gp["rf"], gp["P"]
    nf = len(rf)
    Hm = R._mat(R.FH(*P, rf, g2, gY), nf).transpose(2, 0, 1)
    Gr = R._vec(R.FGR(*P, rf, g2, gY), nf)
    dP = -np.linalg.solve(Hm, Gr.T[:, :, None])[:, :, 0].T
    d = parts_from_P(rf, P, dP, gp["Et"], gp["Br"], g2, gY)
    Om = np.sqrt(d["Om2"])
    d.update(g2=g2, gY=gY, maxdec=gp["maxdec"], min_hess_eig=gp["min_hess_eig"], march_vs_fine=gp["march_vs_fine"],
             chk_Et=float(np.max(np.abs(d["Et"] - (d["C0"] + 1.5 * Om)))),
             chk_Br=float(np.max(np.abs(d["Br"] - (d["Bx"] + d["Bw"] + d["Bb"])))))
    return d


class Parts:
    """splines of the conditional-Gaussian pieces of one gauge profile (C144 Parts verbatim + 2D accessors)."""

    def __init__(self, rf, C0, Bx, Om2e, EtBr):
        r2 = rf * rf
        self.s_c0 = CubicSpline(rf, r2 * C0)
        self.s_bx = CubicSpline(rf, r2 * Bx)
        self.s_om2 = CubicSpline(rf, Om2e)
        self.s_bo = CubicSpline(rf, r2 * EtBr)

    def lo(self, r):
        r2 = r * r
        return 3.0 / (8 * r2) + (self.s_c0(r) + self.s_bx(r)) / r2 + 1.5 * np.sqrt(np.maximum(self.s_om2(r), 0.0))

    def bw(self, r):
        o2 = self.s_om2(r)
        return 3.0 / 16 * (self.s_om2(r, 1) / (2 * o2)) ** 2

    def g(self, r):
        return 3.0 / (8 * r * r) + self.s_bo(r) / (r * r)

    def c0bx(self, r):                       # 2D: C0 + B_x (C140 build '2D_ex': c0 + bx)
        return (self.s_c0(r) + self.s_bx(r)) / (r * r)

    def om2(self, r):
        return self.s_om2(r)


class DMC:
    """Stage-3 R_rot correction E_tot^ex - E_tot^G (C140 Model.gR, tilt 0): dE/dr(0) = 0, natural, frozen beyond."""

    def __init__(self):
        with open(os.path.join(S3DIR, "S3_prop.json"), encoding="utf-8") as f:
            pj = json.load(f)
        self.kn = np.array(pj["rot_knots"], float)
        self.kv = np.array(pj["rot_dEtot"], float)
        self.f = CubicSpline(self.kn, self.kv, bc_type=((1, 0.0), "natural"))

    def __call__(self, x):
        x = np.asarray(x, float)
        return np.where(x <= self.kn[-1], self.f(np.minimum(x, self.kn[-1])), self.kv[-1])


def bump(r, A, r0, w):
    return A * np.exp(-((np.asarray(r, float) - r0) / w) ** 2)


_PCACHE = {}


def prof_pc_path(mt, als, u):
    return os.path.join(PROFDIR, "prof_mt%.2f_as%.4f_u%.4f.npz" % (mt, als, u))


def get_profile(mt, als, u, conv, CS):
    """'pc': profile with (g2, gY) of (mt, als) at mu(u) (cached npz); 'c144': C144 (172.6, 0.1180) profile (read-only)."""
    key = ("c144", round(u, 4)) if conv == "c144" else ("pc", round(mt, 2), round(als, 4), round(u, 4))
    if key in _PCACHE:
        return _PCACHE[key]
    if conv == "c144":
        d = dict(np.load(os.path.join(C144PROF, "prof_u%.4f.npz" % u)))
    else:
        fn = prof_pc_path(mt, als, u)
        if not os.path.exists(fn):
            c = CS(mt, als, mu_node(u))
            d0 = compute_profile(c["g2"], c["gY"])
            keep = ("rf", "C0", "Bx", "Om2e", "EtBr", "Bw", "g2", "gY", "chk_Et", "chk_Br", "maxdec", "min_hess_eig")
            os.makedirs(PROFDIR, exist_ok=True)
            tmp = fn[:-4] + "_tmp%d.npz" % os.getpid()
            np.savez(tmp, u=u, mt=mt, als=als, **{k: np.asarray(d0[k]) for k in keep})
            os.replace(tmp, fn)
        d = dict(np.load(fn))
    out = (Parts(d["rf"], d["C0"], d["Bx"], d["Om2e"], d["EtBr"]),
           dict(g2=float(d["g2"]), gY=float(d["gY"]), chk_Et=float(d["chk_Et"]), chk_Br=float(d["chk_Br"])))
    if len(_PCACHE) > 64:
        _PCACHE.clear()
    _PCACHE[key] = out
    return out


# ------------------------------------------------------------------------------------------------ [C144] 1D solvers
def frob(U, r):
    sel = r <= 0.06
    A = np.vstack([np.ones(sel.sum()), r[sel], r[sel] ** 2, r[sel] ** 3]).T
    coef = np.linalg.lstsq(A, (r * r * U)[sel], rcond=None)[0]
    beta, alpha = float(coef[0]), float(coef[1])
    return 0.5 + math.sqrt(0.25 + 2 * beta), alpha


def _fr(cols, r):
    fr = [frob(U, r) for U in cols]
    return np.array([x[0] for x in fr]), np.array([x[1] for x in fr])


def solve_res(cols, r, m, guess, h=None, spread=0.03):
    """Siegert resonances (R14 Numerov + 2nd-order WKB outgoing condition, vectorised Muller)."""
    h = R.H_N if h is None else h
    Ut = np.array(cols).T.copy()
    s, al = _fr(cols, r)

    def fun(E):
        return R.jost(E, Ut, r, m, s, al, h)[0]
    g = np.asarray(guess, complex)
    E, _ = R.muller(fun, g - spread, g + spread, g + 0j)
    rs = np.abs(fun(E)) / np.maximum(np.abs(R.jost(E, Ut, r, m, s, al, h)[1]), 1e-300)
    return E, rs


def badres(E, g, dq=0.35, lr=3.5):
    g = np.asarray(g, complex)
    gam = -2 * E.imag
    gg = -2 * g.imag
    with np.errstate(all="ignore"):
        ok = (np.isfinite(E) & (gam > -1e-9) & (np.abs(E.real - g.real) < dq)
              & (np.abs(np.log(np.maximum(gam, 1e-300) / np.maximum(gg, 1e-300))) < lr))
    return ~ok


def solve1d(cols, r, m, guess, spread=0.03):
    """1D roots; columns failing the sanity test are re-seeded from a real-axis |f| scan (+-0.08) and re-solved."""
    guess = np.asarray(guess, complex)
    E, rs = solve_res(cols, r, m, guess, spread=spread)
    bad = badres(E, guess, 0.06, 3.0)
    nres = 0
    if bad.any():
        jj = np.where(bad)[0]
        sub = [cols[j] for j in jj]
        Ut = np.array(sub).T.copy()
        s, al = _fr(sub, r)
        offs = np.arange(-0.08, 0.08001, 0.002)
        cen = guess[jj].real
        av = np.array([np.abs(R.jost(cen + o + 0j, Ut, r, m, s, al)[0]) for o in offs])
        i0 = int(np.argmin(np.abs(offs)))
        gg = np.empty(len(jj), complex)
        for k in range(len(jj)):
            a_ = av[:, k]
            loc = [i for i in range(1, len(offs) - 1) if a_[i] < a_[i - 1] and a_[i] < a_[i + 1]]
            i = min(loc, key=lambda i_: abs(i_ - i0)) if loc else i0
            gg[k] = cen[k] + offs[i] + 1j * guess[jj[k]].imag
        E2, rs2 = solve_res(sub, r, m, gg, spread=0.002)
        E[jj], rs[jj] = E2, rs2
        nres = len(jj)
        bad = badres(E, guess, 0.1, 4.0)
    return E, rs, bad, nres


def U1d(model, Pp, dmc, r, c, prm, dvmc=None):
    lam = c["lam"]
    tree = lam * r ** 4 / 4
    if model in ("MU", "MU_CW"):
        return Pp.g(r) + tree + C.dV(model, r, lam, c["g2"], c["gY"], c["yt"])
    dv = dvmc if dvmc is not None else C.dV("MC", r, lam, c["g2"], c["gY"], c["yt"])
    if model == "MC":
        return Pp.g(r) + tree + dv
    if model == "1D_ex":
        return Pp.g(r) + dmc(r) + tree + dv
    if model == "lo":
        return Pp.lo(r) + dmc(r) + tree + dv
    if model == "Ueff":
        return Pp.lo(r) + dmc(r) + bump(r, prm["A"], prm["r0"], prm["w"]) + tree + dv
    if model == "UeffG":
        return Pp.lo(r) + bump(r, prm["A"], prm["r0"], prm["w"]) + tree + dv
    raise ValueError(model)


# ------------------------------------------------------------------------------------------------ [C140] 2D solver
def rgrid(Rout, h):
    m = int(round((Rout - R0) / h))
    return R0 + h * np.arange(m + 3), m


class SGrid:
    """sine DVR on (0, Smax) for v = s chi (3D s-wave), Dirichlet at both ends."""

    def __init__(self, N, Smax):
        self.N, self.Smax = int(N), float(Smax)
        j = np.arange(1, self.N + 1)
        self.s = j * self.Smax / (self.N + 1)
        self.s2 = self.s ** 2
        U = math.sqrt(2.0 / (self.N + 1)) * np.sin(np.pi * np.outer(j, j) / (self.N + 1))
        self.T = (U * (0.5 * (np.pi * j / self.Smax) ** 2)[None, :]) @ U.T


def bc_data(U1, om2, sg, h, m, order=2):
    """Siegert/decaying 2nd-order WKB log-derivative in the local adiabatic w-channels at r_m -> Y(E) (P, N, N)."""
    P = U1.shape[0]
    if sg is None:
        eps = np.zeros((P, 5, 1))
        Phi = np.ones((P, 1, 1))
    else:
        N = sg.N
        eps = np.empty((P, 5, N))
        Phi = np.empty((P, N, N))
        for p in range(P):
            for j, i in enumerate(range(m - 2, m + 3)):
                w, V = np.linalg.eigh(sg.T + np.diag(0.5 * om2[p, i] * sg.s2))
                eps[p, j] = w
                if i == m:
                    Phi[p] = V
    Uc = U1[:, m - 2:m + 3][:, :, None] + eps
    Um_ = Uc[:, 2]
    U1d_ = (Uc[:, 0] - 8 * Uc[:, 1] + 8 * Uc[:, 3] - Uc[:, 4]) / (12 * h)
    U2d = (-Uc[:, 0] + 16 * Uc[:, 1] - 30 * Uc[:, 2] + 16 * Uc[:, 3] - Uc[:, 4]) / (12 * h * h)

    def Y(E):
        d = E[:, None] - Um_
        kk = np.where(d.real > 0, np.sqrt(2 * d + 0j), 1j * np.sqrt(-2 * d + 0j))
        if order == 0:
            y = 1j * kk
        else:
            k1 = -U1d_ / kk
            k2 = -U2d / kk - U1d_ ** 2 / kk ** 3
            y = 1j * kk - k1 / (2 * kk) + (1j / (2 * kk)) * (3 * k1 ** 2 / (4 * kk ** 2) - k2 / (2 * kk))
        return np.einsum("pab,pb,pcb->pac", Phi, y, Phi)
    return Y


def prop(E, Dd, Tn, h, m, ph0, ph1, Yf):
    """batched renormalised matrix Numerov u'' = 2(V - E)u, F_i = W_i u_i, R_i = F_{i+1} F_i^{-1};
    returns (complex sign, log|.|) of det(u'_m - Y u_m) = det(A) det(F_m) (entire in E)."""
    P, n, N = Dd.shape
    dg = np.arange(N)
    cc = h * h / 6.0
    Eb = E[:, None]

    def Wm(i):
        A = np.empty((P, N, N), complex)
        A[:] = Tn
        A[:, dg, dg] += 1.0 - cc * (Dd[:, i, :] - Eb)
        return A
    W0 = Wm(0)
    Rp = (ph1 / ph0)[:, None, None] * (Wm(1) @ np.linalg.inv(W0))
    s0, l0 = np.linalg.slogdet(W0)
    sgn = s0.astype(complex)
    la = l0 + N * np.log(ph0)
    s_, l_ = np.linalg.slogdet(Rp)
    sgn, la = sgn * s_, la + l_
    kW, kR, kRi = {}, {}, {}
    for i in range(1, m + 2):
        Wi = np.linalg.inv(Wm(i))
        Rpi = np.linalg.inv(Rp)
        Rc = 12.0 * Wi - Rpi
        Rc[:, dg, dg] -= 10.0
        if i <= m - 1:
            s_, l_ = np.linalg.slogdet(Rc)
            sgn, la = sgn * s_, la + l_
        if i >= m - 2:
            kW[i] = Wi
            kR[i] = Rc
        if i - 1 >= m - 2:
            kRi[i - 1] = Rpi
        Rp = Rc
    kW[m + 2] = np.linalg.inv(Wm(m + 2))
    G1 = kR[m]
    G2 = kR[m + 1] @ kR[m]
    Gm1 = kRi[m - 1]
    Gm2 = kRi[m - 2] @ kRi[m - 1]
    Up = (kW[m - 2] @ Gm2 - 8 * (kW[m - 1] @ Gm1) + 8 * (kW[m + 1] @ G1) - kW[m + 2] @ G2) / (12 * h)
    A = Up - Yf(E) @ kW[m]
    sA, lA = np.linalg.slogdet(A)
    return sgn * sA, la + lA


def muller2(fun, x0, x1, x2, tol=1e-10, maxit=40):
    """[C140] muller (maxit 60 -> 40)."""
    f0, f1, f2 = fun(x0), fun(x1), fun(x2)
    act = np.ones(x0.shape, bool)
    it = 0
    last = np.full(x0.shape, np.inf)
    with np.errstate(all="ignore"):
        for it in range(maxit):
            h1 = x1 - x0
            h2 = x2 - x1
            d1 = (f1 - f0) / h1
            d2 = (f2 - f1) / h2
            A = (d2 - d1) / (h2 + h1)
            B = A * h2 + d2
            disc = np.sqrt(B * B - 4 * A * f2)
            den = np.where(np.abs(B + disc) >= np.abs(B - disc), B + disc, B - disc)
            dx = -2 * f2 / den
            ok = act & np.isfinite(dx)
            dx = np.where(ok, dx, 0)
            last = np.where(ok, np.abs(dx), last)
            x3 = x2 + dx
            x0, x1, x2 = x1, x2, x3
            f0, f1 = f1, f2
            f2 = fun(x3)
            act = act & (np.abs(dx) > tol * np.maximum(1, np.abs(x3)))
            if not act.any():
                break
    return x2, it, act, last


def phases(U1, r):
    s, al = frob(U1, r)
    return r[0] ** s * (1 + al / s * r[0]), r[1] ** s * (1 + al / s * r[1]), s, al


def make_fun(U1, OM, r, m, h, sg, order, ph0, ph1):
    Dd = U1[:, :, None] + 0.5 * OM[:, :, None] * sg.s2[None, None, :]
    Tn = (-(h * h / 6.0) * sg.T).astype(complex)
    Yf = bc_data(U1, OM, sg, h, m, order)
    cn = {"n": 0}

    def fun(E):
        cn["n"] += 1
        s_, la = prop(np.asarray(E, complex), Dd, Tn, h, m, ph0, ph1, Yf)
        if "c" not in cn:
            cn["c"] = la.copy()
        return s_ * np.exp(np.clip(la - cn["c"], -700.0, 700.0))
    return fun, cn


def window_roots(fun, centers, gims, window, d=0.001):
    """D112 local-window rule: |f| on a real window around each seed, interior local minimum closest to the seed,
    Muller with small spacing (the summed outgoing phase of ~N open channels makes wide starts fall into box zeros)."""
    half, step = window
    offs = np.arange(-half, half + 0.5 * step, step)
    P = len(centers)
    av = np.array([np.abs(fun(centers + o + 0j)) for o in offs])
    i0 = int(np.argmin(np.abs(offs)))
    jmin, noloc = np.empty(P, int), np.zeros(P, bool)
    for jj in range(P):
        a_ = av[:, jj]
        loc = [i for i in range(1, len(offs) - 1) if a_[i] < a_[i - 1] and a_[i] < a_[i + 1]]
        if loc:
            jmin[jj] = min(loc, key=lambda i_: abs(i_ - i0))
        else:
            jmin[jj], noloc[jj] = i0, True
    g = centers + offs[jmin] - 0.5j * gims
    E, it, act, last = muller2(fun, g - d, g + d, g + 0j)
    return E, dict(it=int(it), act=act, noloc=noloc, Emin=centers + offs[jmin],
                   depth=[float(av[jmin[j], j] / np.median(av[:, j])) for j in range(P)])


def build2d(Pp, dmc, r, c, var):
    U = 3.0 / (8 * r * r) + Pp.c0bx(r) + c["lam"] * r ** 4 / 4 + C.dV("MC", r, c["lam"], c["g2"], c["gY"], c["yt"])
    if var == "2D_ex":
        U = U + dmc(r)
    elif var != "2D_G":
        raise ValueError(var)
    return U, Pp.om2(r)


def flags2d(E, Eb, center, gim, half, act, actb, noloc):
    if not np.isfinite(E):
        return ["nan"]
    Q, g = float(E.real), float(-2 * E.imag)
    fl = []
    if act:
        fl.append("muller_nc")
    if not g > 0:
        fl.append("gamma<=0")
    if abs(Q - center) > half:
        fl.append("outside_window")
    if g > 0 and gim > 0 and abs(math.log(g / gim)) > 1.0:
        fl.append("gamma_far_from_seed")
    if (not np.isfinite(Eb)) or actb or abs(Eb - E) > max(2e-7, 2e-3 * max(g, 0.0)):
        fl.append("box_sensitive")
    if noloc:
        fl.append("no_local_min")
    return fl


def solve2d_block(U1, OM, r2, m2, h, cfg, ph0, ph1, centers, gims, window, seed=None):
    sg = SGrid(cfg["N"], cfg["Smax"])
    fun, cnt = make_fun(U1, OM, r2, m2, h, sg, int(cfg["order"]), ph0, ph1)
    P = U1.shape[0]
    if seed is not None:
        E, it, act, _ = muller2(fun, seed - 1e-4, seed + 1e-4, seed + 0j)
        wd = dict(it=int(it), act=act, noloc=np.zeros(P, bool), Emin=seed.real, depth=[None] * P)
    else:
        E, wd = window_roots(fun, centers, gims, window)
    sgb = SGrid(BOXCHK["N"], BOXCHK["Smax"])
    funb, cntb = make_fun(U1, OM, r2, m2, h, sgb, int(cfg["order"]), ph0, ph1)
    Eok = np.where(np.isfinite(E), E, centers - 0.5j * gims)
    Eb, itb, actb, _ = muller2(funb, Eok - 1e-4, Eok + 1e-4, Eok + 0j, maxit=25)
    return E, Eb, wd, actb, cnt["n"] + 1.6 * cntb["n"]


def job2d(spec):
    fn = os.path.join(SH2, spec["tag"] + ".json")
    if os.path.exists(fn):
        return jload(fn)
    t0 = time.time()
    CS = Couplings(KIND, cal=spec["cal"])
    dmc = DMC()
    prm = spec["prm"]
    u, var = float(spec["u"]), spec["var"]
    cfg = dict(FAST)
    cfg.update(spec.get("cfg") or {})
    h, Rout = float(cfg["h"]), float(cfg["Rout"])
    r2, m2 = rgrid(Rout, h)
    r1, m1 = R.grid(16.0)
    cols = spec["cols"]
    P = len(cols)
    U1, OM = np.empty((P, len(r2))), np.empty((P, len(r2)))
    ph0, ph1 = np.empty(P), np.empty(P)
    c1d, g1d, rows = [], [], []
    m1name = "Ueff" if var == "2D_ex" else "UeffG"
    for j, cc in enumerate(cols):
        conv = cc.get("conv", "pc")
        c = CS(cc["mt"], cc["als"], mu_node(u))
        Pp, pinfo = get_profile(cc["mt"], cc["als"], u, conv, CS)
        U1[j], OM[j] = build2d(Pp, dmc, r2, c, var)
        ph0[j], ph1[j], s, al = phases(U1[j], r2)
        c1d.append(U1d(m1name, Pp, dmc, r1, c, prm))
        e0 = c144_E(cc["als"], cc["mt"], u)
        dq, gr = OFF1D[m1name]
        g1d.append(complex(e0.real + dq, e0.imag * gr))
        rows.append(dict(mt=cc["mt"], als=cc["als"], conv=conv, u=u, frob2d=[s, al], g2=c["g2"], gY=c["gY"],
                         yt=c["yt"], lam=c["lam"], prof_g2=pinfo["g2"], prof_gY=pinfo["gY"], prof_chk=pinfo["chk_Et"]))
    E1, rs1, bad1, nres1 = solve1d(c1d, r1, m1, np.array(g1d))
    seed = None
    if spec.get("seed") is not None:
        seed = np.array([complex(a, b) for a, b in spec["seed"]])
        centers, gims = seed.real.copy(), -2 * seed.imag
    else:
        centers = E1.real + DQ2D[var]
        gims = np.maximum(-2 * E1.imag, 1e-14)
    window = tuple(spec.get("window") or WIN[var])
    E2, Eb, wd, actb, nev = solve2d_block(U1, OM, r2, m2, h, cfg, ph0, ph1, centers, gims, window, seed)
    fl = [flags2d(E2[j], Eb[j], centers[j], gims[j], window[0] if seed is None else 0.01, wd["act"][j], actb[j],
                  wd["noloc"][j]) for j in range(P)]
    retried, first = np.zeros(P, bool), [list(x) for x in fl]
    jj = [j for j in range(P) if fl[j]]
    if jj and seed is None:                        # retry with the wide D112 window
        jj = np.array(jj)
        E2r, Ebr, wdr, actbr, nevr = solve2d_block(U1[jj], OM[jj], r2, m2, h, cfg, ph0[jj], ph1[jj], centers[jj],
                                                   gims[jj], WIN_WIDE)
        nev += nevr
        for k, j in enumerate(jj):
            f2 = flags2d(E2r[k], Ebr[k], centers[j], gims[j], WIN_WIDE[0], wdr["act"][k], actbr[k], wdr["noloc"][k])
            retried[j] = True
            if len(f2) <= len(fl[j]):
                E2[j], Eb[j], fl[j] = E2r[k], Ebr[k], f2
    for j in range(P):
        rows[j].update(E1=[float(E1[j].real), float(E1[j].imag)], Q1=float(E1[j].real), gamma1=float(-2 * E1[j].imag),
                       resid1=float(rs1[j]), bad1=bool(bad1[j]), E2=[float(E2[j].real), float(E2[j].imag)],
                       Q2=float(E2[j].real), gamma2=float(-2 * E2[j].imag), center=float(centers[j]),
                       gim=float(gims[j]), Emin=float(wd["Emin"][j]), depth=wd["depth"][j],
                       box_dE=float(abs(Eb[j] - E2[j])) if np.isfinite(Eb[j]) else None,
                       flags=fl[j], retried=bool(retried[j]), flags_first=first[j])
    out = dict(tag=spec["tag"], u=u, var=var, cfg=cfg, window=window, cols=rows, n_eval=nev, n_rescan_1d=nres1,
               muller_it=wd["it"], time_s=time.time() - t0)
    wjson(fn, out)
    say("[2D %s] u=%.3f %.0fs nev=%.0f | %s" % (spec["tag"], u, time.time() - t0, nev, "; ".join(
        "%.2f/%.4f g2D=%.4e g1D=%.4e r=%.4f dQ=%.4f%s%s" % (x["mt"], x["als"], x["gamma2"], x["gamma1"],
                                                       x["gamma2"] / x["gamma1"] if x["gamma1"] > 0 else float("nan"),
                                                       x["Q2"] - x["Q1"], " R" if x["retried"] else "",
                                                       (" FLAG:" + ",".join(x["flags"])) if x["flags"] else "")
        for x in rows)))
    return out


# ------------------------------------------------------------------------------------------------ 1D hazard chains
def job1d(spec):
    fn = os.path.join(SH1, spec["tag"] + ".json")
    if os.path.exists(fn):
        return jload(fn)
    t0 = time.time()
    CS = Couplings(KIND, cal=spec["cal"])
    dmc = DMC()
    prm = spec["prm"]
    nodes = [float(x) for x in spec["nodes"]]
    conv = spec["conv"]
    cols = [(c[0], round(float(c[1]), 2), round(float(c[2]), 4)) for c in spec["cols"]]
    pairs = sorted(set((c[1], c[2]) for c in cols))
    r, m = R.grid(16.0)
    nc, nn = len(cols), len(nodes)
    E = np.full((nc, nn), np.nan + 0j)
    LAM, flags, nresc = {}, [], 0
    prevE = prev144 = None
    for i, u in enumerate(nodes):
        cp, Pp, dv = {}, {}, {}
        for (mt, als) in pairs:
            c = CS(mt, als, mu_node(u))
            cp[(mt, als)] = c
            Pp[(mt, als)] = get_profile(mt, als, u, conv, CS)[0]
            dv[(mt, als)] = C.dV("MC", r, c["lam"], c["g2"], c["gY"], c["yt"])
            LAM["%.2f|%.4f|%.4f" % (mt, als, u)] = c["lam"]
        Us = [U1d(mo, Pp[(mt, als)], dmc, r, cp[(mt, als)], prm, dv[(mt, als)]) for (mo, mt, als) in cols]
        e144 = np.array([c144_E(als, mt, u) for (mo, mt, als) in cols])
        seed0 = np.array([complex(e.real + OFF1D[mo][0], e.imag * OFF1D[mo][1]) for e, (mo, _, _) in zip(e144, cols)])
        if i == 0:
            guess = seed0
        else:
            with np.errstate(all="ignore"):
                cont = (prevE.real + (e144.real - prev144.real)) + 1j * prevE.imag * (e144.imag / prev144.imag)
            guess = np.where(np.isfinite(cont) & (cont.imag < 0), cont, seed0)
        Ei, rs, bad, nr = solve1d(Us, r, m, guess)
        nresc += nr
        for j in np.where(bad)[0]:
            flags.append(dict(u=u, col=list(cols[j]), E=[float(Ei[j].real), float(Ei[j].imag)]))
        E[:, i] = Ei
        prevE, prev144 = Ei.copy(), e144
        say("[1D %s] node %d u=%.3f bad=%d rescan=%d %.0fs" % (spec["tag"], i, u, int(bad.sum()), nr, time.time() - t0))
    out = dict(tag=spec["tag"], conv=conv, nodes=nodes, cols=[list(c) for c in cols], E_re=E.real, E_im=E.imag,
               lam=LAM, flags=flags, n_rescan=nresc, time_s=time.time() - t0)
    wjson(fn, out)
    return out


# ------------------------------------------------------------------------------------------------ [C144] hazard post
def integ(u, g, umax=5.0):
    uf = np.linspace(0.0, umax, 20001)
    wf = 1.0 / (HLB * np.sqrt(np.cosh(2 * uf)))
    spl = CubicSpline(u, np.log(np.maximum(g, 1e-300)))
    return float(simpson(np.exp(spl(uf)) * wf, x=uf))


def crit(mts, I):
    o = np.argsort(mts)
    x = np.asarray(mts, float)[o]
    sp = CubicSpline(x, np.log(np.asarray(I, float)[o]))
    out = {}
    for k, t in TGT.items():
        try:
            out[k] = float(brentq(lambda z: sp(z) - math.log(t), x[0], x[-1]))
        except ValueError:
            out[k] = None
    return out, sp


# ------------------------------------------------------------------------------------------------ modes
def time_mode():
    t0 = time.time()
    CS = Couplings(KIND)
    out = dict(kind=KIND, cal=CS.cal)
    cv = {}
    for als in ALS:
        for mt in (171.8, 172.6, 173.8):
            for u in (0.0, 2.5, 5.0):
                cv["%.4f|%.2f|%.1f" % (als, mt, u)] = CS(mt, als, mu_node(u))
    out["couplings"] = cv
    try:
        ref = jload(C140PREP)["dV_MC_lam_m0p010"]
        out["dV_regression"] = {k: [float(C.dV("MC", np.array([float(k)]), -0.010, V11["g2"], V11["gY"], V11["yt"])[0]),
                                    ref[k]] for k in ref}
    except Exception as e:  # noqa: BLE001
        out["dV_regression"] = repr(e)
    d = np.load(PROF_V11)
    pv = parts_from_P(d["rf"], d["P"], d["dP"], d["Et"], d["Br"], V11["g2"], V11["gY"])
    Pp = Parts(pv["rf"], pv["C0"], pv["Bx"], pv["Om2e"], pv["EtBr"])
    dmc = DMC()
    REF = [(-0.010, 4.28674, 4.1535e-4), (-0.0143, 4.27375, 3.7738e-3)]
    reg = {}
    for name, cfg in (("FAST", FAST), ("BASE", BASE)):
        t1 = time.time()
        r2, m2 = rgrid(cfg["Rout"], cfg["h"])
        U1, OM = np.empty((2, len(r2))), np.empty((2, len(r2)))
        ph0, ph1 = np.empty(2), np.empty(2)
        for j, (lam, q, g) in enumerate(REF):
            U1[j], OM[j] = build2d(Pp, dmc, r2, dict(V11, lam=lam), "2D_ex")
            ph0[j], ph1[j], _, _ = phases(U1[j], r2)
        fun, cnt = make_fun(U1, OM, r2, m2, cfg["h"], SGrid(cfg["N"], cfg["Smax"]), cfg["order"], ph0, ph1)
        E, wd = window_roots(fun, np.array([x[1] for x in REF]), np.array([x[2] for x in REF]), WIN["2D_ex"])
        dt = time.time() - t1
        reg[name] = dict(Q=E.real, gamma=-2 * E.imag, rel_vs_D112=[float(-2 * E[j].imag / REF[j][2] - 1) for j in range(2)],
                         dQ_vs_D112=[float(E[j].real - REF[j][1]) for j in range(2)], n_eval=cnt["n"], time_s=dt,
                         s_per_eval_per_problem=dt / cnt["n"] / 2, muller_it=wd["it"], nc=wd["act"])
        say("v11 regression %s: %s" % (name, json.dumps(clean(reg[name]))))
    out["v11_regression_2D"] = reg
    tj = job2d(dict(tag="TEST_2Dex_u0.0000", u=0.0, var="2D_ex", cal=CS.cal, prm=load_prm(),
                    cols=[dict(mt=172.6, als=AS0), dict(mt=173.0, als=AS0)]))
    out["test_job"] = dict(time_s=tj["time_s"], n_eval=tj["n_eval"],
                           cols=[{k: x[k] for k in ("mt", "Q1", "gamma1", "Q2", "gamma2", "flags", "box_dE", "lam",
                                                    "g2", "gY", "yt", "prof_g2", "prof_gY")} for x in tj["cols"]])
    out["time_s"] = time.time() - t0
    wjson(os.path.join(RES, "C150_time.json"), out)
    status({"done": ["timing + v11 regression (C150_time.json)"], "files": ["C150_time.json"]})
    say("time mode done %.0fs" % (time.time() - t0))


def _specs2d(which, cal, prm):
    specs = []
    if which == "main":
        for u in UN11:
            specs.append(dict(tag="2Dex_main_u%.4f" % u, u=u, var="2D_ex", cols=[dict(mt=mt, als=AS0) for mt in MT_MAIN]))
    elif which == "G":
        for u in UN11:
            specs.append(dict(tag="2DG_u%.4f" % u, u=u, var="2D_G", cols=[dict(mt=mt, als=AS0) for mt in MT_G]))
    elif which == "als":
        for u in UN11:
            specs.append(dict(tag="2Dex_als_u%.4f" % u, u=u, var="2D_ex",
                              cols=[dict(mt=mt, als=a) for a in (0.1171, 0.1189) for mt in MT_AS[a]]))
    elif which == "n29":
        for u in UN29:
            if not np.any(np.abs(UN11 - u) < 1e-9):
                specs.append(dict(tag="2Dex_n29_u%.4f" % u, u=u, var="2D_ex",
                                  cols=[dict(mt=172.6, als=AS0), dict(mt=173.0, als=AS0)]))
    elif which == "c144":
        for u in UN11:
            specs.append(dict(tag="2Dex_c144conv_u%.4f" % u, u=u, var="2D_ex",
                              cols=[dict(mt=mt, als=AS0, conv="c144") for mt in (172.4, 172.6, 172.8, 173.0, 173.2)]))
    elif which == "conv":
        for u in (0.0, 2.5, 5.0):
            mj = jload(os.path.join(SH2, "2Dex_main_u%.4f.json" % u))
            sel = [x for x in mj["cols"] if round(x["mt"], 2) in (172.6, 173.0)]
            for name, cfg in CONV.items():
                specs.append(dict(tag="conv_%s_u%.4f" % (name, u), u=u, var="2D_ex", cfg=cfg,
                                  seed=[x["E2"] for x in sel], cols=[dict(mt=x["mt"], als=AS0) for x in sel]))
    else:
        raise ValueError(which)
    for s in specs:
        s.update(cal=cal, prm=prm)
    return specs


def two(which, nw=3):
    t0 = time.time()
    cal, prm = Couplings(KIND).cal, load_prm()
    specs = _specs2d(which, cal, prm)
    status({"stage": "2D %s running (%d node jobs)" % (which, len(specs)),
            "next_command": "python -B C150_twod.py two %s 3   (resumes per node shard in shards2d/)" % which})
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(specs)))) as ex:  # [v12-package] was min(nw, 3)
        outs = list(ex.map(job2d, specs))
    nfl = sum(1 for o in outs for x in o["cols"] if x["flags"])
    nrt = sum(1 for o in outs for x in o["cols"] if x["retried"])
    say("2D %s done: %d jobs, %d flagged, %d retried, %.0fs" % (which, len(outs), nfl, nrt, time.time() - t0))
    status({"done": ["2D %s (%d node jobs, %d flagged, %d retried)" % (which, len(outs), nfl, nrt)]})


def oned(nw=3):
    t0 = time.time()
    cal, prm = Couplings(KIND).cal, load_prm()
    specs = [dict(tag="A_c144_as0.1180", conv="c144", nodes=UN29,
                  cols=[[mo, mt, AS0] for mo in ("Ueff", "MC") for mt in MT_FINE]),
             dict(tag="A_c144_as0.1171", conv="c144", nodes=UN29, cols=[["Ueff", mt, 0.1171] for mt in MT_FINE2]),
             dict(tag="A_c144_as0.1189", conv="c144", nodes=UN29, cols=[["Ueff", mt, 0.1189] for mt in MT_FINE2]),
             dict(tag="B_pc_as0.1180", conv="pc", nodes=UN11,
                  cols=[[mo, mt, AS0] for mo in ("MU", "MU_CW", "MC", "1D_ex", "lo", "Ueff", "UeffG") for mt in MT_MAIN]),
             dict(tag="B_pc_asvar", conv="pc", nodes=UN11,
                  cols=[[mo, mt, a] for a in (0.1171, 0.1189) for mt in MT_AS[a] for mo in ("Ueff", "lo")])]
    for s in specs:
        s.update(cal=cal, prm=prm)
    status({"stage": "1D chains running", "next_command": "python -B C150_twod.py oned 3 (resumes per shard in shards1d/)"})
    with ProcessPoolExecutor(max_workers=max(1, min(nw, len(specs)))) as ex:  # [v12-package] was min(nw, 3)
        outs = list(ex.map(job1d, specs))
    say("1D done: %s %.0fs" % ([(o["tag"], len(o["flags"]), o["n_rescan"]) for o in outs], time.time() - t0))
    status({"done": ["1D chains (%s)" % ", ".join(o["tag"] for o in outs)]})


# ------------------------------------------------------------------------------------------------ post
def post():
    t0 = time.time()
    out = dict(meta=dict(task="T14b / C150 / D122", agent="twod-direct", coupling_source=KIND,
                         recipe="hazard as C138/C144 (xi_H=1/6, u<=5, mu(u)=mu_b/sqrt(cosh 2u)); 2D_ex direct",
                         nodes_main=UN11.tolist(), mt_main=MT_MAIN, cfg_main=FAST, boxcheck=BOXCHK))
    pub144 = jload(os.path.join(C144RES, "C144_hazard.json"))
    pubC = pub144["critical_mt"]
    c138 = jload(C138SUM)
    ueff = jload(os.path.join(C144RES, "C144_ueff.json"))["final"]
    rv = sorted((float(k), float(v)) for k, v in ueff["rel"].items())
    crv = np.polyfit([x[0] for x in rv], [x[1] for x in rv], 2)
    # ---------------- 2D shards
    R2, conv_rows = {}, []
    m2 = dict(n_cols=0, flagged=[], retried=[], rescued=[], max_box_rel=0.0, max_box_abs=0.0, n_eval=0.0, time_s=0.0,
              n_bad1=0, max_rel_2D_minus_seed_Q=0.0)
    for fn in sorted(glob.glob(os.path.join(SH2, "*.json"))):
        d = jload(fn)
        if d["tag"].startswith("TEST"):
            continue
        if d["tag"].startswith("conv"):
            conv_rows.append(d)
            continue
        m2["n_eval"] += d["n_eval"]
        m2["time_s"] += d["time_s"]
        for x in d["cols"]:
            R2[(d["var"], x["conv"], round(x["als"], 4), round(x["mt"], 2), round(x["u"], 4))] = x
            m2["n_cols"] += 1
            m2["n_bad1"] += int(x["bad1"])
            if x["flags"]:
                m2["flagged"].append([d["tag"], x["mt"], x["als"], x["flags"]])
            if x["retried"]:
                m2["retried"].append([d["tag"], x["mt"], x["als"], x["flags_first"], x["flags"]])
                if not x["flags"]:
                    m2["rescued"].append([d["tag"], x["mt"], x["als"]])
            if x["gamma2"] and x["gamma2"] > 0 and x["box_dE"] is not None:
                m2["max_box_rel"] = max(m2["max_box_rel"], x["box_dE"] / x["gamma2"])
                m2["max_box_abs"] = max(m2["max_box_abs"], x["box_dE"])
            m2["max_rel_2D_minus_seed_Q"] = max(m2["max_rel_2D_minus_seed_Q"], abs(x["Q2"] - x["center"]))
    out["roots_2D"] = m2

    def I2(var, conv, als, mt, nodes, which="gamma2", div=None):
        g = []
        for u in nodes:
            x = R2.get((var, conv, round(als, 4), round(mt, 2), round(float(u), 4)))
            if x is None:
                return None
            if which == "gamma2" and (x["flags"] or not x["gamma2"] > 0):
                return None
            if which == "gamma1" and (x["bad1"] or not x["gamma1"] > 0):
                return None
            v = x[which]
            if div is not None:
                v = v / (1.0 + div(x["lam"]))
            g.append(v)
        return integ(np.array(nodes, float), np.array(g))

    def crit_d(I):
        ks = sorted(k for k, v in I.items() if v is not None)
        if len(ks) < 4:
            return None, None
        return crit(ks, [I[k] for k in ks])
    # ---------------- main 2D direct (alpha_s = 0.1180)
    I2D = {mt: I2("2D_ex", "pc", AS0, mt, UN11) for mt in MT_MAIN}
    IUpc = {mt: I2("2D_ex", "pc", AS0, mt, UN11, "gamma1") for mt in MT_MAIN}
    Ia = {mt: I2("2D_ex", "pc", AS0, mt, UN11, "gamma1", div=lambda l: np.polyval(crv, l)) for mt in MT_MAIN}
    c2D, sp2D = crit_d(I2D)
    cUpc, _ = crit_d(IUpc)
    cA, _ = crit_d(Ia)
    main = dict(P1={"%.2f" % k: (math.exp(-v) if v is not None else None) for k, v in I2D.items()},
                I={"%.2f" % k: v for k, v in I2D.items()}, critical_mt=c2D,
                P1_at={k: float(math.exp(-math.exp(sp2D(v)))) for k, v in
                       (("172.2", 172.2), ("172.52", 172.52), ("172.6", 172.6), ("173.0", 173.0))} if sp2D else None,
                dlnI_dmt={k: float(sp2D(v, 1)) for k, v in (c2D or {}).items() if v is not None} if sp2D else None,
                gamma_ratio_2D_over_Ueff_nodes={"%.2f" % mt: [R2[("2D_ex", "pc", AS0, mt, round(float(u), 4))]["gamma2"]
                                                              / R2[("2D_ex", "pc", AS0, mt, round(float(u), 4))]["gamma1"]
                                                              for u in UN11] for mt in MT_MAIN
                                                if I2D[mt] is not None},
                lam_nodes={"%.2f" % mt: [R2[("2D_ex", "pc", AS0, mt, round(float(u), 4))]["lam"] for u in UN11]
                           for mt in MT_MAIN if I2D[mt] is not None})
    rr = main["gamma_ratio_2D_over_Ueff_nodes"]
    main["ratio_summary"] = {mt: dict(min=min(v), max=max(v), u0=v[0], u5=v[-1], lam_u0=main["lam_nodes"][mt][0],
                                      lam_u5=main["lam_nodes"][mt][-1],
                                      v11_implied_u0=1.0 / (1.0 + float(np.polyval(crv, main["lam_nodes"][mt][0]))))
                             for mt, v in rr.items()}
    out["main_2D_direct_as0.1180"] = main
    # ---------------- 1D shards
    A = {}
    for fn in sorted(glob.glob(os.path.join(SH1, "*.json"))):
        d = jload(fn)
        nodes = np.array(d["nodes"], float)
        Ere, Eim = np.array(d["E_re"], float), np.array(d["E_im"], float)
        badset = set((f["col"][0], round(f["col"][1], 2), round(f["col"][2], 4)) for f in d["flags"])
        sel11 = [i for i, u in enumerate(nodes) if np.any(np.abs(UN11 - u) < 1e-9)]
        for j, (mo, mt, als) in enumerate(d["cols"]):
            key = (mo, d["conv"], round(als, 4), round(mt, 2))
            g = -2 * Eim[j]
            if (mo, round(mt, 2), round(als, 4)) in badset or not np.all(g > 0):
                A[key] = None
                continue
            ent = dict(I_all=integ(nodes, g), n_nodes=len(nodes))
            ent["I11"] = integ(nodes[sel11], g[sel11]) if len(nodes) != 11 else ent["I_all"]
            A[key] = ent

    def critA(mo, conv, als, mts, which):
        I = {mt: (A.get((mo, conv, round(als, 4), mt)) or {}).get(which) for mt in mts}
        return crit_d(I)
    one = {}
    for mo in ("MU", "MU_CW", "MC", "1D_ex", "lo", "Ueff", "UeffG"):
        one[mo] = critA(mo, "pc", AS0, MT_MAIN, "I11")[0]
    out["rows_1D_pc_11nodes_as0.1180"] = one
    reg = dict(Ueff_c144_29nodes_fine={a: critA("Ueff", "c144", float(a), MT_FINE if a == "0.1180" else MT_FINE2,
                                               "I_all")[0] for a in ("0.1171", "0.1180", "0.1189")},
               C144_published=pubC,
               MC_c144_29nodes_fine=critA("MC", "c144", AS0, MT_FINE, "I_all")[0],
               C138_published=c138["central_MC+CW_mu=1/L"])
    reg["Ueff_minus_C144"] = {a: {k: (reg["Ueff_c144_29nodes_fine"][a][k] - pubC[a][k])
                                  if (reg["Ueff_c144_29nodes_fine"].get(a) and reg["Ueff_c144_29nodes_fine"][a].get(k)
                                      and pubC.get(a, {}).get(k)) else None for k in TGT} for a in pubC}
    reg["MC_minus_C138"] = {k: reg["MC_c144_29nodes_fine"][k] - v for k, v in c138["central_MC+CW_mu=1/L"].items()}
    pt = {}                                      # column-by-column P1 regression against the published C144 table
    for a, tb in pub144["P1_table"].items():
        rows_ = []
        for mts_, v in tb.items():
            e = A.get(("Ueff", "c144", round(float(a), 4), round(float(mts_), 2)))
            if e is not None:
                rows_.append([float(mts_), math.exp(-e["I_all"]), v["P1"], math.log(e["I_all"] / v["I"])])
        if rows_:
            pt[a] = dict(n=len(rows_), max_abs_dlnI=max(abs(x[3]) for x in rows_),
                         mine_minus_C144_not_in_table=sorted(set(round(m, 2) for (mo, cv, aa, m) in A
                                                                 if mo == "Ueff" and cv == "c144" and aa == round(float(a), 4)
                                                                 and "%.2f" % m not in tb)), rows=rows_)
    reg["P1_table_vs_C144"] = pt
    nt = {}
    for mo, a, mts in (("Ueff", "0.1180", MT_FINE), ("MC", "0.1180", MT_FINE), ("Ueff", "0.1171", MT_FINE2),
                       ("Ueff", "0.1189", MT_FINE2)):
        c29 = critA(mo, "c144", float(a), mts, "I_all")[0]
        c11 = critA(mo, "c144", float(a), mts, "I11")[0]
        if c29 and c11:
            nt["%s|%s" % (mo, a)] = {k: (c11[k] - c29[k]) if (c11[k] and c29[k]) else None for k in TGT}
    # 2D 11-vs-29 node test at m_t = 172.6, 173.0
    n2 = {}
    for mt in (172.6, 173.0):
        a11 = I2("2D_ex", "pc", AS0, mt, UN11)
        a29 = I2("2D_ex", "pc", AS0, mt, UN29)
        if a11 and a29 and sp2D is not None:
            dl = math.log(a29 / a11)
            n2["%.1f" % mt] = dict(I11=a11, I29=a29, dlnI=dl, dmc_GeV=-dl / float(sp2D(mt, 1)))
    out["node_test"] = dict(oneD_c144_11_minus_29_GeV=nt, twoD_11_vs_29=n2)
    out["regression_1D_vs_published"] = reg
    # ---------------- decomposition (alpha_s = 0.1180) vs C144
    cC11, _ = critA("Ueff", "c144", AS0, MT_MAIN, "I11")
    cC29 = reg["Ueff_c144_29nodes_fine"].get("0.1180")
    dec = {}
    if c2D and cA and cUpc and cC11 and cC29:
        for k in TGT:
            p = pubC["0.1180"][k]
            dec[k] = dict(m_c_2D_direct=c2D[k], m_c_C144_published=p, total=c2D[k] - p,
                          b_coupling_transfer=c2D[k] - cA[k], a_bump_fit_residual=cA[k] - cUpc[k],
                          c_profile_coupling_convention=cUpc[k] - cC11[k],
                          d1_node_set_and_mt_grid=cC11[k] - cC29[k], d2_regression_residual=cC29[k] - p)
    out["decomposition_vs_C144"] = dec
    # direct check of the profile convention at 2D level (if c144-convention 2D shards exist)
    pc2 = {}
    for mt in (172.4, 172.6, 172.8, 173.0, 173.2):
        a = I2("2D_ex", "c144", AS0, mt, UN11)
        b = I2D.get(mt)
        if a and b and sp2D is not None:
            pc2["%.1f" % mt] = dict(dlnI_pc_minus_c144=math.log(b / a), dmc_GeV=-math.log(b / a) / float(sp2D(mt, 1)))
    out["profile_convention_2D_check"] = pc2
    # ---------------- ratio method (2D / U_eff c144 fine) and alpha_s variants
    def ratio_method(als, pts, mts_fine, which_fine):
        r_ = {mt: (I2("2D_ex", "pc", als, mt, UN11), (A.get(("Ueff", "c144", round(als, 4), mt)) or {}).get("I11"))
              for mt in pts}
        ok = [mt for mt in pts if r_[mt][0] and r_[mt][1]]
        if len(ok) < 3:
            return None
        x = np.array(ok)
        y = np.array([math.log(r_[mt][0] / r_[mt][1]) for mt in ok])
        cf = np.polyfit(x - 172.8, y, 2 if len(ok) >= 4 else 1)
        mts_fine = [mt for mt in mts_fine if min(ok) - 0.2 - 1e-9 <= mt <= max(ok) + 0.2 + 1e-9]   # no far extrapolation
        If = {mt: (A.get(("Ueff", "c144", round(als, 4), mt)) or {}).get("I11") for mt in mts_fine}
        If = {mt: v * math.exp(np.polyval(cf, mt - 172.8)) for mt, v in If.items() if v}
        cr, _ = crit_d(If)
        dr, _ = crit_d({mt: r_[mt][0] for mt in ok})
        return dict(points=ok, ln_ratio=y.tolist(), fit=cf.tolist(), fit_resid_max=float(np.max(np.abs(np.polyval(cf, x - 172.8) - y))),
                    critical_ratio_method=cr, critical_direct_spline=dr)
    av = {"0.1180": ratio_method(AS0, MT_MAIN, MT_FINE, "I11")}
    for a in (0.1171, 0.1189):
        av["%.4f" % a] = ratio_method(a, MT_AS[a], MT_FINE2, "I11")
    out["alpha_s_variants"] = av
    try:
        lo, hi = av["0.1171"]["critical_ratio_method"], av["0.1189"]["critical_ratio_method"]
        out["dmc_dalpha_s"] = {k: (hi[k] - lo[k]) / 0.0018 for k in TGT if lo.get(k) and hi.get(k)}
    except Exception:  # noqa: BLE001
        out["dmc_dalpha_s"] = None
    # ---------------- 2D_G
    IG = {mt: I2("2D_G", "pc", AS0, mt, UN11) for mt in MT_G}
    cG, _ = crit_d(IG)
    out["twoD_G_direct"] = dict(critical_mt=cG, P1={"%.2f" % k: (math.exp(-v) if v else None) for k, v in IG.items()})
    # ---------------- numerics (conv)
    nm = {}
    for d in conv_rows:
        for x in d["cols"]:
            ref = R2.get(("2D_ex", "pc", AS0, round(x["mt"], 2), round(x["u"], 4)))
            if ref is None:
                continue
            nm.setdefault(d["tag"].rsplit("_u", 1)[0], []).append(dict(u=x["u"], mt=x["mt"],
                                                                       rel_gamma=x["gamma2"] / ref["gamma2"] - 1,
                                                                       dQ=x["Q2"] - ref["Q2"], flags=x["flags"]))
    nsum = {k: dict(max_abs_rel_gamma=max(abs(y["rel_gamma"]) for y in v), max_abs_dQ=max(abs(y["dQ"]) for y in v),
                    rows=v) for k, v in nm.items()}
    if sp2D is not None and c2D:
        smin = min(float(sp2D(c2D[k], 1)) for k in ("P1=0.9", "P1=0.5") if c2D[k])
        for k in nsum:
            nsum[k]["dmc_bound_GeV"] = nsum[k]["max_abs_rel_gamma"] / smin
    out["numerics_2D"] = nsum
    try:
        out["v11_regression_and_timing"] = jload(os.path.join(RES, "C150_time.json"))
    except Exception:  # noqa: BLE001
        out["v11_regression_and_timing"] = None
    # ---------------- convergence table (alpha_s = 0.1180)
    m0 = c138["central_MC+CW_mu=1/L"]
    alt = c138["error_budget_GeV"]
    pubrows = {"MU": {k: m0[k] + alt[k]["superseded_alternatives_minus_MC"]["MU"] for k in m0},
               "MU_CW": {k: m0[k] + alt[k]["superseded_alternatives_minus_MC"]["MU_CW"] for k in m0},
               "MC": dict(m0), "1D_ex": {"P1=0.9": m0["P1=0.9"] + 0.044, "P1=0.5": m0["P1=0.5"] + 0.062},
               "2D_G": {"P1=0.9": m0["P1=0.9"] + 0.114, "P1=0.5": m0["P1=0.5"] + 0.168},
               "2D_ex_D112": {"P1=0.9": 172.509, "P1=0.5": 173.002}, "C144": pubC["0.1180"]}
    stages = [("MU (D108/R14)", one.get("MU"), pubrows["MU"]), ("MU_CW", one.get("MU_CW"), pubrows["MU_CW"]),
              ("MC+CW (D110)", one.get("MC"), pubrows["MC"]), ("1D_ex (+DMC gauge shape)", one.get("1D_ex"), pubrows["1D_ex"]),
              ("2D_G (2D, no DMC)", cG, pubrows["2D_G"]), ("2D_ex (D112 ratio translation)", None, pubrows["2D_ex_D112"]),
              ("U_eff hazard (C144)", one.get("Ueff"), pubrows["C144"]), ("2D direct (this work, C150)", c2D, None),
              ("+ cross term (T2)", None, None), ("+ NNLO SM mapping (T1)", None, None)]
    tab, prev = [], None
    for name, mine, pub in stages:
        use = mine if mine else (pub if RUN == "run01" else None)     # other runs: moves from recomputed rows only
        row = dict(stage=name, recomputed_this_pipeline=mine, published=pub)
        if use and prev:
            row["move_vs_previous"] = {k: (use[k] - prev[k]) if (use.get(k) is not None and prev.get(k) is not None)
                                       else None for k in ("P1=0.9", "P1=0.5")}
        if use:
            prev = use
        tab.append(row)
    mv = [abs(r_["move_vs_previous"]["P1=0.5"]) for r_ in tab if r_.get("move_vs_previous")
          and r_["move_vs_previous"].get("P1=0.5") is not None]
    out["convergence_table"] = dict(rows=tab, abs_moves_P1_0p5=mv,
                                    monotone_decreasing=bool(all(mv[i + 1] <= mv[i] + 1e-12 for i in range(len(mv) - 1))),
                                    last_two_le_0p05=bool(len(mv) >= 1 and mv[-1] <= 0.05))
    # ---------------- comparison and verdict
    if c2D:
        out["compare_C144"] = {k: dict(two_D_direct=c2D[k], C144=pubC["0.1180"][k], diff=c2D[k] - pubC["0.1180"][k],
                                       pass_0p02=abs(c2D[k] - pubC["0.1180"][k]) <= 0.02) for k in TGT if c2D.get(k)}
    if RUN != "run01":
        crs = {"0.1180": c2D}
        for a in ("0.1171", "0.1189"):
            crs[a] = (av.get(a) or {}).get("critical_ratio_method")
        out["compare_T1_C144_pipeline"] = {a: {k: dict(two_D_direct=crs[a][k], T1=v, diff=crs[a][k] - v)
                                               for k, v in T1_C144PIPE[a].items() if crs.get(a) and crs[a].get(k)}
                                           for a in T1_C144PIPE}
        uc = reg["Ueff_c144_29nodes_fine"]
        out["regression_Ueff_c144_vs_T1"] = {a: {k: (uc[a][k] - v) if (uc.get(a) and uc[a].get(k)) else None
                                                 for k, v in T1_C144PIPE[a].items()} for a in T1_C144PIPE}
        try:
            r1 = jload(os.path.join(os.path.dirname(RES), "run01", "C150_summary.json"))
            c1 = r1["main_2D_direct_as0.1180"]["critical_mt"]
            out["nnlo_row_for_run01_table"] = dict(
                stage="+ NNLO SM mapping (T1 sm_provider central), 2D direct", m_c=c2D, P1_at=main.get("P1_at"),
                run01_2D_direct=c1, move_vs_run01_2D_direct={k: c2D[k] - c1[k] for k in TGT if c2D.get(k) and c1.get(k)},
                note="T2 cross-term row still empty: move is relative to the R14 2D-direct row")
        except Exception as e:  # noqa: BLE001
            out["nnlo_row_for_run01_table"] = repr(e)
    out["code_sha256"] = {fn: sha(os.path.join(HERE, fn)) for fn in sorted(os.listdir(HERE)) if fn.endswith(".py")}
    out["time_s"] = time.time() - t0
    wjson(os.path.join(RES, "C150_summary.json"), out)
    status({"done": ["post -> C150_summary.json"], "files": ["C150_summary.json"],
            "key_numbers": dict(critical_mt_2D_direct=c2D, P1_at=main.get("P1_at"), compare_C144=out.get("compare_C144"))})
    print(json.dumps(clean(dict(main=dict(critical_mt=c2D, P1_at=main.get("P1_at")), compare=out.get("compare_C144"),
                                decomposition=dec, alpha_s=av, dmc_das=out.get("dmc_dalpha_s"), node_test=out["node_test"],
                                regression=reg, rows1D=one, twoD_G=cG, roots=dict((k, v) for k, v in m2.items()
                                                                                  if k not in ("retried",)),
                                numerics={k: dict((kk, vv) for kk, vv in v.items() if kk != "rows") for k, v in nsum.items()},
                                profile_conv_2D=pc2, conv_table=out["convergence_table"])), indent=1))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "post"
    if md == "time":
        time_mode()
    elif md == "two":
        two(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif md == "oned":
        oned(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif md == "post":
        post()
    else:
        raise SystemExit(__doc__)
