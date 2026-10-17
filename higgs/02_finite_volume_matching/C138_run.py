# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_matching/C138_run.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] D110 driver: checks | profiles | static | main | var | mu | post | all.  Static gamma(lam) tables for the MU/ML/MC matching variants at v11 couplings and the round-2 1D hazard with the R14 mapping (historical numbers).  Paper: PRD App. H1 (matching, Table of gamma), error-budget row 'one-loop matching'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C138 driver (D110).  Usage: python -B C138_run.py checks|profiles|static|main|var|mu|post|all|wexport
Hazard recipe = R14/xi_scan: xi_H = 1/6 (a''/a term cancels), couplings at c/L(u) (2-loop + 3-loop calibration),
pole continuation node by node, I = int_0^5 gamma du/(hL_b sqrt(cosh 2u)) (spline of ln gamma + Simpson), P1 = e^-I.
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
import pickle
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C138_core as C  # noqa: E402
from C138_core import R, R14_rge, np as _np  # noqa: E402,F401

AS0, MH = 0.1180, 125.13
UN29 = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0001, 0.125)]), 6)
UN11 = np.round(np.arange(0, 5.0001, 0.5), 6)
CS = {"0.5": 0.5, "1": 1.0, "2": 2.0, "2pi": 2 * math.pi, "4pi": 4 * math.pi}
MT7 = [171.8, 172.0, 172.2, 172.4, 172.6, 172.8, 173.0]
LAMS = [-0.004, -0.006, -0.008, -0.009, -0.010, -0.011, -0.012, -0.0143, -0.016, -0.020]
PCACHE = os.path.join(C.SCR, "C138_profiles.pkl")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, (complex, np.complexfloating)):
        return [clean(o.real), clean(o.imag)]
    return o


def wjson(name, obj):
    p = os.path.join(C.RES, name)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def status(upd):
    p = os.path.join(C.RES, "STATUS.json")
    try:
        with open(p, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {}
    st.update(upd)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(st), f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def say(msg):
    print(msg, flush=True)
    with open(os.path.join(C.RES, "C138_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


# ------------------------------------------------------------------------------------------- couplings / profiles
_SOL = {}


def Ysol(mt, als):
    k = (round(mt, 4), round(als, 5))
    if k not in _SOL:
        _SOL[k] = R14_rge.solve(mt, als, MH)
    return _SOL[k]


def coup(mt, als, mu, cal):
    y = Ysol(mt, als).sol(np.log(mu))
    return dict(gY=float(y[0]), g2=float(y[1]), g3=float(y[2]), yt=float(y[3]), lam=float(y[4] + cal))


def mu_node(u):
    return R14_rge.MU_B / math.sqrt(math.cosh(2 * u))


def profile_keys():
    keys = [("v11", None)]
    for u in UN29:
        keys.append(("1", float(u)))
    for c in ("0.5", "2", "2pi", "4pi"):
        for u in UN11:
            keys.append((c, float(u)))
    return keys


def _prof(args):
    g2, gY = args
    gp = R.gauss_profile(g2, gY)
    return {k: gp[k] for k in ("spl", "beta_g", "alpha_g", "maxdec", "min_hess_eig", "march_vs_fine", "g2", "gY")}


def profiles():
    t0 = time.time()
    keys = profile_keys()
    args = []
    for c, u in keys:
        if c == "v11":
            args.append((C.V11["g2"], C.V11["gY"]))
        else:
            cc = coup(172.6, AS0, CS[c] * mu_node(u), 0.0)
            args.append((cc["g2"], cc["gY"]))
    with ProcessPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(_prof, args, chunksize=2))
    with open(PCACHE, "wb") as f:
        pickle.dump(dict(zip(keys, res)), f)
    say("profiles done: %d in %.1f s" % (len(keys), time.time() - t0))
    return dict(n=len(keys), t=time.time() - t0)


def load_profiles():
    with open(PCACHE, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------------------- column builder
def build(cb, P, u, r, cal):
    """cb: dict(model, c, gphi, mt, als, ct).  returns potential column and Frobenius parameters."""
    c = CS[cb["c"]]
    if cb.get("static"):
        gp = P[("v11", None)] if cb["static"] == "v11" else P[("1", 0.0)]
        cp = dict(C.V11) if cb["static"] == "v11" else coup(172.6, AS0, mu_node(0.0), cal)
        lam, g2, gY, yt = cb["lam"], cp["g2"], cp["gY"], cp["yt"]
        xi, ytt, mut = 1.0, None, None
    else:
        gp = P[(cb["c"], float(u))]
        cc = coup(cb["mt"], cb["als"], c * mu_node(u), cal)
        lam, g2, gY, yt = cc["lam"], cc["g2"], cc["gY"], cc["yt"]
        xi = 1.0
        if cb.get("gphi"):
            c1 = coup(cb["mt"], cb["als"], mu_node(u), cal)
            xi = math.exp(-C.gamma_phi(c1["g2"], c1["gY"], c1["yt"], cb["gphi"]) * math.log(c))
        ytt = mut = None
        if cb.get("ct"):
            ytt = coup(cb["mt"], cb["als"], cb["ct"] * mu_node(u), cal)["yt"]
            mut = cb["ct"] ** 2 if cb.get("ctlog", True) else None   # ctlog False: only y_t scale (RG-consistent)
    Cr = C.C_shift(xi * r, g2, gY, c * c) if cb["model"] == "MUPS" else None
    U = C.U_col(gp, cb["model"], lam, g2, gY, yt, r, c * c, xi, ytt, mut, Cr)
    s, al = C.spar_xi(gp, xi)
    return U, s, al, lam


def badcheck(E_, g_, dq=0.35, lr=3.5):
    gam = -2 * E_.imag
    ok = np.isfinite(E_) & (gam > -1e-9)
    if g_ is not None:
        gg = -2 * g_.imag
        with np.errstate(all="ignore"):
            near = np.abs(E_.real - g_.real) < dq
            ratio = np.abs(np.log(np.maximum(gam, 1e-300) / np.maximum(gg, 1e-300))) < lr
        ok = ok & near & ratio
    else:
        ok = ok & (E_.real > 4.0) & (E_.real < 4.8)
    return ~ok


def continuation(combos, un, tag):
    t0 = time.time()
    P = load_profiles()
    cal, _ = R14_rge.calibration()
    r, m = R.grid(16.0)
    nc, nn = len(combos), len(un)
    E = np.full((nc, nn), np.nan + 0j)
    LAM = np.zeros((nc, nn))
    prev = None
    nhom = 0
    for i, u in enumerate(un):
        cols, ss, aa = [], [], []
        for j, cb in enumerate(combos):
            U, s, al, lam = build(cb, P, u, r, cal)
            cols.append(U)
            ss.append(s)
            aa.append(al)
            LAM[j, i] = lam
        ss, aa = np.array(ss), np.array(aa)
        guess = None if i == 0 else E[:, i - 1].copy()
        Ei, _, _ = R.solve_set(cols, r, m, ss, aa, guess=guess)
        bad = badcheck(Ei, guess)
        if bad.any() and i > 0:
            jj = np.where(bad)[0]
            nhom += len(jj)
            Eh = E[jj, i - 1].copy()
            for t in np.linspace(1.0 / 8, 1.0, 8):
                ch = [(1 - t) * prev[j] + t * cols[j] for j in jj]
                Eh, _, _ = R.solve_set(ch, r, m, ss[jj], aa[jj], guess=Eh)
            Ei[jj] = Eh
        E[:, i] = Ei
        prev = cols
        g = -2 * Ei.imag
        say("[%s] node %d u=%.3f gamma[min,max]=%.3e,%.3e bad=%d %.1fs" % (tag, i, u, np.nanmin(g), np.nanmax(g),
                                                                         int(bad.sum()), time.time() - t0))
    out = dict(tag=tag, u=un, combos=combos, E=E, lam=LAM, cal=cal, n_homotopy=nhom, time_s=time.time() - t0)
    with open(os.path.join(C.SCR, "C138_haz_%s.pkl" % tag), "wb") as f:
        pickle.dump(out, f)
    return out


def combos_main():
    cb = [dict(model="MU", c="1", mt=mt, als=AS0) for mt in (172.0, 172.2, 172.6)]
    for als, grid in ((0.1180, MT7), (0.1171, [x - 0.4 for x in MT7]), (0.1189, [x + 0.4 for x in MT7])):
        cb += [dict(model="MC", c="1", mt=round(mt, 3), als=als) for mt in grid]
    return cb


def combos_var():
    cb = []
    for mod in ("MC", "MC1", "MU_CW", "ML_CW", "MU", "ML", "MUPS", "MC_v32"):
        cb += [dict(model=mod, c="1", mt=mt, als=AS0) for mt in MT7]
    for ct in (0.5, 2.0):
        cb += [dict(model="MC", c="1", mt=mt, als=AS0, ct=ct) for mt in MT7]
    return cb


def combos_top():
    return [dict(model="MC", c="1", mt=mt, als=AS0, ct=ct, ctlog=False) for ct in (0.5, 2.0) for mt in MT7]


def combos_mu():
    cb = []
    for c in ("0.5", "2", "2pi", "4pi"):
        for gph in ("C", None):
            cb += [dict(model="MC", c=c, gphi=gph, mt=mt, als=AS0) for mt in MT7]
    return cb


def static():
    t0 = time.time()
    P = load_profiles()
    cal, _ = R14_rge.calibration()
    r, m = R.grid(16.0)
    combos = [dict(model=mod, c="1", static=st, lam=lam) for st in ("v11", "SM") for mod in C.MODELS for lam in LAMS]
    cols, ss, aa = [], [], []
    for cb in combos:
        U, s, al, _ = build(cb, P, 0.0, r, cal)
        cols.append(U)
        ss.append(s)
        aa.append(al)
    E, it, rs = R.solve_set(cols, r, m, np.array(ss), np.array(aa))
    bad = badcheck(E, None)
    tab = {}
    for j, cb in enumerate(combos):
        tab.setdefault(cb["static"], {}).setdefault(cb["model"], {})[str(cb["lam"])] = dict(
            Q=float(E[j].real), gamma=float(-2 * E[j].imag), resid=float(rs[j]), bad=bool(bad[j]))
    gc = {"P1=0.9": -math.log(0.9) / C.R.ETA_TOT, "P1=0.5": math.log(2) / C.R.ETA_TOT}
    lc = {}
    for st in tab:
        lc[st] = {}
        for mod in tab[st]:
            ls = np.array(sorted(LAMS))
            lg = np.array([math.log(tab[st][mod][str(x)]["gamma"]) for x in ls])
            sp = CubicSpline(ls, lg)
            lc[st][mod] = {k: float(brentq(lambda x: sp(x) - math.log(v), ls[0], ls[-1])) for k, v in gc.items()}
    out = dict(gamma_c=gc, table=tab, lambda_c=lc, muller_iter=int(it), time_s=time.time() - t0,
               regression_D108={"MU -0.010": 2.051e-3, "MU_CW -0.010": 1.510e-3, "ML -0.010": 1.433e-3})
    wjson("C138_static.json", out)
    say("static done %.1f s" % (time.time() - t0))
    return out


def run_group(name):
    if name == "static":
        return static()
    cb, un = {"main": (combos_main(), UN29), "var": (combos_var(), UN11), "mu": (combos_mu(), UN11),
              "top": (combos_top(), UN11)}[name]
    o = continuation(cb, un, name)
    return dict(tag=name, n=len(cb), time_s=o["time_s"], homotopy=o["n_homotopy"])


# ------------------------------------------------------------------------------------------- post
def integ(u, g, umax=5.0):
    uf = np.linspace(0.0, umax, 20001)
    wf = 1.0 / (R.HLB * np.sqrt(np.cosh(2 * uf)))
    spl = CubicSpline(u, np.log(np.maximum(g, 1e-300)))
    return float(simpson(np.exp(spl(uf)) * wf, x=uf))


def crit(mts, I):
    mts, I = np.asarray(mts), np.asarray(I)
    o = np.argsort(mts)
    sp = CubicSpline(mts[o], np.log(I[o]))
    out = {}
    for k, tgt in (("P1=0.9", -math.log(0.9)), ("P1=0.5", math.log(2.0))):
        try:
            out[k] = float(brentq(lambda x: sp(x) - math.log(tgt), mts[o][0], mts[o][-1]))
        except ValueError:
            out[k] = None
    return out


def post():
    res = {}
    sets = {}
    for tag in ("main", "var", "mu", "top"):
        pk = os.path.join(C.SCR, "C138_haz_%s.pkl" % tag)
        if not os.path.exists(pk):
            continue
        with open(pk, "rb") as f:
            d = pickle.load(f)
        G = -2 * d["E"].imag
        for j, cb in enumerate(d["combos"]):
            key = "%s|c=%s|gphi=%s|ct=%s%s|as=%.4f" % (cb["model"], cb["c"], cb.get("gphi"), cb.get("ct"),
                                                      "" if cb.get("ctlog", True) else "(yt only)", cb["als"])
            I = integ(d["u"], G[j])
            sets.setdefault(key + "|" + tag, []).append(dict(mt=cb["mt"], I=I, P1=math.exp(-I), lam_b=d["lam"][j, 0],
                                                             gamma_u0=G[j, 0], gamma_nodes=G[j].tolist()))
    for k, rows in sets.items():
        mts = [x["mt"] for x in rows]
        Is = [x["I"] for x in rows]
        res[k] = dict(rows=rows, crit=crit(mts, Is) if len(rows) >= 4 else None)
    wjson("C138_hazard.json", dict(meta=dict(u_main=UN29, u_var=UN11, xi_H="1/6", u_max=5.0), sets=res))
    say("post done: %d sets" % len(res))
    return res


def main_all():
    status(dict(stage="profiles running"))
    pinfo = profiles()
    status(dict(stage="hazard groups running", profiles=pinfo))
    with ProcessPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(run_group, ["main", "var", "mu", "static"]))
    status(dict(stage="groups done; post running", groups=[o if isinstance(o, dict) and "tag" in o else "static"
                                                            for o in outs]))
    post()
    status(dict(stage="numerics done (C138_static.json, C138_hazard.json); next: C138_run.py report"))


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else "all"
    if md == "all":
        main_all()
    elif md == "profiles":
        profiles()
    elif md in ("static", "main", "var", "mu", "top"):
        run_group(md)
    elif md == "post":
        post()
    elif md == "checks":
        import C138_checks
        C138_checks.run()
