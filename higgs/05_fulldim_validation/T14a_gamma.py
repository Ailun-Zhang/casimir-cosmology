# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_gamma.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Independent width estimates: CAP d ln gamma/dQ (well-localised shift) and multi-dimensional adiabatic WKB.  Paper: PRD H2.

T14a step 3: (i) d ln gamma/dQ of the 2D model itself (CAP FD, well-localised potential shift d*S(r));
(ii) independent gamma estimate (a): multi-dimensional adiabatic WKB along r.
    python -B T14a_gamma.py cap      -> run01/T14a_cap_dlng.json
    python -B T14a_gamma.py wkb      -> run01/T14a_wkb.json (needs T14a_dmc_combined.json + T14a_2d_wall.json for Q_full)
CAP: H_FD(2D_ex, Dirichlet r=11, s=12) - i eta (r-7)^2 theta(r-7), Riss-Meyer optimum on a log eta grid (as C140_cap),
     the eta of the unshifted operator is kept fixed for the shifted ones (derivative not polluted by eta-grid jumps).
Shift shapes (well region, barrier r1 ~ 2.5 untouched):  S_A = 1 (r<1.8), cos^2 taper to 0 at 2.4;
                                                         S_B = 1 (r<1.2), cos^2 taper to 0 at 2.4.
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.integrate import quad
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
import T14a_common as T

sys.stdout.reconfigure(encoding="utf-8")
RC, RWC, SWC = 7.0, 11.0, 12.0
ETAS = np.logspace(-2.5, 0.5, 13)
SHAPES = {"A": (1.8, 2.4), "B": (1.2, 2.4)}
MAIN_WALLS = {-0.010: (4.80, 5.12, 5.44), -0.013: (4.40, 4.64, 4.96)}   # between barrier top and outer turning point
DELTAS = (-0.006, 0.0, 0.006)


def shape(r, a, b):
    t = np.clip((r - a) / (b - a), 0.0, 1.0)
    return np.cos(0.5 * np.pi * t) ** 2


def cap_H(var, lam, h, extra=None):
    nr = int(round(RWC / h)) - 1
    r = h * np.arange(1, nr + 1)
    U1, om2 = T.potentials(var, lam, r)
    if extra is not None:
        U1 = U1 + extra(r)
    ns = int(round(SWC / h)) - 1
    s = h * np.arange(1, ns + 1)
    H = (sparse.kron(T.lap(nr, h), sparse.identity(ns)) + sparse.kron(sparse.identity(nr), T.lap(ns, h))
         + sparse.diags((U1[:, None] + 0.5 * om2[:, None] * s[None, :] ** 2).ravel())).tocsc()
    Wc = np.repeat(np.where(r > RC, (r - RC) ** 2, 0.0), ns)
    return H, sparse.diags(Wc), r, s


def cap_task(args):
    var, lam, h, shp, deltas, target = args
    t0 = time.time()
    H0, Wd, r, s = cap_H(var, lam, h)
    sig = complex(target)
    Es = []
    for eta in ETAS:
        vals = eigs((H0 - 1j * eta * Wd).tocsc(), k=3, sigma=sig, which="LM", return_eigenvectors=False)
        j = int(np.argmin(np.abs(vals - sig)))
        Es.append(complex(vals[j]))
        sig = complex(vals[j])
    Es = np.array(Es)
    dE = np.gradient(Es, np.log(ETAS))
    i = int(np.argmin(np.abs(dE[1:-1]))) + 1
    eta = float(ETAS[i])
    out = dict(var=var, lam=lam, h=h, shape=shp, eta_opt=eta, E0=[Es[i].real, Es[i].imag], stab=float(abs(dE[i])),
               rows=[])
    a, b = SHAPES[shp]
    for d in deltas:
        if d == 0.0:
            E = Es[i]
        else:
            H, _, _, _ = cap_H(var, lam, h, extra=lambda rr: d * shape(rr, a, b))
            vals = eigs((H - 1j * eta * Wd).tocsc(), k=3, sigma=complex(Es[i]) + d * 0.9, which="LM",
                        return_eigenvectors=False)
            E = vals[int(np.argmin(np.abs(vals - (Es[i] + d * 0.9))))]
        out["rows"].append(dict(delta=d, Q=float(E.real), gamma=float(-2 * E.imag)))
    rr = out["rows"]
    lg = np.log([x["gamma"] for x in rr]); qq = np.array([x["Q"] for x in rr])
    out["dlng_dQ"] = float((lg[-1] - lg[0]) / (qq[-1] - qq[0]))
    out["dQ_ddelta"] = float((qq[-1] - qq[0]) / (DELTAS[-1] - DELTAS[0]))
    out["curv_check"] = float((lg[-1] - 2 * lg[1] + lg[0]) / (lg[-1] - lg[0]))
    out["time_s"] = round(time.time() - t0, 1)
    return out


def cap():
    old = T.jload(T.OLDSUM)
    tasks = []
    for lam in (-0.010, -0.013):
        q, g = old["Q"]["2D_ex"][f"{lam}"], old["gamma"]["2D_ex"][f"{lam}"]
        for shp in ("A", "B"):
            tasks.append(("2D_ex", lam, 0.04, shp, DELTAS, complex(q, -g / 2)))
        qg, gg = old["Q"]["2D_G"][f"{lam}"], old["gamma"]["2D_G"][f"{lam}"]
        tasks.append(("2D_G", lam, 0.04, "A", DELTAS, complex(qg, -gg / 2)))
    with ProcessPoolExecutor(max_workers=1) as ex:
        outs = list(ex.map(cap_task, tasks))
    for o in outs:
        o["siegert_ref"] = dict(Q=old["Q"][o["var"]][f"{o['lam']}"], gamma=old["gamma"][o["var"]][f"{o['lam']}"])
        print(o["var"], o["lam"], o["shape"], "dlng/dQ=%.4f dQ/dd=%.4f curv=%.3f eta=%.3g" % (
            o["dlng_dQ"], o["dQ_ddelta"], o["curv_check"], o["eta_opt"]), o["rows"], o["time_s"], flush=True)
    T.wjson(os.path.join(T.RES, "T14a_cap_dlng.json"), dict(tasks=outs, R_c=RC, R_w=RWC, S_w=SWC, shapes=SHAPES))


# ------------------------------------------------------------------------------------------------ WKB along r
def wkb(Ufun, E, r_lo=1.0, r_hi=8.0):
    """Gamow: S = 2 int_{r1}^{r2} sqrt(2(U-E)), T_cl = 2 int_{r0}^{r1} dr / sqrt(2(E-U)); gamma = exp(-S)/T_cl.
    also dS/dE = -2 int dr/kappa (barrier)."""
    rr = np.linspace(r_lo, r_hi, 20001)
    U = Ufun(rr)
    it = int(np.argmax(np.where(rr > 2.5, U, -np.inf)))
    f = lambda x: float(Ufun(np.array([x]))[0] - E)
    r1 = brentq(f, 2.0, rr[it]); r2 = brentq(f, rr[it], r_hi)
    rs = np.linspace(0.02, 2.0, 4001)
    Us = Ufun(rs)
    j = int(np.argmax(Us < E))
    r0 = brentq(f, rs[max(j - 1, 0)], rs[j]) if j > 0 else rs[0]
    S = 2 * quad(lambda x: np.sqrt(max(2 * f(x), 0.0)), r1, r2, limit=400)[0]
    Tcl = 2 * quad(lambda x: 1 / np.sqrt(max(-2 * f(x), 1e-300)), r0, r1, limit=400, points=None)[0]
    dSdE = -2 * quad(lambda x: 1 / np.sqrt(max(2 * f(x), 1e-300)), r1, r2, limit=400)[0]
    return dict(E=E, r0=r0, r1=r1, r2=r2, r_top=float(rr[it]), U_top=float(U[it]), S=S, T_cl=Tcl,
                gamma_wkb=float(np.exp(-S) / Tcl), dSdE=dSdE)


def wkb_main():
    old = T.jload(T.OLDSUM)
    comb = T.jload(os.path.join(T.RES, "T14a_dmc_combined.json"))
    wall = T.jload(os.path.join(T.RES, "T14a_2d_wall.json"))
    M = T.model()
    out = {}
    for lam, tag in ((-0.010, "m010"), (-0.013, "m013")):
        Q2 = old["Q"]["2D_ex"][f"{lam}"]; g2 = old["gamma"]["2D_ex"][f"{lam}"]
        # Delta Q (same wall, weighted over walls) from DMC vs 2D_ex FD
        dqs = []
        for k, v in comb.items():
            if not k.startswith(tag) or v["extrap"].get("E0") is None:
                continue
            rw = v["case"]["rw"]
            if rw not in MAIN_WALLS[lam]:
                continue
            q2w = wall[f"2D_ex|{lam}|{rw:.2f}"]["Q"]
            dqs.append((v["extrap"]["E0"] - q2w, v["extrap"]["se_E0"]))
        if not dqs:
            continue
        w_ = np.array([1 / s ** 2 for _, s in dqs]); d_ = np.array([d for d, _ in dqs])
        dQ = float((w_ * d_).sum() / w_.sum()); sdQ = float(1 / np.sqrt(w_.sum()))
        # potentials along r
        U2 = lambda x: T.u_adiabatic("2D_ex", lam, x)
        U2bw = lambda x: (T.u_adiabatic("2D_ex", lam, x)
                          + 3.0 / 16 * (M.s_om2(x, 1) / (2 * T.potentials("2D_ex", lam, x)[1])) ** 2)
        Ufull = lambda x: T.potentials("1D_ex", lam, x)[0]            # E_tot^ex + B_r^G (+ 3/8r^2, lam r^4/4, dV)
        rec = dict(dQ=dQ, se_dQ=sdQ, Q_2D_siegert=Q2, gamma_2D_siegert=g2)
        rec["2D_lowest_channel"] = wkb(U2, Q2)
        rec["2D_channel_plus_Bw"] = wkb(U2bw, Q2)
        rec["full_adiabatic_at_Qfull"] = wkb(Ufull, Q2 + dQ)
        rec["full_adiabatic_at_Q2D"] = wkb(Ufull, Q2)
        rec["2D_channel_at_Qfull"] = wkb(U2, Q2 + dQ)
        rec["2D_plus_Bw_at_Qfull"] = wkb(U2bw, Q2 + dQ)
        for sgn, lab in ((1, "plus"), (-1, "minus")):
            rec[f"2D_plus_Bw_at_Qfull_{lab}1se"] = wkb(U2bw, Q2 + dQ + sgn * sdQ)
        a = rec["2D_lowest_channel"]
        for key in [k for k in rec if isinstance(rec[k], dict) and k != "2D_lowest_channel"]:
            b = rec[key]
            b["ratio_to_2D_wkb"] = b["gamma_wkb"] / a["gamma_wkb"]
            b["dS_vs_2D"] = b["S"] - a["S"]
            b["ratio_exponent_only"] = float(np.exp(-(b["S"] - a["S"])))
        rec["wkb_over_siegert_2D"] = a["gamma_wkb"] / g2
        out[f"{lam}"] = rec
        print(lam, json.dumps(T.clean(rec))[:1500], flush=True)
    T.wjson(os.path.join(T.RES, "T14a_wkb.json"), out)


if __name__ == "__main__":
    {"cap": cap, "wkb": wkb_main}[sys.argv[1]]()
