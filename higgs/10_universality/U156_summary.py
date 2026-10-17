# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/universality/U156_summary.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent universality (C156/D128, v12 round 3) on 2026-10-14.
r"""[v12 package, 2026-10-15] C156 summary (C156_summary.json): eta scaling, coupling derivatives, sector results, SHA-256 receipts.  Paper: PRD Sec. IV D.

C156 summary: stationary tetragonal shapes at the PPA/PAA holonomy vacua (root of dZ_eff/dbeta), WKB coupling law,
SHA-256 of code and inputs, Wolfram receipt, C156_summary.json.  Usage: python -B U156_summary.py"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import glob
import hashlib
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import U156_twist as TW  # noqa: E402
import U156_centre as CE  # noqa: E402

ROOT = _zp.WORK
RES = TW.OUT
CODE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def J(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def gain(k):
    return math.sqrt(2) * (math.cos(math.pi / 4 * math.sqrt(1 - 4 * k)) if k <= 0.25 else
                           math.cosh(math.pi / 4 * math.sqrt(4 * k - 1)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    eta, coup, tw, hol, cen = J("C156_eta.json"), J("C156_coup.json"), J("C156_twist.json"), J("C156_hol2.json"), J("C156_centre.json")
    # ---- stationary tetragonal shapes at the holonomy vacua
    st = {}
    for name, thq, thl, axis, br in (("PPA_centre", (1 / 3, 1 / 3, 0.5), (0, 0, 0.5), 2, (-0.45, 0.0)),
                                     ("PAA_centre", (1 / 3, 0.5, 0.5), (0, 0.5, 0.5), 0, (0.0, 0.45))):
        h = 1e-3

        def dz(b):
            return (CE.zeff(TW.tet(b + h, axis), thq, thl) - CE.zeff(TW.tet(b - h, axis), thq, thl)) / (2 * h)
        bs = brentq(dz, *br, xtol=1e-7)
        z = CE.zeff(TW.tet(bs, axis), thq, thl)
        zpp = (CE.zeff(TW.tet(bs + h, axis), thq, thl) - 2 * z + CE.zeff(TW.tet(bs - h, axis), thq, thl)) / h ** 2
        # second (orthogonal) aspect mode X' = diag(1,-1,0)/sqrt2 on the two equal axes
        oth = [i for i in range(3) if i != axis]

        def zo(e):
            a = TW.tet(bs, axis)
            a[oth[0]] *= math.exp(2 * e / math.sqrt(2))
            a[oth[1]] *= math.exp(-2 * e / math.sqrt(2))
            return CE.zeff(a, thq, thl)
        zoo = (zo(h) - 2 * z + zo(-h)) / h ** 2
        NC = z / TW.Zs(2, (1, 1, 1), (0, 0, 0))
        xi = 12.8 / (NC * TW.GAM)
        a = TW.tet(bs, axis)
        ratio = math.sqrt(a[axis] / a[oth[0]])
        kt, ko = 3 * zpp / z, 3 * zoo / z
        st[name] = dict(beta_s=bs, side_ratio_special_over_other=ratio, N_C=NC, xi_Q12p8=xi, kappa_tet=kt, kappa_aspect2=ko,
                        kappa_tet_eff=kt / (1 - xi), kappa_aspect2_eff=ko / (1 - xi),
                        gain_tet_eff=gain(kt / (1 - xi)), gain_aspect2_eff=gain(ko / (1 - xi)))
        print(name, json.dumps(st[name]), flush=True)
    # ---- WKB coupling law: dlnI/dlng_i -> predicted dlambda_c/dln g_i = s_i * 2|lam_c|^(3/2)/b_eff vs direct
    import numpy as np
    x0 = json.load(open(os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02/shards"),
                                     "X_main_u0.0000.json"), encoding="utf-8"))
    lamb = {round(c["mt"], 2): c["lam"] for c in x0["cols"]}
    MTV = coup["cols"]
    lv = np.array([abs(lamb[round(m, 2)]) for m in MTV])
    wk = {}
    dl = math.log(1.1) - math.log(0.9)
    for g in ("g2", "gY", "yt"):
        a_, b_ = coup["variants"].get(g + "+"), coup["variants"].get(g + "-")
        if not (a_ and b_ and "dlnI_at_fixed_mt" in a_ and "dlnI_at_fixed_mt" in b_):
            continue
        s = (np.array(a_["dlnI_at_fixed_mt"]) - np.array(b_["dlnI_at_fixed_mt"])) / dl        # dlnI/dln g at fixed lambda
        A = np.vstack([np.ones_like(lv), -lv ** -0.5]).T
        (alpha, beta), *_ = np.linalg.lstsq(A, s, rcond=None)
        wk[g] = dict(s_columns=s.tolist(), alpha=float(alpha), beta=float(beta))
        for k in ("P1=0.9", "P1=0.5", "P1=0.1"):
            be, lc = eta["b_eff_hazard"][k]["b_eff"], abs(eta["b_eff_hazard"][k]["lam_c"])
            Sc = be / math.sqrt(lc)
            bp = beta / be
            pred = 2 * bp - 2 * alpha / Sc                                    # dln|lam_c|/dln g (WKB decomposition)
            D = coup["derivatives"].get(g, {}).get(k, {})
            wk[g][k] = dict(S_c=Sc, dlnb_dlng=bp, dlnlamc_dlng_wkb=pred,
                            dlnlamc_dlng_direct=(-D["dlamc_dlng"] / lc) if D else None,
                            dlamc_dlng_direct=D.get("dlamc_dlng"), dmc_dlng_direct=D.get("dmc_dlng"),
                            pure_zero_mode_scaling=2 - 4 / (3 * Sc))
    # ---- hashes
    code = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(CODE, "U156_*.py")))}
    HX = os.path.join(ROOT, _zp.ZMLIB)
    inp = {f: sha(os.path.join(HX, f)) for f in ("C149_direct.py", "C150_twod.py", "cross_core.py", "cross_kappa.py", "cross_2d.py",
                                                  "cross_var.py", "c150_nnlo_central.py", "sm_provider.py")}
    X02 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02"))
    inp["run02/C149_run02_summary.json"] = sha(os.path.join(X02, "C149_run02_summary.json"))
    for p in sorted(glob.glob(os.path.join(X02, "shards", "X_main_u*.json"))):
        inp["run02/shards/" + os.path.basename(p)] = sha(p)
    out = dict(task="T17 universality (C156/D128)", agent="universality",
               eta=dict(dlamc_dlneta=eta["eta_slopes"], b_eff=eta["b_eff_hazard"], wkb_vs_exact=eta["wkb_vs_exact_duration"],
                        dlnI_dmt=eta["dlnI_dmt_at_mc"], static_u0=eta["static_u0"],
                        D124_cross_check={"hLb x e^-1 (eta x e)": [-0.170, -0.270, -0.394], "hLb x e^+1": [0.214, 0.367, None]}),
               couplings=dict(base=coup.get("base"), derivatives=coup.get("derivatives"), wkb_check=wk,
                              static_u0={v: d.get("static_u0") for v, d in coup.get("variants", {}).items()},
                              s_eff_u0={v: d.get("s_eff_u0") for v, d in coup.get("variants", {}).items()},
                              flags={v: d.get("flags") for v, d in coup.get("variants", {}).items()}),
               spin={k: {x: tw["spin"][k][x] for x in ("N_C", "Z1F", "Ydir", "kappa_Y", "kappa_2", "kappa_3")} for k in tw["spin"]},
               shape_trivial_holonomy={k: {x: tw["shape"][k][x] for x in ("F", "kappa_tet", "stationary_points", "xi0", "Q12.8")}
                                       for k in tw["shape"]},
               holonomy_vacuum={k: {x: v[x] for x in ("E_trivial", "harmonic", "points")} for k, v in hol.items()
                                if isinstance(v, dict) and "harmonic" in v},
               PPA_two_cycles=hol.get("PPA_two_cycles"),
               centre_vacuum={k: {x: v.get(x) for x in ("N_C", "N_C_eff", "kappa_Y", "kappa_2", "kappa_3", "OmegaX2", "r_inv",
                                                        "top_tadpole_Z1", "F", "kappa_tet", "hLb", "eta_inf", "mu_b_GeV",
                                                        "dln_eta_vs_AAA")} for k, v in cen.items()},
               stationary_shapes=st, flux=tw["flux"], sha256=dict(code=code, inputs=inp))
    with open(os.path.join(RES, "C156_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(dict(wkb_check=wk, stationary=st), indent=1))


if __name__ == "__main__":
    main()
