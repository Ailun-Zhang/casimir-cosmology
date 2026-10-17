# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_matching/C138_checks.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] D110 analytic/numerical checks: log cancellation, analyticity in m^2 inside |m| < 2pi/L, Coulomb-mode subtraction, RG consistency gamma_phi^C = (3y_t^2 - 3/2 g2^2 - 1/2 gY^2)/16pi^2.  Paper: PRD App. H1.

C138 analytic/numerical checks for D110 (H1 log cancellation & MS-bar RG consistency, Coulomb-mode
subtraction independence, Goldstone tadpole C(r), BO/non-adiabatic size, MU<->MC potential relation)."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import math
import json
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C138_core as C  # noqa: E402
from C138_core import R, R14_rge  # noqa: E402

PI = math.pi


def delta_p(a, b, p, mu2=1.0):
    """Coulomb-mode Delta with subtraction scalars (b; a; p a): independent algebra check of C.delta (p = 2)."""
    ca, cp = 2 * b / (a * (p - 1)), -2 * b / (a * p * (p - 1))
    K, k = C.K2, C.KK
    wc = np.sqrt((K + a) * (K + b)) / k
    s = ((a + b + a * b / K) / (wc + k) - a / (np.sqrt(K + a) + k) - b / (np.sqrt(K + b) + k)
         - ca * a / (np.sqrt(K + a) + k) - cp * p * a / (np.sqrt(K + p * a) + k))
    t5 = ((a + b) * (a - b) ** 2 - a ** 3 - b ** 3 - ca * a ** 3 - cp * (p * a) ** 3) / 16
    Rp = 0.5 * np.sum(s * C.NQ) + 0.5 * t5 * C.TAILF
    E = lambda x: float(C.estd(x, mu2)[0])  # noqa: E731
    return Rp + E(b) + ca * E(a) + cp * E(p * a)


def run():
    out = {}
    out["R14_fv_internal"] = dict(TB_half=C.TBH, c2P=float(C.FVF.c2[0]), fit_residual=C.FVF.fitres,
                                  switch_jump=C.FVF.jump, ZP12=C.FVF.ZP12)
    pts = [(0.5, -0.05), (1.74, -0.25), (3.0, -0.5), (10.0, -2.0), (25.0, -4.0), (2.0, 0.3)]
    sub = {}
    for a, b in pts:
        d2 = float(C.delta(a, b)[0])
        sub["a=%g,b=%g" % (a, b)] = dict(p2_core=d2, p3=delta_p(a, b, 3.0), p1_5=delta_p(a, b, 1.5),
                                         p2_check=delta_p(a, b, 2.0))
    out["Delta_subtraction_independence"] = sub
    mud = {}
    for a, b in pts:
        num = (float(C.delta(a, b, 4.0)[0]) - float(C.delta(a, b, 1.0)[0])) / math.log(4.0)
        mud["a=%g,b=%g" % (a, b)] = dict(dDelta_dlnmu2=num, expected=-(b * b - 2 * a * b) / (64 * PI ** 2))
    out["Delta_mu_dependence"] = mud
    sl = {}
    for a in (1e-4, 0.1, 1.0, 5.0, 20.0):
        eps = 1e-5
        num = (float(C.delta(a, eps)[0]) - float(C.delta(a, -eps)[0])) / (2 * eps)
        sl["a=%g" % a] = dict(fd=num, analytic=float(C.ddelta_db0(a)[0]))
    out["dDelta_db_at_0"] = sl
    rr = np.array([0.02, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    out["MC_minus_MUCW_at_lam0"] = float(np.max(np.abs(C.dV("MC", rr, 0.0, 0.5277, 0.4426, 0.4284)
                                                      - C.dV("MU_CW", rr, 0.0, 0.5277, 0.4426, 0.4284))))
    out["C_shift"] = {}
    for tag, g2, gY in (("SM_mub", 0.5277, 0.4426), ("v11", 0.5268, 0.4415)):
        cr = C.C_shift(rr, g2, gY)
        out["C_shift"][tag] = {"%g" % x: dict(C=float(c), rC=float(c * x)) for x, c in zip(rr, cr)}
    out["C_shift"]["massless_limit_3TB/2"] = 3 * C.TBH
    # one-loop RG consistency of the explicit mu dependence (H3): mu d/dmu V1 + beta_lam r^4/4 - gamma lam r^4 = 0
    g2, gY, yt, lam = 0.5277, 0.4426, 0.4284, -0.010
    bl1 = float(R14_rge.beta(np.array([gY, g2, 0.5358, yt, lam]), nl=1)[4])
    rg = {}
    for mod, gauge in (("MC", "C"), ("ML_CW", "L"), ("MU_CW", "L")):
        r = np.array([1.0, 2.0, 3.0, 5.0])
        h = 0.01
        expl = (C.dV(mod, r, lam, g2, gY, yt, mu2=math.exp(2 * h)) - C.dV(mod, r, lam, g2, gY, yt, mu2=1.0)) / h
        gph = C.gamma_phi(g2, gY, yt, gauge)
        resid = expl + bl1 * r ** 4 / 4 - gph * lam * r ** 4
        rg[mod] = dict(r=r, explicit_mu_dmu=expl, beta_term=bl1 * r ** 4 / 4, gamma_term=-gph * lam * r ** 4,
                       residual=resid, residual_over_tree_lam_r4=resid / (abs(lam) * r ** 4 / 4), gamma_phi=gph)
    rg["gamma_phi_Landau"] = C.gamma_phi(g2, gY, yt, "L")
    rg["gamma_phi_Coulomb"] = C.gamma_phi(g2, gY, yt, "C")
    rg["beta_lam_1L"] = bl1
    out["RG_one_loop_consistency_lam-0.010"] = rg
    out["BO_corrections_lam-0.010_SM"] = C.bo_corrections(np.array([1.5, 3.0, 5.0]), g2, gY, yt, lam)
    # potential-level relation E_MC(r) vs E_MU_CW(r + C(r)) (v11 couplings, lam = -0.010)
    pc = os.path.join(C.SCR, "C138_profiles.pkl")
    if os.path.exists(pc):
        with open(pc, "rb") as f:
            gp = pickle.load(f)[("v11", None)]
    else:
        gp = R.gauss_profile(C.V11["g2"], C.V11["gY"])
    g2, gY, yt = C.V11["g2"], C.V11["gY"], C.V11["yt"]
    rr = np.array([1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    Cr = C.C_shift(rr, g2, gY)
    rho = rr + Cr
    tree = lambda x: lam * x ** 4 / 4  # noqa: E731
    Ug = lambda x: gp["spl"](x) / x ** 2  # noqa: E731
    dmc = C.dV("MC", rr, lam, g2, gY, yt) - C.dV("MU_CW", rr, lam, g2, gY, yt)
    dmc1 = C.dV("MC1", rr, lam, g2, gY, yt) - C.dV("MU_CW", rr, lam, g2, gY, yt)
    sh_tm = tree(rho) + C.dV("MU_CW", rho, lam, g2, gY, yt) - tree(rr) - C.dV("MU_CW", rr, lam, g2, gY, yt)
    sh_g = Ug(rho) - Ug(rr)
    sh_c = 3 / (8 * rho ** 2) - 3 / (8 * rr ** 2)
    out["potential_relation_v11_lam-0.010"] = dict(
        r=rr, C=Cr, MC_minus_MUCW=dmc, MC1_minus_MUCW=dmc1, Vtree_prime_times_C=lam * rr ** 3 * Cr,
        shift_tree_plus_matching=sh_tm, shift_gauge_zero_modes=sh_g, shift_centrifugal=sh_c,
        full_shift_minus_MC=sh_tm + sh_g + sh_c - dmc, kinetic_factor_drho_dr_sq=(1 - Cr / rr) ** 2)
    # export for the Wolfram gamma check (MC, lam = -0.010, v11 couplings)
    rg_ = np.round(np.arange(0.01, 16.0 + 1e-9, 0.01), 6)
    U = C.U_col(gp, "MC", lam, g2, gY, yt, rg_)
    s, al = C.spar_xi(gp, 1.0)
    with open(os.path.join(C.RES, "wolfram_MC_U_v11_lam-0.010.json"), "w", encoding="utf-8") as f:
        json.dump(dict(r=rg_.tolist(), U=[float(x) for x in U], s=s, alpha=al, lam=lam, model="MC",
                       couplings=C.V11), f)
    from C138_run import wjson
    wjson("C138_checks.json", out)
    print(json.dumps({k: out[k] for k in ("Delta_subtraction_independence", "MC_minus_MUCW_at_lam0")},
                     default=float)[:3000])
    return out


if __name__ == "__main__":
    run()
