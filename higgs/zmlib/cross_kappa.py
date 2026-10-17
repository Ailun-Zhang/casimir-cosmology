# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/cross_kappa.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] One-loop Hessians H11, H33, H3w, Hww(r) of the constant gauge modes in the Higgs background, split by field (W, Higgs, top, light fermions), CCA.  Paper: PRD App. H2'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 stage B -- r-dependent one-loop Hessian of the gauge zero modes (holonomy-cross, D121 Sec. 2).

Body frame (D108 convention: vev component T3=+1/2, Y=1/2; top-type component T3=-1/2), constant Cartan
background g2 x3 T3 + gY w Y (+ x1, x2 non-Cartan, + gluon c).  Per real dof (cross_core): d2E/dphi^2 = s q^2 C(m^2)/(4pi^2).
Charged-component approximation (CCA): every real dof carries its own mass and its charges; Coulomb modes carry the
Goldstone charges with m^2 = a + b (w_C^2 = (k^2+a)(k^2+b)/k^2 = k^2 + a + b + O(ab/k^2)); top vector-like
(each chirality decouples with m_t; axial remainder O(q_A^2 m^2/16pi^2), estimated in D121).
Hessian per spatial direction (canonical x, w):
  H33 = g2^2 sum s n T3^2 C/(4pi^2),  Hww = gY^2 sum s n Y^2 C/(4pi^2),  H3w = g2 gY sum s n T3 Y C/(4pi^2),
  H11: doublets as H33 (CCA, partner map is a bijection); W transverse: 2 pol x [C(aW) + cW^2 C(aZ) + sW^2 C(0)].
2D frame (D112): x3 = -beta w + dx3, <dx3^2> = 1/(2k), <x1^2> = <x2^2> = 1/(2a) per direction:
  E_X(r,w) = E_X^x(r) + 1/2 Om_X^2(r) |w|^2,  E_X^x = (3/2) H11/a + (3/4) H33/k,  Om_X^2 = Hww - 2 beta H3w + beta^2 H33.
SU(3): kappa3(r) = [6 C_P(0) - 2(5 C_A(0) + C_A(m_t^2))]/(4pi^2) (c-units), DeltaU_3 = 1/2 [kappa3(r)-kappa3(0)] <c^2>_3.
Usage: python -B cross_kappa.py stageB
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_core as X  # noqa: E402

PI = math.pi
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585)
PROFILE = os.path.join(X.ROOT, _zp.w("r2/higgs_beyond_bo/run01/C140_profile_v11.npz"))
C3_GAUSS, C3_VIRIAL = 3.66, 3.77
F4 = 1.0 / (4 * PI ** 2)

# light Weyl spinors (AAA, s=-1, 2 real dof each): (name, multiplicity, T3 (D108 convention), Y)
LIGHT = []
for gen in (1, 2, 3):
    LIGHT += [("uL%d" % gen, 3, -0.5, 1 / 6.), ("dL%d" % gen, 3, 0.5, 1 / 6.), ("dR%d" % gen, 3, 0.0, -1 / 3.),
              ("nuL%d" % gen, 1, -0.5, -0.5), ("eL%d" % gen, 1, 0.5, -0.5), ("eR%d" % gen, 1, 0.0, -1.0)]
    if gen < 3:
        LIGHT += [("uR%d" % gen, 3, 0.0, 2 / 3.)]
LIGHT = [x for x in LIGHT if x[0] != "uL3"]          # t_L is massive
TOP = [("tL", 3, -0.5, 1 / 6.), ("tR", 3, 0.0, 2 / 3.)]


def derived_profile():
    d = np.load(PROFILE)
    rf, P, dP = d["rf"], d["P"], d["dP"]
    a, k, b, c = P
    ap, kp, bp, cp = dP
    beta = c / k
    bt = (k * b - c * c) / k
    return dict(rf=rf, a=a, k=k, beta=beta, bt=bt)


def species(r, lam, g2=V11["g2"], gY=V11["gY"], yt=V11["yt"]):
    """list of (name, group, n_real, s, T3, Y, m2, bc) at radius r (scalar)."""
    aW, aZ, b = g2 * g2 / 4 * r * r, (g2 * g2 + gY * gY) / 4 * r * r, lam * r * r
    mt2 = yt * yt / 2 * r * r
    sp = [("W_T+", "W", 2 * 2 // 2, 1, 1.0, 0.0, aW, "P"), ("W_T-", "W", 2, 1, -1.0, 0.0, aW, "P"),
          ("pi_pm", "H", 2, 1, -0.5, 0.5, aW + b, "P"), ("pi_3", "H", 1, 1, 0.5, 0.5, aZ + b, "P"),
          ("h", "H", 1, 1, 0.5, 0.5, 3 * lam * r * r, "P")]
    for nm, mult, t3, y in TOP:
        sp.append((nm, "top", 2 * mult, -1, t3, y, mt2, "A"))
    for nm, mult, t3, y in LIGHT:
        sp.append((nm, "light", 2 * mult, -1, t3, y, 0.0, "A"))
    return sp, dict(aW=aW, aZ=aZ, b=b, mt2=mt2)


_CC = {}


def C(m2, bc):
    key = (round(float(m2), 12), bc)
    if key not in _CC:
        _CC[key] = float(X.curv(m2, bc)[0])
    return _CC[key]


def hessian(r, lam, g2=V11["g2"], gY=V11["gY"], yt=V11["yt"]):
    sp, m = species(r, lam, g2, gY, yt)
    gw = g2 * g2 + gY * gY
    cW2, sW2 = g2 * g2 / gw, gY * gY / gw
    parts = {}
    for nm, grp, n, s, t3, y, m2, bc in sp:
        cc = C(m2, bc)
        p = parts.setdefault(grp, dict(H33=0.0, Hww=0.0, H3w=0.0, H11=0.0))
        p["H33"] += g2 * g2 * s * n * t3 * t3 * cc * F4
        p["Hww"] += gY * gY * s * n * y * y * cc * F4
        p["H3w"] += g2 * gY * s * n * t3 * y * cc * F4
        if grp != "W":
            p["H11"] += g2 * g2 * s * n * t3 * t3 * cc * F4
    parts["W"]["H11"] = g2 * g2 * 2 * (C(m["aW"], "P") + cW2 * C(m["aZ"], "P") + sW2 * C(0.0, "P")) * F4
    tot = {key: sum(parts[g][key] for g in parts) for key in ("H33", "Hww", "H3w", "H11")}
    k3 = (6 * C(0.0, "P") - 2 * (5 * C(0.0, "A") + C(m["mt2"], "A"))) * F4
    k3_0 = (6 * C(0.0, "P") - 12 * C(0.0, "A")) * F4
    return dict(parts=parts, tot=tot, kappa3=k3, dU3_gauss=0.5 * (k3 - k3_0) * C3_GAUSS,
                dU3_virial=0.5 * (k3 - k3_0) * C3_VIRIAL, masses=m)


def d110_higgs_only(r, lam, prof):
    """D110 Sec. 3.6 / D112 Sec. 6 Higgs-only formula (f = 2/3, FV fluctuations from C138_core) for regression (c)."""
    sys.path.insert(0, os.path.join(X.ROOT, _zp.ZMLIB))
    import C138_core as Cc
    g2, gY, f = V11["g2"], V11["gY"], 2.0 / 3
    a, k = np.interp(r, prof["rf"], prof["a"]), np.interp(r, prof["rf"], prof["k"])
    beta, bt = np.interp(r, prof["rf"], prof["beta"]), np.interp(r, prof["rf"], prof["bt"])
    m = Cc.masses(r, lam, g2, gY, 0.0)
    eps = 1e-6
    h2 = (Cc.estd(m["mh2"] + eps) - Cc.estd(m["mh2"] - eps)) / eps
    pW = 2 * Cc.ddelta_db0(g2 * g2 / 4 * r * r)
    pZ = 2 * Cc.ddelta_db0((g2 * g2 + gY * gY) / 4 * r * r)
    half = f / 8 * ((h2 + pZ) * (gY - g2 * beta) ** 2 + 2 * pW * (gY + g2 * beta) ** 2)
    EXx = 0.5 * f * (h2 + pZ + 2 * pW) * 3.0 / 8 * (2 * g2 * g2 / a + g2 * g2 / k)
    return EXx, 2 * half, EXx + half * 1.5 / bt


def stageB():
    prof = derived_profile()
    rg = np.round(np.concatenate([np.arange(0.02, 1.0, 0.02), np.arange(1.0, 8.0001, 0.05)]), 4)
    out = dict(r=rg.tolist(), lams=[-0.008, -0.010, -0.0143])
    arrays = {"r": rg}
    for lam in out["lams"]:
        rows = [hessian(float(x), lam) for x in rg]
        a = np.interp(rg, prof["rf"], prof["a"])
        k = np.interp(rg, prof["rf"], prof["k"])
        beta = np.interp(rg, prof["rf"], prof["beta"])
        tag = f"{lam}"
        H = {key: np.array([x["tot"][key] for x in rows]) for key in ("H33", "Hww", "H3w", "H11")}
        EXx = 1.5 * H["H11"] / a + 0.75 * H["H33"] / k
        OmX2 = H["Hww"] - 2 * beta * H["H3w"] + beta ** 2 * H["H33"]
        dU3 = np.array([x["dU3_gauss"] for x in rows])
        dU3v = np.array([x["dU3_virial"] for x in rows])
        grp = {}
        for g in ("W", "H", "top", "light"):
            Hg = {key: np.array([x["parts"][g][key] for x in rows]) for key in ("H33", "Hww", "H3w", "H11")}
            grp[g] = dict(EXx=1.5 * Hg["H11"] / a + 0.75 * Hg["H33"] / k,
                          OmX2=Hg["Hww"] - 2 * beta * Hg["H3w"] + beta ** 2 * Hg["H33"], **Hg)
        EXx_d, OmX2_d, EXbo_d = d110_higgs_only(rg, lam, prof)
        # Om_eff^2 of D112 (2D frame) for the instability test
        dd = np.load(PROFILE)
        sys.path.insert(0, os.path.join(X.ROOT, _zp.ZMLIB))
        import C140_beyond_bo as B
        q = B.derived(dd["rf"], dd["P"], dd["dP"], V11["g2"], V11["gY"])
        om2e = np.interp(rg, dd["rf"], q["Om2e"])

        def r_inv(tot):
            neg = np.where(tot < 0)[0]
            return float(rg[neg[-1]]) if len(neg) else None
        arrays.update({f"{tag}|{key}": v for key, v in H.items()})
        arrays.update({f"{tag}|EXx": EXx, f"{tag}|OmX2": OmX2, f"{tag}|dU3": dU3, f"{tag}|dU3v": dU3v,
                       f"{tag}|EXx_D110": EXx_d, f"{tag}|OmX2_D110": OmX2_d, f"{tag}|om2e": om2e})
        for g in grp:
            arrays[f"{tag}|EXx_{g}"] = grp[g]["EXx"]
            arrays[f"{tag}|OmX2_{g}"] = grp[g]["OmX2"]
        sel = [0.02, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        tab = {}
        for s_ in sel:
            i = int(np.argmin(np.abs(rg - s_)))
            tab[f"{rg[i]:.2f}"] = dict(H33=H["H33"][i], H11=H["H11"][i], Hww=H["Hww"][i], H3w=H["H3w"][i],
                                       EXx=EXx[i], OmX2=OmX2[i], dU3=dU3[i], kappa3=rows[i]["kappa3"],
                                       EXx_by_group={g: grp[g]["EXx"][i] for g in grp},
                                       OmX2_by_group={g: grp[g]["OmX2"][i] for g in grp},
                                       Higgs_only_D110=dict(EXx=EXx_d[i], OmX2=OmX2_d[i], EX_BO=EXbo_d[i]),
                                       om2e=om2e[i], beta=beta[i], m_t=math.sqrt(rows[i]["masses"]["mt2"]))
        out[tag] = dict(table=tab,
                        r_inverted_Higgs_only_D110=r_inv(om2e + OmX2_d),
                        r_inverted_Higgs_only_exactC=r_inv(om2e + grp["H"]["OmX2"]),
                        r_inverted_full_SM=r_inv(om2e + OmX2),
                        min_OmX2_full=float(OmX2.min()), min_total_full=float((om2e + OmX2).min()),
                        max_abs_Higgs_exactC_minus_D110_OmX2=float(np.max(np.abs(grp["H"]["OmX2"] - OmX2_d))),
                        max_abs_Higgs_exactC_minus_D110_EXx=float(np.max(np.abs(grp["H"]["EXx"] - EXx_d))))
        print(lam, "r_inv D110/exactC/full:", out[tag]["r_inverted_Higgs_only_D110"],
              out[tag]["r_inverted_Higgs_only_exactC"], out[tag]["r_inverted_full_SM"], flush=True)
    r0 = hessian(1e-6, -0.010)
    out["r0_check"] = dict(H33_over_g2sq=r0["tot"]["H33"] / V11["g2"] ** 2, H11_over_g2sq=r0["tot"]["H11"] / V11["g2"] ** 2,
                           Hww_over_gYsq=r0["tot"]["Hww"] / V11["gY"] ** 2, H3w=r0["tot"]["H3w"],
                           kappa3=r0["kappa3"], targets=dict(SU2=-0.24209, Y=0.70036, SU3=-0.39261))
    np.savez(os.path.join(X.RES, "C149_stageB_arrays.npz"), **arrays)
    X.wjson("C149_stageB.json", out)
    print(json.dumps(X.clean({k: v for k, v in out.items() if k not in ("r",)}), indent=1)[:9000])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "stageB":
        stageB()
