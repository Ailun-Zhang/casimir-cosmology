# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/cross_core.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Zero-mode / non-zero-mode cross terms (D121): twisted image sums E_theta(m), curvature C(m), holonomy dependence of all SM fields (P/A spin structures, AAA fermions), kappa_3, kappa_2, kappa_Y.  Paper: PRD App. H2'. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C149 cross_core -- DNEC v12 round 3, agent holonomy-cross (T2, derivation D121).  2026-10-14.

Finite-volume (unit torus, L = 1) one-loop functions of a charged field in a constant (abelian) background
connection, per REAL degree of freedom, periodic (P, integer n) or AAA (A, n + 1/2):

  E_theta(m)  = 1/2 sum_n omega(2 pi (n + theta)) - continuum
              = -(m^2/(4 pi^2)) sum'_w K2(m|w|) cos(2 pi w.theta) / |w|^2      [AAA: x (-1)^{w1+w2+w3}]
              -> -(1/(2 pi^2)) sum'_w cos(2 pi w.theta)/|w|^4  (m -> 0)
  C(m)        = d^2 E_theta / d theta_i^2 |_0 = (1/3) m^2 sum'_w K2(m|w|)  [x sign]  (cubic symmetry, any i)
                periodic: the n = 0 mode is a slow coordinate (zero-mode dynamics) -> subtract 2 pi^2/m.
  small m:    C = (2/3) Z(1; bc) + m^2/6 + O(m^4),  Z(1;P) = pi Z_P(1/2) = -8.913633, Z(1;A) = -2.519356.
  direct sum: C = (2/3) Z(1) + 2 pi^2 sum'_n [ (3 m^2 + 2 k^2)/(3 w^3) - 2/(3k) ] + m^2/6     (valid for m^2 < 0 too)
              (+m^2/6 = minus the 3D continuum of the bracket; the Poisson form is shift invariant)
A field of charge q in a background phi (couplings absorbed) has theta = q phi/(2 pi):
  d^2E/dphi^2 = q^2 C(m)/(4 pi^2) per real dof; fermion: x(-1), Weyl spinor = 2 real dof, Dirac = 4.
v11 App. F:  L d^2E/dc^2 = (1/(3pi^2)) sum_f d_f s_f t_f Z(1; theta_f)  <=>  sum_{real dof} s q^2 (2/3) Z /(4pi^2).
Usage: python -B cross_core.py stageA
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
import time
import numpy as np
from scipy.special import kv, erfc

PI = math.pi
ROOT = _zp.WORK
RES = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run01"))
WMAX = 60          # image shells |w| <= WMAX
NMAX = 60          # momentum shells |n| <= NMAX


def _shells_int(R):
    n = np.arange(-R, R + 1)
    q = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    q = q[(q > 0) & (q <= R * R)]
    N = np.bincount(q)
    qs = np.nonzero(N)[0]
    return qs.astype(float), N[qs].astype(float)


def _shells_half(R):
    v = 2 * np.arange(-R - 1, R + 1) + 1
    q4 = (v[:, None, None] ** 2 + v[None, :, None] ** 2 + v[None, None, :] ** 2).ravel()
    q4 = q4[q4 <= 4 * R * R]
    N = np.bincount(q4)
    qs = np.nonzero(N)[0]
    return qs / 4.0, N[qs].astype(float)


QW, NW = _shells_int(WMAX)                 # image lattice w != 0
SW = (-1.0) ** QW                          # (-1)^{w1+w2+w3} = (-1)^{|w|^2}
QP, NP_ = _shells_int(NMAX)                # periodic momenta n != 0
QA, NA_ = _shells_half(NMAX)               # AAA momenta n + 1/2


def zeta_half():
    """Ewald: Z_P(1/2) = sum'_n |n|^-1, Z_A(1/2) = sum_n |n+1/2|^-1 (analytic continuation) -- R14 formula."""
    q, N = _shells_int(10)
    rr = np.sqrt(q)
    ZP = -3.0 + np.sum(N * erfc(np.sqrt(PI) * rr) / rr) + np.sum(N * np.exp(-PI * q) / (PI * q))
    qh, Nh = _shells_half(10)
    rh = np.sqrt(qh)
    sg = (-1.0) ** q
    ZA = np.sum(Nh * erfc(np.sqrt(PI) * rh) / rh) - 1.0 + np.sum(sg * N * np.exp(-PI * q) / (PI * q))
    return float(ZP), float(ZA)


ZP12, ZA12 = zeta_half()
Z1 = {"P": PI * ZP12, "A": PI * ZA12}      # Z(1;0) cusp removed, Z(1;theta_AAA)  (functional equation, self-dual)
Z2P = float(np.sum(NW / QW ** 2) + 4 * PI / WMAX)   # sum' |w|^-4 (tail ~ 4 pi / W)


def curv_bessel(m, bc):
    """C(m) per real dof from the image sum; periodic: n = 0 slow mode removed."""
    m = np.atleast_1d(np.asarray(m, float))
    sg = SW if bc == "A" else 1.0
    out = np.array([(mm * mm / 3.0) * np.sum(sg * NW * kv(2, mm * np.sqrt(QW))) for mm in m])
    if bc == "P":
        out = out - 2 * PI ** 2 / m
    return out


def curv_direct(m2, bc):
    """C(m^2) per real dof from the momentum sum (zeta-regularised 2/(3k), + m^2/6 continuum correction)."""
    m2 = np.atleast_1d(np.asarray(m2, float))
    q, N = (QA, NA_) if bc == "A" else (QP, NP_)
    k2 = 4 * PI ** 2 * q
    k = np.sqrt(k2)
    out = np.empty(len(m2))
    for i, mm in enumerate(m2):
        w = np.sqrt(k2 + mm)
        s = np.sum(N * ((3 * mm + 2 * k2) / (3 * w ** 3) - 2.0 / (3 * k)))
        tail = -(mm * mm / 4.0) * (2 * PI) ** -5 * 4 * PI / (2 * NMAX ** 2)
        out[i] = (2.0 / 3) * Z1[bc] + 2 * PI ** 2 * (s + tail) + mm / 6.0
    return out


def curv(m2, bc):
    """production curvature: direct sum (handles m^2 < 0); identical to the Bessel form for m^2 > 0 (checked)."""
    return curv_direct(m2, bc)


def energy_bessel(m, theta, bc):
    """E_theta(m) per real dof (all n, continuum removed), theta a 3-vector."""
    th = np.asarray(theta, float)
    n = np.arange(-WMAX, WMAX + 1)
    W = np.stack(np.meshgrid(n, n, n, indexing="ij"), -1).reshape(-1, 3)
    q = (W * W).sum(1)
    sel = (q > 0) & (q <= WMAX * WMAX)
    W, q = W[sel], q[sel].astype(float)
    ph = np.cos(2 * PI * W @ th)
    if bc == "A":
        ph = ph * (-1.0) ** q
    if m == 0:
        return float(-(1 / (2 * PI ** 2)) * np.sum(ph / q ** 2))
    return float(-(m * m / (4 * PI ** 2)) * np.sum(ph * kv(2, m * np.sqrt(q)) / q))


def v11_kappas():
    """r = 0 curvatures (c-units, per (a,i)) from the field content, per real dof s q^2 (2/3) Z /(4 pi^2)."""
    cP, cA = (2.0 / 3) * Z1["P"], (2.0 / 3) * Z1["A"]
    f = 1.0 / (4 * PI ** 2)
    # SU(3): gluons 2 pol x (2 x 1^2 + 4 x (1/2)^2) = 6 ; quarks 6 Dirac x 4 x (1/2) = 12 (AAA, s=-1)
    su3 = dict(gluon=6 * cP * f, quarks=-12 * cA * f)
    # SU(2) (T^3): W 2 pol x 2 x 1 = 4 ; Higgs doublet 4 x 1/4 = 1 ; 12 Weyl doublets x 2 comps x 2 x 1/4 = 12
    su2 = dict(W=4 * cP * f, higgs=1 * cP * f, fermions=-12 * cA * f)
    # U(1)_Y: Higgs 4 x 1/4 = 1 ; fermions 2 x sum_Weyl Y^2 = 2 x 3 x 10/3 = 20
    y = dict(higgs=1 * cP * f, fermions=-20 * cA * f)
    tot = {k: sum(v.values()) for k, v in (("SU3", su3), ("SU2", su2), ("Y", y))}
    spreads = {"SU3": 3.66, "SU2": 1.40, "Y": 0.82}
    dQ = {k: 0.5 * tot[k] * spreads[k] for k in tot}
    formula = {"SU3": (Z1["P"] - 2 * Z1["A"]) / PI ** 2, "SU2": (5 * Z1["P"] - 12 * Z1["A"]) / (6 * PI ** 2),
               "Y": (Z1["P"] - 20 * Z1["A"]) / (6 * PI ** 2)}
    return dict(parts=dict(SU3=su3, SU2=su2, Y=y), kappa=tot, kappa_closed_form=formula,
                v11_targets=dict(SU3=-0.393, SU2=-0.242, Y=0.700), dQ=dQ, dQ_total=sum(dQ.values()),
                dQ_v11=dict(SU3=-0.72, SU2=-0.17, Y=0.29, total=-0.60), dQ_SU3_virial=0.5 * tot["SU3"] * 3.77,
                vanBaal_coeff_half_kappa_SU3=0.5 * tot["SU3"], vanBaal_target=-0.19631,
                Higgs_only_Y_curvature_w_units=0.4415 ** 2 * y["higgs"], D112_Om_X2_small_r=-0.029)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def wjson(name, obj):
    p = os.path.join(RES, name)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def stageA():
    t0 = time.time()
    out = dict(Z_P_half=ZP12, Z_A_half=ZA12, Z1=Z1, targets=dict(Z1_P=-8.913633, Z1_A=-2.519356),
               Z2_sum_w4=Z2P, Z2_over_2pi2=Z2P / (2 * PI ** 2), D108_Z2_over_2pi2=0.8375369)
    out["v11"] = v11_kappas()
    # curvature: Bessel vs direct, small-m expansion
    ms = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
    tab = {}
    for bc in ("P", "A"):
        cb = curv_bessel(ms, bc)
        cd = curv_direct(np.array(ms) ** 2, bc)
        tab[bc] = {f"{m}": dict(bessel=float(b), direct=float(d), diff=float(b - d),
                                small_m=(2.0 / 3) * Z1[bc] + m * m / 6) for m, b, d in zip(ms, cb, cd)}
        tab[bc]["m2=-1"] = dict(direct=float(curv_direct(-1.0, bc)[0]), small_m=(2.0 / 3) * Z1[bc] - 1 / 6.0)
        tab[bc]["m2=0"] = dict(direct=float(curv_direct(0.0, bc)[0]), two_thirds_Z1=(2.0 / 3) * Z1[bc])
    out["curvature"] = tab
    # energy check vs R14 D_P(m^2):  D_P = E_bes(m,0) - m/2 + Z2/(2 pi^2)
    sys.path.insert(0, os.path.join(ROOT, _zp.ZMLIB))
    try:
        import R14_fv
        fv = R14_fv.FV()
        ec = {}
        for m in (1.0, 2.0):
            eb = energy_bessel(m, (0, 0, 0), "P") - m / 2 + Z2P / (2 * PI ** 2)
            ea = energy_bessel(m, (0, 0, 0), "A") - energy_bessel(0, (0, 0, 0), "A")
            ec[f"{m}"] = dict(DP_bessel=eb, DP_R14=float(fv.DP(m * m)[0]), DA_bessel=ea, DA_R14=float(fv.DA(m * m)[0]))
        out["energy_check_vs_R14"] = ec
    except Exception as e:  # noqa: BLE001
        out["energy_check_vs_R14"] = repr(e)
    # curvature from finite differences of the twisted image-sum energy (AAA, m=1, theta along (1,0,0) and (1,1,1))
    h = 1e-3
    fd = {}
    for d in ((1, 0, 0), (1, 1, 1)):
        u = np.array(d, float) / np.linalg.norm(d)
        e0 = energy_bessel(1.0, 0 * u, "A")
        ep = energy_bessel(1.0, h * u, "A")
        em = energy_bessel(1.0, -h * u, "A")
        fd[str(d)] = (ep - 2 * e0 + em) / h ** 2
    out["fd_curvature_AAA_m1"] = dict(fd=fd, bessel=float(curv_bessel(1.0, "A")[0]))
    out["time_s"] = time.time() - t0
    wjson("C149_stageA.json", out)
    print(json.dumps(clean(out), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "stageA":
        stageA()
