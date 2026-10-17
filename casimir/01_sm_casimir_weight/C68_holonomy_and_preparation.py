# PROVENANCE: 04_calculations/code/20261009_claude_R5/C68_holonomy_and_preparation.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] (a) N_C with flat SU(3)-centre and U(1)_Y holonomies (PPP centre 11.19, PPP hypercharge maximum 17.79, AAA centre 22.74 / 9.86, AAA maximal at zero holonomy) and (b) preparation-time sensitivity of the free-field aspect fossil. Paper: PRD Sec. IV B (sec:pa-spin) and App. app:smcasimir. Output <work>/r5/C68_holonomy_preparation/C68_results.json. No D file (R09 revision matrix, R10 row).

C68 (R5, R09 response): (a) Standard Model Casimir weight N_C with flat gauge holonomies, (b) sensitivity of the
shape fossil to the preparation instant.

(a) Twisted Ewald sums of C57 (Z2(G, theta) at the cube). Content (left-handed Weyl basis, per generation x3):
    Q (3x2, Y=1/6), u^c (3, Y=-2/3), d^c (3, Y=1/3), L (2, Y=-1/2), e^c (1, Y=1); Higgs 4 real (Y=1/2);
    gauge bosons (Y=0; gluons in the adjoint of SU(3)); static graviton +1. A flat holonomy along cycle i acts as
    exp(2 pi i theta_i) on a field: U(1)_Y gives theta_i = Y g_i; the SU(3) centre element exp(2 pi i k/3) gives
    theta_i = +-k_i/3 on (anti)triplets and 0 on gluons. Spin structure adds 1/2 on antiperiodic cycles for fermions.
    N_C = [sum_bosons d Z2(1,theta) - sum_fermions Z2(1,theta + spin)] / Z2(1,0).
    Checks reviewer numbers: PPP + centre k=(1,1,1) -> +11.19 (DA); PPP + U(1)_Y g=3/2 -> +7.13, max ~ +17.8 (R1);
    AAA + centre -> 22.74 / 9.86 (DA); AAA maximal at zero hypercharge holonomy (R1).
(b) Leading background L = L_b sqrt(cosh 2ht); shape mode eq. (18) of the paper. Uncorrelated Gaussian prepared at
    h t_p <= 0 with the local natural width s = 1/(2 m |omega|), m = M_P^2 L^3, |omega| = sqrt(kappa) h sech(2 h t_p),
    i.e. s/delta = 3/(2 sqrt2 sqrt(kappa)) (L_b/L)^3 cosh(2 h t_p); also the minimum over s. Outgoing variance in units
    of the floor delta |T12|. Checks DA numbers (natural width: 5.86, 24.4, 128 at ht = -0.25, -0.5, -1).
Output: results/20261009_claude_R5/C68_holonomy_preparation/C68_results.json
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

import C57_spin_structure_Casimir as c57

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C68_holonomy_preparation')
OUT.mkdir(parents=True, exist_ok=True)
G1 = np.eye(3)
_cache = {}


def Z2(theta):
    th = tuple(np.round(np.mod(np.asarray(theta, float), 1.0), 12))
    if th not in _cache:
        _cache[th] = float(c57.Z2(G1, np.array(th)))
    return _cache[th]


Z0 = Z2((0, 0, 0))
FERMIONS = [  # (multiplicity of Weyl fermions per generation, hypercharge, SU(3) rep: +1 triplet, -1 antitriplet, 0 singlet)
    (6, 1 / 6, +1), (3, -2 / 3, -1), (3, 1 / 3, -1), (2, -1 / 2, 0), (1, 1.0, 0)]


def NC(spin, gY=(0, 0, 0), kc=(0, 0, 0)):
    gY, kc, spin = np.asarray(gY, float), np.asarray(kc, float), np.asarray(spin, float)
    tot = 12 * Z0 + 1 * Z0                       # 12 gauge vectors (Y = 0; gluons untwisted by the centre) + graviton
    tot += 4 * 0.5 * Z2(0.5 * gY)                # Higgs: 4 real, Y = 1/2
    for mult, Y, rep in FERMIONS:
        th = Y * gY + rep * kc / 3.0 + spin
        tot -= 3 * mult * Z2(th)                 # three generations
    return tot / Z0


A3, P3 = (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)
res = {"Z2_cube": Z0, "NC_PPP": NC(P3), "NC_AAA": NC(A3)}
res["PPP_centre_111"] = NC(P3, kc=(1, 1, 1))
res["PPP_centre_100"] = NC(P3, kc=(1, 0, 0))
res["AAA_centre_111"] = NC(A3, kc=(1, 1, 1))
res["AAA_centre_100"] = NC(A3, kc=(1, 0, 0))
res["PPP_Y_1.5_iso"] = NC(P3, gY=(1.5, 1.5, 1.5))
gs = np.linspace(0, 6, 241)
scanP = [NC(P3, gY=(g, g, g)) for g in gs]
scanA = [NC(A3, gY=(g, g, g)) for g in gs]
iP = int(np.argmax(scanP)); iA = int(np.argmax(scanA))
res["PPP_Y_iso_scan_max"] = dict(g=float(gs[iP]), NC=float(scanP[iP]))
res["AAA_Y_iso_scan_max"] = dict(g=float(gs[iA]), NC=float(scanA[iA]))
res["AAA_Y_iso_scan_near0"] = [float(v) for v in scanA[:6]]
print(json.dumps(res, indent=1))

# (b) preparation-time sensitivity on the leading background (units h = 1)
C32 = 3 / 2 ** 1.5


def shape_rhs(kappa):
    # beta' = C32 Pi / l^3 ; Pi' = (kappa / C32) l^3 sech^2(2t) beta,  l = L/L_b = sqrt(cosh 2t)
    # (from m beta'' = ..., m = M_P^2 L^3, Pi = M_P^2 L^3 beta' / p0 with p0 = 3 M_P^2 h L_b^3 / 2^{3/2}).
    def f(t, y):
        l3 = np.cosh(2 * t) ** 1.5
        sech2 = 1.0 / np.cosh(2 * t) ** 2
        return [C32 * y[1] / l3, (kappa / C32) * l3 * sech2 * y[0], C32 * y[3] / l3, (kappa / C32) * l3 * sech2 * y[2]]
    return f


def M_from(kappa, t0, t1=12.0):
    s = solve_ivp(shape_rhs(kappa), (t0, t1), [1, 0, 0, 1], method="DOP853", rtol=1e-12, atol=1e-14)
    y = s.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def outgoing_var(kappa, tp, s_over_delta):
    M = M_from(kappa, tp)
    # Var beta_+ / delta for uncorrelated state Var beta = s, Var Pi = delta^2 / s
    return M[0, 0] ** 2 * s_over_delta + M[0, 1] ** 2 / s_over_delta


kE = 13.528578536527492
Mfull = M_from(kE, -12.0)
T12 = abs(Mfull[0, 1])
M0 = M_from(kE, 0.0)
floor = 2 * abs(M0[0, 0] * M0[0, 1])          # |T12| from time symmetry
prep = {"T12_direct": T12, "T12_time_symmetry": floor}
rows = []
for tp in (0.0, -0.1, -0.25, -0.5, -1.0, -2.0):
    l3 = np.cosh(2 * tp) ** 1.5
    s_nat = 3 / (2 * np.sqrt(2) * np.sqrt(kE)) / l3 * np.cosh(2 * tp)
    v_nat = outgoing_var(kE, tp, s_nat) / floor
    best = minimize_scalar(lambda ls: outgoing_var(kE, tp, np.exp(ls)), bounds=(-12, 6), method="bounded")
    rows.append(dict(ht_prep=tp, s_nat_over_delta=s_nat, var_over_floor_natural=v_nat,
                     rms_ratio_natural=float(np.sqrt(v_nat)), var_over_floor_min=float(best.fun / floor)))
    print(rows[-1])
prep["rows"] = rows
res["preparation_time"] = prep
(OUT / "C68_results.json").write_text(json.dumps(res, indent=1))
