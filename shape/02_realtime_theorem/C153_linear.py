# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_linear.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] D125 linear layer: closed-form floor K(kappa) = Gamma(7/4)Gamma(3/4)/(2D|Gamma(5/4+i mu)|^2) and transfer matrix (m11 = 10.554, m12 = 2.4685, m21 = 136.37, m22 = 31.99 at kappa = 19.3237), G tables (leading, next order, T2), monotone-correlation lemma, never-real / phase / adiabatic families, floor conventions (sigma_floor = 0.1037 H_eff convention, 0.0934 next order), shear criterion 3^-3/2. Paper: PRD Sec. VIII A-B (eq:s-floor, eq:s-T12, thm:s-real), App. app:theorem.

C153_linear (shape-hardening, DNEC v12 round 3; derivation D125): linear / closed-form layer.

Leading cosh background L = L_b sqrt(cosh 2t) (t in units 1/H_eff), shape mode
    beta'' + 3 tanh(2t) beta' - kappa sech^2(2t) beta = 0,
C60 normalisation Pi = L^3 beta'/C32 (C32 = 3/2^{3/2}), [beta, Pi] = 2 i delta, delta = 2^{3/2}/(6 X),
X = M_P^2 L_b^3 H_eff.  Transfer Phi(t0 -> t1) on (beta, Pi).
Outputs run01/linear/C153_linear.json:
  1. closed-form floor K(kappa) = Gamma(7/4)Gamma(3/4)/(2 D |Gamma(5/4+i mu)|^2) (D109) vs numerics
     2 |m11 m12| delta X, gain 2*cal G = 2 sqrt2 cosh(pi mu); kappa_E = 19.3237 (C76 r=0.299897), 19.32, 18.0;
  2. G(t_x) tables (two-branch bound G(|t_x|); single-branch future bound G_+(t_x>0)), E and T2;
  3. floor / sigma_E for every X convention in use (D109, V12 directive, C60/C76 delta, v11 table delta);
  4. monotone-correlation lemma check, phase family at the turn (classification: t_x root or never-real class),
     first-order adiabatic family on the contracting branch, linear two-branch fossils;
  5. D109 Prop. 2 shear criterion: exact turning-point bound 3^{-3/2} = 0.19245 and N_pre crossings.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_hardening/run01/linear")
C32 = 3.0 / 2 ** 1.5
KAP_FREE = np.array([13.528581, -6.352387])          # C76 free kappa (E, T2)
R_SM = 0.299897
DELTA0 = 1.5897991e-4                                 # C60 free-benchmark delta (leading background)
TF = 16.0


def phi(kappa, t0, t1):
    """2x2 transfer of (beta, Pi) from t0 to t1 (either direction)."""
    def rhs(t, y):
        c = np.cosh(2 * t)
        return [C32 * y[1] / c ** 1.5, kappa * c ** -0.5 * y[0] / C32,
                C32 * y[3] / c ** 1.5, kappa * c ** -0.5 * y[2] / C32]
    s = solve_ivp(rhs, (t0, t1), [1.0, 0.0, 0.0, 1.0], method="DOP853", rtol=1e-13, atol=1e-16)
    b1, p1, b2, p2 = s.y[:, -1]
    return np.array([[b1, b2], [p1, p2]])


def K_closed(kappa):
    mu = mp.sqrt(4 * mp.mpf(kappa) - 1) / 4
    D = 3 * mp.sqrt(2) / 4
    return float(mp.gamma(mp.mpf(7) / 4) * mp.gamma(mp.mpf(3) / 4) / (2 * D * abs(mp.gamma(mp.mpf(5) / 4 + 1j * mu)) ** 2))


def gain_closed(kappa):
    mu = mp.sqrt(4 * mp.mpf(kappa) - 1) / 4
    return float(mp.sqrt(2) * mp.cosh(mp.pi * mu))


def pm(kappa, t0):
    M = phi(kappa, t0, TF)
    return abs(M[0, 0] * M[0, 1]), M


def Xof(delta):
    return 2 ** 1.5 / (6 * delta)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"code": Path(__file__).name, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "conventions": "C60 normalisation; delta = 2^{3/2}/(6X); t in 1/H_eff"}
    kE30 = KAP_FREE[0] / (1 - R_SM)
    kT30 = KAP_FREE[1] / (1 - R_SM)
    r_f = 1 - KAP_FREE[0] / 18.0                     # rescaled-force completion of the next order
    kT_nx = KAP_FREE[1] / (1 - r_f)
    res["models"] = {"xi030": {"r": R_SM, "kappa_E": kE30, "kappa_T2": kT30},
                     "next_order": {"r_force": r_f, "xi_delta": 0.28, "kappa_E": 18.0, "kappa_T2_rescaled": kT_nx,
                                    "kappa_T2_v11": -8.34}}
    # ---------------------------------------------------------------- 1. floor closed form vs numerics
    fl = {}
    for lab, k in (("19.3237", kE30), ("19.32", 19.32), ("18.79(xi=0.28,no Hessian)", KAP_FREE[0] / 0.72), ("18.0", 18.0)):
        p0, M0 = pm(k, 0.0)
        Kn = 2 * p0 * Xof(1.0) * 1.0                  # 2|m11 m12| delta X with delta = 2^{3/2}/(6X): K = 2 p0 2^{3/2}/6
        Kn = 2 * p0 * 2 ** 1.5 / 6
        pinf, _ = pm(k, -TF)
        fl[lab] = {"kappa": k, "K_closed": K_closed(k), "K_numeric": Kn, "rel_diff": Kn / K_closed(k) - 1,
                   "m11": M0[0, 0], "m12": M0[0, 1], "m21": M0[1, 0], "m22": M0[1, 1],
                   "gain_calG_closed": gain_closed(k), "m11_numeric": M0[0, 0],
                   "G_inf_numeric": pinf / p0, "G_inf_closed_2calG": 2 * gain_closed(k)}
    res["floor_closed_form"] = fl
    # ---------------------------------------------------------------- 2. G tables
    taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 2.0, 4.0, 8.0]
    gt = {}
    for lab, k in (("E_xi030", kE30), ("E_19.32", 19.32), ("E_next18.0", 18.0), ("T2_xi030", kT30),
                   ("T2_next_rescaled", kT_nx), ("T2_v11_-8.34", -8.34)):
        p0, _ = pm(k, 0.0)
        F = 0.5 * float(np.arcsinh(np.sqrt(abs(k))))
        rows = []
        for tau in taus + [F]:
            pmn, _ = pm(k, -tau)
            ppl, _ = pm(k, +tau)
            rows.append({"tau": tau, "G_two_branch": pmn / p0, "sqrtG": np.sqrt(pmn / p0), "G_plus_single_branch": ppl / p0})
        rows.sort(key=lambda r: r["tau"])
        gt[lab] = {"kappa": k, "freeze_start": F, "rows": rows}
    res["G_tables"] = gt
    # ---------------------------------------------------------------- 3. floors for every X convention
    LB, HEFFLB, H_PL = 722.508, 0.00437243, 6.116e-6
    d30 = DELTA0 * (1 - R_SM) ** -0.75
    Xs = {"X_V12directive_exact(L_b=722.508,H_eff L_b=0.00437243)": LB ** 2 * HEFFLB,
          "X_D109(722^2*0.00437243)": 722.0 ** 2 * HEFFLB,
          "X_C60C76_delta(DELTA0*(1-r)^-3/4)": Xof(d30),
          "X_v11_table_delta(2.04e-4)": Xof(2.04e-4),
          "X_plateau_h(722.508^3*6.116e-6)": LB ** 3 * H_PL,
          "X_C152_pair(718.57^2*0.0043949)": 718.57 ** 2 * 0.0043949}
    K30, K1932 = K_closed(kE30), K_closed(19.32)
    res["floor_conventions"] = {k: {"X": v, "delta": 2 ** 1.5 / (6 * v), "sigma_floor_kappa19.3237": np.sqrt(K30 / v),
                                    "sigma_floor_kappa19.32": np.sqrt(K1932 / v)} for k, v in Xs.items()}
    res["H_eff_over_h"] = {"phi_b": 5.5794, "1-exp(-sqrt(2/3)phi_b)": 1 - np.exp(-np.sqrt(2 / 3) * 5.5794),
                           "H_eff/M_P_from_pipeline": HEFFLB / LB, "h/M_P_v11": H_PL,
                           "ratio_pipeline_over_v11": HEFFLB / LB / H_PL,
                           "H_eff/M_P_from_h": H_PL * (1 - np.exp(-np.sqrt(2 / 3) * 5.5794))}
    XE = LB ** 2 * HEFFLB
    nx = {}
    for lab, xi, k in (("xi030_kappa19.3237", R_SM, kE30), ("xi028_kappa18.0(next)", 0.28, 18.0),
                       ("xi028_kappa18.79(energy only)", 0.28, KAP_FREE[0] / 0.72),
                       ("xi02826_kappa18.0(y_t=0.428)", 0.2826, 18.0)):
        X = XE * ((1 - xi) / (1 - R_SM)) ** 0.75
        Xc = Xof(d30) * ((1 - xi) / (1 - R_SM)) ** 0.75
        nx[lab] = {"xi": xi, "kappa_E": k, "K": K_closed(k), "X_exact": X, "sigma_floor_exact": np.sqrt(K_closed(k) / X),
                   "X_C76conv": Xc, "sigma_floor_C76conv": np.sqrt(K_closed(k) / Xc),
                   "delta_C76conv": DELTA0 * (1 - xi) ** -0.75}
    base = nx["xi030_kappa19.3237"]
    for v in nx.values():
        v["ratio_to_xi030"] = v["sigma_floor_exact"] / base["sigma_floor_exact"]
    res["next_order_floor"] = nx
    # ---------------------------------------------------------------- 4. monotone correlation, phase & adiabatic families
    delta = d30
    k = kE30
    p0, M0 = pm(k, 0.0)
    s_opt0 = abs(M0[0, 1] / M0[0, 0]) * delta
    s_nat0 = C32 * delta / np.sqrt(k)
    tgrid = np.linspace(-TF, TF, 641)
    Phis = {float(t): phi(k, 0.0, float(t)) for t in tgrid}

    def cov_at(S0, t):
        P = Phis[float(t)] if float(t) in Phis else phi(k, 0.0, t)
        return P @ S0 @ P.T

    # lemma check: random states at t=0, C(t) monotone
    rng = np.random.default_rng(20261499)
    mono = []
    for _ in range(50):
        a = rng.normal(size=(2, 2))
        S0 = a @ a.T + 1e-3 * np.eye(2)
        Cs = np.array([cov_at(S0, t)[0, 1] for t in tgrid])
        mono.append(bool(np.all(np.diff(Cs) > 0)))
    res["monotone_lemma_check"] = {"n_random_states": 50, "all_strictly_increasing": all(mono)}

    def classify(S0):
        Cs = np.array([cov_at(S0, t)[0, 1] for t in tgrid])
        Cm, Cp = Cs[0], Cs[-1]
        out = {"C_minus_inf_over_delta": Cm / delta, "C_plus_inf_over_delta": Cp / delta}
        if Cm < 0 < Cp:
            i = int(np.flatnonzero(np.diff(np.sign(Cs)) != 0)[0])
            tx = brentq(lambda t: cov_at(S0, t)[0, 1], tgrid[i], tgrid[i + 1], xtol=1e-12)
            out.update(cls="real_once", t_x=tx)
        elif Cm >= 0:
            out.update(cls="never_real_outgoing(C>0 all t)", t_x=None)
        else:
            out.update(cls="never_real_incoming(C<0 all t)", t_x=None)
        Sp = M0 @ S0 @ M0.T                                  # future fossil (t -> +inf)
        Mm = phi(k, 0.0, -TF)
        Sm = Mm @ S0 @ Mm.T                                  # past fossil (t -> -inf)
        fl2 = 2 * p0 * delta
        out.update(sigma_plus=np.sqrt(Sp[0, 0]), sigma_minus=np.sqrt(Sm[0, 0]),
                   var_plus_over_floor=Sp[0, 0] / fl2, var_minus_over_floor=Sm[0, 0] / fl2,
                   max_over_floor=max(Sp[0, 0], Sm[0, 0]) / fl2)
        return out
    fam = []
    for ph in (-50, -20, -10, -7, -5, -4, -3, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 4,
               5, 7, 10, 20, 50):
        c = ph * delta
        S0 = np.array([[s_opt0, c], [c, (delta ** 2 + c ** 2) / s_opt0]])
        fam.append({"phi": ph, **classify(S0)})
    # boundary phases where t_x -> -inf / +inf
    def Cm_of(ph):
        c = ph * delta
        S0 = np.array([[s_opt0, c], [c, (delta ** 2 + c ** 2) / s_opt0]])
        return cov_at(S0, tgrid[0])[0, 1]

    def Cp_of(ph):
        c = ph * delta
        S0 = np.array([[s_opt0, c], [c, (delta ** 2 + c ** 2) / s_opt0]])
        return cov_at(S0, tgrid[-1])[0, 1]
    phs = np.linspace(-60, 60, 1201)
    bm = [float(brentq(Cm_of, phs[i], phs[i + 1])) for i in range(len(phs) - 1) if Cm_of(phs[i]) * Cm_of(phs[i + 1]) < 0]
    bp = [float(brentq(Cp_of, phs[i], phs[i + 1])) for i in range(len(phs) - 1) if Cp_of(phs[i]) * Cp_of(phs[i + 1]) < 0]
    # optimum single-branch focusing phase: minimise future variance
    m11, m12 = M0[0, 0], M0[0, 1]
    c_foc = -m11 * s_opt0 / m12
    res["phase_family"] = {"kappa_E": k, "delta": delta, "s_opt0": s_opt0, "s_nat0": s_nat0, "s_opt0_over_s_nat0": s_opt0 / s_nat0,
                           "floor_var": 2 * p0 * delta, "sigma_floor": np.sqrt(2 * p0 * delta), "rows": fam,
                           "phi_boundaries_C_minus_inf_zero": bm, "phi_boundaries_C_plus_inf_zero": bp,
                           "phi_focus_single_branch_opt": c_foc / delta,
                           "note": "phase applied to E modes; s = s_opt(0); classification by the sign of C at t = -/+ TF"}
    # first-order adiabatic family on the contracting branch (prepared at t_in < 0)
    ad = []
    for t_in in (-0.25, -0.5, -1.0):
        for lab, kk in (("E", kE30), ("T2", kT30)):
            c2 = np.cosh(2 * t_in)
            m = c2 ** 1.5
            H = np.tanh(2 * t_in)
            wl = np.sqrt(abs(kk)) / c2
            W, dlnW = (wl, -2 * H) if wl >= abs(H) else (abs(H), 2 / c2 ** 2 / H)
            g = 3 * H + dlnW
            Vb = delta * C32 / (m * W)
            Vp = delta * m * (W ** 2 + g ** 2 / 4) / (C32 * W)
            Cc = -delta * g / (2 * W)
            S = np.array([[Vb, Cc], [Cc, Vp]])
            Mf = phi(kk, t_in, TF)
            Sf = Mf @ S @ Mf.T
            pt0, _ = pm(kk, 0.0)
            ptin, Mtin = pm(kk, t_in)
            row = {"t_in": t_in, "mode": lab, "kappa": kk, "W": W, "W_is_omega_loc": bool(wl >= abs(H)), "g": g,
                   "Var_beta": Vb, "Var_Pi": Vp, "Cov_over_delta": Cc / delta, "purity_check": (Vb * Vp - Cc ** 2) / delta ** 2,
                   "sigma_plus": np.sqrt(Sf[0, 0]), "var_plus_over_floor": Sf[0, 0] / (2 * pt0 * delta),
                   "G_real_opt_at_t_in": ptin / pt0}
            if lab == "E":
                Mb = phi(kk, t_in, -TF)
                Sb = Mb @ S @ Mb.T
                row["C_minus_inf_over_delta"] = Sb[0, 1] / delta
                row["real_somewhere_before_t_in"] = bool(Sb[0, 1] < 0)
            ad.append(row)
    res["adiabatic_family"] = ad
    # ---------------------------------------------------------------- 5. shear criterion 0.192 (D109 Prop. 2)
    X109 = 722.0 ** 2 * HEFFLB
    d109 = 2 ** 1.5 / (6 * X109)
    kk2 = (19.32, 19.32, -9.07, -9.07, -9.07)

    def ratio(N_pre, mode, c=1.0):
        tx = -0.5 * np.arccosh(np.exp(2 * N_pre))
        tot = 0.0
        for kk in kk2:
            m = np.cosh(2 * tx) ** 1.5
            if mode == "adiabatic":
                W = c * abs(np.tanh(2 * tx))
                S = np.diag([d109 * C32 / (m * W), d109 * m * W / C32])
            else:
                Mx = phi(kk, tx, TF)
                so = abs(Mx[0, 1] / Mx[0, 0]) * d109
                S = np.diag([so, d109 ** 2 / so])
            P = phi(kk, tx, 0.0)
            S0 = P @ S @ P.T
            tot += C32 ** 2 * S0[1, 1] / 12.0
        return tot
    crit = 3 ** -1.5
    sh = {"exact_bound_3^-1.5": crit, "regression_D109": {}, "crossings": {}}
    for N in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        sh["regression_D109"][str(N)] = {"c1": ratio(N, "adiabatic", 1.0), "c3": ratio(N, "adiabatic", 3.0), "opt": ratio(N, "opt")}
    for lab, f in (("c1", lambda N: ratio(N, "adiabatic", 1.0)), ("c3", lambda N: ratio(N, "adiabatic", 3.0))):
        sh["crossings"][lab] = {"N_pre_at_0.19245": brentq(lambda N: f(N) - crit, 0.3, 3.0, xtol=1e-6),
                                "N_pre_at_1": brentq(lambda N: f(N) - 1.0, 0.3, 4.0, xtol=1e-6)}
    sh["opt_N8"] = ratio(8.0, "opt")
    # chi^2 tail for single trajectories of the optimal family at large N_pre (E doublet dominates)
    res["shear_criterion"] = sh
    (OUT / "C153_linear.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    # ---------------------------------------------------------------- print summary
    for lab, v in fl.items():
        print("FLOOR %-26s K=%.6f num=%.6f (rel %.1e)  calG=%.3f  G_inf num=%.3f closed=%.3f  m11=%.4f m12=%.4f m21=%.4f m22=%.4f" % (
            lab, v["K_closed"], v["K_numeric"], v["rel_diff"], v["gain_calG_closed"], v["G_inf_numeric"], v["G_inf_closed_2calG"],
            v["m11"], v["m12"], v["m21"], v["m22"]))
    for lab, v in gt.items():
        print("G %-18s F=%.4f " % (lab, v["freeze_start"]) + " ".join("%.3g:%.4g/%.3g" % (r["tau"], r["G_two_branch"], r["G_plus_single_branch"]) for r in v["rows"]))
    for k_, v in res["floor_conventions"].items():
        print("XCONV %-58s X=%.2f delta=%.5e sig(19.3237)=%.5f sig(19.32)=%.5f" % (k_, v["X"], v["delta"], v["sigma_floor_kappa19.3237"], v["sigma_floor_kappa19.32"]))
    print("HEFF", res["H_eff_over_h"])
    for k_, v in nx.items():
        print("NEXT %-32s K=%.4f sig_exact=%.5f sig_C76conv=%.5f ratio=%.4f" % (k_, v["K"], v["sigma_floor_exact"], v["sigma_floor_C76conv"], v["ratio_to_xi030"]))
    print("LEMMA", res["monotone_lemma_check"])
    pf = res["phase_family"]
    print("PHASE s_opt0=%.4e s_nat0=%.4e ratio=%.4f sigma_floor=%.5f  bnd C(-inf)=0 at %s ; C(+inf)=0 at %s ; focus phi=%.3f" % (
        pf["s_opt0"], pf["s_nat0"], pf["s_opt0_over_s_nat0"], pf["sigma_floor"], pf["phi_boundaries_C_minus_inf_zero"],
        pf["phi_boundaries_C_plus_inf_zero"], pf["phi_focus_single_branch_opt"]))
    for r in fam:
        print("  phi=%6.2f %-34s t_x=%s  var+/fl=%.4g var-/fl=%.4g max/fl=%.4g  C(-inf)/d=%.3g C(+inf)/d=%.3g" % (
            r["phi"], r["cls"], "-" if r["t_x"] is None else "%.4f" % r["t_x"], r["var_plus_over_floor"], r["var_minus_over_floor"],
            r["max_over_floor"], r["C_minus_inf_over_delta"], r["C_plus_inf_over_delta"]))
    for r in ad:
        print("ADIAB", {k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in r.items()})
    print("SHEAR", json.dumps(sh))


if __name__ == "__main__":
    main()
