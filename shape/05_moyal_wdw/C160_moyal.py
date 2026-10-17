# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_tier2/C160_moyal.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Leading Moyal (quantum minus truncated-Wigner) correction to the SM shape-fossil variance (D131 Sec. 1): -2.03e-4 (aspect) and -1.08e-3 (angle) relative at xi = 0.30 for the state real at the turn; C40 toy regression; '1d': exact one-mode quantum vs TWA (1057 s). Paper: PRD Sec. VIII D-E (FOS-08..12).

C160 part 1 (shape-tier2, D131 Sec. 1): leading Moyal (quantum-minus-TWA) correction to the SM shape fossil variance.

Formula (D33/D38, C40/C45 lineage; re-derived in D131 Sec. 1.1):
    Delta Var(q_A(T)) = -delta^2 sum_{BC} c_{ABC}^2 ( Iff_BC Igg_BC - Ifg_BC Ifg_CB ),
    Iff_BC = int_{t0}^{T} W_A f_B f_C dt (etc.),  W_A = V v_A,  V = -3/(mu L),  v_A(t) = f_A(T) g_A(t) - g_A(T) f_A(t),
with H_shape = mu Pi^2/(2 L^3) - (3/(mu L)) Z(G(beta))/Z0, [beta, Pi] = 2 i delta (C60 normalisation), c = d^3 Z/Z0 at the
cube, f,g the fundamental solutions from t0 (state-independent result; depends on the window [t0, T]).
SM: C60 twisted Ewald (15 periodic bosons, 45 antiperiodic Weyl); zero-mode energy as C76 (Z0 -> (1-r) Z_free, shape
derivatives unchanged); next order by C153's rescaled-force completion (all shape derivatives / (1 - r_f)).
Checks: (i) Maxwell content + window [-12,12] must reproduce C40's fixed-cosh toy coefficients (-555.7537, -1.25648);
(ii) Hessian -> kappa (13.528581, -6.352387); (iii) direct double integral; (iv) exact 1D quantum (split-operator) minus
deterministic TWA (Gauss-Hermite over classical trajectories) on the reduced one-mode cubic model.
    python -B C160_moyal.py            -> results/.../shape_tier2/run01/moyal/C160_moyal.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import json, time, hashlib, itertools
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shapelib"))  # package: C60 modules
from C60_twisted_ewald import Casimir, BASIS, CONTENT_SM, CONTENT_MAXWELL, I3   # noqa: E402 (copy)

sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_tier2/run01/moyal")
MU = 3.0 / 2 ** 1.5
R_ZM = 0.299897
X_A = 2282.49                       # D125 (a): X = M_P^2 L_b^3 H_eff, xi = 0.30
DELTA_A = 2 ** 1.5 / (6 * X_A)      # 2.06535e-4 (preferred)
DELTA_B = 1.5897991e-4 * (1 - R_ZM) ** (-0.75)   # C76/C142 draws (convention b)
R_F = 1 - 13.528581 / 18.0          # C153 rescaled-force completion, kappa_E = 18.0
DELTA_NEXT = DELTA_A * (0.70 / 0.72) ** 0.75     # delta at xi = 0.28 (X ~ (1-xi)^{3/4})
TOY_C40 = {"aspect": -555.753661195918, "angle": -1.2564843969255632}


def zline(cas, v, r):
    return cas(expm(2.0 * r * np.einsum("a,aij->ij", v, BASIS)))[0]


def dir_derivs(cas, v, h):
    """f''(0), f'''(0) of f(r) = Z(exp(2 r v.X)) from even/odd polynomial fits at r = +-h..+-4h (O(h^8) truncation)."""
    rs = h * np.arange(1, 5)
    fp = np.array([zline(cas, v, r) for r in rs])
    fm = np.array([zline(cas, v, -r) for r in rs])
    f0 = zline(cas, v, 0.0)
    odd, even = (fp - fm) / 2, (fp + fm) / 2 - f0
    a = np.linalg.solve(np.vstack([rs, rs ** 3, rs ** 5, rs ** 7]).T, odd)
    b = np.linalg.solve(np.vstack([rs ** 2, rs ** 4, rs ** 6, rs ** 8]).T, even)
    return 2 * b[0], 6 * a[1]


def tensors(content, cut=42.0, h=0.02):
    cas = Casimir(content, cut)
    Z0 = cas(I3)[0]
    ee = np.eye(5)
    d2, d3 = {}, {}
    for i in range(5):
        d2[(i,)], d3[(i,)] = dir_derivs(cas, ee[i], h)
    for i, j in itertools.combinations(range(5), 2):
        for sgn in (1, -1):
            d2[(i, j, sgn)], d3[(i, j, sgn)] = dir_derivs(cas, ee[i] + sgn * ee[j], h)
    for i, j, k in itertools.combinations(range(5), 3):
        _, d3[(i, j, k)] = dir_derivs(cas, ee[i] + ee[j] + ee[k], h)
    Hs = np.zeros((5, 5))
    C = np.zeros((5, 5, 5))
    for i in range(5):
        Hs[i, i] = d2[(i,)]
        C[i, i, i] = d3[(i,)]
    for i, j in itertools.combinations(range(5), 2):
        Hs[i, j] = Hs[j, i] = (d2[(i, j, 1)] - d2[(i, j, -1)]) / 4
        p, m = d3[(i, j, 1)], d3[(i, j, -1)]
        ciij = (p - m - 2 * C[j, j, j]) / 6
        cijj = (p + m - 2 * C[i, i, i]) / 6
        for ids in set(itertools.permutations((i, i, j))):
            C[ids] = ciij
        for ids in set(itertools.permutations((i, j, j))):
            C[ids] = cijj
    for i, j, k in itertools.combinations(range(5), 3):
        tog = d3[(i, j, k)]
        pij, pik, pjk = d3[(i, j, 1)], d3[(i, k, 1)], d3[(j, k, 1)]
        cijk = (tog - pij - pik - pjk + C[i, i, i] + C[j, j, j] + C[k, k, k]) / 6
        for ids in itertools.permutations((i, j, k)):
            C[ids] = cijk
    return Z0, Hs, C


def flow(kappa, t0, T, npts):
    """Fundamental matrix columns (f, g): (beta, Pi)(t0) = (1,0) / (0,1); leading cosh background (C60 linear eq.)."""
    def rhs(t, y):
        c = np.cosh(2 * t)
        Y = y.reshape(2, 2)
        A = np.array([[0.0, MU / c ** 1.5], [kappa / (MU * c ** 0.5), 0.0]])
        return (A @ Y).ravel()
    tt = np.linspace(t0, T, npts)
    s = solve_ivp(rhs, (t0, T), np.eye(2).ravel(), t_eval=tt, method="DOP853", rtol=1e-13, atol=1e-16, max_step=0.02)
    F = s.y.T.reshape(-1, 2, 2)
    return tt, F[:, 0, 0], F[:, 0, 1], F[-1]


def coefficients(kap5, c5, t0, T=12.0, npts=4001, direct=True):
    flows = {}
    for k in sorted(set(np.round(kap5, 12))):
        flows[k] = flow(k, t0, T, npts)
    tt = flows[sorted(flows)[0]][0]
    f = np.array([flows[np.round(k, 12)][1] for k in kap5])
    g = np.array([flows[np.round(k, 12)][2] for k in kap5])
    V = -3.0 / (MU * np.sqrt(np.cosh(2 * tt)))
    v = np.array([f[i, -1] * g[i] - g[i, -1] * f[i] for i in range(5)])
    coef, det_list = [], []
    for i in range(5):
        W = V * v[i]
        Iff = np.array([[simpson(W * f[b] * f[c], x=tt) for c in range(5)] for b in range(5)])
        Igg = np.array([[simpson(W * g[b] * g[c], x=tt) for c in range(5)] for b in range(5)])
        Ifg = np.array([[simpson(W * f[b] * g[c], x=tt) for c in range(5)] for b in range(5)])
        det = Iff * Igg - Ifg * Ifg.T
        coef.append(float(-np.sum(c5[i] ** 2 * det)))
        det_list.append(det)
    out = {"t0": t0, "T": T, "npts": npts, "coef": coef}
    if direct:   # independent direct double integral for output modes 0 (aspect) and 2 (angle), coarser grid
        ix = np.arange(0, npts, 4)
        if ix[-1] != npts - 1:
            ix = np.r_[ix, npts - 1]
        t2 = tt[ix]
        dd = []
        for i in (0, 2):
            va = V[ix] * v[i, ix]
            rr = np.array([f[j, ix][:, None] * g[j, ix][None, :] - g[j, ix][:, None] * f[j, ix][None, :] for j in range(5)])
            summ = np.einsum("bc,bij,cij->ij", c5[i] ** 2, rr, rr)
            dd.append(float(-0.5 * simpson(simpson(va[:, None] * va[None, :] * summ, x=t2, axis=1), x=t2, axis=0)))
        out["direct_double"] = {"aspect": dd[0], "angle": dd[1]}
    return out


def transfer(kappa, t0, T=14.0):
    return flow(kappa, t0, T, 3)[3]


def state_var_over_delta(kappa, t0, kind):
    """Var(beta(+inf))/delta for the declared state at t0: 'natural' (C76 width mu/sqrt|kappa|, set at t0=0) or
    'optimal' (s_opt = |M12/M11| delta, Var = 2|M11 M12| delta)."""
    M = transfer(kappa, t0)
    if kind == "optimal":
        return float(2 * abs(M[0, 0] * M[0, 1])), M
    s = MU / np.sqrt(abs(kappa))
    return float(M[0, 0] ** 2 * s + M[0, 1] ** 2 / s), M


# ------------------------------------------------------------------ independent exact 1D check
def one_d_check(kappa, c, delta, s_over_delta, T=8.0, N=4096, Q=1.2, dt=2.5e-4, ngh=20):
    """Reduced model H = mu Pi^2/(2L^3) + V(t)(lam q^2/2 + c q^3/6), V = -3/(mu L), lam = kappa/3, hbar = 2 delta.
    Returns quantum Var (split-operator), TWA Var (Gauss-Hermite over exact classical trajectories) and the formula."""
    hb = 2 * delta
    lam = kappa / 3.0
    s = s_over_delta * delta
    L = lambda t: np.sqrt(np.cosh(2 * t))
    q = -Q + 2 * Q * np.arange(N) / N
    dq = q[1] - q[0]
    k = 2 * np.pi * np.fft.fftfreq(N, dq)
    psi = (2 * np.pi * s) ** -0.25 * np.exp(-q * q / (4 * s)) + 0j
    U = lam * q * q / 2 + c * q ** 3 / 6
    nst = int(round(T / dt))
    for n in range(nst):
        tm = (n + 0.5) * dt
        Lm = L(tm)
        Vm = -3.0 / (MU * Lm)
        ph = np.exp(-1j * Vm * U * dt / (2 * hb))
        psi = ph * psi
        psi = np.fft.ifft(np.exp(-1j * MU * hb * k * k * dt / (2 * Lm ** 3)) * np.fft.fft(psi))
        psi = ph * psi
    rho = np.abs(psi) ** 2 * dq
    norm = float(rho.sum())
    m1 = float((rho * q).sum() / norm)
    varQ = float((rho * q * q).sum() / norm - m1 * m1)
    edge = float(rho[(np.abs(q) > 0.8 * Q)].sum() / norm)
    phat = np.abs(np.fft.fft(psi)) ** 2
    kedge = float(phat[np.abs(k) > 0.5 * np.abs(k).max()].sum() / phat.sum())
    # TWA: exact classical flow of the Gaussian Wigner function
    x, w = np.polynomial.hermite_e.hermegauss(ngh)
    w = w / np.sqrt(2 * np.pi)

    def rhs(t, y):
        Lt = L(t)
        return [MU * y[1] / Lt ** 3, (3.0 / (MU * Lt)) * (lam * y[0] + c * y[0] ** 2 / 2)]
    qs, ws = [], []
    for i in range(ngh):
        for j in range(ngh):
            y0 = [np.sqrt(s) * x[i], delta / np.sqrt(s) * x[j]]
            sol = solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=1e-12, atol=1e-15)
            qs.append(sol.y[0, -1])
            ws.append(w[i] * w[j])
    qs, ws = np.array(qs), np.array(ws)
    mT = float(np.sum(ws * qs))
    varT = float(np.sum(ws * qs * qs) - mT * mT)
    # formula (single mode, window [0, T])
    tt, f, g, M = flow(kappa, 0.0, T, 8001)
    V = -3.0 / (MU * np.sqrt(np.cosh(2 * tt)))
    v = f[-1] * g - g[-1] * f
    W = V * v
    Iff, Igg, Ifg = simpson(W * f * f, x=tt), simpson(W * g * g, x=tt), simpson(W * f * g, x=tt)
    coef = float(-c * c * (Iff * Igg - Ifg * Ifg))
    var_lin = float(M[0, 0] ** 2 * s + M[0, 1] ** 2 * delta ** 2 / s)
    return {"kappa": kappa, "c": c, "delta": delta, "T": T, "N": N, "Q": Q, "dt": dt, "norm": norm, "edge_prob": edge,
            "k_edge_prob": kedge, "var_lin": var_lin, "var_quantum": varQ, "var_TWA": varT,
            "quantum_minus_TWA": varQ - varT, "formula_delta2_coef": float(delta ** 2 * coef),
            "ratio_numeric_over_formula": float((varQ - varT) / (delta ** 2 * coef)) if c != 0 else None,
            "TWA_minus_lin_rel": (varT - var_lin) / var_lin, "quantum_minus_lin_rel": (varQ - var_lin) / var_lin}


def main_1d():
    """Re-run of the exact 1D check with momentum-resolving grids (run01 v1 used N=4096, Q=1.2: k_edge 0.39, aliased;
    kept in C160_moyal.json for the record).  T = 5 (shape frozen; integrals use the same window)."""
    t_start = time.perf_counter()
    Zf, Hf, Cf = tensors(CONTENT_SM)
    kE = 13.528581 / (1 - R_ZM)
    c111 = float(Cf[1, 1, 1] / ((1 - R_ZM) * Zf))
    sod = MU / np.sqrt(kE)
    checks = []
    for cc, dt, N, Q in ((0.0, 2.5e-4, 32768, 1.0), (c111, 2.5e-4, 32768, 1.0), (c111, 1.25e-4, 32768, 1.0),
                         (c111, 2.5e-4, 65536, 1.0), (c111, 2.5e-4, 65536, 2.0), (0.5 * c111, 2.5e-4, 32768, 1.0),
                         (2.0 * c111, 2.5e-4, 32768, 1.0), (0.0, 2.5e-4, 65536, 2.0)):
        r1 = one_d_check(kE, cc, DELTA_A, sod, T=5.0, dt=dt, N=N, Q=Q)
        checks.append(r1)
        print("1D", {k: r1[k] for k in ("c", "dt", "N", "Q", "var_lin", "var_quantum", "var_TWA", "quantum_minus_TWA",
                                         "formula_delta2_coef", "ratio_numeric_over_formula", "edge_prob", "k_edge_prob")},
              flush=True)
    out = {"checks": checks, "c111": c111, "kappa_E": kE, "delta": DELTA_A, "runtime_s": time.perf_counter() - t_start,
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "C160_moyal_1d.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def main():
    if "1d" in sys.argv:
        return main_1d()
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"units": "C60 normalisation: H = mu Pi^2/(2L^3) - (3/(mu L)) Z/Z0, [beta,Pi]=2i delta, t in 1/H_eff, L=sqrt(cosh 2t)"}
    # ---------------- tensors
    Zm, Hm, Cm = tensors(CONTENT_MAXWELL)
    Zf, Hf, Cf = tensors(CONTENT_SM)
    _, _, Cf60 = tensors(CONTENT_SM, cut=60.0, h=0.03)
    kap_m = 3 * np.diag(Hm) / Zm
    kap_f = 3 * np.diag(Hf) / Zf
    res["maxwell"] = {"Z0": Zm, "kappa": kap_m.tolist(), "C001": Cm[0, 0, 1], "C111": Cm[1, 1, 1], "C122": Cm[1, 2, 2],
                      "C234": Cm[2, 3, 4], "C40_ref": {"C001": -40.27112956709544, "C111": 40.27112956507963,
                                                         "C122": 8.716758291782032, "C234": -1.415823761189965,
                                                         "Z0": 16.5323159597617, "kappa_aspect": 3 * 33.0520045147402 / 16.5323159597617}}
    sym = {"E3_C111": Cf[1, 1, 1], "E3_C001": Cf[0, 0, 1], "C000": Cf[0, 0, 0], "C011": Cf[0, 1, 1],
           "ETT_C122": Cf[1, 2, 2], "ETT_C133": Cf[1, 3, 3], "ETT_C033": Cf[0, 3, 3], "ETT_C044": Cf[0, 4, 4],
           "TTT_C234": Cf[2, 3, 4], "C222": Cf[2, 2, 2]}
    res["SM_free"] = {"Z_free": Zf, "kappa_r0": kap_f.tolist(), "kappa_ref_C76": [13.528581, -6.352387],
                      "offdiag_hessian_max": float(np.max(np.abs(Hf - np.diag(np.diag(Hf))))),
                      "cubic_components": sym, "c_over_Zfree": {k: v / Zf for k, v in sym.items()},
                      "cut60_h03_max_abs_change": float(np.max(np.abs(Cf60 - Cf))),
                      "cubic_tensor": Cf.tolist()}
    # ---------------- regression: Maxwell toy, window [-12, 12]
    reg = coefficients(np.array([kap_m[0]] * 2 + [kap_m[2]] * 3), Cm / Zm, -12.0, 12.0, 4801)
    res["regression_C40"] = {"coef_aspect": reg["coef"][0], "coef_angle": reg["coef"][2], "C40": TOY_C40,
                             "rel_diff_aspect": reg["coef"][0] / TOY_C40["aspect"] - 1,
                             "rel_diff_angle": reg["coef"][2] / TOY_C40["angle"] - 1, "direct": reg["direct_double"]}
    print("regression", res["regression_C40"], flush=True)
    # ---------------- SM cases
    cases = {"xi030": dict(kap=np.array([13.528581] * 2 + [-6.352387] * 3) / (1 - R_ZM), c=Cf / ((1 - R_ZM) * Zf),
                           deltas={"a_Heff": DELTA_A, "b_C76": DELTA_B}),
             "next_order": dict(kap=np.array([13.528581] * 2 + [-6.352387] * 3) / (1 - R_F), c=Cf / ((1 - R_F) * Zf),
                                deltas={"a_Heff_xi028": DELTA_NEXT})}
    windows = [("t0=0 natural", 0.0, "natural"), ("t0=0 optimal", 0.0, "optimal"), ("R1 -0.25", -0.25, "optimal"),
               ("R1 -0.5", -0.5, "optimal"), ("R1 -1.0", -1.0, "optimal"), ("R1 F", None, "optimal"),
               ("t0=-12 optimal", -12.0, "optimal")]
    res["SM"] = {}
    for cname, cs in cases.items():
        rows = []
        kE, kT = cs["kap"][0], cs["kap"][2]
        for lab, t0, kind in windows:
            if t0 is None:
                t0 = -0.5 * float(np.arcsinh(np.sqrt(kE)))
            co = coefficients(cs["kap"], cs["c"], t0, 12.0, 4001)
            co2 = coefficients(cs["kap"], cs["c"], t0, 14.0, 8001, direct=False)
            vE, ME = state_var_over_delta(kE, t0, kind)
            vT, MT = state_var_over_delta(kT, t0, kind)
            floorE = state_var_over_delta(kE, 0.0, "optimal")[0]
            row = {"window": lab, "t0": t0, "state": kind, "coef_aspect": co["coef"][0], "coef_angle": co["coef"][2],
                   "coef_aspect_T14_8001": co2["coef"][0], "coef_angle_T14_8001": co2["coef"][2],
                   "direct_double": co["direct_double"], "var_over_delta_aspect": vE, "var_over_delta_angle": vT,
                   "floor_over_delta_aspect": floorE, "rel": {}}
            for dn, d in cs["deltas"].items():
                row["rel"][dn] = {"aspect": d * co["coef"][0] / vE, "angle": d * co["coef"][2] / vT,
                                  "aspect_rel_to_floor": d * co["coef"][0] / floorE,
                                  "sigma_shift_aspect": 0.5 * d * co["coef"][0] / vE}
            rows.append(row)
            print(cname, lab, "%.6g %.6g" % (row["coef_aspect"], row["coef_angle"]),
                  {k: "%.4e/%.4e" % (v["aspect"], v["angle"]) for k, v in row["rel"].items()}, flush=True)
        res["SM"][cname] = {"kappa": cs["kap"].tolist(), "c_E3_111": float(cs["c"][1, 1, 1]),
                            "c_ETT_122": float(cs["c"][1, 2, 2]), "c_TTT_234": float(cs["c"][2, 3, 4]),
                            "deltas": cs["deltas"], "rows": rows}
    # ---------------- exact 1D quantum check (aspect kappa, E3 coupling c_111 along X_D2, natural width)
    kE = 13.528581 / (1 - R_ZM)
    c111 = float(Cf[1, 1, 1] / ((1 - R_ZM) * Zf))
    sod = MU / np.sqrt(kE)
    checks = []
    for cc, dt, N in ((0.0, 2.5e-4, 4096), (c111, 2.5e-4, 4096), (c111, 1.25e-4, 4096), (c111, 2.5e-4, 8192),
                      (0.5 * c111, 2.5e-4, 4096), (2.0 * c111, 2.5e-4, 4096)):
        r1 = one_d_check(kE, cc, DELTA_A, sod, dt=dt, N=N, Q=1.2 if N == 4096 else 2.4)
        checks.append(r1)
        print("1D", {k: r1[k] for k in ("c", "dt", "N", "var_lin", "var_quantum", "var_TWA", "quantum_minus_TWA",
                                         "formula_delta2_coef", "ratio_numeric_over_formula", "edge_prob", "k_edge_prob")},
              flush=True)
    res["one_d_check"] = checks
    res["meta"] = {"runtime_s": time.perf_counter() - t_start, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "ewald_sha256": hashlib.sha256((HERE.parent / "shapelib" / "C60_twisted_ewald.py").read_bytes()).hexdigest(),
                   "delta_a": DELTA_A, "delta_b": DELTA_B, "delta_next": DELTA_NEXT, "r_zm": R_ZM, "r_f": R_F,
                   "BLAS_threads": 1, "processes": 1}
    (OUT / "C160_moyal.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("done %.1f s" % res["meta"]["runtime_s"], flush=True)


if __name__ == "__main__":
    main()
