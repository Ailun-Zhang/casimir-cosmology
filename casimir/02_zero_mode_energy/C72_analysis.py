# PROVENANCE: 04_calculations/code/20261011_claude_followup/C72/C72_analysis.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] C72 analysis (D54): Legendre transform and scaling, exact Gaussian expectations, isospectrality, one-loop couplings and r = xi bounds (pure-gauge lower bound 0.2580, Gaussian upper bound 0.3083, exact-local DMC 0.2993/0.2999/0.3043 for lambda = 0/0.01/0.1), fossil transfer vs xi. Reads <work>/fu/C72/C72_dmc_results.json. Paper: PRD Sec. IV C, Table tab:pa-fzero, App. app:zeromodes.

C72 analysis: zero-mode energy of the SM gauge/Higgs constant modes on T^3 and its effect on the fossil.

Independent re-derivation of the Codex claims (a)-(f) (see D54_zero_mode_energy_DRAFT.md):
  A  sympy: Legendre transform of the reduced constant-mode Lagrangian (general G), the scaling c = g^(2/3) x.
  B  exact Gaussian expectations by Wick moments (sympy, every term of the SU(2) x U(1) x Higgs trial energy),
     SU(N) Gaussian, numerical minimisation (multi-start), Monte-Carlo cross-check of the trial energies.
  C  fixed-volume isospectrality incl. U(1)_Y and the Higgs gradient term with Y = 1/2; Jacobians, symplectic form,
     Laplacian coefficient matrix, Gauss generators.
  D  one-loop couplings at mu = 1/L_b (and 2 pi/L_b, self-consistent C_eff), reference and variational bounds.
  E  fossil transfer (own implementation, leading background), sigma_E, sigma_T2, emptiness budget vs r,
     using DMC exact-local energies from C72_dmc_results.json when present.

    python C72_analysis.py
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C72_common import (RES, C_FREE, GAMMA, NC, KAPPA_E, KAPPA_T2, H_OVER_MP, EPS_SU2_VB, EPS_SU3_PAVEL,  # noqa: E402
                        EPS_QUARTIC4, su_basis, structure_constants, e_gauss_sun, e_gauss_ew, minimize_ew,
                        couplings_one_loop, mu_bounce_gev, v_ew)

LAMS = (0.0, 0.001, 0.01, 0.05, 0.1)
CODEX = dict(eps_SU2=4.116720, eps_SU3=12.5868, gauss_SU2=4.2522335434, gauss_SU3=12.9802461328,
             ew_E=4.4714636207588, ew_abz=(3.45031111953, 1.76278776208, 0.760421324596),
             ew_hess_log=(5.44134, 1.59952, 0.484993), su3_gauss_g05=8.17704266774, Q_upper=12.6485062885,
             frac_pct=(29.531, 29.537, 29.594, 29.809, 30.026), D0=10.5225582277, D0_pct=24.62, SU3_pct=18.55,
             run_g2=0.526764, run_g3=0.539605, run_D0=11.0276706, run_D0_pct=25.8017, kappaE_ref=17.9471532,
             sumkappa_ref=10.6128831)


# ================================================================== A. symbolic Hamiltonian
def part_a():
    import sympy as sp
    g, L = sp.symbols("g L", positive=True)
    c = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"c{a}{i}", real=True))
    cd = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"v{a}{i}", real=True))
    eps = lambda a, b, k: sp.LeviCivita(a, b, k)

    def F2(m):
        tot = 0
        for a in range(3):
            for i in range(3):
                for j in range(3):
                    tot += sum(eps(a, b, k) * m[b, i] * m[k, j] for b in range(3) for k in range(3)) ** 2
        return sp.expand(tot)

    # cube (G = 1): reduced Lagrangian from -(1/4g^2) int sqrt(-g) F F, sqrt(-g) = L^3, h^ij = delta/L^2, unit cell
    lag = L / (2 * g ** 2) * sum(cd[a, i] ** 2 for a in range(3) for i in range(3)) - F2(c) / (4 * g ** 2 * L)
    p = sp.Matrix(3, 3, lambda a, i: sp.diff(lag, cd[a, i]))
    P = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"p{a}{i}", real=True))
    sol = sp.solve([p[a, i] - P[a, i] for a in range(3) for i in range(3)], list(cd), dict=True)[0]
    Ham = sp.expand((sum(P[a, i] * cd[a, i] for a in range(3) for i in range(3)) - lag).subs(sol))
    target = g ** 2 / (2 * L) * sum(P[a, i] ** 2 for a in range(3) for i in range(3)) + F2(c) / (4 * g ** 2 * L)
    res_cube = sp.simplify(Ham - target)
    # scaling c = g^(2/3) x, p = g^(-2/3) pi
    x = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"x{a}{i}", real=True))
    pi = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"q{a}{i}", real=True))
    sub = {c[a, i]: g ** sp.Rational(2, 3) * x[a, i] for a in range(3) for i in range(3)}
    sub.update({P[a, i]: g ** sp.Rational(-2, 3) * pi[a, i] for a in range(3) for i in range(3)})
    LH = sp.expand(L * target.subs(sub, simultaneous=True))
    res_scale = sp.simplify(LH - g ** sp.Rational(2, 3) * (sum(pi[a, i] ** 2 for a in range(3) for i in range(3)) / 2
                                                         + F2(x) / 4))
    # general G, single-index block: Lagrangian (L/2g^2) v^T G^-1 v -> Hamiltonian (g^2/2L) p^T G p
    Gs = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"G{min(i, j)}{max(i, j)}", real=True))
    v = sp.Matrix(3, 1, lambda i, _: sp.Symbol(f"w{i}", real=True))
    pv = sp.Matrix(3, 1, lambda i, _: sp.Symbol(f"k{i}", real=True))
    lagG = (L / (2 * g ** 2) * (v.T * Gs.inv() * v))[0]
    mom = sp.Matrix([sp.diff(lagG, vi) for vi in v])
    vsol = sp.solve(list(mom - pv), list(v), dict=True)[0]
    HG = sp.simplify(((pv.T * v)[0] - lagG).subs(vsol))
    res_G = sp.simplify(HG - (g ** 2 / (2 * L) * (pv.T * Gs * pv))[0])
    return dict(LH_cube_residual=str(res_cube), scaling_residual=str(res_scale), generalG_legendre_residual=str(res_G))


# ================================================================== B. Gaussian trial energies
def wick_expectation(poly, var_of):
    """E[poly] for independent centred Gaussians, var_of[symbol] = variance; poly is a sympy polynomial."""
    import sympy as sp
    syms = list(var_of)
    P = sp.Poly(sp.expand(poly), *syms)
    tot = 0
    for mon, coef in P.terms():
        term = coef
        for s, k in zip(syms, mon):
            if k % 2:
                term = 0
                break
            term *= sp.factorial2(k - 1) * var_of[s] ** (k // 2)
        tot += term
    return sp.simplify(tot)


def part_b_symbolic():
    import sympy as sp
    a, b, z, g2, gY, lam = sp.symbols("a b z g2 gY lam", positive=True)
    c = sp.Matrix(3, 3, lambda A, i: sp.Symbol(f"c{A}{i}", real=True))       # [colour, space]
    u = [sp.Symbol(f"u{i}", real=True) for i in range(3)]
    ph = [sp.Symbol(f"f{k}", real=True) for k in range(4)]
    var = {s: 1 / (2 * a) for s in c}
    var.update({s: 1 / (2 * b) for s in u})
    var.update({s: 1 / (2 * z) for s in ph})
    # SU(2) potential (1/4g2^2) F^2
    F2 = 0
    for A in range(3):
        for i in range(3):
            for j in range(3):
                F2 += sum(sp.LeviCivita(A, B, C) * c[B, i] * c[C, j] for B in range(3) for C in range(3)) ** 2
    e_F = wick_expectation(F2 / (4 * g2 ** 2), var)
    # Higgs gradient, directly from Pauli matrices (no identity used)
    sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    Phi = sp.Matrix([ph[0] + sp.I * ph[1], ph[2] + sp.I * ph[3]]) / sp.sqrt(2)
    grad = 0
    for i in range(3):
        M = sum((c[A, i] * sig[A] / 2 for A in range(3)), sp.zeros(2, 2)) + sp.Rational(1, 2) * u[i] * sp.eye(2)
        vv = M * Phi
        grad += sp.expand((vv.H * vv)[0])
    grad = sp.expand(grad)
    im_part = sp.expand(sp.im(grad))
    grad_re = sp.expand(sp.re(grad))
    cross = sp.expand(grad_re - sum(c[A, i] ** 2 for A in range(3) for i in range(3)) * sum(s ** 2 for s in ph) / 8
                      - sum(s ** 2 for s in u) * sum(s ** 2 for s in ph) / 8)
    e_grad = wick_expectation(grad_re, var)
    e_cross = wick_expectation(cross, var)
    quart = lam * (sum(s ** 2 for s in ph) / 2) ** 2
    e_quart = wick_expectation(quart, var)
    kin = g2 ** 2 / 2 * 9 * a / 2 + gY ** 2 / 2 * 3 * b / 2 + sp.Rational(1, 2) * 4 * z / 2     # <p^2> = width/2 each
    # gY enters only through kinetic normalisation of u with c,u in geometric units -> no gY in the gradient term
    total = sp.simplify(kin + e_F + e_grad + e_quart)
    claim = (sp.Rational(9, 4) * g2 ** 2 * a + sp.Rational(3, 4) * gY ** 2 * b + z + 9 / (4 * g2 ** 2 * a ** 2)
             + 9 / (8 * a * z) + 3 / (8 * b * z) + 3 * lam / (2 * z ** 2))
    return dict(E_F=str(e_F), E_grad=str(sp.simplify(e_grad)), E_cross=str(e_cross), E_quartic=str(e_quart),
                grad_imag_part=str(im_part), total=str(total), total_minus_claim=str(sp.simplify(total - claim)))


def part_b_numeric(rng):
    out = {}
    # structure-constant sums sum_abc f_abc^2 = N(N^2-1) used in <F^2> = 6 sum f^2 sigma^4
    out["sum_f2"] = {N: float((structure_constants(N) ** 2).sum()) for N in (2, 3, 4)}
    for N in (2, 3):
        r = minimize_scalar(lambda al: e_gauss_sun(al, N), bounds=(0.05, 20), method="bounded",
                            options=dict(xatol=1e-13))
        out[f"SU{N}_gauss"] = dict(alpha_num=float(r.x), alpha_closed=float(N ** (1 / 3)), E_num=float(r.fun),
                                   E_closed=float(9 * (N * N - 1) * N ** (1 / 3) / 8))
    # Monte-Carlo (direct potential evaluation) cross-check of the SU(3) Gaussian and the coupled formula
    f3 = structure_constants(3)
    from C72_common import v_sun
    al = 3 ** (1 / 3)
    xs = rng.normal(size=(400000, 8, 3)) / np.sqrt(2 * al)
    mc = 3 * 8 * al / 4 + v_sun(xs, f3)
    out["SU3_gauss_MC"] = dict(mean=float(mc.mean()), se=float(mc.std() / np.sqrt(mc.size)))
    a, b, z, g2, gY, lam = 3.45, 1.76, 0.76, 0.5, 0.46, 0.01
    n = 1_000_000
    y = np.concatenate([rng.normal(size=(n, 9)) / np.sqrt(2 * a * g2 ** 2), rng.normal(size=(n, 3)) / np.sqrt(2 * b * gY ** 2),
                        rng.normal(size=(n, 4)) / np.sqrt(2 * z)], axis=1)       # unit-mass coordinates
    kin = 9 * a * g2 ** 2 / 4 + 3 * b * gY ** 2 / 4 + z
    vals = kin + v_ew(y, g2, gY, lam)
    out["EW_gauss_MC"] = dict(abz=(a, b, z), mean=float(vals.mean()), se=float(vals.std() / np.sqrt(n)),
                              formula=float(e_gauss_ew(a, b, z, g2, gY, lam)))
    # coupled minimisation at the benchmark
    scan = []
    su3g = float(9 * 3 ** (1 / 3) * 0.5 ** (2 / 3))
    for lam in LAMS:
        m = minimize_ew(0.5, 0.46, lam, n_starts=16, seed=11)
        Q = m["E"] + su3g
        scan.append(dict(lam=lam, **m, SU3_gauss=su3g, Q_upper=Q, r_upper=Q / C_FREE))
    out["EW_scan_g05_gY046"] = scan
    return out


# ================================================================== C. isospectrality with U(1)_Y and Higgs
def rand_sl3(rng, scale=0.5):
    """Well-conditioned random S in SL(3,R): S = expm(A), A random traceless (det S = exp(tr A) = 1)."""
    from scipy.linalg import expm
    A = rng.normal(scale=scale, size=(3, 3))
    A -= np.trace(A) / 3 * np.eye(3)
    return expm(A)


def part_c_exact():
    """Exact polynomial identity H_G(Sx, Sy, Phi, S^-T pi) == H_1(x, y, Phi, pi) for a fixed non-diagonal integer
    unimodular S, SU(2) x U(1)_Y (Y = 1/2) x Higgs, all terms incl. the Gauss generators (sympy, exact rationals)."""
    import sympy as sp
    S = sp.Matrix([[1, 1, 0], [1, 2, 1], [0, 1, 2]])
    G = S * S.T
    Gi = G.inv()
    g2, gY = sp.symbols("g2 gY", positive=True)
    x = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"x{a}{i}", real=True))        # [colour, space]
    px = sp.Matrix(3, 3, lambda a, i: sp.Symbol(f"P{a}{i}", real=True))
    y = sp.Matrix(3, 1, lambda i, _: sp.Symbol(f"y{i}", real=True))
    py = sp.Matrix(3, 1, lambda i, _: sp.Symbol(f"Q{i}", real=True))
    ph = [sp.Symbol(f"f{k}", real=True) for k in range(4)]
    Phi = sp.Matrix([ph[0] + sp.I * ph[1], ph[2] + sp.I * ph[3]]) / sp.sqrt(2)
    sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]

    def terms(c, pc, u, pu, G, Gi):
        kin = g2 ** 2 / 2 * sum((pc[a, :] * G * pc[a, :].T)[0] for a in range(3)) + gY ** 2 / 2 * (pu.T * G * pu)[0]
        F = [[[sum(sp.LeviCivita(a, b, k) * c[b, i] * c[k, j] for b in range(3) for k in range(3)) for j in range(3)]
              for i in range(3)] for a in range(3)]
        pot = sum(Gi[i, k] * Gi[j, l] * F[a][i][j] * F[a][k][l] for a in range(3) for i in range(3) for j in range(3)
                  for k in range(3) for l in range(3)) / (4 * g2 ** 2)
        M = [sum((c[a, i] * sig[a] / 2 for a in range(3)), sp.zeros(2, 2)) + sp.Rational(1, 2) * u[i] * sp.eye(2)
             for i in range(3)]
        hig = sum(Gi[i, j] * ((M[i] * Phi).H * (M[j] * Phi))[0] for i in range(3) for j in range(3))
        gauss = [sum(sp.LeviCivita(a, b, k) * c[b, i] * pc[k, i] for b in range(3) for k in range(3) for i in range(3))
                 for a in range(3)]
        return [kin, pot, hig] + gauss

    c = x * S.T                        # c^a_i = S_ij x^a_j
    pc = px * S.inv()                  # p^a_i = (S^-T)_ij pi^a_j
    u = S * y
    pu = S.inv().T * py
    tG = terms(c, pc, u, pu, G, Gi)
    t1 = terms(x, px, y, py, sp.eye(3), sp.eye(3))
    res = [sp.expand(a - b) for a, b in zip(tG, t1)]
    jac = (sp.kronecker_product(sp.eye(3), S)).det() * S.det()
    return dict(S=str(S.tolist()), det_S=str(S.det()), residuals_kin_pot_higgs_gauss=[str(r) for r in res],
                jacobian_config=str(jac), laplacian_matrix=str(sp.simplify(S.inv() * G * S.inv().T)))


def ham_terms(c, u, Phi, pc, pu, G, f, T, g2=0.5, gY=0.46, Y=0.5):
    """Classical constant-mode Hamiltonian terms at shape G (det G = 1), L = 1.
    c, pc: (d,3) [colour, space]; u, pu: (3,); Phi: complex doublet (or None)."""
    Gi = np.linalg.inv(G)
    kin_c = 0.5 * g2 ** 2 * sum(pc[a] @ G @ pc[a] for a in range(c.shape[0]))
    kin_u = 0.5 * gY ** 2 * pu @ G @ pu
    F = np.einsum("abc,bi,cj->aij", f, c, c)
    pot = np.einsum("ik,jl,aij,akl->", Gi, Gi, F, F) / (4 * g2 ** 2)
    gauss = np.einsum("abc,bi,ci->a", f, c, pc)
    hig = 0.0
    if Phi is not None:
        M = [sum(c[a, i] * T[a] for a in range(c.shape[0])) + Y * u[i] * np.eye(2) for i in range(3)]
        hig = sum(Gi[i, j] * (np.conj(M[i] @ Phi) @ (M[j] @ Phi)) for i in range(3) for j in range(3)).real
    return np.array([kin_c, kin_u, pot, hig]), gauss


def part_c(rng, trials=200):
    worst = dict(SU2xU1xHiggs=0.0, SU3=0.0, gauss=0.0, jacobian=0.0, symplectic=0.0, laplacian=0.0)
    for N in (2, 3):
        f = structure_constants(N); T = su_basis(N); d = f.shape[0]
        for _ in range(trials):
            S = rand_sl3(rng)
            G = S @ S.T
            x = rng.normal(size=(d, 3)); px = rng.normal(size=(d, 3))
            y = rng.normal(size=3); py = rng.normal(size=3)
            Phi = (rng.normal(size=2) + 1j * rng.normal(size=2)) if N == 2 else None
            c = x @ S.T                     # c^a_i = S_ij x^a_j
            pc = px @ np.linalg.inv(S)      # p^a_i = (S^-T)_ij pi^a_j
            u = S @ y; pu = np.linalg.inv(S).T @ py
            tG, gG = ham_terms(c, u, Phi, pc, pu, G, f, T)
            t1, g1 = ham_terms(x, y, Phi, px, py, np.eye(3), f, T)
            rel = np.abs(tG - t1).max() / max(np.abs(t1).max(), 1.0)
            worst["SU2xU1xHiggs" if N == 2 else "SU3"] = max(worst["SU2xU1xHiggs" if N == 2 else "SU3"], rel)
            worst["gauss"] = max(worst["gauss"], np.abs(gG - g1).max() / max(np.abs(g1).max(), 1.0))
            Jc = np.kron(np.eye(d), S)       # flattening colour-major: index a*3+i
            Jp = np.kron(np.eye(d), np.linalg.inv(S).T)
            worst["jacobian"] = max(worst["jacobian"], abs(np.linalg.det(Jc) - 1), abs(np.linalg.det(S) - 1),
                                    abs(np.linalg.det(Jp) - 1))
            dx = rng.normal(size=(d, 3))
            worst["symplectic"] = max(worst["symplectic"], abs((pc * (dx @ S.T)).sum() - (px * dx).sum()))
            Si = np.linalg.inv(S)
            worst["laplacian"] = max(worst["laplacian"], np.abs(Si @ G @ Si.T - np.eye(3)).max())
            worst["max_cond_S"] = max(worst.get("max_cond_S", 0.0), np.linalg.cond(S))
    # non-invariance of holonomy periods: U(1) period lattice 2 pi Z^3 in u maps to 2 pi S^-1 Z^3 in y
    S = rand_sl3(rng)
    short = min(np.linalg.norm(np.linalg.inv(S) @ np.array(v)) for v in
                [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (0, 1, 1), (1, 0, -1), (0, 1, -1)])
    worst["example_shortest_period_ratio_after_map"] = float(short)
    return {k: float(v) for k, v in worst.items()}


# ================================================================== D. couplings and bounds
def bounds_at(g2, gY, g3, eps2, eps3, lams=LAMS):
    D0 = eps2 * g2 ** (2 / 3) + eps3 * g3 ** (2 / 3)
    su3g = 9 * 3 ** (1 / 3) * g3 ** (2 / 3) / 1.0
    out = dict(g2=g2, gY=gY, g3=g3, D0=D0, r_D0=D0 / C_FREE, SU3_ref=eps3 * g3 ** (2 / 3),
               r_SU3_ref=eps3 * g3 ** (2 / 3) / C_FREE, SU3_gauss=su3g, upper=[])
    for lam in lams:
        m = minimize_ew(g2, gY, lam, n_starts=8, seed=5)
        out["upper"].append(dict(lam=lam, E_EW_gauss=m["E"], Q_upper=m["E"] + su3g, r_upper=(m["E"] + su3g) / C_FREE,
                                 Q_upper_PavelSU3=m["E"] + eps3 * g3 ** (2 / 3),
                                 r_upper_PavelSU3=(m["E"] + eps3 * g3 ** (2 / 3)) / C_FREE))
    return out


def part_d():
    mu0 = mu_bounce_gev(C_FREE)
    out = dict(mu_1_over_Lb_GeV=mu0)
    run = couplings_one_loop(mu0)
    out["couplings_mu_1_over_Lb"] = run
    out["couplings_mu_2pi_over_Lb"] = couplings_one_loop(2 * np.pi * mu0)
    out["benchmark_g05"] = bounds_at(0.5, 0.46, 0.5, EPS_SU2_VB, EPS_SU3_PAVEL)
    out["running_mu_1_over_Lb"] = bounds_at(run["g2"], run["gY_SM"], run["g3"], EPS_SU2_VB, EPS_SU3_PAVEL)
    r2 = out["couplings_mu_2pi_over_Lb"]
    out["running_mu_2pi_over_Lb"] = bounds_at(r2["g2"], r2["gY_SM"], r2["g3"], EPS_SU2_VB, EPS_SU3_PAVEL, lams=(0.01,))
    # self-consistent scale: 1/L_b computed with C_eff = C_free (1 - r) (reference energy D0)
    r = out["running_mu_1_over_Lb"]["r_D0"]
    for _ in range(20):
        cc = couplings_one_loop(mu_bounce_gev(C_FREE * (1 - r)))
        r = (EPS_SU2_VB * cc["g2"] ** (2 / 3) + EPS_SU3_PAVEL * cc["g3"] ** (2 / 3)) / C_FREE
    out["self_consistent_reference"] = dict(r_D0=r, mu_GeV=cc["mu_GeV"], g2=cc["g2"], g3=cc["g3"], gY=cc["gY_SM"])
    return out


# ================================================================== E. fossil transfer (own implementation)
def transfer(kappa, t_end=14.0):
    """Leading background L = L_b sqrt(cosh 2t) (h = 1). With y1 = beta, y2 = cosh(2t)^(3/2) beta':
    y1' = y2 cosh^(-3/2), y2' = kappa cosh^(-1/2) y1 (equivalent to beta'' + 3 tanh(2t) beta' = kappa sech^2 beta).
    Returns the 2x2 transfer matrix from the bounce t = 0."""
    def rhs(t, y):
        ch = np.cosh(2 * t)
        return [y[1] * ch ** -1.5, kappa * ch ** -0.5 * y[0], y[3] * ch ** -1.5, kappa * ch ** -0.5 * y[2]]
    s = solve_ivp(rhs, (0.0, t_end), [1.0, 0.0, 0.0, 1.0], method="Radau", rtol=1e-11, atol=1e-13)
    y = s.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def fossil(r, C=C_FREE):
    """sigma for the aspect (E) and T2 shape modes when a shape-independent positive energy r*C/L is added.
    delta' = 1/(M_P^2 L_b^3 h) = (h/M_P)^(1/2) (3/C_eff)^(3/4); natural state at the bounce: oscillator ground
    state of |omega| = sqrt|kappa| h; floor = minimum over Gaussian squeezing at the bounce."""
    Ceff = C * (1 - r)
    dprime = np.sqrt(H_OVER_MP) * (3 / Ceff) ** 0.75
    out = dict(r=r, C_eff=Ceff, budget=Ceff / (2 * np.pi), NC_gamma_1mr_over_2pi=NC * GAMMA * (1 - r) / (2 * np.pi),
               delta_paper_units=dprime * np.sqrt(2) / 3, kappa_E=KAPPA_E / (1 - r), kappa_T2=KAPPA_T2 / (1 - r),
               sum_kappa=8 / (1 - r))
    for lab, k0 in (("E", KAPPA_E), ("T2", KAPPA_T2)):
        k = k0 / (1 - r)
        M = transfer(k)
        n11, n12 = M[0, 0], M[0, 1]
        rk = np.sqrt(abs(k))
        out[f"sigma_{lab}_natural"] = float(np.sqrt(dprime * (n11 ** 2 / (2 * rk) + n12 ** 2 * rk / 2)))
        out[f"sigma_{lab}_floor"] = float(np.sqrt(dprime * abs(n11 * n12)))
        out[f"detM_{lab}"] = float(np.linalg.det(M))
    return out


def part_e(dmc, d_part):
    rows = {}
    rows["r0_free"] = fossil(0.0)
    # convergence check of the transfer end time
    rows["r0_free_tend20"] = dict(sigma_E_natural=float(np.sqrt(np.sqrt(H_OVER_MP) * (3 / C_FREE) ** 0.75 *
                                  (lambda M: M[0, 0] ** 2 / (2 * np.sqrt(KAPPA_E)) + M[0, 1] ** 2 * np.sqrt(KAPPA_E) / 2)(
                                      transfer(KAPPA_E, 20.0)))))
    bm, rn = d_part["benchmark_g05"], d_part["running_mu_1_over_Lb"]
    up = lambda blk, lam: [u for u in blk["upper"] if u["lam"] == lam][0]
    rvals = {"ref_D0_g05_literature": bm["r_D0"], "ref_D0_running_literature": rn["r_D0"],
             "upper_gauss_g05_lam0.01": up(bm, 0.01)["r_upper"], "upper_gauss_running_lam0.01": up(rn, 0.01)["r_upper"],
             "upper_gauss_g05_lam0.1": up(bm, 0.1)["r_upper"], "upper_gauss_running_lam0.1": up(rn, 0.1)["r_upper"],
             "paper_r0.10": 0.10, "paper_r0.25": 0.25}
    if dmc is not None:
        for key, val in dmc["r_exact_local"].items():
            rvals["dmc_" + key] = val
    for key, r in rvals.items():
        rows[key] = fossil(r)
    base = rows["r0_free"]
    for key, row in rows.items():
        if "sigma_E_natural" in row and key != "r0_free_tend20":
            row["sigma_E_ratio_to_free"] = row["sigma_E_natural"] / base["sigma_E_natural"]
            row["sigma_T2_ratio_to_free"] = row["sigma_T2_natural"] / base["sigma_T2_natural"]
    return rows


# ================================================================== DMC summary (reads C72_dmc_results.json)
def dmc_summary():
    fn = RES / "C72_dmc_results.json"
    if not fn.exists():
        return None
    from C72_dmc import extrapolate
    d = json.loads(fn.read_text(encoding="utf-8"))
    run = d["running_couplings"]
    ex, fits = {}, {}
    for name in sorted({r["name"] for r in d["rows"]}):
        rr = sorted([r for r in d["rows"] if r["name"] == name], key=lambda r: r["tau"])
        f_all = extrapolate(rr)
        f3 = extrapolate(rr[:3]) if len(rr) >= 4 else None
        fits[name] = dict(fit_all_taus=f_all, fit_three_smallest_taus=f3,
                          per_tau=[(r["tau"], r["tau_eff_mean"], r["E_mixed"], r["se_mixed"], r["E_growth"]) for r in rr])
        # systematic: half the spread between the two fits (4-tau cases) or the extrapolation-minus-smallest-tau shift
        syst = abs(f_all["E0"] - f3["E0"]) / 2 if f3 else abs(f_all["syst_extrap_minus_smallest"]) / 2
        ex[name] = dict(E0=f_all["E0"], se_E0=float(np.hypot(f_all["se_E0"], syst)), se_stat=f_all["se_E0"], syst=syst)
    E = lambda k: ex[k]["E0"]
    S = lambda k: ex[k]["se_E0"]
    eps2, eps3 = E("su2"), E("su3")
    out = dict(eps_SU2_dmc=eps2, se_SU2=S("su2"), eps_SU3_dmc=eps3, se_SU3=S("su3"), quartic4_dmc=E("quartic4_lam1"),
               se_quartic4=S("quartic4_lam1"), eps_SU2_vanBaal=EPS_SU2_VB, eps_SU3_Pavel=EPS_SU3_PAVEL,
               quartic4_paper=EPS_QUARTIC4, r_exact_local={}, Q_exact_local={})
    for lam in (0.0, 0.001, 0.01, 0.05, 0.1):
        k = f"ew_g05_gY046_lam{lam:g}" if lam else "ew_g05_gY046_lam0"
        if k in ex:
            Q = E(k) + eps3 * 0.5 ** (2 / 3)
            se = np.hypot(S(k), S("su3") * 0.5 ** (2 / 3))
            out["Q_exact_local"][f"g05_lam{lam:g}"] = dict(E_EW=E(k), se_EW=S(k), Q=Q, se_Q=float(se))
            out["r_exact_local"][f"g05_lam{lam:g}"] = Q / C_FREE
    for lam in (0.0, 0.01, 0.1):
        k = f"ew_running_lam{lam:g}" if lam else "ew_running_lam0"
        if k in ex:
            Q = E(k) + eps3 * run["g3"] ** (2 / 3)
            se = np.hypot(S(k), S("su3") * run["g3"] ** (2 / 3))
            out["Q_exact_local"][f"running_lam{lam:g}"] = dict(E_EW=E(k), se_EW=S(k), Q=Q, se_Q=float(se))
            out["r_exact_local"][f"running_lam{lam:g}"] = Q / C_FREE
    out["D0_g05_dmc"] = (eps2 + eps3) * 0.5 ** (2 / 3)
    out["r_D0_g05_dmc"] = out["D0_g05_dmc"] / C_FREE
    out["D0_running_dmc"] = eps2 * run["g2"] ** (2 / 3) + eps3 * run["g3"] ** (2 / 3)
    out["r_D0_running_dmc"] = out["D0_running_dmc"] / C_FREE
    # EW part alone: exact-local minus pure SU(2) (the U(1)+Higgs increment), vs the decoupled pure-quartic Higgs
    out["EW_increment_g05_lam0.01"] = E("ew_g05_gY046_lam0.01") - eps2 * 0.5 ** (2 / 3)
    out["decoupled_quartic_higgs_lam0.01"] = E("quartic4_lam1") * 0.01 ** (1 / 3)
    out["fits"] = fits
    out["E0_used"] = ex
    return out


def main():
    rng = np.random.default_rng(20261011)
    RES.mkdir(parents=True, exist_ok=True)
    res = dict(codex_claims=CODEX)
    print("A: symbolic Hamiltonian ...", flush=True)
    res["A_symbolic"] = part_a(); print(res["A_symbolic"], flush=True)
    print("B: Wick expectations ...", flush=True)
    res["B_symbolic"] = part_b_symbolic(); print(res["B_symbolic"], flush=True)
    res["B_numeric"] = part_b_numeric(rng)
    for row in res["B_numeric"]["EW_scan_g05_gY046"]:
        print(f"lam={row['lam']:<6} E_EW={row['E']:.13f} abz=({row['a']:.11f},{row['b']:.11f},{row['z']:.12f}) "
              f"Q_up={row['Q_upper']:.10f} r={100 * row['r_upper']:.4f}% hessLog={row['hess_log_eigs']} "
              f"grad={row['grad_max']:.1e} spread={row['spread_over_starts']:.1e}", flush=True)
    print({k: v for k, v in res["B_numeric"].items() if k != "EW_scan_g05_gY046"}, flush=True)
    print("C: isospectrality ...", flush=True)
    res["C_isospectrality"] = part_c(rng); print(res["C_isospectrality"], flush=True)
    res["C_isospectrality_exact"] = part_c_exact(); print(res["C_isospectrality_exact"], flush=True)
    print("D: couplings ...", flush=True)
    res["D_couplings_bounds"] = part_d()
    d = res["D_couplings_bounds"]
    print(json.dumps({k: d[k] for k in ("mu_1_over_Lb_GeV", "self_consistent_reference")}, indent=0), flush=True)
    for blk in ("benchmark_g05", "running_mu_1_over_Lb", "running_mu_2pi_over_Lb"):
        b = d[blk]
        print(blk, f"g2={b['g2']:.6f} gY={b['gY']:.6f} g3={b['g3']:.6f} D0={b['D0']:.10f} r_D0={100 * b['r_D0']:.4f}% "
              f"SU3ref={b['SU3_ref']:.10f} ({100 * b['r_SU3_ref']:.4f}%)",
              [(u["lam"], round(u["Q_upper"], 10), round(100 * u["r_upper"], 4), round(100 * u["r_upper_PavelSU3"], 4))
               for u in b["upper"]], flush=True)
    dmc = dmc_summary()
    res["DMC_summary"] = dmc
    if dmc is not None:
        print("DMC:", json.dumps(dmc, indent=0), flush=True)
    print("E: fossil ...", flush=True)
    res["E_fossil"] = part_e(dmc, d)
    for k, row in res["E_fossil"].items():
        if "sigma_T2_natural" in row:
            print(f"{k:34s} r={row['r']:.5f} kE={row['kappa_E']:.4f} kT2={row['kappa_T2']:.4f} "
                  f"sE_nat={row['sigma_E_natural']:.6f} sE_floor={row['sigma_E_floor']:.6f} "
                  f"sT2_nat={row['sigma_T2_natural']:.6f} sT2_floor={row['sigma_T2_floor']:.6f} "
                  f"ratioE={row['sigma_E_ratio_to_free']:.4f} budget={row['budget']:.4f}", flush=True)
        else:
            print(k, row, flush=True)
    fn = RES / "C72_analysis_results.json"
    fn.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                  encoding="utf-8")
    print("wrote", fn)


if __name__ == "__main__":
    main()
