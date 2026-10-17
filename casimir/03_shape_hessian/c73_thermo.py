# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_thermo.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Gibbs and fixed-entropy shape-Laplacian tests on free lattice-mode states (D55 Thm 1): Delta F|_T = (8/3)(U - <q>/L) - T^-1 sum Var f_A; flux sector control. Paper: PRD Sec. VII B (Eq. eq:s-nogo), App. app:shapelaplacian. Pool size = env DNEC_WORKERS (default 14).

C73 (i): Gibbs (fixed T) and fixed-entropy shape-Laplacian tests on free lattice-mode states.

(a) mpmath (dps 40): finite level set on the NON-cubic unimodular G_nc (off-diagonal entries):
    vacuum E_vac(G) = -C_free(G)/L  (SM, AAA fermions, static graviton; own Ewald)
    + photon-pair states (+k,-k), energy 2 omega_k, omega_k = 2 pi |G^{-1/2} k| / L
      (3 axis pairs, 6 face-diagonal pairs, 3 double axis pairs; polarization multiplicities)
    + every level duplicated with a shape-independent zero-mode excitation (q0 -> q0+q1).
    Five-direction geodesic second differences, Richardson in eps^2.
(b) numpy: the COMPLETE free Gibbs ensemble (all oscillator occupation states) of the massless
    free SM content: 30 bosonic dof (periodic, k != 0), 90 fermionic dof (AAA), on G_nc.
(c) numpy: a free compact U(1) electric-flux sector, energies (g^2/2L) e^T G e (e in Z^3), which is
    NOT shape independent (Delta = 20/3): the identity acquires the extra term +4 <eps_flux>.
Formulas tested:
    Delta F|_T  = (8/3)(U - <q>/L) - (1/T) sum_A Var(f_A)
    Delta U|_S  = (8/3)(U - <q>/L) - (1/T) sum_A [Var(f_A) - Cov(f_A,E)^2/Var(E)]
Output: results/.../C73/C73_thermo.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time, itertools
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mpmath as mp
from c73_common import (ewald_Z, shape_metric_mp, GeodesicMP, BETA_NC, RES, AAA, PER, richardson,
                        shape_metric_np, GeodesicNP, C_free_np, basis_np)

DPS = 40


def _vac_worker(args):
    Gs, = args
    mp.mp.dps = DPS
    G = mp.matrix([[mp.mpf(x) for x in row] for row in Gs])
    C = (15 * ewald_Z(2, G, PER) - 45 * ewald_Z(2, G, AAA)) / mp.pi ** 2
    return mp.nstr(-C, DPS - 2)


# ============================================================================ (a) mpmath level set
E1 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
FD6 = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]


def build_levels(q0, q1):
    lev = [dict(g=1, ks=[], q=q0)]
    lev += [dict(g=4, ks=[k], q=q0) for k in E1]
    lev += [dict(g=4, ks=[k], q=q0) for k in FD6]
    lev += [dict(g=16, ks=[E1[i], E1[j]], q=q0) for (i, j) in ((0, 1), (0, 2), (1, 2))]
    lev += [dict(g=L["g"], ks=L["ks"], q=q0 + q1) for L in list(lev)]
    return lev


def omega_mp(G, k):
    Gi = mp.inverse(G)
    kv = mp.matrix(k)
    return 2 * mp.pi * mp.sqrt((kv.T * Gi * kv)[0])


def level_energies(G, Evac, levels):
    return [Evac + sum((2 * omega_mp(G, k) for k in L["ks"]), mp.mpf(0)) + L["q"] for L in levels]


def gibbs(Es, gs, T):
    w = [g * mp.exp(-E / T) for E, g in zip(Es, gs)]
    Zp = mp.fsum(w)
    p = [x / Zp for x in w]
    F = -T * mp.log(Zp)
    U = mp.fsum(pi * E for pi, E in zip(p, Es))
    return F, U, p


def solve_T_fixed_S(Es, gs, S0, T):
    for _ in range(200):
        F, U, p = gibbs(Es, gs, T)
        S = (U - F) / T
        varE = mp.fsum(pi * E ** 2 for pi, E in zip(p, Es)) - U ** 2
        dT = (S - S0) / (varE / T ** 3)
        T = T - dT
        if abs(dT) < mp.mpf(10) ** (-(DPS - 4)) * T:
            break
    F, U, p = gibbs(Es, gs, T)
    return T, U


def run_levelset(G0, label, temps, q0, q1, pool):
    mp.mp.dps = DPS
    eps_list = [mp.mpf("1e-6"), mp.mpf("5e-7")]
    geo = GeodesicMP(G0)
    pts = [("c", None, None, G0)]
    for ie, eps in enumerate(eps_list):
        for A in range(5):
            for sg in (+1, -1):
                pts.append((ie, A, sg, geo.point(A, sg * eps)))
    vac = pool.map(_vac_worker, [([[mp.nstr(P[i, j], 55) for j in range(3)] for i in range(3)],) for (_, _, _, P) in pts])
    mp.mp.dps = DPS
    vac = [mp.mpf(v) for v in vac]
    levels = build_levels(mp.mpf(q0), mp.mpf(q1))
    gs = [mp.mpf(L["g"]) for L in levels]
    qs = [L["q"] for L in levels]
    Es_pts = [level_energies(P, vac[i], levels) for i, (_, _, _, P) in enumerate(pts)]

    def lap_from(vals):
        D = []
        for ie, eps in enumerate(eps_list):
            tot = mp.mpf(0)
            for A in range(5):
                ip = 1 + ie * 10 + A * 2
                tot += (vals[ip] - 2 * vals[0] + vals[ip + 1]) / eps ** 2
            D.append(tot)
        return richardson(D)[-1][0], D

    # per-level identity Delta e_s = (8/3) e_s, with e_s = E_s - q_s
    lev_res = []
    for j in range(len(levels)):
        e_vals = [Es_pts[i][j] - qs[j] for i in range(len(pts))]
        Dl, _ = lap_from(e_vals)
        lev_res.append(abs(Dl - mp.mpf(8) / 3 * e_vals[0]))
    vac_lap, _ = lap_from(vac)
    # analytic forces at the center (excitation part; the vacuum gradient is a common shift)
    Sinv = geo.Sinv
    X = geo.X
    fA = []
    for L in levels:
        row = []
        for A in range(5):
            tot = mp.mpf(0)
            for k in L["ks"]:
                kt = Sinv * mp.matrix(k)
                nk = mp.sqrt((kt.T * kt)[0])
                tot += 2 * (-2 * mp.pi * (kt.T * X[A] * kt)[0] / nk)
            row.append(tot)
        fA.append(row)
    out = {"label": label, "q0": q0, "q1": q1, "n_levels": len(levels),
           "vacuum_E": mp.nstr(vac[0], 20),
           "vacuum_lap_over_E": mp.nstr(vac_lap / vac[0], 20),
           "max_level_identity_residual |Delta e_s - (8/3)e_s|": mp.nstr(max(lev_res), 5),
           "temps": []}
    for T in temps:
        T = mp.mpf(T)
        Es0 = Es_pts[0]
        F0, U0, p = gibbs(Es0, gs, T)
        S0 = (U0 - F0) / T
        qbar = mp.fsum(pi * q for pi, q in zip(p, qs))
        varE = mp.fsum(pi * E ** 2 for pi, E in zip(p, Es0)) - U0 ** 2
        var_f, cov_fE = [], []
        for A in range(5):
            m1 = mp.fsum(pi * fA[j][A] for j, pi in enumerate(p))
            m2 = mp.fsum(pi * fA[j][A] ** 2 for j, pi in enumerate(p))
            mE = mp.fsum(pi * fA[j][A] * Es0[j] for j, pi in enumerate(p))
            var_f.append(m2 - m1 ** 2)
            cov_fE.append(mE - m1 * U0)
        base = mp.mpf(8) / 3 * (U0 - qbar)
        rhs_T = base - mp.fsum(var_f) / T
        corr = mp.fsum(c ** 2 for c in cov_fE) / (T * varE)
        proj = [var_f[A] - cov_fE[A] ** 2 / varE for A in range(5)]
        rhs_S = base - mp.fsum(proj) / T
        # numerical Laplacians
        Fv = [gibbs(Es_pts[i], gs, T)[0] for i in range(len(pts))]
        lhs_T, DT = lap_from(Fv)
        Uv = [U0] + [solve_T_fixed_S(Es_pts[i], gs, S0, T)[1] for i in range(1, len(pts))]
        lhs_S, DS = lap_from(Uv)
        out["temps"].append({
            "T_L": float(T), "U": mp.nstr(U0, 20), "F": mp.nstr(F0, 20), "S": mp.nstr(S0, 20),
            "<q>": mp.nstr(qbar, 20), "Var(E)": mp.nstr(varE, 20),
            "(8/3)(U-<q>)": mp.nstr(base, 20),
            "sum_A Var(f_A)/T": mp.nstr(mp.fsum(var_f) / T, 20),
            "fixed_S_correction sum_A Cov(f_A,E)^2/(T Var E)": mp.nstr(corr, 20),
            "Cov(f_A,E)": [mp.nstr(c, 12) for c in cov_fE],
            "projected variances Var(f_A)-Cov^2/VarE (>=0)": [mp.nstr(x, 12) for x in proj],
            "LapF_T numeric (Richardson)": mp.nstr(lhs_T, 25), "LapF_T formula": mp.nstr(rhs_T, 25),
            "LapF_T residual": mp.nstr(lhs_T - rhs_T, 5), "LapF_T raw eps=1e-6": mp.nstr(DT[0], 20),
            "LapU_S numeric (Richardson)": mp.nstr(lhs_S, 25), "LapU_S formula": mp.nstr(rhs_S, 25),
            "LapU_S residual": mp.nstr(lhs_S - rhs_S, 5), "LapU_S raw eps=1e-6": mp.nstr(DS[0], 20),
            "LapU_S - LapF_T (numeric)": mp.nstr(lhs_S - lhs_T, 20)})
    return out


# ============================================================================ (b) numpy full free SM gas
def mode_sets(G0, T, fac=42.0):
    Gi = np.linalg.inv(G0)
    out = {}
    for name, shift in (("B", 0.0), ("F", 0.5)):
        wmax = fac * T
        R = int(np.ceil(wmax / (2 * np.pi) / np.sqrt(np.min(np.linalg.eigvalsh(Gi))))) + 2
        r = np.arange(-R, R + 1) + shift
        K = np.array(list(itertools.product(r, r, r)))
        if shift == 0.0:
            K = K[np.any(K != 0, axis=1)]
        w = 2 * np.pi * np.sqrt(np.einsum('ni,ij,nj->n', K, Gi, K))
        out[name] = K[w <= wmax]
    return out


DB, DF = 30.0, 90.0


def thermo_np(G, T, modes, Evac):
    Gi = np.linalg.inv(G)
    res = {"F": Evac, "U": Evac, "S": 0.0, "VarE": 0.0}
    for name, d in (("B", DB), ("F", DF)):
        K = modes[name]
        w = 2 * np.pi * np.sqrt(np.einsum('ni,ij,nj->n', K, Gi, K))
        x = w / T
        if name == "B":
            n = 1.0 / np.expm1(x)
            res["F"] += d * T * np.sum(np.log1p(-np.exp(-x)))
            v = n * (1 + n)
            res["S"] += d * np.sum((1 + n) * np.log1p(n) - n * np.log(n))
        else:
            n = 1.0 / (np.exp(x) + 1)
            res["F"] -= d * T * np.sum(np.log1p(np.exp(-x)))
            v = n * (1 - n)
            res["S"] -= d * np.sum(n * np.log(n) + (1 - n) * np.log1p(-n))
        res["U"] += d * np.sum(w * n)
        res["VarE"] += d * np.sum(w ** 2 * v)
    return res


def forces_np(geo, T, modes):
    X = geo.X
    out_var = np.zeros(5)
    out_cov = np.zeros(5)
    for name, d in (("B", DB), ("F", DF)):
        Kt = modes[name] @ geo.Sinv.T  # k~ = S^{-1} k  (S symmetric)
        nk = np.linalg.norm(Kt, axis=1)
        w = 2 * np.pi * nk
        x = w / T
        n = 1.0 / np.expm1(x) if name == "B" else 1.0 / (np.exp(x) + 1)
        v = n * (1 + n) if name == "B" else n * (1 - n)
        for A in range(5):
            dw = -2 * np.pi * np.einsum('ni,ij,nj->n', Kt, X[A], Kt) / nk
            out_var[A] += d * np.sum(dw ** 2 * v)
            out_cov[A] += d * np.sum(dw * w * v)
    return out_var, out_cov


def solve_T_np(G, S0, T, modes, Evac):
    for _ in range(100):
        r = thermo_np(G, T, modes, Evac)
        dT = (r["S"] - S0) / (r["VarE"] / T ** 3)
        T -= dT
        if abs(dT) < 1e-15 * T:
            break
    return T, thermo_np(G, T, modes, Evac)


def run_full_gas(G0, temps):
    geo = GeodesicNP(G0)
    eps_list = [0.02, 0.01, 0.005]
    pts = [G0]
    for eps in eps_list:
        for A in range(5):
            for sg in (+1, -1):
                pts.append(geo.point(A, sg * eps))
    vac = [-C_free_np(P) for P in pts]

    def lap(vals):
        D = []
        for ie, eps in enumerate(eps_list):
            tot = 0.0
            for A in range(5):
                ip = 1 + ie * 10 + A * 2
                tot += (vals[ip] - 2 * vals[0] + vals[ip + 1]) / eps ** 2
            D.append(tot)
        return richardson(D)[-1][0], D

    vac_lap = lap(vac)[0]
    rows = []
    for T in temps:
        modes = mode_sets(G0, T)
        c = thermo_np(G0, T, modes, vac[0])
        var_f, cov_fE = forces_np(geo, T, modes)
        base = 8.0 / 3.0 * c["U"]
        rhs_T = base - var_f.sum() / T
        corr = np.sum(cov_fE ** 2) / (T * c["VarE"])
        rhs_S = rhs_T + corr
        Fv = [thermo_np(P, T, modes, vac[i])["F"] for i, P in enumerate(pts)]
        lhs_T, DT = lap(Fv)
        Uv = [c["U"]] + [solve_T_np(P, c["S"], T, modes, vac[i])[1]["U"] for i, P in enumerate(pts) if i > 0]
        lhs_S, DS = lap(Uv)
        rows.append({"T_L": T, "n_boson_modes": int(len(modes["B"])), "n_fermion_modes": int(len(modes["F"])),
                     "U": c["U"], "F": c["F"], "S": c["S"], "(8/3)U": base,
                     "sum_A Var(f_A)/T": var_f.sum() / T,
                     "fixed_S_correction": corr,
                     "projected variances": (var_f - cov_fE ** 2 / c["VarE"]).tolist(),
                     "LapF_T numeric": lhs_T, "LapF_T formula": rhs_T, "LapF_T rel.residual": (lhs_T - rhs_T) / abs(rhs_T),
                     "LapU_S numeric": lhs_S, "LapU_S formula": rhs_S, "LapU_S rel.residual": (lhs_S - rhs_S) / abs(rhs_S),
                     "Richardson D(eps) LapF": DT, "Richardson D(eps) LapU_S": DS})
    return {"vacuum_E": vac[0], "vacuum_lap_over_E": vac_lap / vac[0], "rows": rows,
            "note": "complete free-field Gibbs ensemble (all occupations) of 30 bosonic (periodic, k!=0) + 90 fermionic (AAA) massless dof; zero modes excluded (<q>=0)"}


# ============================================================================ (c) compact U(1) flux sector
def run_flux(G0, gcoup_list, temps):
    geo = GeodesicNP(G0)
    eps_list = [0.02, 0.01, 0.005]
    res = {"single_level_check": [], "thermal": []}
    for e in ((1, 0, 0), (1, 1, 0), (1, 2, -1)):
        e = np.array(e, float)
        f = lambda G: 0.5 * e @ G @ e
        D = []
        for eps in eps_list:
            D.append(sum((f(geo.point(A, eps)) - 2 * f(G0) + f(geo.point(A, -eps))) / eps ** 2 for A in range(5)))
        res["single_level_check"].append({"e": e.tolist(), "Delta eps / eps": richardson(D)[-1][0] / f(G0),
                                          "expected 20/3": 20.0 / 3.0})
    R = 60
    r = np.arange(-R, R + 1)
    Ev = np.array(list(itertools.product(r, r, r)), float)
    q = np.einsum('ni,ij,nj->n', Ev, G0, Ev)
    for gc in gcoup_list:
        for T in temps:
            eps_e = 0.5 * gc ** 2 * q
            w = np.exp(-(eps_e - eps_e.min()) / T)
            mean = np.sum(w * eps_e) / np.sum(w)
            res["thermal"].append({"g": gc, "T_L": T, "<eps_flux>": mean,
                                   "extra term 4<eps_flux> in Delta<E>": 4 * mean,
                                   "equipartition 1.5 T": 1.5 * T})
    # (c2) direct Gibbs test of the flux sector alone: numeric Delta F_flux vs the free-mode formula
    #      (8/3)<eps> - (1/T) sum Var(f_A)  and vs the corrected (8/3)<eps> + 4<eps> - (1/T) sum Var(f_A)
    res["gibbs_flux_sector"] = []
    for gc in gcoup_list:
        for T in temps:
            def Fflux(G):
                ee = 0.5 * gc ** 2 * np.einsum('ni,ij,nj->n', Ev, G, Ev)
                m = ee.min()
                return m - T * np.log(np.sum(np.exp(-(ee - m) / T)))
            D = []
            F0 = Fflux(G0)
            for eps in eps_list:
                D.append(sum((Fflux(geo.point(A, eps)) - 2 * F0 + Fflux(geo.point(A, -eps))) / eps ** 2
                             for A in range(5)))
            lhs = richardson(D)[-1][0]
            ee = 0.5 * gc ** 2 * q
            p = np.exp(-(ee - ee.min()) / T)
            p /= p.sum()
            mean = np.sum(p * ee)
            Et = Ev @ geo.S.T  # e~ = S e
            varsum = 0.0
            for A in range(5):
                fA = gc ** 2 * np.einsum('ni,ij,nj->n', Et, geo.X[A], Et)
                varsum += np.sum(p * fA ** 2) - np.sum(p * fA) ** 2
            free_formula = 8.0 / 3.0 * mean - varsum / T
            res["gibbs_flux_sector"].append({
                "g": gc, "T_L": T, "LapF_flux numeric": lhs,
                "free-mode formula (8/3)<eps>-(1/T)sumVar": free_formula,
                "corrected (20/3)<eps>-(1/T)sumVar": free_formula + 4 * mean,
                "numeric - free formula": lhs - free_formula, "4<eps_flux>": 4 * mean})
    return res


def main():
    t0 = time.time()
    mp.mp.dps = 60
    Gnc = shape_metric_mp(BETA_NC)
    out = {"description": __doc__}
    with Pool(int(os.environ.get("DNEC_WORKERS", "14"))) as pool:  # package: --workers
        out["levelset_Gnc"] = run_levelset(Gnc, "G_nc", [4, 8, 16], "0.75", "2.5", pool)
        mp.mp.dps = 60
        out["levelset_cube"] = run_levelset(mp.eye(3), "cube", [8], "0.75", "2.5", pool)
    Gnc_np = shape_metric_np(BETA_NC)
    out["full_free_SM_gas_Gnc"] = run_full_gas(Gnc_np, [0.6, 1.0, 1.5])
    out["compact_U1_flux_Gnc"] = run_flux(Gnc_np, [0.36, 1.0], [1.0, 3.0])
    out["wall_seconds"] = time.time() - t0
    (RES / "C73_thermo.json").write_text(json.dumps(out, indent=1, default=float))
    for key in ("levelset_Gnc", "levelset_cube"):
        o = out[key]
        print(key, "vac lap/E", o["vacuum_lap_over_E"], "level resid", o["max_level_identity_residual |Delta e_s - (8/3)e_s|"])
        for r in o["temps"]:
            print("  T", r["T_L"], "U", r["U"], "<q>", r["<q>"], "corrS", r["fixed_S_correction sum_A Cov(f_A,E)^2/(T Var E)"])
            print("    LapF", r["LapF_T numeric (Richardson)"], r["LapF_T formula"], r["LapF_T residual"])
            print("    LapU_S", r["LapU_S numeric (Richardson)"], r["LapU_S formula"], r["LapU_S residual"])
    o = out["full_free_SM_gas_Gnc"]
    print("full gas vac lap/E", o["vacuum_lap_over_E"])
    for r in o["rows"]:
        print("  T", r["T_L"], "U", r["U"], "corr", r["fixed_S_correction"], "relres F", r["LapF_T rel.residual"],
              "relres U_S", r["LapU_S rel.residual"], "modes", r["n_boson_modes"], r["n_fermion_modes"])
    print(json.dumps(out["compact_U1_flux_Gnc"], indent=0, default=float)[:1500])
    print("wall", out["wall_seconds"])


if __name__ == "__main__":
    main()
