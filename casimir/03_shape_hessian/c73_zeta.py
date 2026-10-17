# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_zeta.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Twisted Epstein zeta Z_G(1; theta) by analytic continuation, shape-Laplacian eigenvalues at s = 1 (-4/3), van Baal contraction; s = 2 cross-check against casimir/01 C57. Paper: App. app:shapelaplacian, App. app:zeromodes (next-order terms). Pool size = env DNEC_WORKERS (default 14).

C73 (iii) + claim (4): twisted Epstein zeta Z_G(s;theta) at s=1 (analytic continuation),
shape-Laplacian eigenvalues, and the van Baal contraction G_ij M_ij = (2/pi^2) Z_G(1;theta).

All values from own Ewald and Chowla-Selberg code (c73_common.py); cross-check of s=2 values
against the project's C57 Z2 (imported read-only).  Output: results/.../C73/C73_zeta.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
import numpy as np
from c73_common import (ewald_Z, cs_Z, shape_metric_mp, GeodesicMP, symfun_mp, BETA_NC, RES,
                        AAA, PER, richardson)

DPS_VAL = 50
DPS_FD = 40


def _G_to_str(G, n=60):
    return [[mp.nstr(G[i, j], n) for j in range(3)] for i in range(3)]


def _G_from_str(L):
    return mp.matrix([[mp.mpf(x) for x in row] for row in L])


def worker(task):
    mp.mp.dps = task["dps"]
    G = _G_from_str(task["G"])
    a = [mp.mpf(x) for x in task["a"]]
    t0 = time.time()
    if task["method"] == "ewald":
        v = ewald_Z(mp.mpf(task["s"]), G, a, lam=mp.mpf(task.get("lam", "1")))
    else:
        d = task.get("delta")
        v = cs_Z(mp.mpf(task["s"]), G, a, delta=(mp.mpf(d) if d else None))
    return task["key"], mp.nstr(v, task["dps"] - 2), time.time() - t0


def main():
    mp.mp.dps = 60
    I3 = mp.eye(3)
    Gnc = shape_metric_mp(BETA_NC)
    shapes = {"cube": I3, "Gnc": Gnc}
    th = {"theta0": ("0", "0", "0"), "AAA": ("0.5", "0.5", "0.5")}
    tasks = []
    # ---------------- (A) values at dps 50: Ewald (two lambdas) and Chowla-Selberg
    for gk, G in shapes.items():
        for tk, a in th.items():
            for s in ("1", "2"):
                base = dict(G=_G_to_str(G), a=a, s=s, dps=DPS_VAL)
                tasks.append(dict(base, key=("val", gk, tk, s, "ewald_lam1"), method="ewald", lam="1"))
                tasks.append(dict(base, key=("val", gk, tk, s, "ewald_lam0.55"), method="ewald", lam="0.55"))
                delta = "1e-18" if (tk == "theta0" and s == "1") else None
                tasks.append(dict(base, key=("val", gk, tk, s, "chowla_selberg"), method="cs", delta=delta))
        # functional equation partner: Z_{G^{-1}}(1/2; 0)
        mp.mp.dps = 60
        tasks.append(dict(G=_G_to_str(mp.inverse(G)), a=th["theta0"], s="0.5", dps=DPS_VAL,
                          key=("val", gk, "theta0", "0.5_dual", "ewald_lam1"), method="ewald", lam="1"))
    # ---------------- (B) Laplacian eigenvalue checks at dps 40 (geodesic second differences)
    mp.mp.dps = 60
    eps_list = [mp.mpf("1e-6"), mp.mpf("5e-7")]
    combos = []
    for gk in ("cube", "Gnc"):
        for tk in ("theta0", "AAA"):
            for s in ("1", "2"):
                combos.append((gk, tk, s))
    for tk in ("theta0", "AAA"):
        for s in ("0.7", "1.3", "2.6"):
            combos.append(("Gnc", tk, s))
    combos.append(("Gnc", "AAA", "1.5"))
    geos = {gk: GeodesicMP(G) for gk, G in shapes.items()}
    for (gk, tk, s) in combos:
        geo = geos[gk]
        tasks.append(dict(G=_G_to_str(shapes[gk]), a=th[tk], s=s, dps=DPS_FD, method="ewald",
                          key=("lap", gk, tk, s, "center")))
        for ie, eps in enumerate(eps_list):
            for A in range(5):
                for sg in (+1, -1):
                    Gp = geo.point(A, sg * eps)
                    tasks.append(dict(G=_G_to_str(Gp), a=th[tk], s=s, dps=DPS_FD, method="ewald",
                                      key=("lap", gk, tk, s, f"e{ie}_A{A}_{sg:+d}")))
    # ---------------- (C) van Baal contraction: c-Hessian of (2/pi^2)[Z(2;0)-Z(2;c)] contracted with G
    h_list = [mp.mpf("1e-6"), mp.mpf("5e-7")]
    for gk, G in shapes.items():
        S = symfun_mp(G, mp.sqrt)  # columns v_k with sum_k v_k v_k^T = G
        tasks.append(dict(G=_G_to_str(G), a=("0", "0", "0"), s="2", dps=DPS_FD, method="ewald",
                          key=("vb", gk, "theta0", "center")))
        tasks.append(dict(G=_G_to_str(G), a=("0.5", "0.5", "0.5"), s="2", dps=DPS_FD, method="ewald",
                          key=("vb", gk, "AAA", "center")))
        for ih, h in enumerate(h_list):
            for k in range(3):
                for sg in (+1, -1):
                    for tk, a0 in (("theta0", mp.mpf(0)), ("AAA", mp.mpf(1) / 2)):
                        a = [mp.nstr(a0 + sg * h * S[i, k] / (2 * mp.pi), 55) for i in range(3)]
                        tasks.append(dict(G=_G_to_str(G), a=a, s="2", dps=DPS_FD, method="ewald",
                                          key=("vb", gk, tk, f"h{ih}_k{k}_{sg:+d}")))
    print("tasks:", len(tasks), flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("DNEC_WORKERS", "14"))) as pool:  # package: --workers
        out = pool.map(worker, tasks, chunksize=1)
    print("wall", time.time() - t0, flush=True)
    R = {k: v for (k, v, _) in out}
    cpu = sum(x[2] for x in out)

    # ---------------- assemble
    mp.mp.dps = DPS_VAL
    res = {"description": "C73 zeta values, shape-Laplacian eigenvalues, van Baal contraction",
           "Gnc_beta": BETA_NC, "Gnc_matrix": _G_to_str(Gnc, 30), "cpu_seconds": cpu, "values": {}}
    for gk in shapes:
        for tk in th:
            for s in ("1", "2"):
                e1 = mp.mpf(R[("val", gk, tk, s, "ewald_lam1")])
                e2 = mp.mpf(R[("val", gk, tk, s, "ewald_lam0.55")])
                cs = mp.mpf(R[("val", gk, tk, s, "chowla_selberg")])
                res["values"][f"Z_{gk}({s};{tk})"] = {
                    "ewald_lam1": mp.nstr(e1, 40), "ewald_lam0.55": mp.nstr(e2, 40),
                    "chowla_selberg": mp.nstr(cs, 40),
                    "max_abs_diff": mp.nstr(max(abs(e1 - e2), abs(e1 - cs)), 5)}
        z1 = mp.mpf(R[("val", gk, "theta0", "1", "ewald_lam1")])
        zd = mp.mpf(R[("val", gk, "theta0", "0.5_dual", "ewald_lam1")])
        res["values"][f"functional_eq_{gk}"] = {
            "Z_G(1;0)": mp.nstr(z1, 30), "pi*Z_Ginv(1/2;0)": mp.nstr(mp.pi * zd, 30),
            "diff": mp.nstr(z1 - mp.pi * zd, 5)}
    zc0 = mp.mpf(R[("val", "cube", "theta0", "1", "ewald_lam1")])
    zca = mp.mpf(R[("val", "cube", "AAA", "1", "ewald_lam1")])
    res["values"]["M_ii_cube_theta0 = 2 Z_I(1;0)/(3 pi^2)"] = mp.nstr(2 * zc0 / (3 * mp.pi ** 2), 20)
    res["values"]["M_ii_cube_AAA = 2 Z_I(1;AAA)/(3 pi^2)"] = mp.nstr(2 * zca / (3 * mp.pi ** 2), 20)
    z2c0 = mp.mpf(R[("val", "cube", "theta0", "2", "ewald_lam1")])
    z2ca = mp.mpf(R[("val", "cube", "AAA", "2", "ewald_lam1")])
    z2n0 = mp.mpf(R[("val", "Gnc", "theta0", "2", "ewald_lam1")])
    z2na = mp.mpf(R[("val", "Gnc", "AAA", "2", "ewald_lam1")])
    res["values"]["C_free_cube"] = mp.nstr((15 * z2c0 - 45 * z2ca) / mp.pi ** 2, 20)
    res["values"]["C_free_Gnc"] = mp.nstr((15 * z2n0 - 45 * z2na) / mp.pi ** 2, 20)
    res["values"]["gamma = Z_2(I,0)/pi^2"] = mp.nstr(z2c0 / mp.pi ** 2, 15)
    # cross-check with project's C57 (read-only import) at s = 2
    try:
        import _casimir_paths as WP; sys.path.insert(0, str(WP.step_dir("01_sm_casimir_weight")))  # package: C57
        import C57_spin_structure_Casimir as C57
        Gf = np.array([[float(Gnc[i, j]) for j in range(3)] for i in range(3)])
        res["values"]["C57_crosscheck_Gnc"] = {
            "C57_Z2(Gnc,0)": float(C57.Z2(Gf, np.zeros(3))), "own": float(z2n0),
            "C57_Z2(Gnc,AAA)": float(C57.Z2(Gf, np.full(3, 0.5))), "own_AAA": float(z2na)}
    except Exception as exc:  # pragma: no cover
        res["values"]["C57_crosscheck_Gnc"] = f"failed: {exc}"

    # Laplacian checks
    mp.mp.dps = DPS_FD
    lap = {}
    for (gk, tk, s) in combos:
        f0 = mp.mpf(R[("lap", gk, tk, s, "center")])
        D = []
        for ie, eps in enumerate(eps_list):
            tot = mp.mpf(0)
            for A in range(5):
                fp = mp.mpf(R[("lap", gk, tk, s, f"e{ie}_A{A}_+1")])
                fm = mp.mpf(R[("lap", gk, tk, s, f"e{ie}_A{A}_-1")])
                tot += (fp - 2 * f0 + fm) / eps ** 2
            D.append(tot)
        tab = richardson(D)
        Dr = tab[-1][0]
        ss = mp.mpf(s)
        mu = 4 * ss * (2 * ss - 3) / 3
        lap[f"{gk}|{tk}|s={s}"] = {
            "Z": mp.nstr(f0, 25), "Delta_Z_fd_eps1e-6": mp.nstr(D[0], 25),
            "Delta_Z_richardson": mp.nstr(Dr, 25),
            "ratio_Delta_Z_over_Z": mp.nstr(Dr / f0, 25) if f0 != 0 else None,
            "predicted_eigenvalue 4s(2s-3)/3": mp.nstr(mu, 25),
            "residual Delta_Z - mu Z": mp.nstr(Dr - mu * f0, 5)}
    res["laplacian"] = lap

    # van Baal contraction
    vb = {}
    for gk in shapes:
        Gm = shapes[gk]
        for tk in ("theta0", "AAA"):
            Z20 = mp.mpf(R[("vb", gk, "theta0", "center")])
            Zc = mp.mpf(R[("vb", gk, tk, "center")])
            V = lambda Zval: 2 / mp.pi ** 2 * (Z20 - Zval)
            D, slopes = [], []
            for ih, h in enumerate(h_list):
                tot = mp.mpf(0)
                for k in range(3):
                    Vp = V(mp.mpf(R[("vb", gk, tk, f"h{ih}_k{k}_+1")]))
                    Vm = V(mp.mpf(R[("vb", gk, tk, f"h{ih}_k{k}_-1")]))
                    if tk == "theta0":
                        slopes.append((Vp + Vm) / (2 * h))      # -> cusp coefficient 2
                        tot += ((Vp - 2 * h) + (Vm - 2 * h) - 0) / h ** 2   # f = V - 2|c|_{G^-1}, f(0)=0
                    else:
                        tot += (Vp - 2 * V(Zc) + Vm) / h ** 2
                D.append(tot)
            Dr = richardson(D)[-1][0]
            zkey = ("val", gk, tk, "1", "ewald_lam1")
            mp.mp.dps = DPS_VAL
            z1 = mp.mpf(R[zkey])
            mp.mp.dps = DPS_FD
            pred = 2 / mp.pi ** 2 * z1
            ent = {"G_ij M_ij (fd, richardson)": mp.nstr(Dr, 25), "(2/pi^2) Z_G(1;theta)": mp.nstr(pred, 25),
                   "residual": mp.nstr(Dr - pred, 5)}
            if tk == "theta0":
                ent["cusp_slope_estimates (V(hv)+V(-hv))/(2h) -> 2"] = [mp.nstr(x, 15) for x in slopes]
            vb[f"{gk}|{tk}"] = ent
    res["van_baal_contraction"] = vb
    out = RES / "C73_zeta.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["values"], indent=1)[:4000])
    for k, v in lap.items():
        print(k, v["ratio_Delta_Z_over_Z"], v["predicted_eigenvalue 4s(2s-3)/3"], v["residual Delta_Z - mu Z"])
    for k, v in vb.items():
        print(k, v)


if __name__ == "__main__":
    main()
