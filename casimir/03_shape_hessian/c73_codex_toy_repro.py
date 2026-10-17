# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_codex_toy_repro.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Independent recomputation of the four-level toy of the Codex state-response audit (metric typed in as data). Supports PRD App. app:shapelaplacian (no-go with finite-volume matter). Pool size = env DNEC_WORKERS.

C73: independent re-computation of the Codex four-level toy (vacuum + three axis pair levels,
multiplicity 1, no zero-mode energy) on the metric quoted in Codex's state_response_audit.json
(metric numbers typed in here as data; no Codex code is imported or executed).
Own Ewald vacuum E_vac = -C_free(G), analytic forces, mp FD Laplacians (dps 40) with Richardson.
Output: results/.../C73/C73_codex_toy_repro.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
from c73_common import GeodesicMP, RES, richardson
from c73_thermo import _vac_worker, gibbs, solve_T_fixed_S, DPS

G_CODEX = [["1.3525792868995485", "0.0691584867240451", "-0.028315851499890857"],
           ["0.06915848672404509", "0.9257545553267998", "0.03189621266509226"],
           ["-0.02831585149989085", "0.03189621266509226", "0.8034828551059328"]]
CODEX = {8: dict(U=-41.08605377232375, S=1.0836696288834125, corr=0.027681733941915008,
                 can=-114.56514426366283, ise=-114.53746252972091),
         12: dict(U=-39.45957720302424, S=1.2526942713578102, corr=0.05456868355255187,
                  can=-109.72386852269828, ise=-109.66929983914572),
         20: dict(U=-38.14210922301652, S=1.340664497994257, corr=0.06223994317793281,
                  can=-104.97920974874263, ise=-104.91696980556469)}


def main():
    mp.mp.dps = DPS
    G0 = mp.matrix([[mp.mpf(x) for x in row] for row in G_CODEX])
    G0 = (G0 + G0.T) / 2
    geo = GeodesicMP(G0)
    eps_list = [mp.mpf("1e-6"), mp.mpf("5e-7")]
    pts = [G0] + [geo.point(A, sg * e) for e in eps_list for A in range(5) for sg in (+1, -1)]
    with Pool(int(os.environ.get("DNEC_WORKERS", "14"))) as pool:  # package: --workers
        vac = pool.map(_vac_worker, [([[mp.nstr(P[i, j], 55) for j in range(3)] for i in range(3)],) for P in pts])
    mp.mp.dps = DPS
    vac = [mp.mpf(v) for v in vac]
    E = lambda i: [vac[i]] + [vac[i] + 4 * mp.pi * mp.sqrt(mp.inverse(pts[i])[j, j]) for j in range(3)]
    Es_pts = [E(i) for i in range(len(pts))]
    gs = [mp.mpf(1)] * 4
    # analytic forces of the pair levels: d/dt 4 pi |S^{-1} e_j| along S e^{2tX} S
    fA = [[mp.mpf(0)] * 5]
    for j in range(3):
        ej = mp.matrix([1 if i == j else 0 for i in range(3)])
        kt = geo.Sinv * ej
        nk = mp.sqrt((kt.T * kt)[0])
        fA.append([2 * (-2 * mp.pi * (kt.T * geo.X[A] * kt)[0] / nk) for A in range(5)])

    def lap(vals):
        D = []
        for ie, eps in enumerate(eps_list):
            D.append(sum((vals[1 + ie * 10 + 2 * A] - 2 * vals[0] + vals[2 + ie * 10 + 2 * A]) / eps ** 2 for A in range(5)))
        return richardson(D)[-1][0]

    rows = []
    for T in (8, 12, 20):
        Tm = mp.mpf(T)
        F0, U0, p = gibbs(Es_pts[0], gs, Tm)
        S0 = (U0 - F0) / Tm
        varE = mp.fsum(pi * e ** 2 for pi, e in zip(p, Es_pts[0])) - U0 ** 2
        vf, cv = [], []
        for A in range(5):
            m1 = mp.fsum(pi * fA[j][A] for j, pi in enumerate(p))
            vf.append(mp.fsum(pi * fA[j][A] ** 2 for j, pi in enumerate(p)) - m1 ** 2)
            cv.append(mp.fsum(pi * fA[j][A] * Es_pts[0][j] for j, pi in enumerate(p)) - m1 * U0)
        can = mp.mpf(8) / 3 * U0 - mp.fsum(vf) / Tm
        corr = mp.fsum(c ** 2 for c in cv) / (Tm * varE)
        ise = can + corr
        lhsT = lap([gibbs(Es_pts[i], gs, Tm)[0] for i in range(len(pts))])
        lhsS = lap([U0] + [solve_T_fixed_S(Es_pts[i], gs, S0, Tm)[1] for i in range(1, len(pts))])
        c = CODEX[T]
        rows.append({"T_L": T, "U_own": float(U0), "U_codex": c["U"], "S_own": float(S0), "S_codex": c["S"],
                     "corr_own": float(corr), "corr_codex": c["corr"],
                     "canonical_formula_own": float(can), "canonical_codex(eps=2.5e-4)": c["can"],
                     "canonical_numeric_own": float(lhsT), "canonical_residual_own": mp.nstr(lhsT - can, 4),
                     "isentropic_formula_own": float(ise), "isentropic_codex(eps=2.5e-4)": c["ise"],
                     "isentropic_numeric_own": float(lhsS), "isentropic_residual_own": mp.nstr(lhsS - ise, 4)})
    (RES / "C73_codex_toy_repro.json").write_text(json.dumps({"description": __doc__, "rows": rows}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
