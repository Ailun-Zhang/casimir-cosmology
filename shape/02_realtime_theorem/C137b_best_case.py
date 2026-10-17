# PROVENANCE: 04_calculations/code/20261013_claude_v12/state_selection/C137b_best_case.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Best case of an earlier (adiabatic) preparation propagated to the turn (D109 Prop. 2; R_P, shear ratio). Paper: PRD App. app:theorem.

C137b (Claude v12, state_selection): most favourable incoming state for a plateau-supported turn.

v11 background exactly: L = L_b sqrt(cosh 2t) (time unit 1/H_eff), shape mode
    beta'' + 3H beta' - kappa sech^2(2t) beta = 0,  H = tanh 2t,  P = m(t) beta', m = M_P^2 L^3.
Best case: the shape is in an instantaneous adiabatic Gaussian with frequency omega_x = c*H_eff
(c >= 1 is the weakest adiabaticity) at the moment the plateau contraction starts, N_pre e-folds
before the turn (L_x = L_b e^{N_pre}); afterwards v11's own equation applies (frozen + Casimir).
Reports, relative to v11's natural turning-point state (real Gaussian, Var beta = 1/(2 m_b |omega_b|)):
  R_P  = Var P at the turn / natural,  F_out = Var beta at t=+6 / natural outgoing,
  shear ratio in v11 absolute units:  sum_A Var P_A / (12 M_P^4 L_b^6 H^2)
     = sum_A R_A sqrt|kappa_A| / (24 * M_P^2 L_b^3 H_eff),  M_P^2 L_b^3 H_eff = 722^2*0.00437243.
Natural-state outgoing widths reproduce v11's aspect gain as a check (m11 at the turn).
"""
import json, hashlib, os, time
import numpy as np
from scipy.integrate import solve_ivp

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shape_paths as WP  # noqa: E402
OUT = str(WP.wdir("r1/state_selection/run02"))
                   # (package: the work-tree path above replaces the original results path)
MPL3H = 722.0 ** 2 * 0.00437243          # M_P^2 L_b^3 H_eff (xi = 0.30), dimensionless


def transfer(kappa, t0, t1):
    """2x2 transfer of (beta, P/m_b) in units m_b = 1 (L_b = 1)."""
    def rhs(t, y):
        H = np.tanh(2 * t)
        f = kappa / np.cosh(2 * t) ** 2
        # y = [b1, b1', b2, b2']
        return [y[1], -3 * H * y[1] + f * y[0], y[3], -3 * H * y[3] + f * y[2]]
    def run(b, bd):
        s = solve_ivp(rhs, (t0, t1), [b[0], bd[0], b[1], bd[1]], rtol=1e-12, atol=1e-14,
                      method="DOP853")
        return s.y[:, -1]
    return run


def evolve_cov(kappa, t0, t1, cov0):
    """Evolve covariance of (beta, P) with m = L^3 = cosh(2t)^{3/2} (L_b = 1, M_P = 1)."""
    m = lambda t: np.cosh(2 * t) ** 1.5
    def rhs(t, y):
        H = np.tanh(2 * t)
        f = kappa / np.cosh(2 * t) ** 2
        return [y[1], -3 * H * y[1] + f * y[0], y[3], -3 * H * y[3] + f * y[2]]
    # columns: solution with (beta, P) = (1,0) and (0,1) at t0
    y0 = [1.0, 0.0, 0.0, 1.0 / m(t0)]
    s = solve_ivp(rhs, (t0, t1), y0, rtol=1e-12, atol=1e-16, method="DOP853")
    b1, bd1, b2, bd2 = s.y[:, -1]
    M = np.array([[b1, b2], [m(t1) * bd1, m(t1) * bd2]])
    return M @ cov0 @ M.T, M


def natural(kappa):
    w = np.sqrt(abs(kappa))                     # |omega_b| in units H_eff (m_b = 1)
    s = 1.0 / (2 * w)
    return np.diag([s, 1.0 / (4 * s)]), w


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0w = time.time()
    with open(__file__, "rb") as f:
        code_sha = hashlib.sha256(f.read()).hexdigest()
    res = dict(code=os.path.basename(__file__), code_sha256=code_sha, MPL3H=MPL3H, rows=[])
    for kappa in (19.32, -9.07):
        cov_nat, w = natural(kappa)
        cov_out_nat, M_nat = evolve_cov(kappa, 0.0, 6.0, cov_nat)
        for c in (1.0, 3.0):
            for N_pre in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
                tx = -0.5 * np.arccosh(np.exp(2 * N_pre))      # L(tx) = e^{N_pre} L_b
                mx = np.cosh(2 * tx) ** 1.5
                Hx = abs(np.tanh(2 * tx))
                wx = c * max(Hx, 1e-12)
                cov_x = np.diag([1.0 / (2 * mx * wx), mx * wx / 2.0])
                cov_T, _ = evolve_cov(kappa, tx, 0.0, cov_x)
                cov_O, _ = evolve_cov(kappa, 0.0, 6.0, cov_T)
                R_P = cov_T[1, 1] / cov_nat[1, 1]
                F = cov_O[0, 0] / cov_out_nat[0, 0]
                res["rows"].append(dict(kappa=kappa, c=c, N_pre=N_pre, t_x=tx, m_x=mx,
                                        R_P=R_P, R_beta=cov_T[0, 0] / cov_nat[0, 0],
                                        F_out=F, sqrtF=np.sqrt(F),
                                        shear_contrib=R_P * np.sqrt(abs(kappa)) / (24 * MPL3H)))
        res[f"natural_kappa_{kappa}"] = dict(m11_turn_to_out=M_nat[0, 0], var_out_nat=cov_out_nat[0, 0],
                                             floor=abs(M_nat[0, 0] * M_nat[0, 1]))
    # shear ratio (2 E + 3 T2 modes, same N_pre and c)
    tab = {}
    for r in res["rows"]:
        tab.setdefault((r["c"], r["N_pre"]), {})[r["kappa"]] = r
    res["shear_ratio"] = [dict(c=c, N_pre=N, ratio=2 * d[19.32]["shear_contrib"] + 3 * d[-9.07]["shear_contrib"],
                               sqrtF_E=d[19.32]["sqrtF"], R_P_E=d[19.32]["R_P"])
                          for (c, N), d in sorted(tab.items())]
    for s in res["shear_ratio"]:
        print(f"c={s['c']} N_pre={s['N_pre']:5.2f}  R_P(E)={s['R_P_E']:.3e}  sqrt(F_E)={s['sqrtF_E']:.3e}  shear/criterion={s['ratio']:.3e}")
    for k in (19.32, -9.07):
        print(k, res[f"natural_kappa_{k}"])
    res["runtime_s"] = time.time() - t0w
    with open(os.path.join(OUT, "C137b_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
