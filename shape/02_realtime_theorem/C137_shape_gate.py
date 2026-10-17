# PROVENANCE: 04_calculations/code/20261013_claude_v12/state_selection/C137_shape_gate.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Shape gate on v11's background (D109): the outgoing aspect variance of a state that is real at t_x, freezing ratio, inheritance tests; R14-verified. Paper: PRD Sec. VIII B (sec:s-realmoment, Eq. eq:s-Gtx), App. app:theorem (app:theorem).

C137 (Claude v12, branch state_selection): can the shape state of a plateau-supported
Casimir turn be inherited from an adiabatic (incoming-vacuum) past?

Units: reduced M_P = 1, plateau Hubble rate H_eff = sqrt(V_b/3) = 1.
Background energy density (prescribed, Friedmann-consistent toy):
    rho(L) = V_b * g(L/L_x) + A/L^2 - C/L^4,
    g(y) = (1 + y^(3n))^(-1/n)   (plateau for L << L_x, matter-like for L >> L_x),
i.e. a time-reversed reheating -> plateau contraction of N_pre = ln(L_x/L_b) e-folds,
a winding-like shape-stiff component A/L^2 dominating the far past, and the Casimir turn.
Shape mode (one of the five, linearised at a stationary shape):
    beta'' + 3 H beta' + Omega^2(L) beta = 0,
    Omega^2 = A k_w / L^2 - C kt / L^4,   kt = kappa/3 (v11 normalisation),
canonical momentum P = L^3 beta'. Quantum: [beta, P] = i (M_P = 1).
Incoming vacuum: exact positive-frequency solution of the pure-winding era,
    beta ~ exp((-1 + i nu) N), nu = sqrt(3 k_w - 1), N = ln L  (contracting: t up, N down).
Outputs (per N_pre): Var P at the turn relative to v11's natural state, the semiclassical
shear criterion <S> < 4 V_b, and the outgoing (frozen) variance relative to the natural one.
Also: the closed-form freezing time of v11's shape modes, omega/|H| = sqrt|kappa|/|sinh 2ht|.
"""
import json, hashlib, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shape_paths as WP  # noqa: E402
OUT = str(WP.wdir("r1/state_selection/run01"))
                   # (package: the work-tree path above replaces the original results path)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Model:
    def __init__(self, kappa, C=30.0, a=1e-2, k_w=2.0, N_pre=4.0, n=2.0):
        self.Vb = 3.0
        self.C = C
        self.Lb0 = (C / self.Vb) ** 0.25          # turn size without winding
        self.A = a * self.Vb * self.Lb0 ** 2      # winding small at the turn: A/Lb^2 = a V_b
        self.Lx = self.Lb0 * np.exp(N_pre)
        self.n = n
        self.kw = k_w
        self.kt = kappa / 3.0
        self.kappa = kappa
        # exact turn size: rho(Lb) = 0 (g ~ 1 there)
        f = lambda L: self.rho(L)
        lo, hi = 0.5 * self.Lb0, 2.0 * self.Lb0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        self.Lb = 0.5 * (lo + hi)

    def g(self, L):
        y = L / self.Lx
        return (1.0 + y ** (3 * self.n)) ** (-1.0 / self.n)

    def rho(self, L):
        return self.Vb * self.g(L) + self.A / L ** 2 - self.C / L ** 4

    def drho(self, L, eps=1e-6):
        return (self.rho(L * (1 + eps)) - self.rho(L * (1 - eps))) / (2 * eps * L)

    def Om2(self, L):
        return self.A * self.kw / L ** 2 - self.C * self.kt / L ** 4


def mode_N(m, N0, N1):
    """Integrate the incoming vacuum in N = ln L from N0 (far past) down to N1."""
    nu = np.sqrt(3 * m.kw - 1.0)
    L0 = np.exp(N0)
    H0 = -np.sqrt(m.rho(L0) / 3.0)
    # exact pure-winding normalisation |c|^2 = 1/(2 nu sqrt(A/3)); use local H for robustness
    c2 = -1.0 / (2.0 * nu * H0 * L0 ** 3 * np.exp(-2 * N0))
    c = np.sqrt(c2)
    u0 = c * np.exp((-1 + 1j * nu) * N0)
    uN0 = (-1 + 1j * nu) * u0

    def rhs(N, y):
        L = np.exp(N)
        r = m.rho(L)
        H2 = r / 3.0
        dlnH = 0.5 * L * m.drho(L) / r
        u = y[0] + 1j * y[1]
        uN = y[2] + 1j * y[3]
        uNN = -(3.0 + dlnH) * uN - (m.Om2(L) / H2) * u
        return [uN.real, uN.imag, uNN.real, uNN.imag]

    sol = solve_ivp(rhs, (N0, N1), [u0.real, u0.imag, uN0.real, uN0.imag],
                    rtol=1e-10, atol=1e-30, method="DOP853")
    y = sol.y[:, -1]
    u = y[0] + 1j * y[1]
    uN = y[2] + 1j * y[3]
    L1 = np.exp(N1)
    H1 = -np.sqrt(m.rho(L1) / 3.0)
    # Wronskian check in N form: L^3 H (u uN* - u* uN) = i
    W = (L1 ** 3 * H1 * (u * np.conj(uN) - np.conj(u) * uN)).imag
    return u, H1 * uN, W, sol.nfev


def mode_t(m, L_start, ud_start, u_start, t_end):
    """Background + mode in cosmic time from L_start (contracting) through the turn."""
    H_start = -np.sqrt(m.rho(L_start) / 3.0)

    def rhs(t, y):
        L, Ld = y[0], y[1]
        Ldd = L ** 2 * m.drho(L) / 6.0 + L * m.rho(L) / 3.0
        H = Ld / L
        u = y[2] + 1j * y[3]
        ud = y[4] + 1j * y[5]
        udd = -3 * H * ud - m.Om2(L) * u
        return [Ld, Ldd, ud.real, ud.imag, udd.real, udd.imag]

    turn = lambda t, y: y[1]
    turn.terminal = False
    y0 = [L_start, L_start * H_start, u_start.real, u_start.imag, ud_start.real, ud_start.imag]
    sol = solve_ivp(rhs, (0.0, t_end), y0, rtol=1e-11, atol=1e-30, method="DOP853",
                    events=turn, dense_output=True)
    return sol


def natural_out(m, t_out):
    """v11-type natural state at the turn: real Gaussian, Var beta = 1/(2 m_b sqrt(kappa) H)."""
    mb = m.Lb ** 3
    # |Omega| at the turn: instability rate (E) or oscillator frequency (T2); s = 1/(2 m |w|)
    w = np.sqrt(abs(m.Om2(m.Lb)))
    s = 1.0 / (2 * mb * w)
    # two real solutions with (beta, P) = (1, 0) and (0, 1) at the turn
    def rhs(t, y):
        L, Ld = y[0], y[1]
        Ldd = L ** 2 * m.drho(L) / 6.0 + L * m.rho(L) / 3.0
        H = Ld / L
        return [Ld, Ldd, y[3], -3 * H * y[3] - m.Om2(L) * y[2],
                y[5], -3 * H * y[5] - m.Om2(L) * y[4]]
    y0 = [m.Lb, 0.0, 1.0, 0.0, 0.0, 1.0 / mb]
    sol = solve_ivp(rhs, (0.0, t_out), y0, rtol=1e-11, atol=1e-30, method="DOP853")
    b1, b2 = sol.y[2, -1], sol.y[4, -1]          # beta_+ for unit beta_0 and unit P_0
    var_out = b1 ** 2 * s + b2 ** 2 / (4 * s)
    floor = abs(b1 * b2)                          # min over s of b1^2 s + b2^2/(4s) = |b1 b2|
    return dict(s_nat=s, VarP_nat=1.0 / (4 * s), var_out_nat=var_out, floor_out=floor,
                w_turn=w, m11=b1, m12_over_mb=b2 * mb)


def run_case(kappa, N_pre, a=1e-2, k_w=2.0, C=30.0):
    m = Model(kappa, C=C, a=a, k_w=k_w, N_pre=N_pre)
    # far past: deep in the winding era, L_w = V_b Lx^3 / A; start 4 e-folds beyond it
    Lw = m.Vb * m.Lx ** 3 / m.A
    N0 = np.log(Lw) + 8.0
    N1 = np.log(3.0 * m.Lb)
    u, ud, W, nfev = mode_N(m, N0, N1)
    # cosmic-time phase through the turn and out to H t = 6 after it
    t_guess = 8.0
    sol = mode_t(m, np.exp(N1), ud, u, t_guess + 6.0)
    tt = sol.t_events[0]
    t_turn = float(tt[0]) if len(tt) else float("nan")
    yT = sol.sol(t_turn)
    L_T = yT[0]
    uT = yT[2] + 1j * yT[3]
    udT = yT[4] + 1j * yT[5]
    varb_T = abs(uT) ** 2
    varP_T = (L_T ** 3) ** 2 * abs(udT) ** 2
    # outgoing frozen variance at H t = 6 after the turn
    yO = sol.sol(t_turn + 6.0)
    varb_out = abs(yO[2] + 1j * yO[3]) ** 2
    nat = natural_out(m, 6.0)
    S_turn = 5 * varP_T / L_T ** 6          # five modes, same stiffness (upper estimate)
    return dict(kappa=kappa, N_pre=N_pre, a=a, k_w=k_w, C=C, Lb=m.Lb, L_turn_num=L_T,
                t_turn=t_turn, N_total=float(N0 - N1), wronskian_N=float(W),
                VarBeta_turn=varb_T, VarP_turn=varP_T,
                R_P=varP_T / nat["VarP_nat"], R_beta=varb_T / nat["s_nat"],
                shear_ratio=S_turn / (4 * m.Vb),
                var_out=varb_out, F_out=varb_out / nat["var_out_nat"],
                F_out_floor=varb_out / nat["floor_out"], nat=nat, nfev_N=int(nfev))


def frozen_times():
    rows = []
    for name, k in [("E_xi0.30", 19.32), ("T2_xi0.30", 9.07), ("E_free", 13.53), ("T2_free", 6.35)]:
        ht = 0.5 * np.arcsinh(np.sqrt(k))
        efold = 0.5 * np.log(np.cosh(2 * ht))
        rows.append(dict(mode=name, abs_kappa=k, h_t_freeze=ht, efolds_from_turn=efold))
    return rows


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = dict(code=os.path.basename(__file__), code_sha256=sha(__file__),
               frozen=frozen_times(), cases=[])
    for kappa in (19.32, -9.07):
        for N_pre in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            r = run_case(kappa, N_pre)
            res["cases"].append(r)
            print(f"kappa={kappa:6.2f} N_pre={N_pre:4.1f} R_P={r['R_P']:.3e} "
                  f"shear={r['shear_ratio']:.3e} F_out={r['F_out']:.3e} W={r['wronskian_N']:.6f}",
                  flush=True)
    # sensitivity: winding strength and stiffness
    for a, kw in ((1e-3, 2.0), (1e-1, 2.0), (1e-2, 1.0), (1e-2, 5.0)):
        r = run_case(19.32, 3.0, a=a, k_w=kw)
        res["cases"].append(r)
        print(f"sens a={a} kw={kw}: R_P={r['R_P']:.3e} shear={r['shear_ratio']:.3e}", flush=True)
    res["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, "C137_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print("done", res["runtime_s"])
