# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_tier2/C160_b1_wdw.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Minisuperspace WDW test of B1 with one aspect mode (D131 Sec. 4): reality defect D_re of the reconstructed turning-point state, 2.4e-6 at X = 2282.49 (Ai-type, X^-2.4 scaling); R15 counterexample fails as Born-Oppenheimer predicts. Paper: PRD Sec. VIII B (sec:s-equivalent, eq:s-WDW).

C160 part 4 (shape-tier2, D131 Sec. 4): minisuperspace WDW test of B1 ("real Psi <=> real turning-point shape state",
with the single-classical-history condition) with one aspect mode.

Minisuperspace (C60 units; x = ln(L/L_b); beta along X_D2; b = sqrt(X) beta; plateau potential frozen at phi_b, Vhat = 3):
    psi_xx = 6 X psi_bb - U(x,b) psi,   U = 36 X^2 [e^{6x} - zeta(b) e^{2x}],
    zeta = 1 + (kappa/6) beta^2 + (c/6) beta^3   (SM, xi = 0.30: kappa_E = 19.3236, c = c_111 from C160_moyal),
i.e. -p_x^2 + 6 P_beta^2 + 12 X^2 e^{6x} rho = 0 with p = -i d (hbar = 1), X = M_P^2 L_b^3 H_eff.  The equation is real.
Born-Oppenheimer reading: psi ~ K0^{-1/2} [e^{iS} phi_+ + e^{-iS} phi_-], K0 = 6X e^{3x} sqrt(1 - e^{-4x}); phi_- obeys
i d_t phi = H_s phi along the expanding branch (t = t(x) > 0), phi_+ is the contracting branch at -t, with
    H_s = -(1/(2 e^{3x})) d_bb - 3 X e^{-x} (zeta(b) - 1),  x(t) = (1/2) ln cosh 2t.
Turning-point state reconstructed from each branch: chi_exp(0) = U(0,t) phi_-(t), chi_con(0) = U(0,-t) phi_+(t).
Defects: D_sh = 1 - |<chi_con|chi_exp>|/(|chi_con||chi_exp|) (single history), D_re = 1 - |sum chi_exp^2|/|chi_exp|^2
(reality up to a global phase), F_in = fidelity with the declared input shape state.
Data on a slice inside the barrier from the local Airy form: psi = Ai(z) phi_A(b) + Bi(z) phi_B(b), z = (x_t(b) - x)/l(b).
    python -B C160_b1_wdw.py   -> results/.../run01/b1/C160_b1_wdw.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import airy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_tier2/run01/b1")
MOYAL = WP.wfile("r3/shape_tier2/run01/moyal/C160_moyal.json")
MU = 3.0 / 2 ** 1.5
KAP = 13.528581 / (1 - 0.299897)
X_PHYS = 2282.49
A_CAP = 15.0
OUTNAME = "C160_b1_wdw.json"


def c111():
    try:
        d = json.loads(MOYAL.read_text(encoding="utf-8"))
        return float(d["SM"]["xi030"]["c_E3_111"])
    except Exception:
        return 12.50397377273011


class Grid:
    def __init__(self, X, bmax=12.0, nb=384, cubic=True):
        self.X = X
        self.b = np.linspace(-bmax, bmax, nb)
        self.db = self.b[1] - self.b[0]
        n = np.arange(nb)
        dn = n[:, None] - n[None, :]
        with np.errstate(divide="ignore"):
            T = np.where(dn == 0, np.pi ** 2 / 3, 2.0 * (-1.0) ** dn / np.where(dn == 0, 1, dn) ** 2)
        self.D2 = -T / self.db ** 2                      # sinc-DVR d^2/db^2
        beta = self.b / np.sqrt(X)
        self.c = c111() if cubic else 0.0
        raw = KAP / 6 * beta ** 2 + self.c / 6 * beta ** 3
        # Saturation of the shape potential beyond the state support: the exact WDW is an x-initial-value problem whose
        # sub-barrier channels grow like e^{X (zeta-1)^{3/2}}; large-|b| channels (thick barrier, no physical amplitude)
        # would amplify round-off by e^{>300}.  Cap the barrier action at A_CAP: zeta-1 -> eps_c tanh((zeta-1)/eps_c),
        # eps_c = (A_CAP/X)^{2/3}.  The same zeta enters H_s, so the WDW <-> Born-Oppenheimer comparison is exact for
        # this (slightly modified at large |b|) model; b_cap is recorded.
        self.eps_c = (A_CAP / X) ** (2 / 3)
        self.zeta = 1 + self.eps_c * np.tanh(raw / self.eps_c)
        self.b_cap = float(np.sqrt(self.eps_c / (KAP / 6)) * np.sqrt(X))
        self.xt = 0.25 * np.log(self.zeta)
        self.ell = (144 * X * X * self.zeta ** 1.5) ** (-1 / 3)

    def U(self, x):
        return 36 * self.X ** 2 * (np.exp(6 * x) - self.zeta * np.exp(2 * x))

    def Hs_apply(self, t, y):
        x = 0.5 * np.log(np.cosh(2 * t))
        return -(self.D2 @ y) / (2 * np.exp(3 * x)) - 3 * self.X * np.exp(-x) * (self.zeta - 1) * y


def gauss(g, var, gamma=0.0, mean=0.0):
    f = np.exp(-(g.b - mean) ** 2 / (4 * var) + 1j * gamma * g.b ** 2)
    return f / np.sqrt(np.sum(np.abs(f) ** 2) * g.db)


def hermite1(g, var):
    f = g.b * np.exp(-g.b ** 2 / (4 * var))
    return (f / np.sqrt(np.sum(np.abs(f) ** 2) * g.db)).astype(complex)


def wdw(g, phiA, phiB, zs, x_list, sep=False):
    """Slice x_s = -zs*l0 (b = 0 turning point x_t = 0, l0 = (144 X^2)^{-1/3}).  Ai part: local nonseparable Airy
    (z = (x_t(b) - x)/l(b), the decaying solution of each b-channel) unless sep; Bi part always separable (z0), because
    the b-local Bi grows like e^{(2/3) z(b)^{3/2}} where the barrier is thick and is not a meaningful datum there."""
    nb = len(g.b)
    l0 = (144 * g.X ** 2) ** (-1 / 3)
    xs = -zs * l0
    zA, lA = (np.full(nb, zs), np.full(nb, l0)) if sep else ((g.xt - xs) / g.ell, g.ell)
    with np.errstate(over="ignore", invalid="ignore"):
        ai, aip, _, _ = airy(zA)
    psi = ai * phiA
    dpsi = -aip * phiA / lA
    if np.any(phiB != 0):
        _, _, bi, bip = airy(zs)
        psi = psi + bi * phiB
        dpsi = dpsi - bip * phiB / l0
    A6 = 6 * g.X * g.D2

    def rhs(x, y):
        p = y[:nb]
        return np.concatenate((y[nb:], A6 @ p - g.U(x) * p))
    sol = solve_ivp(rhs, (xs, max(x_list)), np.concatenate((psi, dpsi)), method="DOP853", rtol=1e-10, atol=1e-13,
                    t_eval=sorted(x_list))
    return {x: (sol.y[:nb, k], sol.y[nb:, k]) for k, x in enumerate(sol.t)}, int(sol.nfev), xs


def evolve(g, chi, t_from, t_to):
    def rhs(t, y):
        return -1j * g.Hs_apply(t, y)
    s = solve_ivp(rhs, (t_from, t_to), chi.astype(complex), method="DOP853", rtol=1e-10, atol=1e-13)
    return s.y[:, -1]


def metrics(g, psi, dpsi, x, t, chi_in):
    K0 = 6 * g.X * np.exp(3 * x) * np.sqrt(1 - np.exp(-4 * x))
    pm = (psi + 1j * dpsi / K0) * np.sqrt(K0) / 2          # e^{-iS} coefficient (expanding branch at +t)
    pp = (psi - 1j * dpsi / K0) * np.sqrt(K0) / 2          # e^{+iS} coefficient (contracting branch at -t)
    ce = evolve(g, pm, t, 0.0)
    cc = evolve(g, pp, -t, 0.0)
    nrm = lambda f: np.sqrt(np.sum(np.abs(f) ** 2) * g.db)
    ov = lambda f, h: np.sum(np.conj(f) * h) * g.db
    D_sh = 1 - abs(ov(cc, ce)) / (nrm(cc) * nrm(ce))
    D_re = 1 - abs(np.sum(ce * ce) * g.db) / nrm(ce) ** 2
    F_in = abs(ov(chi_in, ce)) / (nrm(chi_in) * nrm(ce)) if chi_in is not None else None
    im_frac = float(nrm(np.imag(psi)) / nrm(psi))
    varb = float(np.sum(np.abs(pm) ** 2 * g.b ** 2) * g.db / nrm(pm) ** 2)
    return {"t": t, "x": x, "D_sh": float(D_sh), "D_re": float(D_re), "F_in": None if F_in is None else float(F_in),
            "branch_norm_ratio": float(nrm(pp) / nrm(pm)), "Im_psi_fraction": im_frac, "var_b_expanding": varb}


def linear_var_b(g, chi, t):
    ch = evolve(g, chi, 0.0, t)
    return float(np.sum(np.abs(ch) ** 2 * g.b ** 2) * g.db / (np.sum(np.abs(ch) ** 2) * g.db))


def main():
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    var0 = MU * 2 ** 1.5 / (6 * np.sqrt(KAP))          # natural Var(b) = X Var(beta) = X mu delta/sqrt(kappa)
    t_list = (0.3, 0.45, 0.6)
    res = {"var_b_natural": var0, "kappa": KAP, "c111": c111(), "runs": []}

    def case(X, name, A, B, chi_in, zs=2.5, cubic=True, nb=384, bmax=12.0, predicted=None, sep=False):
        g = Grid(X, bmax=bmax, nb=nb, cubic=cubic)
        fa, fb = A(g), B(g)
        xl = [0.5 * np.log(np.cosh(2 * t)) for t in t_list]
        sols, nfev, xs = wdw(g, fa, fb, zs, xl, sep=sep)
        cin = chi_in(g) if chi_in else None
        rows = []
        for t, x in zip(t_list, xl):
            key = min(sols, key=lambda k: abs(k - x))
            m = metrics(g, *sols[key], x, t, cin)
            if cin is not None:
                m["var_b_linear_from_input"] = linear_var_b(g, cin, t)
            rows.append(m)
        chk = evolve(g, nat(g), 0.0, max(t_list))
        kk = np.fft.fftfreq(len(g.b), g.db) * 2 * np.pi
        ph = np.abs(np.fft.fft(chk)) ** 2
        grid_check = {"k_edge": float(ph[np.abs(kk) > 0.5 * np.abs(kk).max()].sum() / ph.sum()),
                      "b_edge": float(np.sum(np.abs(chk[np.abs(g.b) > 0.8 * g.b.max()]) ** 2) / np.sum(np.abs(chk) ** 2))}
        rec = {"grid_check_linear_natural_tmax": grid_check, "b_cap": g.b_cap, "eps_c": g.eps_c, "X": X, "case": name, "zs": zs, "sep": sep, "cubic": cubic, "nb": nb, "bmax": bmax, "x_start": xs, "nfev": nfev,
               "predicted_D_re": predicted, "rows": rows}
        res["runs"].append(rec)
        print(json.dumps({"X": X, "case": name, "zs": zs, "nb": nb,
                          "D_re": [round(r["D_re"], 8) for r in rows], "D_sh": [round(r["D_sh"], 8) for r in rows],
                          "F_in": [None if r["F_in"] is None else round(r["F_in"], 8) for r in rows],
                          "Im": [round(r["Im_psi_fraction"], 4) for r in rows], "pred": predicted}), flush=True)
        (OUT / OUTNAME).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    nat = lambda g: gauss(g, var0).astype(complex)
    zero = lambda g: np.zeros(len(g.b), complex)
    h1 = lambda g: hermite1(g, var0)
    if "zs0" in sys.argv:
        # Second set (after run01 v1): Bi data posed AT the b=0 turning point (z_s = 0).  Posing separable Bi data inside
        # the barrier (z_s = 2.5) is ill-conditioned: its O(eta) mismatch with the true nonseparable Bi-type solution
        # converts into the Ai-type solution, which grows ~480x relative by the turning point.
        global OUTNAME
        OUTNAME = "C160_b1_wdw_zs0.json"
        for X in (300.0, 1000.0, X_PHYS):
            case(X, "A_sep_zs0_control", nat, zero, nat, zs=0.0, predicted=0.0, sep=True)
            case(X, "B1_zs0_Ai_chi0_plus_Bi_h1", nat, h1, None, zs=0.0, predicted=1.0, sep=True)
            case(X, "B2_zs0_eps0.3", nat, lambda g: 0.3 * hermite1(g, var0), None, zs=0.0, predicted=2 * 0.09 / 1.09, sep=True)
            case(X, "B3_zs0_Ai_plus_0.7Bi_same_chi0", nat, lambda g: 0.7 * gauss(g, var0).astype(complex), nat, zs=0.0,
                 predicted=0.0, sep=True)
            case(X, "Bonly_zs0_h1", zero, h1, h1, zs=0.0, predicted=0.0, sep=True)
        chic = lambda g: gauss(g, var0, gamma=0.6)
        for X in (300.0, 1000.0):
            case(X, "C_Ai_complex_chi0_gamma0.6", chic, zero, chic, predicted=None)
        res["complex_case_D_re_of_input"] = float(1 - abs(np.sum(chic(Grid(300.0)) ** 2) * Grid(300.0).db))
        res["meta"] = {"runtime_s": time.perf_counter() - t_start, "set": "zs0",
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (OUT / OUTNAME).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        return
    for X in (100.0, 300.0, 1000.0, X_PHYS):
        case(X, "A_Ai_real", nat, zero, nat, predicted=0.0)
    for X in (100.0, 300.0, 1000.0):
        case(X, "A_sep_control", nat, zero, nat, predicted=0.0, sep=True)
        case(X, "B1_Ai_chi0_plus_Bi_h1_eps1", nat, h1, None, predicted=1.0, sep=True)
    X = 300.0
    case(X, "A_Ai_real_zs5", nat, zero, nat, zs=5.0, predicted=0.0)
    case(X, "A_Ai_real_nb512_bmax16", nat, zero, nat, nb=512, bmax=16.0, predicted=0.0)
    case(X, "A_Ai_real_nb512_bmax12", nat, zero, nat, nb=512, bmax=12.0, predicted=0.0)
    case(X, "A_Ai_real_quadratic", nat, zero, nat, cubic=False, predicted=0.0)
    case(X, "B2_eps0.3", nat, lambda g: 0.3 * hermite1(g, var0), None, predicted=2 * 0.09 / 1.09, sep=True)
    case(X, "B3_Ai_plus_0.7Bi_same_chi0", nat, lambda g: 0.7 * gauss(g, var0).astype(complex), nat, predicted=0.0, sep=True)
    case(X, "B4_Bi_other_width_even", nat, lambda g: gauss(g, 4 * var0).astype(complex), None, predicted=None, sep=True)
    gam = 0.6
    chic = lambda g: gauss(g, var0, gamma=gam)
    case(X, "C_Ai_complex_chi0_gamma0.6", chic, zero, chic, predicted=None)
    res["meta"] = {"runtime_s": time.perf_counter() - t_start,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "complex_case_gamma": gam,
                   "complex_case_D_re_exact_input": float(1 - abs(np.sum(chic(Grid(300.0)) ** 2) * Grid(300.0).db))}
    (OUT / "C160_b1_wdw.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("done %.0f s" % res["meta"]["runtime_s"], flush=True)


if __name__ == "__main__":
    main()
