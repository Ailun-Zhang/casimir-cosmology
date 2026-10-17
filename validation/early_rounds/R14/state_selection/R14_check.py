# -*- coding: utf-8 -*-
"""
R14 independent check of D109 (state selection), DNEC v12.  Written without opening C137*.

Primary method: closed-form solution of the v11 shape-mode equation
    beta'' + 3H beta' - kappa h^2 sech^2(2ht) beta = 0,  H = h tanh 2ht,  L = L_b sqrt(cosh 2ht)
in z = 1/(1+e^{4ht})  (z: 1 -> 0 as t: -inf -> +inf).  It becomes Gauss' equation
    z(1-z) beta_zz + (1/4 - z/2) beta_z - (kappa/4) beta = 0 ,
basis w1 = 2F1(a,b;1/4;z), w2 = z^{3/4} 2F1(a+3/4,b+3/4;7/4;z), a+b=-1/2, ab=kappa/4.
P = M_P^2 L^3 dbeta/dt = -q beta_z, q = (4z(1-z))^{1/4};  q*W[w1,w2] = 3 sqrt2/4 = D.
Exact transfer (beta,P)(t_x) -> (beta_+, P_+):  M = [[q w2'/D, w2/D], [q w1', w1]].
Units h = M_P = L_b = 1, [beta,P] = i.
Cross-checks: ODE in u = tanh(2t) (transfer to the turn) and ODE in t (DOP853, to t=14).
"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import time
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

OUT = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 40
Q14 = mp.mpf(1) / 4
Q34 = mp.mpf(3) / 4


def Dconst():
    return 3 * mp.sqrt(2) / 4


def gain_closed(k):
    k = mp.mpf(k)
    if 4 * k > 1:
        return mp.sqrt(2) * mp.cosh(mp.pi * mp.sqrt(4 * k - 1) / 4)
    return mp.sqrt(2) * mp.cos(mp.pi * mp.sqrt(1 - 4 * k) / 4)


class Mode:
    def __init__(self, kstr):
        self.k = mp.mpf(kstr)
        r = mp.sqrt(mp.mpc(Q14 - self.k))
        self.a = -Q14 + r / 2
        self.b = -Q14 - r / 2
        self.maximag = 0.0
        self.maxdeterr = 0.0

    def M(self, t):
        t = mp.mpf(t)
        z = 1 / (1 + mp.exp(4 * t))
        omz = 1 / (1 + mp.exp(-4 * t))
        a, b = self.a, self.b
        c, a2, b2, c2 = Q14, a + Q34, b + Q34, mp.mpf(7) / 4
        F1 = mp.hyp2f1(a, b, c, z)
        dF1 = a * b / c * mp.hyp2f1(a + 1, b + 1, c + 1, z)
        F2 = mp.hyp2f1(a2, b2, c2, z)
        dF2 = a2 * b2 / c2 * mp.hyp2f1(a2 + 1, b2 + 1, c2 + 1, z)
        w2 = z ** Q34 * F2
        dw2 = Q34 * z ** (-Q14) * F2 + z ** Q34 * dF2
        q = (4 * z * omz) ** Q14
        D = Dconst()
        vals = [q * dw2 / D, w2 / D, q * dF1, F1]
        im = max(float(abs(mp.im(v)) / (1 + abs(v))) for v in vals)
        self.maximag = max(self.maximag, im)
        m = [mp.re(v) for v in vals]
        Mx = mp.matrix([[m[0], m[1]], [m[2], m[3]]])
        det = m[0] * m[3] - m[1] * m[2]
        self.maxdeterr = max(self.maxdeterr, float(abs(det - 1)))
        return Mx


def inv2(M):
    return mp.matrix([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])


def mm(A, B):
    return A * B


def f(x, n=8):
    return float(mp.nstr(x, n)) if not isinstance(x, float) else x


# ---------------- ODE cross-checks (independent of the hypergeometric algebra) -------------
def ode_t_transfer(k, tx, t_end=14.0):
    k = float(k)

    def rhs(t, y):
        ch = np.cosh(2 * t)
        return [y[1] * ch ** -1.5, k * ch ** -0.5 * y[0]]

    cols = []
    for y0 in ([1.0, 0.0], [0.0, 1.0]):
        s = solve_ivp(rhs, (float(tx), t_end), y0, method="DOP853", rtol=1e-12, atol=1e-13)
        cols.append(s.y[:, -1])
    return np.array([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]])


def ode_u_turn(k, tx):
    k = float(k)
    ux = np.tanh(2 * float(tx))
    omu2 = 1.0 / np.cosh(2 * float(tx)) ** 2

    def rhs(u, y):
        return [y[1], (0.5 * u * y[1] + 0.25 * k * y[0]) / ((1 - u) * (1 + u))]

    cols = []
    for (b0, P0) in ((1.0, 0.0), (0.0, 1.0)):
        bu0 = P0 / (2 * omu2 ** 0.25)
        s = solve_ivp(rhs, (ux, 0.0), [b0, bu0], method="DOP853", rtol=1e-12, atol=1e-14)
        bT, buT = s.y[:, -1]
        cols.append((bT, 2 * buT))
    return np.array([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]])


def tx_of_N(N):
    return -mp.acosh(mp.e ** (2 * mp.mpf(N))) / 2


# ---------------------------------------------------------------------------------------------
t0 = time.time()
R = {"method": "hypergeometric closed form in z=1/(1+e^{4ht}); ODE cross-checks in u=tanh2ht and t",
     "units": "h=M_P=L_b=1, [beta,P]=i, P=L^3 dbeta/dt"}

kE, kT = "19.32", "-9.07"
kE0, kT0 = "13.53", "-6.35"

# ---- claim 1: freezing
frz = {}
for ks in (kE, kT, kE0, kT0):
    k = abs(mp.mpf(ks))
    tf = mp.asinh(mp.sqrt(k)) / 2
    frz[ks] = {"h_t_freeze": f(tf, 6), "efolds_from_turn": f(mp.log(1 + k) / 4, 5),
               "omega_over_H_check_at_t=0.7": f(mp.sqrt(k) / mp.cosh(1.4) / mp.tanh(1.4), 8),
               "sqrtk_over_sinh_at_t=0.7": f(mp.sqrt(k) / mp.sinh(1.4), 8),
               "total_phase_sqrtk_pi_over_2": f(mp.sqrt(k) * mp.pi / 2, 5)}
R["claim1_freezing"] = frz


def mode_block(ks, tlist):
    md = Mode(ks)
    M0 = md.M(0)
    m11, m12, m21, m22 = M0[0, 0], M0[0, 1], M0[1, 0], M0[1, 1]
    floor = abs(m11 * m12)
    k = abs(md.k)
    sn = 1 / (2 * mp.sqrt(k))                     # natural width at the turn (v_b = 3)
    x = sn / (abs(m12 / m11) / 2)
    nat_over_floor = (x + 1 / x) / 2
    sn294 = 1 / (2 * mp.sqrt(k * mp.mpf("2.94") / 3))
    x294 = sn294 / (abs(m12 / m11) / 2)
    out = {"m11": f(m11), "m12": f(m12), "m21": f(m21), "m22": f(m22), "floor_var=|m11m12|": f(floor),
           "natural_over_floor_var(vb=3)": f(nat_over_floor, 8), "x_nat(vb=3)": f(x, 6),
           "natural_over_floor_var(vb=2.94 width only)": f((x294 + 1 / x294) / 2, 8)}
    # full transfer: far past limit (t=-40, high precision)
    with mp.workdps(90):
        mdf = Mode(ks)
        Mf = mdf.M(-40)
        T11, T12 = Mf[0, 0], Mf[0, 1]
    out["T11_numeric(t=-40)"] = f(T11, 10)
    out["T11_closed_gain"] = f(gain_closed(ks), 10)
    out["T12_numeric"] = f(T12, 10)
    out["T12_from_2m11m12"] = f(2 * m11 * m12, 10)
    out["T11_from_m"] = f(m11 * m22 + m12 * m21, 10)
    a, b = md.a, md.b
    T12g = mp.gamma(mp.mpf(7) / 4) * mp.gamma(Q34) / (mp.gamma(1 - a) * mp.gamma(1 - b)) / Dconst()
    out["T12_gamma_closed"] = f(mp.re(T12g), 10)
    out["G_inf_numeric"] = f(abs(T11 * T12) / floor, 10)
    out["G_inf_closed=2|gain|"] = f(2 * abs(gain_closed(ks)), 10)
    # G(t): real state at t_x=-t; bound on future branch / floor ; and future-branch bound for t_x=+t
    G = {}
    for t in tlist:
        Mx = md.M(-mp.mpf(t))
        Mp = md.M(mp.mpf(t))
        sx = 1 / (2 * mp.sqrt(k) * mp.cosh(2 * mp.mpf(t)) ** mp.mpf(0.5))   # natural width of that moment
        Vnat_x = Mx[0, 0] ** 2 * sx + Mx[0, 1] ** 2 / (4 * sx)
        Vnat_turn = m11 ** 2 * sn + m12 ** 2 / (4 * sn)
        G[str(t)] = {"G_past(t_x=-t)": f(abs(Mx[0, 0] * Mx[0, 1]) / floor, 8),
                     "rms_sqrtG": f(mp.sqrt(abs(Mx[0, 0] * Mx[0, 1]) / floor), 6),
                     "future_branch_bound_if_real_at_+t (/floor)": f(abs(Mp[0, 0] * Mp[0, 1]) / floor, 6),
                     "natural_width_at_t_x: Var+/floor": f(Vnat_x / floor, 6),
                     "natural_width_at_t_x: rms vs natural-at-turn": f(mp.sqrt(Vnat_x / Vnat_turn), 6)}
    out["G_table"] = G
    out["max_rel_imag"] = md.maximag
    out["max_det_err"] = md.maxdeterr
    return md, M0, out


tl = [0.1, 0.25, 0.5, 1, 1.0932, 2, 4, 8, 16]
mdE, M0E, outE = mode_block(kE, tl)
mdT, M0T, outT = mode_block(kT, tl)
mdE0, M0E0, outE0 = mode_block(kE0, [0.1, 0.25, 0.5])
mdT0, M0T0, outT0 = mode_block(kT0, [0.1])
R["mode_E_19.32"] = outE
R["mode_T2_-9.07"] = outT
R["mode_E_13.53_free"] = outE0
R["mode_T2_-6.35_free"] = outT0

# monotonicity scan of G_E and minimum of G_T2
with mp.workdps(30):
    mdE30, mdT30 = Mode(kE), Mode(kT)
    M0E30, M0T30 = mdE30.M(0), mdT30.M(0)
    fE = abs(M0E30[0, 0] * M0E30[0, 1])
    fT = abs(M0T30[0, 0] * M0T30[0, 1])
    ts = [i * 0.05 for i in range(0, 161)]
    gE = [float(abs(mdE30.M(-t)[0, 0] * mdE30.M(-t)[0, 1]) / fE) for t in ts]
    gT = []
    for t in ts:
        Mx = mdT30.M(-t)
        gT.append(float(abs(Mx[0, 0] * Mx[0, 1]) / fT))
mono = all(gE[i + 1] >= gE[i] for i in range(len(gE) - 1))
iT = int(np.argmin(gT))
first_below1 = next((ts[i] for i in range(1, len(ts)) if gT[i] < 1), None)
R["scan"] = {"G_E_monotone_on_[0,8]": mono, "G_E_at_0.05_grid_first10": gE[:10],
             "G_T2_min": gT[iT], "G_T2_argmin_t": ts[iT], "G_T2_first_t_below_1": first_below1,
             "G_T2_at_8": gT[-1]}

# ---- claim 4: best case (adiabatic Gaussian, omega_x = c|H|, set at L_x = e^{N_pre})
K0 = mp.mpf(722) ** 2 * mp.mpf("0.00437243")          # M_P^2 L_b^3 h
K0alt = mp.sqrt(2) / (3 * mp.mpf("2.04e-4"))           # from v11 delta = 2.04e-4
best = {}
for N in (0.25, 0.5, 1, 1.5, 2, 3, 4):
    tx = tx_of_N(N)
    mx = mp.e ** (3 * mp.mpf(N))
    Hx = mp.tanh(2 * abs(tx))
    row = {"t_x": f(tx, 6)}
    Rs = {}
    for name, md, M0 in (("E", mdE, M0E), ("T2", mdT, M0T)):
        k = abs(md.k)
        Mx = md.M(tx)
        Mt = inv2(M0) * Mx
        m11, m12 = M0[0, 0], M0[0, 1]
        sn = 1 / (2 * mp.sqrt(k))
        Vout_nat = m11 ** 2 * sn + m12 ** 2 / (4 * sn)
        VPnat = mp.sqrt(k) / 2
        for c in (1, 3):
            om = c * Hx
            vb, vp = 1 / (2 * mx * om), mx * om / 2
            VP = Mt[1, 0] ** 2 * vb + Mt[1, 1] ** 2 * vp
            Vout = Mx[0, 0] ** 2 * vb + Mx[0, 1] ** 2 * vp
            Rs[(name, c)] = VP / VPnat
            row[f"{name}_c{c}_R_P"] = f(VP / VPnat, 6)
            row[f"{name}_c{c}_out_rms_over_nat"] = f(mp.sqrt(Vout / Vout_nat), 6)
        Rmin = abs(Mt[1, 0] * Mt[1, 1]) / VPnat
        Rs[(name, "min")] = Rmin
        row[f"{name}_Rmin(optimal real width at t_x)"] = f(Rmin, 6)
        row[f"{name}_cstar"] = f(abs(Mt[1, 0] / Mt[1, 1]) / (mx * Hx), 6)
        row[f"{name}_G(|t_x|)"] = f(abs(Mx[0, 0] * Mx[0, 1]) / abs(m11 * m12), 6)
    for c in (1, 3, "min"):
        num = 2 * Rs[("E", c)] * mp.sqrt(abs(mdE.k)) + 3 * Rs[("T2", c)] * mp.sqrt(abs(mdT.k))
        row[f"shear_over_criterion_c{c}"] = f(num / (24 * K0), 5)
        row[f"shear_over_criterion_c{c}_K0alt"] = f(num / (24 * K0alt), 5)
        row[f"shear_E_only_c{c}"] = f(2 * Rs[("E", c)] * mp.sqrt(abs(mdE.k)) / (24 * K0), 5)
    best[str(N)] = row
R["claim4_best_case"] = best
R["K0=M_P^2L_b^3h"] = f(K0, 8)
R["K0alt_from_delta2.04e-4"] = f(K0alt, 8)

# ---- ODE cross-checks
xc = {}
for t in (0, -0.5, -1, -2, -4):
    Mh = mdE.M(t)
    Mo = ode_t_transfer(kE, t)
    xc[f"E_Minf_t={t}"] = {"hyper": [f(Mh[0, 0]), f(Mh[0, 1])], "odeT": [float(Mo[0, 0]), float(Mo[0, 1])]}
for N in (0.5, 2):
    tx = tx_of_N(N)
    for name, md, M0, ks in (("E", mdE, M0E, kE), ("T2", mdT, M0T, kT)):
        Mt = inv2(M0) * md.M(tx)
        Mu = ode_u_turn(ks, float(tx))
        xc[f"{name}_Mturn_N={N}"] = {"hyper": [f(Mt[i, j]) for i in range(2) for j in range(2)],
                                     "odeU": [float(Mu[i, j]) for i in range(2) for j in range(2)]}
R["crosschecks"] = xc
R["runtime_s"] = time.time() - t0

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
