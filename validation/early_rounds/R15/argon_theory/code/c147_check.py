# -*- coding: utf-8 -*-
"""R15 r15-argon-theory: independent check of C147/D119 (width rule B3, Theta-floor on sigma_N).
Own minisuperspace integration (flat T^3, lapse 1, M_P=1 reduced):
  H^2 = (phidot^2/2 + V - D/L^4)/3,  Hdot = -phidot^2/2 + (2/3) D/L^4,  phiddot = -3 H phidot - V'.
Inputs taken from D119 header (model inputs, not results): h, D, phi_b(ref).
Time unit 1/h. P = L^3 phidot (canonical, unit comoving volume), m = L_b^3, hbar = 1.
"""
import json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import beta as Bfun
from numpy.polynomial.hermite_e import hermegauss

t0 = time.time()
OUT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R15_v12_round2\argon_theory\code\c147_results.json"
h = 6.1159966e-6; D = 29.92; kap = np.sqrt(2.0 / 3.0); phib_ref = 5.579356768512762
Dt = D / h**2
def v(p):   e = np.exp(-kap * p); return 3 * (1 - e)**2              # V/h^2
def vp(p):  e = np.exp(-kap * p); return 6 * kap * e * (1 - e)
def vpp(p): e = np.exp(-kap * p); return 6 * kap**2 * e * (2 * e - 1)

def Lb_of(phi, dphi):   # turning point p_L=0: L^4 (V + phidot^2/2) = D
    return (Dt / (v(phi) + 0.5 * dphi**2))**0.25

def rhs(t, y):
    lnL, H, p, dp = y
    return [H, -0.5 * dp**2 + (2.0 / 3.0) * Dt * np.exp(-4 * lnL), dp, -3 * H * dp - vp(p)]

def ev_end(t, y):   # epsilon_H = 1  <=>  -Hdot - H^2 = 0 (upward)
    lnL, H, p, dp = y
    return -(-0.5 * dp**2 + (2.0 / 3.0) * Dt * np.exp(-4 * lnL)) - H**2
ev_end.terminal = True; ev_end.direction = 1

def Npost(phi_b, P_b=0.0, full=False):
    dphi = 0.0
    for _ in range(60):
        Lb = Lb_of(phi_b, dphi)
        new = P_b / (Lb**3 * h)
        if abs(new - dphi) < 1e-20: break
        dphi = new
    Lb = Lb_of(phi_b, dphi)
    sol = solve_ivp(rhs, [0, 5000], [np.log(Lb), 0.0, phi_b, dphi], method="DOP853",
                    rtol=1e-12, atol=1e-13, events=ev_end)
    ye = sol.y_events[0][0]
    N = ye[0] - np.log(Lb)
    if full:
        lnL, H, p, dp = ye
        cres = H**2 - (0.5 * dp**2 + v(p) - Dt * np.exp(-4 * lnL)) / 3
        return N, Lb, sol.t_events[0][0], cres, p
    return N, Lb

res = {}
N0, Lb0, tend, cres, phi_end = Npost(phib_ref, full=True)
res["ref"] = {"phi_b": phib_ref, "N_post_to_eps1": N0, "L_b": Lb0, "v_b=V_b/h^2": v(phib_ref),
              "hL_b": h * Lb0, "tau_end_h": tend, "constraint_residual_end": cres, "phi_end": phi_end}
m = Lb0**3
dphi_fd = 1e-4; dP = 1.0
A = (Npost(phib_ref + dphi_fd)[0] - Npost(phib_ref - dphi_fd)[0]) / (2 * dphi_fd)
B = (Npost(phib_ref, +dP)[0] - Npost(phib_ref, -dP)[0]) / (2 * dP)
B2 = (Npost(phib_ref, +0.3)[0] - Npost(phib_ref, -0.3)[0]) / 0.6
res["linear_response"] = {"A=dN/dphi_b": A, "B=dN/dP_b": B, "B_step0.3": B2, "m=L_b^3": m}
# slow-roll and free-flight estimates
x = kap * phib_ref
A_sr = 0.75 * kap * (np.exp(x) - 1)
hV = h * np.sqrt(v(phib_ref) / 3)
tauhV = Bfun(0.75, 0.5) / 4          # int_0^inf sech^{3/2}(u) du / 2 ; L = L_b cosh(2 h_V t)^{1/2}
tau_ff = tauhV / hV
res["analytic"] = {"A_slowroll": A_sr, "tau*h_V(free flight)": tauhV, "tau*h": tau_ff * h,
                   "B_freeflight=A tau/m": A * tau_ff / m, "tau_from_B=mB/A (x h)": m * B / A * h}
floor = np.sqrt(abs(A * B)); s_opt = np.sqrt(abs(B / (2 * A)))
# B3 width
Vp = h**2 * vp(phib_ref); Vpp = h**2 * vpp(phib_ref)
s_B3 = (4 * m**2 * Vp)**(-1.0 / 3.0)
F = m * Vp; k = m * abs(Vpp)
coef = [16.0, 0.0, 0.0, -4 * F**2 * m**2, -k**2 * m**2]
roots = np.roots(coef); a_q = max(r.real for r in roots if abs(r.imag) < 1e-9 * abs(r) and r.real > 0)
s_B3q = (1 / (4 * a_q))**0.5
s_C = (2 * m * np.sqrt(abs(Vpp)))**-0.5
def VarH(s):
    a = 1 / (4 * s**2)
    # Cov_sym(P^2, x^2) = -1/2 for a real Gaussian  =>  constant term -V''/4 (sigma-independent)
    return a**2 / (2 * m**2) + m**2 * Vp**2 / (4 * a) + m**2 * Vpp**2 / (32 * a**2) - Vpp / 4
def sigN_lin(s): return np.sqrt(A**2 * s**2 + B**2 / (4 * s**2))
res["widths"] = {"sigma_B3_slope_only=(4m^2V')^-1/3": s_B3, "sigma_B3_quartic": s_B3q,
                 "sigma_Codex=(2m sqrt|V''|)^-1/2": s_C, "slope/curv ratio 2mV'^2/|V''|^1.5": 2 * m * Vp**2 / abs(Vpp)**1.5,
                 "sigma_opt_floor": s_opt, "VarH(B3)": VarH(s_B3q), "VarH(Codex)": VarH(s_C),
                 "sigma_P(B3)": 1 / (2 * s_B3q)}
res["sigma_N_linear"] = {"floor_sqrt|AB|": floor, "B3": sigN_lin(s_B3q), "Codex": sigN_lin(s_C),
                         "B3/floor": sigN_lin(s_B3q) / floor,
                         "floor/N_post": floor / N0, "delta_v11=sqrt2/(3 h m)": np.sqrt(2) / (3 * h * m)}
# nonlinear Wigner push-forward with Gauss-Hermite (real Gaussian => W = product Gaussian)
def sigN_GH(s, n=7):
    z, w = hermegauss(n); w = w / w.sum(); sP = 1 / (2 * s)
    vals = np.array([[Npost(phib_ref + s * zi, sP * zj)[0] for zj in z] for zi in z])
    W = np.outer(w, w); mu = (W * vals).sum(); var = (W * (vals - mu)**2).sum()
    sk = (W * (vals - mu)**3).sum() / var**1.5
    return np.sqrt(var), sk
gB3 = sigN_GH(s_B3q); gopt = sigN_GH(s_opt)
res["sigma_N_nonlinear_GH7x7"] = {"B3": gB3[0], "B3_skew": gB3[1], "opt": gopt[0], "opt_skew": gopt[1]}
# one-parameter family and p(N) ~ 1/N check
fam = []
for pb in (4.9, 5.3, 5.5, phib_ref, 5.7, 5.9, 6.3):
    Nn, Lb = Npost(pb)
    Ad = (Npost(pb + 1e-4)[0] - Npost(pb - 1e-4)[0]) / 2e-4
    fam.append({"phi_b": pb, "N_post": Nn, "L_b": Lb, "dN/dphi": Ad, "N/(dN/dphi)": Nn / Ad})
res["family"] = fam
# misc quick checks of header numbers
Vb = h**2 * v(phib_ref)
res["misc"] = {"ell_L=(48D/L_b)^-1/3": (48 * D / Lb0)**(-1 / 3), "S_E=sqrt12*D^(3/4)V^(-1/4)*I": np.sqrt(12) * 0.8740191847640 * D**0.75 * Vb**-0.25,
               "24pi^2/V": 24 * np.pi**2 / Vb}
res["runtime_s"] = time.time() - t0
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
