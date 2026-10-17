# -*- coding: utf-8 -*-
"""R15 r15-argon-theory: independent check of C146/D118.
Independent of producer code. Only uses the conventions stated in D118 Sec.0-1:
  T(lam;G) = [sum_k/A - int] log(k^2+lam) = -(1/pi) sum_{m!=0} c_m (sqrt(lam)/|m|_G) K1(sqrt(lam)|m|_G),
  aspect G=diag(e^{2t},e^{-2t}), shear G=exp(2 t sigma_x), det G = 1 (fixed area), L=1.
Route used here (different from producer's Chowla-Selberg row split):
  (sqrt(lam)/r) K1(sqrt(lam) r) = (1/4) int_0^inf ds s^-2 e^{-lam s} e^{-r^2/(4s)}
  => h_c(lam) = d_t^2 T = -(1/pi) int_0^inf dx e^{-lam/(4x)} H_c(x),
     H_c(x) = d_t^2 sum_{m!=0} c_m exp(-x |m|_G^2)  (Gaussian lattice-sum Hessian, x = 1/(4s)).
  If H_c(x) has a fixed sign for ALL x>0, h_c(lam) has the opposite fixed sign for ALL lam>=0.
"""
import json, time
import numpy as np
import mpmath as mp
from scipy import integrate, special

t0 = time.time()
OUT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R15_v12_round2\argon_theory\code\c146_results.json"
res = {}
mp.mp.dps = 30
G = mp.catalan
PI = np.pi

# ---------------- 1. lattice constants and closed forms ----------------
def sum_m_inv4(z):          # sum_m 1/(z+m)^4
    w = mp.pi * z
    c, ct = mp.csc(w), mp.cot(w)
    return mp.pi**4 / 3 * (2 * c**2 * ct**2 + c**4)

def sum_m_inv_r4(y):        # sum_m 1/(m^2+y^2)^2
    return mp.pi / (2 * y**3) * mp.coth(mp.pi * y) + mp.pi**2 / (2 * y**2) * mp.csch(mp.pi * y)**2

G4_direct = 2 * mp.zeta(4) + 2 * sum(mp.re(sum_m_inv4(1j * n)) for n in range(1, 40))
G4_closed = mp.gamma(mp.mpf(1) / 4)**8 / (960 * mp.pi**2)
A1_direct = (2 * mp.zeta(4) + 2 * sum(sum_m_inv_r4(n) for n in range(1, 40))
             + mp.pi * (mp.zeta(3) - sum(mp.mpf(n)**-3 for n in range(1, 40))))  # exact tail: S(n)->pi/(2n^3)
A1_closed = 2 * mp.pi**2 / 3 * G
res["G4_i_direct"] = float(G4_direct); res["G4_i_closed"] = float(G4_closed)
res["Z_PP_direct"] = float(A1_direct); res["Z_PP_closed_2pi2G_over3"] = float(A1_closed)

closed = {
    "gamma_PP": -G / 3, "gamma_AA": G / 6,
    "H_asp_PP": -4 * G / 3 - 6 / mp.pi**2 * G4_closed,
    "H_sh_PP": -4 * G / 3 + 6 / mp.pi**2 * G4_closed,
    "H_asp_AA": 2 * G / 3 + 9 / mp.pi**2 * G4_closed,
    "H_sh_AA": 2 * G / 3 - 9 / mp.pi**2 * G4_closed,
}
res["closed_forms"] = {k: float(v) for k, v in closed.items()}
res["P2_coefficient_28G_3_plus_90G_6"] = float(28 * G / 3 + 90 * G / 6)
res["P2_73G_over_3"] = float(mp.mpf(73) / 3 * G)

# ---------------- 2. Gaussian lattice-sum Hessians ----------------
def lattice(M, shifted):
    r = np.arange(-M, M + 1, dtype=float)
    if shifted:
        r = r + 0.5
    a, b = np.meshgrid(r, r, indexing="ij")
    return a.ravel(), b.ravel()

def gauss_H(x, kind, lat):
    """d_t^2 sum c_m exp(-x Q_t(m)) at t=0. lat: 'PP' (Z^2, c=1), 'AA' (Z^2, c=(-1)^{m1+m2}),
    'S' (Z^2+(1/2,1/2), c=1; Poisson dual of AA)."""
    x = np.atleast_1d(np.asarray(x, float))
    M = int(np.ceil(np.sqrt(90.0 / x.min()))) + 2
    if lat == "S":
        p1, p2 = lattice(M, True); w = np.ones_like(p1)
    else:
        p1, p2 = lattice(M, False)
        keep = ~((p1 == 0) & (p2 == 0)); p1, p2 = p1[keep], p2[keep]
        w = np.ones_like(p1) if lat == "PP" else 1.0 - 2.0 * ((np.abs(p1 + p2).astype(int)) % 2)
    r2 = p1**2 + p2**2
    Aq = (p1**2 - p2**2)**2 if kind == "asp" else 4.0 * p1**2 * p2**2   # (Q')^2/4
    out = np.empty_like(x)
    for i0 in range(0, len(x), 256):
        X = x[i0:i0 + 256, None]
        out[i0:i0 + 256] = np.sum(w * (4 * X**2 * Aq - 4 * X * r2) * np.exp(-X * r2), axis=1)
    return out

dual = {"PP": "PP", "AA": "S"}

def H_full(x, kind, tw):
    x = np.atleast_1d(np.asarray(x, float)); out = np.empty_like(x)
    big = x >= PI
    if big.any():
        out[big] = gauss_H(x[big], kind, tw)
    sm = ~big
    if sm.any():
        out[sm] = (PI / x[sm]) * gauss_H(PI**2 / x[sm], kind, dual[tw])
    return out

# Poisson consistency for AA (direct vs dual) at a few x
pc = {}
for kind in ("asp", "sh"):
    for x in (0.7, 1.0, PI, 5.0):
        d = gauss_H(x, kind, "AA")[0]
        q = (PI / x) * gauss_H(PI**2 / x, kind, "S")[0]
        pc[f"{kind}_x{x:.4f}"] = [d, q, abs(d - q) / abs(d)]
res["poisson_check_AA"] = pc

# sign scan on [pi, 400] for the three functions that cover all x>0
ys = np.logspace(np.log10(PI), np.log10(400.0), 6000)
scan = {}
for kind in ("asp", "sh"):
    for lat in ("PP", "AA", "S"):
        Hv = gauss_H(ys, kind, lat)
        # relative to leading-shell magnitude to avoid underflow issues
        sgn = np.sign(Hv[np.abs(Hv) > 1e-300])
        scan[f"{kind}_{lat}"] = {"n_pos": int((sgn > 0).sum()), "n_neg": int((sgn < 0).sum()),
                                 "min": float(Hv.min()), "max": float(Hv.max()),
                                 "at_pi": float(gauss_H(PI, kind, lat)[0])}
res["gauss_hessian_sign_scan_y_in_[pi,400]"] = scan

# ---------------- 3. flat Hessians from the Gaussian route ----------------
def quad(f, a=PI, b=400.0):
    v, e = integrate.quad(f, a, b, limit=500, epsabs=0, epsrel=1e-12, points=[5, 10, 20, 40])
    return v

flat = {}
for tw in ("PP", "AA"):
    for kind in ("asp", "sh"):
        Sdd = quad(lambda x: x * gauss_H(x, kind, tw)[0] + PI**3 * gauss_H(x, kind, dual[tw])[0] / x**2)
        flat[f"H_{kind}_{tw}"] = -Sdd / (2 * PI**2)
        # curvature moment <s^2> for heat-kernel b2 correction at the tip
        num = quad(lambda x: gauss_H(x, kind, tw)[0] / x + gauss_H(x, kind, dual[tw])[0] / PI)
        flat[f"s2mom_{kind}_{tw}"] = num / (16 * Sdd)
res["flat_hessians_gauss_route"] = flat
res["flat_minus_closed"] = {k: flat[k] - res["closed_forms"][k] for k in res["closed_forms"] if k in flat}

# leading curvature correction at the cigar tip: K(s;tip)=(4 pi s)^-1 [1 + b2 s^2 + ...],
# b2 = (box R)/180 + R^2/360 with R = 4/R0^2, box R = -16/R0^4  =>  b2 = -2/(45 R0^4)
pred = {}
for key in ("asp_PP", "sh_PP", "asp_AA", "sh_AA"):
    s2 = flat[f"s2mom_{key}"]
    pred[key] = {f"R0={R0}": -2.0 / 45.0 * s2 / R0**4 for R0 in (1, 2, 4, 16)}
res["tip_relative_deviation_pred_b2"] = pred

# ---------------- 4. spectral kernel h_c(lam) for all lam ----------------
def h_kernel(lam, kind, tw):
    f = lambda x: (np.exp(-lam / (4 * x)) * gauss_H(x, kind, tw)[0]
                   + PI * np.exp(-lam * x / (4 * PI**2)) * gauss_H(x, kind, dual[tw])[0] / x)
    return -quad(f) / PI

kap = np.concatenate([[0.0], np.logspace(-4, np.log10(45.0), 60)])
hk = {}
for tw in ("PP", "AA"):
    for kind in ("asp", "sh"):
        vals = np.array([h_kernel(k**2, kind, tw) for k in kap])
        hk[f"{kind}_{tw}"] = {"h(kappa=0)": float(vals[0]), "min": float(vals.min()), "max": float(vals.max()),
                              "n_pos": int((vals > 0).sum()), "n_neg": int((vals < 0).sum()),
                              "h(kappa=1)": float(h_kernel(1.0, kind, tw)),
                              "h(kappa=45)": float(vals[-1])}
res["spectral_kernel_scan_kappa_0_to_45"] = hk

# Bessel-sum cross-check at kappa = 0.5, 1, 3 (analytic t-derivatives)
def h_bessel(lam, kind, tw, M=60):
    r = np.arange(-M, M + 1, dtype=float)
    a, b = np.meshgrid(r, r, indexing="ij"); a, b = a.ravel(), b.ravel()
    keep = ~((a == 0) & (b == 0)); a, b = a[keep], b[keep]
    w = np.ones_like(a) if tw == "PP" else 1.0 - 2.0 * ((np.abs(a + b).astype(int)) % 2)
    Q = a**2 + b**2; u = np.sqrt(lam * Q)
    Q1 = 2 * (a**2 - b**2) if kind == "asp" else 4 * a * b
    Q2 = 4 * Q
    g1 = -lam**2 * special.kv(2, u) / (2 * u**2)
    g2 = lam**3 * special.kv(3, u) / (4 * u**3)
    return -(1 / PI) * np.sum(w * (g2 * Q1**2 + g1 * Q2))

bx = {}
for k in (0.5, 1.0, 3.0):
    for tw in ("PP", "AA"):
        for kind in ("asp", "sh"):
            hb = h_bessel(k**2, kind, tw); hg = h_kernel(k**2, kind, tw)
            bx[f"{kind}_{tw}_kappa{k}"] = [hb, hg, abs(hb - hg) / abs(hb)]
res["bessel_vs_gauss_route"] = bx

# Kronecker limit: h_PP(0) = d_t^2 log(tau2 |eta(tau)|^4)
def logdet(tau):
    q = mp.exp(2j * mp.pi * tau)
    prod = mp.mpf(1)
    for n in range(1, 80):
        prod *= (1 - q**n)
    t2 = mp.im(tau)
    return mp.log(t2) - mp.pi * t2 / 3 + 4 * mp.log(abs(prod))
f_asp = lambda t: logdet(1j * mp.exp(-2 * t))
f_sh = lambda t: logdet((mp.sinh(2 * t) + 1j) / mp.cosh(2 * t))
res["kronecker_hPP0"] = {"asp": float(mp.diff(f_asp, 0, 2)), "sh": float(mp.diff(f_sh, 0, 2)),
                         "gauss_route_asp": hk["asp_PP"]["h(kappa=0)"], "gauss_route_sh": hk["sh_PP"]["h(kappa=0)"]}

# SM-like flat totals (per unit, bosons PP, fermion dof with loop sign)
c = res["closed_forms"]
res["SM_like_flat"] = {
    "AA_fermions": {"asp": 28 * c["H_asp_PP"] - 90 * c["H_asp_AA"], "sh": 28 * c["H_sh_PP"] - 90 * c["H_sh_AA"]},
    "PP_fermions": {"asp": 28 * c["H_asp_PP"] - 90 * c["H_asp_PP"], "sh": 28 * c["H_sh_PP"] - 90 * c["H_sh_PP"]},
}
res["runtime_s"] = time.time() - t0
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
