#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R15 r15-timesym: independent check of D113 lemma B3 (inverted oscillator, centered Gaussians).

Independent route (not C141's code):
  A. sympy, direct Gaussian-moment integrals of H psi / psi (no pre-derived formula used);
     closed-form identity Var H - k/2m = (4(a^2+b^2) - k m)^2 / (32 a^2 m^2);
     Riccati flow alpha' = -i(2 alpha^2/m + k/2) and explicit orbit a = a0 sech(2|w|t), b = a0 tanh(2|w|t).
  B. split-step Fourier evolution of the natural state (generic m, k): Var H conservation,
     (a,b) extracted from <x^2>, Re<xp>, Gaussianity overlap, time-reversal psi(-t) = psi(t)*,
     autocorrelation C(t) = (cosh |w| t)^(-1/2).
  C. global minimisation of Var H over (a>0, b) from random starts: flat direction = half circle.
  D. energy distribution of the natural state P0(E) = |Gamma(1/4 + iE/2|w|)|^2/(2 pi sqrt(2 pi)|w|):
     Var = k/2m by an independent route; spectral filtering f(H) psi0 (f real) gives REAL states with
     Var H << k/2m, i.e. the minimum is a statement inside the centered-Gaussian family only.
"""
import json, math, time
import numpy as np
import sympy as sp
from scipy import optimize, special

OUT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R15_v12_round2\time_symmetry"
res = {"script": "R15_B3_check.py", "agent": "r15-timesym", "date": "2026-10-13"}
T0 = time.time()

# ---------------- Part A: symbolic ----------------
x = sp.symbols('x', real=True)
a, k, m = sp.symbols('a k m', positive=True)
b, t = sp.symbols('b t', real=True)
alpha = a - sp.I * b
psi = sp.exp(-alpha * x**2)


def ratio_H(sgn):
    Hpsi = -sp.diff(psi, x, 2) / (2 * m) + sgn * k * x**2 / 2 * psi
    return sp.expand(sp.simplify(Hpsi / psi))


def avg(poly):
    P = sp.Poly(sp.expand(poly), x)
    den = sp.integrate(sp.exp(-2 * a * x**2), (x, -sp.oo, sp.oo))
    tot = 0
    for (n,), c in P.terms():
        tot += c * sp.integrate(x**n * sp.exp(-2 * a * x**2), (x, -sp.oo, sp.oo))
    return sp.simplify(tot / den)


sym = {}
store = {}
for name, sgn in (("inverted", -1), ("normal", +1)):
    r = ratio_H(sgn)
    EH = sp.simplify(avg(r))
    EH2 = avg(sp.expand(r * sp.conjugate(r)))
    var = sp.factor(sp.simplify(EH2 - EH**2))
    sym[name] = {"H_psi_over_psi": str(r), "<H>": str(EH), "VarH": str(var)}
    store[name] = (r, EH, var)

r_inv, EH_inv, var_inv = store["inverted"]
var_nor = store["normal"][2]
D113 = (16*a**4 + 32*a**2*b**2 + 8*a**2*k*m + 16*b**4 - 8*b**2*k*m + k**2*m**2) / (32*a**2*m**2)
checks = {}
checks["VarH_inv_minus_D113_formula"] = str(sp.simplify(var_inv - D113))
checks["identity_inv: VarH - k/2m - (4(a^2+b^2)-km)^2/(32a^2m^2)"] = str(
    sp.simplify(var_inv - k / (2*m) - (4*(a**2 + b**2) - k*m)**2 / (32*a**2*m**2)))
checks["identity_nor: VarH - [(4(a^2+b^2)-km)^2+16kmb^2]/(32a^2m^2)"] = str(
    sp.simplify(var_nor - ((4*(a**2 + b**2) - k*m)**2 + 16*k*m*b**2) / (32*a**2*m**2)))
bc = sp.sqrt(k*m/4 - a**2)
checks["<H> on circle a^2+b^2=km/4"] = str(sp.simplify(EH_inv.subs(b, bc)))
checks["VarH on circle minus k/2m"] = str(sp.simplify(var_inv.subs(b, bc) - k/(2*m)))
# b = 0 member
a_nat = sp.sqrt(k*m)/2
checks["<x^2> of b=0 member: 1/(4a) - 1/(2 m |w|)"] = str(sp.simplify(1/(4*a_nat) - 1/(2*m*sp.sqrt(k/m))))
# Riccati: psi = N exp(-alpha x^2), i dpsi/dt = H psi  => -i alpha' = c2 (x^2 coefficient of H psi/psi)
c2 = sp.Poly(r_inv, x).coeff_monomial(x**2)
c0 = sp.Poly(r_inv, x).coeff_monomial(1)
checks["c2 + (2 alpha^2/m + k/2)"] = str(sp.simplify(c2 + (2*alpha**2/m + k/2)))
checks["c0 - alpha/m"] = str(sp.simplify(c0 - alpha/m))
w_s = sp.sqrt(k/m)
a0_s = sp.sqrt(k*m)/2
a_t = a0_s / sp.cosh(2*w_s*t)
b_t = a0_s * sp.tanh(2*w_s*t)
alpha_t = a_t - sp.I*b_t
ric_res = sp.diff(alpha_t, t) - sp.I*c2.subs({a: a_t, b: b_t}, simultaneous=True)
ric_simpl = sp.simplify(sp.expand(ric_res.rewrite(sp.exp)))
checks["Riccati residual for a0 sech(2wt) - i a0 tanh(2wt)"] = str(ric_simpl)
num_pts = []
for kv, mv, tv in ((0.9, 1.7, 0.37), (2.3, 0.4, -1.1), (1.0, 1.0, 2.5)):
    num_pts.append(complex(ric_res.subs({k: kv, m: mv, t: tv}).evalf()))
checks["Riccati residual numeric (3 pts)"] = [abs(z) for z in num_pts]
# angle flow on circle: alpha = a0 exp(-i theta) => theta' = 2|w| cos(theta) > 0 on (-pi/2, pi/2)
th = sp.symbols('theta', real=True)
alpha_th = a0_s * sp.exp(-sp.I*th)
# alpha' = -i(2 alpha^2/m + k/2) ; alpha' = -i a0 theta' e^{-i theta}  => theta' = (2alpha^2/m + k/2)/(a0 e^{-i theta})
theta_dot = sp.simplify(((2*alpha_th**2/m + k/2) / (a0_s*sp.exp(-sp.I*th))).rewrite(sp.cos))
checks["theta_dot on min circle"] = str(sp.simplify(sp.expand_complex(theta_dot)))
res["A_symbolic"] = {"per_hamiltonian": sym, "checks": checks}

# ---------------- Part B: split-step numerics ----------------
mN, kN = 1.7, 0.9
w = math.sqrt(kN / mN)
a0 = math.sqrt(kN * mN) / 2
Lbox, N = 40.0, 8192
xs = np.linspace(-Lbox, Lbox, N, endpoint=False)
dx = xs[1] - xs[0]
ps = 2 * np.pi * np.fft.fftfreq(N, d=dx)
V = -kN * xs**2 / 2


def Happly(f):
    return np.fft.ifft(ps**2 / (2 * mN) * np.fft.fft(f)) + V * f


def observables(f):
    nrm = np.sum(np.abs(f)**2) * dx
    x2 = np.sum(xs**2 * np.abs(f)**2) * dx / nrm
    pf = np.fft.ifft(ps * np.fft.fft(f))
    xp = np.real(np.sum(np.conj(f) * xs * pf) * dx) / nrm
    Hf = Happly(f)
    EH = np.real(np.sum(np.conj(f) * Hf) * dx) / nrm
    EH2 = np.sum(np.abs(Hf)**2) * dx / nrm
    aa = 1 / (4 * x2)
    bb = 2 * aa * xp
    g = np.exp(-(aa - 1j * bb) * xs**2)
    g /= np.sqrt(np.sum(np.abs(g)**2) * dx)
    ov = abs(np.sum(np.conj(g) * f) * dx) / math.sqrt(nrm)
    return dict(norm=float(nrm), x2=float(x2), a=float(aa), b=float(bb), EH=float(EH),
                VarH=float(EH2 - EH**2), gauss_overlap=float(ov))


def evolve(f, dt, nsteps):
    half = np.exp(-1j * V * dt / 2)
    kin = np.exp(-1j * ps**2 / (2 * mN) * dt)
    for _ in range(nsteps):
        f = half * np.fft.ifft(kin * np.fft.fft(half * f))
    return f


psi0 = (2 * a0 / np.pi)**0.25 * np.exp(-a0 * xs**2)
dt = 2.5e-4
tcheck = [0.25, 0.5, 1.0, 1.5, 2.0]
rows = []
fF, fB, tprev = psi0.copy(), psi0.copy(), 0.0
o0 = observables(psi0)
rows.append(dict(t=0.0, **o0, a_th=a0, b_th=0.0))
for tc in tcheck:
    nst = int(round((tc - tprev) / dt))
    fF = evolve(fF, dt, nst)
    fB = evolve(fB, -dt, nst)
    tprev = tc
    for sgn, f in ((+1, fF), (-1, fB)):
        tt = sgn * tc
        o = observables(f)
        o.update(t=tt, a_th=a0 / math.cosh(2 * w * tt), b_th=a0 * math.tanh(2 * w * tt),
                 circle_resid=o["a"]**2 + o["b"]**2 - kN * mN / 4,
                 C_num=complex(np.sum(np.conj(psi0) * f) * dx).__repr__(),
                 C_th=(math.cosh(w * tt))**-0.5)
        rows.append(o)
    rows.append(dict(t=tc, TR_maxdiff_psi_minus_t_vs_conj_psi_t=float(np.max(np.abs(fB - np.conj(fF))))))
res["B_numerics"] = {"m": mN, "k": kN, "w": w, "a0": a0, "k_over_2m": kN / (2 * mN),
                     "x2_natural_expected_1_over_2m|w|": 1 / (2 * mN * w), "grid": [Lbox, N, dt],
                     "rows": rows}
# off-circle Gaussian: Var H conserved but > k/2m, orbit not on the min circle
ga, gb = 0.3, 0.15
g0 = np.exp(-(ga - 1j * gb) * xs**2)
g0 /= np.sqrt(np.sum(np.abs(g0)**2) * dx)
og0 = observables(g0)
og1 = observables(evolve(g0, dt, 4000))
varf = lambda A, B: kN / (2 * mN) + (4 * (A**2 + B**2) - kN * mN)**2 / (32 * A**2 * mN**2)
res["B_offcircle"] = {"a,b": [ga, gb], "VarH_formula": varf(ga, gb), "t0": og0, "t1": og1}

# ---------------- Part C: global minimisation ----------------
rng = np.random.default_rng(20261013)
mins = []
for _ in range(40):
    s0 = [math.log(rng.uniform(0.02, 3.0)), rng.uniform(-2.0, 2.0)]
    rr = optimize.minimize(lambda z: varf(math.exp(z[0]), z[1]), s0, method="Nelder-Mead",
                           options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000))
    A, B = math.exp(rr.x[0]), rr.x[1]
    mins.append([A, B, rr.fun])
mins = np.array(mins)
res["C_minimisation"] = {
    "k_over_2m": kN / (2 * mN),
    "min_found": float(mins[:, 2].min()), "max_of_minima": float(mins[:, 2].max()),
    "max_|a^2+b^2-km/4|": float(np.max(np.abs(mins[:, 0]**2 + mins[:, 1]**2 - kN * mN / 4))),
    "b_range_of_minimisers": [float(mins[:, 1].min()), float(mins[:, 1].max())],
    "bound_|b|<sqrt(km)/2": math.sqrt(kN * mN) / 2,
}

# ---------------- Part D: energy distribution & real filtered states ----------------
def P0(E, ww):
    return np.exp(2 * np.real(special.loggamma(0.25 + 1j * E / (2 * ww)))) / (2 * np.pi * math.sqrt(2 * np.pi) * ww)


E = np.linspace(-80 * w, 80 * w, 400001)
P = P0(E, w)
nrmP = np.trapezoid(P, E)
meanP = np.trapezoid(E * P, E) / nrmP
varP = np.trapezoid(E**2 * P, E) / nrmP - meanP**2
Ccheck = []
for tau in (0.3, 1.0, 2.5):
    Cn = np.trapezoid(P * np.exp(-1j * E * tau), E)
    Ccheck.append([tau, abs(Cn - (math.cosh(w * tau))**-0.5)])
filt = []
for epsr in (0.5, 0.3, 0.1):
    eps = epsr * w
    Pf = P * np.exp(-E**2 / (2 * eps**2))
    nf = np.trapezoid(Pf, E)
    vf = np.trapezoid(E**2 * Pf, E) / nf
    filt.append(dict(eps_over_w=epsr, VarH=float(vf), ratio_to_k_over_2m=float(vf / (kN / (2 * mN))),
                     surviving_weight=float(nf)))
res["D_energy_distribution"] = {"norm": float(nrmP), "mean": float(meanP), "var": float(varP),
                                "k_over_2m": kN / (2 * mN), "C_tau_check_absdiff": Ccheck,
                                "filtered_real_states": filt}
res["runtime_s"] = time.time() - T0

with open(OUT + r"\R15_B3_results.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False, default=str)

# concise console summary
print("== A symbolic ==")
for kk_, vv in checks.items():
    print(f"  {kk_}: {vv}")
print("  VarH(inverted) =", sym["inverted"]["VarH"])
print("== B numerics (m=%.2f k=%.2f |w|=%.5f a0=%.6f k/2m=%.6f) ==" % (mN, kN, w, a0, kN / (2 * mN)))
for r_ in rows:
    if "TR_maxdiff_psi_minus_t_vs_conj_psi_t" in r_:
        print("  t=%+.2f  TR max|psi(-t)-psi(t)*| = %.3e" % (r_["t"], r_["TR_maxdiff_psi_minus_t_vs_conj_psi_t"]))
    else:
        print("  t=%+.2f a=%.8f (th %.8f) b=%+.8f (th %+.8f) VarH=%.10f <H>=%+.2e ov=%.10f circ=%s C=%s Cth=%s" % (
            r_["t"], r_["a"], r_["a_th"], r_["b"], r_["b_th"], r_["VarH"], r_["EH"], r_["gauss_overlap"],
            ("%.2e" % r_["circle_resid"]) if "circle_resid" in r_ else "-", r_.get("C_num", "-"),
            ("%.8f" % r_["C_th"]) if "C_th" in r_ else "-"))
print("  off-circle:", json.dumps(res["B_offcircle"], default=str)[:400])
print("== C minimisation ==", json.dumps(res["C_minimisation"]))
print("== D energy distribution ==", json.dumps(res["D_energy_distribution"]))
print("runtime %.1f s" % res["runtime_s"])
