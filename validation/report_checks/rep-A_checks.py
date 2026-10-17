# rep-A checks of the intermediate steps added to report chapters 1-5, 13-17 (2026-10-16).
import sys, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp, quad
mp.mp.dps = 30
rng = np.random.default_rng(1)

def tf_sym(n=3):
    a = rng.normal(size=(n, n)); a = a + a.T
    return a - np.trace(a) / n * np.eye(n)

print('A. Bianchi I invariants: E4 = W^2 - 2 Ric^2 + 2/3 R^2 with W^2, Ricci from the frame formulas')
for _ in range(3):
    H, Hd = rng.normal(), rng.normal()
    s, u = tf_sym(), tf_sym()
    S = np.trace(s @ s); P = s @ s - S / 3 * np.eye(3)
    R00 = -(3 * Hd + 3 * H**2 + S)
    Rij = (Hd + 3 * H**2) * np.eye(3) + u + 3 * H * s
    R = -R00 + np.trace(Rij)
    Ric2 = R00**2 + np.trace(Rij @ Rij)
    W2 = 2 * np.trace((u + H * s + 2 * P) @ (u + H * s + 2 * P))
    E4a = W2 - 2 * Ric2 + 2 / 3 * R**2
    E4b = 24 * H**4 + 24 * H**2 * Hd - (12 * H**2 + 4 * Hd) * S - 8 * H * np.trace(u @ s) + 8 * np.trace(u @ P) + 8 * H * np.trace(s @ s @ s)
    a, c = rng.uniform(0.5, 2), rng.uniform(0.5, 2)
    Q0 = 2 * c * np.trace((u - 2 * (a / c - 1) * P) @ (u - 2 * (a / c - 1) * P)) + 4 * a / 3 * (2 - a / c) * S**2
    Pcal = 4 * (c + 2 * a) * np.trace(u @ s) + 8 * (c - a) * np.trace(s @ s @ s) + (2 * c + 12 * a) * H * S - 24 * a * H**3
    lhs = c * W2 - a * E4b; rhs = Q0 + 4 * a * Hd * (S - 6 * H**2) + H * Pcal
    print('  E4 diff %.2e   general-H square diff %.2e   trP2-S2/6 %.2e   trs4-S2/2 %.2e' % (E4a - E4b, lhs - rhs, np.trace(P @ P) - S**2 / 6, np.trace(s @ s @ s @ s) - S**2 / 2))

print('B. rotating principal axes: extra term in tr[u-2(rho-1)P]^2')
O = np.linalg.qr(rng.normal(size=(3, 3)))[0]; sv = np.array([0.7, -0.2, -0.5]); sd = np.array([0.1, 0.3, -0.4]); Xi = rng.normal(size=(3, 3)); Xi = Xi - Xi.T
D = np.diag(sv); u_rot = O @ (np.diag(sd) + Xi @ D - D @ Xi) @ O.T
Pm = O @ np.diag(sv**2 - (sv**2).sum() / 3) @ O.T; rho = 1.3
full = np.trace((u_rot - 2 * (rho - 1) * Pm) @ (u_rot - 2 * (rho - 1) * Pm))
diag_part = np.sum((sd - 2 * (rho - 1) * (sv**2 - (sv**2).sum() / 3))**2)
extra = 2 * sum((sv[i] - sv[j])**2 * Xi[i, j]**2 for i in range(3) for j in range(i + 1, 3))
print('  full - diag - extra = %.2e' % (full - diag_part - extra))

print('C. ANEC and conformal duration')
I52 = mp.quad(lambda x: mp.sech(x)**mp.mpf(2.5), [-mp.inf, mp.inf]); I12 = mp.quad(lambda x: mp.sech(x)**mp.mpf(0.5), [-mp.inf, mp.inf])
print('  int sech^5/2 =', mp.nstr(I52, 12), ' sqrt(pi)G(5/4)/G(7/4) =', mp.nstr(mp.sqrt(mp.pi) * mp.gamma(1.25) / mp.gamma(1.75), 12))
print('  ANEC coeff 2*int =', mp.nstr(2 * I52, 10), ' closed sqrt2 G(1/4)^2/(3 sqrt pi) =', mp.nstr(mp.sqrt(2) * mp.gamma(0.25)**2 / (3 * mp.sqrt(mp.pi)), 10))
J = mp.quad(lambda x: mp.tanh(x)**2 * mp.sech(x)**mp.mpf(0.5), [-mp.inf, mp.inf])
print('  J/(h w_b) = int tanh^2 sech^1/2 =', mp.nstr(J, 12), ' vs 2*int sech^5/2 =', mp.nstr(2 * I52, 12), ' ratio I12/I52 =', mp.nstr(I12 / I52, 12))
eta = mp.quad(lambda x: 1 / mp.sqrt(mp.cosh(2 * x)), [0, mp.inf])
print('  eta_inf*hLb =', mp.nstr(eta, 10), ' G(1/4)^2/(4 sqrt(2pi)) =', mp.nstr(mp.gamma(0.25)**2 / (4 * mp.sqrt(2 * mp.pi)), 10))
w1 = mp.quad(lambda t: 1 / mp.sqrt(4 * t**3 - 4 * t), [1, 2, mp.inf])
print('  lemniscatic omega1 = int_1^inf dt/sqrt(4t^3-4t) =', mp.nstr(w1, 10), ';  eta_inf/(1.31103/0.00437243) =', 1.31103 / 0.00437243)

print('D. Ewald sums at the cube')
def Z2(theta, R=7):
    th = np.array(theta, float); tot = 0.0; rng_ = range(-R, R + 1)
    for n1 in rng_:
        for n2 in rng_:
            for n3 in rng_:
                n = np.array([n1, n2, n3])
                if n.any():
                    Q = float(n @ n); tot += math.cos(2 * math.pi * th @ n) * (1 + math.pi * Q) * math.exp(-math.pi * Q) / Q**2
                m = n - th; Pm = float(m @ m)
                if Pm > 1e-14:
                    x = math.pi * Pm; G = 2 * math.exp(-x) / math.sqrt(x) - 2 * math.sqrt(math.pi) * math.erfc(math.sqrt(x))
                    tot += math.pi**2 * math.sqrt(x) * G
    tot -= math.pi**2 / 2
    if np.allclose(th, np.round(th)): tot += 2 * math.pi**2
    return tot
zp, za = Z2([0, 0, 0]), Z2([.5, .5, .5])
print('  Z2(1,0) = %.10f  Z2(1,AAA) = %.10f  ratio = %.6f  gamma_C = %.7f' % (zp, za, za / zp, zp / math.pi**2))
print('  N_C AAA = %.4f ; N_C PPP = -30' % (15 - 45 * za / zp))
print('  even/odd split: Z_even = %.4f, Z_odd = %.4f' % ((zp + za) / 2, (zp - za) / 2))

print('E. Gaussian SU(N) bound 9/8 (N^2-1) N^(1/3):', 9 / 8 * 3 * 2**(1 / 3), 9 / 8 * 8 * 3**(1 / 3))

print('F. gains, transfer, floor')
def gain(k): return float(mp.sqrt(2) * mp.cosh(mp.pi / 4 * mp.sqrt(4 * k - 1))) if k > 0.25 else float(mp.sqrt(2) * mp.cos(mp.pi / 4 * mp.sqrt(1 - 4 * k)))
def T12(k):
    mu = mp.sqrt(4 * k - 1) / 4 if k > 0.25 else None
    return float(mp.gamma(1.75) * mp.gamma(0.75) / abs(mp.gamma(1.25 + 1j * mu))**2)
for k in [8 / 5, 8 / 5 / 0.7001, 6.0, 13.5286, 19.3237, 18.0, -6.3524, -9.074]:
    print('  kappa=%8.4f  gain=%.6f' % (k, gain(k)) + ('  T12=%.4f' % T12(k) if k > 0.25 else ''))
c32 = 3 / 2**1.5
def transfer(k, t0, t1):
    def f(t, y):
        L3 = np.cosh(2 * t)**1.5
        return [c32 * y[1] / L3, k / c32 * L3 / np.cosh(2 * t)**2 * y[0]]
    cols = []
    for y0 in ([1, 0], [0, 1]):
        s = solve_ivp(f, [t0, t1], y0, rtol=1e-12, atol=1e-14, method='DOP853'); cols.append(s.y[:, -1])
    return np.array(cols).T
k = 19.3237
M = transfer(k, 0, 14); T = transfer(k, -14, 14); Rm = np.diag([1, -1])
print('  m =', np.round(M, 4).tolist(), ' T =', np.round(T, 4).tolist())
print('  T - M R M^-1 R max =', np.abs(T - M @ Rm @ np.linalg.inv(M) @ Rm).max(), '  2 m11 m12 =', 2 * M[0, 0] * M[0, 1])
K = math.sqrt(2) / 3 * T12(k); X = 2282.49; delta = math.sqrt(2) / (3 * X)
print('  K(19.3237)=%.4f K(19.32)=%.4f sigma_floor=%.5f delta=%.4e' % (K, math.sqrt(2) / 3 * T12(19.32), math.sqrt(K / X), delta))
Xn = X * ((1 - 0.28) / (1 - 0.2999))**0.75; Kn = math.sqrt(2) / 3 * T12(18.0)
print('  next order: X=%.2f K(18)=%.4f sigma=%.5f ratio=%.4f' % (Xn, Kn, math.sqrt(Kn / Xn), math.sqrt(Kn / Xn) / math.sqrt(K / X)))
m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
s_star = abs(m12 / m11) * delta; s_nat = 1 / (2 * X * math.sqrt(k)); x = s_nat / s_star
print('  s_star=%.4e s_nat=%.4e x=%.4f (x+1/x)/2-1=%.5f' % (s_star, s_nat, x, (x + 1 / x) / 2 - 1))
print('  never-real width 2 m12 m22 =', 2 * m12 * m22, ' var=', 2 * m12 * m22 * delta, ' rms=', math.sqrt(2 * m12 * m22 * delta), ' /natural =', 2 * m12 * m22 * delta / s_nat)
print('  G(inf)=2T11=', 2 * T[0, 0], '  T11 T12/(m11 m12)=', T[0, 0] * T[0, 1] / (m11 * m12))
for tx in [0.1, 0.25, 0.5, 1.0]:
    Mx = transfer(k, -tx, 14); print('   G(%.2f) = %.3f' % (tx, Mx[0, 0] * Mx[0, 1] / (m11 * m12)))
h = 1e-4; dl = (0.5 * math.log(T12(k * (1 + h))) - 0.5 * math.log(T12(k * (1 - h)))) / (2 * h)
print('  dln sigma/dln kappa (fixed X) =', dl)

print('G. background numbers')
gC = zp / math.pi**2; NC = 15 - 45 * za / zp; C = NC * gC; Heff = 0.98949 * 6.116e-6
for f0 in [0, 0.2999]:
    Lb = ((1 - f0) * C / (3 * Heff**2))**0.25
    print('  f0=%.4f  C_eff=%.4f  L_b M=%.2f  h L_b=%.5f  Heff L_b=%.6f' % (f0, (1 - f0) * C, Lb, 6.116e-6 * Lb, Heff * Lb))
Lb = 722.508; print('  X(L_b=722.508)=', Lb**3 * 0.00437243 / Lb, ' X closed=', ((1 - 0.2999) * C / 3)**0.75 / math.sqrt(Heff))
print('  mu_b = %.4e  L_b/l_P = %.1f  L_b[m] = %.3e  1/(hLb)=%.1f' % (2.435e18 / 722, 722 * math.sqrt(8 * math.pi), 722 * math.sqrt(8 * math.pi) * 1.616255e-35, 1 / 0.004418))
print('  eps = sqrt(N_C) h/M = %.3e   (hLb/4pi)^2 = %.2e   R L_b^2 = 12 (hLb)^2 = %.2e' % (math.sqrt(NC) * 6.116e-6, (0.0044 / 4 / math.pi)**2, 12 * 0.0044**2))
print('  H_eff/h = 1-exp(-sqrt(2/3) 5.5794) =', 1 - math.exp(-math.sqrt(2 / 3) * 5.5794), '  v_b = 3 (Heff/h)^2 =', 3 * 0.98949**2)
print('  (1-f0)^(1/4)=%.4f  (1-f0)^(-3/4)=%.4f' % (0.7001**0.25, 0.7001**-0.75))

print('H. emptiness, photon pair, misc')
print('  gamma/2pi = %.5f ; *N_eff(0.2999) = %.4f ; free %.4f; f0 for 5: %.4f' % (gC / 2 / math.pi, gC / 2 / math.pi * 0.7001 * NC, gC / 2 / math.pi * NC, 1 - 5 / (gC / 2 / math.pi * NC)))
print('  fermion quanta: %.3f' % (gC / 2 / math.pi * 0.7001 * NC / (math.sqrt(3) / 2)))
print('  C_b free = %.3f ; C_b(0.30) = %.3f ; side ratio e^{sqrt(3/2) 0.4511} = %.4f' % (C - 4 * math.pi, 0.7001 * C - 4 * math.pi, math.exp(math.sqrt(1.5) * 0.4511)))
print('  4pi/C = %.3f ; 4pi/(0.7 C) = %.3f ; 2 sqrt3 pi = %.3f ; /C = %.3f' % (4 * math.pi / C, 4 * math.pi / (0.7001 * C), 2 * math.sqrt(3) * math.pi, 2 * math.sqrt(3) * math.pi / C))
print('  Rayleigh median sqrt(2 ln2) = %.4f' % math.sqrt(2 * math.log(2)))
print('  freeze: E %.4f T2 %.4f; e-folds %.3f %.3f; angle phase %.3f' % (0.5 * math.asinh(math.sqrt(19.3237)), 0.5 * math.asinh(math.sqrt(9.074)), 0.5 * math.log(math.sqrt(1 + 19.3237)), 0.5 * math.log(math.sqrt(1 + 9.074)), math.sqrt(9.074) * math.pi / 2))
print('  3^(-3/2) = %.5f ; Airy width (144 X^2)^(-1/3) = %.3e' % (3**-1.5, (144 * 2282.49**2)**(-1 / 3)))
gY = 0.44277; print("  't Hooft: 2pi^2/gY^2 = %.2f ; / C = %.3f ; N_C needed = %.1f" % (2 * math.pi**2 / gY**2, 2 * math.pi**2 / gY**2 / C, 2 * math.pi**2 / gY**2 / gC))
print('  a/c SM = %.4f' % ((995.5 / 360) / (283 / 120)))
print('  large-gauge copy 4pi/g^(2/3), g=0.53: %.2f' % (4 * math.pi / 0.53**(2 / 3)))
print('  pure gauge share: %.4f' % ((12.587 * 0.5396**(2 / 3) + 4.1167 * 0.5268**(2 / 3)) / C))
Xs = [np.diag([1, -1, 0]) / math.sqrt(2), np.diag([1, 1, -2]) / math.sqrt(6)] + [(np.eye(3)[:, [i]] @ np.eye(3)[[j], :] + np.eye(3)[:, [j]] @ np.eye(3)[[i], :]) / math.sqrt(2) for i, j in [(0, 1), (0, 2), (1, 2)]]
y = rng.normal(size=3); print('  completeness: sum (yXy)^2/|y|^4 = %.6f ; sum X^2 = %s' % (sum((y @ Xa @ y)**2 for Xa in Xs) / (y @ y)**2, np.round(sum(Xa @ Xa for Xa in Xs), 6).diagonal().tolist()))

print('I. slow-roll plateau (first order), end at eps=1')
a = math.sqrt(2 / 3)
def eps(p): e = math.exp(-a * p); return (4 / 3) * e**2 / (1 - e)**2
def eta_(p): e = math.exp(-a * p); return (4 / 3) * (2 * e**2 - e) / (1 - e)**2
from scipy.optimize import brentq
p_end = brentq(lambda p: eps(p) - 1, 0.1, 5)
def Nof(p): return quad(lambda q: 1 / math.sqrt(2 * eps(q)), p_end, p)[0]
for N in [55.722, 50.33]:
    p = brentq(lambda q: Nof(q) - N, p_end + 0.01, 10)
    print('  N=%.3f phi=%.4f  n_s=%.5f  r=%.5f  (1-2/N=%.5f, 12/N^2=%.5f)' % (N, p, 1 - 6 * eps(p) + 2 * eta_(p), 16 * eps(p), 1 - 2 / N, 12 / N**2))
