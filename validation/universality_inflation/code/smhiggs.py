# smhiggs.py -- ui-blind (V5), own implementation, 2026-10-15
# SM RGEs to three loops (Buttazzo et al. 2013, arXiv:1307.3536, App. B, eqs. (96)-(101); y_b=y_tau=0),
# d/d ln mu^2, GUT g1^2 = 5 gY^2/3.  xi runs as (xi-1/6)*gamma_m (gamma_m = d ln m^2/d ln mu^2);
# MMR 2021 use the one-loop gamma_m (their eq. 2.15).  alpha (R^2/144 coefficient) runs with MMR eq. (2.16).
# Curved-space one-loop potential: Markkanen-Nurmi-Rajantie-Stopyra 2018 (arXiv:1804.02020) Tables 1-2,
# eq. (5.2) with H^2 -> R/12 (as MMR 2021 eqs. 2.9-2.13); RG improvement mu=mu*(h,R) with DeltaV_loops=0.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

PI = np.pi
K1 = 1.0 / (16 * PI**2)


def betas(Y, nloop=3):
    """d(G1,G2,G3,Yt,L)/d ln mu^2, G=g^2 (G1 GUT), Yt=yt^2, L=lambda (V = lambda |H|^4)."""
    G1, G2, G3, Yt, L = Y
    k = K1; k2 = k * k; k3 = k2 * k
    b1 = G1 * G1 * (k * 41 / 10 + k2 * (44 * G3 / 5 + 27 * G2 / 10 + 199 * G1 / 50 - 17 * Yt / 10))
    b2 = G2 * G2 * (k * (-19 / 6) + k2 * (12 * G3 + 35 * G2 / 6 + 9 * G1 / 10 - 3 * Yt / 2))
    b3 = G3 * G3 * (k * (-7) + k2 * (-26 * G3 + 9 * G2 / 2 + 11 * G1 / 10 - 2 * Yt))
    bt = Yt * (k * (9 * Yt / 2 - 8 * G3 - 9 * G2 / 4 - 17 * G1 / 20)
               + k2 * (Yt * (-12 * Yt - 12 * L + 36 * G3 + 225 * G2 / 16 + 393 * G1 / 80)
                       + 6 * L * L - 108 * G3**2 - 23 * G2**2 / 4 + 1187 * G1**2 / 600
                       + 9 * G3 * G2 + 19 * G3 * G1 / 15 - 9 * G2 * G1 / 20))
    bl = (k * (L * (12 * L + 6 * Yt - 9 * G2 / 2 - 9 * G1 / 10) - 3 * Yt**2
               + 9 * G2**2 / 16 + 27 * G1**2 / 400 + 9 * G2 * G1 / 40)
          + k2 * (L**2 * (-156 * L - 72 * Yt + 54 * G2 + 54 * G1 / 5)
                  + L * Yt * (-3 * Yt / 2 + 40 * G3 + 45 * G2 / 4 + 17 * G1 / 4)
                  + L * (-73 * G2**2 / 16 + 1887 * G1**2 / 400 + 117 * G2 * G1 / 40)
                  + Yt**2 * (15 * Yt - 16 * G3 - 4 * G1 / 5)
                  + Yt * (-9 * G2**2 / 8 - 171 * G1**2 / 200 + 63 * G2 * G1 / 20)
                  + 305 * G2**3 / 32 - 3411 * G1**3 / 4000 - 289 * G2**2 * G1 / 160
                  - 1677 * G2 * G1**2 / 800))
    if nloop >= 3:
        b1 += G1 * G1 * k3 * (Yt * (189 * Yt / 16 - 29 * G3 / 5 - 471 * G2 / 32 - 2827 * G1 / 800)
                              + L * (-9 * L / 5 + 9 * G2 / 10 + 27 * G1 / 50)
                              + 297 * G3**2 / 5 + 789 * G2**2 / 64 - 388613 * G1**2 / 24000
                              - 3 * G3 * G2 / 5 - 137 * G3 * G1 / 75 + 123 * G2 * G1 / 160)
        b2 += G2 * G2 * k3 * (Yt * (147 * Yt / 16 - 7 * G3 - 729 * G2 / 32 - 593 * G1 / 160)
                              + L * (-3 * L + 3 * G2 / 2 + 3 * G1 / 10)
                              + 81 * G3**2 + 324953 * G2**2 / 1728 - 5597 * G1**2 / 1600
                              + 39 * G3 * G2 - G3 * G1 / 5 + 873 * G2 * G1 / 160)
        b3 += (G3 * G3 * k3 * (Yt * (15 * Yt - 40 * G3 - 93 * G2 / 8 - 101 * G1 / 40)
                               + 65 * G3**2 / 2 + 109 * G2**2 / 8 - 523 * G1**2 / 120
                               + 21 * G3 * G2 + 77 * G3 * G1 / 15 - 3 * G2 * G1 / 40)
               + G3**5 * k2 * k2 * (-2472.28))
        bt += Yt * k3 * (Yt**2 * (58.6028 * Yt + 198 * L - 157 * G3 - 1593 * G2 / 16 - 2437 * G1 / 80)
                         + L * Yt * (15 * L / 4 + 16 * G3 - 135 * G2 / 2 - 127 * G1 / 10)
                         + Yt * (363.764 * G3**2 + 16.990 * G2**2 - 24.422 * G1**2 + 48.370 * G3 * G2
                                 + 18.074 * G3 * G1 + 34.829 * G2 * G1)
                         + L**2 * (-36 * L + 45 * G2 + 9 * G1)
                         + L * (-171 * G2**2 / 16 - 1089 * G1**2 / 400 + 117 * G2 * G1 / 40)
                         - 619.35 * G3**3 + 169.829 * G2**3 + 16.099 * G1**3 + 73.654 * G3**2 * G2
                         - 15.096 * G3**2 * G1 - 21.072 * G3 * G2**2 - 22.319 * G3 * G1**2
                         - 321 / 20 * G3 * G2 * G1 - 4.743 * G2**2 * G1 - 4.442 * G2 * G1**2)
        bl += k3 * (L**3 * (6011.35 * L + 873 * Yt - 387.452 * G2 - 77.490 * G1)
                    + L**2 * Yt * (1768.26 * Yt + 160.77 * G3 - 359.539 * G2 - 63.869 * G1)
                    + L**2 * (-790.28 * G2**2 - 185.532 * G1**2 - 316.64 * G2 * G1)
                    + L * Yt**2 * (-223.382 * Yt - 662.866 * G3 - 5.470 * G2 - 21.015 * G1)
                    + L * Yt * (356.968 * G3**2 - 319.664 * G2**2 - 74.8599 * G1**2 + 15.1443 * G3 * G2
                                + 17.454 * G3 * G1 + 5.615 * G2 * G1)
                    + L * G2**2 * (-57.144 * G3 + 865.483 * G2 + 79.638 * G1)
                    + L * G1**2 * (-8.381 * G3 + 61.753 * G2 + 28.168 * G1)
                    + Yt**3 * (-243.149 * Yt + 250.494 * G3 + 74.138 * G2 + 33.930 * G1)
                    + Yt**2 * (-50.201 * G3**2 + 15.884 * G2**2 + 15.948 * G1**2 + 13.349 * G3 * G2
                               + 17.570 * G3 * G1 - 70.356 * G2 * G1)
                    + Yt * G3 * (16.464 * G2**2 + 1.016 * G1**2 + 11.386 * G2 * G1)
                    + Yt * G2**2 * (62.500 * G2 + 13.041 * G1)
                    + Yt * G1**2 * (10.627 * G2 + 11.117 * G1)
                    + G3 * (7.536 * G2**3 + 0.663 * G1**3 + 1.507 * G2**2 * G1 + 1.105 * G2 * G1**2)
                    - 114.091 * G2**4 - 1.508 * G1**4 - 37.889 * G2**3 * G1 + 6.500 * G2**2 * G1**2
                    - 1.543 * G2 * G1**3)
    return np.array([b1, b2, b3, bt, bl])


def gamma_m(Y, nloop=1):
    """d ln m^2 / d ln mu^2 (Buttazzo eq. 100), truncated at nloop."""
    G1, G2, G3, Yt, L = Y
    k = K1; k2 = k * k; k3 = k2 * k
    g = k * (6 * L + 3 * Yt - 9 * G2 / 4 - 9 * G1 / 20)
    if nloop >= 2:
        g += k2 * (L * (-30 * L - 36 * Yt + 36 * G2 + 36 * G1 / 5)
                   + Yt * (-27 * Yt / 4 + 20 * G3 + 45 * G2 / 8 + 17 * G1 / 8)
                   - 145 * G2**2 / 32 + 1671 * G1**2 / 800 + 9 * G2 * G1 / 16)
    if nloop >= 3:
        g += k3 * (L**2 * (1026 * L + 297 * Yt / 2 - 192.822 * G2 - 38.564 * G1)
                   + L * Yt * (347.394 * Yt + 80.385 * G3 - 318.591 * G2 - 59.699 * G1)
                   + L * (-64.5145 * G2**2 - 65.8056 * G1**2 - 37.8231 * G2 * G1)
                   + Yt**2 * (154.405 * Yt - 209.24 * G3 - 3.82928 * G2 - 7.50769 * G1)
                   + Yt * (178.484 * G3**2 - 102.627 * G2**2 - 27.721 * G1**2 + 7.572 * G3 * G2
                           + 8.727 * G3 * G1 + 11.470 * G2 * G1)
                   + G2**2 * (-28.572 * G3 + 301.724 * G2 + 9.931 * G1)
                   + G1**2 * (-4.191 * G3 + 9.778 * G2 + 8.378 * G1))
    return g


def buttazzo_matching(Mt, Mh, alphas, MW=80.3692):
    """MS-bar couplings at mu=Mt, Buttazzo et al. eqs. (55),(57)-(60)."""
    lam = 0.12604 + 0.00206 * (Mh - 125.15) - 0.00004 * (Mt - 173.34)
    yt = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (alphas - 0.1184) / 0.0007
    g2 = 0.64779 + 0.00004 * (Mt - 173.34) + 0.00011 * (MW - 80.384) / 0.014
    gY = 0.35830 + 0.00011 * (Mt - 173.34) - 0.00020 * (MW - 80.384) / 0.014
    g3 = 1.1666 + 0.00314 * (alphas - 0.1184) / 0.0007 - 0.00046 * (Mt - 173.34)
    return dict(g2=g2, gY=gY, g3=g3, yt=yt, lam=lam)


class SMRun:
    """Runs SM couplings from mu0 upward; stores splines in s = ln(mu^2/GeV^2).
    Also K_n(s) = exp(int gamma_m^(n) ds) and J2_n(s) = int K_n^2 ds for n = 1, 3 (xi and alpha running)."""

    def __init__(self, mu0, c0, nloop=3, mumax=1e20, npts=4001):
        self.mu0 = mu0
        self.s0 = 2 * np.log(mu0)
        s1 = 2 * np.log(mumax)
        G1 = 5.0 / 3.0 * c0["gY"]**2
        y0 = [G1, c0["g2"]**2, c0["g3"]**2, c0["yt"]**2, c0["lam"], 0.0, 0.0, 0.0, 0.0]

        def rhs(s, y):
            Y = y[:5]
            return np.concatenate([betas(Y, nloop), [gamma_m(Y, 1), gamma_m(Y, 3),
                                                     np.exp(2 * y[5]), np.exp(2 * y[6])]])
        sol = solve_ivp(rhs, (self.s0, s1), y0, method="DOP853", rtol=1e-12, atol=1e-15,
                        dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.sg = np.linspace(self.s0, s1, npts)
        Yg = sol.sol(self.sg)
        self.sp = [CubicSpline(self.sg, Yg[i]) for i in range(9)]
        self.smin, self.smax = self.sg[0], self.sg[-1]
        self.sol = sol

    def coup(self, s):
        s = min(max(s, self.smin), self.smax)
        G1, G2, G3, Yt, L = (self.sp[i](s) for i in range(5))
        return float(G1), float(G2), float(G3), float(Yt), float(L)

    def lam(self, mu):
        return float(self.sp[4](2 * np.log(mu)))

    def K(self, s, xiloop=1):
        s = min(max(s, self.smin), self.smax)
        return float(np.exp(self.sp[5 if xiloop == 1 else 6](s)))

    def J2(self, s, xiloop=1):
        s = min(max(s, self.smin), self.smax)
        return float(self.sp[7 if xiloop == 1 else 8](s))

    def xi(self, s, c, xiloop=1):
        return 1.0 / 6.0 + c * self.K(s, xiloop)

    def alpha(self, s, c, xiloop=1):
        s = min(max(s, self.smin), self.smax)
        return (288 * c * c * self.J2(s, xiloop) - 66.366666666666667 * (s - self.s0)) / (32 * PI**2)


# ---- curved-space one-loop terms: Markkanen et al. 2018 Tables 1-2 ----
_n = np.array([2, 6, -2, 1, 3, -1, -12, -12, -12, -12, -12, -12, -4, -4, -4, 1, 2, 1, -2, -1,
               1, 3, -1, 8, 24, -8, -2, -2, -2, -1, -8], dtype=float)
_d = np.array([1.5, 5 / 6, 1.5, 1.5, 5 / 6, 1.5] + [1.5] * 9 + [1.5] * 5
              + [1.5, 5 / 6, 1.5, 1.5, 5 / 6, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
_np = np.array([-34 / 15, -34 / 5, 4 / 15, -17 / 15, -17 / 5, 2 / 15] + [38 / 5] * 6 + [38 / 15] * 3
               + [-2 / 15, -4 / 15, -2 / 15, 4 / 15, 2 / 15]
               + [-17 / 15, -17 / 5, 2 / 15, -136 / 15, -136 / 5, 16 / 15, 19 / 15, 19 / 15, 19 / 15,
                  2 / 15, 16 / 15])
assert len(_n) == len(_d) == len(_np) == 31
# consistency: sum(n theta^2 + n') at h=0 must equal 2*16pi^2*... beta_alpha structure (checked in tests)


def masses2(h2, H2, G2, gY2, Yt, L, xi, zeta=1.0):
    mW2 = G2 * h2 / 4; mZ2 = (G2 + gY2) * h2 / 4; mt2 = Yt * h2 / 2
    mh2 = 3 * L * h2; mc2 = L * h2; a = 12 * (xi - 1 / 6) * H2
    return np.array([mW2 + H2, mW2 + H2, mW2 - 2 * H2, mZ2 + H2, mZ2 + H2, mZ2 - 2 * H2,
                     mt2 + H2, H2, H2, H2, H2, H2, H2, H2, H2,
                     mh2 + a, mc2 + zeta * mW2 + a, mc2 + zeta * mZ2 + a,
                     zeta * mW2 - 2 * H2, zeta * mZ2 - 2 * H2,
                     H2, H2, -2 * H2, H2, H2, -2 * H2, H2, H2, H2, -2 * H2, -2 * H2])


class Veff:
    def __init__(self, run, c, xiloop=1, zeta=1.0, noalpha=False, light_fields=True):
        self.run, self.c, self.xiloop, self.zeta = run, c, xiloop, zeta
        self.noalpha = noalpha
        self.mask = np.ones(31, dtype=bool)
        if not light_fields:          # diagnostic: drop Table-2 fields (i=21..31) from DeltaV_loops
            self.mask[20:] = False

    def F(self, s, h2, H2):
        G1, G2, G3, Yt, L = self.run.coup(s)
        xi = self.run.xi(s, self.c, self.xiloop)
        M2 = masses2(h2, H2, G2, 0.6 * G1, Yt, L, xi, self.zeta)
        lg = np.log(np.abs(M2)) - s
        return float(np.sum((_n * M2 * M2 * (lg - _d) + _np * H2 * H2 * lg)[self.mask]))

    def sstar(self, h2, H2):
        s0 = np.log(h2 + H2)
        a, b = s0 - 6.0, s0 + 6.0
        fa, fb = self.F(a, h2, H2), self.F(b, h2, H2)
        it = 0
        while fa * fb > 0 and it < 8:
            a -= 6.0; b += 6.0
            fa, fb = self.F(a, h2, H2), self.F(b, h2, H2)
            it += 1
        if fa * fb > 0:
            raise RuntimeError("no mu* bracket")
        return brentq(self.F, a, b, args=(h2, H2), xtol=1e-12, rtol=1e-14)

    def V(self, h, R):
        H2 = R / 12.0
        h2 = h * h
        s = self.sstar(h2, H2)
        G1, G2, G3, Yt, L = self.run.coup(s)
        xi = self.run.xi(s, self.c, self.xiloop)
        al = 0.0 if self.noalpha else self.run.alpha(s, self.c, self.xiloop)
        return 0.5 * xi * R * h2 + 0.25 * L * h2 * h2 + al * R * R / 144.0

    def barrier(self, R, nscan=61):
        """returns (DeltaV, h_bar, mu*(h_bar)) ; DeltaV=0 if no barrier."""
        H = np.sqrt(R / 12.0)
        V0 = self.V(0.0, R)
        lh = np.linspace(np.log(0.2 * H), np.log(400 * H), nscan)
        dv = np.array([self.V(np.exp(x), R) - V0 for x in lh])
        i = int(np.argmax(dv))
        if dv[i] <= 0 or i == 0:
            return 0.0, 0.0, 0.0
        if i == nscan - 1:
            raise RuntimeError("barrier beyond scan range")
        res = minimize_scalar(lambda x: -(self.V(np.exp(x), R) - V0), bracket=(lh[i - 1], lh[i], lh[i + 1]),
                              tol=1e-10)
        hb = float(np.exp(res.x))
        return float(-res.fun), hb, float(np.exp(0.5 * self.sstar(hb * hb, R / 12.0)))

    def B_HM(self, R):
        dV, hb, mus = self.barrier(R)
        return 384 * PI**2 * dV / R**2, hb, mus
