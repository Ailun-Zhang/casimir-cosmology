# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/R14_fv.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/R14_fv.py on 2026-10-14 by agent holonomy-cross (C149/D121).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_matching/R14_fv.py on 2026-10-13 by agent higgs-beyond-bo (C140/D112).
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12/higgs_survival/xi_scan/R14_fv.py on 2026-10-13 by agent higgs-matching (C138/D110).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Finite-volume one-loop sums on the 3-torus (periodic P / antiperiodic A): Ewald/Bessel sums Z_theta, D_theta(m^2), tadpoles T_B/2 = -0.1128925, T_F/2 = -0.0319080, quartic constant c2.  Paper: PRD App. H1 (finite-volume matching). Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

R14: independent finite-volume matching functions per real d.o.f. on the unit cube (energies in 1/L).

E_th(x)  = -(x^2/4pi^2) sum_{m!=0} w_m K_2(x|m|)/|m|^2   (full mode sum minus continuum; w=1 periodic,
           w=(-1)^{m1+m2+m3} antiperiodic AAA)
D_P(mu2) = E_P(x) - E_P(0) - x/2   (periodic, zero mode removed)        x = sqrt(mu2)
D_A(mu2) = E_A(x) - E_A(0)         (antiperiodic)
Small and tachyonic mu2: D = Z_th(1/2)/(8pi) mu2 + [c2 - log|mu2|/(64pi^2)] mu2^2
                           + 1/2 sum_{j>=3} binom(1/2,j) (2pi)^{1-2j} Z_th(j-1/2) mu2^j
Z_th(1/2) by Ewald (own derivation), Z_th(j-1/2) by direct shells + tail; c2 and E_th(0) by least squares on
0.7<=x<=2.4 where both representations converge (residual tests the log coefficient and the Bessel sums).
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import numpy as np
from scipy.special import kv, erfc, binom
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
PI = math.pi
RL = 60
JMAX = 80
XSW = 1.6


def shells_int(Rm):
    n = np.arange(-Rm, Rm + 1)
    q = (n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :] ** 2).ravel()
    q = q[(q > 0) & (q <= Rm * Rm)]
    N = np.bincount(q)
    qs = np.nonzero(N)[0]
    return qs.astype(float), N[qs].astype(float)


def shells_half(Rm):
    v = 2 * np.arange(-Rm - 1, Rm + 1) + 1
    q4 = (v[:, None, None] ** 2 + v[None, :, None] ** 2 + v[None, None, :] ** 2).ravel()
    q4 = q4[q4 <= 4 * Rm * Rm]
    N = np.bincount(q4)
    qs = np.nonzero(N)[0]
    return qs / 4.0, N[qs].astype(float)


def zeta_half():
    q, N = shells_int(10)
    rr = np.sqrt(q)
    ZP = -3.0 + np.sum(N * erfc(np.sqrt(PI) * rr) / rr) + np.sum(N * np.exp(-PI * q) / (PI * q))
    qh, Nh = shells_half(10)
    rh = np.sqrt(qh)
    sg = (-1.0) ** q
    ZA = np.sum(Nh * erfc(np.sqrt(PI) * rh) / rh) - 1.0 + np.sum(sg * N * np.exp(-PI * q) / (PI * q))
    return float(ZP), float(ZA)


class FV:
    def __init__(self):
        self.qP, self.NP = shells_int(RL)
        self.qA, self.NA = shells_half(RL)
        self.ZP12, self.ZA12 = zeta_half()
        self.a = {0: self._coeffs(0), 1: self._coeffs(1)}
        self.c2 = {}
        self.E0 = {}
        self.fitres = {}
        xs = np.linspace(0.7, 2.4, 35)
        for tw in (0, 1):
            Eb = self.ebessel(xs, tw)
            known = self._ser_known(xs ** 2, tw)
            lhs = Eb - (0.0 if tw else xs / 2) - known
            A = np.vstack([np.ones_like(xs), xs ** 4]).T
            sol = np.linalg.lstsq(A, lhs, rcond=None)[0]
            self.E0[tw], self.c2[tw] = float(sol[0]), float(sol[1])
            self.fitres[tw] = float(np.max(np.abs(lhs - A @ sol)))
        # tabulate mu2>=0 branch on x grid: series below XSW, Bessel above
        xg = np.linspace(0.0, 16.0, 3201)
        self.xg = xg
        self.tab = {}
        for tw in (0, 1):
            lo = xg <= XSW
            vals = np.empty_like(xg)
            vals[lo] = self.dser(xg[lo] ** 2, tw)
            hi = ~lo
            vals[hi] = self.ebessel(xg[hi], tw) - self.E0[tw] - (0.0 if tw else xg[hi] / 2)
            self.tab[tw] = CubicSpline(xg, vals)
        self.jump = {tw: float(self.ebessel(np.array([XSW]), tw)[0] - self.E0[tw] - (0 if tw else XSW / 2)
                               - self.dser(np.array([XSW ** 2]), tw)[0]) for tw in (0, 1)}

    def zdirect(self, s, tw):
        q, N = (self.qA, self.NA) if tw else (self.qP, self.NP)
        return float(np.sum(N * q ** (-s)) + 4 * PI * RL ** (3 - 2 * s) / (2 * s - 3))

    def _coeffs(self, tw):
        a = np.zeros(JMAX + 1)
        z12 = self.ZA12 if tw else self.ZP12
        a[1] = z12 / (8 * PI)
        for j in range(3, JMAX + 1):
            a[j] = 0.5 * binom(0.5, j) * (2 * PI) ** (1 - 2 * j) * self.zdirect(j - 0.5, tw)
        return a

    def _ser_known(self, mu2, tw):
        mu2 = np.asarray(mu2, float)
        a = self.a[tw]
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(mu2 != 0, np.log(np.abs(mu2)), 0.0)
        out = a[1] * mu2 - mu2 ** 2 * lg / (64 * PI ** 2)
        p = mu2 ** 3
        for j in range(3, JMAX + 1):
            out = out + a[j] * p
            p = p * mu2
        return out

    def dser(self, mu2, tw):
        mu2 = np.asarray(mu2, float)
        lim = (0.8 * 4 * PI ** 2) if tw == 0 else (0.8 * 3 * PI ** 2)
        if np.any(mu2 < -lim):
            raise ValueError("tachyonic mu2 beyond series radius")
        return self._ser_known(mu2, tw) + self.c2[tw] * mu2 ** 2

    def ebessel(self, x, tw):
        x = np.atleast_1d(np.asarray(x, float))
        q, N = self.qP, self.NP
        W = N * ((-1.0) ** q if tw else 1.0)
        sq = np.sqrt(q)
        return np.array([-(xx * xx / (4 * PI ** 2)) * np.sum(W * kv(2, xx * sq) / q) for xx in x])

    def D(self, mu2, tw):
        mu2 = np.atleast_1d(np.asarray(mu2, float))
        out = np.empty_like(mu2)
        neg = mu2 < 0
        if np.any(neg):
            out[neg] = self.dser(mu2[neg], tw)
        pos = ~neg
        if np.any(pos):
            out[pos] = self.tab[tw](np.sqrt(mu2[pos]))
        return out

    def DP(self, mu2):
        return self.D(mu2, 0)

    def DA(self, mu2):
        return self.D(mu2, 1)


def main():
    fv = FV()
    res = {}
    res["Z_P(1/2)"] = fv.ZP12
    res["Z_A(1/2)"] = fv.ZA12
    res["T_B/2=Z_P/(8pi)"] = fv.ZP12 / (8 * PI)
    res["T_F/2=Z_A/(8pi)"] = fv.ZA12 / (8 * PI)
    res["target_T_B/2"] = -0.1128924797204
    res["target_T_F/2"] = -0.0319080184183
    res["c2"] = fv.c2
    res["E_theta(0)"] = fv.E0
    res["fit_max_residual"] = fv.fitres
    res["switch_jump"] = fv.jump
    res["D_P(1e-5)/1e-5"] = float(fv.DP(1e-5)[0] / 1e-5)
    res["D_A(1e-5)/1e-5"] = float(fv.DA(1e-5)[0] / 1e-5)
    res["D_P(-1e-5)/-1e-5"] = float(fv.DP(-1e-5)[0] / -1e-5)
    res["D_P(mu2=4)"] = float(fv.DP(4.0)[0])
    res["D_P(mu2=4)_D108_wolfram"] = -0.38450643
    res["D_P(x=14)+7"] = float(fv.DP(196.0)[0] + 7.0)
    res["D108_Z2_over_2pi2"] = 0.8375369
    # independent check of tachyonic branch: direct shell sum of the convergent remainder at mu2=-10
    mu2 = -10.0
    k = 2 * PI * np.sqrt(fv.qP)
    rem = 0.5 * np.sum(fv.NP * (np.sqrt(k * k + mu2) - k - mu2 / (2 * k) + mu2 ** 2 / (8 * k ** 3)))
    # tail beyond RL: 1/2 * sum (mu2^3/16) k^-5  ~ (mu2^3/32) (2pi)^-5 * 4pi/(2 RL^2)
    rem += (mu2 ** 3 / 32.0) * (2 * PI) ** -5 * 4 * PI / (2 * RL ** 2)
    direct = fv.a[0][1] * mu2 + (fv.c2[0] - math.log(abs(mu2)) / (64 * PI ** 2)) * mu2 ** 2 + rem
    res["tachyon_check_mu2=-10"] = dict(series=float(fv.DP(mu2)[0]), direct_shells=float(direct))
    # tabulate a few values for the report
    res["table"] = {f"{m2:g}": dict(DP=float(fv.DP(m2)[0]), DA=float(fv.DA(m2)[0]) if m2 >= -20 else None)
                    for m2 in (-10.0, -3.0, -1.0, 0.25, 1.0, 4.0, 9.0, 25.0)}
    p = _zp.w("r1/R14_checks/results.json")
    d = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    d["check2_fv"] = res
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
