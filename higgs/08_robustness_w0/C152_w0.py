# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/higgs_robustness/C152_w0.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new module by agent higgs-robustness (C152/D124, v12 round 3) on 2026-10-15; uses C152_core (same dir), which
#   imports the holonomy_cross C149/C150 code unchanged.
r"""[v12 package, 2026-10-15] w0 by the B3 rule (time-symmetric real initial state with minimal energy variance in the centred Gaussian family): w0(172.2/172.6/173.0/173.4) = 0.9945/0.9907/0.9852/0.9801, critical masses with P = w0 P1.  Paper: PRD App. H3.

C152 w0 (T15 / B3 rule for the metastable Higgs zero mode at the turn, u = 0, round-3 central model).

H_2D(u=0) = -1/2 d2/dr2 + U1(r) + T_s + 1/2 Om^2(r) s^2 + W_L(r, s)   (C149 run02 pieces: 2D_ex + NNLO + cross terms;
r = |Phi| (Phi in R^4, reduced u = r^{3/2} R), s = |w| (w in R^3, v = s chi), sine DVR in s as in the hazard code).
Family (B3, D113): real (time-symmetric) Gaussians centred at the symmetric point Phi = 0, w = 0; by SO(4) x SO(3)
invariance the most general one is psi = exp(-|Phi|^2/2 sr^2 - |w|^2/2 ss^2) -> u v = r^{3/2} e^{-r^2/2sr^2} s e^{-s^2/2ss^2}.
Selection: minimise Var(H) over (sr, ss).  w0 = |<phi_res|psi>|^2 with phi_res = (a) walled quasi-bound state (Dirichlet
wall at 1.1/1.2/1.4 x outer turning point), (b) CAP c-normalised resonance (172.6 only).  Comparisons: energy-minimising
Gaussian; D111-type radial Gaussian centred at the radial well minimum (direct truncation at r = 0).
Usage: python -B C152_w0.py [mt ...]
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import time
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh, eigs
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C152_core as S  # noqa: E402

W, CD = S.W, S.CD
HH = 0.01
RMAX = 16.0


def pieces(mt, als=0.1180, l=0.0):
    mu = S.MU_B * math.exp(l)
    CS = W.Couplings(W.KIND)
    dmc = W.DMC()
    cfg = dict(W.FAST)
    r2, m2 = W.rgrid(float(cfg["Rout"]), float(cfg["h"]))
    sg = W.SGrid(cfg["N"], cfg["Smax"])
    c = CS(mt, als, mu)
    g3 = CD.g3_at(mt, als, mu)
    Pp, _ = S.parts(mt, als, l, c, False)
    U0, OM0 = W.build2d(Pp, dmc, r2, c, "2D_ex")
    inc, _ = CD.cross_increments(c, g3)
    U1 = U0 + inc["dC0"](r2) / r2 ** 2 + inc["dBx"](r2) / r2 ** 2 + inc["dU3"](r2)
    OM = OM0 + inc["dOm2e"](r2) + inc["omx_nl"](r2)
    Wl = CD.V_light_c(r2, sg.s, inc["beta"](r2), c["g2"], c["gY"])
    return r2, U1, OM, Wl, sg, c


def build(mt):
    r2, U1, OM, Wl, sg, c = pieces(mt)
    r = HH * np.arange(1, int(round(RMAX / HH)))
    Ureg = CubicSpline(r2, U1 - 3.0 / (8 * r2 * r2))(r) + 3.0 / (8 * r * r)
    Om = CubicSpline(r2, OM)(r)
    Wr = CubicSpline(r2, Wl, axis=0)(r)
    N = sg.N
    V = Ureg[:, None] + 0.5 * Om[:, None] * sg.s2[None, :] + Wr
    Vad = np.array([np.linalg.eigvalsh(sg.T + np.diag(V[i] - Ureg[i]))[0] for i in range(len(r))]) + Ureg
    return dict(r=r, V=V, Vad=Vad, sg=sg, N=N, c=c)


def hmat(B, M, cap=None):
    N = B["N"]
    d2 = sps.diags([np.ones(M - 1), -2 * np.ones(M), np.ones(M - 1)], [-1, 0, 1]) * (-0.5 / HH ** 2)
    H = sps.kron(d2, sps.identity(N)) + sps.kron(sps.identity(M), sps.csr_matrix(B["sg"].T))
    Vd = B["V"][:M].ravel().astype(complex if cap is not None else float)
    if cap is not None:
        rc, eta = cap
        Vd = Vd - 1j * eta * np.repeat(np.maximum(B["r"][:M] - rc, 0.0) ** 2, N)
    return (H + sps.diags(Vd)).tocsc()


def gauss(B, sr, ss, M):
    r, s = B["r"][:M], B["sg"].s
    g = (r ** 1.5 * np.exp(-r * r / (2 * sr * sr)))[:, None] * (s * np.exp(-s * s / (2 * ss * ss)))[None, :]
    g = g.ravel()
    return g / np.linalg.norm(g)


def radial_gauss(B, rm, sig, chi, M):
    u = np.exp(-(B["r"][:M] - rm) ** 2 / (2 * sig * sig))
    g = (u[:, None] * chi[None, :]).ravel()
    return g / np.linalg.norm(g)


def moments(H, g):
    Hg = H @ g
    e = float(g @ Hg)
    return e, float(Hg @ Hg) - e * e


def run(mt):
    t0 = time.time()
    B = build(mt)
    r, Vad = B["r"], B["Vad"]
    Lb, Qb, Gb = S.base_table()[round(mt, 2)]
    k0 = int(np.argmin(np.abs(Lb)))
    Q2D, G2D = float(Qb[k0]), float(Gb[k0])
    iw = int(np.argmin(Vad[(r > 0.3) & (r < 4)])) + int(np.argmax(r > 0.3))
    sel = (r > r[iw]) & (r < 14)
    itop = int(np.argmax(np.where(sel, Vad, -np.inf)))
    iout = itop + int(np.argmax(Vad[itop:] < Q2D))
    rec = dict(mt=mt, couplings=B["c"], Q_2D_root=Q2D, gamma_2D_root=G2D, r_well_min=float(r[iw]), V_well_min=float(Vad[iw]),
               r_top=float(r[itop]), V_top=float(Vad[itop]), r_out=float(r[iout]))
    Mfull = len(r)
    Hf = hmat(B, Mfull)
    states = {}
    for fac in (1.1, 1.2, 1.4):
        M = int(np.argmax(r > fac * r[iout]))
        Hw = hmat(B, M)
        w, v = eigsh(Hw, k=6, sigma=Q2D, which="LM")
        wt = [float(np.sum(np.abs(v[:, j].reshape(M, B["N"])[r[:M] < r[itop]]) ** 2)) for j in range(len(w))]
        j = int(np.argmax(wt))
        ph = np.zeros(Mfull * B["N"])
        ph[:M * B["N"]] = v[:, j] / np.linalg.norm(v[:, j])
        states["wall%.1f" % fac] = ph
        rec["wall%.1f" % fac] = dict(r_wall=float(r[M - 1]), E=float(w[j]), E_minus_Q2D=float(w[j] - Q2D),
                                     inner_weight=wt[j])
    phi = states["wall1.2"]
    # B3: minimum energy variance in the centred Gaussian family; and the energy-minimising Gaussian
    f_var = lambda p: moments(Hf, gauss(B, math.exp(p[0]), math.exp(p[1]), Mfull))[1]  # noqa: E731
    f_en = lambda p: moments(Hf, gauss(B, math.exp(p[0]), math.exp(p[1]), Mfull))[0]  # noqa: E731
    pe = minimize(f_en, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=400)).x
    pv = minimize(f_var, pe, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-12, maxiter=400)).x
    for tag, p in (("B3_minvar", pv), ("Emin", pe)):
        g = gauss(B, math.exp(p[0]), math.exp(p[1]), Mfull)
        e, var = moments(Hf, g)
        rec[tag] = dict(sigma_r=float(math.exp(p[0])), sigma_s=float(math.exp(p[1])), E=e, Var=var, dE=math.sqrt(max(var, 0)),
                        w0={k: float((st @ g) ** 2) for k, st in states.items()})
    # width sensitivity around the B3 state
    sens = {}
    for fr in (0.7, 0.85, 1.15, 1.3):
        g = gauss(B, fr * math.exp(pv[0]), math.exp(pv[1]), Mfull)
        sens["sr x %.2f" % fr] = float((phi @ g) ** 2)
    for fs in (0.7, 1.3):
        g = gauss(B, math.exp(pv[0]), fs * math.exp(pv[1]), Mfull)
        sens["ss x %.2f" % fs] = float((phi @ g) ** 2)
    rec["B3_width_sensitivity_w0"] = sens
    # D111-type construction: radial Gaussian centred at the radial well minimum, direct truncation at r = 0,
    # s-part = adiabatic ground channel at r_m; widths 1x / 2x the harmonic width of V_ad at r_m
    sg = B["sg"]
    ev, evec = np.linalg.eigh(sg.T + np.diag(B["V"][iw] - B["V"][iw, 0] + 0.0))
    chi = evec[:, 0] * np.sign(evec[0, 0])
    w2 = float((Vad[iw + 5] - 2 * Vad[iw] + Vad[iw - 5]) / (5 * HH) ** 2)
    sh = 1.0 / math.sqrt(math.sqrt(max(w2, 1e-6)))
    d111 = dict(r_m=float(r[iw]), omega2=w2, sigma_harm=sh)
    for fw in (1.0, 2.0):
        g = radial_gauss(B, r[iw], fw * sh, chi, Mfull)
        e, var = moments(Hf, g)
        d111["%gx" % fw] = dict(w0=float((phi @ g) ** 2), E=e, dE=math.sqrt(max(var, 0)))
    rec["D111_type_radial_gaussian"] = d111
    # CAP c-product check (one column)
    if abs(mt - 172.6) < 1e-6:
        cap = {}
        for eta in (0.05, 0.2):
            rc = float(r[iout] + 1.0)
            Hc = hmat(B, Mfull, cap=(rc, eta))
            try:
                wc, vc = eigs(Hc, k=4, sigma=Q2D - 0.5j * G2D, which="LM")
                j = int(np.argmin(np.abs(wc - (Q2D - 0.5j * G2D))))
                f = vc[:, j] / np.sqrt(vc[:, j] @ vc[:, j])
                g = gauss(B, math.exp(pv[0]), math.exp(pv[1]), Mfull)
                cap["eta=%.2f" % eta] = dict(r_c=rc, E=[float(wc[j].real), float(wc[j].imag)], gamma=float(-2 * wc[j].imag),
                                             w0_cprod=float(abs((f @ g) ** 2)), w0_cprod_complex=[float((f @ g) ** 2).real
                                                                                                  if False else float(((f @ g) ** 2).real),
                                                                                                  float(((f @ g) ** 2).imag)])
            except Exception as ex:  # noqa: BLE001
                cap["eta=%.2f" % eta] = dict(error=repr(ex))
        rec["CAP"] = cap
    rec["time_s"] = time.time() - t0
    return rec


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mts = [float(a) for a in sys.argv[1:]] or [172.2, 172.6, 173.0, 173.4]
    out = dict(model="C149 run02 central (2D_ex + NNLO + cross), u = 0, alpha_s = 0.1180", grid=dict(h=HH, rmax=RMAX, N=24, Smax=12),
               cols={})
    fn = os.path.join(S.RES, "C152_w0_B3.json")
    for mt in mts:
        rec = run(mt)
        out["cols"]["%.2f" % mt] = rec
        W.wjson(fn, out)
        print(json.dumps(W.clean({k: rec[k] for k in rec if k not in ("couplings",)}), indent=1), flush=True)
