# -*- coding: utf-8 -*-
"""
r15_rt.py -- R15 independent check of D111 (agent r15-realtime, 2026-10-13).  D111 code/results NOT opened.

Own implementation:
  * potential: MU model tables built here from the R14 code (R14_res.U_radial, R14_rge; reuse allowed by brief),
    or a D111-format npz table (C144 final model = INPUT for extra item 4).  U = G(r;u) + lam(1/L) r^4/4,
    G (non-quartic part) linear in u between nodes, lam(1/L) exact from the running table.
  * TDSE in the conformal r-frame ALL THE WAY to u_sw (no lens map during evolution):
      i d_eta psi = [-1/2 d_r^2 + U(r;u) - (1-6xi)(hL)^2 r^2 - iW] psi ,  d eta = du/hL,
    2nd-order FD (Dirichlet at r=0), Crank-Nicolson, step d_eta = min(0.02, cE/(hL)^4);
    absorber: from r_c(u) = cummax[max(10, 1.5 r_top^phys(u), 10 hL)] (r_top^phys = barrier top of
    U + 6 xi (hL)^2 r^2 = physical-frame barrier in r units) the potential is smoothly flattened and a quadratic
    CAP acts.  Absorbed norm counts as runaway.
  * at u_sw: lens map only as a change of variables at one instant, y = r/hL,
    Phi(y) = sqrt(hL) exp(-i hL tanh2u r^2/2) psi(r); discrete Wigner function by FFT, summed into cells;
    each cell -> cosmic-time classical trajectory
      y'' + 3 tanh(2u) y' + 12 xi y + G'(hL y;u)/(hL)^3 + d/dy[lam y^4/4] = 0,
    lam = lam(1/L) ('L')  or  lam(mu), mu = sqrt(phi^2+h^2+L^-2) with its dlam/dy term ('phi').
    runaway: y > max(40, 3 y_b);  bounded: u > max(u_sw+1.5, 7.5), y < y_b and E_q < 0.85 V_b.
Usage: python -B r15_rt.py tables|smoke|phase1|c144|post
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import traceback
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.lapack import zgtsv
from scipy.integrate import cumulative_trapezoid, quad
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
R14DIR = os.path.join(ROOT, r"04_calculations\code\20261013_claude_v12\higgs_survival\xi_scan")
OUT = os.path.join(ROOT, r"07_reviews\R15_v12_round2\higgs_realtime")
TABDIR = os.path.join(OUT, "tables")
RUNS = os.path.join(OUT, "runs")
C144 = os.path.join(ROOT, r"04_calculations\results\20261013_claude_v12_r2\higgs_final\run01\tables"
                          r"\U_table_final_mt%.1f.npz")
for _d in (TABDIR, RUNS):
    os.makedirs(_d, exist_ok=True)
HLB = 0.00437243
MU_B = 2.435e18 / 722.0
LN_MUB = math.log(MU_B)
LN_H = math.log(HLB * MU_B)          # h in GeV:  hL_b = h L_b, mu_b = 1/L_b
XI6 = 1.0 / 6.0
NWORK = 3

_UF = np.linspace(0.0, 10.0, 400001)
_FF = cumulative_trapezoid(1.0 / np.sqrt(np.cosh(2 * _UF)), _UF, initial=0.0)


def Fint(u):
    return float(np.interp(u, _UF, _FF))


def hl(u):
    return HLB * np.sqrt(np.cosh(2 * np.asarray(u, float)))


def eta_rem(u):
    f = lambda v: math.sqrt(2.0) * math.exp(-v) / math.sqrt(1.0 + math.exp(-4 * v))  # noqa: E731
    return quad(f, u, np.inf, epsabs=1e-13, epsrel=1e-12)[0] / HLB


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, (complex, np.complexfloating)):
        return [clean(o.real), clean(o.imag)]
    return o


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "compute_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def first_min_max(V, rr, rstart=0.3):
    d = np.diff(V)
    j0 = int(np.searchsorted(rr, rstart))
    mins = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    mins = mins[mins >= j0]
    jm = int(mins[0]) if len(mins) else j0
    maxs = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    maxs = maxs[maxs > jm]
    jt = int(maxs[0]) if len(maxs) else None
    return jm, jt


# ============================================================================================ potential tables
class Table:
    def __init__(self, model, mt):
        self.model, self.mt = model, mt
        if model == "MU":
            d = np.load(os.path.join(TABDIR, "MU_mt%.1f.npz" % mt))
            r, un, G, lamn, lnmu, lamg = d["r"], d["u"], d["G"], d["lam"], d["lnmu"], d["lamg"]
        elif model == "C144":
            d = np.load(C144 % mt, allow_pickle=True)
            r, un, U, lamn = d["r"], d["u"], d["U"], d["lam"]
            G = U - lamn[:, None] * r[None, :] ** 4 / 4.0
            lnmu, lamg = d["logmu"], d["lam_run"]
        else:
            raise ValueError(model)
        self.r, self.u, self.G, self.lamn = r, un, G, lamn
        self.dr = float(r[1] - r[0])
        self.R = float(r[-1])
        self.lam_spl = CubicSpline(lnmu, lamg)
        self.beta_spl = self.lam_spl.derivative()
        self.lnlo, self.lnhi = float(lnmu[0]), float(lnmu[-1])
        nf = 80
        self.slope = np.array([np.polyfit(r[-nf:], g[-nf:], 1)[0] for g in G])
        self.Gp = np.gradient(G, r, axis=1)

    def lam(self, lnmu):
        return self.lam_spl(np.clip(lnmu, self.lnlo, self.lnhi))

    def beta(self, lnmu):
        x = np.asarray(lnmu, float)
        return np.where((x > self.lnlo) & (x < self.lnhi), self.beta_spl(np.clip(x, self.lnlo, self.lnhi)), 0.0)

    def lnmuL(self, u):
        return LN_MUB - 0.5 * (np.logaddexp(2 * u, -2 * u) - math.log(2.0))

    def lamL(self, u):
        return float(self.lam(self.lnmuL(u)))

    def uidx(self, u):
        un = self.u
        if u <= un[0]:
            return 0, 0.0
        if u >= un[-1]:
            return len(un) - 2, 1.0
        k = int(np.searchsorted(un, u) - 1)
        return k, float((u - un[k]) / (un[k + 1] - un[k]))

    def Gx(self, k, rr):
        out = np.interp(rr, self.r, self.G[k])
        hi = rr > self.R
        out[hi] = self.G[k, -1] + self.slope[k] * (rr[hi] - self.R)
        return out

    def Gpx(self, k, rr):
        out = np.interp(rr, self.r, self.Gp[k])
        out[rr > self.R] = self.slope[k]
        return out

    def U_at(self, u, rr):
        k, t = self.uidx(u)
        return (1 - t) * self.Gx(k, rr) + t * self.Gx(k + 1, rr) + self.lamL(u) * rr ** 4 / 4

    def G_on(self, rg):
        out = np.empty((len(self.u), len(rg)))
        ins = rg <= self.R + 1e-9
        idx = np.rint(rg[ins] / self.dr).astype(int) - 1
        if np.max(np.abs(self.r[idx] - rg[ins])) > 1e-9:
            raise ValueError("TDSE grid must be a sub-lattice of the table grid")
        out[:, ins] = self.G[:, idx]
        x = rg[~ins] - self.R
        out[:, ~ins] = self.G[:, -1][:, None] + self.slope[:, None] * x[None, :]
        return out


def _prof(args):
    sys.path.insert(0, R14DIR)
    import R14_res as R
    return R.gauss_profile(*args)


def build_tables():
    t0 = time.time()
    sys.path.insert(0, R14DIR)
    import R14_res as R
    import R14_rge as RG
    cal, _ = RG.calibration()
    un = np.round(np.concatenate([np.arange(0, 3.0, 0.25), np.arange(3.0, 5.0, 0.125),
                                  np.arange(5.0, 7.0001, 0.25)]), 6)
    mus = RG.MU_B / np.sqrt(np.cosh(2 * un))
    Yc = RG.solve(172.6, 0.1180, 125.13).sol(np.log(mus))
    with ProcessPoolExecutor(max_workers=2) as ex:
        profs = list(ex.map(_prof, [(float(Yc[1, i]), float(Yc[0, i])) for i in range(len(un))]))
    log("profiles done %.0fs cal=%.6g" % (time.time() - t0, cal))
    r = 0.0025 * np.arange(1, 7001)
    sols = {}
    for mt in (172.2, 172.6):
        sol = RG.solve(mt, 0.1180, 125.13)
        sols[mt] = sol
        Y = sol.sol(np.log(mus))
        lam = Y[4] + cal
        G = np.empty((len(un), len(r)))
        for i in range(len(un)):
            U = R.U_radial(profs[i], "MU", float(lam[i]), float(Yc[1, i]), float(Yc[0, i]), float(Y[3, i]), r, 0.0)
            G[i] = U - lam[i] * r ** 4 / 4
        lng = np.linspace(math.log(mt) + 0.01, math.log(2.9e19), 6000)
        lamg = sol.sol(lng)[4] + cal
        np.savez(os.path.join(TABDIR, "MU_mt%.1f.npz" % mt), r=r, u=un, G=G, lam=lam, lnmu=lng, lamg=lamg,
                 yt=Y[3], g2=Yc[1], gY=Yc[0], cal=cal)
    log("tables written %.0fs" % (time.time() - t0))
    # Siegert poles (R14 machinery, continuation in u), xi = 1/6 (no curvature term), u <= 5
    rs, m = R.grid(16.0)
    sel = [i for i in range(len(un)) if un[i] <= 5.0 + 1e-9]
    mts = (172.2, 172.6)
    Ys = {mt: sols[mt].sol(np.log(mus)) for mt in mts}
    E_prev, prev, Es, nh = None, None, [], 0
    for i in sel:
        cols = np.array([R.U_radial(profs[i], "MU", float(Ys[mt][4, i] + cal), float(Yc[1, i]), float(Yc[0, i]),
                                    float(Ys[mt][3, i]), rs, 0.0) for mt in mts])
        s_, a_, _ = R.spar(profs[i])
        sl, al = np.full(2, s_), np.full(2, a_)
        E, it, res = R.solve_set(list(cols), rs, m, sl, al, guess=E_prev)
        if E_prev is not None:
            with np.errstate(all="ignore"):
                bad = (~np.isfinite(E)) | (np.abs(E.real - E_prev.real) > 0.25) | (np.abs(np.log(
                    np.maximum(-2 * E.imag, 1e-300) / np.maximum(-2 * E_prev.imag, 1e-300))) > 3.5)
            if bad.any():
                Eh = E_prev.copy()
                for t in np.linspace(0.125, 1.0, 8):
                    Eh, _, _ = R.solve_set(list((1 - t) * prev + t * cols), rs, m, sl, al, guess=Eh)
                E = Eh
                nh += 1
        Es.append(E.copy())
        E_prev, prev = E.copy(), cols
    Es = np.array(Es)
    out = dict(u=un[sel], cal=cal, n_homotopy=nh, time_s=time.time() - t0)
    for j, mt in enumerate(mts):
        out["mt=%.1f" % mt] = dict(Q=Es[:, j].real, gamma=-2 * Es[:, j].imag)
    write_json(os.path.join(TABDIR, "poles_xi16.json"), out)
    log("poles done %.0fs; gamma(172.6,u=0)=%.4e Q=%.5f" % (time.time() - t0, -2 * Es[0, 1].imag, Es[0, 1].real))


# ============================================================================================ TDSE (r-frame)
def tdse(cfg):
    t0 = time.time()
    tb = Table(cfg["model"], cfg["mt"])
    xi = float(cfg["xi"])
    dr = cfg.get("dr", 0.0025)
    uend = float(cfg["u_end"])
    fcap, Rc0 = cfg.get("fcap", 1.5), cfg.get("Rc0", 10.0)
    ell0, w0, w1, W0 = cfg.get("ell0", 1.0), cfg.get("w0", 6.0), cfg.get("w1", 8.0), cfg.get("W0", 30.0)
    deta0, cE = cfg.get("deta0", 0.02), cfg.get("cE", 2e-4)
    eta_early, deta_early = cfg.get("eta_early", 0.0), cfg.get("deta_early", 0.004)
    snaps = [round(x, 6) for x in cfg.get("snaps", [])]
    diag_u = [round(x, 6) for x in cfg.get("diag_u", [])]
    res_u = [round(x, 6) for x in cfg.get("res_u", [])]
    # ---- absorber schedule
    ug = np.round(np.arange(0.0, uend + 0.0100001, 0.01), 6)
    rx = np.concatenate([np.arange(0.3, tb.R, 0.005), np.arange(tb.R, 200.0, 0.01)])
    rtp = np.empty(len(ug))
    for i, u in enumerate(ug):
        Wp = tb.U_at(u, rx) + 6 * xi * float(hl(u)) ** 2 * rx ** 2
        jm, jt = first_min_max(Wp, rx, 0.3)
        rtp[i] = rx[jt] if jt is not None else 150.0
    rc_s = np.maximum.accumulate(np.minimum(np.maximum(np.maximum(Rc0, fcap * rtp), 10 * hl(ug)), 160.0))
    hle = float(hl(uend))
    rmax = float(rc_s[-1]) + max(w0, w1 * hle) + 3 * max(1.0, hle)
    N = int(math.ceil(rmax / dr))
    r = dr * np.arange(1, N + 1)
    Gg = tb.G_on(r)
    dGg = np.diff(Gg, axis=0)
    r2 = r * r
    r4q = r2 * r2 / 4

    def Vraw(u):
        k, t = tb.uidx(u)
        return Gg[k] + t * dGg[k] + tb.lamL(u) * r4q - (1 - 6 * xi) * float(hl(u)) ** 2 * r2

    def pot(u):
        V = Vraw(u)
        rc = float(np.interp(u, ug, rc_s))
        jc = min(int(rc / dr) - 1, N - 3)
        h = float(hl(u))
        ell, wc = ell0 * max(1.0, h), max(w0, w1 * h)
        Vc, Vp = V[jc], min((V[jc + 1] - V[jc - 1]) / (2 * dr), 0.0)
        x = r[jc + 1:] - r[jc]
        V[jc + 1:] = Vc + Vp * ell * (1 - np.exp(-x / ell))
        return V, W0 * np.minimum(1.0, x / wc) ** 2, jc, rc

    # ---- initial state (u = 0)
    V0 = Vraw(0.0)
    jm, jt = first_min_max(V0, r, 0.3)
    r_min, r_top0 = r[jm], r[jt]

    def box(rw):
        n = int(rw / dr)
        w, v = eigh_tridiagonal(1.0 / dr ** 2 + V0[:n], np.full(n - 1, -0.5 / dr ** 2), select="i",
                                select_range=(0, 0))
        return float(w[0]), v[:, 0], n
    E, v, n = box(r_top0 + 1.5)
    r_exit = None
    for _ in range(3):
        jx = jt + int(np.argmax(V0[jt:] <= E))
        r_exit = r[jx]
        E, v, n = box(1.2 * r_exit)
    Q0 = E
    st = cfg.get("state", "b12")
    om = math.sqrt((V0[jm + 1] - 2 * V0[jm] + V0[jm - 1]) / dr ** 2)
    if st == "b12":
        psi0 = np.zeros(N)
        psi0[:n] = v
    else:
        fac = {"gauss05": 0.5, "gauss1": 1.0, "gauss2": 2.0}[st]
        sig = fac / math.sqrt(2 * om)
        psi0 = np.exp(-(r - r_min) ** 2 / (4 * sig ** 2))
    psi0 = psi0 / math.sqrt(np.sum(psi0 ** 2) * dr)
    if psi0[jm] < 0:
        psi0 = -psi0
    kin, off = 1.0 / dr ** 2, -0.5 / dr ** 2

    def resonance(u, Qg, psi):
        V, Wt, jc, rc = pot(u)
        dgl = (kin + V).astype(complex)
        dgl[jc + 1:] -= 1j * Wt
        H = sps.diags([np.full(N - 1, off), dgl, np.full(N - 1, off)], [-1, 0, 1], format="csc")
        vals, vecs = spla.eigs(H, k=6, sigma=Qg - 2e-4j, which="LM")
        _, jtt = first_min_max(Vraw(u), r, 0.3)
        jtt = jtt if jtt is not None else jc
        cand = []
        for i in range(len(vals)):
            ph = vecs[:, i]
            loc = np.sum(np.abs(ph[:jtt]) ** 2) / np.sum(np.abs(ph) ** 2)
            if loc > 0.5 and abs(vals[i].real - Qg) < 0.3:
                cand.append((abs(vals[i] - Qg), i))
        i = min(cand)[1]
        ph = vecs[:, i] / np.sqrt(np.sum(vecs[:, i] ** 2) * dr)
        c = np.sum(ph * psi) * dr
        return complex(vals[i]), float(abs(c) ** 2)

    recs, res_rec, snapf = [], [], {}
    Qg = Q0

    def diagnostics(u, eta, psi):
        V = Vraw(u)
        _, jtt = first_min_max(V, r, 0.3)
        rc = float(np.interp(u, ug, rc_s))
        jc = min(int(rc / dr) - 1, N - 3)
        p2 = np.abs(psi) ** 2 * dr
        cs = np.cumsum(p2)
        return dict(u=u, eta=eta, hL=float(hl(u)), rc=rc, N_tot=float(cs[-1]), N_rc=float(cs[jc]),
                    r_top=(float(r[jtt]) if jtt is not None else None),
                    P_in_top=(float(cs[jtt]) if jtt is not None else None))

    psi = psi0.astype(complex)
    u, eta, nstep = 0.0, 0.0, 0
    if 0.0 in res_u:
        E0, w0r = resonance(0.0, Qg, psi)
        res_rec.append(dict(u=0.0, E=E0, w=w0r))
        Qg = E0.real
    recs.append(diagnostics(0.0, 0.0, psi))
    marks = sorted(set([m_ for m_ in diag_u + snaps + res_u + [round(uend, 6)] if m_ > 1e-9]))
    mi = 0
    while u < uend - 1e-12:
        h = float(hl(u))
        dmax = deta_early if eta < eta_early else min(deta0, cE / h ** 4)
        du = h * dmax
        while mi < len(marks) and marks[mi] <= u + 1e-12:
            mi += 1
        nxt = marks[mi]
        if u + du >= nxt - 1e-12:
            du = nxt - u
        u1 = u + du
        de = (Fint(u1) - Fint(u)) / HLB
        V, Wt, jc, rc = pot(0.5 * (u + u1))
        z = 0.5j * de
        dg = 1.0 + z * (kin + V - Q0)
        dg[jc + 1:] += 0.5 * de * Wt
        b = z * off
        rhs = (2.0 - dg) * psi
        rhs[1:] -= b * psi[:-1]
        rhs[:-1] -= b * psi[1:]
        bv = np.full(N - 1, b, complex)
        out = zgtsv(bv, dg, bv.copy(), rhs)
        if out[4] != 0:
            raise RuntimeError("zgtsv info=%d" % out[4])
        psi = out[3]
        u, eta, nstep = u1, eta + de, nstep + 1
        if abs(u - nxt) < 1e-9:
            u = nxt
            if u in diag_u or u in snaps or u in res_u or u == round(uend, 6):
                recs.append(diagnostics(u, eta, psi))
            if u in res_u:
                Eu, wu = resonance(u, Qg, psi)
                res_rec.append(dict(u=u, E=Eu, w=wu))
                Qg = Eu.real
            if u in snaps:
                rcs = float(np.interp(u, ug, rc_s))
                jcs = min(int(rcs / dr) - 1, N - 3)
                fn = os.path.join(RUNS, "%s_snap%.2f.npz" % (cfg["tag"], u))
                np.savez(fn, psi=psi[:jcs + 1], dr=dr, u=u, rc=rcs, N_tot=np.sum(np.abs(psi) ** 2) * dr)
                snapf["%.2f" % u] = fn
    return dict(cfg=cfg, Q0=Q0, r_min=r_min, r_top0=r_top0, r_exit=r_exit, omega=om, N=N, rmax=rmax,
                nstep=nstep, rc_sched={"%.2f" % x: float(np.interp(x, ug, rc_s)) for x in
                                       (0, 3, 4, 4.5, 5, 5.5, 6, 6.5) if x <= uend + 1e-9},
                records=recs, resonances=res_rec, snaps=snapf, t_tdse=time.time() - t0)


# ============================================================================================ Wigner cells
def wigner_cells(psi, dr, u, nyc=400, npc=300):
    h = float(hl(u))
    chirp = h * math.tanh(2 * u)
    n = len(psi)
    r = dr * np.arange(1, n + 1)
    rc = r[-1]
    lt = 0.5 * max(1.0, h)
    win = np.ones(n)
    s = r > rc - lt
    win[s] = np.cos(0.5 * np.pi * (r[s] - (rc - lt)) / lt) ** 2
    norm_pre = float(np.sum(np.abs(psi) ** 2) * dr)
    phi = np.concatenate([[0.0 + 0.0j], math.sqrt(h) * np.exp(-0.5j * chirp * r * r) * psi * win])
    dy = dr / h
    norm_post = float(np.sum(np.abs(phi) ** 2) * dy)
    nf = 1 << int(math.ceil(math.log2(4 * len(phi))))
    A = np.abs(np.fft.fft(phi, nf)) ** 2
    pk = 2 * np.pi * np.fft.fftfreq(nf, d=dy)
    o = np.argsort(pk)
    cp = np.cumsum(A[o])
    cp /= cp[-1]
    p_lo = float(pk[o][np.searchsorted(cp, 1e-9)])
    p_hi = float(pk[o][min(np.searchsorted(cp, 1 - 1e-9), nf - 1)])
    pad = 0.05 * (p_hi - p_lo)
    p_lo, p_hi = p_lo - pad, p_hi + pad
    pabs = max(abs(p_lo), abs(p_hi))
    yext = len(phi) * dy
    Dy = yext / nyc
    q = max(1, min(int(math.floor(math.pi / (2 * 1.3 * dy * pabs))), int(math.floor(Dy / (2 * dy)))))
    phf = phi[::q]
    dyf = q * dy
    ny = len(phf)
    yv = dyf * np.arange(ny)
    nfft = 1 << int(math.ceil(math.log2(4 * ny)))
    dp = math.pi / (nfft * dyf)
    pn = np.fft.fftfreq(nfft) * nfft * dp
    Dp = (p_hi - p_lo) / npc
    jcell = np.clip(np.floor((pn - p_lo) / Dp).astype(np.int64), 0, npc - 1)
    outside = (pn < p_lo) | (pn >= p_hi)
    icell = np.minimum((yv / Dy).astype(np.int64), nyc - 1)
    Wc = np.zeros(nyc * npc)
    wout = 0.0
    blk = max(1, (1 << 21) // nfft)
    for k0 in range(0, ny, blk):
        ks = np.arange(k0, min(ny, k0 + blk))
        mmax = int(np.max(np.minimum(ks, ny - 1 - ks)))
        mm = np.arange(-mmax, mmax + 1)
        Kp = ks[:, None] + mm[None, :]
        Km = ks[:, None] - mm[None, :]
        val = (Kp < ny) & (Km >= 0)
        C = np.where(val, np.conj(phf[np.clip(Kp, 0, ny - 1)]) * phf[np.clip(Km, 0, ny - 1)], 0.0)
        Cf = np.zeros((len(ks), nfft), complex)
        Cf[:, mm % nfft] = C
        Wr = np.fft.ifft(Cf, axis=1).real * dyf
        wout += float(Wr[:, outside].sum())
        idx = (icell[ks][:, None] * npc + jcell[None, :]).ravel()
        Wc += np.bincount(idx, weights=Wr.ravel(), minlength=nyc * npc)
    Wc = Wc.reshape(nyc, npc)
    tot = float(Wc.sum())
    if tot != 0:
        Wc *= norm_post / tot          # tiny subsampling correction (reported)
    yc = (np.arange(nyc) + 0.5) * Dy
    pc = p_lo + (np.arange(npc) + 0.5) * Dp
    neg = float(-Wc[Wc < 0].sum())
    return dict(W=Wc, yc=yc, pc=pc, M=h ** 3, norm_pre=norm_pre, norm_post=norm_post, sum_raw=tot,
                w_outside_p=wout, neg_weight=neg, q=q, p_range=[p_lo, p_hi], y_ext=yext, ny=ny, nfft=nfft)


# ============================================================================================ classical stage
def classical(tb, xi, presc, u0, cells, du=0.004, Ucl=30.0, wmin=1e-12):
    Yg, Pg = np.meshgrid(cells["yc"], cells["pc"], indexing="ij")
    sel = np.abs(cells["W"]) > wmin
    y = Yg[sel].astype(float)
    v = Pg[sel] / cells["M"]
    w = cells["W"][sel]
    n = len(y)
    stat = np.zeros(n, np.int8)
    lam0 = tb.lamL(u0)
    yb0 = math.sqrt(12 * xi / abs(lam0)) if lam0 < 0 else 50.0
    Yrun = max(40.0, 3.0 * yb0)
    ubchk = max(u0 + 1.5, 7.5)
    ygrid = np.linspace(1e-3, 400.0, 40000)

    def Vq(yy, u):
        if presc == "L":
            return 6 * xi * yy ** 2 + tb.lamL(u) * yy ** 4 / 4
        s_ = yy * yy + 1 + 1 / float(hl(u)) ** 2
        return 6 * xi * yy ** 2 + tb.lam(LN_H + 0.5 * np.log(s_)) * yy ** 4 / 4

    def barrier(u):
        V = Vq(ygrid, u)
        _, jt = first_min_max(V, ygrid, 0.0)
        if jt is None:
            return float("inf"), float("inf")
        return float(ygrid[jt]), float(V[jt])

    def acc(yy, vv, u):
        h = float(hl(u))
        k, t = tb.uidx(u)
        rr = h * yy
        gp = (1 - t) * tb.Gpx(k, rr) + t * tb.Gpx(k + 1, rr)
        F = 12 * xi * yy + gp / h ** 3
        if presc == "L":
            F = F + tb.lamL(u) * yy ** 3
        else:
            s_ = yy * yy + 1 + 1 / h ** 2
            ln = LN_H + 0.5 * np.log(s_)
            F = F + tb.lam(ln) * yy ** 3 + 0.25 * yy ** 5 * tb.beta(ln) / s_
        return -3 * math.tanh(2 * u) * vv - F

    def rk4(yy, vv, u, h_):
        k1y, k1v = vv, acc(yy, vv, u)
        k2y = vv + 0.5 * h_ * k1v
        k2v = acc(yy + 0.5 * h_ * k1y, k2y, u + 0.5 * h_)
        k3y = vv + 0.5 * h_ * k2v
        k3v = acc(yy + 0.5 * h_ * k2y, k3y, u + 0.5 * h_)
        k4y = vv + h_ * k3v
        k4v = acc(yy + h_ * k3y, k4y, u + h_)
        return yy + h_ / 6 * (k1y + 2 * k2y + 2 * k3y + k4y), vv + h_ / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

    u = u0
    act = np.arange(n)
    nst = 0
    tdec = np.full(n, np.nan)
    while act.size and u < u0 + Ucl - 1e-12:
        ya, va = y[act].copy(), v[act].copy()
        sm = ya < 0.4
        if sm.any():
            ys, vs = ya[sm], va[sm]
            for i in range(20):
                ys, vs = rk4(ys, vs, u + i * du / 20, du / 20)
                ng = ys < 0
                ys[ng], vs[ng] = -ys[ng], -vs[ng]
            ya[sm], va[sm] = ys, vs
        bg = ~sm
        if bg.any():
            ya[bg], va[bg] = rk4(ya[bg], va[bg], u, du)
        ng = ya < 0
        ya[ng], va[ng] = -ya[ng], -va[ng]
        u += du
        nst += 1
        y[act], v[act] = ya, va
        run = (ya > Yrun) | ~np.isfinite(ya) | ~np.isfinite(va)
        stat[act[run]] = 2
        tdec[act[run]] = u
        if u >= ubchk - 1e-9 and nst % 10 == 0:
            yb, Vb = barrier(u)
            E = 0.5 * va ** 2 + Vq(ya, u)
            bnd = (~run) & (ya < yb) & (E < 0.85 * Vb)
            stat[act[bnd]] = 1
            tdec[act[bnd]] = u
        act = act[stat[act] == 0]
    und = act.copy()
    if und.size:
        yb, Vb = barrier(u)
        stat[und] = np.where(y[und] < yb, 1, 2)
    yb, Vb = barrier(u0)
    return dict(presc=presc, u0=u0, P_bound=float(w[stat == 1].sum()), P_run_cl=float(w[stat == 2].sum()),
                W_undecided=float(w[und].sum()) if und.size else 0.0, n_traj=int(n), n_undecided=int(und.size),
                Yrun=Yrun, yb_u0=yb, Vb_u0=Vb, lamL_u0=lam0, u_end=u, steps=nst)


# ============================================================================================ jobs
def tagof(model, mt, xi, var="main"):
    return "%s_%.1f_xi%.4f_%s" % (model, mt, xi, var)


def job(cfg):
    t0 = time.time()
    fn = os.path.join(RUNS, cfg["tag"] + ".json")
    if os.path.exists(fn) and not cfg.get("force"):
        return cfg["tag"], "skip"
    try:
        res = tdse(cfg)
        cl = {}
        if res["snaps"]:
            tb = Table(cfg["model"], cfg["mt"])
            for us, f in res["snaps"].items():
                d = np.load(f)
                clist = cfg.get("cells", [(400, 300)]) if abs(float(d["u"]) - 6.0) < 1e-6 else [(320, 240)]
                for (nyc, npc) in clist:
                    tw = time.time()
                    cells = wigner_cells(d["psi"], float(d["dr"]), float(d["u"]), nyc, npc)
                    tw = time.time() - tw
                    for presc in ("phi", "L"):
                        tc = time.time()
                        c = classical(tb, float(cfg["xi"]), presc, float(d["u"]), cells)
                        c.update(norm_pre=cells["norm_pre"], norm_post=cells["norm_post"],
                                 sum_raw=cells["sum_raw"], w_outside_p=cells["w_outside_p"],
                                 neg_weight=cells["neg_weight"], q=cells["q"], p_range=cells["p_range"],
                                 N_tot=float(d["N_tot"]), rc=float(d["rc"]), t_wig=tw, t_cl=time.time() - tc)
                        cl["usw=%s,%s,%dx%d" % (us, presc, nyc, npc)] = c
        res["classical"] = cl
        res["time_s"] = time.time() - t0
        write_json(fn, res)
        msg = " ".join("%s:%.4f" % (k, v["P_bound"]) for k, v in cl.items())
        return cfg["tag"], "ok %.0fs %s" % (time.time() - t0, msg)
    except Exception:
        write_json(os.path.join(RUNS, cfg["tag"] + ".err.json"), dict(cfg=cfg, tb=traceback.format_exc()))
        return cfg["tag"], "ERR " + traceback.format_exc().splitlines()[-1]


def run_pool(cfgs):
    log("pool start: %d jobs" % len(cfgs))
    with ProcessPoolExecutor(max_workers=NWORK) as ex:
        futs = {ex.submit(job, c): c["tag"] for c in cfgs}
        for f in as_completed(futs):
            tag, msg = f.result()
            log("  %s %s" % (tag, msg))


DIAG = [0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
XI_MU = {172.2: [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, XI6],
         172.6: [0.04, 0.06, 0.10, 0.11, 0.12, 0.13, 0.14, XI6]}
XI_C144 = {172.6: [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, XI6],
           172.2: [0.02, 0.03, 0.04, 0.05, 0.06, XI6],
           173.0: [0.08, 0.10, 0.11, 0.12, 0.13, 0.14, XI6]}


def cfgs_item1():
    out = []
    for st in ("b12", "gauss1", "gauss05", "gauss2"):
        out.append(dict(model="MU", mt=172.6, xi=XI6, state=st, u_end=5.0, snaps=[], tag="item1_MU_172.6_" + st,
                        diag_u=list(np.round(np.arange(0.125, 5.0001, 0.125), 6)), res_u=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                        eta_early=(0.0 if st == "b12" else 50.0)))
    return out


def cfgs_main():
    out = []
    for mt, xs in XI_MU.items():
        for xi in xs:
            c = dict(model="MU", mt=mt, xi=xi, u_end=6.5, snaps=[5.5, 6.0, 6.5], diag_u=DIAG, tag=tagof("MU", mt, xi))
            if mt == 172.6 and xi in (0.04, 0.12):
                c["cells"] = [(400, 300), (600, 450)]
            out.append(c)
    return out


def cfgs_variants():
    out = []
    V = dict(dr5=dict(dr=0.005), dt=dict(cE=5e-5, deta0=0.01), cap2=dict(fcap=2.0), capW=dict(W0=12.0, w1=12.0))
    for name, kw in V.items():
        for xi in ((0.04, 0.12, XI6) if name == "dr5" else (0.04, 0.12)):
            c = dict(model="MU", mt=172.6, xi=xi, u_end=6.0, snaps=[6.0], diag_u=DIAG, tag=tagof("MU", 172.6, xi, name))
            c.update(kw)
            out.append(c)
    return out


def cfgs_c144():
    out = []
    for mt, xs in XI_C144.items():
        for xi in xs:
            if mt == 172.6:
                out.append(dict(model="C144", mt=mt, xi=xi, u_end=6.5, snaps=[5.5, 6.0, 6.5], diag_u=DIAG,
                                tag=tagof("C144", mt, xi)))
            else:
                out.append(dict(model="C144", mt=mt, xi=xi, u_end=6.0, snaps=[6.0], diag_u=DIAG,
                                tag=tagof("C144", mt, xi)))
    return out


# ============================================================================================ post
def xi_threshold(xs, Ps, frac=0.99):
    xs, Ps = np.asarray(xs, float), np.asarray(Ps, float)
    o = np.argsort(xs)
    xs, Ps = xs[o], Ps[o]
    tgt = frac * Ps[-1]
    below = [k for k in range(len(xs) - 1) if Ps[k] < tgt]
    if not below:
        return dict(xi_c=None, note="all grid points >= target")
    k = max(below)
    x1, x2, p1, p2 = xs[k], xs[k + 1], Ps[k], Ps[k + 1]
    lin = x1 + (tgt - p1) * (x2 - x1) / (p2 - p1)
    try:
        pc = PchipInterpolator(xs, Ps)
        xx = np.linspace(x1, x2, 20001)
        pv = pc(xx)
        j = int(np.argmax(pv >= tgt))
        pch = float(xx[j])
    except Exception:
        pch = None
    return dict(xi_c=float(lin), xi_c_pchip=pch, bracket=[float(x1), float(x2)], target=float(tgt))


def post():
    S = {}
    # ---- item 1
    P = json.load(open(os.path.join(TABDIR, "poles_xi16.json"), encoding="utf-8"))
    un = np.array(P["u"])
    g = np.array(P["mt=172.6"]["gamma"])
    spl = CubicSpline(un, np.log(g))
    uf = np.linspace(0, 5.0, 40001)
    I = cumulative_trapezoid(np.exp(spl(uf)) / hl(uf), uf, initial=0.0)
    tail = float(g[-1] * eta_rem(5.0))
    it1 = dict(Phaz_u5=float(math.exp(-I[-1])), Phaz_inf=float(math.exp(-I[-1] - tail)), gamma_u0=float(g[0]),
               Q_u0=float(P["mt=172.6"]["Q"][0]))
    g2 = np.array(P["mt=172.2"]["gamma"])
    spl2 = CubicSpline(un, np.log(g2))
    I2 = cumulative_trapezoid(np.exp(spl2(uf)) / hl(uf), uf, initial=0.0)
    it1["Phaz_inf_172.2"] = float(math.exp(-I2[-1] - g2[-1] * eta_rem(5.0)))
    for st in ("b12", "gauss1", "gauss05", "gauss2"):
        fn = os.path.join(RUNS, "item1_MU_172.6_%s.json" % st)
        if not os.path.exists(fn):
            continue
        d = json.load(open(fn, encoding="utf-8"))
        w0 = d["resonances"][0]["w"]
        rows = []
        for rec in d["records"]:
            uu = rec["u"]
            ph = float(math.exp(-np.interp(uu, uf, I)))
            rows.append(dict(u=uu, P_in_top=rec["P_in_top"], N_rc=rec["N_rc"],
                             ratio_in=(rec["P_in_top"] / (w0 * ph) if rec["P_in_top"] else None),
                             ratio_rc=rec["N_rc"] / (w0 * ph)))
        proj = []
        for rr in d["resonances"]:
            uu = rr["u"]
            ph = float(math.exp(-np.interp(uu, uf, I)))
            gs = float(math.exp(spl(uu)))
            gc = -2 * rr["E"][1]
            proj.append(dict(u=uu, w=rr["w"], ratio_w=rr["w"] / (w0 * ph), gamma_cap=gc, gamma_siegert=gs,
                             cap_vs_siegert=gc / gs - 1, Q_cap=rr["E"][0]))
        last = d["records"][-1]
        Pinf = last["P_in_top"] * math.exp(-tail)
        it1[st] = dict(w0=w0, Q0=d["Q0"], r_min=d["r_min"], r_top0=d["r_top0"], r_exit=d["r_exit"],
                       omega=d["omega"], P_in_top_u5=last["P_in_top"], P_inf=Pinf,
                       P_inf_over_Phaz_inf_minus1=Pinf / it1["Phaz_inf"] - 1,
                       P_inf_over_w0Phaz_inf_minus1=Pinf / (w0 * it1["Phaz_inf"]) - 1,
                       max_dev_ratio_in_u_ge_0p5=max(abs(x["ratio_in"] - 1) for x in rows if x["u"] >= 0.5 and x["ratio_in"]),
                       ratio_in_u5=rows[-1]["ratio_in"], projections=proj, rows=rows)
    S["item1"] = it1
    # ---- item 2/3 + variants + C144
    allres = {}
    for fn in sorted(os.listdir(RUNS)):
        if fn.endswith(".json") and not fn.endswith(".err.json") and not fn.startswith("item1"):
            d = json.load(open(os.path.join(RUNS, fn), encoding="utf-8"))
            allres[fn[:-5]] = d
    tabs, xic = {}, {}
    for key, d in allres.items():
        parts = key.split("_")
        if len(parts) != 4:
            continue
        model, mt, xis, var = parts
        xi = float(xis[2:])
        for ck, c in d["classical"].items():
            usw, presc, cells = ck.split(",")
            k2 = "%s|%s|%s|%s|%s|%s" % (model, mt, var, usw, presc, cells)
            tabs.setdefault(k2, {})[round(xi, 4)] = dict(P=c["P_bound"], Prun=c["P_run_cl"], und=c["W_undecided"],
                                                         Nrc=c["norm_pre"], neg=c["neg_weight"])
    for k2, row in tabs.items():
        xs = sorted(row)
        if round(XI6, 4) in row and len(xs) >= 3:
            xic[k2] = xi_threshold(xs, [row[x]["P"] for x in xs])
    S["P_surv"] = {k: {("%.4f" % x): v for x, v in sorted(r.items())} for k, r in tabs.items()}
    S["xi_c"] = xic
    S["tdse_losses"] = {k: dict(records=[(r["u"], r["N_tot"], r["N_rc"]) for r in d["records"]],
                                nstep=d["nstep"], N=d["N"], t=d.get("time_s"), rc=d["rc_sched"])
                        for k, d in allres.items()}
    write_json(os.path.join(OUT, "r15_summary.json"), S)
    # compact print
    print("ITEM1:", json.dumps(clean({k: (v if not isinstance(v, dict) else
                                         {kk: vv for kk, vv in v.items() if kk not in ("rows", "projections")})
                                     for k, v in it1.items()}), indent=0)[:3000])
    for st in ("b12", "gauss1"):
        if st in it1:
            print(st, "proj:", [(p["u"], round(p["ratio_w"], 6), round(p["cap_vs_siegert"], 5)) for p in it1[st]["projections"]])
    for k2 in sorted(tabs):
        row = tabs[k2]
        print(k2, " ".join("%.4f:%.4f" % (x, row[x]["P"]) for x in sorted(row)),
              "| xi_c=", (None if k2 not in xic else (xic[k2].get("xi_c"), xic[k2].get("xi_c_pchip"))))


def smoke():
    c = dict(model="C144", mt=172.6, xi=0.10, u_end=5.6, snaps=[5.6], diag_u=[1.0, 4.0, 5.0], dr=0.005,
             res_u=[0.0], tag="smoke_C144", cells=[(120, 90)], force=True)
    t0 = time.time()
    tb = Table("C144", 172.6)
    print("C144 lam nodes vs lamL:", np.max(np.abs(np.array([tb.lamL(x) for x in tb.u]) - tb.lamn)))
    print("C144 U(r=1.22,u=0)=", np.interp(1.22, tb.r, tb.G[0] + tb.lamn[0] * tb.r ** 4 / 4), "slope", tb.slope[[0, -1]])
    tag, msg = job(c)
    print(tag, msg, "%.0fs" % (time.time() - t0))
    d = json.load(open(os.path.join(RUNS, "smoke_C144.json"), encoding="utf-8"))
    print(json.dumps(clean({k: d[k] for k in ("Q0", "r_min", "r_top0", "r_exit", "omega", "N", "nstep", "rc_sched",
                                               "resonances", "records", "classical")}), indent=0)[:4000])


class LfrzTable:
    """diagnostic prescription: lambda(1/L) frozen at u = ucap in the classical stage (instead of running on)."""
    def __init__(self, tb, ucap=7.0):
        self.tb, self.ucap = tb, ucap

    def __getattr__(self, k):
        return getattr(self.tb, k)

    def lamL(self, u):
        return self.tb.lamL(min(u, self.ucap))


RCDIR = os.path.join(OUT, "reclass")
os.makedirs(RCDIR, exist_ok=True)


def job2(cfg):
    if cfg.get("kind") != "reclass":
        return job(cfg)
    t0 = time.time()
    fn = os.path.join(RCDIR, cfg["tag"] + ".json")
    if os.path.exists(fn):
        return cfg["tag"], "skip"
    try:
        mdl = cfg.get("model", "MU")
        d = np.load(os.path.join(RUNS, "%s_snap%.2f.npz" % (tagof(mdl, cfg["mt"], cfg["xi"]), cfg["usw"])))
        tb = LfrzTable(Table(mdl, cfg["mt"]), 7.0)
        cells = wigner_cells(d["psi"], float(d["dr"]), float(d["u"]), 400, 300)
        c = classical(tb, float(cfg["xi"]), "L", float(d["u"]), cells)
        c.update(cfg=cfg, time_s=time.time() - t0)
        write_json(fn, c)
        return cfg["tag"], "ok %.0fs Lfrz7 P=%.4f" % (time.time() - t0, c["P_bound"])
    except Exception:
        return cfg["tag"], "ERR " + traceback.format_exc().splitlines()[-1]


def cfgs_phase2():
    out = []
    for xi in XI_C144[172.6]:
        out.append(dict(model="C144", mt=172.6, xi=xi, u_end=6.0, snaps=[5.5, 6.0], diag_u=DIAG,
                        tag=tagof("C144", 172.6, xi)))
    for mt, xs in ((172.2, [0.04, 0.05, 0.06, 0.07, XI6]), (172.6, [0.04, 0.06, 0.10, 0.11, 0.12, XI6])):
        for xi in xs:
            out.append(dict(kind="reclass", mt=mt, xi=xi, usw=6.0, tag="RC_MU_%.1f_xi%.4f" % (mt, xi)))
    for mt in (172.2, 173.0):
        for xi in XI_C144[mt]:
            out.append(dict(model="C144", mt=mt, xi=xi, u_end=6.0, snaps=[6.0], diag_u=DIAG,
                            tag=tagof("C144", mt, xi)))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "phase3":
    log("pool3 start (C144 Lfrz7 reclass)")
    c3 = [dict(kind="reclass", model="C144", mt=172.6, xi=xi, usw=6.0, tag="RC_C144_172.6_xi%.4f" % xi)
          for xi in (0.05, 0.06, 0.07, 0.08, 0.09, XI6)]
    with ProcessPoolExecutor(max_workers=NWORK) as ex:
        for tag_, msg_ in ex.map(job2, c3):
            log("  %s %s" % (tag_, msg_))
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "phase2":
    log("pool2 start")
    with ProcessPoolExecutor(max_workers=NWORK) as ex:
        futs = {ex.submit(job2, c): c["tag"] for c in cfgs_phase2()}
        for f in as_completed(futs):
            tag_, msg_ = f.result()
            log("  %s %s" % (tag_, msg_))
    sys.exit(0)

if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else "post"
    if md == "tables":
        build_tables()
    elif md == "smoke":
        smoke()
    elif md == "phase1":
        run_pool(cfgs_item1() + cfgs_main() + cfgs_variants())
    elif md == "c144":
        run_pool(cfgs_c144())
    elif md == "post":
        post()
