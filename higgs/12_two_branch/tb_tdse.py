# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/two_branch/tb_tdse.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Two-branch (two-time) analysis by TDSE through the turn on the D111 potential tables: GH decoherence functional, linear (Goldstein-Page / Margenau-Hill) two-time probability P2_lin = P1^2/w0, Frechet bounds.  Paper: PRD App. H3. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C159 / D130 (two-branch, T7): two-arrow joint survival in the D111 1D zero-mode model (r frame, xi_H = 1/6).

Turnaround state psi0 (real) at u = 0; H(eta) real and even in eta  =>  psi(-eta) = conj(psi(eta)).
Branch events, Heisenberg picture at the turnaround:
    E+ = U(T,0)^+ P U(T,0),   E- = U(-T,0)^+ P U(-T,0) = K E+ K   (K = complex conjugation),
    P = 1[r < r_top(u_T)]   (same 'in well' cut as D111 P_in).
Everything needed follows from two forward, well-localised evolutions (CAP is then exact):
    A = P psi(T),   P psi(-T) = conj(A),   X = P U(T,-T) conj(A)   (CN through the turnaround with H(|eta|))
Gell-Mann-Hartle time-ordered set C_ab = P_b(T) P_a(-T), D(h,h') = <C_h' psi0 | C_h psi0>:
    D[(in,in),(in,in)]   = |X|^2 = G
    D[(out,in),(out,in)] = |A - X|^2
    D[(in,in),(out,in)]  = <A - X | X> = K - G           (b = out block: same modulus, opposite sign, by unitarity)
Kirkwood-Dirac / linear (Goldstein-Page) value: q(in,in) = <psi0| E+ E- |psi0> = <A|X> = K (real for real psi0).
Usage: python -B tb_tdse.py smoke | full
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
sys.dont_write_bytecode = True
import json
import math
import time
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import eigsh
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hr_core as C  # noqa: E402
from hr_potential import PotentialTable  # noqa: E402

ROOT = _zp.WORK
TAB = os.path.join(ROOT, _zp.w("r2/higgs_realtime/run01/tables"))
OUT = os.path.join(ROOT, _zp.w("r3/two_branch/run01"))
UN = np.round(np.arange(0, 5.0001, 0.25), 6)
UT = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
BASE = dict(dr=0.01, rmax=16.0, r_c=10.0, cap0=20.0, deta=0.025)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hazard(un, gam, uq):
    """copy of hr_part1.hazard: exp(-int gamma d eta) from CAP pole nodes."""
    spl = CubicSpline(un, np.log(gam))
    uf = np.linspace(0, max(uq), 40001)
    f = np.exp(spl(uf)) / C.hL(uf)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(uf))])
    return np.exp(-np.interp(uq, uf, I))


def make_states(g, lm0, vecs, poles, U0, wanted):
    """same constructions as hr_part1 (D111 sec 1.3)."""
    x, dx = g.x, g.dx
    st = {}
    a = np.real(vecs[0]).copy()
    a[x > lm0["r_exit"]] = 0
    st["a_gamow_real"] = a
    for f in (1.0, 1.2):
        nw = int(np.searchsorted(x, f * lm0["r_exit"]))
        sub = C.Grid(dx, nw * dx)
        Hs = C.h_sparse(sub, 0.5, U0[:nw]).real
        ev, evec = eigsh(Hs, k=8, sigma=poles[0].real, which="LM")
        ov = [abs(np.sum(vecs[0][:nw] * evec[:, j]) * dx) ** 2 / (np.sum(evec[:, j] ** 2) * dx) for j in range(len(ev))]
        j = int(np.argmax(ov))
        b = np.zeros(g.N)
        b[:nw] = evec[:, j]
        st["b_wall%.1f" % f] = b
    i = lm0["i_min"]
    om = math.sqrt(max((U0[i + 1] - 2 * U0[i] + U0[i - 1]) / dx ** 2, 1e-6))
    sho = 1 / math.sqrt(2 * om)
    for f in (0.5, 1.0, 2.0):
        st["c_gauss%.1f" % f] = np.exp(-(x - lm0["r_min"]) ** 2 / (4 * (f * sho) ** 2)) * (1 - np.exp(-(x / 0.3) ** 2))
    return {k: v for k, v in st.items() if wanted is None or k in wanted}


def run(args):
    mt, opt, tag, wanted, uts = args
    t0 = time.time()
    o = dict(BASE, **opt)
    tp = os.path.join(TAB, "U_table_MU_mt%.1f.npz" % mt)
    tab = PotentialTable(tp)
    g = C.Grid(o["dr"], o["rmax"])
    fr = C.RFrame(tab, g, 1.0 / 6, r_c=o["r_c"], cap0=o["cap0"])
    x, dx = g.x, g.dx
    lm = C.landmarks(x, fr.U_real(0.0), 4.4)
    E, v = C.resonance(g, fr.V(0.0), 4.4 - 0.0005j, k=8, inner=lm["i_top"])
    poles, vecs = [E], [v]
    for u in UN[1:]:
        E, v = C.resonance(g, fr.V(u), poles[-1], k=1)
        poles.append(E)
        vecs.append(v)
    poles = np.array(poles)
    gam = -2 * poles.imag
    lms = [C.landmarks(x, fr.U_real(u), poles[i].real) for i, u in enumerate(UN)]
    states = make_states(g, lms[0], vecs, poles, fr.U_real(0.0), wanted)
    # common time grid (eta >= 0), potential cached at the CN midpoints |eta| = (k - 1/2) h
    eta5 = float(C.eta_of_u(5.0))
    n = int(math.ceil(eta5 / o["deta"]))
    h = eta5 / n
    kT = {ut: int(round(float(C.eta_of_u(ut)) / h)) for ut in uts}
    kmax = max(kT.values())
    Vr = np.empty((kmax, g.N))
    for k in range(1, kmax + 1):
        Vr[k - 1] = fr.V(float(C.u_of_eta((k - 0.5) * h))).real
    cap = fr.cap
    E0 = float(poles[0].real)
    Ph = hazard(UN, gam, np.array([0.0] + list(uts)))[1:]
    g5, erem = float(gam[-1]), C.eta_rem(5.0)

    def step(psi, k):
        return C.cn_step(psi, g, 0.5, Vr[k - 1] - 1j * cap - E0, h)

    out = dict(mt=mt, tag=tag, opt=o, h=h, n=n, kT={str(k): v for k, v in kT.items()},
               poles=[[float(p.real), float(p.imag)] for p in poles], u_nodes=UN.tolist(),
               r_top_nodes=[l["r_top"] for l in lms], landmarks_u0=lms[0], table=tp, table_sha256=sha(tp),
               code_sha256={f: sha(os.path.join(HERE if os.path.exists(os.path.join(HERE, f)) else _zp.ZMLIB, f)) for f in ("tb_tdse.py", "hr_core.py", "hr_potential.py")},
               gamma5=g5, eta_rem5=erem, states={})
    inv = {v: u for u, v in kT.items()}
    for nm, s0 in states.items():
        psi = s0.astype(complex)
        psi /= math.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        w0 = float(abs(np.sum(vecs[0] * psi) * dx) ** 2)
        snaps = {}
        p = psi.copy()
        for k in range(1, kmax + 1):
            p = step(p, k)
            if k in inv:
                snaps[inv[k]] = p.copy()
        recs = []
        for ut in uts:
            kk = kT[ut]
            iu = int(np.argmin(np.abs(UN - ut)))
            itop = int(np.searchsorted(x, lms[iu]["r_top"]))
            A = np.zeros(g.N, complex)
            A[:itop] = snaps[ut][:itop]
            q = np.conj(A)
            for j in range(1, 2 * kk + 1):
                q = step(q, (kk - j + 1) if j <= kk else (j - kk))
            X = np.zeros(g.N, complex)
            X[:itop] = q[:itop]
            P1 = float(np.sum(np.abs(A) ** 2) * dx)
            G = float(np.sum(np.abs(X) ** 2) * dx)
            K = complex(np.sum(np.conj(A) * X) * dx)
            Y2 = float(np.sum(np.abs(A - X) ** 2) * dx)
            Doff = K - G
            recs.append(dict(
                u_T=ut, eta_T=kk * h, P1=P1, P_haz=float(Ph[list(uts).index(ut)]),
                w_T=float(abs(np.sum(vecs[iu] * snaps[ut]) * dx) ** 2),
                G_GH=G, K_re=K.real, K_im=K.imag, Y2=Y2, Doff_re=Doff.real, Doff_im=Doff.imag,
                eps=abs(Doff) / math.sqrt(max(G * Y2, 1e-300)),
                marg_plus_in=G + Y2, marg_viol=G + Y2 - P1,
                R_K=K.real * w0 / P1 ** 2, R_G=G * w0 ** 2 / P1 ** 3,
                frechet_lo=max(0.0, 2 * P1 - 1), GH_below_frechet=bool(G < 2 * P1 - 1),
                wX=float(abs(np.sum(vecs[iu] * q) * dx) ** 2), norm_X_full=float(np.sum(np.abs(q) ** 2) * dx)))
        inf = None
        r5 = [r for r in recs if abs(r["u_T"] - 5.0) < 1e-9]
        if r5:
            r5 = r5[0]
            f = math.exp(-g5 * erem)
            P1i, Ki, Gi = r5["P1"] * f, r5["K_re"] * f ** 2, r5["G_GH"] * f ** 3
            inf = dict(P1=P1i, P2_KD=Ki, G_GH=Gi, P1sq=P1i ** 2, P1sq_over_w0=P1i ** 2 / w0,
                       frechet_lo=max(0.0, 2 * P1i - 1), frechet_hi=P1i, KD_over_P1sq=Ki / P1i ** 2,
                       KD_over_P1sq_over_w0=Ki * w0 / P1i ** 2, G_over_P1cube_w0sq=Gi * w0 ** 2 / P1i ** 3)
        out["states"][nm] = dict(w0=w0, recs=recs, inf=inf)
    out["time_s"] = time.time() - t0
    os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
    with open(os.path.join(OUT, "shards", "tdse_%s_mt%.1f.json" % (tag.replace(":", "_").replace(",", "_").replace("=", ""), mt)),
              "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=1)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    t0 = time.time()
    if mode == "smoke":
        r = run((172.6, dict(deta=0.05), "smoke", ["a_gamow_real", "c_gauss1.0"], [1.0, 5.0]))
        for nm, s in r["states"].items():
            print(nm, "w0=%.5f" % s["w0"])
            for rec in s["recs"]:
                print("  u_T=%.1f P1=%.5f Phaz=%.5f K=%.6f%+.1ej G=%.6f eps=%.4f R_K=%.5f R_G=%.5f margviol=%.5f"
                      % (rec["u_T"], rec["P1"], rec["P_haz"], rec["K_re"], rec["K_im"], rec["G_GH"], rec["eps"],
                         rec["R_K"], rec["R_G"], rec["marg_viol"]))
            print("  inf:", json.dumps(s["inf"]))
        print("time %.1f s" % r["time_s"])
        sys.exit(0)
    jobs = [(mt, {}, "base", None, UT) for mt in (172.6, 172.2, 172.0)]
    var = [dict(deta=0.0125), dict(r_c=11.5, cap0=40.0), dict(dr=0.007), dict(cap0=10.0)]
    jobs += [(172.6, v, "var:" + ",".join("%s=%s" % kv for kv in v.items()), ["a_gamow_real", "c_gauss1.0"], [1.0, 3.0, 5.0])
             for v in var]
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(run, jobs))
    meta = dict(code=os.path.abspath(__file__), base=BASE, time_s=time.time() - t0, xi_H=1.0 / 6, workers=2)
    with open(os.path.join(OUT, "tdse_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, runs=res), f, indent=1)
    print("done %.1f s" % (time.time() - t0))
