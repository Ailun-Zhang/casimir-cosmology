# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_3d.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] 3D (r, s, zeta) model with an SU(2) collective width coordinate (TDVP mass): gamma_3D/gamma_2D.  Paper: PRD H2.

T14a item 3(b): 3D (r, s=|w|, zeta) model -- D112 2D_ex plus an SU(2) collective coordinate (fulldim-dmc, 2026-10-14).

Collective coordinate: global width scale of the D108/D112 conditional SU(2) Gaussian,
    precisions (a*, a*, k*) -> (a*, a*, k*) e^q,  zeta = e^{-q/2}  (zeta = rms-width scale, zeta = 1 at the D108 optimum),
    mean <x^3_i> = -beta* w_i unchanged.
Potential E(r,s,zeta) from the D112 closed forms with a,k -> a e^q, k e^q (Gaussian moments exact):
    C0(zeta) = C0 + (3/4)(2a+k+k beta^2)(zeta^-2 - 1) + (3/16) r^2 (2g^2/a+g^2/k)(zeta^2 - 1)
               + [(3/4)g^2/a^2 + (3/2)g^2/(a k)](zeta^4 - 1)            (angular 3(k-a)^2/(akr^2) is scale invariant)
    Om_eff^2(zeta) = Om_eff^2 + (2 g^2 beta^2/a)(zeta^2 - 1) + (4 a beta^2/r^2 + k beta'^2/2)(zeta^-2 - 1)
Kinetic: TDVP breathing mass (thawed-Gaussian; equals GOA h2/gamma^2 for quadratic H, GOA gamma = G_qq = 9/8):
    T_zeta = -(1/(2 mu)) d^2/dzeta^2,  mu(r) = sum_j sigma_j^2 = 3 (1/a* + 1/(2k*))     (zeta is the flat coordinate)
Zero-point bookkeeping (the 2D_ex surface already contains the SU(2) zero-point energy):
    'ref'  : subtract eps(r) = E0_zeta[T + E(r,s_ref,.)] - E(r,s_ref,1), s_ref^2 = min(3/(2 Om), 5)  (keeps the adiabatic
             s-dependence of the zeta relaxation, i.e. the SU(2) width responding to w)
    'local': subtract eps(r,s) at every s (BO limit = 2D_ex exactly; pure non-adiabatic test)
No q-r derivative coupling (condition; <d_q phi|d_r phi> = (1/8) sum_j d_r ln P_j kept out, see D123).
Numerics: r FD (h, Dirichlet at R), s sine-DVR (N_s on (0,S)), zeta sine-DVR (N_z on (0,Z)); 2D reference = zeta frozen at 1
on the SAME r/s grids; CAP -i eta (r-7)^2 (r>7) as T14a_gamma; ratio gamma_3D/gamma_2D at equal eta.
    python -B T14a_3d.py test | run
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
from scipy.interpolate import CubicSpline
import T14a_common as T

sys.stdout.reconfigure(encoding="utf-8")
G2, GY = T.V11["g2"], T.V11["gY"]
OUT = os.path.join(T.RES, "T14a_3d.json")


def sine_dvr(n, L):
    j = np.arange(1, n + 1)
    x = j * L / (n + 1)
    U = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(j, j) * np.pi / (n + 1))
    Tk = U @ np.diag(0.5 * (j * np.pi / L) ** 2) @ U.T
    return x, Tk


class Prof:
    def __init__(self):
        M = T.model()
        rf, q = M.rf, M.q
        self.M = M
        self.sa = CubicSpline(rf, q["a"]); self.sk = CubicSpline(rf, q["k"])
        self.sb = CubicSpline(rf, q["beta"]); self.sbp = CubicSpline(rf, q["betap"])

    def pars(self, r):
        return self.sa(r), self.sk(r), self.sb(r), self.sbp(r)


def potential(P, lam, r, s, z, var="2D_ex"):
    """V[r, s, z] including 3/(8r^2), lam r^4/4, dV, dE_dmc, B_x (z = 1 -> D112 2D_ex)."""
    U1, om2 = T.potentials(var, lam, r)                     # U1 = 3/8r^2 + lam r^4/4 + dV + C0 + Bx + corr
    a, k, b, bp = P.pars(r)
    A = (0.75 * (2 * a + k + k * b * b))[:, None, None]
    Bm = (3.0 / 16 * r * r * (2 * G2 ** 2 / a + G2 ** 2 / k))[:, None, None]
    Cq = (0.75 * G2 ** 2 / a ** 2 + 1.5 * G2 ** 2 / (a * k))[:, None, None]
    D2 = (2 * G2 ** 2 * b * b / a)[:, None, None]
    Dm2 = (4 * a * b * b / (r * r) + 0.5 * k * bp * bp)[:, None, None]
    zz = z[None, None, :]
    c0z = U1[:, None, None] + A * (zz ** -2 - 1) + Bm * (zz ** 2 - 1) + Cq * (zz ** 4 - 1)
    om2z = om2[:, None, None] + D2 * (zz ** 2 - 1) + Dm2 * (zz ** -2 - 1)
    ss = s[None, :, None]
    return c0z + 0.5 * om2z * ss * ss, om2, (a, k)


def build(lam, h=0.05, R=11.0, Ns=22, S=12.5, Nz=12, Z=2.6, zpe="ref", cap=True):
    P = Prof()
    nr = int(round(R / h)) - 1
    r = h * np.arange(1, nr + 1)
    s, Ts = sine_dvr(Ns, S)
    z, Tz0 = sine_dvr(Nz, Z)
    V, om2, (a, k) = potential(P, lam, r, s, z)
    mu = 3 * (1 / a + 1 / (2 * k))
    # zero-point bookkeeping
    iz1 = None
    eps = np.zeros((nr, Ns))
    sref2 = np.minimum(1.5 / np.sqrt(om2), 5.0)
    for i in range(nr):
        Tz = Tz0 / mu[i]
        if zpe == "ref":
            Vzi = potential(P, lam, r[i:i + 1], np.sqrt(sref2[i:i + 1]), z)[0][0, 0]
            Vri = potential(P, lam, r[i:i + 1], np.sqrt(sref2[i:i + 1]), np.array([1.0]))[0][0, 0, 0]  # E(r_i,s_ref,1)
            e0 = np.linalg.eigvalsh(Tz + np.diag(Vzi))[0]
            eps[i, :] = e0 - Vri
        else:
            for j in range(Ns):
                e0 = np.linalg.eigvalsh(Tz + np.diag(V[i, j]))[0]
                V1 = potential(P, lam, r[i:i + 1], s[j:j + 1], np.array([1.0]))[0][0, 0, 0]
                eps[i, j] = e0 - V1
    V3 = V - eps[:, :, None]
    # 2D reference on the same grids: zeta frozen at 1
    V2 = potential(P, lam, r, s, np.array([1.0]))[0][:, :, 0]
    Tr = T.lap(nr, h)
    Is, Iz = sparse.identity(Ns), sparse.identity(Nz)
    # 3D: T_r x I x I + sum_i e_i e_i^T x (Ts x I + I x Tz/mu_i) + diag V3
    blocks = []
    for i in range(nr):
        blocks.append(sparse.kron(sparse.csr_matrix(Ts), Iz) + sparse.kron(Is, sparse.csr_matrix(Tz0 / mu[i])))
    H3 = (sparse.kron(Tr, sparse.identity(Ns * Nz)) + sparse.block_diag(blocks)
          + sparse.diags(V3.ravel())).tocsc()
    H2 = (sparse.kron(Tr, Is) + sparse.kron(sparse.identity(nr), sparse.csr_matrix(Ts)) + sparse.diags(V2.ravel())).tocsc()
    Wr = np.where(r > 7.0, (r - 7.0) ** 2, 0.0)
    W3 = sparse.diags(np.repeat(Wr, Ns * Nz)); W2 = sparse.diags(np.repeat(Wr, Ns))
    info = dict(lam=lam, h=h, R=R, Ns=Ns, S=S, Nz=Nz, Z=Z, zpe=zpe, n3=H3.shape[0], n2=H2.shape[0],
                eps_at={f"{x}": float(eps[int(np.argmin(np.abs(r - x))), 0]) for x in (0.5, 1.0, 2.0, 3.0, 4.5)},
                mu_at={f"{x}": float(mu[int(np.argmin(np.abs(r - x)))]) for x in (0.5, 1.0, 2.0, 3.0, 4.5)})
    return H3, H2, W3, W2, r, s, z, info


def bound(lam, rw=9.0, **kw):
    H3, H2, _, _, r, s, z, info = build(lam, R=rw, **kw)
    e3 = eigsh(H3, k=2, sigma=4.3, which="LM", return_eigenvectors=False)
    e2 = eigsh(H2, k=2, sigma=4.3, which="LM", return_eigenvectors=False)
    return dict(info, Q3=float(min(e3)), Q2=float(min(e2)), dQ_3D_minus_2D=float(min(e3) - min(e2)))


def cap_pair(lam, target, etas=np.logspace(-2.0, 0.5, 9), **kw):
    H3, H2, W3, W2, r, s, z, info = build(lam, **kw)
    out = dict(info, rows=[])
    sig2, sig3 = complex(target), complex(target)
    for eta in etas:
        v2 = eigs((H2 - 1j * eta * W2).tocsc(), k=3, sigma=sig2, which="LM", return_eigenvectors=False)
        e2 = v2[int(np.argmin(np.abs(v2 - sig2)))]
        v3 = eigs((H3 - 1j * eta * W3).tocsc(), k=3, sigma=sig3, which="LM", return_eigenvectors=False)
        e3 = v3[int(np.argmin(np.abs(v3 - sig3)))]
        sig2, sig3 = complex(e2), complex(e3)
        out["rows"].append(dict(eta=float(eta), E2=[e2.real, e2.imag], E3=[e3.real, e3.imag],
                                ratio=float(e3.imag / e2.imag)))
    E2 = np.array([complex(*x["E2"]) for x in out["rows"]])
    d = np.gradient(E2, np.log(etas))
    i = int(np.argmin(np.abs(d[1:-1]))) + 1
    x = out["rows"][i]
    out.update(i_opt=i, eta_opt=x["eta"], Q2=x["E2"][0], gamma2=-2 * x["E2"][1], Q3=x["E3"][0], gamma3=-2 * x["E3"][1],
               ratio_3D_2D=x["ratio"], dQ=x["E3"][0] - x["E2"][0])
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    res = T.jload(OUT) if os.path.exists(OUT) else {}
    t0 = time.time()
    if mode == "test":
        o = bound(0.0, rw=9.0, h=0.08, Ns=18, Nz=10)
        print("lam0 test", json.dumps(T.clean(o)), "%.1fs" % (time.time() - t0), flush=True)
        return
    old = T.jload(T.OLDSUM)
    jobs = [("lam0_ref", lambda: bound(0.0, rw=9.0, zpe="ref")),
            ("lam0_local", lambda: bound(0.0, rw=9.0, zpe="local")),
            ("lam0_ref_Nz16", lambda: bound(0.0, rw=9.0, zpe="ref", Nz=16, Z=3.0))]
    for lam in (-0.010, -0.013):
        tg = complex(old["Q"]["2D_ex"][f"{lam}"], -old["gamma"]["2D_ex"][f"{lam}"] / 2)
        jobs.append((f"cap_{lam}_ref", lambda lam=lam, tg=tg: cap_pair(lam, tg, zpe="ref")))
        jobs.append((f"cap_{lam}_local", lambda lam=lam, tg=tg: cap_pair(lam, tg, zpe="local")))
    jobs.append(("cap_-0.01_ref_Nz16_Ns26", lambda: cap_pair(-0.010, complex(old["Q"]["2D_ex"]["-0.01"],
                                                                       -old["gamma"]["2D_ex"]["-0.01"] / 2),
                                                        zpe="ref", Nz=16, Z=3.0, Ns=26)))
    for name, fn in jobs:
        if name in res:
            continue
        t1 = time.time()
        o = fn()
        o["time_s"] = round(time.time() - t1, 1)
        res[name] = o
        T.wjson(OUT, res)
        keys = [k for k in ("Q3", "Q2", "dQ_3D_minus_2D", "ratio_3D_2D", "gamma2", "gamma3", "dQ", "eta_opt", "time_s")
                if k in o]
        print(name, {k: o[k] for k in keys}, flush=True)


if __name__ == "__main__":
    main()
