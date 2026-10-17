# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/C140_cap.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] D112 cross-check of the 2D resonance widths with a complex absorbing potential (CAP).  Paper: PRD App. H2.

C140 independent cross-check (D112 Sec. 3.6; coordinator diagnostics (a)-(c)).  Claude 2026-10-13.

Complex-absorbing-potential (CAP) resonances of the SAME 2D (r, |w|) operator and of the 1D BO operators by sparse
second-order finite differences -- a different discretisation (FD instead of sine-DVR x matrix Numerov) and a different
outgoing-wave treatment (CAP instead of the Siegert WKB condition) from C140_beyond_bo.py.
    H_CAP(eta) = H_FD - i eta (r - R_c)^2 theta(r - R_c)        (acts on r only; w keeps its Dirichlet box)
For each eta on a log grid the eigenvalue nearest the target is tracked; the optimal eta minimises |eta dE/deta|
(Riss-Meyer stabilisation).  Also: probability inside the barrier (r < r_top) of the 2D CAP eigenvector, its
conditional <s^2>(r) and the share of |psi|^2 at s > 0.75 S_max (no leakage along w), and a Dirichlet-wall
stabilisation scan (real eigenvalues vs. outer wall) for the 2D model at lambda = -0.0143.
Usage: python -B C140_cap.py
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
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C140_beyond_bo as B  # noqa: E402

RC, RW, SW = 7.0, 11.0, 12.0
ETAS = np.logspace(-2.5, 0.5, 13)


def lap(n, h):
    return sparse.diags([np.full(n - 1, -0.5 / h ** 2), np.full(n, 1.0 / h ** 2), np.full(n - 1, -0.5 / h ** 2)],
                        [-1, 0, 1])


def hfd(M, var, lam, h, Rw=RW, Sw=SW):
    nr = int(round(Rw / h)) - 1
    r = h * np.arange(1, nr + 1)
    U1, om2 = B.build(M, var, "MC", lam, r)
    if om2 is None:
        return (lap(nr, h) + sparse.diags(U1)).tocsc(), r, None
    ns = int(round(Sw / h)) - 1
    s = h * np.arange(1, ns + 1)
    H = (sparse.kron(lap(nr, h), sparse.identity(ns)) + sparse.kron(sparse.identity(nr), lap(ns, h))
         + sparse.diags((U1[:, None] + 0.5 * om2[:, None] * s[None, :] ** 2).ravel())).tocsc()
    return H, r, s


def task(args):
    var, lam, h, target = args
    t0 = time.time()
    M = B.Model()
    H, r, s = hfd(M, var, lam, h)
    ns = 1 if s is None else len(s)
    W = np.repeat(np.where(r > RC, (r - RC) ** 2, 0.0), ns)
    Es, V = [], []
    sig = complex(target)
    for eta in ETAS:
        A = (H - 1j * eta * sparse.diags(W)).tocsc()
        vals, vecs = eigs(A, k=3, sigma=sig, which="LM")
        j = int(np.argmin(np.abs(vals - sig)))
        Es.append(complex(vals[j]))
        sig = complex(vals[j])
        if s is not None:
            V.append(vecs[:, j])
    Es = np.array(Es)
    lg = np.log(ETAS)
    dE = np.gradient(Es, lg)
    i = int(np.argmin(np.abs(dE[1:-1]))) + 1
    Es_vec = V[i] if s is not None else None
    out = dict(var=var, lam=lam, h=h, target=[target.real, target.imag], E_eta=[[e.real, e.imag] for e in Es],
               eta=ETAS.tolist(), i_opt=i, eta_opt=float(ETAS[i]), Q=float(Es[i].real), gamma=float(-2 * Es[i].imag),
               stab=float(abs(dE[i])), time_s=time.time() - t0)
    if s is not None:
        p2 = np.abs(Es_vec.reshape(len(r), len(s))) ** 2
        rho = p2.sum(1)
        tot = rho.sum()
        out["P_r_lt_5"] = float(rho[r < 5.0].sum() / tot)
        out["P_s_gt_0.75Smax"] = float(p2[:, s > 0.75 * SW].sum() / tot)
        out["cond_s2"] = {f"{x}": float((p2[int(np.argmin(np.abs(r - x)))] * s ** 2).sum()
                                         / p2[int(np.argmin(np.abs(r - x)))].sum()) for x in (0.6, 1.0, 1.5, 2.5, 4.0)}
    return out


def stab_task(args):
    """real Dirichlet-wall stabilisation scan for the 2D model (lowest eigenvalues vs outer wall R_w)."""
    var, lam, h, Rws = args
    M = B.Model()
    res = {}
    for Rw in Rws:
        H, r, s = hfd(M, var, lam, h, Rw=Rw)
        vals = eigsh(H, k=6, sigma=4.25, which="LM", return_eigenvectors=False)
        res[f"{Rw}"] = sorted(vals.tolist())
    return dict(var=var, lam=lam, h=h, levels=res)


def main():
    t0 = time.time()
    S = B.jload("C140_summary.json")
    tasks = []
    for lam in (-0.010, -0.0143):
        for var in ("2D_ex", "1D_ex", "1D_G"):
            q, g = S["Q"][var][f"{lam}"], S["gamma"][var][f"{lam}"]
            for h in ((0.04, 0.03) if var.startswith("2D") else (0.04, 0.02, 0.01)):
                tasks.append((var, lam, h, complex(q, -g / 2)))
    with ProcessPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(task, tasks))
        stab = list(ex.map(stab_task, [("2D_ex", -0.0143, 0.04, [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])]))
    rat = {}
    for lam in (-0.010, -0.0143):
        for h in (0.04, 0.03):
            a = [o for o in outs if o["var"] == "2D_ex" and o["lam"] == lam and o["h"] == h]
            b = [o for o in outs if o["var"] == "1D_ex" and o["lam"] == lam and o["h"] == 0.04]
            if a and b:
                rat[f"{lam}|h2D={h}"] = dict(cap_2D_over_1Dex=a[0]["gamma"] / b[0]["gamma"],
                                            numerov=S["gamma"]["2D_ex"][f"{lam}"] / S["gamma"]["1D_ex"][f"{lam}"])
    B.wjson("C140_cap.json", dict(tasks=outs, ratios=rat, stabilisation=stab, R_c=RC, R_w=RW, S_w=SW,
                                   time_s=time.time() - t0))
    for o in outs:
        print(o["var"], o["lam"], o["h"], "Q=%.6f gamma=%.4e eta_opt=%.3g stab=%.2e" % (o["Q"], o["gamma"], o["eta_opt"],
                                                                                    o["stab"]),
              {k: o[k] for k in ("P_r_lt_5", "P_s_gt_0.75Smax", "cond_s2") if k in o}, flush=True)
    print("ratios", rat, flush=True)
    print("stabilisation", stab, flush=True)


if __name__ == "__main__":
    main()
