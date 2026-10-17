# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_common.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Common pieces of the full-dimensional validation (D123/T14a): v11 couplings, MC+CW matching, Dirichlet wall at the gauge-invariant radius r = |phi| = r_w, radial guide function, same-wall 2D model.  Paper: PRD App. H2. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

T14a / C151 common layer (fulldim-dmc, DNEC v12 round 3, 2026-10-14).

* 2D (r,|w|) operator of D112 (2D_ex, 2D_G) via the copied C140_beyond_bo.Model/build (unchanged physics).
* Same-wall 2D eigenvalues: sparse 2nd-order FD with Dirichlet at r = r_w (and s = S_w), Richardson h = 0.08/0.04/0.02.
* Radial guide f(r) for the full-dimensional DMC: ground state of a 1D radial problem
      -1/2 (f'' + 3 f'/r) + U_g(r) f = eps f ,  f(r_w) = 0 ,
  with U_g = lam r^4/4 + dV(r) + C0 + B_x + dE_dmc + (3/2) Om_eff  (adiabatic 2D_ex channel without B_w), so that the
  local energy of psi_T = G(x,w) f(r) is  E_L = 1/2 sum alpha - 1/2 sum alpha^2 y^2 + eps - U_g(r) + V(y)  (smooth at r_w).
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import C140_beyond_bo_t14a as B  # [v12-package] renamed T14a copy  # noqa: E402  (copy; RES redirected to T14a run dir)

ROOT = B.ROOT
RES = os.path.join(ROOT, _zp.w("r3/fulldim_dmc/run01"))
SHARDS = os.path.join(RES, "shards")
OLDSUM = os.path.join(B.RES_OLD, "C140_summary.json")
V11 = B.V11
C72_LAM0, C72_SE = 4.4478058, 0.0013152717
Q2D_LAM0 = 4.450863          # D112 2D_ex (Numerov), FD Richardson 4.450855
# D112 Sec. 3.5 hazard conversion
DLNI_DMT = {"0.9": 5.717, "0.5": 3.513}
_M = None


def model():
    global _M
    if _M is None:
        _M = B.Model()
    return _M


def clean(o):
    return B.clean(o)


def wjson(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def match_of(lam):
    return "L0" if lam == 0.0 else "MC"


def potentials(var, lam, r, tilt=0.0):
    """U1 (incl. 3/(8r^2)) and om2 of the D112 operator; U_ad = U1 + 1.5 Om_eff (lowest adiabatic w channel)."""
    U1, om2 = B.build(model(), var, match_of(lam), lam, np.asarray(r, float), tilt)
    return U1, om2


def u_adiabatic(var, lam, r):
    U1, om2 = potentials(var, lam, r)
    return U1 + 1.5 * np.sqrt(om2)


# ------------------------------------------------------------------------------------------------ same-wall 2D FD
def lap(n, h):
    return sparse.diags([np.full(n - 1, -0.5 / h ** 2), np.full(n, 1.0 / h ** 2), np.full(n - 1, -0.5 / h ** 2)],
                        [-1, 0, 1])


def fd2d_wall(var, lam, rw, h, Sw=12.0, sigma=None, k=4, extra_U=None, tilt=0.0):
    """lowest eigenvalues of the 2D operator with Dirichlet walls at r = rw, s = Sw (u = r^{3/2} f, v = s chi)."""
    nr = int(round(rw / h)) - 1
    assert abs((nr + 1) * h - rw) < 1e-9, (rw, h)
    r = h * np.arange(1, nr + 1)
    U1, om2 = potentials(var, lam, r, tilt)
    if extra_U is not None:
        U1 = U1 + extra_U(r)
    ns = int(round(Sw / h)) - 1
    s = h * np.arange(1, ns + 1)
    H = (sparse.kron(lap(nr, h), sparse.identity(ns)) + sparse.kron(sparse.identity(nr), lap(ns, h))
         + sparse.diags((U1[:, None] + 0.5 * om2[:, None] * s[None, :] ** 2).ravel())).tocsc()
    if sigma is None:
        sigma = float(np.min(U1 + 1.5 * np.sqrt(om2))) - 0.05
    vals, vecs = eigsh(H, k=k, sigma=sigma, which="LM")
    o = np.argsort(vals)
    return vals[o], vecs[:, o], r, s


def richardson(e):
    """e = [E(0.08), E(0.04), E(0.02)] -> h^2,h^4 Richardson."""
    return (64 * e[2] - 20 * e[1] + e[0]) / 45, (4 * e[2] - e[1]) / 3


# ------------------------------------------------------------------------------------------------ radial guide
def radial_guide(lam, rw, h=0.001, var="2D_ex"):
    """ground state of -1/2 u'' + [3/(8r^2) + U_g] u = eps u on (0, rw), u(rw) = 0; returns dict with spline data."""
    n = int(round(rw / h)) - 1
    r = h * np.arange(1, n + 1)
    U1, om2 = potentials(var, lam, r)
    Ufull = U1 + 1.5 * np.sqrt(om2)                  # includes 3/(8r^2)
    d = 1.0 / h ** 2 + Ufull
    e = np.full(n - 1, -0.5 / h ** 2)
    w, v = eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
    u = v[:, 0] * np.sign(v[n // 3, 0])
    f = u / r ** 1.5
    # f(0) from even fit on the first points
    sel = r < 0.05
    A = np.vstack([np.ones(sel.sum()), r[sel] ** 2, r[sel] ** 4]).T
    c = np.linalg.lstsq(A, f[sel], rcond=None)[0]
    rr = np.concatenate([[0.0], r, [rw]])
    ff = np.concatenate([[c[0]], f, [0.0]])
    ff = ff / ff.max()
    Ug = Ufull - 3.0 / (8 * r * r)                   # potential seen by f in the 4D radial Laplacian
    Ug0 = np.polyfit(r[sel] ** 2, Ug[sel], 2)[-1]
    UU = np.concatenate([[Ug0], Ug, [Ug[-1] + (Ug[-1] - Ug[-2])]])
    return dict(r=rr, f=ff, U=UU, eps=float(w[0]), rw=rw, h=h, var=var, lam=lam)


class Guide:
    """vectorised evaluation of ln f, f'/f and U_g on walkers."""

    def __init__(self, g):
        self.rw = g["rw"]
        self.eps = g["eps"]
        r, f = g["r"], g["f"]
        self.sf = CubicSpline(r, f, bc_type=((1, 0.0), "not-a-knot"))
        self.sdf = self.sf.derivative()
        self.sU = CubicSpline(r, g["U"], bc_type=((1, 0.0), "not-a-knot"))

    def eval(self, rr):
        """returns ok (inside wall, f>0), ln f, f'/f, and the exact radial local-energy part -1/2 (f''+3f'/r)/f of the
        SPLINE guide (fully consistent with the drift and the Metropolis ratio)."""
        f = self.sf(rr)
        df = self.sdf(rr)
        d2f = self.sf(rr, 2)
        ok = (rr < self.rw) & (f > 0)
        fs = np.where(ok, f, 1.0)
        rs = np.maximum(rr, 1e-9)
        lap = np.where(rr > 1e-7, d2f + 3 * df / rs, 4 * d2f)
        return (ok, np.where(ok, np.log(fs), -np.inf), np.where(ok, df / fs, 0.0),
                np.where(ok, -0.5 * lap / fs, 0.0))


def guide_consistency(g, n=400):
    """max |-1/2 (f''+3f'/r)/f + U - eps| on r in [0.05, rw - 0.15] (spline vs ODE)."""
    G = Guide(g)
    rr = np.linspace(0.05, g["rw"] - 0.15, n)
    f = G.sf(rr)
    lapf = G.sf(rr, 2) + 3 * G.sf(rr, 1) / rr
    res = -0.5 * lapf / f + G.sU(rr) - g["eps"]
    return float(np.max(np.abs(res)))


def dv_table(lam, rmax, h=0.002):
    """dV_MC(r) spline (0 for lam = 0 local model)."""
    rr = np.arange(0.0, rmax + 3 * h, h)
    rr[0] = 1e-6
    if lam == 0.0:
        vals = np.zeros_like(rr)
    else:
        vals = B.dVm("MC", rr, lam)
    return CubicSpline(rr, vals)
