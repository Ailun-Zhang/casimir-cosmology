"""Optional item 3: Gamma(lambda) on a dense grid in x = |lambda|^{-1/2} (Method B, e = 1) and the
local slope  -d ln Gamma / d x.  Also the Born-Oppenheimer WKB barrier action for comparison:
    V_BO(r) = (3/2) e r + lambda r^4 / 4,   S(E) = int_{r1}^{r2} sqrt(2 (V_BO - E)) dr,
    -d ln Gamma/dx ~ 2 dS/dx (E = eps(lambda)), and the E -> 0, lambda -> 0 limit 2 S -> sqrt(2) pi e x.
Writes ../slope_results.json.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
import mpmath as mp
from scipy.integrate import quad
from scipy.optimize import brentq
from ho_mp import resonance_mp

HERE = os.path.dirname(os.path.abspath(__file__))
NR, NS, TH = 120, 60, np.pi / 8


def gam(lam, e=1.0):
    Ed, Em = resonance_mp(e, lam, NR, 1.0, NS, 1.0, TH, 0.0, sigma=1.9 * e ** (2 / 3))
    return float(Em.real), float(-2 * Em.imag)


def wkb_S(lam, E, e=1.0):
    V = lambda r: 1.5 * e * r + 0.25 * lam * r**4
    rm = (1.5 * e / abs(lam)) ** (1 / 3)
    if V(rm) <= E:
        return float("nan")
    r1 = brentq(lambda r: V(r) - E, 1e-12, rm)
    r2 = brentq(lambda r: V(r) - E, rm, 10 * rm + 10)
    val, _ = quad(lambda r: np.sqrt(max(2 * (V(r) - E), 0.0)), r1, r2, limit=200)
    return val


def main():
    out = {"basis": [NR, NS, TH], "dense": [], "slopes": []}
    xs = np.arange(1.0, 10.01, 0.25)
    for x in xs:
        lam = -1.0 / x**2
        eps, G = gam(lam)
        out["dense"].append(dict(x=x, lam=lam, eps=eps, Gamma=G, lnGamma=float(np.log(G))))
        print(f"x={x:6.3f} lam={lam:.6f} eps={eps:.12f} Gamma={G:.10e}", flush=True)
    dx = 0.05
    for lam0 in [-0.11174, -0.05, -0.03, -0.02, -0.01003]:
        x0 = abs(lam0) ** -0.5
        lnG = {}
        epsd = {}
        for k in (-2, -1, 0, 1, 2):
            x = x0 + k * dx
            eps, G = gam(-1.0 / x**2)
            lnG[k] = np.log(G)
            epsd[k] = eps
        d1 = (lnG[-2] - 8 * lnG[-1] + 8 * lnG[1] - lnG[2]) / (12 * dx)
        d1_3pt = (lnG[1] - lnG[-1]) / (2 * dx)
        # BO-WKB comparison: 2 dS/dx at fixed E = eps(lam0) and with E = eps(lam)
        S = {k: wkb_S(-1.0 / (x0 + k * dx) ** 2, epsd[0]) for k in (-1, 1)}
        Se = {k: wkb_S(-1.0 / (x0 + k * dx) ** 2, epsd[k]) for k in (-1, 1)}
        row = dict(lam=lam0, x=x0, slope=-d1, slope_3pt=-d1_3pt,
                   wkb_2dSdx_fixedE=(2 * (S[1] - S[-1]) / (2 * dx)),
                   wkb_2dSdx=(2 * (Se[1] - Se[-1]) / (2 * dx)),
                   wkb_2S=2 * wkb_S(lam0, epsd[0]), sqrt2pi_x=np.sqrt(2) * np.pi * x0)
        out["slopes"].append(row)
        print(row, flush=True)
    json.dump(out, open(os.path.join(HERE, "..", "slope_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
