"""Method A production runs: smooth ECS on a 2D finite-difference grid + flux formula.

Writes ../methodA_results.json (all runs).  Runs variants in a small process pool.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, time, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "methodA_results.json")

# case: (e, lam, R0, Rf_list, w, theta, Smax)
CASES = {
    "e=1,lam=-1": (1.0, -1.0, 2.6, [1.6, 2.0], 1.5, 0.6, 10.0),
    "e=1,lam=-0.11174": (1.0, -0.11174, 4.6, [3.6, 4.0], 2.0, 0.5, 10.0),
    "e=1,lam=-0.03": (1.0, -0.03, 6.8, [5.8, 6.2], 2.0, 0.5, 10.0),
    "e=1,lam=-0.01003": (1.0, -0.01003, 10.0, [9.0, 9.4], 2.0, 0.5, 10.0),
    "e=0.873,lam=-0.02286387": (0.873, -0.03 * 0.873**2, 7.1, [6.0, 6.5], 2.0, 0.5, 11.0),
    "e=1,lam=-0.05": (1.0, -0.05, 6.0, [5.0, 5.4], 2.0, 0.5, 10.0),
    "e=1,lam=-0.02": (1.0, -0.02, 8.2, [7.0, 7.5], 2.0, 0.5, 10.0),
}


def variants(key):
    e, lam, R0, Rf, w, th, Smax = CASES[key]
    base = dict(e=e, lam=lam, R0=R0, Rf_list=Rf, w=w, theta=th, Smax=Smax, p=4, mask_c=50.0,
                a=6.0, kind="erf")
    tail = 2.0 if lam == -1.0 else 2.5
    out = []
    hs = [0.05, 0.04, 0.03] + ([0.025, 0.02] if lam == -1.0 else [])
    for h in hs:
        out.append(dict(base, h=h, Rmax=R0 + w + tail, tag=f"h={h}"))
    out.append(dict(base, h=0.04, p=5, Rmax=R0 + w + tail, tag="h=0.04,p=5"))
    out.append(dict(base, h=0.04, theta=th - 0.15, Rmax=R0 + w + tail + 1.0, tag=f"h=0.04,theta={th-0.15:.2f}"))
    dR = 0.4 if lam == -1.0 else 0.7
    out.append(dict(base, h=0.04, R0=R0 + dR, Rmax=R0 + dR + w + tail, tag=f"h=0.04,R0+{dR}"))
    out.append(dict(base, h=0.04, Smax=Smax + 2.0, mask_c=70.0, Rmax=R0 + w + tail, tag="h=0.04,Smax+2,mask70"))
    for v in out:
        v["key"] = key
    return out


def work(v):
    import ecs_grid as eg
    t = time.time()
    G = eg.ECSGrid(v["e"], v["lam"], v["h"], v["Rmax"], v["h"], v["Smax"], v["R0"], v["w"], v["theta"],
                   p=v["p"], mask_c=v["mask_c"], kind=v["kind"], a=v["a"])
    sigma = 1.9 * v["e"] ** (2.0 / 3.0)
    E, x, hist, res = G.solve(sigma, nit=8)
    fl = G.flux_gamma(x, v["Rf_list"])
    r = dict(v)
    r.update(n=int(G.n), eps=float(E.real), Gamma_eig=float(-2 * E.imag), resid=float(res),
             flux=[dict(R=float(R), J=float(J), N=float(N), Gamma_flux=float(Gm)) for (R, J, N, Gm) in fl],
             seconds=time.time() - t)
    return r


def main(keys=None, nproc=3):
    keys = keys or list(CASES)
    jobs = [v for k in keys for v in variants(k)]
    # largest jobs first
    jobs.sort(key=lambda v: -(v["Rmax"] / v["h"]) * (v["Smax"] / v["h"]))
    results = []
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))["runs"]
        except Exception:
            results = []
    done = {(r["key"], r["tag"]) for r in results}
    jobs = [v for v in jobs if (v["key"], v["tag"]) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(work, jobs):
            results.append(r)
            fl = " ".join(f"{f['Gamma_flux']:.10e}" for f in r["flux"])
            print(f"{r['key']:28s} {r['tag']:24s} n={r['n']:7d} eps={r['eps']:.13f} "
                  f"Geig={r['Gamma_eig']:.10e} Gflux={fl} res={r['resid']:.1e} {r['seconds']:.0f}s",
                  flush=True)
            json.dump({"method": "A: smooth ECS (erf switch) on 2D finite-difference grid, flux formula",
                       "runs": results}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    keys = sys.argv[1:] or None
    main(keys)
