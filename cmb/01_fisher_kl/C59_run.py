# PROVENANCE: 04_calculations/code/20261009_claude_R5/obs_fisher/C59_run.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Resumable parallel production of torus covariances and finite-difference derivatives (113 tasks: conv, main, full, exact) into r5/obs_fisher/cache (1.3 GB, not shipped). Paper: PRD Sec. X (Fisher).

C59_run: resumable parallel production of torus covariances and finite-difference derivatives.

Every task writes one cache file results/.../obs_fisher/cache/<key>.npz (packed upper triangles,
float64, per parity block) and is skipped if the file exists.  Analysis is done by C59_analyze.py.

  python C59_run.py conv     kmax / taper convergence set (L/dLSS = 0.95, 1.0, 1.2)
  python C59_run.py main     production scan (all L), x_max = XMAIN
  python C59_run.py full     5 shape + 3 rotation generators at L = dLSS (block-structure check)
  python C59_run.py exact    finite deformations (benchmarks A, B) for the quadratic-regime check
Task kinds
  base  : C(0)
  fd    : dC_h = [C(+h)-C(-h)]/2h and dC_2h = [C(+2h)-C(-2h)]/4h, h = 1e-3, for one generator
  fd1   : dC_h only (block-structure check)
  defm  : C(beta) for a finite deformation (along X_d, or a full 5-vector draw)
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
import time
import numpy as np
from C59_torus import OUT, Transfer, TorusCov, kmatrix, SHAPE, ROTATIONS

CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
LMAX = 30
H = 1e-3
XMAIN = 160.0
S_MAIN = [0.9, 0.95, 0.985, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
S_FINE = [1.02, 1.07, 1.15]          # extra points near the detectability crossover
S_CONV = [0.95, 1.0, 1.2]
X_CONV = [80.0, 120.0, 160.0, 240.0]
SIG = {"A": dict(E=0.0589, T=0.0066), "B": dict(E=0.0179, T=0.0069)}


def key(kind, s, xmax, taper=0.25, wmode="deformed", gen="none", tag=""):
    return f"{kind}_s{s:.3f}_x{xmax:g}_t{taper:g}_{wmode[0]}_{gen}{tag}"


def pack(c):
    return c[np.triu_indices(c.shape[0])]


def unpack(v, n):
    c = np.zeros((n, n)); iu = np.triu_indices(n)
    c[iu] = v
    return c + np.triu(c, 1).T


def load(name):
    """Return dict name -> [block0, block1] of full symmetric matrices, plus meta."""
    a = np.load(CACHE / (name + ".npz"))
    meta = json.loads(str(a["meta"]))
    out = {}
    for f in meta["fields"]:
        out[f] = [unpack(a[f"{f}_p{p}"], meta["dims"][p]) for p in (0, 1)]
    return out, meta


def run_task(task):
    kind, s, xmax, taper, wmode, gen, extra = task
    tag = extra.get("tag", "")
    name = key(kind, s, xmax, taper, wmode, gen, tag)
    fn = CACHE / (name + ".npz")
    if fn.exists():
        return name, "cached", 0.0
    t0 = time.perf_counter()
    tr = Transfer(LMAX)
    tc = TorusCov(tr, LMAX)
    L0 = s * 2 * tr.chi
    if kind == "base":
        mats = {"C0": [(np.eye(3), 1.0)]}
    elif kind in ("fd", "fd1"):
        M = lambda a: kmatrix(gen, a)
        mats = {"dCh": [(M(H), 1 / (2 * H)), (M(-H), -1 / (2 * H))]}
        if kind == "fd":
            mats["dC2h"] = [(M(2 * H), 1 / (4 * H)), (M(-2 * H), -1 / (4 * H))]
    elif kind == "defm":
        if "beta5" in extra:
            mats = {"Cb": [(kmatrix(beta5=extra["beta5"]), 1.0)]}
        else:
            mats = {"Cb": [(kmatrix(gen, extra["amp"]), 1.0)]}
    else:
        raise ValueError(kind)
    allM = [m for lst in mats.values() for m, _ in lst]
    nlist, R = tc.mode_list(L0, xmax, allM)
    res = {}
    infos = []
    for f, lst in mats.items():
        acc = None
        for M, coef in lst:
            C, info = tc.compute(L0, M, xmax, nlist, taper=taper, wmode=wmode)
            infos.append(info)
            acc = [coef * c for c in C] if acc is None else [a + coef * c for a, c in zip(acc, C)]
        res[f] = acc
    dims = [int(c.shape[0]) for c in next(iter(res.values()))]
    meta = dict(kind=kind, s=s, L0_Mpc=L0, chi=tr.chi, xmax=xmax, taper=taper, wmode=wmode, gen=gen,
                extra={k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in extra.items()},
                h=H, lmax=LMAX, dims=dims, fields=list(res), mode_list_radius=R,
                mode_list_half=int(len(nlist)), modes_half_used=[i["modes_half_used"] for i in infos],
                seconds=time.perf_counter() - t0, pid=os.getpid())
    arrs = {f"{f}_p{p}": pack(res[f][p]) for f in res for p in (0, 1)}
    tmp = CACHE / (name + ".tmp.npz")
    np.savez(tmp, meta=json.dumps(meta), **arrs)
    os.replace(tmp, fn)
    return name, "done", meta["seconds"]


def cost(task):
    kind, s, xmax = task[:3]
    n = {"base": 1, "fd": 4, "fd1": 2, "defm": 1}[kind]
    return n * s ** 3 * xmax ** 3


def tasks_for(stage):
    T = []
    if stage == "conv":
        for s in S_CONV:
            for x in X_CONV:
                T.append(("base", s, x, 0.25, "deformed", "none", {}))
                for g in ("Xd", "X12"):
                    T.append(("fd", s, x, 0.25, "deformed", g, {}))
        # taper / fixed-list variants at L = dLSS, x = XMAIN
        for taper, wmode in ((0.10, "deformed"), (1e-9, "undeformed"), (0.25, "undeformed")):
            T.append(("base", 1.0, XMAIN, taper, wmode, "none", {}))
            for g in ("Xd", "X12"):
                T.append(("fd", 1.0, XMAIN, taper, wmode, g, {}))
    elif stage == "conv2":
        # matched-circle threshold: x_max convergence of F_E near L = d_LSS
        for s in (0.985, 1.0, 1.05):
            for x in (240.0, 320.0):
                T.append(("base", s, x, 0.25, "deformed", "none", {}))
                T.append(("fd", s, x, 0.25, "deformed", "Xd", {}))
                if x == 240.0:
                    T.append(("fd", s, x, 0.25, "deformed", "X12", {}))
    elif stage == "fine":
        for s in S_FINE:
            T.append(("base", s, XMAIN, 0.25, "deformed", "none", {}))
            for g in ("Xd", "X12"):
                T.append(("fd", s, XMAIN, 0.25, "deformed", g, {}))
            for b, sg in SIG.items():
                T.append(("defm", s, XMAIN, 0.25, "deformed", "Xd",
                          {"amp": float(np.sqrt(2) * sg["E"]), "tag": f"_bench{b}"}))
    elif stage == "main":
        for s in S_MAIN:
            T.append(("base", s, XMAIN, 0.25, "deformed", "none", {}))
            for g in ("Xd", "X12"):
                T.append(("fd", s, XMAIN, 0.25, "deformed", g, {}))
    elif stage == "full":
        for g in SHAPE + ROTATIONS:
            if g not in ("Xd", "X12"):
                T.append(("fd1", 1.0, XMAIN, 0.25, "deformed", g, {}))
    elif stage == "exact":
        rng = np.random.default_rng(20261010)
        z = rng.normal(size=5)
        for s in S_MAIN:
            for b, sg in SIG.items():
                T.append(("defm", s, XMAIN, 0.25, "deformed", "Xd",
                          {"amp": float(np.sqrt(2) * sg["E"]), "tag": f"_bench{b}"}))
        for s in (1.0, 1.2):
            for b, sg in SIG.items():
                beta5 = [float(z[0] * sg["E"]), float(z[1] * sg["E"])] + [float(v * sg["T"]) for v in z[2:]]
                T.append(("defm", s, XMAIN, 0.25, "deformed", "draw", {"beta5": beta5, "tag": f"_bench{b}"}))
    else:
        raise ValueError(stage)
    return T


def main():
    from multiprocessing import Pool
    stages = sys.argv[1:]
    # stages are dispatched in the order given; within a stage, most expensive first
    seen = set(); todo = []; uniq = []
    for st in stages:
        block = []
        for t in tasks_for(st):
            k = key(t[0], t[1], t[2], t[3], t[4], t[5], t[6].get("tag", ""))
            if k in seen:
                continue
            seen.add(k); uniq.append(t)
            if not (CACHE / (k + ".npz")).exists():
                block.append(t)
        block.sort(key=cost, reverse=True)
        todo += block
    print(f"{len(uniq)} tasks, {len(todo)} to run", flush=True)
    t0 = time.perf_counter()
    with Pool(min(int(os.environ.get("DNEC_WORKERS", "8")), max(1, len(todo)))) as pool:  # package: --workers
        for name, status, sec in pool.imap_unordered(run_task, todo):
            print(f"[{time.perf_counter() - t0:8.1f}s] {status} {name} ({sec:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
