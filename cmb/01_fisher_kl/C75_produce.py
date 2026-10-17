# PROVENANCE: 04_calculations/code/20261011_claude_followup/C75/C75_produce.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Covariances of the v11 readability-window scan (stages validate, main, xconv, l40, dirs) with the C59 builder; 668 MB cache (not shipped). Pool size = env DNEC_WORKERS (default 6) or -jN.

C75_produce: torus T+E covariances for the CMB readability window of the zero-mode-corrected fossil.

Question (task C75, 2026-10-11 follow-up): the paper (v10, Sec. IX and App. J) finds that a typical free-field fossil,
|beta_E| = sqrt(2) sigma_E along X_d with sigma_E = 0.0589, is readable (2D >= 9, D = exact KL divergence of the T+E
covariance from that of the cube of equal volume, l <= 30) only for 1.045 < L0/d_LSS < 1.08. A positive zero-mode energy
Q/L = r |E_C| raises sigma_E to 0.0925 (r = 0.25) or 0.1041 (r = 0.30) (C71). This script produces the covariances needed
to redo the window for those widths.

Conventions are those of C59_run's 'base' and 'defm' tasks, unchanged: the covariance code C59_torus.py (imported read-only
from code/20261009_claude_R5/obs_fisher; bytecode writing disabled so nothing is written there), Transfer(30) on the
CAMB v2 transfer table, x_max = 160, smooth taper over the last quarter, taper evaluated at the deformed |k| ('deformed'),
mode list selected with a margin covering the deformation (mode_list(L0, xmax, [M])), L0 = s * 2 chi_*,
k_n = (2 pi / L0) M n with M = expm(-amp X_d) (kmatrix('Xd', amp)), amp = sqrt(2) sigma_E, i.e. sides L0 e^{+sigma},
L0 e^{-sigma}, L0.

Covariances that already exist (this task's cache, or read-only the R5 C59/C70 caches, see C75_common.find) are never
recomputed.

Stages (python C75_produce.py <stage> [<stage> ...]; resumable):
  validate  fresh recomputation of the free fossil (sigma_E = 0.0589) and of the cube at s = 1.07 (own cache only, so it
            is a genuine recomputation; compared with the C59 cache by C75_analyze.py)
  main      cubes on the comparison grid s' = 1.020 ... 1.200 (step 0.005) and at the three circle bounds
            s_c = 0.985 e^{sigma}; fossils for sigma_E in {0.0589, 0.0925, 0.1041} on the fossil grids and at s_c
  xconv     x_max = 240 convergence check of fossil and cube near the upper edges
  l40       l <= 40 covariances (Transfer(40)) of fossil and cube near the upper edges
  dirs      orientation check inside the E doublet: deformation along +X_d2 / -X_d2 with the same |beta|
Output: results/20261011_claude_followup/C75/cache/<key>.npz (packed upper triangles + JSON meta, C59 format).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True          # never write __pycache__ into the read-only R5 code folder
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C75_common import (CACHE, C59_CODE, SIGMAS, TAPER, XMAIN, key, find, pack, fossil_points,  # noqa: E402
                        cube_points)
sys.path.insert(1, str(C59_CODE))
from C59_torus import Transfer, TorusCov, kmatrix  # noqa: E402  (read-only import of the project's covariance code)

_TR = {}


def transfer(lmax):
    if lmax not in _TR:
        _TR[lmax] = Transfer(lmax)
    return _TR[lmax]


def mmat(kind, sig, gen):
    if kind == "base":
        return np.eye(3)
    amp = float(np.sqrt(2.0) * sig)
    if gen == "Xd":
        return kmatrix("Xd", amp)
    if gen == "Xd2p":
        return kmatrix(beta5=[0.0, amp, 0.0, 0.0, 0.0])
    if gen == "Xd2m":
        return kmatrix(beta5=[0.0, -amp, 0.0, 0.0, 0.0])
    raise ValueError(gen)


def run_task(task):
    kind, s, xmax, lmax, sig, gen, own_only = task
    name = key(kind, s, xmax, lmax, sig, gen)
    if find(kind, s, xmax, lmax, sig, gen, own_only=own_only) is not None:
        return name, "cached", 0.0
    fn = CACHE / (name + ".npz")
    t0 = time.perf_counter()
    tr = transfer(lmax)
    tc = TorusCov(tr, lmax)
    L0 = s * 2 * tr.chi
    M = mmat(kind, sig, gen)
    nlist, R = tc.mode_list(L0, xmax, [M])
    C, info = tc.compute(L0, M, xmax, nlist, taper=TAPER, wmode="deformed")
    field = "C0" if kind == "base" else "Cb"
    meta = dict(kind=kind, s=s, L0_Mpc=L0, chi=tr.chi, xmax=xmax, taper=TAPER, wmode="deformed", gen=gen,
                sigma_E=sig, amp=(None if sig is None else float(np.sqrt(2.0) * sig)), M=np.asarray(M).tolist(),
                lmax=lmax, dims=[int(c.shape[0]) for c in C], fields=[field], mode_list_radius=R,
                mode_list_half=int(len(nlist)), modes_half_used=info["modes_half_used"],
                transfer=tr.path, seconds=time.perf_counter() - t0, pid=os.getpid())
    tmp = CACHE / (name + ".tmp.npz")
    np.savez(tmp, meta=json.dumps(meta), **{f"{field}_p{p}": pack(C[p]) for p in (0, 1)})
    os.replace(tmp, fn)
    return name, "done", meta["seconds"]


# upper-edge neighbourhoods used by the robustness stages, chosen from the main-stage results (C75_analyze.log):
# T+E upper edges 1.0784 (free), 1.1194 (r25), 1.1334 (r30) -> grid points 1.08, 1.12, 1.13 next to them
XCONV_POINTS = {"free": [1.08], "r25": [1.12], "r30": [1.13]}
L40_POINTS = {"free": [1.08], "r25": [1.12], "r30": [1.13]}
DIRS_POINTS = {"r25": [1.12], "r30": [1.13]}


def tasks_for(stage):
    T = []
    if stage == "validate":
        T.append(("defm", 1.07, XMAIN, 30, SIGMAS["free"], "Xd", True))
        T.append(("base", 1.07, XMAIN, 30, None, "none", True))
    elif stage == "main":
        for s in cube_points():
            T.append(("base", s, XMAIN, 30, None, "none", False))
        for b, sig in SIGMAS.items():
            for s in fossil_points(b):
                T.append(("defm", s, XMAIN, 30, sig, "Xd", False))
    elif stage == "xconv":
        for b, pts in XCONV_POINTS.items():
            for s in pts:
                T.append(("base", s, 240.0, 30, None, "none", True))
                T.append(("defm", s, 240.0, 30, SIGMAS[b], "Xd", True))
    elif stage == "l40":
        for b, pts in L40_POINTS.items():
            for s in pts:
                T.append(("base", s, XMAIN, 40, None, "none", True))
                T.append(("defm", s, XMAIN, 40, SIGMAS[b], "Xd", True))
    elif stage == "dirs":
        for b, pts in DIRS_POINTS.items():
            for s in pts:
                for g in ("Xd2p", "Xd2m"):
                    T.append(("defm", s, XMAIN, 30, SIGMAS[b], g, True))
    else:
        raise ValueError(stage)
    return T


def cost(task):
    kind, s, xmax, lmax = task[:4]
    return s ** 3 * xmax ** 3 * (lmax / 30.0) ** 4


def main():
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[1:] or ["validate", "main"]
    nproc = int(os.environ.get("DNEC_WORKERS", "6"))  # package: --workers (-jN still overrides)
    if stages and stages[0].startswith("-j"):
        nproc = int(stages.pop(0)[2:])
    seen, todo, n_all = set(), [], 0
    for st in stages:
        block = []
        for t in tasks_for(st):
            k = key(t[0], t[1], t[2], t[3], t[4], t[5])
            if k in seen:
                continue
            seen.add(k)
            n_all += 1
            if find(t[0], t[1], t[2], t[3], t[4], t[5], own_only=t[6]) is None:
                block.append(t)
        block.sort(key=cost, reverse=True)
        todo += block
    print(f"stages {stages}: {n_all} tasks, {len(todo)} to run: " + ", ".join(
        key(*t[:6]) for t in todo), flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    with Pool(min(nproc, len(todo))) as pool:
        for name, status, sec in pool.imap_unordered(run_task, todo):
            print(f"[{time.perf_counter() - t0:8.1f}s] {status} {name} ({sec:.1f}s)", flush=True)
    print(f"total wall time {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
