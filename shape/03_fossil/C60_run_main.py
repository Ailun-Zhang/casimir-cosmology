# PROVENANCE: 04_calculations/code/20261009_claude_R5/nonlinear_SM/C60_run_main.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Free-field (xi = 0) Wigner-surrogate ensembles through the nonlinear turning-point map (D49b): main 128, hi 2048 (v11 Fig. 5 free-field draws), amplitude/convergence controls. Paper: PRD Sec. VIII D-E, App. app:nonlinear.

C60_run_main: Wigner-surrogate ensembles through the nonlinear turning-point -> outgoing map, SM all-antiperiodic.

Draws (predeclared): rng = default_rng(20261010); Z = rng.standard_normal((128, 10));
    beta0_A = sqrt(k s_A) Z[:, A],  Pi0_A = delta / sqrt(k s_A) Z[:, 5 + A],   A = 0..4  (modes: E, E, T2, T2, T2),
with delta = 1.5897991e-4, s_E = 0.28837 delta, s_T = 0.42083 delta (natural widths).  No rejection.
Sets (tag: description):
  main     N=128, k=1 (natural widths)                           [headline]
  sens3    N=64 (first 64 rows of the same Z), k=3                 [3x natural widths, pure Gaussian state]
  amp3     N=64 (same rows), beta0 and Pi0 multiplied by 3         [supplementary stress test, k_beta=9, k_Pi=9]
  amp5/amp8/amp12  same with multipliers 5, 8, 12                  [supplementary survival-margin scan]
           (generic tags sens<k>, amp<k> are accepted)
  ic_can   first 32 draws of main, canonical-chart momentum        [initial-momentum convention]
  ic_trf   first 32 draws of main, G-trace-free projected momentum [cruder O(beta) convention]
  conv_tol first 16 draws of main, rtol 2e-10 / atol 2e-12
  conv_cut first 16 draws of main, Ewald cut 60
  hi       N=2048, seed 20261011, k=1                              [supplementary high-statistics check]
Per-draw JSON in results/.../draws/<tag>_<i>.json (resumable: existing files are skipped).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import re
import time
import argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shapelib"))  # package: C60 modules
from C60_nonlinear_integrator import *
from C60_twisted_ewald import OUT

SEED = 20261010
SEED_HI = 20261011
DELTA = 1.5897991e-4
S_NAT = np.array([0.28837, 0.28837, 0.42083, 0.42083, 0.42083]) * DELTA
DRAWS = OUT / "draws"
N_MAIN = 128


def z_main():
    return np.random.default_rng(SEED).standard_normal((N_MAIN, 10))


def z_hi(n=2048):
    return np.random.default_rng(SEED_HI).standard_normal((n, 10))


def make_specs(tag):
    z = z_main()
    base = dict(ic="on", rtol=1e-11, atol=1e-13, cut=42.0)

    def spec(i, zrow, ks=1.0, ka=1.0, **kw):
        s = ks * S_NAT
        d = dict(base)
        d.update(kw)
        d.update(tag=tag, i=i, ks=ks, amp=ka,
                 beta0=(ka * np.sqrt(s) * zrow[:5]).tolist(), Pi0=(ka * DELTA / np.sqrt(s) * zrow[5:]).tolist())
        return d
    if tag == "main":
        return [spec(i, z[i]) for i in range(N_MAIN)]
    mt = re.fullmatch(r"(sens|amp)([0-9]+(?:\.[0-9]+)?)", tag)
    if mt:                      # sens<k>: Var beta -> k s (pure Gaussian); amp<k>: beta0, Pi0 multiplied by k (stress test)
        k = float(mt.group(2))
        if mt.group(1) == "sens":
            return [spec(i, z[i], ks=k) for i in range(64)]
        return [spec(i, z[i], ka=k) for i in range(64)]
    if tag == "ic_can":
        return [spec(i, z[i], ic="canonical") for i in range(32)]
    if tag == "ic_trf":
        return [spec(i, z[i], ic="trfree") for i in range(32)]
    if tag == "conv_tol":
        return [spec(i, z[i], rtol=2e-10, atol=2e-12) for i in range(16)]
    if tag == "conv_cut":
        return [spec(i, z[i], cut=60.0) for i in range(16)]
    if tag == "hi":
        zh = z_hi()
        return [spec(i, zh[i]) for i in range(len(zh))]
    raise ValueError(tag)


_MODELS = {}


def get_model(rtol, atol, cut):
    key = (rtol, atol, cut)
    if key not in _MODELS:
        _MODELS[key] = NLModel(CONTENT_SM, cut=cut, rtol=rtol, atol=atol)
    return _MODELS[key]


def work(spec):
    path = DRAWS / ("%s_%04d.json" % (spec["tag"], spec["i"]))
    if path.exists():
        return 0
    model = get_model(spec["rtol"], spec["atol"], spec["cut"])
    rec = model.forward(spec["beta0"], spec["Pi0"], ic=spec["ic"], label="%s_%d" % (spec["tag"], spec["i"]))
    rec.update(tag=spec["tag"], i=spec["i"], ks=spec["ks"], amp=spec["amp"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["main", "sens3", "amp3", "ic_can", "ic_trf", "conv_tol", "conv_cut"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("DNEC_WORKERS", "6")))  # package: --workers
    args = ap.parse_args()
    procs = max(1, args.procs)  # package: no cap (each draw is independent; results do not depend on the worker count)
    DRAWS.mkdir(parents=True, exist_ok=True)
    # manifest of the predeclared draws and linear transfers
    z = z_main()
    man = {"seed": SEED, "seed_hi": SEED_HI, "delta": DELTA, "s_nat": S_NAT.tolist(), "N_main": N_MAIN,
           "Z_checksum_sum_z_times_col_index": float(np.sum(z * np.arange(1, 11))),
           "note": "beta0=sqrt(k s) Z[:, :5], Pi0=delta/sqrt(k s) Z[:, 5:]; rng=numpy default_rng(seed).standard_normal((N,10))"}
    (OUT / "C60_draw_manifest.json").write_text(json.dumps(man, indent=1))
    for tag in args.tags:
        specs = make_specs(tag)
        todo = [s for s in specs if not (DRAWS / ("%s_%04d.json" % (s["tag"], s["i"]))).exists()]
        t0 = time.time()
        done = 0
        if todo:
            with Pool(procs) as pool:
                for r in pool.imap_unordered(work, todo, chunksize=2):
                    done += r
        print("%-8s total %d, ran %d, %.1f s" % (tag, len(specs), done, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
