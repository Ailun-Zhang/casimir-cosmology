# PROVENANCE: 04_calculations/code/20261011_claude_followup/C75/C75_common.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Configuration and cache lookup of the v11 readability window (D57): own cache fu/C75/cache, then the C59 / C70 caches (work tree r5/...). Paper: PRD Sec. X (window of v11, superseded for the ensemble by cmb/02).

C75_common: shared configuration, cache keys and cache lookup for C75_produce.py / C75_analyze.py.

Cache lookup order (never recompute a covariance that already exists with identical conventions):
  1. this task's cache   results/20261011_claude_followup/C75/cache/<key>.npz            (keys with s to 5 decimals)
  2. R5 C59 cache        results/20261009_claude_R5/obs_fisher/cache  (read-only)
       cubes  base_s{s:.3f}_x160_t0.25_d_none.npz
       free fossil (sigma_E = 0.0589 only; amp = sqrt(2)*0.0589, identical) defm_s{s:.3f}_x160_t0.25_d_Xd_benchA.npz
  3. R5 C70 cache        results/20261009_claude_R5/C70_profile_fine/cache  (read-only; cubes 1.030 ... 1.095)
External files are used only for x_max = 160, l <= 30, taper 0.25 'deformed' and s with at most 3 decimals.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package: chain root (cmb/)
import _cmb_paths as WP  # noqa: E402
C59_CODE = HERE  # package: the C59 modules live in this directory
RES = WP.wdir("fu/C75")
CACHE = RES / "cache"
R5 = WP.wdir("r5")
C59_CACHE = R5 / "obs_fisher" / "cache"
C70_CACHE = R5 / "C70_profile_fine" / "cache"

TAPER = 0.25
XMAIN = 160.0
CIRCLE = 0.985                       # matched-circle bound on the shortest cycle, units of d_LSS
THR = 4.5                            # 2D = 9
SIGMAS = {"free": 0.0589, "r25": 0.0925, "r30": 0.1041}
R_OF = {"free": 0.0, "r25": 0.25, "r30": 0.30}
FREE_AMP = float(np.sqrt(2.0) * SIGMAS["free"])          # = C59 benchmark A amplitude (checked in analysis)

GRID_CUBE = [round(1.02 + 0.005 * i, 3) for i in range(37)]      # comparison cubes 1.020 ... 1.200
GRID_FOSSIL = {
    "free": [1.05, 1.07, 1.075, 1.08, 1.085, 1.09, 1.10, 1.14, 1.15, 1.16, 1.20],
    "r25": [1.07, 1.08, 1.085, 1.09, 1.095, 1.10, 1.105, 1.11, 1.115, 1.12, 1.125, 1.13, 1.14, 1.16, 1.18, 1.20],
    "r30": [1.07, 1.08, 1.09, 1.095, 1.10, 1.105, 1.11, 1.115, 1.12, 1.125, 1.13, 1.14, 1.16, 1.18, 1.20],
}


def s_circle(sig):
    """Circle bound on s = L0/d_LSS for sides L0 e^{+sigma}, L0 e^{-sigma}, L0: L0 e^{-sigma} > 0.985 d_LSS."""
    return CIRCLE * math.exp(sig)


def s_circle_point(sig):
    """Sample point at the circle bound, rounded UP to 5 decimals so that it is (just) allowed."""
    return math.ceil(s_circle(sig) * 1e5 - 1e-9) / 1e5


def fossil_points(b):
    return sorted(set(GRID_FOSSIL[b]) | {s_circle_point(SIGMAS[b])})


def cube_points():
    return sorted(set(GRID_CUBE) | {s_circle_point(v) for v in SIGMAS.values()})


def key(kind, s, xmax=XMAIN, lmax=30, sig=None, gen="Xd"):
    k = f"{kind}_s{s:.5f}_x{xmax:g}_t{TAPER:g}_d"
    k += "_none" if kind == "base" else f"_{gen}_sig{sig:.4f}"
    return k + ("" if lmax == 30 else f"_l{lmax}")


def _three_dp(s):
    return abs(round(s, 3) - s) < 1e-9


def find(kind, s, xmax=XMAIN, lmax=30, sig=None, gen="Xd", own_only=False):
    """Path of an existing covariance with these conventions, or None."""
    p = CACHE / (key(kind, s, xmax, lmax, sig, gen) + ".npz")
    if p.exists() or own_only:
        return p if p.exists() else None
    if xmax != XMAIN or lmax != 30 or not _three_dp(s):
        return None
    if kind == "base":
        for d in (C59_CACHE, C70_CACHE):
            q = d / f"base_s{s:.3f}_x160_t0.25_d_none.npz"
            if q.exists():
                return q
    elif kind == "defm" and gen == "Xd" and sig is not None and abs(sig - SIGMAS["free"]) < 1e-12:
        q = C59_CACHE / f"defm_s{s:.3f}_x160_t0.25_d_Xd_benchA.npz"
        if q.exists():
            return q
    return None


def pack(c):
    return c[np.triu_indices(c.shape[0])]


def unpack(v, n):
    c = np.zeros((n, n))
    iu = np.triu_indices(n)
    c[iu] = v
    return c + np.triu(c, 1).T


def load(path):
    a = np.load(path)
    meta = json.loads(str(a["meta"]))
    f = meta["fields"][0]
    return [unpack(a[f"{f}_p{p}"], meta["dims"][p]) for p in (0, 1)], meta


def get(kind, s, xmax=XMAIN, lmax=30, sig=None, gen="Xd"):
    p = find(kind, s, xmax, lmax, sig, gen)
    if p is None:
        raise FileNotFoundError(key(kind, s, xmax, lmax, sig, gen))
    C, meta = load(p)
    meta["_path"] = str(p)
    return C, meta


def amp_of(meta):
    """Deformation amplitude |beta| along the generator (C75 meta: 'amp'; C59 meta: extra.amp)."""
    if meta.get("amp") is not None:
        return float(meta["amp"])
    return float((meta.get("extra") or {}).get("amp", 0.0))


def modes_of(meta):
    m = meta.get("modes_half_used")
    return int(m[0]) if isinstance(m, list) else (None if m is None else int(m))
