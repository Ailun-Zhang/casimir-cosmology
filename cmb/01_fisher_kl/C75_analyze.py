# PROVENANCE: 04_calculations/code/20261011_claude_followup/C75/C75_analyze.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Exact Gaussian KL window of the zero-mode-corrected fossil (l <= 30): validation against the C59 cache, D(s) scans, upper edges, profiled volume. The optional external Codex sigma table (an external Codex workspace, data only) is not shipped. Paper: PRD Sec. X; v11 Fig. 7 (a)(b).

C75_analyze: readability window of the zero-mode-corrected fossil (exact Gaussian KL, T+E and T, l <= 30).

Reads covariances through C75_common.get (this task's cache first, then read-only the R5 C59 / C70 caches; nothing is
written outside results/20261011_claude_followup/C75).

KL divergence D(C_f || C_c) = 1/2 [tr(C_c^{-1} C_f) - n - ln det(C_c^{-1} C_f)], summed over the two parity blocks
(opposite-parity blocks vanish identically). Implemented here in two ways:
  eig    : Cholesky C_c = L L^T, eigenvalues lam of L^{-1} C_f L^{-T}, D = 1/2 sum(lam - 1 - ln lam)   (primary)
  logdet : D = 1/2 [tr(cho_solve(C_c, C_f)) - n + ln det C_c - ln det C_f] with Cholesky log-determinants   (check)
Both after the congruence diag(C_c)^{-1/2} (KL invariant). T alone = the T entries of each parity block.

Window definitions (paper v10, Sec. IX):
  circle bound      : shortest cycle L0 e^{-sigma} > 0.985 d_LSS  <=>  s = L0/d_LSS > s_c = 0.985 e^{sigma}
  readable          : 2D >= 9 (D >= 4.5), D against the same-size cube (equal volume), l <= 30, full sky, no noise
  upper edge        : largest grid s with D >= 4.5, log-linear interpolation of D to the next grid point (C59 rule);
                      a monotone cubic (PCHIP) interpolation of ln D is reported as a check
  profiled          : D_prof(s) = min over comparison cubes s' (grid 1.020..1.200, step 0.005) of D(C_f(s) || C_c(s')),
                      with a parabolic refinement of the minimum in s'

    python C75_analyze.py            -> results/20261011_claude_followup/C75/C75_results.json, C75_D_table.csv
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigvalsh, cho_factor, cho_solve
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C75_common import (RES, CACHE, R5, C59_CACHE, C70_CACHE, C59_CODE, XMAIN, CIRCLE, THR, SIGMAS, R_OF,  # noqa
                        FREE_AMP, GRID_CUBE, key, find, load, get, amp_of, modes_of, s_circle, s_circle_point,
                        fossil_points)

C59_JSON = R5 / "obs_fisher" / "C59_shape_fisher.json"
C70_JSON = R5 / "C70_profile_fine" / "C70_results.json"
PAPER = {"TE": {1.05: 13.0, 1.07: 6.7, 1.10: 1.5, 1.15: 0.25}}   # v10 Sec. IX (free fossil, T+E, l <= 30)


# ------------------------------------------------------------------------------------------------ index sets
def labels(p, lmax):
    """(X, l, m) of parity block p in the C59 layout: all T entries first, then all E entries."""
    return np.array([(X, l, m) for X in (0, 1) for l in range(2, lmax + 1) if l % 2 == p
                     for m in range(-l, l + 1)], int)


_LAB = {}


def index(p, lmax, lsel, fields):
    k = (p, lmax, lsel, fields)
    if k not in _LAB:
        lab = labels(p, lmax)
        fl = {"T": [0], "E": [1], "TE": [0, 1]}[fields]
        _LAB[k] = np.flatnonzero((lab[:, 1] <= lsel) & np.isin(lab[:, 0], fl))
    return _LAB[k]


# ------------------------------------------------------------------------------------------------ KL
def kl_eig(P, Q):
    L = cholesky(Q, lower=True, check_finite=False)
    W = solve_triangular(L, P, lower=True, check_finite=False)
    W = solve_triangular(L, W.T, lower=True, check_finite=False)
    lam = eigvalsh(0.5 * (W + W.T), check_finite=False)
    if lam.min() <= 0:
        raise ValueError("non positive definite")
    return float(0.5 * np.sum(lam - 1.0 - np.log(lam))), lam


def kl_logdet(P, Q):
    cq = cho_factor(Q, lower=True, check_finite=False)
    tr = float(np.trace(cho_solve(cq, P, check_finite=False)))
    ldq = 2.0 * float(np.sum(np.log(np.diag(cq[0]))))
    ldp = 2.0 * float(np.sum(np.log(np.diag(cholesky(P, lower=True, check_finite=False)))))
    return 0.5 * (tr - P.shape[0] + ldq - ldp)


def D(Cf, Cc, fields, lmax=30, lsel=30, check=False):
    """Exact KL D(N(0,Cf) || N(0,Cc)) summed over parity blocks; optional logdet cross-check."""
    tot, tot2, lmin, lmx = 0.0, 0.0, np.inf, 0.0
    for p in (0, 1):
        ix = index(p, lmax, lsel, fields)
        d = 1.0 / np.sqrt(np.diag(Cc[p])[ix])
        P = Cf[p][np.ix_(ix, ix)] * d[:, None] * d[None, :]
        Q = Cc[p][np.ix_(ix, ix)] * d[:, None] * d[None, :]
        v, lam = kl_eig(P, Q)
        tot += v
        lmin, lmx = min(lmin, float(lam.min())), max(lmx, float(lam.max()))
        if check:
            tot2 += kl_logdet(P, Q)
    out = dict(D=tot, lam_min=lmin, lam_max=lmx)
    if check:
        out["D_logdet"] = tot2
    return out


# ------------------------------------------------------------------------------------------------ helpers
def crossing_up(s, y, thr=THR):
    """Largest s with y >= thr; log-linear interpolation to the next grid point (C59 rule)."""
    s = np.asarray(s, float); y = np.asarray(y, float)
    ok = y >= thr
    if not ok.any():
        return None
    i = int(np.flatnonzero(ok)[-1])
    if i == len(s) - 1:
        return f">{s[-1]:g}"
    y0, y1 = np.log(y[i]), np.log(y[i + 1])
    return float(s[i] + (np.log(thr) - y0) / (y1 - y0) * (s[i + 1] - s[i]))


def crossing_pchip(s, y, thr=THR):
    s = np.asarray(s, float); y = np.asarray(y, float)
    ok = y >= thr
    if not ok.any() or ok[-1]:
        return None
    i = int(np.flatnonzero(ok)[-1])
    f = PchipInterpolator(s, np.log(y))
    a, b = s[i], s[i + 1]
    for _ in range(80):
        m = 0.5 * (a + b)
        if f(m) >= np.log(thr):
            a = m
        else:
            b = m
    return float(0.5 * (a + b))


def parabola_min(xs, ys):
    """Vertex of the parabola through the grid minimum and its two neighbours (None at the grid edge)."""
    i = int(np.argmin(ys))
    if i == 0 or i == len(xs) - 1:
        return None, None
    x0, x1, x2 = xs[i - 1:i + 2]
    y0, y1, y2 = ys[i - 1:i + 2]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    B = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
    C = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / den
    xv = -B / (2 * A)
    return float(xv), float(C - B ** 2 / (4 * A))


def rel(A, B):
    return max(float(np.abs(A[p] - B[p]).max() / np.abs(B[p]).max()) for p in (0, 1))


# ------------------------------------------------------------------------------------------------ validation
def validation():
    out = {"from_C59_cache": {}, "fresh_vs_cached": {}, "determinism_cubes": {}, "kl_methods": {}}
    c59 = json.loads(C59_JSON.read_text(encoding="utf-8"))["exact_KL_check"]
    c70 = json.loads(C70_JSON.read_text(encoding="utf-8"))["fossils"]
    for s in (1.02, 1.05, 1.07, 1.10, 1.15, 1.20):
        Cf, mf = load(C59_CACHE / f"defm_s{s:.3f}_x160_t0.25_d_Xd_benchA.npz")
        Cc, _ = load(C59_CACHE / f"base_s{s:.3f}_x160_t0.25_d_none.npz")
        row = {"amp": amp_of(mf), "amp_equals_sqrt2_0.0589": abs(amp_of(mf) - FREE_AMP) < 1e-14}
        for fl in ("TE", "T"):
            r = D(Cf, Cc, fl, check=True)
            ref59 = c59.get(f"{s:g}", {}).get("Xd_benchA", {}).get(f"{fl}_l30_none", {}).get("KL_exact")
            ref70 = c70.get(f"{s:.2f}", {}).get(fl, {}).get("D_same_size")
            row[fl] = dict(D=r["D"], D_logdet=r["D_logdet"], C59_json=ref59, C70_json=ref70,
                           paper=PAPER.get(fl, {}).get(s))
        out["from_C59_cache"][f"{s:.2f}"] = row
        print(f"[validate C59 cache] s={s:.2f}: D_TE={row['TE']['D']:.4f} (C59 json {row['TE']['C59_json']}, "
              f"paper {row['TE']['paper']}), D_T={row['T']['D']:.4f}", flush=True)
    # fresh recomputation (this task's own cache) versus the C59 cache, s = 1.07
    Cf_new, m_new = load(find("defm", 1.07, sig=SIGMAS["free"], own_only=True))
    Cc_new, mc_new = load(find("base", 1.07, own_only=True))
    Cf_old, m_old = load(C59_CACHE / "defm_s1.070_x160_t0.25_d_Xd_benchA.npz")
    Cc_old, mc_old = load(C59_CACHE / "base_s1.070_x160_t0.25_d_none.npz")
    fv = dict(amp_fresh=amp_of(m_new), amp_cached=amp_of(m_old),
              modes_half_used_fresh=modes_of(m_new), modes_half_used_cached=modes_of(m_old),
              cube_modes_fresh=modes_of(mc_new), cube_modes_cached=modes_of(mc_old),
              max_rel_diff_fossil=rel(Cf_new, Cf_old), max_rel_diff_cube=rel(Cc_new, Cc_old))
    for fl in ("TE", "T"):
        fv[f"D_{fl}_fresh_fresh"] = D(Cf_new, Cc_new, fl)["D"]
        fv[f"D_{fl}_fresh_vs_cachedcube"] = D(Cf_new, Cc_old, fl)["D"]
        fv[f"D_{fl}_cached_cached"] = D(Cf_old, Cc_old, fl)["D"]
    out["fresh_vs_cached"]["s1.07_free_fossil"] = fv
    print(f"[validate fresh 1.07] rel diff fossil {fv['max_rel_diff_fossil']:.2e}, cube {fv['max_rel_diff_cube']:.2e}; "
          f"D_TE fresh {fv['D_TE_fresh_fresh']:.4f} vs cached {fv['D_TE_cached_cached']:.4f}", flush=True)
    # other fresh free fossils that coincide with cached ones
    for s in (1.05, 1.10, 1.15, 1.20):
        p = find("defm", s, sig=SIGMAS["free"], own_only=True)
        if p is not None:
            A, _ = load(p)
            B, _ = load(C59_CACHE / f"defm_s{s:.3f}_x160_t0.25_d_Xd_benchA.npz")
            out["fresh_vs_cached"][f"s{s:.2f}_free_fossil_max_rel_diff"] = rel(A, B)
    # cubes: fresh versus C59 and C70 caches where both exist
    for s in GRID_CUBE:
        p = find("base", s, own_only=True)
        if p is None:
            continue
        A, _ = load(p)
        for tag, fn in (("C59", C59_CACHE / f"base_s{s:.3f}_x160_t0.25_d_none.npz"),
                        ("C70", C70_CACHE / f"base_s{s:.3f}_x160_t0.25_d_none.npz")):
            if fn.exists():
                B, _ = load(fn)
                out["determinism_cubes"][f"{s:.3f}_{tag}"] = rel(A, B)
    if out["determinism_cubes"]:
        print(f"[validate cubes] {len(out['determinism_cubes'])} fresh cubes vs R5 caches, worst rel diff "
              f"{max(out['determinism_cubes'].values()):.2e}", flush=True)
    # KL implementation cross-check with the project's kl_gauss (imported read-only)
    sys.path.insert(1, str(C59_CODE))
    from C59_torus import kl_gauss  # noqa: E402
    tot = 0.0
    for p in (0, 1):
        ix = index(p, 30, 30, "TE")
        d = 1.0 / np.sqrt(np.diag(Cc_old[p])[ix])
        tot += kl_gauss(Cf_old[p][np.ix_(ix, ix)] * d[:, None] * d[None, :],
                        Cc_old[p][np.ix_(ix, ix)] * d[:, None] * d[None, :])[0]
    out["kl_methods"] = dict(D_TE_s1p07_project_kl_gauss=tot, D_TE_s1p07_this_eig=fv["D_TE_cached_cached"])
    # geometry of the deformed lattices actually used: k_n = (2 pi/L0) M n, diagonal M -> sides L0 / M_ii
    out["fossil_geometry"] = {}
    for b, sig in SIGMAS.items():
        _, m = load(find("defm", 1.20, sig=sig, own_only=True))
        Mg = np.array(m["M"])
        sides = (1.0 / np.diag(Mg)).tolist()
        out["fossil_geometry"][b] = dict(M=m["M"], offdiag_max=float(np.abs(Mg - np.diag(np.diag(Mg))).max()),
                                         sides_over_L0=sides, expected=[math.exp(-sig) ** -1, math.exp(-sig), 1.0],
                                         volume_factor=float(np.prod(sides)))
    return out


# ------------------------------------------------------------------------------------------------ main scan
def scan_task(args):
    """All comparison cubes for one fossil (sigma label b, size s): D_TE and D_T versus cube size s'."""
    b, s = args
    sig = SIGMAS[b]
    Cf, mf = get("defm", s, sig=sig)
    Cs, _ = get("base", s)
    same = {fl: D(Cf, Cs, fl, check=True) for fl in ("TE", "T")}
    prof = {"TE": {}, "T": {}}
    if s >= s_circle(sig) - 1e-12:
        for sp in GRID_CUBE:
            Cc, _ = get("base", sp)
            for fl in ("TE", "T"):
                prof[fl][f"{sp:.3f}"] = D(Cf, Cc, fl)["D"]
    return b, s, dict(amp=amp_of(mf), modes_half_used=modes_of(mf), path=mf["_path"], same=same, prof=prof)


def window_summary(b, rows):
    sig = SIGMAS[b]
    sc = s_circle(sig)
    pts = sorted(rows)
    allowed = [s for s in pts if s >= sc - 1e-12]
    res = dict(sigma_E=sig, r=R_OF[b], amp_beta_d=float(np.sqrt(2) * sig), s_circle=sc,
               s_circle_sample_point=s_circle_point(sig), shortest_cycle_factor=math.exp(-sig))
    for fl in ("TE", "T"):
        ys = [rows[s]["same"][fl]["D"] for s in allowed]
        mono = bool(np.all(np.diff(ys) < 0))
        up = crossing_up(allowed, ys)
        up_p = crossing_pchip(allowed, ys)
        Dc = rows[allowed[0]]["same"][fl]["D"]
        ent = dict(D_at_circle_bound=Dc, monotone_decreasing_above_circle=mono, upper_edge_loglin=up,
                   upper_edge_pchip=up_p)
        if isinstance(up, float) and Dc >= THR:
            ent.update(window=[sc, up], width=up - sc, width_efolds=math.log(up / sc),
                       shortest_cycle_range=[CIRCLE, up * math.exp(-sig)])
        else:
            ent.update(window=None, width=0.0, width_efolds=0.0, shortest_cycle_range=None)
        # profiled over comparison cube sizes
        prof_rows = []
        for s in allowed:
            pr = rows[s]["prof"][fl]
            xs = np.array([float(k) for k in pr]); vs = np.array(list(pr.values()))
            i = int(np.argmin(vs))
            xv, yv = parabola_min(xs, vs)
            prof_rows.append(dict(s=s, D_same=rows[s]["same"][fl]["D"], D_min_grid=float(vs[i]),
                                  s_prime_min=float(xs[i]), D_min_parabola=yv, s_prime_parabola=xv,
                                  at_grid_edge=bool(i in (0, len(xs) - 1)),
                                  reduction=1.0 - float(vs[i]) / rows[s]["same"][fl]["D"]))
        dmin = [min(p["D_min_grid"], p["D_min_parabola"] if p["D_min_parabola"] is not None else np.inf)
                for p in prof_rows]
        upp = crossing_up(allowed, dmin)
        ent["profiled"] = dict(rows=prof_rows, upper_edge_loglin=upp, D_at_circle_bound=dmin[0],
                               window=([sc, upp] if isinstance(upp, float) and dmin[0] >= THR else None))
        if isinstance(upp, float) and dmin[0] >= THR:
            ent["profiled"]["shortest_cycle_range"] = [CIRCLE, upp * math.exp(-sig)]
            ent["profiled"]["upper_edge_shift"] = (upp - up) if isinstance(up, float) else None
        res[fl] = ent
    return res


def orientation_note():
    """Analytic: |beta_E| = sqrt(2) sigma at angle theta in the (X_d, X_d2) doublet gives sides
    L0 exp((2/sqrt3) sigma cos(theta - phi_k)), phi_k = pi/6, 5pi/6, 3pi/2. Shortest-side exponent per unit sigma:
    X_d (theta = 0): -1; +X_d2: -2/sqrt3; -X_d2: -1/sqrt3; uniform-theta mean: -(2/sqrt3)(3 sqrt3/(2 pi)) = -3/pi."""
    th = np.linspace(0.0, 2 * np.pi, 200001)[:-1]
    phis = np.array([np.pi / 6, 5 * np.pi / 6, 3 * np.pi / 2])
    expo = (2 / np.sqrt(3)) * np.cos(th[:, None] - phis[None, :]).min(axis=1)
    out = dict(exponent_per_sigma={"Xd": -1.0, "+Xd2": -2 / math.sqrt(3), "-Xd2": -1 / math.sqrt(3),
                                   "uniform_mean_numeric": float(expo.mean()), "uniform_mean_exact": -3 / math.pi})
    out["circle_bound_s"] = {b: {g: CIRCLE * math.exp(-e * sig) for g, e in
                                 (("Xd", -1.0), ("+Xd2", -2 / math.sqrt(3)), ("-Xd2", -1 / math.sqrt(3)),
                                  ("uniform_mean", -3 / math.pi))}
                             for b, sig in SIGMAS.items()}
    return out


def inputs():
    """Provenance of the benchmark widths: C71 (project, read-only) and the external Codex table (read as data only)."""
    out = {"benchmarks_used": SIGMAS}
    c71 = R5 / "C71_zero_mode_sensitivity" / "C71_results.json"
    if c71.exists():
        rows = json.loads(c71.read_text(encoding="utf-8"))["rows"]
        out["C71_sigma_E_natural"] = {f"{r['r']:g}": r["sigma_E_natural"] for r in rows}
    cdx = WP.step_dir("01_fisher_kl", "inputs", "direction4_zero_mode_sensitivity.json")  # package: external Codex table, not shipped (optional)
    if cdx.exists():
        try:
            rows = json.loads(cdx.read_text(encoding="utf-8"))["rows"]
            out["Codex_sigma_aspect_leading"] = {f"r={r['fraction_of_free_vacuum_budget']:.4f}":
                                                 r["sigma_aspect_natural_leading_estimate"] for r in rows}
        except Exception as e:  # data only; never fatal
            out["Codex_read_error"] = repr(e)
    return out


def extras():
    """Optional stages (x_max = 240, l <= 40, orientation inside E), analysed when their files exist."""
    out = {"xconv": {}, "l40": {}, "dirs": {}}
    for b, sig in SIGMAS.items():
        for s in fossil_points(b):
            f240, c240 = find("defm", s, 240.0, 30, sig), find("base", s, 240.0)
            if f240 is not None and c240 is not None:
                A, _ = load(f240); B, _ = load(c240)
                A0, _ = get("defm", s, sig=sig); B0, _ = get("base", s)
                out["xconv"][f"{b}_s{s:.3f}"] = {fl: dict(x160=D(A0, B0, fl)["D"], x240=D(A, B, fl)["D"])
                                                 for fl in ("TE", "T")}
            f40, c40 = find("defm", s, XMAIN, 40, sig), find("base", s, XMAIN, 40)
            if f40 is not None and c40 is not None:
                A, _ = load(f40); B, _ = load(c40)
                A0, _ = get("defm", s, sig=sig); B0, _ = get("base", s)
                out["l40"][f"{b}_s{s:.3f}"] = {fl: dict(l30_from_l30_files=D(A0, B0, fl)["D"],
                                                        l30_from_l40_files=D(A, B, fl, lmax=40, lsel=30)["D"],
                                                        l40=D(A, B, fl, lmax=40, lsel=40)["D"])
                                               for fl in ("TE", "T")}
            for g in ("Xd2p", "Xd2m"):
                fg = find("defm", s, XMAIN, 30, sig, g)
                if fg is not None:
                    A, ma = load(fg); B, _ = get("base", s); A0, _ = get("defm", s, sig=sig)
                    Mg = np.array(ma["M"])
                    sides = sorted((1.0 / np.abs(np.diag(Mg))).tolist())      # diagonal M: side_i = L0 / M_ii
                    out["dirs"][f"{b}_s{s:.3f}_{g}"] = dict(
                        sides_over_L0=sides, shortest_over_dLSS=s * sides[0], allowed_by_circles=s * sides[0] > CIRCLE,
                        D_TE=D(A, B, "TE")["D"], D_T=D(A, B, "T")["D"],
                        D_TE_Xd=D(A0, B, "TE")["D"], D_T_Xd=D(A0, B, "T")["D"])
    return out


def main():
    from multiprocessing import Pool
    t0 = time.perf_counter()
    val = validation()
    jobs = [(b, s) for b in SIGMAS for s in fossil_points(b)]
    rows = {b: {} for b in SIGMAS}
    with Pool(int(os.environ.get("DNEC_WORKERS", "6"))) as pool:  # package: --workers
        for b, s, r in pool.imap_unordered(scan_task, jobs):
            rows[b][s] = r
            print(f"[{time.perf_counter() - t0:7.1f}s] {b} s={s:.5f}: D_TE={r['same']['TE']['D']:.4f} "
                  f"D_T={r['same']['T']['D']:.4f}", flush=True)
    summ = {b: window_summary(b, rows[b]) for b in SIGMAS}
    # free fossil on the paper's coarse grid (circle bound excluded, as in v10): edge by the same log-linear rule
    coarse = [s for s in (1.05, 1.07, 1.10, 1.15, 1.20) if s in rows["free"]]
    free_coarse = {fl: crossing_up(coarse, [rows["free"][s]["same"][fl]["D"] for s in coarse]) for fl in ("TE", "T")}
    # quadratic-regime check: D ratio to the free fossil at equal s versus (sigma/sigma_free)^2
    ratios = {}
    for b in ("r25", "r30"):
        q = (SIGMAS[b] / SIGMAS["free"]) ** 2
        ratios[b] = {"sigma_ratio_squared": q}
        for s in sorted(set(rows[b]) & set(rows["free"])):
            ratios[b][f"{s:.3f}"] = {fl: rows[b][s]["same"][fl]["D"] / rows["free"][s]["same"][fl]["D"]
                                     for fl in ("TE", "T")}
    kl_check = max(abs(r["same"][fl]["D"] - r["same"][fl]["D_logdet"]) / r["same"][fl]["D"]
                   for b in rows for r in rows[b].values() for fl in ("TE", "T"))
    out = dict(
        scope="exact Gaussian KL D(C_fossil(s) || C_cube), T+E and T, 2<=l<=30, full sky, no noise, x_max=160, "
              "taper 0.25 (deformed), CAMB transfer v2; fossil |beta_E| = sqrt(2) sigma_E along X_d (sides L0e^{+-sigma}, "
              "L0); s = L0/d_LSS; cube of equal volume unless profiled",
        conventions=dict(circle_bound=CIRCLE, threshold_D=THR, crossing="largest s with D>=4.5, log-linear in D",
                         cube_grid=GRID_CUBE, fossil_grids={b: fossil_points(b) for b in SIGMAS}),
        inputs=inputs(),
        validation=val,
        free_fossil_edge_on_paper_coarse_grid=dict(grid=coarse, upper_edge=free_coarse),
        kl_eig_vs_logdet_max_rel_diff=kl_check,
        D_same_size={b: {f"{s:.5f}": {fl: rows[b][s]["same"][fl]["D"] for fl in ("TE", "T")} for s in sorted(rows[b])}
                     for b in SIGMAS},
        covariance_files={b: {f"{s:.5f}": rows[b][s]["path"] for s in sorted(rows[b])} for b in SIGMAS},
        eigenvalue_range={b: {f"{s:.5f}": [rows[b][s]["same"]["TE"]["lam_min"], rows[b][s]["same"]["TE"]["lam_max"]]
                              for s in sorted(rows[b])} for b in SIGMAS},
        D_profile={b: {f"{s:.5f}": rows[b][s]["prof"] for s in sorted(rows[b]) if rows[b][s]["prof"]["TE"]}
                   for b in SIGMAS},
        windows=summ, ratio_to_free=ratios, orientation=orientation_note(), extras=extras(),
        seconds=time.perf_counter() - t0)
    (RES / "C75_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(RES / "C75_D_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "sigma_E", "s=L0/dLSS", "shortest_cycle/dLSS", "allowed_by_circles", "D_TE_same",
                    "D_T_same", "D_TE_profiled_min", "s_prime_TE_min", "D_T_profiled_min", "s_prime_T_min"])
        for b in SIGMAS:
            sig = SIGMAS[b]
            prow = {fl: {p["s"]: p for p in summ[b][fl]["profiled"]["rows"]} for fl in ("TE", "T")}
            for s in sorted(rows[b]):
                pe, pt = prow["TE"].get(s), prow["T"].get(s)
                w.writerow([b, sig, f"{s:.5f}", f"{s * math.exp(-sig):.5f}", s >= s_circle(sig) - 1e-12,
                            f"{rows[b][s]['same']['TE']['D']:.5f}", f"{rows[b][s]['same']['T']['D']:.5f}",
                            "" if pe is None else f"{pe['D_min_grid']:.5f}", "" if pe is None else pe["s_prime_min"],
                            "" if pt is None else f"{pt['D_min_grid']:.5f}", "" if pt is None else pt["s_prime_min"]])
    print(f"\nfree fossil, paper coarse grid {coarse}: upper edge {free_coarse}")
    for b in SIGMAS:
        sm = summ[b]
        print(f"\n== {b}: sigma_E={sm['sigma_E']}, circle bound s_c={sm['s_circle']:.5f}")
        for fl in ("TE", "T"):
            e = sm[fl]
            print(f"  {fl}: D(s_c)={e['D_at_circle_bound']:.3f}, upper edge {e['upper_edge_loglin']} "
                  f"(pchip {e['upper_edge_pchip']}), window {e['window']}, shortest cycle {e['shortest_cycle_range']}")
            pr = e["profiled"]
            print(f"      profiled: D(s_c)={pr['D_at_circle_bound']:.3f}, upper edge {pr['upper_edge_loglin']}, "
                  f"window {pr['window']}")
    print(f"\nKL eig vs logdet max rel diff {kl_check:.2e}; wrote {RES / 'C75_results.json'} "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
