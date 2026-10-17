# PROVENANCE: 04_calculations/code/20261009_claude_R5/obs_fisher/C59_validate.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Unit checks of the covariance builder: harmonics, half lattice, parity blocks, continuum limit.

C59_validate: unit checks of the torus covariance builder.

  python C59_validate.py units      harmonics vs scipy + quadrature orthonormality, half-lattice, timing
  python C59_validate.py parity     explicit complex sum over n and -n: opposite-parity blocks and
                                    agreement with the paired real-basis builder
  python C59_validate.py continuum  large-L0 continuum limit (6 d_LSS, l<=20; 3 d_LSS, l<=30), parallel
Outputs: results/20261009_claude_R5/obs_fisher/validate/C59_validate_<name>.json
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
import time
import numpy as np
from C59_torus import (OUT, Transfer, TorusCov, real_ylm, half_lattice, kmatrix, window, kl_gauss)

VDIR = OUT / "validate"
VDIR.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    (VDIR / f"C59_validate_{name}.json").write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def units():
    from scipy.special import sph_harm_y_all
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    lmax = 40
    mine = real_ylm(v, lmax)
    th = np.arccos(np.clip(v[:, 2], -1, 1)); ph = np.arctan2(v[:, 1], v[:, 0])
    ally = sph_harm_y_all(lmax, lmax, th, ph)
    ref = np.empty_like(mine)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            val = ally[l, abs(m)]
            r = val.imag if m < 0 else val.real
            ref[:, l * l + l + m] = r * (1.0 if m == 0 else np.sqrt(2) * (-1.0) ** m)
    harm_err = float(np.max(np.abs(mine - ref)))
    # quadrature orthonormality: Gauss-Legendre in cos(theta) x uniform phi
    nth, nph = 64, 128
    z, wz = np.polynomial.legendre.leggauss(nth)
    phis = 2 * np.pi * np.arange(nph) / nph
    Z, P = np.meshgrid(z, phis, indexing="ij")
    s = np.sqrt(1 - Z ** 2)
    u = np.column_stack([(s * np.cos(P)).ravel(), (s * np.sin(P)).ravel(), Z.ravel()])
    w = (np.repeat(wz, nph) * 2 * np.pi / nph)
    Yq = real_ylm(u, 30)
    G = (Yq * w[:, None]).T @ Yq
    ortho_err = float(np.max(np.abs(G - np.eye(len(G)))))
    # half lattice vs brute force
    R = 7.3
    q = np.arange(-8, 9)
    nn = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    nn = nn[(np.sum(nn ** 2, 1) <= R * R) & np.any(nn != 0, 1)]
    hl = half_lattice(R)
    sh = set(map(tuple, hl)); sneg = set(map(tuple, -hl))
    lat_ok = (len(hl) * 2 == len(nn)) and not (sh & sneg) and (sh | sneg) == set(map(tuple, nn))
    # parity of Y under inversion
    par_err = float(np.max(np.abs(real_ylm(-v, 10) - real_ylm(v, 10) * np.repeat((-1.0) ** np.arange(11), 2 * np.arange(11) + 1)[None, :])))
    # window
    xx = np.linspace(0, 10, 1001); W = window(xx, 5, 8)
    # timing of one covariance batch at l=30 (T+E)
    tr = Transfer(30)
    tc = TorusCov(tr, 30)
    L0 = 2 * tr.chi
    nlist, Rn = tc.mode_list(L0, 60.0, [np.eye(3)])
    t0 = time.perf_counter()
    C, info = tc.compute(L0, np.eye(3), 60.0, nlist)
    dt = time.perf_counter() - t0
    out = dict(real_harmonic_max_abs_diff_vs_scipy_C35_convention_l40=harm_err,
               quadrature_orthonormality_max_abs_err_l30=ortho_err,
               half_lattice_matches_bruteforce_R7p3=bool(lat_ok), half_lattice_count=int(len(hl)),
               inversion_parity_max_err=par_err,
               window_checks=dict(W_at_x1=float(window(5.0, 5, 8)), W_at_x2=float(window(8.0, 5, 8)),
                                  monotone=bool(np.all(np.diff(W) <= 1e-15))),
               timing_l30_TE_cube_L0_dLSS_xmax60=dict(seconds=dt, **info,
                                                      dims=[int(c.shape[0]) for c in C]))
    save("units", out)
    print(json.dumps(out, indent=1))


def parity():
    """Complex explicit sum over all n (both signs) with i^{l-l'}: compare with paired real builder."""
    lmax, xmax = 10, 30.0
    tr = Transfer(lmax)
    tc = TorusCov(tr, lmax)
    L0 = 0.93 * 2 * tr.chi
    M = kmatrix(beta5=[0.05, -0.03, 0.04, 0.02, -0.06])     # generic (non-cubic) shape
    nlist, _ = tc.mode_list(L0, xmax, [M])
    Cp, _ = tc.compute(L0, M, xmax, nlist, taper=0.25)
    full = np.concatenate([nlist, -nlist]).astype(float)
    kv = 2 * np.pi / L0 * full @ M.T
    k = np.linalg.norm(kv, axis=1); keep = k * tr.chi < xmax
    kv, k = kv[keep], k[keep]; x = k * tr.chi
    w = (4 * np.pi) ** 2 / L0 ** 3 * 2 * np.pi ** 2 * tr.calP(k) / k ** 3 * window(x, 0.75 * xmax, xmax)
    Y = real_ylm(kv / k[:, None], lmax)
    D = tr.eval(x)
    lab = np.array([(X, l, m) for X in (0, 1) for l in range(2, lmax + 1) for m in range(-l, l + 1)])
    A = (Y[:, lab[:, 1] ** 2 + lab[:, 1] + lab[:, 2]] * D[:, lab[:, 0] * tr.nl + lab[:, 1] - 2]
         * np.sqrt(w)[:, None]) * (1j) ** (-lab[:, 1])[None, :]
    C = (A.conj().T @ A)
    tail = tr.tail(0.75 * xmax, xmax)
    from C59_torus import add_diag_tl
    Cr = np.zeros(C.shape); add_diag_tl(Cr, lab, tail)
    C = C + Cr
    scale = np.sqrt(np.abs(np.diag(C).real))
    Cn = C / scale[:, None] / scale[None, :]
    par = lab[:, 1] % 2
    opp = par[:, None] != par[None, :]
    res = dict(imag_part_max_normalized=float(np.max(np.abs(Cn.imag))),
               opposite_parity_block_max_normalized=float(np.max(np.abs(Cn[opp]))))
    diffs = []
    for p in (0, 1):
        labp = tc.blocks[p]["labels"]
        idx = [int(np.flatnonzero((lab[:, 0] == a) & (lab[:, 1] == b) & (lab[:, 2] == c))[0]) for a, b, c in labp]
        sub = C.real[np.ix_(idx, idx)]
        sc = np.sqrt(np.diag(sub))
        diffs.append(float(np.max(np.abs(sub - Cp[p]) / sc[:, None] / sc[None, :])))
    res["paired_vs_explicit_max_normalized_diff"] = diffs
    res["setup"] = dict(lmax=lmax, xmax=xmax, L0_over_dLSS=0.93, beta5=[0.05, -0.03, 0.04, 0.02, -0.06],
                        modes_full=int(len(k)))
    save("parity", res)
    print(json.dumps(res, indent=1))


def _continuum_task(args):
    s, lmax, xmax, part, nparts = args
    tr = Transfer(lmax); tc = TorusCov(tr, lmax)
    L0 = s * 2 * tr.chi
    nlist, R = tc.mode_list(L0, xmax, [np.eye(3)])
    sub = nlist[part::nparts]
    C, info = tc.compute(L0, np.eye(3), xmax, sub)
    # remove the tail from all but part 0 (each part added it once)
    if part != 0:
        tail = tr.tail(0.75 * xmax, xmax)
        from C59_torus import add_diag_tl
        for p, blk in enumerate(tc.blocks):
            neg = {k: -v for k, v in tail.items()}
            add_diag_tl(C[p], blk["labels"], neg)
    return s, lmax, xmax, C, info


def continuum():
    from multiprocessing import Pool
    jobs = []
    cases = [(6.0, 20, 60.0), (3.0, 30, 120.0)]
    nparts = 4
    for s, lmax, xmax in cases:
        for part in range(nparts):
            jobs.append((s, lmax, xmax, part, nparts))
    with Pool(int(os.environ.get("DNEC_WORKERS", "8"))) as pool:  # package: --workers
        res = pool.map(_continuum_task, jobs)
    out = {}
    for s, lmax, xmax in cases:
        parts = [r for r in res if r[0] == s]
        C = [sum(r[3][p] for r in parts) for p in (0, 1)]
        tr = Transfer(lmax); tc = TorusCov(tr, lmax)
        cov = tc.cover()
        rows = {}
        diag_dev, off_corr, te_dev = [], [], []
        per_l = {l: [] for l in range(2, lmax + 1)}
        kls = {}
        for p in (0, 1):
            lab = tc.blocks[p]["labels"]
            d = np.diag(C[p]); dc = np.diag(cov[p])
            for i, (X, l, m) in enumerate(lab):
                per_l[l].append((X, d[i] / dc[i] - 1))
            sc = 1 / np.sqrt(dc)
            cn = C[p] * sc[:, None] * sc[None, :]
            ccn = cov[p] * sc[:, None] * sc[None, :]
            off = cn - ccn
            np.fill_diagonal(off, 0.0)
            off_corr.append(float(np.max(np.abs(off))))
        for fields in ("T", "TE"):
            for lsel in sorted({10, 20, lmax}):
                if lsel > lmax:
                    continue
                kl = 0.0
                for p in (0, 1):
                    ix = tc.select(p, lsel, fields)
                    sc = tc.scale(p)[ix]
                    kl += kl_gauss(C[p][np.ix_(ix, ix)] * sc[:, None] * sc[None, :],
                                   cov[p][np.ix_(ix, ix)] * sc[:, None] * sc[None, :])[0]
                kls[f"{fields}_l{lsel}"] = kl
        dl = {l: dict(T_mean=float(np.mean([v for X, v in per_l[l] if X == 0])),
                      T_maxabs=float(np.max(np.abs([v for X, v in per_l[l] if X == 0]))),
                      E_mean=float(np.mean([v for X, v in per_l[l] if X == 1])),
                      E_maxabs=float(np.max(np.abs([v for X, v in per_l[l] if X == 1]))))
              for l in range(2, lmax + 1)}
        out[f"L0_{s:g}dLSS_l{lmax}_x{xmax:g}"] = dict(
            L0_over_dLSS=s, lmax=lmax, xmax=xmax, modes_full=int(sum(r[4]["modes_full_used"] for r in parts)),
            max_abs_diag_rel_dev=float(max(max(abs(v) for X, v in per_l[l]) for l in per_l)),
            max_abs_offdiag_minus_cover_normalized=float(max(off_corr)),
            KL_torus_vs_cover=kls, per_l_diag_rel_dev=dl)
        print(s, lmax, xmax, "maxdiag", out[f"L0_{s:g}dLSS_l{lmax}_x{xmax:g}"]["max_abs_diag_rel_dev"],
              "maxoff", max(off_corr), "KL", kls, flush=True)
    save("continuum", out)


if __name__ == "__main__":
    {"units": units, "parity": parity, "continuum": continuum}[sys.argv[1]]()
