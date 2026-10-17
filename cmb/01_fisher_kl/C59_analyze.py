# PROVENANCE: 04_calculations/code/20261009_claude_R5/obs_fisher/C59_analyze.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Topology KL, shape Fisher sigma(beta_E) = 0.023/0.073/0.25 at L0 = 1.05/1.1/1.2 d_LSS (l <= 30, ideal full sky), convergence, block structure, exact-KL quadratic check -> C59_shape_fisher.json. Needs the C59_run cache (--full). Paper: PRD Sec. X (CMB-23).

C59_analyze: topology KL, shape Fisher, convergence, block structure, fossil detectability,
exact-KL quadratic-regime check.  Reads the C59_run cache; writes
  C59_shape_fisher.json, C59_shape_fisher_summary.csv, C59_shape_fisher_figure.npz
Fisher:  F_AB = 1/2 Tr[C^-1 d_A C C^-1 d_B C]  (central differences h=1e-3, 2e-3, Richardson).
Noise:   T cosmic-variance limited; E white noise N_l = (w_P arcmin / T_CMB)^2, beams neglected.
Sky cut: F -> f_sky F (mode counting).
Runs in one process with one BLAS thread by default ("pool" argument: 8 processes).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import csv
import json
import sys
import time
import numpy as np
from scipy.linalg import cholesky
from C59_torus import OUT, Transfer, TorusCov, kl_gauss, noise_EE, SHAPE, ROTATIONS, _whiten
from C59_run import CACHE, key, load, S_MAIN, S_FINE, S_CONV, X_CONV, XMAIN, SIG, LMAX, H

NOISE = {"none": 0.0, "P60": 60.0, "LB2p16": 2.16}
LSEL = [10, 20, 30]
FIELDS = ["T", "E", "TE"]
FSKY = [1.0, 0.7]
S_ALL = sorted(set(S_MAIN + S_FINE))
# A: task benchmark (sigma_E=0.0589); A587: coordinator's quoted SM value (0.0587); B: Maxwell toy
SIG_REPORT = {"A": SIG["A"], "A587": dict(E=0.0587, T=0.0066), "B": SIG["B"]}
CONV2_S, CONV2_X = (0.985, 1.0, 1.05), (160.0, 240.0, 320.0)


def combos():
    for f in FIELDS:
        for l in LSEL:
            for nz in (["none"] if f == "T" else list(NOISE)):
                yield f, l, nz


def cname(f, l, nz):
    return f"{f}_l{l}_{nz}"


_TC = {}


def tcov():
    if "tc" not in _TC:
        tr = Transfer(LMAX)
        _TC["tc"] = TorusCov(tr, LMAX)
        _TC["cover"] = _TC["tc"].cover()
    return _TC["tc"], _TC["cover"]


def sel(tc, C, p, ix, sc, nz=None):
    c = C[p][np.ix_(ix, ix)] * sc[:, None] * sc[None, :]
    if nz is not None and nz != "none":
        nd = tc.noise_diag(p, noise_EE(NOISE[nz], tc.tr.nl))[ix] * sc ** 2
        c[np.diag_indices(len(ix))] += nd
    return c


def exists(name):
    return (CACHE / (name + ".npz")).exists()


def analyze_config(cfg):
    """Topology KL and diagonal Fisher (X_d, X_12) for one (s, xmax, taper, wmode)."""
    s, x, taper, wmode = cfg
    tc, cover = tcov()
    kb = key("base", s, x, taper, wmode)
    if not exists(kb):
        return cfg, None
    C0 = load(kb)[0]["C0"]
    der = {}
    for g in ("Xd", "X12"):
        kf = key("fd", s, x, taper, wmode, g)
        if exists(kf):
            d = load(kf)[0]
            der[g] = (d["dCh"], d["dC2h"])
    res = {}
    for f, l, nz in combos():
        r = {}
        kl, klr = 0.0, 0.0
        acc = {g: np.zeros(3) for g in der}
        cross = 0.0
        for p in (0, 1):
            ix = tc.select(p, l, f)
            sc = tc.scale(p)[ix]
            c0 = sel(tc, C0, p, ix, sc, nz)
            cc = sel(tc, cover, p, ix, sc, nz)
            kl += kl_gauss(c0, cc)[0]
            klr += kl_gauss(cc, c0)[0]
            if der:
                L = cholesky(c0, lower=True, check_finite=False)
                TR = {}
                for g, (dh, d2h) in der.items():
                    th = _whiten(L, sel(tc, dh, p, ix, sc))
                    t2 = _whiten(L, sel(tc, d2h, p, ix, sc))
                    tr_ = (4 * th - t2) / 3
                    acc[g] += 0.5 * np.array([np.sum(th * th), np.sum(t2 * t2), np.sum(tr_ * tr_)])
                    TR[g] = tr_
                if len(TR) == 2:
                    cross += 0.5 * float(np.sum(TR["Xd"] * TR["X12"]))
        r["KL_topology"] = float(kl)
        r["KL_topology_reverse"] = float(klr)
        for g, lab in (("Xd", "E"), ("X12", "T")):
            if g in acc:
                r[f"F_{lab}_h"], r[f"F_{lab}_2h"], r[f"F_{lab}"] = map(float, acc[g])
                r[f"FD_relerr_{lab}"] = float(abs(acc[g][0] - acc[g][2]) / acc[g][2])
        if len(der) == 2:
            r["F_cross_E_T_normalized"] = float(cross / np.sqrt(acc["Xd"][2] * acc["X12"][2]))
        res[cname(f, l, nz)] = r
    return cfg, res


def analyze_full():
    """8x8 Fisher (5 shape + 3 rotation generators) at L = dLSS, h = 1e-3."""
    tc, _ = tcov()
    s, x = 1.0, XMAIN
    gens = SHAPE + ROTATIONS
    d = {}
    for g in gens:
        kind = "fd" if g in ("Xd", "X12") else "fd1"
        kf = key(kind, s, x, 0.25, "deformed", g)
        if not exists(kf):
            return None
        d[g] = load(kf)[0]["dCh"]
    C0 = load(key("base", s, x))[0]["C0"]
    out = {}
    for f, l, nz in [("T", 30, "none"), ("E", 30, "none"), ("TE", 30, "none"), ("TE", 20, "none"),
                     ("TE", 10, "none"), ("TE", 30, "P60")]:
        F = np.zeros((8, 8))
        for p in (0, 1):
            ix = tc.select(p, l, f)
            sc = tc.scale(p)[ix]
            L = cholesky(sel(tc, C0, p, ix, sc, nz), lower=True, check_finite=False)
            T = [_whiten(L, sel(tc, d[g], p, ix, sc)) for g in gens]
            for a in range(8):
                for b in range(a, 8):
                    F[a, b] += 0.5 * float(np.sum(T[a] * T[b]))
        F = np.triu(F) + np.triu(F, 1).T
        dg = np.sqrt(np.diag(F))
        Fn = F / dg[:, None] / dg[None, :]
        off = Fn - np.diag(np.diag(Fn))
        out[cname(f, l, nz)] = dict(
            generators=gens, F=F.tolist(), F_normalized=Fn.tolist(),
            max_abs_normalized_offdiag=float(np.max(np.abs(off))),
            max_abs_normalized_shape_rotation=float(np.max(np.abs(Fn[:5, 5:]))),
            E_degeneracy_Xd2_over_Xd=float(F[1, 1] / F[0, 0]),
            T2_degeneracy_X13_over_X12=float(F[3, 3] / F[2, 2]),
            T2_degeneracy_X23_over_X12=float(F[4, 4] / F[2, 2]),
            T1_degeneracy_J13_over_J12=float(F[6, 6] / F[5, 5]),
            T1_degeneracy_J23_over_J12=float(F[7, 7] / F[5, 5]),
            F_E=float(F[0, 0]), F_T2=float(F[2, 2]), F_rot=float(F[5, 5]))
    return out


def analyze_exact(s):
    """KL between deformed and cube covariances vs the Fisher quadratic form."""
    tc, _ = tcov()
    x = XMAIN
    kb = key("base", s, x)
    if not exists(kb):
        return s, None
    C0 = load(kb)[0]["C0"]
    out = {}
    jobs = []
    for b in SIG:
        jobs.append((f"Xd_bench{b}", key("defm", s, x, 0.25, "deformed", "Xd", f"_bench{b}")))
        jobs.append((f"draw_bench{b}", key("defm", s, x, 0.25, "deformed", "draw", f"_bench{b}")))
    for lab, kn in jobs:
        if not exists(kn):
            continue
        Cb, meta = load(kn)
        Cb = Cb["Cb"]
        r = dict(meta_extra=meta["extra"])
        for f, l, nz in combos():
            kf, kr = 0.0, 0.0
            for p in (0, 1):
                ix = tc.select(p, l, f)
                sc = tc.scale(p)[ix]
                a = sel(tc, Cb, p, ix, sc, nz)
                c = sel(tc, C0, p, ix, sc, nz)
                kf += kl_gauss(a, c)[0]
                kr += kl_gauss(c, a)[0]
            r[cname(f, l, nz)] = dict(KL_deformed_vs_cube=float(kf), KL_cube_vs_deformed=float(kr))
        out[lab] = r
    return s, out


def crossing(svals, yvals, thr):
    """Largest s at which y >= thr, log-linear interpolation; None if never, '>max' if always."""
    s = np.asarray(svals, float); y = np.asarray(yvals, float)
    ok = y >= thr
    if not ok.any():
        return None
    i = int(np.flatnonzero(ok)[-1])
    if i == len(s) - 1:
        return f">{s[-1]:g}"
    y0, y1 = np.log(y[i]), np.log(y[i + 1])
    return float(s[i] + (np.log(thr) - y0) / (y1 - y0) * (s[i + 1] - s[i]))


def main():
    t0 = time.perf_counter()
    cfgs = [(s, XMAIN, 0.25, "deformed") for s in S_ALL]
    cfgs += [(s, x, 0.25, "deformed") for s in S_CONV for x in X_CONV]
    cfgs += [(s, x, 0.25, "deformed") for s in CONV2_S for x in CONV2_X]
    cfgs += [(1.0, XMAIN, 0.10, "deformed"), (1.0, XMAIN, 1e-9, "undeformed"), (1.0, XMAIN, 0.25, "undeformed")]
    cfgs = list(dict.fromkeys(cfgs))
    if "pool" in sys.argv[1:]:
        from multiprocessing import Pool
        with Pool(int(os.environ.get("DNEC_WORKERS", "8"))) as pool:  # package: --workers
            cres = dict(pool.map(analyze_config, cfgs))
            eres = dict(pool.map(analyze_exact, S_ALL))
    else:
        cres = dict(map(analyze_config, cfgs))
        eres = dict(map(analyze_exact, S_ALL))
    full = analyze_full()
    print(f"analysis {time.perf_counter() - t0:.1f}s", flush=True)
    tr = Transfer(LMAX)
    chi = tr.chi
    # ---------------- main table / detectability ----------------
    main_ = {}
    rows = []
    for s in S_ALL:
        r = cres.get((s, XMAIN, 0.25, "deformed"))
        if not r:
            continue
        main_[f"{s:g}"] = {}
        for f, l, nz in combos():
            q = r[cname(f, l, nz)]
            if "F_E" not in q or "F_T" not in q:
                continue
            for fs in FSKY:
                FE, FT = fs * q["F_E"], fs * q["F_T"]
                sE, sT = FE ** -0.5, FT ** -0.5
                d = dict(L_over_dLSS=s, L0_Mpc=2 * chi * s, fields=f, lmax=l, noise=nz, fsky=fs,
                         F_E=FE, F_T=FT, sigmaE_Fisher=sE, sigmaT_Fisher=sT,
                         min_beta_d_3sigma=3 * sE, min_ln_axis_ratio_3sigma=3 * np.sqrt(2) * sE,
                         KL_topology=q["KL_topology"], FD_relerr_E=q["FD_relerr_E"], FD_relerr_T=q["FD_relerr_T"],
                         note="F_E not converged in x_max (matched-circle threshold)" if abs(s - 1.0) < 1e-9 else "")
                for b, sg in SIG_REPORT.items():
                    d[f"dKL_{b}"] = 0.5 * (2 * FE * sg["E"] ** 2 + 3 * FT * sg["T"] ** 2)
                    d[f"SNR_E_{b}"] = sg["E"] / sE
                    d[f"SNR_T_{b}"] = sg["T"] / sT
                main_[f"{s:g}"][f"{cname(f, l, nz)}_fsky{fs:g}"] = d
                rows.append(d)
    # ---------------- crossings (range where the fossil is measurable / detectable) ----------------
    cross = {}
    for f, l, nz in combos():
        for fs in FSKY:
            tag = f"{cname(f, l, nz)}_fsky{fs:g}"
            ss = [s for s in S_ALL if f"{s:g}" in main_ and tag in main_[f"{s:g}"] and s >= 1.0]
            if len(ss) < 2:
                continue
            ent = {}
            for b in SIG_REPORT:
                for m, thr in (("SNR_E", 1.0), ("SNR_E", 3.0), ("SNR_T", 1.0), ("dKL", 1.0), ("dKL", 5.0)):
                    ys = [main_[f"{s:g}"][tag][f"{m}_{b}"] for s in ss]
                    ent[f"{m}_{b}>={thr:g}"] = crossing(ss, ys, thr)
            ys = [main_[f"{s:g}"][tag]["KL_topology"] for s in ss]
            ent["KL_topology>=1"] = crossing(ss, ys, 1.0)
            cross[tag] = ent
    # ---------------- convergence ----------------
    conv = {}
    for s in sorted(set(S_CONV) | set(CONV2_S)):
        conv[f"{s:g}"] = {}
        for x in sorted(set(X_CONV) | set(CONV2_X)):
            r = cres.get((s, x, 0.25, "deformed"))
            if r:
                conv[f"{s:g}"][f"{x:g}"] = {c: {k: r[c][k] for k in ("F_E", "F_T", "KL_topology", "FD_relerr_E", "FD_relerr_T") if k in r[c]}
                                            for c in ("T_l30_none", "E_l30_none", "TE_l30_none", "TE_l20_none", "TE_l10_none",
                                                      "T_l20_none", "T_l10_none", "TE_l30_P60")}
    variants = {}
    for cfg in [(1.0, XMAIN, 0.25, "deformed"), (1.0, XMAIN, 0.10, "deformed"),
                (1.0, XMAIN, 1e-9, "undeformed"), (1.0, XMAIN, 0.25, "undeformed")]:
        r = cres.get(cfg)
        if r:
            variants[f"taper{cfg[2]:g}_{cfg[3]}"] = {c: {k: r[c][k] for k in ("F_E", "F_T", "KL_topology") if k in r[c]}
                                                    for c in ("T_l30_none", "E_l30_none", "TE_l30_none", "TE_l20_none")}
    # lmax saturation of F at selected L (l<=10, 20, 30 from the same covariances)
    lsat = {}
    for s in S_ALL:
        r = cres.get((s, XMAIN, 0.25, "deformed"))
        if r:
            lsat[f"{s:g}"] = {f: {f"l{l}": dict(F_E=r[cname(f, l, 'none')].get("F_E"), F_T=r[cname(f, l, 'none')].get("F_T"))
                                  for l in LSEL} for f in FIELDS}
    # ---------------- exact KL check ----------------
    exact = {}
    for s in S_ALL:
        e = eres.get(s)
        r = cres.get((s, XMAIN, 0.25, "deformed"))
        if not e or not r:
            continue
        exact[f"{s:g}"] = {}
        for lab, q in e.items():
            ent = {"deformation": q["meta_extra"]}
            for c in q:
                if c == "meta_extra":
                    continue
                FE, FT = r[c]["F_E"], r[c]["F_T"]
                if lab.startswith("Xd"):
                    amp = q["meta_extra"]["amp"]
                    quad = 0.5 * FE * amp ** 2
                else:
                    b5 = np.array(q["meta_extra"]["beta5"])
                    quad = 0.5 * float(np.sum(np.array([FE, FE, FT, FT, FT]) * b5 ** 2))
                ent[c] = dict(KL_exact=q[c]["KL_deformed_vs_cube"], KL_exact_reverse=q[c]["KL_cube_vs_deformed"],
                              KL_quadratic=quad, ratio_exact_over_quadratic=q[c]["KL_deformed_vs_cube"] / quad)
            exact[f"{s:g}"][lab] = ent
    # ---------------- topology KL table ----------------
    kltab, kltab_rev = {}, {}
    for s in S_ALL:
        r = cres.get((s, XMAIN, 0.25, "deformed"))
        if r:
            kltab[f"{s:g}"] = {c: r[c]["KL_topology"] for c in r}
            kltab_rev[f"{s:g}"] = {c: r[c]["KL_topology_reverse"] for c in r}
    tj = json.loads((OUT / "C59_transfer.json").read_text(encoding="utf-8"))
    val = {}
    for nm in ("units", "parity", "continuum"):
        fn = OUT / "validate" / f"C59_validate_{nm}.json"
        if fn.exists():
            val[nm] = json.loads(fn.read_text(encoding="utf-8"))
    out = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), script=os.path.abspath(__file__),
        conventions=dict(dLSS_Mpc=2 * chi, chi_star_Mpc=chi, s="L0/d_LSS with L0 = V^(1/3); cube L_short = L0",
                         fixed="volume V = L0^3 fixed under shape changes (det G = 1)",
                         xmax_main=XMAIN, taper_fraction=0.25, lmax_computed=LMAX, fd_steps=[H, 2 * H],
                         noise_uK_arcmin=NOISE, T_noise="cosmic-variance limited", fsky=FSKY,
                         fsky_rule="F -> fsky F (mode counting)", benchmarks=SIG_REPORT,
                         fisher="F=1/2 Tr[C^-1 dC C^-1 dC], fixed V, cube, X_d (E irrep), X_12 (T2 irrep)",
                         dKL="<dKL> = 1/2 (2 F_E sigma_E^2 + 3 F_T sigma_T^2)",
                         beams="neglected (b_l^2 >= 0.978 at l<=40 for FWHM <= 30 arcmin)"),
        transfer=dict(version=tj.get("version"), normalization_check=tj["normalization_check"],
                      normalization_check_raw_camb_transfer=tj.get("normalization_check_raw_camb_transfer"),
                      low_k_truncation_fix={k: v for k, v in tj.get("low_k_truncation_fix", {}).items()
                                            if k != "continuation_validation_l14_17"}),
        validation=val,
        topology_KL=kltab, topology_KL_reverse_cover_vs_torus=kltab_rev, detectability=main_,
        crossings_L_over_dLSS=cross, convergence_xmax=conv, window_variants_L1=variants, lmax_saturation=lsat,
        block_structure_L1=full, exact_KL_check=exact,
        raw_config_results={f"s{c[0]:g}_x{c[1]:g}_t{c[2]:g}_{c[3]}": v for c, v in cres.items() if v})
    (OUT / "C59_shape_fisher.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    # ---------------- CSV ----------------
    cols = ["L_over_dLSS", "fields", "lmax", "noise", "fsky", "sigmaE_Fisher", "sigmaT_Fisher", "F_E", "F_T",
            "dKL_A", "SNR_E_A", "SNR_T_A", "dKL_A587", "SNR_E_A587", "SNR_T_A587", "dKL_B", "SNR_E_B", "SNR_T_B",
            "min_beta_d_3sigma", "min_ln_axis_ratio_3sigma", "KL_topology", "FD_relerr_E", "FD_relerr_T", "note"]
    with open(OUT / "C59_shape_fisher_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in rows:
            w.writerow([d[c] if isinstance(d[c], str) else (f"{d[c]:.6g}") for c in cols])
    # ---------------- figure arrays ----------------
    fig = {"L_over_dLSS": np.array([s for s in S_ALL if f"{s:g}" in main_])}
    for f, l, nz in combos():
        for fs in FSKY:
            tag = f"{cname(f, l, nz)}_fsky{fs:g}"
            vals = [main_[f"{s:g}"].get(tag) for s in fig["L_over_dLSS"]]
            if any(v is None for v in vals):
                continue
            for k in ("sigmaE_Fisher", "sigmaT_Fisher", "dKL_A", "dKL_B", "dKL_A587", "SNR_E_A", "SNR_T_A",
                      "SNR_E_B", "SNR_T_B", "SNR_E_A587", "KL_topology"):
                fig[f"{tag}__{k}"] = np.array([v[k] for v in vals])
    fig["topology_KL_reverse__TE_l30_none"] = np.array([kltab_rev[f"{s:g}"]["TE_l30_none"] for s in fig["L_over_dLSS"]])
    fig["topology_KL_reverse__T_l30_none"] = np.array([kltab_rev[f"{s:g}"]["T_l30_none"] for s in fig["L_over_dLSS"]])
    fig["benchmark_names"] = np.array(list(SIG_REPORT))
    fig["benchmark_sigma_E"] = np.array([v["E"] for v in SIG_REPORT.values()])
    fig["benchmark_sigma_T"] = np.array([v["T"] for v in SIG_REPORT.values()])
    np.savez(OUT / "C59_shape_fisher_figure.npz", **fig)
    # ---------------- console summary ----------------
    print("topology KL (noise-free, full sky):")
    for s in kltab:
        q = kltab[s]
        print(f"  L/dLSS={s:>6}: T20={q['T_l20_none']:.3g} T30={q['T_l30_none']:.3g} "
              f"TE20={q['TE_l20_none']:.3g} TE30={q['TE_l30_none']:.3g}")
    print("T+E l30 CV fsky1: s, sigE_F, sigT_F, SNR_E_A, SNR_T_A, SNR_E_B, SNR_T_B, dKL_A, dKL_B")
    for s in main_:
        d = main_[s].get("TE_l30_none_fsky1")
        if d:
            print(f"  {s:>6} {d['sigmaE_Fisher']:.4g} {d['sigmaT_Fisher']:.4g} {d['SNR_E_A']:.3g} {d['SNR_T_A']:.3g} "
                  f"{d['SNR_E_B']:.3g} {d['SNR_T_B']:.3g} {d['dKL_A']:.3g} {d['dKL_B']:.3g}  FDerr {d['FD_relerr_E']:.1e} {d['FD_relerr_T']:.1e}")
    for tag in ("TE_l30_none_fsky1", "TE_l30_P60_fsky0.7", "T_l30_none_fsky1"):
        print(tag, cross.get(tag))


if __name__ == "__main__":
    main()
