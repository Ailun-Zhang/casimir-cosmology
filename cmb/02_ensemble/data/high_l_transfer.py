r"""C59_transfer: CAMB scalar T and E transfer functions, l = 2..40, on a fine k grid.

Normalization convention (checked against CAMB's own unlensed C_l):
    C_l^{XY} = 4 pi \int dln k  calP(k) Delta^X_l(k) Delta^Y_l(k),    X,Y in {T,E}
    calP(k)  = A_s (k/k_pivot)^(n_s-1),  P_zeta(k) = 2 pi^2 calP(k)/k^3.
CAMB 2.0.4 get_cmb_transfer_data('scalar') returns sources [T, E]; the E row omits the
spin-2 factor sqrt((l+2)!/(l-2)!), which is multiplied in here, so Delta^E below is the
full E-mode transfer (TE sign follows CAMB's convention).

FIX (v2, 2026-10-10): CAMB 2.0.4 returns Delta_l(k) == 0 below a fixed node k chi* = 14.217 for
18 <= l <= 26 and jumps to a finite value there (9.0%, 4.2%, 1.8%, 0.73%, 0.28%, 0.10% of
max|Delta_l| for l = 18..23).  Accuracy boosts do not move it (C59_transfer_cutoff_scan.py).  CAMB's
own C_l carry the same truncation, so the C_l check cannot see it, but on a lattice the discontinuity
produces spurious shape derivatives for modes near x = 14.2.  Below the first computed node x_c(l)
we now continue Delta_l(x) = g_l(x) * poly2(x - x_c), poly2 fitted to Delta_l/g_l on [x_c, x_c+3],
g_l = j_l(x) (T) and j_l(x)/x^2 (E).  The method is validated on l = 14..17 (full CAMB range) by
artificial truncation at the same relative jump.  Patching is applied only where the jump exceeds 1e-6.

Output (results/20261009_claude_R5/obs_fisher):
    C59_transfer.npz  : x = k chi_* uniform grid (step 0.01, 0..400), DT[l-2,i], DE[l-2,i],
                        tails beyond x=400, native CAMB grid, CAMB unlensed raw C_l.
    C59_transfer.json : metadata + normalization check + low-k patch record and validation.
Run with PYTHONPATH pointing at the project runtime_libs (CAMB 2.0.4).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

COSMO = dict(H0=67.4, ombh2=0.02237, omch2=0.1200, mnu=0.06, nnu=3.044, omk=0.0,
             tau=0.0544, TCMB=2.7255)
PRIM = dict(As=2.1e-9, ns=0.9649, pivot=0.05)
LMAX_T = 100
XSTEP = 0.01      # fine grid step in x = k chi_*
XEND = 400.0      # fine grid end; contribution beyond is stored as a tail integral
ACC = dict(AccuracyBoost=3, lAccuracyBoost=3, lSampleBoost=50, IntkAccuracyBoost=3)


def calP(k):
    return PRIM["As"] * (np.asarray(k) / PRIM["pivot"]) ** (PRIM["ns"] - 1.0)


def _shape(x, l, field):
    from scipy.special import spherical_jn
    j = spherical_jn(l, x)
    return j if field == "T" else j / x ** 2


def continue_low_k(xq, d, l, field, x_start=None, width=3.0, tol=1e-6):
    """Fill d (native grid) below the first non-zero node (or below x_start) by g_l * poly2."""
    nz = np.flatnonzero(d != 0)
    i0 = int(nz[0]) if x_start is None else int(np.searchsorted(xq, x_start))
    xc = float(xq[i0])
    jump = float(abs(d[i0]) / np.max(np.abs(d)))
    if jump < tol or i0 == 0:
        return d, dict(l=l, field=field, x_c=xc, jump_over_max=jump, patched=False)
    fit = (xq >= xc) & (xq <= xc + width)
    coef = np.polyfit(xq[fit] - xc, d[fit] / _shape(xq[fit], l, field), 2)
    low = (xq < xc) & (xq > 0.5)
    out = d.copy()
    out[xq < xc] = 0.0
    out[low] = _shape(xq[low], l, field) * np.polyval(coef, xq[low] - xc)
    return out, dict(l=l, field=field, x_c=xc, jump_over_max=jump, patched=True, n_fit=int(fit.sum()),
                     poly_coef=coef.tolist())


def validate_continuation(xq, dT, dE, lv):
    """Truncate l=14..17 (complete in CAMB) where |Delta| first reaches the l=18 jump level, continue,
    compare with CAMB below the truncation point (errors relative to max|Delta_l|)."""
    rows = []
    for l in range(14, 18):
        i = l - 2
        for field, d in (("T", dT[i]), ("E", dE[i])):
            frac = np.abs(d) / np.max(np.abs(d))
            for level in (0.09, 0.04):
                it = int(np.flatnonzero(frac >= level)[0])
                cont, _ = continue_low_k(xq, d, l, field, x_start=xq[it])
                below = (xq < xq[it]) & (xq > 0.5)
                err = np.abs(cont[below] - d[below]) / np.max(np.abs(d))
                rows.append(dict(l=l, field=field, truncation_level=level, x_t=float(xq[it]),
                                 max_err_over_max=float(err.max()),
                                 max_err_over_value_at_xt=float(err.max() / frac[it])))
    return rows


def run_camb():
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(**COSMO)
    pars.InitPower.set_params(As=PRIM["As"], ns=PRIM["ns"], pivot_scalar=PRIM["pivot"], r=0.0)
    pars.set_for_lmax(100, lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.DoLensing = False
    for key, val in ACC.items():
        setattr(pars.Accuracy, key, val)
    t0 = time.perf_counter()
    res = camb.get_results(pars)
    tr = res.get_cmb_transfer_data("scalar")
    der = res.get_derived_params()
    chi = float(res.comoving_radial_distance(der["zstar"]))
    cls = res.get_unlensed_scalar_cls(lmax=120, raw_cl=True)   # TT, EE, BB, TE (dimensionless)
    meta = dict(camb_version=camb.__version__, camb_path=camb.__file__,
                seconds=time.perf_counter() - t0, zstar=float(der["zstar"]),
                rstar=float(der["rstar"]), chi_star_Mpc=chi,
                num_sources=int(tr.NumSources), nq=int(len(tr.q)))
    return tr, cls, chi, meta


def main():
    tr, cls, chi, meta = run_camb()
    L = np.asarray(tr.L)
    lv = np.arange(2, LMAX_T + 1)
    missing = [int(l) for l in lv if l not in L]
    if missing:
        raise RuntimeError(f"CAMB omitted multipoles {missing}")
    ix = [int(np.where(L == l)[0][0]) for l in lv]
    q = np.asarray(tr.q)
    spin2 = np.sqrt(np.array([(l + 2) * (l + 1) * l * (l - 1) for l in lv], float))
    dT_raw = tr.delta_p_l_k[0][ix, :].copy()
    dE_raw = tr.delta_p_l_k[1][ix, :] * spin2[:, None]
    xq = q * chi
    dxq = np.diff(xq)

    # ---- v2: continue CAMB's per-l low-k truncation (see module docstring) ----
    cont_validation = validate_continuation(xq, dT_raw, dE_raw, lv)
    dT_native, dE_native = dT_raw.copy(), dE_raw.copy()
    patches = []
    for i, l in enumerate(lv):
        dT_native[i], pT = continue_low_k(xq, dT_raw[i], int(l), "T")
        dE_native[i], pE = continue_low_k(xq, dE_raw[i], int(l), "E")
        patches += [pT, pE]

    camb_cl = {"TT": cls[lv, 0], "EE": cls[lv, 1], "TE": cls[lv, 3]}

    def integrate(dT, dE):
        x = np.arange(0.0, XEND + 0.5 * XSTEP, XSTEP)
        splT = CubicSpline(xq, dT, axis=1, extrapolate=False)
        splE = CubicSpline(xq, dE, axis=1, extrapolate=False)
        DT = np.nan_to_num(splT(x))   # x < first native node (x~0.03): 0 (Delta_l ~ x^l there)
        DE = np.nan_to_num(splE(x))
        good = x > 0
        w = 4 * np.pi * calP(x[good] / chi)
        lnx = np.log(x[good])
        cl_fine = {name: simpson(w[None, :] * a[:, good] * b[:, good], x=lnx, axis=1)
                   for name, a, b in (("TT", DT, DT), ("EE", DE, DE), ("TE", DT, DE))}
        xt = np.geomspace(XEND, xq[-1], 400001)
        wt = 4 * np.pi * calP(xt / chi)
        aT, aE = splT(xt), splE(xt)
        tail = {"TT": simpson(wt * aT * aT, x=np.log(xt), axis=1),
                "EE": simpson(wt * aE * aE, x=np.log(xt), axis=1),
                "TE": simpson(wt * aT * aE, x=np.log(xt), axis=1)}
        cl = {kk: cl_fine[kk] + tail[kk] for kk in cl_fine}
        rel = {"TT": cl["TT"] / camb_cl["TT"] - 1, "EE": cl["EE"] / camb_cl["EE"] - 1,
               "TE_over_sqrtTTEE": (cl["TE"] - camb_cl["TE"]) / np.sqrt(camb_cl["TT"] * camb_cl["EE"]),
               "TE_plain": cl["TE"] / camb_cl["TE"] - 1}
        return x, DT, DE, splT, splE, cl, tail, rel

    # raw CAMB transfer: tests the normalization convention itself (same truncation as CAMB's C_l)
    _, _, _, _, _, cl_raw, _, rel_raw = integrate(dT_raw, dE_raw)
    # patched transfer: the product used downstream
    x, DT, DE, splT, splE, cl_ours, tail, rel = integrate(dT_native, dE_native)
    # native trapezoid in ln k with CAMB-like weights (diagnostic only, raw transfer)
    wq = np.gradient(q) / q
    native = {"TT": 4 * np.pi * np.sum(calP(q) * dT_raw ** 2 * wq, axis=1),
              "EE": 4 * np.pi * np.sum(calP(q) * dE_raw ** 2 * wq, axis=1)}
    # interpolation self-check: 4-point Lagrange on the fine grid vs native spline at random x
    rng = np.random.default_rng(59)
    xs = rng.uniform(1.0, 390.0, 20000)
    u = xs / XSTEP; i = np.floor(u).astype(int); t = u - i
    w0 = -t * (t - 1) * (t - 2) / 6; w1 = (t + 1) * (t - 1) * (t - 2) / 2
    w2 = -(t + 1) * t * (t - 2) / 2; w3 = (t + 1) * t * (t - 1) / 6
    lagT = DT[:, i - 1] * w0 + DT[:, i] * w1 + DT[:, i + 1] * w2 + DT[:, i + 2] * w3
    lagE = DE[:, i - 1] * w0 + DE[:, i] * w1 + DE[:, i + 1] * w2 + DE[:, i + 2] * w3
    interp_err = dict(T=float(np.max(np.abs(lagT - splT(xs))) / np.max(np.abs(DT))),
                      E=float(np.max(np.abs(lagE - splE(xs))) / np.max(np.abs(DE))))

    np.savez_compressed(OUT / "C59_transfer.npz", x=x, DT=DT, DE=DE, lvals=lv, chi=chi,
                        tail_TT=tail["TT"], tail_EE=tail["EE"], tail_TE=tail["TE"], xend=XEND,
                        xstep=XSTEP, q_native=q, dT_native=dT_native, dE_native=dE_native,
                        dT_native_raw=dT_raw, dE_native_raw=dE_raw,
                        camb_TT=camb_cl["TT"], camb_EE=camb_cl["EE"], camb_TE=camb_cl["TE"],
                        cl_TT=cl_ours["TT"], cl_EE=cl_ours["EE"], cl_TE=cl_ours["TE"],
                        As=PRIM["As"], ns=PRIM["ns"], pivot=PRIM["pivot"], version=2)
    patched = [p for p in patches if p["patched"]]
    out = dict(
        script=str(Path(__file__).resolve()), python=sys.version, numpy=np.__version__, version=2,
        cosmology=COSMO, primordial=PRIM, camb_accuracy=ACC, **meta,
        dLSS_Mpc=2 * chi, lvals=[2, LMAX_T],
        low_k_truncation_fix=dict(
            description="CAMB Delta_l==0 below x_c(l) (x_c=14.217 for 18<=l<=26) with a finite jump; "
                        "continued as g_l*poly2 (g=j_l for T, j_l/x^2 for E) fitted on [x_c, x_c+3]",
            patched=[dict(l=p["l"], field=p["field"], x_c=p["x_c"], jump_over_max=p["jump_over_max"]) for p in patched],
            continuation_validation_l14_17=cont_validation,
            max_validation_err_over_max=float(max(r["max_err_over_max"] for r in cont_validation)),
            cl_change_from_patch_max_rel=dict(
                TT=float(np.max(np.abs(cl_ours["TT"] / cl_raw["TT"] - 1))),
                EE=float(np.max(np.abs(cl_ours["EE"] / cl_raw["EE"] - 1))),
                TE_over_sqrtTTEE=float(np.max(np.abs(cl_ours["TE"] - cl_raw["TE"]) / np.sqrt(cl_raw["TT"] * cl_raw["EE"]))))),
        normalization_check_raw_camb_transfer=dict(
            max_abs_rel_TT=float(np.max(np.abs(rel_raw["TT"]))), max_abs_rel_EE=float(np.max(np.abs(rel_raw["EE"]))),
            max_abs_TE_over_sqrtTTEE=float(np.max(np.abs(rel_raw["TE_over_sqrtTTEE"]))),
            TE_sign_agreement=bool(np.all(np.sign(cl_raw["TE"]) == np.sign(camb_cl["TE"]))),
            pass_1em3=bool(max(np.max(np.abs(rel_raw["TT"])), np.max(np.abs(rel_raw["EE"])),
                               np.max(np.abs(rel_raw["TE_over_sqrtTTEE"]))) < 1e-3)),
        native_grid=dict(x_first=float(xq[0]), x_last=float(xq[-1]),
                         max_dx_below_x300=float(dxq[xq[:-1] < 300].max()),
                         median_dx_below_x300=float(np.median(dxq[xq[:-1] < 300]))),
        fine_grid=dict(step_x=XSTEP, x_end=XEND, n=int(len(x)), interpolation="4-point Lagrange cubic on uniform x grid",
                       lagrange_vs_native_spline_max_rel=interp_err),
        E_normalization="Delta^E = sqrt((l+2)!/(l-2)!) * CAMB delta_p_l_k[1]",
        normalization_formula="C_l^XY = 4 pi int dlnk calP Delta^X Delta^Y",
        normalization_check=dict(
            max_abs_rel_TT=float(np.max(np.abs(rel["TT"]))),
            max_abs_rel_EE=float(np.max(np.abs(rel["EE"]))),
            max_abs_TE_over_sqrtTTEE=float(np.max(np.abs(rel["TE_over_sqrtTTEE"]))),
            max_abs_rel_TE_plain=float(np.max(np.abs(rel["TE_plain"]))),
            TE_sign_agreement=bool(np.all(np.sign(cl_ours["TE"]) == np.sign(camb_cl["TE"]))),
            native_trapezoid_max_rel_TT=float(np.max(np.abs(native["TT"] / camb_cl["TT"] - 1))),
            native_trapezoid_max_rel_EE=float(np.max(np.abs(native["EE"] / camb_cl["EE"] - 1))),
            tail_fraction_beyond_x400_max=dict(TT=float(np.max(np.abs(tail["TT"] / cl_ours["TT"]))),
                                               EE=float(np.max(np.abs(tail["EE"] / cl_ours["EE"])))),
            per_l=dict(l=lv.tolist(), TT=rel["TT"].tolist(), EE=rel["EE"].tolist(),
                       TE_over_sqrtTTEE=rel["TE_over_sqrtTTEE"].tolist()),
            pass_1em3=bool(max(np.max(np.abs(rel["TT"])), np.max(np.abs(rel["EE"])),
                               np.max(np.abs(rel["TE_over_sqrtTTEE"]))) < 1e-3)),
        camb_unlensed_raw_cl=dict(l=lv.tolist(), TT=camb_cl["TT"].tolist(), EE=camb_cl["EE"].tolist(),
                                  TE=camb_cl["TE"].tolist()),
        units="k in Mpc^-1, chi in Mpc, x=k chi_*, Delta and C_l dimensionless (Delta T/T_CMB)^2")
    (OUT / "C59_transfer.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    nc = out["normalization_check"]
    print("chi_* =", chi, " d_LSS =", 2 * chi)
    print("raw CAMB transfer vs CAMB C_l:", out["normalization_check_raw_camb_transfer"])
    print("patched: max rel TT", nc["max_abs_rel_TT"], " EE", nc["max_abs_rel_EE"], " TE/sqrt(TTEE)",
          nc["max_abs_TE_over_sqrtTTEE"], " TE plain", nc["max_abs_rel_TE_plain"], " pass", nc["pass_1em3"])
    fx = out["low_k_truncation_fix"]
    print("patched rows:", [(p["l"], p["field"], round(p["jump_over_max"], 6)) for p in fx["patched"] if p["jump_over_max"] > 1e-4])
    print("continuation validation max err/max:", fx["max_validation_err_over_max"])
    for r in fx["continuation_validation_l14_17"]:
        print("   ", r)
    print("C_l change from patch:", fx["cl_change_from_patch_max_rel"])
    print("native grid", out["native_grid"], " interp", interp_err)


if __name__ == "__main__":
    main()

