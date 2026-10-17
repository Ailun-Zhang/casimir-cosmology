# PROVENANCE: 04_calculations/code/20261009_claude_R5/nonlinear_SM/C60_analyze.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Summary statistics of the free-field ensembles (C60_summary.json: survival, NL/linear variance ratios, geometry, median side ratio 1.10). Paper: PRD Sec. VIII D (tab:s-fossil, free-field row).

C60_analyze: summary statistics of the Wigner-surrogate ensembles (reads results/.../draws/*.json)."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shapelib"))  # package: C60 modules
from C60_nonlinear_integrator import linear_transfer, geometry
from C60_twisted_ewald import OUT, BASIS, shape_from_beta
from C60_run_main import DELTA, S_NAT

DRAWS = OUT / "draws"
TYPE = np.array([0, 0, 1, 1, 1])          # 0 aspect, 1 angle


def load(tag):
    recs = [json.loads(p.read_text()) for p in sorted(DRAWS.glob(tag + "_*.json"))]
    return sorted(recs, key=lambda r: r["i"])


def cp_interval(k, n, conf=0.95):
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta_dist.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a, k + 1, n - k)
    return float(lo), float(hi)


def lin_coeffs():
    v = json.loads((OUT / "C60_validation_ewald.json").read_text())["b_cube_SM"]
    ka, kt = v["kappa_E"], v["kappa_T"]
    Ma, Mt = linear_transfer(ka)[0], linear_transfer(kt)[0]
    return {"kappa_E": ka, "kappa_T": kt, "aspect": Ma.tolist(), "angle": Mt.tolist()}


def stats_set(tag, lc, nboot=2000, seed=1):
    recs = load(tag)
    N = len(recs)
    out = {"tag": tag, "N": N}
    if N == 0:
        return out
    st = {}
    for r in recs:
        st[r["status"]] = st.get(r["status"], 0) + 1
    out["status_counts"] = st
    surv = [r for r in recs if r.get("survivor")]
    ns = len(surv)
    out["n_survivors"] = ns
    out["survival_fraction"] = ns / N
    out["survival_CP95"] = cp_interval(ns, N)
    done = [r for r in recs if r["status"] in ("complete",)]
    if done:
        out["gates"] = {"max_constraint": max(r["gate_constraint_max"] for r in done),
                        "max_detG_minus1": max(r["gate_detG_minus1_max"] for r in done),
                        "max_G_change_10_12": max(r["gate_G_change_10_to_12"] for r in done),
                        "max_trGinvM": max(r["gate_trGinvM_max"] for r in done),
                        "all_gates_pass": int(sum(r["gates_pass"] for r in done)), "n_complete": len(done)}
    out["runtime_mean_s"] = float(np.mean([r["runtime_s"] for r in recs]))
    if ns == 0:
        return out
    ks, amp = surv[0]["ks"], surv[0]["amp"]
    B0 = np.array([r["beta0"] for r in surv])
    P0 = np.array([r["Pi0"] for r in surv])
    BP = np.array([r["beta_plus"] for r in surv])
    m11 = np.array([lc["aspect"][0] if t == 0 else lc["angle"][0] for t in TYPE])
    m12 = np.array([lc["aspect"][1] if t == 0 else lc["angle"][1] for t in TYPE])
    BL = m11 * B0 + m12 * P0
    s = ks * S_NAT
    var_ana = amp ** 2 * (m11 ** 2 * s + m12 ** 2 * DELTA ** 2 / s)
    sq_nl, sq_lin = BP ** 2, BL ** 2
    var_nl, var_lin = sq_nl.mean(0), sq_lin.mean(0)
    asp, ang = TYPE == 0, TYPE == 1
    out["per_mode"] = {"var_NL": var_nl.tolist(), "var_lin_analytic": var_ana.tolist(), "var_lin_paired_sample": var_lin.tolist(),
                       "ratio_NL_over_lin_paired": (var_nl / var_lin).tolist(), "ratio_NL_over_analytic": (var_nl / var_ana).tolist(),
                       "mean_beta_plus": BP.mean(0).tolist(), "mean_beta_plus_stderr": (BP.std(0, ddof=1) / np.sqrt(ns)).tolist()}

    def rms(a, mask):
        return float(np.sqrt(a[:, mask].mean()))
    out["rms"] = {"aspect_NL": rms(sq_nl, asp), "angle_NL": rms(sq_nl, ang), "total5_NL": float(np.sqrt(sq_nl.sum(1).mean())),
                  "aspect_lin_paired": rms(sq_lin, asp), "angle_lin_paired": rms(sq_lin, ang), "total5_lin_paired": float(np.sqrt(sq_lin.sum(1).mean())),
                  "aspect_lin_analytic": float(np.sqrt(var_ana[asp].mean())), "angle_lin_analytic": float(np.sqrt(var_ana[ang].mean())),
                  "total5_lin_analytic": float(np.sqrt(var_ana.sum()))}
    # paired differences
    D = BP - BL
    out["paired_diff"] = {"rel_rms_all": float(np.sqrt((D ** 2).sum() / (BL ** 2).sum())),
                          "rel_rms_aspect": float(np.sqrt((D[:, asp] ** 2).sum() / (BL[:, asp] ** 2).sum())),
                          "rel_rms_angle": float(np.sqrt((D[:, ang] ** 2).sum() / (BL[:, ang] ** 2).sum())),
                          "per_draw_rel_norm_median": float(np.median(np.linalg.norm(D, axis=1) / np.linalg.norm(BL, axis=1))),
                          "per_draw_rel_norm_p90": float(np.percentile(np.linalg.norm(D, axis=1) / np.linalg.norm(BL, axis=1), 90)),
                          "per_draw_rel_norm_max": float(np.max(np.linalg.norm(D, axis=1) / np.linalg.norm(BL, axis=1))),
                          "max_abs_diff": float(np.max(np.abs(D)))}
    # bootstrap of variance ratios
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ns, (nboot, ns))
    num_t = sq_nl.sum(1)
    den_t = sq_lin.sum(1)

    def ratio_boot(num, den):
        r = num[idx].mean(1) / den[idx].mean(1)
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    out["variance_ratio_NL_over_lin_paired"] = {
        "aspect_pooled": float(sq_nl[:, asp].sum() / sq_lin[:, asp].sum()),
        "aspect_pooled_boot95": ratio_boot(sq_nl[:, asp].sum(1), sq_lin[:, asp].sum(1)),
        "angle_pooled": float(sq_nl[:, ang].sum() / sq_lin[:, ang].sum()),
        "angle_pooled_boot95": ratio_boot(sq_nl[:, ang].sum(1), sq_lin[:, ang].sum(1)),
        "total": float(num_t.sum() / den_t.sum()), "total_boot95": ratio_boot(num_t, den_t)}
    out["variance_ratio_NL_over_analytic"] = {
        "aspect_pooled": float(sq_nl[:, asp].mean() / var_ana[asp].mean()), "angle_pooled": float(sq_nl[:, ang].mean() / var_ana[ang].mean()),
        "total": float(sq_nl.sum(1).mean() / var_ana.sum())}
    # covariance
    C_nl = BP.T @ BP / ns
    C_lin = BL.T @ BL / ns
    sd = np.sqrt(np.diag(C_nl))
    corr = C_nl / np.outer(sd, sd)
    sdl = np.sqrt(np.diag(C_lin))
    corrl = C_lin / np.outer(sdl, sdl)
    off = ~np.eye(5, dtype=bool)
    out["covariance"] = {"C_NL": C_nl.tolist(), "C_lin_paired": C_lin.tolist(), "C_lin_analytic_diag": var_ana.tolist(),
                         "frob_rel_diff_NL_vs_lin_paired": float(np.linalg.norm(C_nl - C_lin) / np.linalg.norm(C_lin)),
                         "max_offdiag_corr_NL": float(np.max(np.abs(corr[off]))), "max_offdiag_corr_lin_paired": float(np.max(np.abs(corrl[off]))),
                         "expected_noise_corr_1_over_sqrtN": float(1 / np.sqrt(ns))}
    # amplitude
    out["max_abs_beta_plus"] = float(np.max(np.abs(BP)))
    out["max_abs_beta_plus_lin_paired"] = float(np.max(np.abs(BL)))
    out["max_abs_beta_path"] = float(max(r["max_abs_beta_path"] for r in surv))
    # geometry
    geo = [r["geometry_plus"] for r in surv]
    Gl = [geometry(shape_from_beta(b)) for b in BL]
    out["geometry"] = {
        "lnL_median": float(np.median([g["ln_Lmax_over_Lmin"] for g in geo])), "lnL_max": float(np.max([g["ln_Lmax_over_Lmin"] for g in geo])),
        "angle_dev_deg_median": float(np.median([g["max_angle_deviation_deg"] for g in geo])), "angle_dev_deg_max": float(np.max([g["max_angle_deviation_deg"] for g in geo])),
        "lnL_median_lin_paired": float(np.median([g["ln_Lmax_over_Lmin"] for g in Gl])), "lnL_max_lin_paired": float(np.max([g["ln_Lmax_over_Lmin"] for g in Gl])),
        "angle_dev_deg_median_lin_paired": float(np.median([g["max_angle_deviation_deg"] for g in Gl])),
        "angle_dev_deg_max_lin_paired": float(np.max([g["max_angle_deviation_deg"] for g in Gl]))}

    # cubic invariant tr(B^3) skewness (prolate/oblate asymmetry), zero for the linear Gaussian map
    def tr3(Bv):
        Bm = np.einsum("na,aij->nij", Bv, BASIS)
        return np.einsum("nij,njk,nki->n", Bm, Bm, Bm)

    def skew(Bv):
        t3 = tr3(Bv)
        t2 = (Bv ** 2).sum(1)
        return float(t3.mean() / (t2.mean() ** 1.5)), t3, t2
    sk_nl, t3, t2 = skew(BP)
    sk_lin, t3l, t2l = skew(BL)
    bs = (t3[idx].mean(1) / t2[idx].mean(1) ** 1.5)
    out["cubic_invariant_skewness"] = {"NL": sk_nl, "NL_boot95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                       "lin_paired": sk_lin, "note": "<tr B^3>/<tr B^2>^{3/2}, B = sum beta_A X_A"}
    # relative departure vs |beta_lin| (nonlinearity scaling): regress per-draw rel diff on |BL|
    nb = np.linalg.norm(BL, axis=1)
    nd = np.linalg.norm(D, axis=1)
    out["departure_vs_amplitude"] = {"slope_norm_diff_over_norm_lin_vs_norm_lin": float(np.polyfit(nb, nd / nb, 1)[0]),
                                     "median_norm_lin": float(np.median(nb)), "max_norm_lin": float(np.max(nb)),
                                     "corr_reldiff_normlin": float(np.corrcoef(nb, nd / nb)[0, 1])}
    out["angle_over_aspect_norm_median"] = {
        "NL": float(np.median(np.linalg.norm(BP[:, 2:], axis=1) / np.linalg.norm(BP[:, :2], axis=1))),
        "lin_paired": float(np.median(np.linalg.norm(BL[:, 2:], axis=1) / np.linalg.norm(BL[:, :2], axis=1)))}
    out["_arrays"] = {"BP": BP, "BL": BL, "B0": B0, "P0": P0, "idx": [r["i"] for r in surv]}
    return out


def write_table(tag, s):
    a = s["_arrays"]
    hdr = ["i"] + ["beta0_%d" % k for k in range(5)] + ["Pi0_%d" % k for k in range(5)] + \
          ["beta_plus_NL_%d" % k for k in range(5)] + ["beta_plus_lin_%d" % k for k in range(5)]
    rows = [",".join(hdr)]
    for n, i in enumerate(a["idx"]):
        vals = [i] + list(a["B0"][n]) + list(a["P0"][n]) + list(a["BP"][n]) + list(a["BL"][n])
        rows.append(",".join(("%d" % v) if k == 0 else ("%.10e" % v) for k, v in enumerate(vals)))
    (OUT / ("C60_%s_table.csv" % tag)).write_text("\n".join(rows) + "\n")


def compare_to_main(tag, ref="main"):
    a = {r["i"]: r for r in load(tag)}
    b = {r["i"]: r for r in load(ref)}
    ids = sorted(set(a) & set(b))
    rows = []
    for i in ids:
        if a[i].get("survivor") and b[i].get("survivor"):
            ba, bb = np.array(a[i]["beta_plus"]), np.array(b[i]["beta_plus"])
            rows.append((np.linalg.norm(ba - bb), np.linalg.norm(bb), np.max(np.abs(ba - bb))))
    if not rows:
        return {}
    rows = np.array(rows)
    return {"n": len(rows), "n_status_differs": int(sum(1 for i in ids if a[i]["status"] != b[i]["status"])),
            "rel_rms_diff_vs_main": float(np.sqrt((rows[:, 0] ** 2).sum() / (rows[:, 1] ** 2).sum())),
            "max_abs_diff_beta_plus": float(rows[:, 2].max()),
            "max_per_draw_rel_norm_diff": float((rows[:, 0] / rows[:, 1]).max())}


def main():
    lc = lin_coeffs()
    rep = {"linear_coefficients": lc, "sets": {}, "comparisons": {}}
    tags = ["main", "sens3", "amp3"] + [t for t in ("amp5", "amp8", "amp12") if (DRAWS / (t + "_0000.json")).exists()] + ["hi"]
    for tag in tags:
        s = stats_set(tag, lc, nboot=2000 if tag != "hi" else 1000)
        if tag == "main" and "_arrays" in s:
            write_table(tag, s)
        s.pop("_arrays", None)
        rep["sets"][tag] = s
    for tag in ("ic_can", "ic_trf", "conv_tol", "conv_cut"):
        rep["comparisons"][tag] = compare_to_main(tag)
        rep["comparisons"][tag]["status_counts"] = stats_set(tag, lc, nboot=10).get("status_counts")
    (OUT / "C60_summary.json").write_text(json.dumps(rep, indent=1))
    for tag, s in rep["sets"].items():
        if s["N"] == 0:
            continue
        print("== %s: N=%d survivors=%d (%.4f, CP95 %.4f-%.4f) status %s" % (tag, s["N"], s["n_survivors"], s["survival_fraction"], *s["survival_CP95"], s["status_counts"]))
        if s["n_survivors"] == 0:
            continue
        g = s["gates"]
        print("   gates: constraint %.1e detG %.1e Gchange %.1e trGinvM %.1e pass %d/%d" % (g["max_constraint"], g["max_detG_minus1"], g["max_G_change_10_12"], g["max_trGinvM"], g["all_gates_pass"], g["n_complete"]))
        r = s["rms"]
        print("   rms NL aspect %.5f angle %.5f total %.5f | lin paired %.5f %.5f %.5f | lin analytic %.5f %.5f %.5f" % (
            r["aspect_NL"], r["angle_NL"], r["total5_NL"], r["aspect_lin_paired"], r["angle_lin_paired"], r["total5_lin_paired"],
            r["aspect_lin_analytic"], r["angle_lin_analytic"], r["total5_lin_analytic"]))
        v = s["variance_ratio_NL_over_lin_paired"]
        print("   var ratio NL/lin(paired): aspect %.4f [%.4f,%.4f] angle %.4f [%.4f,%.4f] total %.4f [%.4f,%.4f]" % (
            v["aspect_pooled"], *v["aspect_pooled_boot95"], v["angle_pooled"], *v["angle_pooled_boot95"], v["total"], *v["total_boot95"]))
        pm = s["per_mode"]
        print("   per-mode var ratio NL/lin_paired:", np.round(pm["ratio_NL_over_lin_paired"], 3), " NL/analytic:", np.round(pm["ratio_NL_over_analytic"], 3))
        d = s["paired_diff"]
        print("   paired diff rel rms: all %.4f aspect %.4f angle %.4f | per-draw norm median %.4f p90 %.4f max %.4f" % (
            d["rel_rms_all"], d["rel_rms_aspect"], d["rel_rms_angle"], d["per_draw_rel_norm_median"], d["per_draw_rel_norm_p90"], d["per_draw_rel_norm_max"]))
        print("   max|beta_+| NL %.4f (lin paired %.4f, path max %.4f)" % (s["max_abs_beta_plus"], s["max_abs_beta_plus_lin_paired"], s["max_abs_beta_path"]))
        ge = s["geometry"]
        print("   lnLmax/Lmin median %.4f max %.4f (lin %.4f %.4f) | angle dev deg median %.4f max %.4f (lin %.4f %.4f)" % (
            ge["lnL_median"], ge["lnL_max"], ge["lnL_median_lin_paired"], ge["lnL_max_lin_paired"],
            ge["angle_dev_deg_median"], ge["angle_dev_deg_max"], ge["angle_dev_deg_median_lin_paired"], ge["angle_dev_deg_max_lin_paired"]))
        c = s["covariance"]
        print("   cov: frob rel diff %.4f; max offdiag corr NL %.3f lin %.3f (noise 1/sqrtN %.3f); skew tr B^3: NL %.3f [%.3f,%.3f] lin %.3f" % (
            c["frob_rel_diff_NL_vs_lin_paired"], c["max_offdiag_corr_NL"], c["max_offdiag_corr_lin_paired"], c["expected_noise_corr_1_over_sqrtN"],
            s["cubic_invariant_skewness"]["NL"], *s["cubic_invariant_skewness"]["NL_boot95"], s["cubic_invariant_skewness"]["lin_paired"]))
        print("   mean beta_+:", np.round(pm["mean_beta_plus"], 5), "+-", np.round(pm["mean_beta_plus_stderr"], 5),
              "| median |beta_angle|/|beta_aspect| NL %.4f lin %.4f | dev slope %.3f" % (
                  s["angle_over_aspect_norm_median"]["NL"], s["angle_over_aspect_norm_median"]["lin_paired"],
                  s["departure_vs_amplitude"]["slope_norm_diff_over_norm_lin_vs_norm_lin"]))
    for tag, c in rep["comparisons"].items():
        print("-- %s vs main: %s" % (tag, c))


if __name__ == "__main__":
    main()
