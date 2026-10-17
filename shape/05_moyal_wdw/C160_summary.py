# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_tier2/C160_summary.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Collects the four C160 parts into r3/shape_tier2/run01/C160_summary.json (D131).

C160 summary: collect key numbers of the four parts and sha256 of code / main inputs -> run01/C160_summary.json."""
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import json, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package: chain root (shape/)
import _shape_paths as WP  # noqa: E402
ROOT = None  # package: research-tree inputs (C40/C45 toy Moyal code, D-files) are not shipped
RUN = WP.wdir("r3/shape_tier2/run01")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    p = RUN / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    out = {"task": "T8 shape-tier2", "ids": "C160 / D131", "code_sha256": {}, "inputs_sha256": {}}
    for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.wl")):  # package: PROVENANCE.txt -> shape/PACKAGE_PATCHES.json
        out["code_sha256"][p.name] = sha(p)
    ins = ["04_calculations/code/20261007_074655UTC/model_completion/C40_cubic_moyal.py",
           "04_calculations/results/20261007_074655UTC/model_completion/C40_cubic_moyal_results.json",
           "04_calculations/code/20261007_074655UTC/model_completion/C45_actual_cubic_moyal.py",
           "04_calculations/derivations/20261013_claude_v12_r2/shape_nonlinear_threshold/D114_SHAPE_NONLINEAR_THRESHOLD_CN.md",
           "04_calculations/derivations/20261014_claude_v12_r3/shape_hardening/D125_SHAPE_HARDENING_CN.md",
           "04_calculations/derivations/20261013_claude_v12_r2/time_symmetry_origin/D113_TIME_SYMMETRY_ORIGIN_CN.md",
           "04_calculations/derivations/20261013_claude_v12_r2/time_symmetry_origin/D113_R15_CORRECTION_ADDENDUM_CN.md"]
    for p in ins:
        if ROOT is not None and (ROOT / p).exists():
            out["inputs_sha256"][p] = sha(ROOT / p)
    m = load("moyal/C160_moyal.json")
    if m:
        out["moyal"] = {"regression_C40": m["regression_C40"],
                        "SM_cubic_over_Z0_xi030": {k: m["SM"]["xi030"][k] for k in ("c_E3_111", "c_ETT_122", "c_TTT_234")},
                        "rows": {c: [{k: r[k] for k in ("window", "t0", "coef_aspect", "coef_angle", "rel")}
                                     for r in m["SM"][c]["rows"]] for c in m["SM"]}}
    m1 = load("moyal/C160_moyal_1d.json")
    if m1:
        out["moyal_1d"] = [{k: c[k] for k in ("c", "dt", "N", "Q", "quantum_minus_TWA", "formula_delta2_coef",
                                               "ratio_numeric_over_formula", "TWA_minus_lin_rel")} for c in m1["checks"]]
    pl = load("plateau/C160_plateau_summary.json")
    if pl:
        out["plateau"] = {k: {kk: v[kk] for kk in ("n", "frac", "wilson95", "D114_frac", "n_flips", "paired_diff_frac",
                                                   "sigma_lin_aspect_plateau", "sigma_E_NL_survivors",
                                                   "median_rel_dbeta_plus_joint", "max_constraint_survivors") if kk in v}
                          for k, v in pl["sets"].items()}
        sm = load("plateau/smoke.json")
        if sm:
            out["plateau_background"] = sm["bg_vs_cosh"]
    sh = load("shear/C160_shear_wdw.json")
    if sh:
        out["shear"] = {"physical": sh["physical"], "fits": sh["fits"], "domain_check": sh["domain_check"]}
    b1 = load("b1/C160_b1_wdw.json")
    if b1:
        out["b1"] = [{"X": r["X"], "case": r["case"], "predicted_D_re": r["predicted_D_re"],
                      "D_re": [x["D_re"] for x in r["rows"]], "D_sh": [x["D_sh"] for x in r["rows"]],
                      "F_in": [x["F_in"] for x in r["rows"]], "Im_psi": [x["Im_psi_fraction"] for x in r["rows"]],
                      "grid_check": r.get("grid_check_linear_natural_tmax")} for r in b1["runs"]]
    b0 = load("b1/C160_b1_wdw_zs0.json")
    if b0:
        out["b1_zs0"] = [{"X": r["X"], "case": r["case"], "zs": r["zs"], "predicted_D_re": r["predicted_D_re"],
                          "D_re": [x["D_re"] for x in r["rows"]], "D_sh": [x["D_sh"] for x in r["rows"]],
                          "F_in": [x["F_in"] for x in r["rows"]], "Im_psi": [x["Im_psi_fraction"] for x in r["rows"]]}
                         for r in b0["runs"]]
        out["b1_note"] = ("b1 (first set): cases A_Ai_real (valid) and the z_s=2.5 separable Bi cases (ill-conditioned, "
                          "superseded by b1_zs0; see D131 Sec. 4.2(d)); B4 at z_s=2.5 was stopped (contaminated).")
    out["status"] = {"item1_SM_Moyal": "done", "item2_plateau": "done (0/2560 fate flips vs D114)",
                     "item3_shear_quantum_volume": "done", "item4_B1_WDW": "done (one aspect mode)"}
    (RUN / "C160_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("written", RUN / "C160_summary.json")


if __name__ == "__main__":
    main()
