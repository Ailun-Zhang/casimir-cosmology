# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_summary.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Collects the linear (this step), nonlinear (shapelib/C153_nl), ensemble (cmb/02 C153_ensemble) and emptiness (shape/01) layers into r3/shape_hardening/run01/C153_summary.json (the file the paper's number keys refer to).

C153_summary: collect linear / nonlinear / ensemble / emptiness layers into run01/C153_summary.json (D125)."""
import json, hashlib, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package: chain root (shape/)
import _shape_paths as WP  # noqa: E402
RUN = WP.wdir("r3/shape_hardening/run01")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


lin = json.loads((RUN / "linear" / "C153_linear.json").read_text(encoding="utf-8"))
nl = json.loads((RUN / "C153_nl_summary.json").read_text(encoding="utf-8"))
ens = json.loads((RUN / "ensemble" / "C153_ensemble.json").read_text(encoding="utf-8"))
emp = json.loads((RUN / "emptiness" / "C153_emptiness.json").read_text(encoding="utf-8"))
inputs = {
    "S2_profiles.json": WP.wfile("ai/cmb_ensemble/run01/results/S2_profiles.json"),
    "C56_SM_AAA.json": WP.wfile("r5/C56_preparation_gate/C56_SM_AAA.json"),
    "C55_window_gate.json": WP.wfile("r5/C55_window_gate/C55_window_gate.json"),
    "C55_window_gate_refined.json": WP.wfile("r5/C55_window_gate/C55_window_gate_refined.json"),
}
codes = sorted(p.name for p in HERE.glob("*.py")) + ["C153_verify.wl"]
g = {r["tau"]: r for r in nl["grid"]}
grid_tab = [{"tau": t, "two_branch": r["two_branch"]["p"], "two_branch_wilson95": r["two_branch"]["wilson95"],
             "A_future_from_minus_tau": r["A_future_of_minus_tau"]["p"], "B_future_from_plus_tau": r["B_future_of_plus_tau"]["p"],
             "single_branch_plus_tau": r["single_branch_plus_tau(B and seg[-tau,0] of A)"]["p"],
             "sigma_lin_A": r["sigma_lin_A"], "sigma_lin_B": r["sigma_lin_B"], "n": r["n"],
             "G_two_branch_linear": next((x["G_two_branch"] for x in lin["G_tables"]["E_xi030"]["rows"] if abs(x["tau"] - t) < 1e-9), 1.0 if t == 0 else None),
             "G_plus_single_linear": next((x["G_plus_single_branch"] for x in lin["G_tables"]["E_xi030"]["rows"] if abs(x["tau"] - t) < 1e-9), 1.0 if t == 0 else None)}
            for t, r in sorted(g.items())]
summary = {
    "task": "C153 / D125 (T16 + T18 shape + T9), agent shape-hardening",
    "codes_sha256": {c: sha(HERE / c) for c in codes},
    "inputs_sha256": {k: sha(v) for k, v in inputs.items() if Path(v).exists()},
    "provenance_file": "shape/PACKAGE_PATCHES.json",  # package (research tree: shape_hardening/PROVENANCE.txt)
    "A1_conditional_theorem": {"grid": grid_tab, "t_star_two_branch": nl.get("t_star_two_branch"),
                               "t_star_forward_only": nl.get("t_star_A_future_of_minus_tau"),
                               "monotone_lemma_check": lin["monotone_lemma_check"], "width_scan": nl.get("width"),
                               "mirror_check": json.loads((RUN / "smoke.json").read_text(encoding="utf-8")).get("mirror_check")},
    "A2_never_real": {"phase_family_linear": lin["phase_family"], "phase_family_NL": nl["phase"], "never_real_pair_NL": nl["never"],
                      "adiabatic_linear": lin["adiabatic_family"], "adiabatic_NL": nl["adiab"]},
    "B_next_order": {"floor_closed_form": lin["floor_closed_form"], "G_tables": {k: lin["G_tables"][k] for k in ("E_xi030", "E_next18.0", "T2_xi030", "T2_next_rescaled", "T2_v11_-8.34")},
                     "next_order_floor": lin["next_order_floor"], "NL": nl["next"], "NL_amp_threshold": nl["next_amp_threshold"],
                     "t_star_next_universal_curve": nl.get("next_t_star_from_universal_curve"),
                     "ensemble": {s: {v: {m: {k: ens["results"][s]["variants"][v][m][k] for k in ("Pj_max", "L0_Pj_max", "integrated_measure", "win_joint", "win_cond", "rep_point_window")}
                                          for m in ("mid", "lo", "hi")} for v in ("TE80_grid", "TE30_grid")} for s in ens["results"]}},
    "B_floor_unification": {"floor_conventions": lin["floor_conventions"], "H_eff_over_h": lin["H_eff_over_h"]},
    "B_shear_criterion": lin["shear_criterion"],
    "C_emptiness": {"quanta_budget": emp["quanta_budget"], "xi_where_budget_equals": emp["xi_where_budget_equals"],
                    "higgs_mass_term": emp["higgs_mass_term"], "reheating_excess": emp["reheating_excess"],
                    "preparation_window": emp["preparation_window"], "scaling_check": emp.get("scaling_check")},
    "wolfram_receipt": "r3/shape_hardening/run01/wolfram_receipts.json",
}
(RUN / "C153_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
print("written C153_summary.json;", len(grid_tab), "grid rows")
for r in grid_tab:
    print(" tau=%.3f two=%.4f %s A=%.4f B=%.4f single+=%.4f sigA=%.3f G=%s" % (r["tau"], r["two_branch"], [round(x, 3) for x in r["two_branch_wilson95"]],
          r["A_future_from_minus_tau"], r["B_future_from_plus_tau"], r["single_branch_plus_tau"], r["sigma_lin_A"], r["G_two_branch_linear"]))
