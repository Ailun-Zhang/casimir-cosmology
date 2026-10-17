"""Copy the archived production results used by the S-group (shape) figures into ./data/shape and record their
provenance in ./data/SOURCES_shape.json (original path relative to the research root, SHA-256, size, use).

Run only inside the research tree (ROOT = three levels above this folder); the released package ships the copied
files and SOURCES_shape.json, so the figure scripts never read outside ./data. Files are copied, never moved; the
originals are read-only. The two sets of 2048 nonlinear fossil draws (one JSON file per draw) are condensed into one
CSV each (beta_plus, geometry, survivor flags); SOURCES records the directory, the file count and a combined SHA-256
(SHA-256 of the sorted "name sha256" lines of the individual files).

Agent fig-shape, 2026-10-15.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = HERE / "data" / "shape"
SOURCES = HERE / "data" / "SOURCES_shape.json"

RES = "04_calculations/results"
R5 = f"{RES}/20261009_claude_R5"
FU = f"{RES}/20261011_claude_followup"
CR1 = f"{RES}/20261013_claude_v12"
CR2 = f"{RES}/20261013_claude_v12_r2"
CR3 = f"{RES}/20261014_claude_v12_r3"
AI = f"{RES}/20261013_argon_codex_intake"
R16 = "07_reviews/R16_v12_round3"

# dest name, source (relative to ROOT), use, figures
FILES = [
    ("C137c_results.json", f"{CR1}/state_selection/run03/C137c_results.json",
     "C137c (D109): optimal real-state variance over the floor, opt_over_floor, and natural/optimal width ratio "
     "s_opt_over_s_adiabatic at t_x for kappa_E=19.32 and kappa_T2=-9.07", ["prd_turning_state"]),
    ("C153_summary.json", f"{CR3}/shape_hardening/run01/C153_summary.json",
     "C153 (D125) run01: G(|t_x|) tables, closed-form transfer elements, floor conventions, next-order floor, "
     "conditional-theorem grid S2(|t_x|), never-real pairs, next-order NL survival, emptiness budget and preparation "
     "window, reheating excess, next-order ensemble summaries",
     ["prd_turning_state", "prd_nonlinear_survival", "prd_casimir_shape", "prd_emptiness_gate", "prd_fossil_shape",
      "prd_fossil_sky"]),
    ("C153_run02_summary.json", f"{CR3}/shape_hardening/run02/C153_run02_summary.json",
     "C153 run02 (N=2048): S2 at h|t_x|=0.15-0.35, t_*(99%)=0.237, amplitude scan s=2.5-3.5 and s99, next order",
     ["prd_turning_state", "prd_nonlinear_survival"]),
    ("C153_ensemble.json", f"{CR3}/shape_hardening/run01/ensemble/C153_ensemble.json",
     "C153 ensemble: P_joint(L0) and P_cond(L0) of the CMB fossil for sigma_E variants incl. next order 0.0940",
     ["prd_fossil_sky"]),
    ("C142_summary.json", f"{CR2}/shape_nonlinear_threshold/run01/C142_summary.json",
     "C142 (D114): nonlinear survival vs amplitude factor s (N=256), free-field and r=0.5 variants, real states set "
     "at t_x<0 (R1)", ["prd_nonlinear_survival"]),
    ("C160_summary.json", f"{CR3}/shape_tier2/run01/C160_summary.json",
     "C160 (D131): plateau-background variant of the survival runs (paired, fates unchanged)",
     ["prd_nonlinear_survival"]),
    ("D114_own.json", f"{R16}/unverified_items/results/D114_own.json",
     "R16 T19 independent implementation (N=2048): survival vs s, critical-amplification quantiles s99/s90/s50, R1 "
     "states, plateau reference", ["prd_nonlinear_survival"]),
    ("BLIND_RESULTS.json", f"{R16}/shape_blind/results/BLIND_RESULTS.json",
     "R16 V4 blind check: G(|t_x|) grid, nonlinear survival S(t_x) with N=2048 and t_*(90/99%)",
     ["prd_turning_state", "prd_nonlinear_survival"]),
    ("C63_results.json", f"{R5}/C63_rolling_fossil/C63_results.json",
     "C63: free-field SM (f0=0) and Maxwell-toy shape stiffness, delta, natural and optimal widths",
     ["prd_turning_state"]),
    ("C76_summary.json", f"{FU}/C76/C76_summary.json",
     "C76: f0=0.30 linear transfer (aspect, angle), natural widths, analytic linear sigma_E, sigma_T2",
     ["prd_turning_state", "prd_fossil_shape"]),
    ("C57_spin_structures.json", f"{R5}/C57_spin_structure/C57_spin_structures.json",
     "C57: signed Casimir weight N_C of the SM for the spin-structure classes", ["prd_casimir_shape"]),
    ("C55_window_gate.json", f"{R5}/C55_window_gate/C55_window_gate.json",
     "C55: log10 R of photons from a reheating coupling in the time-symmetric remote contraction (97 modes)",
     ["prd_emptiness_gate"]),
    ("C55_window_gate_refined.json", f"{R5}/C55_window_gate/C55_window_gate_refined.json",
     "C55 refined (401 modes)", ["prd_emptiness_gate"]),
    ("C56_SM_AAA.json", f"{R5}/C56_preparation_gate/C56_SM_AAA.json",
     "C56: excitation per non-conformal degree of freedom vs preparation time T (SM AAA)", ["prd_emptiness_gate"]),
    ("C56_NC100.json", f"{R5}/C56_preparation_gate/C56_NC100.json",
     "C56: same for the N_C=100 toy (drawn in v11 Fig. 3b; not drawn in v12, kept for reference)", []),
    ("C58_results.json", f"{R5}/C58_turning_point_states/C58_results.json",
     "C58: free-field natural outgoing variances (linear reference of the free-field fossil)", ["prd_fossil_shape"]),
    ("C64_results.json", f"{R5}/C64_plateau_ns_r/C64_results.json",
     "C64: exact n_s and r of the plateau vs N_*", ["prd_primordial"]),
    ("S3_postcheck.json", f"{AI}/cmb_ensemble/run01/results/S3_postcheck.json",
     "C145/D117 production ensemble (pool-min profile): P_joint, P_cond vs L0 for TE80/TE40/TE30/T80/noise variants, "
     "read edges", ["prd_fossil_sky"]),
    ("fossil_table_v2.csv", f"{CR3}/cleanup_t12/run01/fossil_table_v2.csv",
     "T12 cleanup: KL divergence D_prof of 55 fossils vs shortest cycle u (columns xi=sin3theta, s=L0/d_LSS)",
     ["prd_fossil_sky"]),
]

DRAWS = [
    ("fossil_draws_C76_ind.csv", f"{FU}/C76/draws", "ind_*.json",
     "C76: 2048 independent-seed nonlinear draws, f0=0.2999, natural turning-point state", ["prd_fossil_shape"]),
    ("fossil_draws_R5_hi.csv", f"{R5}/nonlinear_SM/draws", "hi_*.json",
     "C60: 2048 nonlinear draws, free fields (f0=0), natural turning-point state", ["prd_fossil_shape"]),
]


# files written in this folder rather than copied: name, origin, use, figures
DERIVED = [
    ("casimir_slice_AAA.json",
     "generated by figures/gen_casimir_slice_shape.py (Ewald sums copied unchanged from "
     "04_calculations/code/20261009_claude_R5/C57_spin_structure_Casimir.py)",
     "Delta rho_C/|rho_C| of the AAA Standard Model on the (beta_d, beta_a) shape slice", ["prd_casimir_shape"]),
    ("external_primordial_bands.json",
     "transcribed from published results (bib keys inside); values as in 06_paper_draft/figures/v11/"
     "fig6_primordial.py and PRD sections/s09_spectra.tex",
     "n_s 68% bands (SPT+ACT+Planck; P-ACT-LB2), BK18 r_T bound, LiteBIRD target", ["prd_primordial"]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def condense(src_dir: Path, pattern: str, out: Path) -> tuple[int, str]:
    paths = sorted(src_dir.glob(pattern), key=lambda p: int(p.stem.split("_")[-1]))
    lines = []
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "beta_plus_d", "beta_plus_d2", "beta_plus_12", "beta_plus_13", "beta_plus_23",
                    "ln_Lmax_over_Lmin", "max_angle_deviation_deg", "survivor", "gates_pass"])
        for p in paths:
            lines.append(f"{p.name} {sha256(p)}")
            r = json.loads(p.read_text(encoding="utf-8"))
            g = r["geometry_plus"]
            w.writerow([r["i"], *[repr(float(x)) for x in r["beta_plus"]], repr(float(g["ln_Lmax_over_Lmin"])),
                        repr(float(g["max_angle_deviation_deg"])), int(bool(r["survivor"])), int(bool(r["gates_pass"]))])
    combined = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return len(paths), combined


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DEST.mkdir(parents=True, exist_ok=True)
    rec = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "agent": "fig-shape",
           "generator": "figures/_collect_data_shape.py", "root": "research tree root (paths below are relative to it)",
           "files": {}}
    for name, src, use, figs in FILES:
        s = ROOT / src
        d = DEST / name
        if not d.exists() or sha256(d) != sha256(s):
            shutil.copy2(s, d)
        rec["files"][f"shape/{name}"] = {"source": src, "sha256": sha256(d), "bytes": d.stat().st_size,
                                         "use": use, "figures": figs}
        print(f"copied  {name:32s} {d.stat().st_size:>9d}")
    for name, src_dir, pattern, use, figs in DRAWS:
        d = DEST / name
        n, comb = condense(ROOT / src_dir, pattern, d)
        rec["files"][f"shape/{name}"] = {"source": f"{src_dir}/{pattern} ({n} files, condensed)",
                                         "sources_combined_sha256": comb, "sha256": sha256(d),
                                         "bytes": d.stat().st_size, "use": use, "figures": figs,
                                         "columns": "beta_plus (5 shape moduli: X_d, X_d2, X_12, X_13, X_23), "
                                                    "geometry_plus.ln_Lmax_over_Lmin, max_angle_deviation_deg, "
                                                    "survivor, gates_pass"}
        print(f"condensed {name:30s} {n} draws")
    for name, origin, use, figs in DERIVED:
        d = DEST / name
        if d.exists():
            rec["files"][f"shape/{name}"] = {"source": origin, "sha256": sha256(d), "bytes": d.stat().st_size,
                                             "use": use, "figures": figs}
            print(f"recorded {name}")
    old = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else {}
    for k, v in old.get("files", {}).items():      # keep entries added later by hand (e.g. derived slices)
        rec["files"].setdefault(k, v)
    SOURCES.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", SOURCES)


if __name__ == "__main__":
    main()
