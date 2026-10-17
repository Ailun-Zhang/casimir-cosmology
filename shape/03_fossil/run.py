# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""shape/03_fossil: floor and natural state, sigma_E (0.104 / 0.094 next order / 0.059 free), side difference, nonlinear ratio.

--quick (~2 min): C58, C62, C63, C77, C82, C143 (all seconds); the archived free-field (C60, 2592) and xi = 0.30 (C76,
          4224) trajectories are unpacked and re-analysed (C60_analyze, shapelib/C76 analyze); 2 + 2 archived
          trajectories are re-integrated bit for bit.
--full  : C60_run_main (all tags incl. hi) and C76 run (main, hi, ind) from scratch, then the analyses.
The next-order floor 0.0934 and G(inf) are computed in shape/02 (C153_linear); next-order kappa in casimir/03 (C83).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _shape_paths as WP  # noqa: E402

LIB = "../shapelib/"


def main():
    st = Step(__file__, chain="shape")
    a = st.parse()
    w = str(a.workers)
    for s, lab in (("C58_turning_point_states.py", "C58 turning-point states"), ("C62_SM_pivot_calibration.py", "C62 pivot"),
                   ("C63_rolling_fossil.py", "C63 rolling fossil"), ("C77_prep_time_zero_mode.py", "C77 preparation time"),
                   ("C82_fossil_r_dependence.py", "C82 fossil vs xi")):
        st.py(s, label=lab)
    st.py("C143_observables.py", st.w("r2/observables/run01/C143_observables.json"), label="C143 observables")
    if a.mode == "full":
        st.py("C60_run_main.py", "--procs", w, "--tags", "main", "sens3", "amp3", "amp5", "amp8", "amp12", "ic_can", "ic_trf",
              "conv_tol", "conv_cut", "hi", label="C60 free-field ensembles")
        st.py(LIB + "C76_nonlinear_zero_mode.py", "run", "--tags", "main", "hi", "ind", "--procs", w, label="C76 xi=0.30 ensembles")
    else:
        WP.stage(["r5/nonlinear_SM/draws", "fu/C76/draws"])
        st.py(LIB + "redraw_check.py", "--driver", "C60", "--picks", "hi:5", "main:17", "--out", "checks/redraw_C60.json",
              label="redraw C60 (2 draws)")
        st.py(LIB + "redraw_check.py", "--driver", "C76", "--picks", "ind:3", "hi:40", "--out", "checks/redraw_C76.json",
              label="redraw C76 (2 draws)")
    WP.stage(["r5/nonlinear_SM/C60_validation_ewald.json"])   # archived: its C40 cross-check needs the research tree
    st.py("C60_analyze.py", label="C60 analysis")
    st.py(LIB + "C76_nonlinear_zero_mode.py", "analyze", label="C76 analysis")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
