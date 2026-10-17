# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""shape/04_nonlinear_survival: survival thresholds s50/s90/s99, R1 states, plateau background, next order.

--quick (~5 min): unpack the archived trajectories (D114 6656, plateau 2560, C76 4224, C153 run01/run02), re-run the
          production analyses (shapelib/C142 analyze, C160_plateau analyze, shapelib/C153_nl analyze, C153_run02 analyze)
          and re-integrate 2 + 2 + 2 archived trajectories bit for bit.
--full  : C142 run (6656, ~40 min on 5 processes originally), C160_plateau run + smoke (2560, ~1 h on 2-4 processes),
          C153_run02 / C153_nl next group (resumed if shape/02 --full ran in the same work tree), then the analyses.
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
    if a.mode == "full":
        WP.stage(["fu/C76/draws"])                       # C142 regression input (recomputed by shape/03 --full)
        st.py(LIB + "C142_shape_nonlinear_threshold.py", "run", "--group", "all", "--procs", w, label="C142 run (D114)")
        st.py(LIB + "C142_shape_nonlinear_threshold.py", "smoke", label="C142 smoke")
        st.py("C160_plateau.py", "run", "--procs", w, label="C160 plateau run")
        st.py("C160_plateau.py", "smoke", label="C160 plateau smoke")
        st.py(LIB + "C153_nl.py", "run", "--groups", "next", "--procs", w, label="C153 next-order group")
        st.py(LIB + "C153_run02.py", "run", "--procs", w, label="C153 run02")
    else:
        WP.stage(["r2/shape_nonlinear_threshold/run01/draws", "r3/shape_tier2/run01/plateau/draws", "fu/C76/draws",
                  "r3/shape_hardening/run01/draws", "r3/shape_hardening/run02/draws"])
        st.py(LIB + "redraw_check.py", "--driver", "C142", "--picks", "amp_s3:21", "r1_tx-0.5:8", "--out", "checks/redraw_C142.json",
              label="redraw C142 (2 draws)")
        st.py(LIB + "redraw_check.py", "--driver", "C160_plateau", "--picks", "pl_amp_s5:12", "pl_r1_tx-1.0:3",
              "--out", "checks/redraw_C160_plateau.json", label="redraw C160 plateau (2 draws)")
        st.py(LIB + "redraw_check.py", "--driver", "C153_run02", "--picks", "lo_amp_s2.5:33", "nx_amp_s3.5:2",
              "--out", "checks/redraw_C153_run02_amp.json", label="redraw C153_run02 amplitude sets (2 draws)")
    WP.stage(["r3/shape_hardening/run01/linear/C153_linear.json"])     # from shape/02 (never overwrites fresh files)
    st.py(LIB + "C142_shape_nonlinear_threshold.py", "analyze", label="C142 analysis")
    st.py("C160_plateau.py", "analyze", label="C160 plateau analysis")
    st.py(LIB + "C153_nl.py", "analyze", label="C153 run01 analysis (next-order amplitude threshold)")
    st.py(LIB + "C153_run02.py", "analyze", label="C153 run02 analysis (s99)")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
