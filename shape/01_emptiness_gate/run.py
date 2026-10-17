# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""shape/01_emptiness_gate: quanta budget, reheating coupling 10^94, latest preparation ~7 e-folds, photon branch.

--quick (~2 min): C55 --quick (2 slices x 25 modes), C55b time-reversal cross-check, C153_emptiness --check on the
          archived C55/C56 production files (budgets, R, T_max, 3-dof crossing + direct scaling check), C80 photon branch.
--full  : C55 production (7 slices x 97 modes) and refined (2 x 401), C56 NC100 and SM_AAA (~20-25 min each on 11
          processes in the original run), then the same chain on the fresh files.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _shape_paths as WP  # noqa: E402


def main():
    st = Step(__file__, chain="shape")
    a = st.parse()
    w = str(a.workers)
    if a.mode == "full":
        st.py("C55_contracting_window_gate.py", "--procs", w, label="C55 production (97 q)")
        st.py("C55_contracting_window_gate.py", "--procs", w, "--nq", "401", "--qmax", "30", "--rtol", "1e-12",
              "--slices", "1,85", "--label", "_refined", label="C55 refined (401 q)")
        st.py("C56_preparation_time_gate.py", "--procs", w, label="C56 NC100")
        st.py("C56_preparation_time_gate.py", "--procs", w, "--hLb", "0.0048300839914022006", "--NC", "25.515",
              "--label", "SM_AAA", "--Ts", "4,5,6,6.5,7,7.25,7.5,8", label="C56 SM_AAA")
    else:
        st.py("C55_contracting_window_gate.py", "--quick", "--procs", w, label="C55 quick")
        WP.stage(["r5/C55_window_gate/C55_window_gate.json", "r5/C55_window_gate/C55_window_gate_refined.json",
                  "r5/C56_preparation_gate/C56_SM_AAA.json", "r5/C56_preparation_gate/C56_NC100.json"])
    st.py("C55b_time_reversal_check.py", label="C55b time-reversal check")
    st.py("C153_emptiness.py", "--check", label="C153 emptiness (v12 inputs) + scaling check")
    st.py("C80_photon_branch_r.py", label="C80 photon-pair branch")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
