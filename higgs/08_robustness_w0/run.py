# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""08_robustness_w0: background sensitivity of the critical mass, the inflation-scale item (vi), the B3-rule w0 and
the background bookkeeping (D124/C152).  Base: the run02 node-direct pipeline (C149 run02 shards), as in the paper.

--quick (~3 min): C152_core 'test' (re-solves two node columns at l = 0 with fresh and baseline profiles and compares
          with the run02 base table), then post / w0 / accounting on the archived C152 shards.
--full  (original 1403 s with 3 workers; ~7 min with 12): C152_core all -0.3147 -0.8803 0.2705 -0.014531
          (47 scale-shifted node jobs: l-grid +0.28/-5.15/-5.60 and direct shifts delta_mu for the r-family, the
          h e^{-+1} family and the content variant), then post, w0, accounting.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

RUN = "r3/higgs_robustness/run01"
INPUTS = ["r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
          "r2/higgs_beyond_bo/run01/C140_summary.json", "r2/higgs_beyond_bo/run01/C140_prep.json",
          "r2/higgs_matching/run01/C138_summary.json", "r2/higgs_final/run01/C144_ueff.json",
          "r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/hazard",
          "r3/twod_direct_sm/run02/shards2d", "r3/twod_direct_sm/run02/C150_summary.json",
          "r3/holonomy_cross/run02/shards", "r3/holonomy_cross/run02/C149_run02_summary.json",
          "r3/realtime_2d/run01/C157_xi_quick.json"]
DELTAS = ["-0.3147", "-0.8803", "0.2705", "-0.014531"]


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh or a.mode == "full":
        st.clear([RUN + "/shards", RUN + "/C152_post.json", RUN + "/C152_w0_B3.json", RUN + "/C152_core_test.json"])
    st.stage(INPUTS)
    st.py("C152_core.py", "test")
    if a.mode == "quick":
        st.stage([RUN + "/shards"])
    else:
        st.py("C152_core.py", "all", *DELTAS)
    st.py("C152_post.py")
    st.py("C152_w0.py")
    st.py("C152_account.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
