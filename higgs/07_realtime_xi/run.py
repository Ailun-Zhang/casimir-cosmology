# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""07_realtime_xi: real-time evolution through the turn and the non-minimal-coupling condition xi_c (D129 chain,
C161 run05 version = paper centre).

--quick (~1-2 min): node regression (xi_regress: the D129 run01 node is reproduced with the run02 cross terms; run05
          difference), xi_c extraction from the archived run05 real-time runs (rt_quick post -> C157_xi_quick.json),
          comparison with run02 and the universal xi_c(P1) reading (xi_compare).
--full  (~8-10 min with 8 workers; C161: nodes 178 s + calib 8 s + tables + 55 runs 264 s): node evaluation of the
          run05 model (nm_model nodes), calibration to the run05 2D widths, potential tables, 55 real-time runs
          (5 masses x 11 xi_H), post, comparison.
The run02-based D129 extras quoted with the xi condition (SM/alpha_s bands, 2D (r,|w|) real-time check, R15 items:
C157_run02_summary.json, C157_twod.json, C157_r15_items.json) are archived in expected/ for reference; their
production scripts (realtime_2d: twod_rt.py, run02_sm.py, r15_items.py; ~35 min) are not part of this package.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

X5 = "w/claims_run05/run01/xi_run05"
INPUTS = ["r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
          "r2/higgs_beyond_bo/run01/C140_summary.json", "r2/higgs_beyond_bo/run01/C140_prep.json",
          "r2/higgs_matching/run01/C138_summary.json", "r2/higgs_final/run01/C144_ueff.json",
          "r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/hazard",
          "r3/twod_direct_sm/run02/shards2d", "r3/twod_direct_sm/run02/C150_summary.json",
          "r3/holonomy_cross/run02/shards", "r3/holonomy_cross/run05/shards",
          "r3/holonomy_cross/run04/C149_top_chiral.json", "r3/holonomy_cross/run04/C149_run04_summary.json",
          "r3/realtime_2d/run01/C157_xi_quick.json", "r3/realtime_2d/run02/C157_run02_summary.json",
          "r3/realtime_2d/run01/nodes/node_mt172.60_u0.0000.npz"]


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh or a.mode == "full":
        st.clear([X5 + "/part2", X5 + "/nodes", X5 + "/tables", X5 + "/C157_xi_quick.json", X5 + "/C157_calib.json"])
    st.stage(INPUTS)
    st.py("xi_regress.py")
    if a.mode == "quick":
        st.stage([X5 + "/part2", X5 + "/C157_calib.json"])
    else:
        st.py("nm_model.py", "nodes", a.workers)
        st.py("nm_model.py", "calib")
        st.py("nm_model.py", "tables")
        st.py("rt_quick.py", "run", a.workers, "main")
    st.py("rt_quick.py", "post")
    st.py("xi_compare.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
