# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""10_universality: universality of the finite-volume threshold lam_c (D128/C156): eta (transit-duration) scaling,
coupling derivatives, spin-structure (PPA/PAA) holonomy vacua and 't Hooft flux sectors.  Base: run02 node widths.

--quick (~2-4 min): U156_eta (from the run02 node shards), U156_twist, U156_hol2, U156_centre recomputed;
          U156_summary on the archived coupling-variation shards; U156_fix2 post (run02 repair) on archived shards.
--full: additionally U156_coup run/post (2D direct hazard with g2, gY, y_t x (1 +- 0.1); 39 node jobs) and
          U156_fix2 run (repair of the large-r cross-term functional).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

U1, U2 = "r3/universality/run01", "r3/universality/run02"
INPUTS = ["r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
          "r2/higgs_beyond_bo/run01/C140_summary.json", "r2/higgs_beyond_bo/run01/C140_prep.json",
          "r2/higgs_matching/run01/C138_summary.json", "r2/higgs_final/run01/C144_ueff.json",
          "r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/hazard",
          "r3/twod_direct_sm/run02/shards2d", "r3/twod_direct_sm/run02/C150_summary.json",
          "r3/holonomy_cross/run02/shards", "r3/holonomy_cross/run02/C149_run02_summary.json"]


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh or a.mode == "full":
        st.clear([U1 + "/shards", U2 + "/shards"])
    st.stage(INPUTS)
    st.py("U156_eta.py")
    st.py("U156_twist.py")
    st.py("U156_hol2.py")
    st.py("U156_centre.py")
    if a.mode == "quick":
        st.stage([U1 + "/shards", U1 + "/C156_coup.json", U1 + "/wolfram_receipts.json", U2 + "/shards",
                  U2 + "/repair_rb.json"])
    else:
        st.stage([U1 + "/wolfram_receipts.json"])
        st.py("U156_coup.py", "run", a.workers)
        st.py("U156_coup.py", "post")
        st.py("U156_fix2.py", "run")
    st.py("U156_summary.py")
    st.py("U156_fix2.py", "post")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
