# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""06_hazard_criterion: hazard along the turn -> P1(m_t, alpha_s), critical m_t at P1 = 0.9/0.5/0.1, lam_c(mu_b),
mbar_t(mbar_t), and the convergence table (D122 Sec. 0.3 + D121 run02-run05).

--quick (~30 s): uses the run05 node widths in the work tree (recomputed by 04 --full in the same work tree, otherwise
          the archived shards from 04/expected), runs the production post/final (04_cross_terms/C149_direct5.py),
          the mbar conversion (01_sm_mapping/R3_mbar.py) and hazard_table.py (independent quadrature cross-check and the
          convergence table).
--full:  identical (the expensive node widths belong to 04_cross_terms --full; run run_all.py --full for the chain).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

STAGE = ["r3/holonomy_cross/run05/shards", "r3/holonomy_cross/run05/C149_coulomb_check.json",
         "r3/holonomy_cross/run04/C149_run04_summary.json", "r3/holonomy_cross/run04/C149_top_chiral.json",
         "r3/holonomy_cross/run02/C149_run02_summary.json",
         "r3/twod_direct_sm/run02/C150_summary.json", "r3/twod_direct_sm/run02/shards2d",
         "r3/twod_direct_sm/run01/C150_summary.json",
         "r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
         "r2/higgs_beyond_bo/run01/C140_summary.json", "r2/higgs_matching/run01/C138_summary.json"]


def main():
    st = Step(__file__)
    a = st.parse()
    st.stage(STAGE)       # never overwrites files already recomputed in this work tree
    st.py("../04_cross_terms/C149_direct5.py", "post", label="C149 run05 post (hazard, crit, P1, lam_c)")
    st.py("../04_cross_terms/C149_direct5.py", "final", label="C149 run05 final (summary)")
    st.py("../01_sm_mapping/R3_mbar.py", label="mbar(mbar) of the critical masses",
          env=dict(R3_RUN="run05", R3_SUMMARY="C149_run05_summary.json", R3_OUT="R3_mbar_run05.json"))
    st.py("hazard_table.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
