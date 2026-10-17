# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""04_cross_terms: zero-mode/non-zero-mode cross terms and the node-direct 2D resonance widths (D121 run02-run05).

--quick   (~2 min)  recompute the exact top chiral table (C149_top), the RG-closure check (C149_coulomb), one node
                    job (u = 0, m_t = 172.6 and 173.0; full run05 physics) and the hazard/critical m_t from the
                    archived run05 node shards; compare with expected/.
--full    (~10 min with 11 workers; original 2 x 8 min with 3 workers)  recompute all 22 run05 node jobs
                    (11 nodes x {alpha_s = 0.1180: 10 masses; 0.1171/0.1189: 5+5 masses}) from scratch, incl. the T14b
                    base profiles (computed on the fly if 03 has not been run in the same work tree), then post/final.
Usage: python run.py [--quick|--full] [--workers N] [--work DIR] [--fresh]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

COMMON = ["r1/higgs_survival/run02_stage3/S3_prop.json",
          "r2/higgs_beyond_bo/run01/C140_profile_v11.npz", "r2/higgs_beyond_bo/run01/C140_summary.json",
          "r2/higgs_beyond_bo/run01/C140_prep.json", "r2/higgs_matching/run01/C138_summary.json",
          "r2/higgs_final/run01/C144_ueff.json", "r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/hazard",
          "r3/twod_direct_sm/run02/shards2d", "r3/twod_direct_sm/run02/C150_summary.json",
          "r3/holonomy_cross/run04/C149_run04_summary.json"]
RUN5 = "r3/holonomy_cross/run05"


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh:
        st.clear([RUN5, "r3/holonomy_cross/run04/C149_top_chiral.json"])
    st.stage(COMMON)
    # exact top chiral bubble table (input of run04/run05) and the RG-closure check (cheap, always recomputed)
    st.py("C149_top.py")
    st.py("C149_coulomb.py")
    if a.mode == "quick":
        test = st.w(RUN5 + "/shards/TEST_u0.0000.json")
        if os.path.exists(test):
            os.remove(test)
        st.py("C149_direct5.py", "test")                       # one node job, recomputed
        st.stage([RUN5 + "/shards"])                            # archived node shards (TEST is recomputed above)
        st.py("C149_direct5.py", "post")
        st.py("C149_direct5.py", "final")
    else:
        st.clear([RUN5 + "/shards", RUN5 + "/post"])      # full: node jobs are cached by shard file -> start clean
        st.py("C149_direct5.py", "test")
        st.py("C149_direct5.py", "run", "main")
        st.py("C149_direct5.py", "run", "als")
        st.py("C149_direct5.py", "post")
        st.py("C149_direct5.py", "final")
    # C161 step 6 (added for R18): field-by-field split of the run05 cross-term shift, from the archived node shards of
    # cross_decomp.py (66 node jobs, not rerun here) and the run05 summaries computed above
    st.stage(["w/claims_run05/run01/cross_decomp/shards"])
    st.py("cross_decomp.py", "post")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
