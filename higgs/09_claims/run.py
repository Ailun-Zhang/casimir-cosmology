# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""09_claims: claims layer at the run05 centre (D126 v2 + correction addendum; C161/D132): P1bar, Bayes factors,
exclusion thresholds, FCC-ee / HL-LHC discriminating power, (m_t, alpha_s) contour data, rebuilt Table 1.

--quick (~2-4 min): C154 core + correction on the run05 centre (inputs C154_inputs_v2_run05.json), independent direct
          Monte Carlo of P1bar (seeded, 4e6 samples), FCC-ee exclusion strength (fcc_exclusion.py: central P1 and
          B_ex at the FCC-ee threshold, R18), two-branch / B3 critical variants, Table 1 rebuild (uses the
          archived matching-scale scan summary), claims with the rebuilt composite (t1new), run02 regression
          (C154_inputs_v2_regress_run02.json must reproduce the archived round-3 claim_positioning/run02 numbers),
          run02-vs-run05 comparison.
--full: additionally the matching-scale scan in the run05 node pipeline (mu_scan.py run/post: 66 node jobs).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

NEW = "w/claims_run05/run01"
INPUTS = ["r3/holonomy_cross/run05/post/C149_run02_summary.json", "r3/holonomy_cross/run05/C149_run05_summary.json",
          "r3/holonomy_cross/run05/shards", "r3/holonomy_cross/run02/C149_run02_summary.json",
          "r3/sm_mapping/run01/hazard_runs/C144_hazard_central.json", "r3/realtime_2d/run01/C157_summary.json",
          "r3/inflation_stability/run01/C155_summary.json", "r3/two_branch/run01/C159_summary.json",
          "r3/higgs_robustness/run01/C152_mc_w0.json", "r3/claim_positioning/run02"]
SCAN_INPUTS = ["r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
               "r2/higgs_beyond_bo/run01/C140_summary.json", "r2/higgs_beyond_bo/run01/C140_prep.json",
               "r2/higgs_matching/run01/C138_summary.json", "r2/higgs_final/run01/C144_ueff.json",
               "r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/hazard",
               "r3/twod_direct_sm/run02/shards2d", "r3/twod_direct_sm/run02/C150_summary.json",
               "r3/holonomy_cross/run04/C149_top_chiral.json", "r3/holonomy_cross/run04/C149_run04_summary.json"]


def main():
    st = Step(__file__)
    a = st.parse()
    st.stage(INPUTS)
    if a.mode == "full":
        # the scan shards are produced only by full runs (never staged from the archive) and the job resumes per
        # shard file; --fresh forces a complete recomputation
        if a.fresh:
            st.clear([NEW + "/mu_scan"])
        st.stage(SCAN_INPUTS)
        st.py("mu_scan.py", "run", "c1", "c0.5", "c2", "c2pi", "ct0.5", "ct2")   # 6 variants x 11 nodes
        st.py("mu_scan.py", "post")
    else:
        st.stage([NEW + "/mu_scan/mu_scan_summary.json"])
    st.py("C154_core_v2.py", env=dict(C154_CFG=os.path.join(HERE, "C154_inputs_v2_run05.json")), label="C154 core run05")
    st.py("C154_correction_v2c.py", env=dict(C154_CFG=os.path.join(HERE, "C154_inputs_v2_run05.json")),
          label="C154 correction run05")
    st.py("fcc_exclusion.py", env=dict(C154_CFG=os.path.join(HERE, "C154_inputs_v2_run05.json")))   # R18 DA-01
    st.py("pbar_direct_mc.py")
    st.py("crit_variants.py")
    st.py("table1_rebuild.py")
    t1 = st.w(NEW + "/C154_inputs_v2_run05_t1new.json")            # written by table1_rebuild (work tree)
    st.py("C154_core_v2.py", env=dict(C154_CFG=t1), label="C154 core t1new")
    st.py("C154_correction_v2c.py", env=dict(C154_CFG=t1), label="C154 correction t1new")
    rg = os.path.join(HERE, "C154_inputs_v2_regress_run02.json")
    st.py("C154_core_v2.py", env=dict(C154_CFG=rg), label="C154 core regress run02")
    st.py("C154_correction_v2c.py", env=dict(C154_CFG=rg), label="C154 correction regress run02")
    st.py("compare_claims.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
