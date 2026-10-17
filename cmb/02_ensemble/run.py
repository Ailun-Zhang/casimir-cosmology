# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""cmb/02_ensemble: C145/D117 E-plane fossil ensemble, read edge u_max, P(read|L0) window (TE l <= 80, ideal sky).

--quick (~1 min): on the archived run01 tables (KL table, profiles, noise): postcheck_ens (S3_postcheck: peak 0.344 at
          1.075, P >= 0.1 window [1.018, 1.183], measure 0.0422), make_fossil_table_v2, C153_ensemble (next order 0.377 at
          1.065), priors_ens.
--reduced: run_ens.py --smoke (l <= 12, x_max = 60; transfer rows l <= 12 shipped as inputs/C59_transfer_l12.npz):
          every covariance build, cross-KL, profile, Brent, noise and S3 analysis path from scratch, compared with the
          archived smoke run.
--full  : run_ens.py, run_ens.py --refine, then the quick chain on fresh tables.  Uses the l <= 80 transfer table
          shipped in data/C59_transfer.npz (85 MB; CAMB 2.0.4 via data/high_l_transfer.py; env DNEC_CMB_TRANSFER
          overrides), and preferably the snapshot cache
          (3.3 GB) for the S1 regression; it writes a 15.5 GB covariance cache (run01/cache, 163 files).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _cmb_paths as WP  # noqa: E402


def main():
    st = Step(__file__, chain="cmb")
    a = st.parse()
    w = str(a.workers)
    if a.mode == "reduced":
        os.environ["DNEC_CMB_TRANSFER"] = os.path.join(HERE, "inputs", "C59_transfer_l12.npz")
        st.py("run_ens.py", "--smoke", "--workers", w, label="C145 smoke run (l up to 12)")
    else:
        if a.mode == "full":
            if not os.environ.get("DNEC_CMB_TRANSFER") and os.path.exists(os.path.join(HERE, "data", "C59_transfer.npz")):
                os.environ["DNEC_CMB_TRANSFER"] = os.path.join(HERE, "data", "C59_transfer.npz")   # shipped (R18)
            st.py("run_ens.py", "--workers", w, label="C145 run01")
            st.py("run_ens.py", "--refine", "--workers", w, label="C145 run01 refine")
        else:
            WP.stage(["ai/cmb_ensemble/run01/results/" + f for f in ("kl_table.json", "S2_profiles.json", "S2_brent.json",
                      "S3_analysis.json", "S4_convergence.json", "S4_noise.json", "fossil_table.csv")])
        WP.stage(["fu/C79/C79_results.json"])
        st.py("postcheck_ens.py", label="C145 postcheck (pool-min profile, edges, windows)")
        st.py("make_fossil_table_v2.py", label="fossil_table_v2")
        st.py("C153_ensemble.py", label="C153 ensemble for other sigma_E")
        st.py("priors_ens.py", label="C145 size-prior experiments")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
