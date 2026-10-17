# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""cmb/01_fisher_kl: torus T+E covariance, shape Fisher and exact KL (ideal full sky, l <= 30/40).

--quick (~5 min): C59_validate units + parity; 'C75_produce.py validate' builds the free fossil and the cube at
          L0 = 1.07 d_LSS from scratch with the shipped CAMB 2.0.4 transfer table, kl_quick.py evaluates the exact KL
          (end-to-end check of transfer -> covariance -> KL against the archive); C75_summary on the archived C75 table.
--full  : C59_transfer (+ cutoff scan) if CAMB 2.0.4 is importable, C59_run conv/conv2/fine/main/full/exact (covariance tasks,
          1.3 GB cache), C59_analyze, C59_validate continuum, C75_produce all stages (668 MB cache), C75_analyze,
          C75_summary.  Original runs: several hours on 8 processes (R5, 2026-10-09/10; follow-up 2026-10-11).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _cmb_paths as WP  # noqa: E402

import importlib.util  # noqa: E402


def main():
    st = Step(__file__, chain="cmb")
    a = st.parse(extra=lambda ap: ap.add_argument("--c59-only", action="store_true",
                                                  help="--full without the C75 (v11 window) production and the continuum test"))
    w = str(a.workers)
    st.py("C59_validate.py", "units", label="C59 validate units")
    st.py("C59_validate.py", "parity", label="C59 validate parity")
    if a.mode == "full":
        if importlib.util.find_spec("camb") is not None:
            st.py("C59_transfer.py", label="C59 CAMB transfer table")
            st.py("C59_transfer_cutoff_scan.py", label="C59 transfer cutoff scan")
        else:
            st.stage(["r5/obs_fisher/C59_transfer.json"])   # metadata of the shipped CAMB 2.0.4 table (read by C59_analyze)
        for stage in ("conv", "conv2", "fine", "main", "full", "exact"):   # all production stages of 2026-10-09/10
            st.py("C59_run.py", stage, label="C59 run " + stage)
        if a.c59_only:   # continuum unit test not recomputed in this mode: archived result (read by C59_analyze)
            st.stage(["r5/obs_fisher/validate/C59_validate_continuum.json"])
        st.py("C59_analyze.py", label="C59 analyze")
        if not a.c59_only:
            st.py("C59_validate.py", "continuum", label="C59 validate continuum")
            st.stage(["r5/C70_profile_fine/C70_results.json", "r5/C71_zero_mode_sensitivity/C71_results.json"])
            st.py("C75_produce.py", "-j" + w, "validate", "main", "xconv", "l40", "dirs", label="C75 produce")
            st.py("C75_analyze.py", label="C75 analyze")
    else:
        st.py("C75_produce.py", "-j2", "validate", label="C75 validate builds (fossil + cube at 1.07)")
        st.py("kl_quick.py", label="exact KL of the fresh pair")
        st.stage(["fu/C75/C75_results.json"])
    if not (a.mode == "full" and a.c59_only):
        st.py("C75_summary.py", label="C75 summary")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
