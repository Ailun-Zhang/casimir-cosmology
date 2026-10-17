# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""casimir/02_zero_mode_energy: energy of the SM gauge/Higgs zero modes on T^3 (xi = 0.2999) and next order.

--quick : C72 DMC timing test (2 short DMC runs, seeded), C72 analysis + summary on the archived production DMC energies,
          C78 (next-order holonomy curvature) and C81 (Higgs zero-mode mass).
--full  : the production DMC (all cases, ~10 min on 8 workers in the original run) and then the same chain.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402


def main():
    st = Step(__file__, chain="casimir")
    a = st.parse()
    if a.mode == "full":
        st.clear(["fu/C72/C72_dmc_results.json"])
        st.py("C72_dmc.py", label="C72 DMC production (all cases)")
    else:
        st.py("C72_dmc.py", "--test", label="C72 DMC timing test")
        st.stage(["fu/C72/C72_dmc_results.json"])      # archived production DMC energies
    st.py("C72_analysis.py", label="C72 analysis (D54 parts A-E)")
    st.py("C72_summary.py", label="C72 summary")
    st.py("C78_next_order_zero_mode.py", label="C78 next-order holonomy curvature")
    st.py("C81_higgs_zero_mode_mass.py", label="C81 Higgs zero-mode mass")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
