# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""shape/05_moyal_wdw: SM Moyal correction, shear criterion with quantum volume, B1 WDW test, B3 width lemma.

--quick (~8 min): C141 (SymPy, stdout compared with the archive), C160_moyal (111 s), C160_shear_wdw (235 s), four B1
          cases via b1_cases.py (incl. the physical X = 2282.49 Ai-type case), C160_summary (archived 1d/b1/plateau parts).
--full  : additionally C160_moyal 1d (exact one-mode quantum check, ~18 min), the complete z_s = 0 B1 set (~18 min) and
          all 15 valid z_s = 2.5 B1 cases (b1_cases.py; the production z_s = 2.5 run was stopped in the contaminated case B4).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _shape_paths as WP  # noqa: E402

B1_QUICK = ['A_Ai_real@100', 'A_sep_control@100', 'B1_Ai_chi0_plus_Bi_h1_eps1@100', 'A_Ai_real@2282.49']
B1_FULL = ['A_Ai_real@100', 'A_Ai_real@300', 'A_Ai_real@1000', 'A_Ai_real@2282.49', 'A_sep_control@100', 'B1_Ai_chi0_plus_Bi_h1_eps1@100', 'A_sep_control@300', 'B1_Ai_chi0_plus_Bi_h1_eps1@300', 'A_sep_control@1000', 'B1_Ai_chi0_plus_Bi_h1_eps1@1000', 'A_Ai_real_zs5@300', 'A_Ai_real_nb512_bmax16@300', 'A_Ai_real_nb512_bmax12@300', 'A_Ai_real_quadratic@300', 'B2_eps0.3@300', 'B3_Ai_plus_0.7Bi_same_chi0@300']


def main():
    st = Step(__file__, chain="shape")
    a = st.parse()
    st.py("C141_minvar_lemma.py", label="C141 minvar lemma")
    st.py("../shapelib/textcheck.py", "logs/05_moyal_wdw/C141_minvar_lemma.out.txt",
          "r2/time_symmetry_origin/run01/C141_minvar_lemma_output.txt", "checks/C141_textcheck.json", label="C141 text check")
    st.py("C160_moyal.py", label="C160 Moyal (SM)")
    st.py("C160_shear_wdw.py", label="C160 shear criterion with quantum volume")
    if a.mode == "full":
        st.py("C160_moyal.py", "1d", label="C160 Moyal exact 1D check")
        st.py("C160_b1_wdw.py", "zs0", label="C160 B1 WDW z_s = 0 set")
        st.py("b1_cases.py", *B1_FULL, "--out", "r3/shape_tier2/run01/b1/b1_cases.json", label="B1 WDW z_s = 2.5 cases")
    else:
        st.py("b1_cases.py", *B1_QUICK, "--out", "r3/shape_tier2/run01/b1/b1_cases.json", label="B1 WDW selected cases")
    WP.stage(["r3/shape_tier2/run01/moyal/C160_moyal_1d.json", "r3/shape_tier2/run01/b1/C160_b1_wdw.json",
              "r3/shape_tier2/run01/b1/C160_b1_wdw_zs0.json", "r3/shape_tier2/run01/plateau/C160_plateau_summary.json",
              "r3/shape_tier2/run01/plateau/smoke.json"])   # archived parts (never overwrites fresh files)
    st.py("C160_summary.py", label="C160 summary")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
