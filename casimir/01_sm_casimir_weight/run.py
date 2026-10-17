# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""casimir/01_sm_casimir_weight: SM signed Casimir weight, spin structures, holonomy sectors.

--quick = --full (about 10 s): C57 (eight spin structures, cube Hessians) and C68 (holonomy sectors, preparation time).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402


def main():
    st = Step(__file__, chain="casimir")
    a = st.parse()
    st.py("C57_spin_structure_Casimir.py", label="C57 spin structures, N_C, cube Hessians")
    st.py("C68_holonomy_and_preparation.py", label="C68 holonomy sectors and preparation time")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
