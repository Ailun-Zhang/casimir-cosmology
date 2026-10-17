# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""shape/06_primordial_spectra: n_s and r of the plateau after the turn.

--quick = --full: C64 (mode equations for 12 values of N_* and 4 initial-state tests).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402


def main():
    st = Step(__file__, chain="shape")
    a = st.parse()
    st.py("C64_plateau_ns_r.py", label="C64 n_s and r")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
