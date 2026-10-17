# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""casimir/03_shape_hessian: shape Laplacian / trace identity, no-go generalisations, gain, next-order Hessian.

--quick : C73 symbolic (SymPy) and gain checks, C83 next-order stiffnesses (inputs fu/C78, fu/C81 from casimir/02,
          recomputed there or staged from its expected/).
--full  : additionally the mpmath (dps 40-60) multiprocess tests C73 thermo, zeta, abelian zero point, toy reproduction.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402


def main():
    st = Step(__file__, chain="casimir")
    a = st.parse()
    st.stage(["fu/C78/C78_results.json", "fu/C81/C81_results.json"])   # from casimir/02 (never overwrites fresh files)
    st.py("c73_symbolic.py", label="C73 symbolic S1-S7")
    st.py("c73_gain.py", label="C73 gain and guaranteed lower bound")
    st.py("C83_next_order_shape_hessian.py", label="C83 next-order shape Hessian")
    if a.mode == "full":
        for s in ("c73_thermo.py", "c73_zeta.py", "c73_abelian_zeropoint.py", "c73_codex_toy_repro.py"):
            st.py(s, label="C73 " + s[4:-3])
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
