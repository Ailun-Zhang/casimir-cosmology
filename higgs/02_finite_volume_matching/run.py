# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""02_finite_volume_matching: consistent one-loop finite-volume matching (D110/C138).

--quick (~1-2 min): C138_run.py checks (log cancellation, Coulomb-mode subtraction, RG consistency, T_B/2, c2) and
          static (gamma(lam) for MU/ML/MU_CW/ML_CW/MC/MC1 at v11 and SM couplings, lam_c of the static criterion).
--full  (~5 min original, 3 workers): C138_run.py all (adds profiles, the round-2 1D hazard main/var/mu sets with
          the R14 mapping, post -> C138_summary.json: MC+CW critical 172.349/172.768 [historical; superseded by the
          2D chain] and the matching error items used in PRL Table 1 'one-loop matching').
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh:
        st.clear(["r2/higgs_matching/run01"])
    st.py("C138_run.py", "checks")
    st.py("C138_run.py", "profiles")      # conditional-Gaussian gauge profiles (cache in <work>/scratch)
    st.py("C138_run.py", "static")
    if a.mode == "full":
        st.py("C138_run.py", "all")
    st.py("matching_potential.py")         # R18 DA-04: matching vs zero-point confinement (reads package data)
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
