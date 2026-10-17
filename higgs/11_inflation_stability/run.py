# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""11_inflation_stability: all-mode Higgs stability during inflation (D127/C155; MMR2021-type bound).

--quick (~1-3 min): MMR 2021 / 2022 reproduction (run_repro) and the xi_inf table / C155 summary (build_plane) from
          the archived scan shards.
--full: additionally the 95-job xi_inf(m_t, alpha_s) scan (run_main.py) before build_plane.
Note: the (m_t, xi_H) plane xi_plane_mt_xiH.csv uses inputs/xi_c_zero_mode.json = round-2 zero-mode values (archived
run; header "R3 values pending").  Do not plot it; the current zero-mode xi_c is in 07_realtime_xi.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

RUN = "r3/inflation_stability/run01"


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh or a.mode == "full":
        st.clear([RUN + "/shards", RUN + "/repro"])
    st.stage([RUN + "/inputs"])
    st.py("run_repro.py")
    if a.mode == "quick":
        st.stage([RUN + "/shards"])
    else:
        st.py("run_main.py", a.workers)
    st.py("build_plane.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
