# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""12_two_branch: two-branch (two-time) survival through the turn (D130/C159): GH decoherence functional (no
decoherence: interference eps 0.9948-0.99997), linear two-time probability P2_lin = P1^2/w0, Frechet bounds,
critical masses for P2 = P1^2 (run02 base).

--quick (~1-2 min): rebuild the D111 potential tables (zmlib/hr_potential.py build, ~12 s), static-spectrum
          cross-check (tb_static), summary (tb_post) from the archived TDSE shards.
--full: additionally the TDSE runs (tb_tdse.py full) before the summary.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

RUN = "r3/two_branch/run01"
TAB = "r2/higgs_realtime/run01/tables"


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh or a.mode == "full":
        st.clear([RUN + "/shards", RUN + "/tdse_results.json"])
    st.stage(["r3/holonomy_cross/run02/C149_run02_summary.json"])
    if not os.path.exists(st.w(TAB + "/U_table_MU_mt172.6.npz")):
        st.py("../zmlib/hr_potential.py", "build", st.w(TAB), label="D111 potential tables")
    st.py("tb_static.py")
    if a.mode == "quick":
        st.stage([RUN + "/shards", RUN + "/tdse_results.json"])
    else:
        st.py("tb_tdse.py", "full")
    st.py("tb_post.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
