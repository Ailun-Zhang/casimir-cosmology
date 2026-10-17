# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""05_fulldim_validation: walled full-dimensional DMC, same-wall 2D, CAP/WKB/3D width estimates (D123/C151/T14a).

--quick (~2-3 min): T14a_profile (barrier and walls); ONE production DMC shard (m010_rw5.12, tau 0.04, k 0;
          SHA-256 seed) recomputed and compared bit-for-bit; tau -> 0 extrapolation (combine) and the summary
          (T14a_post) on the archived 66 shards + archived 2D/CAP/WKB/3D tables -> Delta Q, Delta m_c.
--full  (original ~30 min DMC with 3 workers + ~10 min other stages; ~15 min with 12 workers): all 66 DMC shards,
          combine, same-wall 2D FD, CAP, WKB, 3D model, post.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

RUN = "r3/fulldim_dmc/run01"
INPUTS = ["r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
          "r2/higgs_beyond_bo/run01/C140_prep.json", "r2/higgs_beyond_bo/run01/C140_summary.json",
          "r2/higgs_beyond_bo/run01/C140_lam0.json", "r2/higgs_matching/run01/C138_summary.json"]


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh or a.mode == "full":
        st.clear([RUN])
    st.stage(INPUTS)
    st.py("T14a_profile.py")
    if a.mode == "quick":
        one = st.w(RUN + "/shards/m010_rw5.12_tau0.040_k0.json")
        if os.path.exists(one):
            os.remove(one)
        st.py("dmc_one_shard.py", label="DMC shard m010_rw5.12 tau0.04 k0")
        st.stage([RUN + "/shards", RUN + "/T14a_2d_wall.json", RUN + "/T14a_cap_dlng.json", RUN + "/T14a_wkb.json",
                  RUN + "/T14a_3d.json"])
        st.py("T14a_dmc.py", "combine")
        st.py("T14a_post.py")
    else:
        st.py("T14a_dmc.py", "run", "all", a.workers)
        st.py("T14a_2d.py")
        st.py("T14a_gamma.py", "cap")
        st.py("T14a_gamma.py", "wkb")
        st.py("T14a_3d.py", "run")
        st.py("T14a_post.py")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
