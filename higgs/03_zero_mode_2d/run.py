# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""03_zero_mode_2d: 2D (r, |w|) zero-mode model (D112/C140) and the direct 2D hazard at SM running couplings
(D122/C150 = T14b; run02 with the NNLO mapping is the no-cross-term baseline of the PRL centre).

--quick (~3 min): C140 'lam0' (lam = 0 ground state 2D_ex 4.45086 vs full-dimensional DMC 4.4478(13));
          one production node job of T14b run02 (u = 0, 10 masses) recomputed and compared with the archived shard;
          C150 post on the (archived + recomputed) node shards for run02 (NNLO) and run01 (R14 mapping):
          critical m_t 172.5826/173.0775/173.5784 (run02) and 172.5185/173.0140/173.5155 (run01), convergence table.
--full (~10-15 min with 11 workers; original ~30 min with 3): T14b run02 from scratch: 1D seed chains (oned),
          2D main + als node jobs (incl. all gauge profiles), post.  run01 (R14 mapping, historical rows of the
          convergence table) is reproduced from its archived shards; its raw recomputation uses the same commands
          with C150_RUN=run01 C150_COUPLINGS=R14 (the 'c144' variant also needs the round-2 C144 profiles, 26 MB,
          not shipped).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402

ZM = os.path.join(os.path.dirname(HERE), "zmlib")
NNLO = dict(C150_COUPLINGS="module:" + os.path.join(ZM, "c150_nnlo_central.py"), C150_RUN="run02")
R14 = dict(C150_COUPLINGS="R14", C150_RUN="run01")
INPUTS = ["r1/higgs_survival/run02_stage3/S3_prop.json", "r2/higgs_beyond_bo/run01/C140_profile_v11.npz",
          "r2/higgs_beyond_bo/run01/C140_prep.json", "r2/higgs_matching/run01/C138_summary.json",
          "r2/higgs_final/run01/C144_ueff.json", "r2/higgs_final/run01/C144_hazard.json", "r2/higgs_final/run01/hazard"]
T2 = "r3/twod_direct_sm/run02"


def main():
    st = Step(__file__)
    a = st.parse()
    if a.fresh:
        st.clear([T2, "r2/higgs_beyond_bo/run01/C140_lam0.json"])
    st.stage(INPUTS)
    st.py("../zmlib/C140_beyond_bo.py", "lam0", label="C140 lam0")
    if a.mode == "quick":
        st.stage([T2 + "/shards1d"])                                   # 1D seeds of the 2D root search
        node = st.w(T2 + "/shards2d/2Dex_main_u0.0000.json")
        if os.path.exists(node):
            os.remove(node)
        st.py("c150_one_node.py", "main", "0.0", env=NNLO, label="C150 node u=0 (run02)")
        st.stage([T2 + "/shards2d", "r3/twod_direct_sm/run01/C150_summary.json"])
        st.py("../zmlib/C150_twod.py", "post", env=NNLO, label="C150 post run02")
        st.stage(["r3/twod_direct_sm/run01/shards1d", "r3/twod_direct_sm/run01/shards2d",
                  "r3/twod_direct_sm/run01/C150_time.json"])
        st.py("../zmlib/C150_twod.py", "post", env=R14, label="C150 post run01")
    else:
        # full: never reuse node shards (job1d/job2d return cached shard files, which may have been staged from the
        # archive by another step in this work tree); gauge profiles are pure functions of the couplings and are kept
        st.clear([T2 + "/shards2d", T2 + "/C150_summary.json"])
        # 1D chains (C150_twod.py oned) = root-search seeds + 1D rows; their 'c144' variants need the round-2 C144
        # gauge profiles (r2/higgs_final/run01/profiles, 37 files, 26 MB; produced by the round-2 C144_model.py,
        # not packaged) -> the archived 1D shards are staged; the 2D roots do not depend on the seed (verified).
        st.stage([T2 + "/shards1d"])
        st.py("../zmlib/C150_twod.py", "two", "main", a.workers, env=NNLO, label="C150 two main run02")
        st.py("../zmlib/C150_twod.py", "two", "als", a.workers, env=NNLO, label="C150 two als run02")
        st.stage(["r3/twod_direct_sm/run01/C150_summary.json"])
        st.py("../zmlib/C150_twod.py", "post", env=NNLO, label="C150 post run02")
        st.stage(["r3/twod_direct_sm/run01/shards1d", "r3/twod_direct_sm/run01/shards2d",
                  "r3/twod_direct_sm/run01/C150_time.json"])
        st.py("../zmlib/C150_twod.py", "post", env=R14, label="C150 post run01")
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
