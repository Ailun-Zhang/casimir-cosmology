# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""01_sm_mapping: SM couplings at the bounce scale mu_b = Mbar_P/722 (D120/C148; compiled SMDR 1.3 cross-check run02).

--quick (~1-2 min): lam/y_t/g at mu_b for (172.60, 0.1180, 125.13) and at the run05 critical masses (sm_point.py);
          mbar_t(mbar_t) of the run05 critical pole masses (R3_mbar.py); C148_selftest.py; run02_post.py analyze
          (compiled SMDR fits [archived] vs the Python port); if third_party/smdr is built, one compiled refit
          (172.60, all scales x1) is recomputed and compared with the archived fit.
--full: additionally C148_bench.py and all 38 compiled refits (run02_fits.py; needs third_party/smdr built and, on
          Windows, DNEC_MINGW_BIN pointing to the MinGW bin directory), then run02_post.py analyze.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common import paths as P  # noqa: E402
from dnec_common.runner import Step  # noqa: E402

STAGE = ["r3/sm_mapping/run01/C148_bench.json", "r3/sm_mapping/run01/C148_summary.json",
         "r3/sm_mapping/run02/hazard_runs", "r2/higgs_final/run01/C144_hazard.json",
         "r3/holonomy_cross/run05/C149_run05_summary.json"]


def compiled_refit(st):
    exe = os.path.join(P.smdr_dir(), "c148_fit" + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(exe):
        print("[01] compiled SMDR not built (third_party/smdr/fetch_and_build.py) -> compiled refit skipped")
        return False
    out = st.w("r3/sm_mapping/run02/fits_quick/fit_xi1_Mt172.60.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = os.environ.get("DNEC_MINGW_BIN", "") + os.pathsep + env.get("PATH", "")
    args = [exe, "fit", "172.6000", "125.1300", "0.11800", "1", "1", "1", "1", "1", "172.6000", "1e-10", out]
    st.timed("compiled refit xi1 172.60", subprocess.run, args, cwd=P.smdr_dir(), env=env, capture_output=True)
    # turn the text output into JSON for the checker
    sys.path.insert(0, HERE)
    import run02_fits as F  # noqa: E402
    import json
    d = F.parse(out)
    with open(out[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    exp = F.parse(os.path.join(HERE, "expected", "r3", "sm_mapping", "run02", "fits", "fit_xi1_Mt172.60.txt"))
    with open(out[:-4] + "_archived.json", "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=1)
    return True


def main():
    st = Step(__file__)
    a = st.parse()
    st.stage(STAGE)
    st.py("sm_point.py")
    st.py("R3_mbar.py", env=dict(R3_RUN="run05", R3_SUMMARY="C149_run05_summary.json", R3_OUT="R3_mbar_run05.json"))
    st.py("C148_selftest.py")
    # absolute-stability bound m_stab and m0 with the production mapping (rep-B, report Sec. 10.5; ~6 s)
    out_abs = st.w("w/abs_stability/abs_stability.json")
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    st.py("abs_stability.py", "--out", out_abs, label="abs_stability m_stab m0")
    if a.mode == "full":
        st.py("C148_bench.py")
        if os.path.exists(os.path.join(P.smdr_dir(), "c148_fit" + (".exe" if os.name == "nt" else ""))):
            fits = st.w("r3/sm_mapping/run02/fits")
            if a.fresh and os.path.isdir(fits):
                shutil.rmtree(fits)
            st.py("run02_fits.py", env=dict(C148_FITW=str(a.workers)))
        else:
            print("[01] compiled SMDR missing: using the archived fits for run02_post")
            st.stage(["r3/sm_mapping/run02/fits"])
    else:
        st.stage(["r3/sm_mapping/run02/fits"])
    st.py("run02_post.py", "analyze")
    built = compiled_refit(st)
    if not a.no_verify:
        st.verify("checks.json", extra=dict(compiled_smdr_available=built))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
