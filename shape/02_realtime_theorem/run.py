# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""shape/02_realtime_theorem: real-moment theorem, G(t_x), closed-form floor, conditional theorem t_*.

--quick (~5-10 min): C137/C137b/C137c and C153_linear (linear theorem, G(t_x), floor closed form); the archived run01
          (17 920) and run02 (28 672) trajectories are unpacked and re-analysed by the production analyses
          (shapelib/C153_nl.py analyze, C153_run02.py analyze -> S2 grid, t_*(99%) = 0.237, ...); 6 + 4 archived
          trajectories are re-integrated bit for bit (shapelib/redraw_check.py); C153_summary.
--full  : all trajectories from scratch (C153_nl run: grid next phase never adiab + width, smoke; C153_run02 run):
          46 592 trajectories, 3438 + 221 + ~6800 s on 4 processes in the original run (~45 min on 15 workers),
          then the same analyses.  --fresh deletes the draws of a previous full run first (otherwise resumed).
Optional: env WOLFRAMSCRIPT -> the two .wl cross-checks are run and their output stored in the work tree.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _shape_paths as WP  # noqa: E402

import subprocess  # noqa: E402

LIB = "../shapelib/"
DRAWS = ["r3/shape_hardening/run01/draws", "r3/shape_hardening/run02/draws"]


def main():
    st = Step(__file__, chain="shape")
    a = st.parse()
    w = str(a.workers)
    st.py("C137_shape_gate.py", label="C137 shape gate")
    st.py("C137b_best_case.py", label="C137b best case")
    st.py("C137c_uncorrelated_optimum.py", label="C137c G(t_x)")
    st.py("C153_linear.py", label="C153 linear layer")
    if a.mode == "full":
        if a.fresh:
            st.clear(DRAWS)
        st.py(LIB + "C153_nl.py", "run", "--groups", "grid", "next", "phase", "never", "adiab", "--procs", w, label="C153 run01 main groups")
        st.py(LIB + "C153_nl.py", "run", "--groups", "width", "--procs", w, label="C153 run01 width")
        st.py(LIB + "C153_nl.py", "smoke", label="C153 smoke and mirror check")
        st.py(LIB + "C153_run02.py", "run", "--procs", w, label="C153 run02 (N = 2048)")
    else:
        WP.stage(DRAWS + ["r3/shape_hardening/run01/smoke.json"])
        st.py(LIB + "redraw_check.py", "--driver", "C153_nl", "--picks", "g_tau0.300_A:0", "g_tau0.300_B:0", "g_tau0.300_A:7",
              "nv1.2_out:3", "ph1.0_A:5", "nx_amp_s5:2", "--out", "checks/redraw_C153_nl.json", label="redraw C153_nl (6 draws)")
        st.py(LIB + "redraw_check.py", "--driver", "C153_run02", "--picks", "lo_t_tau0.250_A:11", "lo_t_tau0.250_B:11",
              "nx_t_tau0.300_A:4", "lo_amp_s3.0:9", "--out", "checks/redraw_C153_run02.json", label="redraw C153_run02 (4 draws)")
    st.py(LIB + "C153_nl.py", "analyze", label="C153 run01 analysis")
    st.py(LIB + "C153_run02.py", "analyze", label="C153 run02 analysis")
    WP.stage(["r3/shape_hardening/run01/emptiness/C153_emptiness.json", "r3/shape_hardening/run01/ensemble/C153_ensemble.json",
              "r3/shape_hardening/run01/wolfram_receipts.json"])        # from shape/01 and cmb/02 (never overwrites fresh files)
    st.py("C153_summary.py", label="C153 summary")
    ws = os.environ.get("WOLFRAMSCRIPT")
    if ws:
        for wl in ("C137_verify.wl", "C153_verify.wl"):
            r = subprocess.run([ws, "-file", os.path.join(HERE, wl)], capture_output=True, text=True)
            with open(st.w("checks/%s.out.txt" % wl), "w", encoding="utf-8") as f:
                f.write(r.stdout + r.stderr)
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
