# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""cmb/03_joint_state: C147 joint inflaton-shape state, size distribution p(N) ~ 1/N, readability <~ 1.5%.

--quick = --full (~1-2 min): joint_state.py (55 s on 2 workers originally) and the SymPy slope check b3_slope_symbolic.py
          (stdout compared line by line with the archive).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dnec_common.runner import Step  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import _cmb_paths as WP  # noqa: E402

import json  # noqa: E402


def main():
    st = Step(__file__, chain="cmb")
    a = st.parse()
    WP.stage(["fu/C79/C79_results.json"])                 # archived C79 table (shipped in cmb/02 expected/)
    st.py("joint_state.py", label="C147 joint state")
    st.py("b3_slope_symbolic.py", label="B3 slope symbolic")
    norm = lambda p: [x.rstrip() for x in open(p, encoding="utf-8-sig", errors="replace").read().replace("\r\n", "\n").split("\n") if x.strip()]
    new = norm(st.w("logs/03_joint_state/B3_slope_symbolic.out.txt"))
    old = norm(WP.find_archived("ai/joint_state/run01/b3_slope_symbolic.txt"))
    rep = {"n_lines_produced": len(new), "n_lines_archived": len(old),
           "n_diff_lines": sum(1 for x, y in zip(new, old) if x != y) + abs(len(new) - len(old))}
    os.makedirs(st.w("checks"), exist_ok=True)
    with open(st.w("checks/b3_textcheck.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    if not a.no_verify:
        st.verify("checks.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
