"""Build every v12 figure (PRL and PRD) from ./data.

Each entry runs one figure script in a fresh interpreter (python -B <script> [args]); a failing script is reported
and the others still run. `python make_all_figures.py` builds all; `python make_all_figures.py <substring> ...` builds
the entries whose script name contains one of the substrings.

Groups: H = Higgs criterion (fig-higgs), S = trace/Casimir/shape/fossil/CMB (fig-shape; append your entries in the
S block only).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

FIGS = [
    # ---------------------------------------------------------------- H group (fig-higgs)
    ("fig_prl_fig1_threshold.py", ["prl"]),
    ("fig_prd_threshold.py", []),
    ("fig_prl_fig2_criterion.py", ["prl"]),
    ("fig_prd_criterion.py", []),
    ("fig_prl_fig3_xi.py", []),
    ("fig_prd_higgs_wavefunction.py", []),
    ("fig_prd_gamma_validation.py", []),
    ("fig_prd_hazard_realtime.py", []),
    ("fig_prd_cross_terms.py", []),
    ("fig_prd_universality.py", []),
    # ---------------------------------------------------------------- S group (fig-shape): append below
    ("fig_prd_turning_state.py", []),
    ("fig_prd_nonlinear_survival.py", []),
    ("fig_prd_trace.py", []),
    ("gen_casimir_slice_shape.py", []),          # regenerates data/shape/casimir_slice_AAA.json (about 20 s)
    ("fig_prd_casimir_shape.py", []),
    ("fig_prd_emptiness_gate.py", []),
    ("fig_prd_fossil_shape.py", []),
    ("fig_prd_primordial.py", []),
    ("fig_prd_fossil_sky.py", []),
    # technical report v1.1 (static box and abelian Higgs model; data in data/higgs/{static_box_c170,abelian_c171})
    ("fig_rep_v11_static_box.py", []),
    ("fig_rep_v11_kappa.py", []),
    ("fig_rep_v11_abelian_width.py", []),
    ("fig_rep_v11_abelian_threshold.py", []),
]


def main(argv: list[str]) -> int:
    sel = [f for f in FIGS if not argv or any(a in f[0] for a in argv)]
    bad = []
    for script, args in sel:
        if not (HERE / script).exists():
            print(f"[skip] {script} (not present)")
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, "-B", script, *args], cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok = r.returncode == 0
        print(f"[{'ok' if ok else 'FAIL'}] {script} {' '.join(args)} ({time.time() - t0:.1f} s)")
        if not ok:
            bad.append(script)
            print(r.stdout[-2000:], r.stderr[-3000:], sep="\n")
    print("failed:", bad if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
