# PROVENANCE: new driver written 2026-10-15 by code-higgs (v12 code/data package).
"""Run the Higgs zero-mode chain (all result directories, dependency order) and print the verification summary.

    python higgs/run_higgs.py --quick [--workers N]          # ~15-20 min on 16 cores
    python higgs/run_higgs.py --full  [--workers N]          # ~2-3 h on 16 cores (see higgs/README.md)
    python higgs/run_higgs.py --quick --only 04_cross_terms 06_hazard_criterion

All steps share one work tree (<out>/higgs/<mode>), so a --full run feeds freshly computed upstream results (e.g.
the T14b node shards and gauge profiles of 03) into the downstream steps; a step run alone stages archived inputs.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["01_sm_mapping", "02_finite_volume_matching", "03_zero_mode_2d", "04_cross_terms", "05_fulldim_validation",
         "06_hazard_criterion", "07_realtime_xi", "08_robustness_w0", "09_claims", "10_universality",
         "11_inflation_stability", "12_two_branch"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(15, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()
    mode = "--full" if a.full else "--quick"
    steps = [s for s in STEPS if not a.only or s in a.only]
    rec = []
    for s in steps:
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-B", os.path.join(HERE, s, "run.py"), mode, "--workers", str(a.workers)])
        rec.append((s, rc, time.time() - t0))
        print("== %s rc=%d %.0f s" % (s, rc, time.time() - t0), flush=True)
        if rc != 0 and not a.keep_going:
            break
    print("\nstep timings:")
    for s, rc, dt in rec:
        print("  %-28s rc=%d %7.0f s" % (s, rc, dt))
    subprocess.call([sys.executable, "-B", os.path.join(os.path.dirname(HERE), "tools", "verify_summary.py")])
    sys.exit(max([r for _, r, _ in rec] or [0]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
