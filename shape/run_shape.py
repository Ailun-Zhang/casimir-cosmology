# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""Run every step of the shape chain in dependency order.

    python shape/run_shape.py --quick | --full | --reduced  [--workers N]

--reduced runs the reduced-size path check where a step defines one ([]) and --quick elsewhere.
Each step writes its own VERIFY.json; the exit code is non-zero if any step failed or any check failed.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['01_emptiness_gate', '02_realtime_theorem', '03_fossil', '04_nonlinear_survival', '05_moyal_wdw', '06_primordial_spectra']
REDUCED = []


def main():
    args = sys.argv[1:] or ["--quick"]
    mode = "full" if "--full" in args else ("reduced" if "--reduced" in args else "quick")
    rows, bad = [], 0
    for s in STEPS:
        a = list(args)
        if mode == "reduced" and s not in REDUCED:
            a = ["--quick" if x == "--reduced" else x for x in a]
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-B", os.path.join(HERE, s, "run.py")] + a)
        ok = None
        try:
            with open(os.path.join(HERE, s, "VERIFY.json"), encoding="utf-8") as f:
                v = json.load(f)
            m = "reduced" if (mode == "reduced" and s in REDUCED) else ("full" if mode == "full" else "quick")
            ok = v.get(m, {}).get("all_pass")
        except Exception:  # noqa: BLE001
            pass
        bad += int(rc != 0 or ok is False)
        rows.append("%-26s rc=%d all_pass=%s %6.0f s" % (s, rc, ok, time.time() - t0))
    print("\n".join(rows))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
