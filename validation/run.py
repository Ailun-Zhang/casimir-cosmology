# PROVENANCE: new driver written 2026-10-16 by rev-code (R18 revision, R1/MH-02) for the v12 code/data package.
"""validation/: the archived blind reimplementations (keystone, V1-V5), the report's own checks (rep-A) and their
comparison with the main calculation.

    python validation/run.py [--quick | --full] [--record]

--quick / --full (identical): (1) compare_blind.py: 23 items, each blind number found in the archived blind results
of this folder, each main number in the archived package outputs (or the report table named), difference within the
pass criterion; (2) reruns report_checks/rep-A_checks.py and rep-A_checks_part2.py (about 2 s) and compares every
number of their output with the archived output report_checks/expected/*.out.txt (relative 1e-9, absolute 1e-12).
The blind codes themselves are archived outputs: they are not rerun here (they need their own builds of mr, SMDR and
TSIL and hours of DMC/hazard runs; see README.md).  Record: output/validation/run_<mode>.json; exit 1 on a failure.
--record rewrites report_checks/expected/*.out.txt from a fresh run.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from dnec_common import paths as P  # noqa: E402

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
REP = ["rep-A_checks", "rep-A_checks_part2"]


def py(script):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-B", script], cwd=os.path.dirname(script), env=env, capture_output=True)
    return p.returncode, p.stdout.decode("utf-8"), p.stderr.decode("utf-8", "replace"), round(time.time() - t0, 1)


def same_numbers(a, b, rel=1e-9, ab=1e-12):
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return False, "line count %d vs %d" % (len(la), len(lb))
    worst = 0.0
    for i, (x, y) in enumerate(zip(la, lb)):
        nx, ny = NUM.findall(x), NUM.findall(y)
        if len(nx) != len(ny) or NUM.sub("#", x) != NUM.sub("#", y):
            return False, "line %d differs in text: %r vs %r" % (i + 1, x[:80], y[:80])
        for u, v in zip(nx, ny):
            d = abs(float(u) - float(v))
            if d > ab + rel * max(abs(float(u)), abs(float(v))):
                return False, "line %d: %s vs %s" % (i + 1, u, v)
            worst = max(worst, d)
    return True, "max |diff| %.3g" % worst


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mode = "full" if "--full" in sys.argv else "quick"
    res, ok = {"mode": mode, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": []}, True
    rc, out, err, dt = py(os.path.join(HERE, "compare_blind.py"))
    print(out.rstrip())
    res["steps"].append(dict(step="compare_blind", rc=rc, seconds=dt, passed=rc == 0))
    ok &= rc == 0
    for n in REP:
        rc, out, err, dt = py(os.path.join(HERE, "report_checks", n + ".py"))
        exp = os.path.join(HERE, "report_checks", "expected", n + ".out.txt")
        if "--record" in sys.argv:
            with open(exp, "w", encoding="utf-8", newline="\n") as f:
                f.write(out.replace("\r\n", "\n"))
        good, why = (False, "rc=%d %s" % (rc, err[-300:])) if rc else same_numbers(out.replace("\r\n", "\n"),
                                                                               open(exp, encoding="utf-8").read())
        print("%-4s %-22s %s (%.1f s)" % ("PASS" if good else "FAIL", n, why, dt))
        res["steps"].append(dict(step=n, rc=rc, seconds=dt, passed=good, detail=why))
        ok &= good
    res["all_pass"] = ok
    p = P.out("validation", "run_%s.json" % mode, mkdir=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("validation %s: %s -> %s" % (mode, "all passed" if ok else "FAILED", p))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
