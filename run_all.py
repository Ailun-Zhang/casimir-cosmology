# PROVENANCE: new driver written 2026-10-15 by code-finalize (v12 code/data package).
"""One-command reproduction of the DNEC v12 code/data package.

    python run_all.py --quick [--workers N]      # every chain, quick mode, + figures   (~25 min on 16 cores)
    python run_all.py --full                     # list the long computations and their run times (nothing runs)
    python run_all.py --full --yes [--workers N] # run them (~7-8 h on 16 cores; see README.md)
    python run_all.py --quick --chains higgs cmb --no-figures

--quick recomputes something non-trivial in every result directory with the production code (single nodes, shards,
analysis layers, re-integrated trajectories, covariance builds) and compares it with the archived production values
in each directory's expected/ (tolerances in checks.json; result in VERIFY.json); it also runs the reduced-size
covariance -> KL -> profile path of cmb/02, rebuilds all figures from figures/data and checks the paper numbers
(numbers/numbers_v12.json -> numbers.tex in sync; every number against the archived outputs).
--full reruns the production computations (see the table printed by --full); cmb/01 runs the C59 production
(--c59-only; add --with-c75 for the C75 window production), cmb/02 runs its reduced path (the full l <= 80 ensemble
needs an 85 MB transfer table and writes a 15.5 GB cache; see cmb/README.md).

Outputs go to <package>/output/ (env DNEC_OUT); a summary is written to output/run_all_<mode>.json.  The exit code is
0 only if every step ran and every non-optional check passed in this run.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dnec_common import paths as P  # noqa: E402

CHAINS = ["casimir", "higgs", "shape", "cmb"]
FULL_PLAN = [  # (chain, what, wall time on the 16-core test machine, from the chain READMEs)
    ("casimir", "01-03 --full: Ewald sums, DMC of the zero-mode Hamiltonians, shape Hessian", "~11 min"),
    ("higgs", "01-12 --full: SMDR refits (if built), FV matching, 2D node jobs, run05 cross terms, DMC shards, real time, claims scan, universality", "~2.6 h"),
    ("shape", "01-06 --full: about 62 600 nonlinear trajectories (C153, C142, C76, C60, plateau), Moyal/WDW", "~3.2 h"),
    ("cmb", "01 --full --c59-only: C59 covariance production (128 tasks, 1.3 GB cache) + analysis", "~1 h"),
    ("cmb", "01 --full (with --with-c75): + C75 v11-window production (668 MB cache)", "+ several h (optional)"),
    ("cmb", "02 --reduced: covariance -> KL -> profile -> analysis path (l <= 12); 02 --full (l <= 80 table shipped in cmb/02_ensemble/data; ~70 min on 12 workers, 15.5 GB cache) only by hand: python cmb/02_ensemble/run.py --full", "~30 s"),
    ("cmb", "03 --full", "~20 s"),
    ("figures", "make_all_figures.py: all PRL/PRD figures from figures/data", "~2 min"),
]


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def call(cmd, rec, label):
    print("\n=== %s: %s" % (label, " ".join(os.path.relpath(c, HERE) if os.path.isabs(c) else c for c in cmd[1:])), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=HERE)
    rec.append(dict(label=label, rc=rc, seconds=round(time.time() - t0, 1)))
    print("=== %s: rc=%d, %.0f s" % (label, rc, time.time() - t0), flush=True)
    return rc


def collect(chains, mode_of, start):
    """Per result directory: did this run write VERIFY.json for the expected mode, and did it pass?"""
    rows = []
    for ch in chains:
        for v in sorted(glob.glob(os.path.join(HERE, ch, "*", "VERIFY.json"))):
            step = os.path.basename(os.path.dirname(v))
            out_rec = P.out("verify", ch, step, "VERIFY.json")   # R18: run records live under output/
            if os.environ.get("DNEC_RECORD") != "1" and os.path.exists(out_rec):
                v = out_rec
            m = mode_of(ch, step)
            try:
                with open(v, encoding="utf-8-sig") as f:
                    r = json.load(f).get(m, {})
            except Exception as exc:  # noqa: BLE001
                r = {"error": str(exc)}
            fresh = bool(r.get("utc")) and r["utc"] >= start
            rows.append(dict(step="%s/%s" % (ch, step), mode=m, fresh=fresh, all_pass=r.get("all_pass"),
                             n_pass=r.get("n_pass"), n_checks=r.get("n_checks"), n_skipped=r.get("n_skipped", 0),
                             utc=r.get("utc")))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--yes", action="store_true", help="with --full: really start the multi-hour production runs")
    ap.add_argument("--workers", type=int, default=max(1, min(15, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--chains", nargs="*", default=CHAINS, choices=CHAINS)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--with-c75", action="store_true", help="with --full: include the C75 production of cmb/01")
    a = ap.parse_args()
    mode = "full" if a.full else "quick"
    if a.full and not a.yes:
        print("run_all.py --full would run (wall times on an Intel Core Ultra X9 378H, 16 cores, Windows 11):")
        for ch, what, t in FULL_PLAN:
            print("  %-8s %-11s %s" % (ch, t, what))
        print("Total about 7-8 h (plus the optional C75 production). Start with:  python run_all.py --full --yes")
        return 0
    py, w = sys.executable, str(a.workers)
    start, t_all, rec = utc_now(), time.time(), []
    print("DNEC v12 package: mode=%s workers=%s output=%s" % (mode, w, P.out_root()), flush=True)
    for ch in a.chains:
        if ch == "higgs":
            call([py, "-B", P.pkg("higgs", "run_higgs.py"), "--" + mode, "--workers", w, "--keep-going"], rec, ch)
        elif ch == "cmb" and mode == "full":
            c01 = ["--full", "--workers", w] + ([] if a.with_c75 else ["--c59-only"])
            call([py, "-B", P.pkg("cmb", "01_fisher_kl", "run.py")] + c01, rec, "cmb/01")
            call([py, "-B", P.pkg("cmb", "02_ensemble", "run.py"), "--reduced", "--workers", w], rec, "cmb/02 reduced")
            call([py, "-B", P.pkg("cmb", "03_joint_state", "run.py"), "--full", "--workers", w], rec, "cmb/03")
        else:
            call([py, "-B", P.pkg(ch, "run_%s.py" % ch), "--" + mode, "--workers", w], rec, ch)
            if ch == "cmb":
                call([py, "-B", P.pkg("cmb", "02_ensemble", "run.py"), "--reduced", "--workers", w], rec, "cmb/02 reduced")
    if not a.no_figures:
        call([py, "-B", P.pkg("figures", "make_all_figures.py")], rec, "figures")
    call([py, "-B", P.pkg("validation", "run.py"), "--" + mode], rec, "validation")   # R18: blind checks + report checks
    call([py, "-B", P.pkg("numbers", "make_numbers_tex.py"), "--check"], rec, "numbers.tex sync")
    call([py, "-B", P.pkg("numbers", "check_numbers.py")], rec, "numbers check")

    def mode_of(ch, step):
        if ch == "cmb" and step == "02_ensemble" and mode == "full":
            return "reduced"
        return mode

    rows = collect(a.chains, mode_of, start)
    if "cmb" in a.chains and mode == "quick":   # the extra reduced run of cmb/02
        rows += [dict(r, step=r["step"] + " (reduced)") for r in collect(["cmb"], lambda c, s: "reduced", start)
                 if r["step"] == "cmb/02_ensemble"]
    ok_steps = all(r["fresh"] and r["all_pass"] for r in rows) and len(rows) > 0
    ok_calls = all(r["rc"] == 0 for r in rec)
    total = time.time() - t_all
    print("\n%-36s %-8s %-6s %s" % ("result directory", "mode", "fresh", "checks"))
    for r in rows:
        print("%-36s %-8s %-6s %s/%s%s%s" % (r["step"], r["mode"], r["fresh"], r["n_pass"], r["n_checks"],
                                            " (%s skipped)" % r["n_skipped"] if r["n_skipped"] else "",
                                            "" if r["all_pass"] else "  FAIL"))
    for r in rec:
        print("  %-16s rc=%d %8.0f s" % (r["label"], r["rc"], r["seconds"]))
    print("TOTAL %.0f s (%.1f min); all steps fresh and passing: %s; all drivers rc=0: %s"
          % (total, total / 60, ok_steps, ok_calls))
    summ = dict(mode=mode, start_utc=start, end_utc=utc_now(), total_seconds=round(total, 1), workers=a.workers,
                python=sys.version.split()[0], cpu_count=os.cpu_count(), calls=rec, steps=rows,
                all_pass=ok_steps and ok_calls)
    with open(P.out("run_all_%s.json" % mode, mkdir=True), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    return 0 if (ok_steps and ok_calls) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
