# PROVENANCE: new helper written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""Bit-level re-computation of selected archived nonlinear shape trajectories (used by the --quick runs).

For a production driver (C153_nl, C153_run02, C142, C76, C60 = C60_run_main, C160_plateau) the listed draws are rebuilt
by the driver's own spec builder (same seeds), integrated by the driver's own ``work()`` (module global DRAWS pointed at
a scratch directory) and compared key by key with the archived per-draw JSON, read directly from the zip archive
shipped in ``<step>/expected/``.  Only 'runtime_s' is ignored.  Output (work tree): n_compared, n_mismatch,
n_fate_mismatch, max_abs_diff, max_rel_diff, per-draw fates.

    python redraw_check.py --driver C153_nl --picks g_tau0.300_A:0 g_tau0.300_B:0 --out checks/redraw_C153_nl.json
"""
import argparse
import importlib
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
for _p in (CHAIN, CHAIN / "04_nonlinear_survival", CHAIN / "03_fossil", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _shape_paths as WP  # noqa: E402

DRIVERS = {  # module, archive (mirror path of the zip), member pattern
    "C153_nl": ("C153_nl", "r3/shape_hardening/run01/draws.zip", "draws/{set}/{i:04d}.json"),
    "C153_run02": ("C153_run02", "r3/shape_hardening/run02/draws.zip", "draws/{set}/{i:04d}.json"),
    "C142": ("C142_shape_nonlinear_threshold", "r2/shape_nonlinear_threshold/run01/draws.zip", "draws/{set}/{i:04d}.json"),
    "C76": ("C76_nonlinear_zero_mode", "fu/C76/draws.zip", "draws/{tag}_{i:04d}.json"),
    "C60": ("C60_run_main", "r5/nonlinear_SM/draws.zip", "draws/{tag}_{i:04d}.json"),
    "C160_plateau": ("C160_plateau", "r3/shape_tier2/run01/plateau/draws.zip", "draws/{set}/{i:04d}.json"),
}
IGNORE = {"runtime_s"}


def all_specs(name, m):
    if name == "C153_nl":
        return [s for g in m.BUILD for s in m.BUILD[g]()]
    if name == "C153_run02":
        return m.all_specs()
    if name == "C142":
        return m.specs_core() + m.specs_extra()
    if name == "C76":
        return [s for tag in ("main", "hi", "ind") for s in m.specs(tag, m.R_ZM)]
    if name == "C60":
        tags = ("main", "sens3", "amp3", "amp5", "amp8", "amp12", "ic_can", "ic_trf", "conv_tol", "conv_cut", "hi")
        return [s for t in tags for s in m.make_specs(t)]
    if name == "C160_plateau":
        return m.specs(("amp", "r1"))
    raise KeyError(name)


def label(s):
    return "%s:%d" % (s.get("set", s.get("tag")), s["i"])


def _job(args):
    name, spec, scratch = args
    m = importlib.import_module(DRIVERS[name][0])
    m.DRAWS = Path(scratch)                    # the production work() writes DRAWS/<set>/<i>.json (or DRAWS/<tag>_<i>.json)
    m.work(spec)
    return label(spec)


def compare(c, e, path, acc, ta, tr):
    if isinstance(e, dict):
        if not isinstance(c, dict):
            acc["mism"].append(path + ": type")
            return
        for k, v in e.items():
            if k in IGNORE:
                continue
            if k not in c:
                acc["mism"].append(path + "/" + k + ": missing")
                continue
            compare(c[k], v, path + "/" + k, acc, ta, tr)
    elif isinstance(e, list):
        if not isinstance(c, list) or len(c) != len(e):
            acc["mism"].append(path + ": list length")
            return
        for k, (x, y) in enumerate(zip(c, e)):
            compare(x, y, "%s/%d" % (path, k), acc, ta, tr)
    elif isinstance(e, bool) or e is None or isinstance(e, str):
        if c != e:
            acc["mism"].append("%s: %r vs %r" % (path, c, e))
    else:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            acc["mism"].append("%s: %r vs %r" % (path, c, e))
            return
        if math.isnan(e) and math.isnan(c):
            return
        d = abs(c - e)
        acc["max_abs"] = max(acc["max_abs"], d)
        if e != 0:
            acc["max_rel"] = max(acc["max_rel"], d / abs(e))
        if d > ta + tr * abs(e):
            acc["mism"].append("%s: %r vs %r" % (path, c, e))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--driver", required=True, choices=sorted(DRIVERS))
    ap.add_argument("--picks", nargs="+", required=True, help="<set or tag>:<i>")
    ap.add_argument("--out", required=True, help="report path relative to the work tree")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DNEC_WORKERS", "4")))
    ap.add_argument("--tol-abs", type=float, default=1e-12)
    ap.add_argument("--tol-rel", type=float, default=1e-9)
    a = ap.parse_args()
    t0 = time.time()
    mod, zrel, member = DRIVERS[a.driver]
    m = importlib.import_module(mod)
    want = set(a.picks)
    specs = [s for s in all_specs(a.driver, m) if label(s) in want]
    missing = sorted(want - {label(s) for s in specs})
    if missing:
        raise SystemExit("unknown picks: %s" % missing)
    zpath = WP.find_archived(zrel)
    if zpath is None:
        raise SystemExit("archive not found in the package: " + zrel)
    rep = {"driver": a.driver, "archive": zrel, "picks": a.picks, "records": {}, "n_compared": 0, "n_mismatch": 0,
           "n_fate_mismatch": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0, "tolerances": [a.tol_abs, a.tol_rel]}
    with tempfile.TemporaryDirectory() as scratch:
        with Pool(min(a.workers, len(specs))) as pool:
            for _ in pool.imap_unordered(_job, [(a.driver, s, scratch) for s in specs]):
                pass
        with zipfile.ZipFile(zpath) as zf:
            for s in specs:
                rel = member.format(**s)
                new = json.loads((Path(scratch) / rel.split("/", 1)[1]).read_text(encoding="utf-8"))
                old = json.loads(zf.read(rel).decode("utf-8"))
                acc = {"mism": [], "max_abs": 0.0, "max_rel": 0.0}
                compare(new, old, "", acc, a.tol_abs, a.tol_rel)
                rep["n_compared"] += 1
                rep["n_mismatch"] += int(bool(acc["mism"]))
                rep["n_fate_mismatch"] += int(new.get("fate", new.get("status")) != old.get("fate", old.get("status")))
                rep["max_abs_diff"] = max(rep["max_abs_diff"], acc["max_abs"])
                rep["max_rel_diff"] = max(rep["max_rel_diff"], acc["max_rel"])
                rep["records"][label(s)] = {"fate": new.get("fate", new.get("status")),
                                            "fate_archived": old.get("fate", old.get("status")),
                                            "max_abs_diff": acc["max_abs"], "first_mismatches": acc["mism"][:5]}
    rep["runtime_s"] = time.time() - t0
    out = WP.wfile(a.out)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("%s: %d draws, %d mismatching, %d fate changes, max|diff| %.3g (rel %.3g), %.0f s" % (
        a.driver, rep["n_compared"], rep["n_mismatch"], rep["n_fate_mismatch"], rep["max_abs_diff"], rep["max_rel_diff"],
        rep["runtime_s"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
