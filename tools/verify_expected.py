# PROVENANCE: new tool written 2026-10-15 by code-higgs (v12 code/data package).
"""Compare freshly computed outputs with the archived production values (``expected/``) under stated tolerances.

A *checks file* (JSON) lists the comparisons of one result directory::

    {"step": "04_cross_terms",
     "checks": [
       {"id": "mc_P0.5", "what": "critical m_t^pole, P1 = 0.5, alpha_s = 0.1180", "paper": "PRL Table 1",
        "computed": {"file": "WORK:r3/holonomy_cross/run05/C149_run05_summary.json", "key": "crit/0.1180/P1=0.5"},
        "expected": {"file": "expected/r3/holonomy_cross/run05/C149_run05_summary.json", "key": "crit/0.1180/P1=0.5"},
        "tol_abs": 0.001},
       {"id": "shards", "mode": "tree", "computed": {...}, "expected": {...}, "tol_abs": 1e-9, "tol_rel": 1e-6,
        "ignore": ["time_s", "prof"]}
     ]}

File prefixes: ``WORK:`` = the work directory given with --work (or ``work=`` in ``run_checks``), ``PKG:`` = the
package root, anything else is relative to the directory of the checks file.  ``expected`` may also be
``{"value": 172.5024}``.  ``mode`` is ``scalar`` (default) or ``tree`` (all numeric leaves below the key are
compared; strings/bools must match unless ignored).  A check passes if |c - e| <= tol_abs or
|c - e| <= tol_rel * |e|.  Missing files/keys count as failures (with the reason recorded).
Non-finite values: NaN matches NaN (and null, the JSON-safe form written by ``dnec_common.io``), +-inf matches the
same infinity.  A tree check passes when no leaf differs and at least one leaf (number, string, bool or null) was
compared, so trees made only of strings are valid.  (Fixes 2026-10-15 by code-finalize.)

Usage::  python tools/verify_expected.py CHECKS.json --work DIR [--out VERIFY.json] [--mode quick|full]
"""
import argparse
import datetime
import json
import math
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
from dnec_common.io import get_key, load_json, write_json  # noqa: E402


def _resolve(spec_file, base_dir, work):
    if spec_file.startswith("WORK:"):
        return os.path.normpath(os.path.join(work, spec_file[5:]))
    if spec_file.startswith("PKG:"):
        return os.path.normpath(os.path.join(PKG_ROOT, spec_file[4:]))
    return os.path.normpath(os.path.join(base_dir, spec_file))


def _rel(path):
    """Package-relative path for the record (basename if the file is on another drive, e.g. DNEC_OUT elsewhere)."""
    try:
        return os.path.relpath(path, PKG_ROOT).replace(os.sep, "/")
    except ValueError:
        return os.path.basename(path)


def _fetch(spec, base_dir, work):
    if "value" in spec:
        return spec["value"], "literal"
    path = _resolve(spec["file"], base_dir, work)
    if path.lower().endswith(".csv"):
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        obj = rows
        if "row" in spec:                       # {"row": {"col": "value"}, "col": "name"}
            sel = [r for r in rows if all(r.get(k) == v for k, v in spec["row"].items())]
            if len(sel) != 1:
                raise KeyError("csv row selector matched %d rows" % len(sel))
            val = sel[0][spec["col"]]
            try:
                val = float(val)
            except ValueError:
                pass
            return val, _rel(path)
    else:
        obj = load_json(path)
    return get_key(obj, spec.get("key", "")), _rel(path)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _nonfinite(x):
    return isinstance(x, float) and not math.isfinite(x)


def _same_nonfinite(c, e):
    """NaN == NaN (also NaN vs null, the JSON-safe form of non-finite values); inf == inf of the same sign."""
    if (c is None or (_nonfinite(c) and math.isnan(c))) and (e is None or (_nonfinite(e) and math.isnan(e))):
        return True
    return _nonfinite(c) and _nonfinite(e) and c == e


def _close(c, e, ta, tr):
    if c is None or e is None or _nonfinite(c) or _nonfinite(e):
        return _same_nonfinite(c, e), (0.0 if _same_nonfinite(c, e) else None)
    d = abs(float(c) - float(e))
    ok = (ta is not None and d <= ta) or (tr is not None and d <= tr * abs(float(e)))
    if ta is None and tr is None:
        ok = d == 0.0
    return ok, d


def _tree(c, e, ta, tr, ignore, path, acc):
    if isinstance(e, dict):
        if not isinstance(c, dict):
            acc["mismatch"].append((path, "type"))
            return
        for k, v in e.items():
            if k in ignore:
                continue
            if k not in c:
                acc["mismatch"].append((path + "/" + k, "missing"))
                continue
            _tree(c[k], v, ta, tr, ignore, path + "/" + k, acc)
    elif isinstance(e, list):
        if not isinstance(c, list) or len(c) != len(e):
            acc["mismatch"].append((path, "list length"))
            return
        for i, (ci, ei) in enumerate(zip(c, e)):
            _tree(ci, ei, ta, tr, ignore, "%s/%d" % (path, i), acc)
    elif _num(e) or e is None:
        acc["n_leaves"] += 1
        if e is None or c is None or _nonfinite(e) or (_num(c) and _nonfinite(c)):
            if not _same_nonfinite(c, e):
                acc["mismatch"].append((path, "%r vs %r (non-finite/null)" % (c, e)))
            return
        if not _num(c):
            acc["mismatch"].append((path, "type"))
            return
        acc["n"] += 1
        ok, d = _close(c, e, ta, tr)
        acc["max_abs"] = max(acc["max_abs"], d)
        if e != 0:
            acc["max_rel"] = max(acc["max_rel"], d / abs(e))
        if not ok:
            acc["mismatch"].append((path, "%r vs %r" % (c, e)))
    else:                                   # strings, bools: exact match
        acc["n_leaves"] += 1
        if c != e:
            acc["mismatch"].append((path, "%r vs %r" % (c, e)))


def run_checks(checks_path, work, mode="quick", extra=None):
    """Evaluate a checks file; return the result dict (not written)."""
    base = os.path.dirname(os.path.abspath(checks_path))
    spec = load_json(checks_path)
    res = []
    for ch in spec["checks"]:
        if ch.get("modes") and mode not in ch["modes"]:
            continue
        r = dict(id=ch["id"], what=ch.get("what", ""), paper=ch.get("paper", ""),
                 tol_abs=ch.get("tol_abs"), tol_rel=ch.get("tol_rel"), mode=ch.get("mode", "scalar"))
        cfile = ch["computed"].get("file", "")
        if ch.get("optional") and cfile and not os.path.exists(_resolve(cfile, base, work)):
            r.update(passed=None, skipped=True, note=ch.get("skip_note", "optional input not available"))
            res.append(r)
            continue
        try:
            c, cf = _fetch(ch["computed"], base, work)
            e, ef = _fetch(ch["expected"], base, work)
            r.update(computed_from=cf, expected_from=ef)
            if r["mode"] == "tree":
                acc = dict(n=0, n_leaves=0, max_abs=0.0, max_rel=0.0, mismatch=[])
                _tree(c, e, ch.get("tol_abs"), ch.get("tol_rel"), set(ch.get("ignore", [])), "", acc)
                r.update(n_leaves=acc["n"], n_leaves_all=acc["n_leaves"], max_abs_diff=acc["max_abs"],
                         max_rel_diff=acc["max_rel"], n_mismatch=len(acc["mismatch"]),
                         first_mismatches=acc["mismatch"][:8],
                         passed=len(acc["mismatch"]) == 0 and acc["n_leaves"] > 0)
            else:
                if isinstance(e, str) or isinstance(c, str):
                    ok, d = (c == e), None
                else:
                    ok, d = _close(c, e, ch.get("tol_abs"), ch.get("tol_rel"))
                r.update(computed=c, expected=e, abs_diff=d, passed=bool(ok))
        except Exception as exc:  # noqa: BLE001
            r.update(passed=False, error="%s: %s" % (type(exc).__name__, exc))
        res.append(r)
    try:
        import numpy
        import scipy
        vers = dict(numpy=numpy.__version__, scipy=scipy.__version__)
    except Exception:  # noqa: BLE001
        vers = {}
    out = dict(step=spec.get("step"), mode=mode, utc=datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), python=platform.python_version(), platform=platform.platform(),
        processor=platform.processor(), cpu_count=os.cpu_count(), **vers,
        n_checks=len(res), n_pass=sum(1 for r in res if r.get("passed")),
        n_skipped=sum(1 for r in res if r.get("skipped")),
        all_pass=all(r.get("passed") for r in res if not r.get("skipped")) and len(res) > 0, checks=res)
    if extra:
        out.update(extra)
    return out


def write_verify(verify_path, result):
    """Merge one mode's result into VERIFY.json (keeps the other mode's record)."""
    old = {}
    if os.path.exists(verify_path):
        try:
            old = load_json(verify_path)
        except Exception:  # noqa: BLE001
            old = {}
    old["step"] = result.get("step")
    old[result["mode"]] = result
    old["summary"] = {m: dict(all_pass=old[m].get("all_pass"), n_pass=old[m].get("n_pass"),
                              n_checks=old[m].get("n_checks"), utc=old[m].get("utc"))
                      for m in ("quick", "full", "reduced") if m in old}
    write_json(verify_path, old, sha=False)
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checks")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    ap.add_argument("--mode", default="quick")
    a = ap.parse_args()
    r = run_checks(a.checks, a.work, a.mode)
    if a.out:
        write_verify(a.out, r)
    for c in r["checks"]:
        print("%-5s %-28s %s" % ("SKIP" if c.get("skipped") else ("PASS" if c.get("passed") else "FAIL"), c["id"],
                                 c.get("error") or ("diff=%s" % (c.get("abs_diff") if c["mode"] == "scalar"
                                                                 else c.get("max_abs_diff")))))
    print("all_pass:", r["all_pass"])
    sys.exit(0 if r["all_pass"] else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
