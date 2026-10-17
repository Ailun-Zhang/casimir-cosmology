# PROVENANCE: build logic adapted from 04_calculations/code/20261014_claude_v12_r3/sm_mapping/build_smdr.py and r02.py
# (C148 run02, sm-mapping) on 2026-10-15 by code-higgs; download/verification steps are new.
"""Fetch, verify and build SMDR 1.3 (Martin & Robertson, arXiv:1907.02500) with TSIL 1.46 and 3VIL 2.02.

SMDR is NOT redistributed in this package.  This script downloads the official tarball, checks its SHA-256, unpacks
it under third_party/smdr/build/ and compiles the three libraries, the stock applications and the package driver
higgs/01_sm_mapping/c148_fit.c (used by run02_fits.py for the compiled matching-scale refits).

    python third_party/smdr/fetch_and_build.py all                # download + verify + unpack + libs + apps + driver
    python third_party/smdr/fetch_and_build.py all --tarball PATH # use a local copy of smdr-1.3.tar.gz (verified)
    python third_party/smdr/fetch_and_build.py {fetch|unpack|libs|apps|driver|test}
Options: --cc gcc   --ar ar   (default: gcc/ar on PATH, or $DNEC_MINGW_BIN/gcc.exe on Windows)

Windows: MinGW-w64 gcc (tested: gcc 5.3 from the conda env 'cbuild' = conda-forge m2w64 toolchain); flags
  -O2 -D__USE_MINGW_ANSI_STDIO=1 (long double printf), -I mingw_stubs (empty execinfo.h), 3VIL/SMDR with -fcommon,
  TSIL with its Makefile flags (-DTSIL_SIZE_LONG), executables linked with a 64 MB stack (-Wl,--stack,67108864).
  The stock calc_fit.exe hit an access violation on the test machine; c148_fit.c (a verbatim copy of SMDR_Fit_Inputs
  with the evaluation scales as arguments) is used instead.
Linux/macOS: system gcc/clang; same manual build (no MinGW flags); `ulimit -s unlimited` recommended for the fits.
Equivalent upstream route on POSIX: `make` in tsil-1.46, 3vil-v2.02 and SMDR-1.3 (see SMDR README.txt).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
URL = "https://www.niu.edu/spmartin/smdr/smdr-1.3.tar.gz"      # official page: https://www.niu.edu/spmartin/SMDR/
VERSION = "1.3"
SHA256 = "398b3947a6e5e02244d5adcd45911a3825e056a1b6f65bcf203975da0fc91b41"
NBYTES = 2220791
DL = os.path.join(HERE, "download")
TAR = os.path.join(DL, "smdr-1.3.tar.gz")
BUILD = os.path.join(HERE, "build")
S = os.environ.get("DNEC_SMDR") or os.path.join(BUILD, "SMDR-1.3")
TS, TV = os.path.join(S, "tsil-1.46"), os.path.join(S, "3vil-v2.02")
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
DRIVER = os.path.join(PKG, "higgs", "01_sm_mapping", "c148_fit.c")
LOG = os.path.join(BUILD, "logs")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tools(a):
    mb = os.environ.get("DNEC_MINGW_BIN", "")
    cc = a.cc or (os.path.join(mb, "gcc.exe") if (WIN and mb) else "gcc")
    ar = a.ar or (os.path.join(mb, "ar.exe") if (WIN and mb) else "ar")
    env = dict(os.environ)
    if WIN and mb:
        env["PATH"] = mb + os.pathsep + env.get("PATH", "")
    extra = ["-O2"] + (["-D__USE_MINGW_ANSI_STDIO=1", "-I" + os.path.join(HERE, "mingw_stubs")] if WIN else [])
    return cc, ar, env, extra


def run(cmd, cwd, env):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")
    return p.returncode, p.stdout + p.stderr


def fetch(a):
    """Download (or copy --tarball) into third_party/smdr/download/ and verify the SHA-256."""
    os.makedirs(DL, exist_ok=True)
    if a.tarball:
        shutil.copy2(a.tarball, TAR)
        src = "local copy: " + os.path.basename(a.tarball)
    elif not os.path.exists(TAR):
        print("downloading", URL, flush=True)
        with urllib.request.urlopen(URL, timeout=120) as r, open(TAR + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(TAR + ".part", TAR)
        src = URL
    else:
        src = "already present"
    h = sha256(TAR)
    ok = h == SHA256 and os.path.getsize(TAR) == NBYTES
    rec = {"smdr-1.3.tar.gz": dict(url=URL, version=VERSION, source=src, date=time.strftime("%Y-%m-%d"),
                                   bytes=os.path.getsize(TAR), sha256=h, sha256_expected=SHA256, verified=ok)}
    with open(os.path.join(DL, "SOURCES.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("sha256 %s -> %s" % (h, "OK" if ok else "MISMATCH"), flush=True)
    if not ok:
        sys.exit("SHA-256 mismatch: refusing to build")


def unpack(a):
    if os.path.exists(os.path.join(S, "src", "smdr.h")):
        print("already unpacked:", S)
        return
    os.makedirs(BUILD, exist_ok=True)
    with tarfile.open(TAR) as tf:
        tf.extractall(BUILD, filter="data")
    for sub in ("tsil-1.46.tar.gz", "3vil-v2.02.tar.gz"):
        p = os.path.join(S, sub)
        if os.path.exists(p) and not os.path.isdir(p.replace(".tar.gz", "")):
            with tarfile.open(p) as tf:
                tf.extractall(S, filter="data")
    print("unpacked into", S, flush=True)


def mkvars(path):
    txt = open(path, encoding="utf-8", errors="replace").read().replace("\\\n", " ")
    v = {}
    for ln in txt.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*(#.*)?$", ln)
        if m and m.group(1) not in v:
            v[m.group(1)] = m.group(2).strip()
    return v


def build_lib(d, srcs, cflags, lib, outdir, a):
    cc, ar, env, _ = tools(a)
    jobs = [(s, [cc] + cflags + ["-c", s, "-o", s[:-2] + ".o"]) for s in srcs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda j: (j[0],) + run(j[1], d, env), jobs))
    bad = [(s, out[-800:]) for s, rc, out in res if rc != 0]
    if bad:
        return dict(ok=False, failed=bad[:5], n_failed=len(bad))
    rc, out = run([ar, "rcs", os.path.join(outdir, lib)] + [s[:-2] + ".o" for s in srcs], d, env)
    return dict(ok=rc == 0, n_obj=len(srcs), warnings=sum(o.count("warning") for _, _, o in res), ar=out[-300:])


def libs(a):
    _, _, _, extra = tools(a)
    rep = {}
    for d, pre, lib in ((TS, "TSIL", "libtsil.a"), (TV, "TVIL", "lib3vil.a")):
        v = mkvars(os.path.join(d, "Makefile"))
        flags = [x for k in (pre + "_RESCALE", pre + "_SIZE") for x in v.get(k, "").split()]
        flags += extra + (["-fcommon"] if pre == "TVIL" else [])
        t0 = time.time()
        rep[lib] = dict(flags=flags, **build_lib(d, v[pre + "_SRCS"].split(), flags, lib, d, a), s=round(time.time() - t0, 1))
        print(lib, json.dumps(rep[lib])[:300], flush=True)
    v = mkvars(os.path.join(S, "src", "Makefile"))
    flags = ["-I" + TS, "-I" + TV] + extra + ["-fcommon"]
    t0 = time.time()
    rep["libsmdr.a"] = dict(flags=flags, **build_lib(os.path.join(S, "src"), v["SMDR_SRCS"].split(), flags, "libsmdr.a", S, a),
                            s=round(time.time() - t0, 1))
    print("libsmdr.a", json.dumps(rep["libsmdr.a"])[:300], flush=True)
    for h in (os.path.join(S, "src", "smdr.h"), os.path.join(TS, "tsil.h"), os.path.join(TV, "3vil.h")):
        shutil.copy(h, S)
    for lib, d in (("libtsil.a", TS), ("lib3vil.a", TV)):
        shutil.copy(os.path.join(d, lib), S)
    shutil.copy(os.path.join(S, "benchmark_data", "ReferenceModel.dat"), S)
    if not all(x.get("ok") for x in rep.values()):
        sys.exit("library build failed: %s" % json.dumps(rep)[:2000])
    return rep


def link(src, exe, a, inc_src=False):
    cc, _, env, extra = tools(a)
    inc = ["-I" + S, "-I" + TS, "-I" + TV] + (["-I" + os.path.join(S, "src")] if inc_src else [])
    lflags = ["-Wl,--stack,67108864"] if WIN else []
    return run([cc] + inc + extra + [src, "-L" + S, "-lsmdr", "-l3vil", "-ltsil", "-lm"] + lflags + ["-o", exe], S, env)


def apps(a):
    out = {}
    for app in ("calc_all", "calc_fit", "calc_RGrun"):
        rc, txt = link(os.path.join(S, "applications", app + ".c"), os.path.join(S, app + EXE), a)
        out[app] = rc
        print(app, "rc=%d" % rc, txt[-300:], flush=True)
    return out


def driver(a):
    rc, txt = link(DRIVER, os.path.join(S, "c148_fit" + EXE), a, inc_src=True)
    print("c148_fit rc=%d" % rc, txt[-600:], flush=True)
    if rc != 0:
        sys.exit("driver build failed")


def test(a):
    """One c148_fit refit (172.60, 125.13, 0.1180, all evaluation scales x1), non-interactive, with a timeout; the
    result is compared with the archived production fit by higgs/01_sm_mapping/run.py (check compiled_refit_172p60).
    The stock application calc_all is optional (--with-calc-all): with MinGW-w64 gcc 5.3 on Windows it did not finish
    within 4 h on the test machine (killed; 14,076 s CPU) and the stock calc_fit hit an access violation in round 3 -
    neither is used by the package."""
    _, _, env, _ = tools(a)
    os.makedirs(LOG, exist_ok=True)
    ok = True
    if a.with_calc_all:
        try:
            p = subprocess.run([os.path.join(S, "calc_all" + EXE)], cwd=S, env=env, capture_output=True, text=True,
                               errors="replace", timeout=a.timeout, stdin=subprocess.DEVNULL)
            open(os.path.join(LOG, "calc_all_stdout.txt"), "w", encoding="utf-8").write(p.stdout + p.stderr)
            print("calc_all rc=%d (stdout -> build/logs/calc_all_stdout.txt)" % p.returncode)
        except subprocess.TimeoutExpired:
            print("calc_all: timeout after %d s (known behaviour of the MinGW build; not used by the package)" % a.timeout)
    out = os.path.join(LOG, "fit_xi1_Mt172.60.txt")
    t0 = time.time()
    try:
        p = subprocess.run([os.path.join(S, "c148_fit" + EXE), "fit", "172.6000", "125.1300", "0.11800", "1", "1", "1", "1",
                            "1", "172.6000", "1e-10", out], cwd=S, env=env, capture_output=True, text=True, errors="replace",
                           timeout=a.timeout, stdin=subprocess.DEVNULL)
        ok = p.returncode == 0 and os.path.exists(out)
        print("c148_fit rc=%d %.1f s -> %s" % (p.returncode, time.time() - t0, os.path.relpath(out, PKG)))
        if ok:
            print(open(out, encoding="utf-8").read()[:600])
    except subprocess.TimeoutExpired:
        ok = False
        print("c148_fit: timeout after %d s" % a.timeout)
    if not ok:
        sys.exit("c148_fit test failed")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["all", "fetch", "unpack", "libs", "apps", "driver", "test"])
    ap.add_argument("--tarball")
    ap.add_argument("--cc")
    ap.add_argument("--ar")
    ap.add_argument("--with-calc-all", action="store_true", help="also run the stock calc_all (slow/hangs with MinGW)")
    ap.add_argument("--timeout", type=int, default=600, help="per-executable timeout in seconds for 'test'")
    a = ap.parse_args()
    steps = ["fetch", "unpack", "libs", "apps", "driver", "test"] if a.step == "all" else [a.step]
    for s in steps:
        t0 = time.time()
        globals()[s](a)
        print("== %s done in %.1f s" % (s, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
