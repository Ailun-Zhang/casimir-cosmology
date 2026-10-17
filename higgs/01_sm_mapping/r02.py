# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/r02.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] run02 helpers (compiled SMDR 1.3): status, unpack of the SMDR tarball (now done by third_party/smdr/fetch_and_build.py).

C148 run02 (sm-mapping): helpers + step 'prep' (contract, STATUS, unpack SMDR into smdr_build, build-info, screen)."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import re
import sys
import json
import time
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r3util as U  # noqa: E402

RES2 = os.path.join(U.ROOT, _zp.w("r3/sm_mapping/run02"))
BUILD = _zp.smdr("..")  # [v12-package] SMDR tree from third_party/smdr
TAR = _zp.smdr("../../download/smdr-1.3.tar.gz")
SRCDIR = os.path.join(BUILD, "SMDR-1.3")


def status2(upd, done=None, files=None):
    p = os.path.join(RES2, "STATUS.json")
    try:
        st = U.jload(p)
    except Exception:  # noqa: BLE001
        st = {"task": "C148 run02: compiled SMDR 1.3 -> true matching-scale variation (Q = Mt/4..4Mt), cross-check "
                      "of the Python port, hazard Delta m_c, rewritten SM-side budget", "agent": "sm-mapping",
              "done": [], "files": [], "key_numbers": {}, "remaining": [], "complete": False}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(st.get(k), dict):
            st[k].update(v)
        else:
            st[k] = v
    for x in (done or []):
        if x not in st["done"]:
            st["done"].append(x)
    for x in (files or []):
        if x not in st["files"]:
            st["files"].append(x)
    st["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    U.wjson(p, st)


def prep():
    os.makedirs(RES2, exist_ok=True)
    U.wjson(os.path.join(RES2, "execution_contract.json"), dict(
        task="run02 (coordinator 20:59Z): compile SMDR 1.3 (+TSIL, 3VIL) with MinGW gcc, verify vs shipped reference "
             "outputs, refit MSbar parameters from on-shell inputs with observables evaluated at Q = xi*default "
             "(xi = 1/4,1/2,1,2,4), run to mu_b and nodes (own RGE and SMDR RG), full hazard Delta m_c, cross-check "
             "the Python port, rewrite budget (pole, mbar), append 'run02 appendix' to D120; run01 untouched",
        build_dir=os.path.relpath(BUILD, U.ROOT), results=os.path.relpath(RES2, U.ROOT),
        compiler=r"gcc (MinGW-w64 5.3, conda env cbuild), mingw32-make; CFLAGS += -O2 "
                 "-D__USE_MINGW_ANSI_STDIO=1"))
    if not os.path.exists(SRCDIR):
        os.makedirs(BUILD, exist_ok=True)
        with tarfile.open(TAR) as tf:
            tf.extractall(BUILD, filter="data")
    for sub in ("tsil-1.46.tar.gz", "3vil-v2.02.tar.gz"):
        p = os.path.join(SRCDIR, sub)
        d = os.path.join(SRCDIR, sub.replace(".tar.gz", ""))
        if os.path.exists(p) and not os.path.exists(d):
            with tarfile.open(p) as tf:
                names = tf.getnames()
                tf.extractall(SRCDIR, filter="data")
            print("unpacked", sub, "->", sorted(set(n.split("/")[0] for n in names)))
    status2({"stage": "prep done (unpacked)", "next_command": "build SMDR with mingw32-make (see r02 STATUS)"},
            done=["run02 prep: contract, unpack to smdr_build"], files=["execution_contract.json"])
    for mf in ("Makefile", "src/Makefile", "applications/Makefile"):
        print("=====", mf)
        print(open(os.path.join(SRCDIR, mf), encoding="utf-8", errors="replace").read())
    for d in os.listdir(SRCDIR):
        p = os.path.join(SRCDIR, d)
        if os.path.isdir(p) and (d.startswith("tsil") or d.startswith("3vil")):
            print("=====", d, sorted(os.listdir(p)))
            mk = os.path.join(p, "Makefile")
            if os.path.exists(mk):
                txt = open(mk, encoding="utf-8", errors="replace").read()
                print("\n".join(l for l in txt.splitlines() if re.match(r"^(CC|CFLAGS|TSIL_|LIB|AR|RANLIB|OPT|\w+ ?=|all|install|clean)", l))[:2500])
    rd = open(os.path.join(SRCDIR, "README.txt"), encoding="utf-8", errors="replace").read().splitlines()
    hits = [i for i, l in enumerate(rd) if re.search(r"(?i)\b(make|install|compil|gcc|TSIL_PATH|3VIL_PATH|calc_fit|SMDR_Fit_Inputs)\b", l)]
    shown = set()
    for i in hits[:40]:
        for j in range(max(0, i - 1), min(len(rd), i + 3)):
            if j not in shown:
                shown.add(j)
                print("README %d: %s" % (j + 1, rd[j]))
    bad = re.compile(r"\b(system|popen|_popen|ShellExecute\w*|WinExec|CreateProcess\w*|execv\w*|execl\w*|unlink|remove|"
                     r"rename|socket|connect|URLDownload\w*)\s*\(")
    flagged = []
    for root, _, fs in os.walk(SRCDIR):
        for fn in fs:
            if fn.endswith((".c", ".h", ".cpp")):
                for k, l in enumerate(open(os.path.join(root, fn), encoding="utf-8", errors="replace")):
                    if bad.search(l):
                        flagged.append("%s:%d: %s" % (os.path.relpath(os.path.join(root, fn), SRCDIR), k + 1, l.strip()[:120]))
    print("SECURITY SCREEN hits:", len(flagged))
    print("\n".join(flagged[:40]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "prep":
        prep()
