# PROVENANCE: new module written 2026-10-15 by code-higgs (v12 code/data package).
"""Common driver for ``run.py`` of each result directory (``--quick`` / ``--full`` / ``--reduced``).

    from dnec_common.runner import Step
    st = Step(__file__, chain="higgs")          # step dir = directory of run.py
    a = st.parse()                              # --quick (default) | --full | --reduced, --workers N, --work DIR
    st.stage(["r3/twod_direct_sm/run02/shards2d"])   # copy archived inputs into the work dir (if absent)
    st.py("C149_direct5.py", "post")            # run a script of this step dir, stdout/err -> work log, timed
    st.verify("checks.json")                     # compare with expected/ -> VERIFY.json (per mode)

Work directory: ``<out_root>/<chain>/<mode>`` (env ``DNEC_<CHAIN>_WORK`` overrides, e.g. DNEC_HIGGS_WORK); it
mirrors the archive layout ``r1|r2|r3/<calculation>/<run>/...`` so that computed and archived files share
relative paths.  Archived inputs are looked up, in order, in ``<chain>/data/``, this step's ``expected/`` and
the other steps' ``expected/`` directories of the same chain.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

from . import paths as P


class Step:
    def __init__(self, run_file, chain="higgs"):
        self.dir = os.path.dirname(os.path.abspath(run_file))
        self.name = os.path.basename(self.dir)
        self.chain = chain
        self.chain_dir = os.path.dirname(self.dir)
        self.timings = {}
        self.args = None
        self.work = None

    # ------------------------------------------------------------------ arguments / work dir
    def parse(self, argv=None, extra=None):
        ap = argparse.ArgumentParser(description="%s/%s driver" % (self.chain, self.name))
        g = ap.add_mutually_exclusive_group()
        g.add_argument("--quick", action="store_true", help="fast reproduction + verification (default)")
        g.add_argument("--full", action="store_true", help="full production recomputation")
        g.add_argument("--reduced", action="store_true", help="reduced-size run of the long computation")
        ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
        ap.add_argument("--work", help="work directory (default <out>/<chain>/<mode>)")
        ap.add_argument("--fresh", action="store_true", help="delete this step's outputs in the work dir first")
        ap.add_argument("--no-verify", action="store_true")
        if extra:
            extra(ap)
        a = ap.parse_args(argv)
        a.mode = "full" if a.full else ("reduced" if a.reduced else "quick")
        self.args = a
        env_key = "DNEC_%s_WORK" % self.chain.upper()
        self.work = os.path.abspath(a.work or os.environ.get(env_key) or P.out(self.chain, a.mode))
        os.makedirs(self.work, exist_ok=True)
        os.environ[env_key] = self.work            # inherited by subprocesses and imported modules
        os.environ["DNEC_WORKERS"] = str(a.workers)
        print("[%s] mode=%s work=%s workers=%d" % (self.name, a.mode, self.work, a.workers), flush=True)
        return a

    def w(self, rel):
        return os.path.normpath(os.path.join(self.work, rel))

    # ------------------------------------------------------------------ archived inputs
    def archive_roots(self):
        roots = [os.path.join(self.chain_dir, "data"), os.path.join(self.dir, "expected")]
        roots += sorted(x for x in glob.glob(os.path.join(self.chain_dir, "*", "expected")) if x not in roots)
        return [r for r in roots if os.path.isdir(r)]

    def find_archived(self, rel):
        for r in self.archive_roots():
            p = os.path.join(r, rel)
            if os.path.exists(p):
                return p
        return None

    def stage(self, rels, overwrite=False):
        """Copy archived files/dirs (relative mirror paths) into the work dir unless already present."""
        rec = {}
        for rel in rels:
            dst = self.w(rel)
            src = self.find_archived(rel)
            if src is None:
                raise FileNotFoundError("archived input not found in package: %s" % rel)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                n = 0
                for d, _, files in os.walk(src):
                    for fn in files:
                        if fn.endswith(".sha256"):
                            continue
                        s = os.path.join(d, fn)
                        t = os.path.join(dst, os.path.relpath(s, src))
                        if overwrite or not os.path.exists(t):
                            os.makedirs(os.path.dirname(t), exist_ok=True)
                            shutil.copy2(s, t)
                            n += 1
                rec[rel] = "dir (%d files copied)" % n
            else:
                if overwrite or not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    rec[rel] = "copied"
                else:
                    rec[rel] = "present"
        return rec

    def clear(self, rels):
        for rel in rels:
            p = self.w(rel)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)

    # ------------------------------------------------------------------ running
    def py(self, script, *args, label=None, env=None, cwd=None, check=True):
        """Run ``python -B <script> args`` (script relative to this step dir); log to <work>/logs/."""
        label = label or ("%s %s" % (script, " ".join(str(x) for x in args))).strip()
        logdir = self.w(os.path.join("logs", self.name))
        os.makedirs(logdir, exist_ok=True)
        tag = log_tag(label)
        e = dict(os.environ)
        e.update({k: str(v) for k, v in (env or {}).items()})
        e["PYTHONIOENCODING"] = "utf-8"
        cmd = [sys.executable, "-B", os.path.join(self.dir, script)] + [str(x) for x in args]
        t0 = time.time()
        print("[%s] run: %s" % (self.name, label), flush=True)
        with open(os.path.join(logdir, tag + ".out.txt"), "w", encoding="utf-8") as fo, \
                open(os.path.join(logdir, tag + ".err.txt"), "w", encoding="utf-8") as fe:
            p = subprocess.run(cmd, cwd=cwd or self.dir, env=e, stdout=fo, stderr=fe)
        dt = time.time() - t0
        self.timings[label] = round(dt, 2)
        print("[%s]   -> rc=%d  %.1f s" % (self.name, p.returncode, dt), flush=True)
        if check and p.returncode != 0:
            err = open(os.path.join(logdir, tag + ".err.txt"), encoding="utf-8", errors="replace").read()
            raise RuntimeError("%s failed (rc=%d):\n%s" % (label, p.returncode, err[-3000:]))
        return p.returncode

    def timed(self, label, fn, *a, **k):
        t0 = time.time()
        r = fn(*a, **k)
        self.timings[label] = round(time.time() - t0, 2)
        return r

    # ------------------------------------------------------------------ verification
    def verify(self, checks="checks.json", extra=None):
        sys.path.insert(0, P.pkg("tools"))
        import verify_expected as V  # noqa: E402
        res = V.run_checks(os.path.join(self.dir, checks), self.work, self.args.mode,
                           extra=dict(timings_s=self.timings, workers=self.args.workers, **(extra or {})))
        # R18: a run writes its record under <out>/verify/<chain>/<step>/VERIFY.json (started from the shipped record,
        # so the other modes are kept); the shipped VERIFY.json changes only with DNEC_RECORD=1
        shipped = os.path.join(self.dir, "VERIFY.json")
        rec = shipped
        if os.environ.get("DNEC_RECORD") != "1":
            rec = P.out("verify", os.path.basename(self.chain_dir), self.name, "VERIFY.json", mkdir=True)
            if not os.path.exists(rec) and os.path.exists(shipped):
                shutil.copy2(shipped, rec)
        V.write_verify(rec, res)
        for c in res["checks"]:
            print("  %-4s %-30s %s" % ("SKIP" if c.get("skipped") else ("PASS" if c.get("passed") else "FAIL"), c["id"],
                                      c.get("error") or ("|diff|=%s" % (c.get("abs_diff") if c["mode"] == "scalar"
                                                                       else c.get("max_abs_diff")))), flush=True)
        print("[%s] %s: %d/%d checks passed" % (self.name, self.args.mode, res["n_pass"], res["n_checks"]), flush=True)
        return res


def log_tag(label, maxlen=80):
    """File-name-safe tag of a run label: each character illegal in Windows/POSIX file names (<>:"/\\|?* and control
    characters) and each whitespace character becomes '_' (one for one, so existing tags such as
    'B3_slope_symbolic' are unchanged); never empty (fix 2026-10-15, code-finalize)."""
    tag = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]', "_", label)[:maxlen]
    return tag or "run"


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
