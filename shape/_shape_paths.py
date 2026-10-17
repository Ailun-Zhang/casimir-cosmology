# PROVENANCE: new module written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""Work-tree paths of the shape chain (imported as ``WP`` by every packaged script of shape/).

The production scripts originally addressed files through an absolute project root.  In the package every such path
was rewritten (see ``PACKAGE_PATCHES.json``) to the *work tree*, which mirrors the archive layout of the research
project: ``r5/`` (round R5, 2026-10-09), ``fu/`` (follow-up, 2026-10-11), ``r1/ r2/ r3/`` (v12 research rounds 1-3,
2026-10-13/14), ``ai/`` (Argon/Codex intake, 2026-10-13), so a recomputed file and its archived copy in
``<step>/expected/`` have the same relative path.

Work-tree root: env ``DNEC_SHAPE_WORK`` (set by every ``run.py``: ``<out>/shape/<quick|full|reduced>``), default
``<out>/shape/manual`` (``<out>`` = env ``DNEC_OUT`` or ``<package>/output``).
"""
import glob
import os
import shutil
import sys
from pathlib import Path

CHAIN = "shape"
CHAIN_DIR = Path(__file__).resolve().parent
PKG = CHAIN_DIR.parent
if str(PKG) not in sys.path:
    sys.path.append(str(PKG))
from dnec_common import paths as _P  # noqa: E402

ENV = "DNEC_SHAPE_WORK"


def work_root():
    return Path(os.environ.get(ENV) or _P.out(CHAIN, "manual")).resolve()


def wdir(rel=""):
    """Directory in the work tree (created)."""
    p = work_root() / rel if rel else work_root()
    p.mkdir(parents=True, exist_ok=True)
    return p


def wfile(rel):
    """File in the work tree (parent directory created)."""
    p = work_root() / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def step_dir(name, *parts):
    """Path inside a result directory of this chain, e.g. step_dir('01_sm_casimir_weight')."""
    return CHAIN_DIR.joinpath(name, *parts)


def pkg(*parts):
    return PKG.joinpath(*parts)


def find_archived(rel):
    """Archived copy of a mirror path in any expected/ or data/ directory of casimir/, shape/, cmb/
    (this chain first)."""
    chains = [CHAIN] + [c for c in ("casimir", "shape", "cmb") if c != CHAIN]
    for c in chains:
        for root in [PKG / c / "data"] + sorted(Path(x) for x in glob.glob(str(PKG / c / "*" / "expected"))):
            p = root / rel
            if p.exists():
                return p
    return None


def stage(rels, overwrite=False):
    """Copy archived inputs (mirror paths, files or directories) into the work tree unless already present;
    a '<dir>.zip' archive is unpacked when '<dir>' itself is not shipped."""
    rec = {}
    for rel in rels:
        dst = work_root() / rel
        src = find_archived(rel)
        if src is None:
            z = find_archived(rel.rstrip("/") + ".zip")
            if z is None:
                raise FileNotFoundError("archived input not found in the package: %s" % rel)
            import zipfile
            with zipfile.ZipFile(z) as zf:
                # [code-finalize 2026-10-16] unpack every member that is not yet in the work tree: an existing (e.g.
                # empty, pre-created) directory no longer blocks the unpacking; fresh files are never overwritten
                # unless overwrite=True.  (Before: an existing '<dir>' was taken as 'present' -> empty C76 draws.)
                members = [m for m in zf.namelist() if not m.endswith("/") and (overwrite or not (dst.parent / m).exists())]
                if members:
                    zf.extractall(dst.parent, members=members)
            rec[rel] = "unzipped %s (%d files)" % (z.name, len(members)) if members else "present"
            continue
        if src.is_dir():
            n = 0
            for d, _, files in os.walk(src):
                for fn in files:
                    if fn.endswith(".sha256"):
                        continue
                    s = Path(d) / fn
                    t = dst / s.relative_to(src)
                    if overwrite or not t.exists():
                        t.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(s, t)
                        n += 1
            rec[rel] = "dir (%d files copied)" % n
        elif overwrite or not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rec[rel] = "copied"
        else:
            rec[rel] = "present"
    return rec
