# PROVENANCE: new module written 2026-10-15 by code-higgs (v12 code/data package).
"""Path shim of the Higgs zero-mode chain (imported as ``_zp`` by every packaged module).

The production scripts originally addressed files through an absolute project root.  In the package every such
path was rewritten mechanically (see ``PACKAGE_PATCHES.json``) to one of:

* ``_zp.w("r3/holonomy_cross/run05")``  -- the *work tree*, which mirrors the archive layout
  ``r1|r2|r3/<calculation>/<run>/...`` (r1 = round 1, 2026-10-13; r2 = round 2; r3 = round 3, 2026-10-14).
  Root: env ``DNEC_HIGGS_WORK`` (set by ``run.py``: ``<out>/higgs/<quick|full|reduced>``), default
  ``<out>/higgs/manual``.  The directory (or, for a file path, its parent) is created on demand.
* ``_zp.ZMLIB``                          -- this directory (former per-run code directories).
* ``_zp.smdr("src/betas.c")``            -- the SMDR 1.3 tree built by ``third_party/smdr/fetch_and_build.py``.

Archived inputs are copied into the work tree by ``run.py`` (``dnec_common.runner.Step.stage``) before a step runs.
"""
import os
import sys

ZMLIB = os.path.dirname(os.path.abspath(__file__))
HIGGS = os.path.dirname(ZMLIB)
PKG = os.path.dirname(HIGGS)
if PKG not in sys.path:
    sys.path.append(PKG)
from dnec_common import paths as _P  # noqa: E402

WORK = os.path.abspath(os.environ.get("DNEC_HIGGS_WORK") or _P.out("higgs", "manual"))


def w(rel=""):
    """Absolute path in the work tree; creates the directory (no suffix) or the parent directory (file)."""
    p = os.path.normpath(os.path.join(WORK, rel)) if rel else WORK
    d = p if not os.path.splitext(p)[1] else os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    return p


def smdr(rel=""):
    """Path inside the SMDR 1.3 source/build tree (env DNEC_SMDR overrides)."""
    return os.path.normpath(os.path.join(_P.smdr_dir(), rel)) if rel else _P.smdr_dir()


def step(name, *parts):
    """Path inside a result directory of the Higgs chain, e.g. step('01_sm_mapping', 'c148_fit.c')."""
    return os.path.normpath(os.path.join(HIGGS, name, *parts))
