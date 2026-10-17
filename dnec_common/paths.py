# PROVENANCE: new module written 2026-10-15 by code-higgs (v12 code/data package).
"""Package-relative paths for the DNEC v12 code/data release.

Every script of the package resolves files through this module, never through absolute paths.

Layout (all relative to the package root ``PKG_ROOT`` = the directory that contains ``dnec_common/``):

* ``pkg(*parts)``          -- a file shipped with the package (code, ``expected/``, ``inputs/``, ``data/``).
* ``out(*parts)``          -- writable output tree, default ``<PKG_ROOT>/output``; override with env ``DNEC_OUT``.
* ``third_party(*parts)``  -- third-party sources/builds, default ``<PKG_ROOT>/third_party``;
                              override with env ``DNEC_THIRD_PARTY``.
* ``smdr_dir()``           -- compiled SMDR 1.3 tree (see ``third_party/smdr/fetch_and_build.py``);
                              override with env ``DNEC_SMDR``.

Environment overrides are read at call time, so a driver may set them before importing step code.
The interface is deliberately tiny and stable (used by higgs/, shape/, cmb/ and figures/).
"""
import os

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pkg(*parts):
    """Path of a file shipped inside the package (read-only by convention)."""
    return os.path.normpath(os.path.join(PKG_ROOT, *parts))


def out_root():
    """Root of the writable output tree (env DNEC_OUT, default <pkg>/output)."""
    return os.path.normpath(os.environ.get("DNEC_OUT") or os.path.join(PKG_ROOT, "output"))


def out(*parts, mkdir=False):
    """Path below the output root; with mkdir=True the parent directory (or the directory itself if the
    last component has no suffix) is created."""
    p = os.path.normpath(os.path.join(out_root(), *parts))
    if mkdir:
        d = p if not os.path.splitext(p)[1] else os.path.dirname(p)
        os.makedirs(d, exist_ok=True)
    return p


def third_party(*parts):
    """Third-party tree (env DNEC_THIRD_PARTY, default <pkg>/third_party)."""
    base = os.environ.get("DNEC_THIRD_PARTY") or os.path.join(PKG_ROOT, "third_party")
    return os.path.normpath(os.path.join(base, *parts))


def smdr_dir():
    """Directory of the unpacked/compiled SMDR 1.3 tree (env DNEC_SMDR,
    default <third_party>/smdr/build/SMDR-1.3)."""
    return os.path.normpath(os.environ.get("DNEC_SMDR") or third_party("smdr", "build", "SMDR-1.3"))


def rel(path):
    """Path relative to the package root (for logs/records; never write absolute paths into outputs)."""
    try:
        return os.path.relpath(path, PKG_ROOT).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return os.path.basename(path)
