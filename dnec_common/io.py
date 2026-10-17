# PROVENANCE: new module written 2026-10-15 by code-higgs (v12 code/data package).
"""Small I/O helpers shared by the package: JSON with numpy cleaning, SHA-256 of files, key-path lookup.

``write_json(path, obj, sha=True)`` writes UTF-8 JSON (numpy scalars/arrays and complex numbers converted,
non-finite floats -> null) and, unless ``sha=False``, a sidecar ``<path>.sha256`` holding the SHA-256 of the
written bytes.  ``get_key(obj, "a/b/0/c")`` walks dicts and lists with '/'-separated keys (list indices as ints).
"""
import hashlib
import json
import math
import os

try:  # numpy is optional for this module
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None


def clean(o):
    """Convert numpy / complex / non-finite values into JSON-safe Python objects."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if _np is not None:
        if isinstance(o, _np.ndarray):
            return clean(o.tolist())
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            o = float(o)
        if isinstance(o, _np.complexfloating):
            o = complex(o)
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, complex):
        return [clean(o.real), clean(o.imag)]
    return o


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj, sha=True, indent=1):
    """Write JSON atomically; return the SHA-256 of the file (also stored in <path>.sha256 if sha=True)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=indent)
        f.write("\n")
    os.replace(tmp, path)
    digest = sha256_file(path)
    if sha:
        with open(path + ".sha256", "w", encoding="utf-8") as f:
            f.write("%s  %s\n" % (digest, os.path.basename(path)))
    return digest


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:   # tolerate a UTF-8 BOM (files edited on Windows)
        return json.load(f)


def get_key(obj, keypath, sep="/"):
    """Walk a nested dict/list by a key path such as 'crit/0.1180/P1=0.9' or 'cols/0/gamma'."""
    cur = obj
    if keypath in ("", None):
        return cur
    parts = list(keypath) if isinstance(keypath, (list, tuple)) else keypath.split(sep)   # list form: keys may contain '/'
    for k in parts:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur
