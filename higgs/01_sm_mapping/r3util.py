# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/r3util.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Helpers of the SM-mapping step (paths, JSON I/O, contract).

C148 / D120 (agent sm-mapping, DNEC v12 round 3): shared paths, json/STATUS helpers."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import math
import time
import hashlib

ROOT = _zp.WORK
CODE = os.path.join(ROOT, _zp.ZMLIB)
RES = os.path.join(ROOT, _zp.w("r3/sm_mapping/run01"))
LIT = _zp.smdr("../../download")  # [v12-package] was the literature batch holding the SMDR tarball/sources
DER = os.path.join(ROOT, _zp.w("derivations/20261014_claude_v12_r3/sm_mapping"))
R2 = os.path.join(ROOT, _zp.ZMLIB)
R2RES = os.path.join(ROOT, _zp.w("r2/higgs_final/run01"))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            o = float(o)
    except ImportError:
        pass
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, complex):
        return [clean(o.real), clean(o.imag)]
    return o


def wjson(path, obj):
    if not os.path.isabs(path):
        path = os.path.join(RES, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def jload(path):
    if not os.path.isabs(path):
        path = os.path.join(RES, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def status(upd, done=None, files=None):
    """merge into RES/STATUS.json (lists in 'done'/'files' are appended uniquely)."""
    p = os.path.join(RES, "STATUS.json")
    try:
        st = jload(p)
    except Exception:  # noqa: BLE001
        st = {"task": "C148/D120 T1 SM mapping NNLO (3-loop RGE + NNLO matching, errors, critical m_t)",
              "agent": "sm-mapping", "done": [], "files": [], "key_numbers": {}, "remaining": [],
              "complete": False}
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
    wjson(p, st)
    return st


def contract():
    c = {
        "task": "C148/D120 T1 sm_mapping: own 3-loop SM RGE + NNLO matching at mu0=M_t; couplings at mu_b=MPbar/722 "
                "and along the bounce nodes mu(u)=mu_b/sqrt(cosh 2u), u in [0,5]; SM-side errors (matching scale, "
                "truncation, scheme) + m_t definition; critical m_t recomputed through a copied C144 hazard.",
        "agent": "sm-mapping", "date": "2026-10-14",
        "inputs": {
            "buttazzo_2013": "02_literature/20261007_074655UTC/model_completion/C52_Buttazzo_etal_2013.pdf (+_embedded.txt)",
            "r2_rge": _zp.ZMLIB,
            "r2_hazard": _zp.ZMLIB,
            "downloads": "02_literature/batches/20261014_claude_v12_r3_L37/sm_mapping/ (SOURCES.json)"},
        "method": [
            "3-loop SM RGE (g1,g2,g3,yt,lambda; yb,ytau optional) transcribed from Buttazzo App.B Eqs.(96)-(101) + "
            "4-loop QCD; cross-checked against an independent source (SMDR source betas or literature) term by term",
            "matching at mu=M_t: (i) Buttazzo Eqs.(55)-(60) (+ exact 4-loop QCD yt shift via RunDec); "
            "(ii) SMDR (Martin-Robertson 2019) fit formulas / reference point; compare",
            "errors by recomputation: mu0 in [M_t/2, 2M_t]; loop truncation (2L vs 3L vs 3L+4L terms, threshold "
            "higher orders); scheme (inputs set, tadpole convention via (i) vs (ii)); parametric (M_h, alpha_s) "
            "separately; m_t definition (pole / MSbar mt(mt) / MC)",
            "propagation: copied C144 hazard with couplings from sm_provider; alpha_s = 0.1180 (+0.1171, 0.1189); "
            "full hazard rerun for main error variants; linearized m_c <-> lambda_c consistency"],
        "pass_criteria": {"SM_side_error_on_m_c": "<= +-0.1 GeV (matching scale + truncation + scheme, computed by "
                                                  "own code); otherwise report honestly with dominant source",
                          "new_central_m_c_vs_r2_173.02": "difference explained"},
        "budget": {"tool_calls": 110, "tokens_target": 800000, "tokens_hard": 1600000, "workers": 4},
        "outputs": {"results": _zp.w("r3/sm_mapping/run01/"),
                    "code": _zp.ZMLIB,
                    "derivation": _zp.w("derivations/20261014_claude_v12_r3/sm_mapping/D120_SM_MAPPING_NNLO_CN.md"),
                    "summary": "C148_summary.json"}}
    wjson("execution_contract.json", c)
    return c


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        os.makedirs(RES, exist_ok=True)
        contract()
        status({"stage": "init", "next_command": "python -I -B dl_sources.py  (downloads + SOURCES.json)",
                "remaining": ["downloads", "RGE 3L", "matching", "benchmarks", "errors", "hazard", "D120"]},
               done=["execution_contract.json written"], files=["execution_contract.json", "STATUS.json"])
        print("init ok", RES)
