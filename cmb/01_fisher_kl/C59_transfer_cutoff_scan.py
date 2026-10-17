# PROVENANCE: 04_calculations/code/20261009_claude_R5/obs_fisher/C59_transfer_cutoff_scan.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] CAMB accuracy-boost scan of the l = 18..26 low-k transfer cutoff (needs CAMB 2.0.4; --full only).

C59_transfer_cutoff_scan: diagnose CAMB's per-l low-k truncation of the CMB transfer functions.

CAMB 2.0.4 (C59_transfer settings) returns Delta_l(k) == 0 for k chi* < 14.217 when l >= 18, with a
finite jump at the first computed node.  This scans accuracy settings and reports, per l, the first
non-zero native node x_c(l) and |Delta_l(x_c)| / max|Delta_l| (jump size).  Output: JSON in results.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package: chain root (cmb/)
import _cmb_paths as WP  # noqa: E402
OUT = WP.wdir("r5/obs_fisher/validate")
OUT.mkdir(parents=True, exist_ok=True)
COSMO = dict(H0=67.4, ombh2=0.02237, omch2=0.1200, mnu=0.06, nnu=3.044, omk=0.0, tau=0.0544, TCMB=2.7255)
BASE = dict(AccuracyBoost=3, lAccuracyBoost=3, lSampleBoost=50, IntkAccuracyBoost=3)


def run(extra, lmax_set=100):
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(**COSMO)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, pivot_scalar=0.05, r=0.0)
    pars.set_for_lmax(lmax_set, lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.DoLensing = False
    for k, v in {**BASE, **extra}.items():
        setattr(pars.Accuracy, k, v)
    res = camb.get_results(pars)
    tr = res.get_cmb_transfer_data("scalar")
    chi = float(res.comoving_radial_distance(res.get_derived_params()["zstar"]))
    L = np.asarray(tr.L); x = np.asarray(tr.q) * chi
    rows = {}
    for l in range(14, 41):
        if l not in L:
            continue
        d = tr.delta_p_l_k[0][int(np.where(L == l)[0][0])]
        nz = np.flatnonzero(d != 0)
        rows[l] = (float(x[nz[0]]), float(abs(d[nz[0]]) / np.max(np.abs(d))))
    return rows


def main():
    trials = {"baseline": {}, "BessIntBoost3": dict(BessIntBoost=3), "BessIntBoost10": dict(BessIntBoost=10),
              "BesselBoost3": dict(BesselBoost=3), "SourcekAccuracyBoost3": dict(SourcekAccuracyBoost=3),
              "TransferkBoost3": dict(TransferkBoost=3)}
    if len(sys.argv) > 1:
        trials = {k: trials[k] for k in sys.argv[1:]}
    out = {}
    for name, extra in trials.items():
        try:
            rows = run(extra)
        except Exception as exc:          # noqa: BLE001
            out[name] = str(exc); print(name, exc); continue
        out[name] = {str(l): dict(x_first=v[0], jump_over_max=v[1]) for l, v in rows.items()}
        worst = max(rows.items(), key=lambda kv: kv[1][1])
        print(f"{name}: l=18 x_c={rows[18][0]:.3f} jump={rows[18][1]:.2e}; l=30 x_c={rows[30][0]:.3f} "
              f"jump={rows[30][1]:.2e}; worst l={worst[0]} jump={worst[1][1]:.2e}", flush=True)
    (OUT / "C59_transfer_cutoff_scan.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
