# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/c150_nnlo_central.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261014_claude_v12_r3/twod_direct_sm/c150_nnlo_central.py on 2026-10-14 by agent holonomy-cross (C149 run02).
# PROVENANCE: new adapter by twod-direct (C150 run02, 2026-10-14) around T1's sm_provider.py (copied unchanged next to this file).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Adapter couplings(mt, als, mu) -> sm_provider variant 'central' (M_h = 125.13) for C150_twod (env C150_COUPLINGS=module:<this file>).

couplings(mt, als, mu) -> dict(g2, gY, yt, lam) from T1 (C148/D120) sm_provider, variant 'central'
(SMDR25 NNLO matching at mu0 = M_t, full 3-loop + higher RGE, no constant calibration), M_h = 125.13 as in C144/T1.
Used via env C150_COUPLINGS=module:<this file>.  Identical call to T1's C144_model.coup (cal = 0)."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sm_provider as SP  # noqa: E402

MH = 125.13
VARIANT = os.environ.get("C150_PROVIDER_VARIANT", "central")


def couplings(mt, als, mu):
    c = SP.coup_point(mt, als, MH, mu, VARIANT)
    return dict(g2=c["g2"], gY=c["gY"], yt=c["yt"], lam=c["lam"])
