"""Blind Method-B solver (B1) at the full-precision couplings of C171 points j10 and j21 (C171_results.json#table).
Usage: python -I blind_at_c171_exact.py <B1 code dir>"""
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, sys.argv[1])
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from ho_mp import resonance_mp  # noqa: E402

C171 = {"j10": (-0.11173797183976203, 1.8506521094818096, 0.006115927402293394, (150, 70, 1.0, 1.0)),
        "j21": (-0.010028169057997869, 1.8987196506819004, 5.079042891085601e-14, (200, 90, 0.7, 0.7))}
for name, (lam, e0, g0, (Nr, Ns, beta, gamma)) in C171.items():
    _, E = resonance_mp(1.0, lam, Nr, beta, Ns, gamma, np.pi / 8, 0.0, 1.9, 40)
    eps, gam = float(E.real), float(-2 * E.imag)
    print(f"{name} Lambda={lam!r} blind eps={mp.nstr(E.real, 14)} Gamma={mp.nstr(-2 * E.imag, 12)} | "
          f"d_eps={eps - e0:+.2e} rel_Gamma={gam / g0 - 1:+.2e}", flush=True)
