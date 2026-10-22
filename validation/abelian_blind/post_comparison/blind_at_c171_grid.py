"""Evaluate the blind Method-B solver (B1) at the exact couplings of the C171 grid points j10 and j21, after the blind
results were delivered, for a like-for-like comparison.  Usage: python -I blind_at_c171_grid.py <B1 code dir> <out.json>"""
import json
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, sys.argv[1])
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from ho_mp import resonance_mp  # noqa: E402

PI = np.pi
POINTS = {"j10": -0.11173797, "j21": -0.01002817}   # C171_gamma_table.csv, column Lambda (8 decimals)
BASES = {"j10": [(150, 70, 1.0, 1.0, PI / 8), (150, 70, 1.0, 1.0, PI / 6), (120, 60, 1.0, 1.0, PI / 8)],
         "j21": [(200, 90, 0.7, 0.7, PI / 8), (250, 100, 0.7, 0.7, PI / 8), (200, 90, 1.0, 1.0, PI / 8)]}
out = {}
for name, lam in POINTS.items():
    runs = []
    for Nr, Ns, beta, gamma, th in BASES[name]:
        _, E = resonance_mp(1.0, lam, Nr, beta, Ns, gamma, th, 0.0, 1.9, 40)
        eps, Gam = float(E.real), float(-2 * E.imag)
        runs.append({"Nr": Nr, "Ns": Ns, "beta": beta, "gamma": gamma, "theta": th, "eps": mp.nstr(E.real, 16),
                     "Gamma": mp.nstr(-2 * E.imag, 13)})
        print(name, lam, Nr, Ns, beta, round(th, 4), runs[-1]["eps"], runs[-1]["Gamma"], flush=True)
    out[name] = {"Lambda": lam, "runs": runs}
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
