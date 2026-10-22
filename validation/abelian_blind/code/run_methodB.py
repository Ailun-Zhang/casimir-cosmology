"""Method B production runs: HO product basis + complex rotation, mp Rayleigh quotient.

Writes ../methodB_results.json with every individual run and a per-(e,lambda) summary.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, time, sys
import numpy as np
import mpmath as mp
from ho_mp import resonance_mp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "methodB_results.json")

CASES = [
    (1.0, -1.0), (1.0, -0.11174), (1.0, -0.03), (1.0, -0.01003),
    (0.873, -0.03 * 0.873**2),
    (1.0, -0.05), (1.0, -0.02),
]
# variations: (Nr, Ns, beta, gamma, theta, phi)
PI = np.pi
VARS = [
    (60, 30, 1.0, 1.0, PI / 8, 0.0),
    (80, 40, 1.0, 1.0, PI / 8, 0.0),
    (100, 50, 1.0, 1.0, PI / 8, 0.0),
    (120, 60, 1.0, 1.0, PI / 8, 0.0),
    (150, 70, 1.0, 1.0, PI / 8, 0.0),
    (150, 70, 1.0, 1.0, PI / 6, 0.0),
    (150, 70, 1.0, 1.0, PI / 12, 0.0),
    (150, 70, 1.0, 1.0, PI / 8, -PI / 16),
    (150, 70, 0.7, 0.7, PI / 8, 0.0),
    (150, 70, 1.5, 1.5, PI / 8, 0.0),
    (150, 70, 0.7, 1.5, PI / 8, 0.0),
]


# extra (larger) bases for the smallest width, where the (0.7,0.7) basis at 150x70 is under-resolved
EXTRA = {
    "e=1,lam=-0.01003": [(200, 90, 0.7, 0.7, PI / 8, 0.0), (250, 100, 0.7, 0.7, PI / 8, 0.0),
                         (200, 90, 1.0, 1.0, PI / 8, 0.0), (200, 90, 1.5, 1.5, PI / 8, 0.0)],
}


def summarize(rows, e, lam):
    """Reference = (150,70,beta=gamma=1,theta=pi/8,phi=0).  Error = max over
    (a) |ref - (120,60) same family| (basis-size convergence) and
    (b) deviation of every other variant family (beta,gamma,theta,phi) at its largest basis."""
    ref = [r for r in rows if r["Nr"] == 150 and r["beta"] == 1.0 and r["gamma"] == 1.0
           and abs(r["theta"] - PI / 8) < 1e-12 and r["phi"] == 0.0][0]
    e0, G0 = float(ref["eps"]), float(ref["Gamma"])
    fam = {}
    for r in rows:
        if r["Nr"] < 120:
            continue
        k = (r["beta"], r["gamma"], round(r["theta"], 12), r["phi"])
        if k not in fam or r["Nr"] > fam[k]["Nr"]:
            fam[k] = r
    prev = [r for r in rows if r["Nr"] == 120 and r["beta"] == 1.0 and r["gamma"] == 1.0
            and abs(r["theta"] - PI / 8) < 1e-12 and r["phi"] == 0.0][0]
    cand = list(fam.values()) + [prev]
    de = max(abs(float(r["eps"]) - e0) for r in cand)
    dG = max(abs(float(r["Gamma"]) - G0) for r in cand)
    seq = [(r["Nr"], r["Ns"], r["Gamma"]) for r in rows
           if r["beta"] == 1.0 and r["gamma"] == 1.0 and abs(r["theta"] - PI / 8) < 1e-12 and r["phi"] == 0.0]
    return dict(e=e, lam=lam, eps=ref["eps"], Gamma=ref["Gamma"], eps_abs_err=de, Gamma_rel_err=dG / G0,
                basis_sequence=seq)


def main(cases=CASES, out=OUT):
    results = {"method": "B: 2D-HO(m=0) x 3D-HO(l=0) product basis, complex rotation r->r e^{i theta} "
                         "(optional s->s e^{i phi}), ARPACK shift-invert eigenvector, 40-digit c-product "
                         "Rayleigh quotient", "runs": [], "summary": {}}
    old = []
    if os.path.exists(out):
        try:
            old = json.load(open(out))["runs"]
        except Exception:
            old = []
    for (e, lam) in cases:
        key = f"e={e:.6g},lam={lam:.10g}"
        rows = []
        for (Nr, Ns, b, g, th, ph) in VARS + EXTRA.get(key, []):
            hit = [r for r in old if r["e"] == e and r["lam"] == lam and r["Nr"] == Nr and r["Ns"] == Ns
                   and r["beta"] == b and r["gamma"] == g and r["theta"] == th and r["phi"] == ph]
            if hit:
                row = hit[0]
            else:
                t = time.time()
                Ed, Em = resonance_mp(e, lam, Nr, b, Ns, g, th, ph, sigma=1.9 * e ** (2 / 3))
                row = dict(e=e, lam=lam, Nr=Nr, Ns=Ns, beta=b, gamma=g, theta=th, phi=ph,
                           eps=mp.nstr(Em.real, 20), Gamma=mp.nstr(-2 * Em.imag, 20),
                           eps_double=Ed.real, Gamma_double=-2 * Ed.imag, seconds=time.time() - t)
            rows.append(row)
            results["runs"].append(row)
            print(key, Nr, Ns, b, g, round(th, 4), round(ph, 4), row["eps"], row["Gamma"],
                  f"{row['seconds']:.1f}s", flush=True)
        results["summary"][key] = summarize(rows, e, lam)
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    return results


if __name__ == "__main__":
    main()
