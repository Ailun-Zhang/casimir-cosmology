# PROVENANCE: copied from 06_paper_draft/figures/v11/fig6_primordial.py on 2026-10-15 by fig-shape; restyled with
# style_v12, tensor ratio written r_T (v12 notation), N_* markers and external bands read from data files.
"""PRD figure prd_primordial (Sec. IX, label fig:c-primordial) - one conclusion: a prepared Casimir turning point leaves
the plateau-inflation predictions untouched; n_s and r_T depend only on the e-folds N_* after pivot exit, i.e. on
reheating.

(a) exact scalar tilt of the plateau V = 3 Mbar_P^2 h^2 (1 - e^{-sqrt(2/3) phi/Mbar_P})^2 vs N_* (C64, second-order
    adiabatic states). Coloured band: N_* between the weak visible decay (c_v = 0.01, 50.33) and instant reheating
    (55.72). Grey bands: 68% intervals of published Lambda-CDM marginals (data/shape/external_primordial_bands.json).
(b) tensor-to-scalar ratio vs N_*, with the BICEP/Keck 2018 bound r_0.05 < 0.036 (95%) and the LiteBIRD target
    delta r_T < 0.001.
One-dimensional comparisons with Lambda-CDM marginals, not a likelihood analysis of the model.
Data: C64_results.json, external_primordial_bands.json.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import style_v12 as st
import _helpers_shape as hs

st.use()
d = st.load_json("shape/C64_results.json")
EXT = st.load_json("shape/external_primordial_bands.json")
N = np.array([r["Nstar"] for r in d["rows"]], float)
ns = np.array([r["ns"] for r in d["rows"]])
rr = np.array([r["r"] for r in d["rows"]])
o = np.argsort(N)
N, ns, rr = N[o], ns[o], rr[o]
special = [x for x in N if abs(x - round(x)) > 1e-6]      # the two reheating histories (non-integer N_*)
N_weak, N_inst = min(special), max(special)
Nf = np.linspace(N.min(), N.max(), 300)
nsf = CubicSpline(N, ns)(Nf)
rf = np.exp(CubicSpline(N, np.log(rr))(Nf))

fig, (a, b) = plt.subplots(1, 2, figsize=(st.DOUBLE, 2.3), gridspec_kw={"wspace": 0.32})
# display labels for the two published bands (values from the data file, in its order)
SHOWN = ("CMB: SPT+ACT+Planck", "ACT+Planck with DESI DR2 BAO")
for band, col, lab in zip(EXT["ns_bands_68"], (st.GREY_FILL2, st.GREY_FILL), SHOWN):
    mu, sig = band["mean"], band["sigma"]
    a.axhspan(mu - sig, mu + sig, color=col, lw=0, zorder=0)
    for edge in (mu - sig, mu + sig):
        a.axhline(edge, color=st.HAIR, lw=0.5, zorder=0.5)
    a.text(41, mu, lab, ha="left", va="center", fontsize=7, color=st.INK2, zorder=4)
a.axvspan(N_weak, N_inst, color=st.BLUE, alpha=0.12, lw=0, zorder=1)
a.plot(Nf, nsf, color=st.INK, lw=1.3, zorder=3)
for Ns, lab in ((N_inst, "instant reheating"), (N_weak, "weak decay")):
    v = ns[np.argmin(np.abs(N - Ns))]
    a.plot([Ns], [v], "o", ms=4.5, color=st.BLUE, mec=st.SURFACE, mew=0.6, zorder=5)
    a.annotate(f"{lab}: {v:.4f}", (Ns, v), xytext=(4, -10), textcoords="offset points", fontsize=7, color=st.INK2)
a.set_xlim(40, 70)
a.set_ylim(0.950, 0.982)
a.set_xlabel(r"$N_*$ (e-folds after pivot exit)")
a.set_ylabel(r"scalar tilt $n_s$")
st.panel_label(a, "(a)", x=0.03, y=0.96)

rmax = EXT["r_upper_95"]["value"]
rtgt = EXT["r_target"]["value"]
b.axvspan(N_weak, N_inst, color=st.BLUE, alpha=0.12, lw=0)
b.plot(Nf, rf, color=st.INK, lw=1.3)
b.axhline(rmax, color=st.INK2, lw=0.8)
b.fill_between([40, 70], rmax, 0.1, color=st.GREY_FILL, lw=0)
b.text(44, rmax * 1.25, rf"excluded by BICEP/Keck: $r_T>{rmax:g}$ (95%)", fontsize=7, color=st.INK2,
       va="bottom")
b.axhline(rtgt, color=st.INK2, lw=0.8, ls=(0, (3, 2)))
b.text(41, rtgt * 1.08, rf"LiteBIRD target $\delta r_T<{rtgt:g}$", fontsize=7, color=st.INK2, va="bottom")
for Ns in (N_inst, N_weak):
    v = rr[np.argmin(np.abs(N - Ns))]
    b.plot([Ns], [v], "o", ms=4.5, color=st.BLUE, mec=st.SURFACE, mew=0.6, zorder=5)
    b.annotate(f"{v:.5f}", (Ns, v), xytext=(5, 3) if Ns == N_inst else (-5, -11), textcoords="offset points",
               fontsize=7, color=st.INK2, ha="left" if Ns == N_inst else "right")
b.set_yscale("log")
b.set_xlim(40, 70)
b.set_ylim(5e-4, 0.1)
hs.plain_log(b, "y")
b.set_xlabel(r"$N_*$ (e-folds after pivot exit)")
b.set_ylabel(r"tensor-to-scalar ratio $r_T$")
st.panel_label(b, "(b)", x=0.03, y=0.96)

print(f"N_* {N_weak}/{N_inst}; n_s {ns[np.argmin(abs(N - N_inst))]:.5f}/{ns[np.argmin(abs(N - N_weak))]:.5f}")
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_primordial", width=st.DOUBLE, papers=("prd",)))
