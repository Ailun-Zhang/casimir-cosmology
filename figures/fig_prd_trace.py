# PROVENANCE: copied from 06_paper_draft/figures/v11/fig1_trace.py on 2026-10-15 by fig-shape; restyled with
# style_v12 (decimal log labels, reduced Planck mass in the labels as in the v12 text); curves unchanged.
"""PRD figure prd_trace (Sec. III, label fig:pa-trace) - one conclusion: a conformal sector cannot turn a flat
anisotropic universe around; a positive potential must pay the trace bill while negative (Casimir) energy pays the
energy bill.

(a) largest Hdot/S at H = 0 with non-negative extra trace (Eq. pa-rate) vs the anomaly strength
    y = a S/(24 pi^2 Mbar_P^2), for varrho = a/c of a conformal scalar (1/3), a Dirac fermion (11/18), varrho = 1 and a
    Maxwell field (31/18): always below -5/54.
(b) least negative extra trace required for a turning rate nu = Hdot/S (Eq. pa-budget), varrho = 31/18.
(c) least negative CFT energy zeta_min = -rho_CFT/(Mbar_P^2 S) when a positive-energy component with equation of state
    w supplies the trace (Eq. pa-joint); for w >= 1/3 (shaded) the obstruction applies.
Analytic curves (v11 D11, D12, D18); no data file is read.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

import style_v12 as st
import _helpers_shape as hs

st.use()
fig, (a, b, c) = plt.subplots(1, 3, figsize=(st.DOUBLE, 2.25), gridspec_kw={"wspace": 0.42})
y = np.geomspace(1e-2, 1e2, 400)

# (a) envelope by varrho (ordinal ramp, light -> dark = increasing varrho)
cases = [(1 / 3, "scalar"), (11 / 18, "Dirac"), (1.0, r"$\varrho=1$"), (31 / 18, "Maxwell")]
for (r, lab), col in zip(cases, st.RAMP):
    env = -(1 + 2 * (2 - r) * y) / (6 * (1 + y))
    a.plot(y, env, color=col, lw=1.2)
    yl = -(1 + 2 * (2 - r) * 25.0) / (6 * 26.0)
    below = r > 1.5
    up = {"scalar": 0.041, "Dirac": 0.032}.get(lab, 0.012)  # clear of the own curve, which rises to the left
    a.text(25.0, yl - 0.03 if below else yl + up, lab, fontsize=7.5, color=st.INK2, ha="right",
           va="top" if below else "bottom")
a.axhline(-5 / 54, color=st.INK2, lw=0.7, ls=(0, (3, 2)))
a.text(1.3e-2, -5 / 54 + 0.015, r"collider bound $-5/54$", fontsize=7, color=st.INK2, va="bottom")
a.axhline(0, color=st.MUTED, lw=0.5)
a.set_xscale("log")
a.set_xlim(1e-2, 1e2)
a.set_ylim(-0.62, 0.10)
hs.plain_log(a, "x")
a.yaxis.set_major_formatter(lambda v, p: hs.fmt(v, 1) if abs(v) > 1e-9 else "0")
a.set_xlabel(r"$y=aS/(24\pi^2\bar M_{\rm P}^2)$")
a.set_ylabel(r"largest $\dot H/S$ at $H=0$")
st.panel_label(a, "(a)", x=0.04, y=0.96)

# (b) required negative extra trace at varrho = 31/18
for nu, col in zip([0.0, 0.5, 1.0], st.RAMP[1:]):
    req = 1 + 6 * nu * (1 + y) + (5 / 9) * y
    b.plot(y, req, color=col, lw=1.2)
    b.text(1.15e2, req[-1], rf"$\nu={nu:g}$", color=st.INK2, fontsize=7, va="center", ha="left")
b.set_xscale("log")
b.set_yscale("log")
b.set_xlim(1e-2, 1e2)
hs.plain_log(b, "x")
hs.plain_log(b, "y")
b.set_xlabel(r"$y=aS/(24\pi^2\bar M_{\rm P}^2)$")
b.set_ylabel(r"$-T_{\rm extra}/(\bar M_{\rm P}^2S)$ needed")
st.panel_label(b, "(b)", x=0.04, y=0.96)
b.text(0.96, 0.05, r"$\varrho=31/18$", transform=b.transAxes, ha="right", fontsize=7, color=st.INK2)

# (c) least negative CFT energy vs w
w = np.linspace(-1, 0.31, 500)
for nu, col in [(0.0, st.RAMP[1]), (0.5, st.RAMP[3])]:
    for yy, ls in [(0.0, "-"), (1.0, (0, (4, 2)))]:
        z = 0.5 + (1 + 6 * nu * (1 + yy) + 5 * yy / 9) / (1 - 3 * w)
        c.plot(w, z, color=col, ls=ls, lw=1.2, label=rf"$\nu={nu:g},\ y={yy:g}$")
c.axvspan(1 / 3, 0.5, color=st.GREY_FILL2, lw=0)
c.text(0.37, 1.0, r"extra trace $\geq0$", fontsize=7, color=st.INK2, rotation=90, ha="center", va="bottom")
for wv, lab in [(-1.0, "vacuum"), (0.0, "dust")]:
    z0 = 0.5 + 1 / (1 - 3 * wv)
    c.plot([wv], [z0], "o", ms=4, color=st.INK, mec=st.SURFACE, mew=0.6, zorder=5)
    c.annotate(f"{lab}: {z0:g}", (wv, z0), xytext=(4, -10), textcoords="offset points", fontsize=7, color=st.INK2)
c.set_yscale("log")
c.set_xlim(-1.04, 0.44)
c.set_ylim(0.3, 400)
hs.plain_log(c, "y")
c.xaxis.set_major_formatter(lambda v, p: hs.fmt(v, 1) if abs(v) > 1e-9 else "0")
c.set_xlabel(r"$w$ of the positive-energy component")
c.set_ylabel(r"$\zeta_{\min}=-\rho_{\rm CFT}/(\bar M_{\rm P}^2S)$")
c.legend(loc="upper left", fontsize=7, ncol=1, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
st.panel_label(c, "(c)", x=0.90, y=0.96)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_trace", width=st.DOUBLE, papers=("prd",)))
