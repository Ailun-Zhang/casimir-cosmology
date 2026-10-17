# PROVENANCE: replaces 06_paper_draft/figures/v11/fig7_fossil_sky.py (copied 2026-10-15 by fig-shape for the panel
# layout and the circle-bound shading); the v11 data (C59 l<=30 Fisher widths, C75 typical fossil, C79 nine-realization
# ensemble) are superseded by the C145/D117 ensemble at l<=80 with profiled cube volume and by its next-order rerun.
"""PRD figure prd_fossil_sky (Sec. X, label fig:c-sky) - one conclusion: whether the microwave background reads the
shape of the torus is decided by the shortest cycle alone (read edge u ~ 1.02-1.04 d_LSS); over the predicted
distribution of shapes the probability of a readable fossil with no matched circles peaks at 0.34 near
L0 = 1.075 d_LSS (0.38 at next order) and exceeds 0.1 only for 1.018 < L0/d_LSS < 1.183; the integrated readable
measure, 0.042, does not depend on sigma_E.

(a) Kullback-Leibler divergence D_prof of 55 computed fossils from the closest circle-free cube (temperature and E
    modes, 2 <= l <= 80, full sky, cube volume profiled over the pool) vs their shortest cycle u (T12 cleanup table,
    fossil_table_v2.csv: column s = L0/d_LSS, xi = sin 3 theta); dotted: readability criterion D = 4.5; band: read
    edges u_max of the 15 interpolated shapes (S3_postcheck TE80 edges, all of lo/hi/chord: 1.021-1.039); grey:
    excluded by matched circles (u < 0.985).
(b) P_joint(L0) = P(readable and no circles) (solid) and P_cond = P(readable | no circles) (dashed) for the E-plane
    Gaussian ensemble, sigma_E = 0.1041 (thick; S3_postcheck TE80_grid, band: lo/hi edge interpolations) and the
    next-order sigma_E = 0.0940 (thin; C153 ensemble); dots: peaks (archived refined maxima); bar: P_joint >= 0.1.
Data: fossil_table_v2.csv, S3_postcheck.json, C153_ensemble.json, C153_summary.json.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

import style_v12 as st
import _helpers_shape as hs

st.use()
TAB = st.load_csv("shape/fossil_table_v2.csv")
S3 = st.load_json("shape/S3_postcheck.json")
ENS = st.load_json("shape/C153_ensemble.json")
NXS = st.load_json("shape/C153_summary.json")["B_next_order"]["ensemble"]
U_CIRCLE = 0.985
D_READ = 4.5

fig, (a, b) = plt.subplots(1, 2, figsize=(st.DOUBLE, 2.5), gridspec_kw={"wspace": 0.28})

# (a) KL divergence vs shortest cycle -------------------------------------------------------------------------------------
assert len(TAB) == 55
u = np.array([float(r["u"]) for r in TAB])
D = np.array([float(r["TE80_prof"]) for r in TAB])
lab = np.array([r["lab"] for r in TAB])
edges = S3["variants"]["TE80_grid"]["edges"]
ends = np.array([[e["lo"], e["hi"], e["chord"]] for e in edges.values()])   # all edge interpolations
e_lo, e_hi = ends.min(), ends.max()
a.axvspan(0.95, U_CIRCLE, color=st.GREY_FILL, lw=0, zorder=0)
a.axvspan(e_lo, e_hi, color=st.AQUA_FILL, lw=0, zorder=0)
a.axhline(D_READ, color=st.INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
for key, m, name in (("obl", "v", "oblate"), ("tri", "o", "triaxial"), ("pro", "^", "prolate")):
    k = lab == key
    a.plot(u[k], D[k], m, ms=3.6, mfc=st.SURFACE if key == "tri" else st.ORANGE, mec=st.ORANGE, mew=0.9,
           ls="none", zorder=4, label=name)
a.text(0.9865, 0.0125, "matched\ncircles", fontsize=7, color=st.INK2, ha="left", va="bottom", linespacing=1.0)
a.text(1.135, D_READ * 1.25, rf"$D={D_READ:g}$", fontsize=7, color=st.INK2, ha="right", va="bottom")
a.annotate(f"readout edges\n{e_lo:.3f}–{e_hi:.3f}", (0.5 * (e_lo + e_hi), 300), xytext=(1.06, 300), fontsize=7,
           color=st.INK2, ha="left", va="center", linespacing=1.0,
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=1, shrinkB=1))
a.set_yscale("log")
a.set_xlim(0.975, 1.14)
a.set_ylim(1e-2, 3e3)
hs.plain_log(a, "y")
a.set_xlabel(r"shortest cycle $u$ ($d_{\rm LSS}$)")
a.set_ylabel(r"KL divergence $D$ (nats)")
a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), fontsize=7, handletextpad=0.2, borderaxespad=0.3,
         labelspacing=0.25)
st.panel_label(a, "(a)", x=0.03, y=0.96)

# (b) ensemble readability ------------------------------------------------------------------------------------------------
L3 = np.array(S3["L0_pick"])
mid = S3["variants"]["TE80_grid"]["modes"]["mid"]
lo, hi = (S3["variants"]["TE80_grid"]["modes"][k] for k in ("lo", "hi"))
Le = np.array(ENS["L0_pick"])
nx = ENS["results"]["0.0940"]["variants"]["TE80_grid"]["mid"]
nxs = NXS["0.0940"]["TE80_grid"]["mid"]
hs.check("next-order peak (ensemble file vs summary)", nx["Pj_max"], nxs["Pj_max"], 1e-12)
b.axvspan(0.95, U_CIRCLE, color=st.GREY_FILL, lw=0, zorder=0)


def smooth(L, P, start_zero=True):
    L, P = np.asarray(L, float), np.asarray(P, float)
    if start_zero:
        L, P = np.r_[U_CIRCLE, L], np.r_[0.0, P]
    f = PchipInterpolator(L, P)
    x = np.linspace(L[0], L[-1], 400)
    return x, f(x)


xb, ylo = smooth(L3, lo["Pj"])
_, yhi = smooth(L3, hi["Pj"])
b.fill_between(xb, np.minimum(ylo, yhi), np.maximum(ylo, yhi), color=st.ORANGE, alpha=0.22, lw=0, zorder=1)
for L, m, lw in ((L3, mid, 1.5), (Le, nx, 0.8)):
    x, y = smooth(L, m["Pj"])
    b.plot(x, y, color=st.ORANGE, lw=lw, zorder=3)
    x, y = smooth(L, m["Pc"], start_zero=False)
    b.plot(x, y, color=st.ORANGE, lw=lw, ls=(0, (4, 2)), zorder=3)
for m in (mid, nxs):
    b.plot([m["L0_Pj_max"]], [m["Pj_max"]], "o", ms=3.6, color=st.ORANGE, mec=st.SURFACE, mew=0.6, zorder=5)
w_lo, w_hi = mid["win_joint"]["0.1"][:2]
wn_lo, wn_hi = nxs["win_joint"]["0.1"][:2]
b.plot([w_lo, w_hi], [0.1, 0.1], color=st.INK, lw=1.6, solid_capstyle="butt", zorder=4)
b.text(0.5 * (w_lo + w_hi), 0.078, r"$P_{\rm joint}\geq0.1$: " + f"{w_lo:.3f}–{w_hi:.3f}", fontsize=7,
       color=st.INK2, ha="center", va="top")
meas, meas_nx = mid["mean_span"], nxs["integrated_measure"]
b.text(1.292, 0.95, rf"$\int P_{{\rm joint}}\,d\ln L_0={meas:.3f}$, {meas_nx:.3f}" + "\n"
       + rf"peak $P_{{\rm joint}}={mid['Pj_max']:.3f}$ at {mid['L0_Pj_max']:.3f}," + "\n"
       + f"{nxs['Pj_max']:.3f} at {nxs['L0_Pj_max']:.3f}", fontsize=7, color=st.INK2, ha="right", va="top",
       linespacing=1.2)
h = [b.plot([], [], color=st.ORANGE, lw=1.5, label=r"$\sigma_E=0.104$")[0],
     b.plot([], [], color=st.ORANGE, lw=0.8, label=r"$\sigma_E=0.094$")[0],
     b.plot([], [], color=st.INK2, lw=1.0, label=r"$P_{\rm joint}$")[0],
     b.plot([], [], color=st.INK2, lw=1.0, ls=(0, (4, 2)), label=r"$P_{\rm cond}$")[0]]
b.legend(handles=h, loc="center right", bbox_to_anchor=(1.0, 0.52), fontsize=7, handlelength=1.8,
         borderaxespad=0.3, labelspacing=0.25)
b.set_xlim(0.975, 1.30)
b.set_ylim(0, 1.03)
b.set_xlabel(r"torus side $L_0$ ($d_{\rm LSS}$)")
b.set_ylabel("probability")
st.panel_label(b, "(b)", x=0.03, y=0.96)

print(f"edges {e_lo:.4f}-{e_hi:.4f}; peaks {mid['Pj_max']:.4f}@{mid['L0_Pj_max']} / {nxs['Pj_max']:.4f}@"
      f"{nxs['L0_Pj_max']}; windows {w_lo}-{w_hi} / {wn_lo}-{wn_hi}; measures {meas:.4f}/{meas_nx:.4f}; "
      f"u range {u.min():.4f}-{u.max():.4f}, D range {D.min():.3g}-{D.max():.3g}")
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_fossil_sky", width=st.DOUBLE, papers=("prd",)))
