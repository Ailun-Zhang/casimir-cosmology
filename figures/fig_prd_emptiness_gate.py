# PROVENANCE: copied from 06_paper_draft/figures/v11/fig3_emptiness_gate.py on 2026-10-15 by fig-shape; updated to the
# v12 benchmark (zero-mode share f0 = 0.30: R includes the zero-mode energy, budget of "about five" quanta, h L_b =
# 0.0044) and restyled with style_v12; the N_C = 100 toy is no longer shown.
"""PRD figure prd_emptiness_gate (Sec. VI, label fig:s-gate) - one conclusion: a Casimir turning point must be almost
empty, so its state has to be prepared at, or within about 7 e-folds of, the turning point; a reheating coupling in a
time-symmetric past overfills it by more than 10^94.

(a) ratio R of the comoving energy of visible photons produced by the reheating coupling (c_v = 0.01) during the
    time-symmetric remote contraction to the comoving Casimir energy that supports the turn, for a photon vacuum
    declared at the j-th extremum of the inflaton oscillation before the final climb (C55: 97 and 401 modes), including
    the zero-mode energy, log10 R + log10[1/(1 - f0)] (C153: +0.155 at f0 = 0.30). R >= 1 forbids the turn.
(b) excitation sum_n |n| N_n per non-conformal light degree of freedom at the turn for a natural state declared T
    e-folds before it (C56, SM AAA), on the f0 = 0.30 history (h L_b = 0.0044): the de Sitter scaling of the mode
    labels shifts the C56 curves (h L_b = 0.00483) by ln(0.00483/0.0044) in T (checked against C153). Band: the two
    superhorizon state choices (instantaneous ground state, conformal vacuum). Horizontal lines: the budget
    0.2666 N_C^eff shared by three non-conformal degrees of freedom at f0 = 0.30 (solid) and for free fields (dotted);
    vertical line: T = ln(2 pi/(h L_b)), when the whole torus is inside the Hubble radius.
Data: C55_window_gate.json, C55_window_gate_refined.json, C56_SM_AAA.json, C153_summary.json.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

import style_v12 as st
import _helpers_shape as hs

st.use()
coarse = st.load_json("shape/C55_window_gate.json")
refined = st.load_json("shape/C55_window_gate_refined.json")
C56 = st.load_json("shape/C56_SM_AAA.json")
EMP = st.load_json("shape/C153_summary.json")["C_emptiness"]

fig, (a, b) = plt.subplots(1, 2, figsize=(st.DOUBLE, 2.45), gridspec_kw={"wspace": 0.3})

# (a) remote window photons, including the zero-mode energy -------------------------------------------------------------
rh = EMP["reheating_excess"]["xi0.30"]
shift = rh["shift_log10"]
jc = np.array([s["turning_point_index_before_climb"] for s in coarse["slices"]])
lc = np.array([s["log10_R"] for s in coarse["slices"]]) + shift
jr = np.array([s["turning_point_index_before_climb"] for s in refined["slices"]])
lr = np.array([s["log10_R"] for s in refined["slices"]]) + shift
for j, v in zip(jr, lr):                                  # reproduces C153 reheating_excess (refined slices)
    hs.check(f"log10R(j={j})", v, rh["log10R"][str(int(j))], 1e-9)
a.plot(jc, lc, "-", color=st.INK2, lw=1.0, zorder=2)
a.plot(jc, lc, "o", ms=4.5, color=st.INK, mec=st.SURFACE, mew=0.6, zorder=3, label="97 photon modes")
a.plot(jr, lr, "D", ms=5, mfc=st.SURFACE, mec=st.ORANGE, mew=1.1, zorder=4, label="401 photon modes")
a.axhline(0, color=st.INK, lw=0.8)
a.fill_between([0.7, 120], -12, 0, color=st.GREY_FILL2, lw=0)
a.text(1.0, -5.5, r"turning point allowed: $R<1$", fontsize=7, color=st.INK2, va="center")
for j, v in zip(jr, lr):
    a.annotate(f"{v:.1f}", (j, v), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
               fontsize=7.5, color=st.INK2)
a.set_xscale("log")
a.set_xlim(0.7, 120)
a.set_ylim(-12, 122)
hs.plain_log(a, "x")
a.set_xlabel(r"inflaton oscillation extremum $j$ before the final climb")
a.set_ylabel(r"$\log_{10}R,\ R=\rho_\gamma/|\rho_C|$")
a.text(0.97, 0.50, "reheating coupling active\nduring remote contraction\n" + r"($c_v=0.01$, time symmetric," + "\n"
       + rf"$f_0=0.30$)", transform=a.transAxes, ha="right", va="center", fontsize=7, color=st.INK2)
a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=7, handletextpad=0.3)
st.panel_label(a, "(a)", x=0.03, y=0.96)

# (b) preparation-time gate on the f0 = 0.30 history ---------------------------------------------------------------------
pw = EMP["preparation_window"]["hLb0.0044"]
hLb = 0.0044
dT = np.log(C56["hLb"] / hLb)
sc = EMP["scaling_check"][0]
hs.check("de Sitter shift of T", sc["T_new"] - sc["T_old"], dT, 1e-6)
T = np.array([r["T"] for r in C56["rows"]]) + dT
g = np.array([r["total_ground"] for r in C56["rows"]])
cf = np.array([r["total_conformal"] for r in C56["rows"]])
lo, hi = np.minimum(g, cf), np.maximum(g, cf)
qb = EMP["quanta_budget"]
bud30 = qb["0.299897 (v11 DMC, lambda=+0.01)"]["budget_sum_n"] / 3.0
budfree = qb["0 (free fields)"]["budget_sum_n"] / 3.0
hs.check("free budget C153 vs C56", 3 * budfree, C56["budget_sum_n_quanta"], 1e-6)
# crossing of the budget: the archived value (C153, from the exact de Sitter scaling); log-linear interpolation of the
# eight C56 nodes agrees to 0.03 e-fold, which only checks consistency
T_cross = pw["budget_crossing_3dof"]["ground_xi0.30"]
hs.check("crossing f0=0.30 (coarse interpolation)", np.interp(np.log(bud30), np.log(g), T), T_cross, 1e-2)
T_max = pw["T_max_ln(2pi/hLb)"]
hs.check("T_max", T_max, np.log(2 * np.pi / hLb), 1e-9)
floor = 1e-8
keep = hi >= floor
b.fill_between(T[keep], np.clip(lo[keep], floor, None), hi[keep], color=st.BLUE, alpha=0.18, lw=0)
b.plot(T[keep], hi[keep], color=st.BLUE, lw=1.2)
b.plot(T[keep], hi[keep], "o", ms=3, color=st.BLUE, mec=st.SURFACE, mew=0.4)
b.axhline(bud30, color=st.INK2, lw=0.8)
b.axhline(budfree, color=st.INK2, lw=0.8, ls=(0, (1, 1.5)))
b.axvline(T_max, color=st.MUTED, lw=0.7)
XL = 2.0 + 0.03 * 6.2   # left edge of the in-panel labels = that of "(b)" and the SM label (axes fraction 0.03)
b.text(XL, min(bud30, budfree) * 0.6, rf"budget per degree of freedom, $f_0=0.30$: {bud30:.2f}", fontsize=7,
       color=st.INK2, va="top")
b.text(XL, max(bud30, budfree) * 1.5, f"free fields: {budfree:.2f}", fontsize=7, color=st.INK2, va="bottom")
b.annotate(rf"$T={T_cross:.2f}$", (T_cross, bud30), xytext=(5.15, 1.5e-3), fontsize=7, color=st.INK2,
           ha="center", va="top", arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=1, shrinkB=2))
b.text(T_max - 0.06, 3e-6, rf"$\ln(2\pi/h L_b)={T_max:.2f}$", rotation=90, fontsize=7, color=st.MUTED, ha="right",
       va="bottom")
b.text(XL, 2.5e-8, r"$<10^{-8}$ for $T\leq4$", fontsize=7, color=st.INK2, va="bottom")
b.set_yscale("log")
b.set_ylim(1e-8, 1e4)
b.set_xlim(2, 8.2)
b.set_xlabel(r"preparation time $T$ before the turning point (e-folds)")
b.set_ylabel(r"$\sum_{n}|n|\,N_{n}$ per degree of freedom")
b.text(0.03, 0.45, r"SM, $f_0=0.30$, $hL_b=0.0044$", transform=b.transAxes, fontsize=7, color=st.INK2,
       va="top")
st.panel_label(b, "(b)", x=0.03, y=0.96)

print(f"shift {shift:.4f}; log10R refined {lr}; dT {dT:.5f}; budget {bud30:.4f}/{budfree:.4f}; crossing {T_cross:.4f}"
      f" (C153 {pw['budget_crossing_3dof']['ground_xi0.30']:.4f}); T_max {T_max:.4f}")
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_emptiness_gate", width=st.DOUBLE, papers=("prd",)))
