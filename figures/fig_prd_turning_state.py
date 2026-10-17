# PROVENANCE: copied from 06_paper_draft/figures/v11/fig4_turning_point_state.py on 2026-10-15 by fig-shape, then
# rewritten for v12: Standard Model with the zero-mode share f0 = 0.30 as the benchmark (free fields as the variant),
# leading turning-point solution throughout, new panel (d) G(|t_x|) with the survival thresholds t_*, style_v12.
"""PRD figure prd_turning_state (Sec. VIII A-B, label fig:s-turning) - one conclusion: a shape state that is real at
the turning point leaves the turn at the quantum floor, and reality at any other moment t_x costs at least the factor
G(|t_x|) in outgoing variance, which the survival of the shape limits to h|t_x| < 0.24 (99%) and 0.36 (90%).

(a) 1-sigma Wigner ellipses of the Standard Model aspect mode (f0 = 0.30, kappa_E = 19.32) in its natural turning-point
    state at ht = 0, +-1, +-6, in units of the outgoing widths; past (dashed) and future (solid) are mirror images.
(b) rms width of the aspect (orange) and angle (blue) modes vs ht for the natural state: f0 = 0.30 (solid) and free
    fields (dashed); the widths freeze within |ht| ~ 2 at the fossil values.
(c) outgoing variance in units of the floor |T12| delta vs the turning-point width s/s_*: the curve (x + 1/x)/2 with
    the natural widths (filled: f0 = 0.30, open: free fields).
(d) G(|t_x|) at leading order (kappa_E = 19.32) and next order (kappa_E = 18.0), the single-branch bound G_+(t_x) for
    t_x > 0, the saturation 2*Gain, the onset of freezing, and the two-branch survival thresholds t_*(99%) = 0.237 and
    t_*(90%) = 0.355 (h^-1, N = 2048).

All curves are computed here on the leading solution with the stiffnesses, widths and delta read from ./data/shape;
the script first checks that it reproduces the archived transfer matrices and G tables (C153) and the natural/optimal
width ratio (C137c). Data: C153_summary.json, C153_run02_summary.json, C137c_results.json, C76_summary.json,
C63_results.json.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

import style_v12 as st
import _helpers_shape as hs

st.use()
C153 = st.load_json("shape/C153_summary.json")
R02 = st.load_json("shape/C153_run02_summary.json")
C137 = st.load_json("shape/C137c_results.json")
C76 = st.load_json("shape/C76_summary.json")
C63 = st.load_json("shape/C63_results.json")

hs.selftest()
GT = C153["B_next_order"]["G_tables"]
K_E, K_N = GT["E_xi030"]["kappa"], GT["E_next18.0"]["kappa"]
K_T2 = GT["T2_xi030"]["kappa"]
DELTA = C153["B_floor_unification"]["floor_conventions"]["X_V12directive_exact(L_b=722.508,H_eff L_b=0.00437243)"]["delta"]
T_FREEZE = GT["E_xi030"]["freeze_start"]

# natural widths in units of delta (convention free); f0 = 0.30 from C76, free fields from C63
snat_E = C76["s_nat"][0] / C76["delta"]
snat_T = C76["s_nat"][2] / C76["delta"]
hs.check("kappa_E C76 vs C153", C76["kappa"][0], K_E, 1e-12)
hs.check("kappa_T2 C76 vs C153", C76["kappa"][2], K_T2, 1e-12)
mE, mT = hs.transfer(K_E, 0.0), hs.transfer(K_T2, 0.0)
sstar_E, sstar_T = abs(mE[0, 1] / mE[0, 0]), abs(mT[0, 1] / mT[0, 0])
row0 = {(r["kappa"], r["t_x"]): r for r in C137["rows"]}
hs.check("natural/optimal width (C137c)", snat_E / sstar_E, 1.0 / row0[(19.32, 0.0)]["s_opt_over_s_adiabatic"], 2e-4)
free = C63["SM_AAA"]
fE, fT = free["modes"]["aspect"], free["modes"]["angle"]
# delta for free fields in the same (H_eff, leading-solution) convention: X ~ (N_C^eff)^(3/4) at fixed H, so
# delta_free = delta_{f0} (1 - f0)^(3/4) with f0 = C76 "r" = 0.299897; this reproduces the archived free-field
# fossil 0.0587 of C63 (rolling plateau history) to 0.04% (checked below).
DELTA_FREE = DELTA * (1.0 - C76["r"]) ** 0.75


def outgoing_rms(kappa, x_nat, delta):
    m = hs.transfer(kappa, 0.0)
    floor = 2 * abs(m[0, 0] * m[0, 1]) * delta
    return np.sqrt(floor * 0.5 * (x_nat + 1 / x_nat))


def rms_path(kappa, s_over_delta, delta, ts):
    M = hs.transfer_path(kappa, 0.0, ts)
    C0 = np.diag([s_over_delta, 1.0 / s_over_delta])
    C = np.einsum("tij,jk,tlk->til", M, C0, M)
    return np.sqrt(C[:, 0, 0] * delta)


fig = plt.figure(figsize=(st.DOUBLE, 4.55))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.95], wspace=0.30)
bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.95], wspace=0.30)
a, b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
c, d = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])

# (a) Wigner ellipses of the f0 = 0.30 aspect mode -------------------------------------------------------------------
# in units of the outgoing widths; at ht = +-6 the state is frozen (needles along the diagonals, clipped by the frame)
T_ELL = (0.0, 0.5, 6.0)
ts = np.array([*T_ELL, hs.T_INF])
M = hs.transfer_path(K_E, 0.0, ts)
C0 = np.diag([snat_E, 1.0 / snat_E])
C = np.einsum("tij,jk,tlk->til", M, C0, M)
sb, sp = np.sqrt(C[-1, 0, 0]), np.sqrt(C[-1, 1, 1])
P = np.diag([1.0, -1.0])
greys = {0.0: st.INK, 0.5: st.INK2, 6.0: "#b4b3ab"}
for k, tt in enumerate(T_ELL):
    for sign, ls in ((+1, "-"), (-1, (0, (3, 2)))):
        if tt == 0 and sign < 0:
            continue
        Ck = C[k] if sign > 0 else P @ C[k] @ P
        Cn = np.diag([1 / sb, 1 / sp]) @ Ck @ np.diag([1 / sb, 1 / sp])
        w, v = np.linalg.eigh(Cn)
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        a.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fill=(tt == 0),
                            fc=st.GREY_FILL, ec=greys[tt], lw=1.1 if tt else 0.9, ls=ls, zorder=3 if tt else 4))
a.annotate(r"$ht=0$", (0.0, -0.02), xytext=(-0.05, -0.26), textcoords="data", fontsize=7, color=st.INK,
           ha="right", arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=1, shrinkB=1))
a.text(0.322, 0.125, r"$+0.5$", fontsize=7, color=st.INK2, ha="left", va="center")
a.text(0.322, -0.125, r"$-0.5$", fontsize=7, color=st.INK2, ha="left", va="center")
a.text(0.33, 0.395, r"$+6$", fontsize=7, color=st.MUTED, ha="right", va="center")
a.text(0.33, -0.395, r"$-6$", fontsize=7, color=st.MUTED, ha="right", va="center")
a.set_xlim(-0.45, 0.45)
a.set_ylim(-0.45, 0.45)
a.set_aspect("equal")
a.set_xticks([-0.4, 0, 0.4])
a.set_yticks([-0.4, 0, 0.4])
a.xaxis.set_major_formatter(lambda v, p: hs.fmt(v, 1) if v else "0")
a.yaxis.set_major_formatter(lambda v, p: hs.fmt(v, 1) if v else "0")
a.set_xlabel(r"$\beta/\sigma_\beta(\infty)$")
a.set_ylabel(r"$\Pi/\sigma_\Pi(\infty)$")
st.panel_label(a, "(a)", x=0.04, y=0.96)
a.texts[-1].set(bbox=dict(boxstyle="square,pad=0.1", fc=st.SURFACE, ec="none"), zorder=6)  # over the ht = -6 needle

# (b) rms widths vs time ---------------------------------------------------------------------------------------------
tpos = np.linspace(0.0, 8.0, 401)
x_free_E = fE["s_nat_over_delta"] / fE["s_star_over_delta"]
x_free_T = fT["s_nat_over_delta"] / fT["s_star_over_delta"]
curves = [  # kappa, s_nat/delta on the leading solution, delta, colour, style
    (K_E, snat_E, DELTA, st.ORANGE, "-"),
    (K_T2, snat_T, DELTA, st.BLUE, "-"),
    (fE["kappa"], x_free_E * abs(np.divide(*hs.transfer(fE["kappa"], 0.0)[0, ::-1])), DELTA_FREE, st.ORANGE,
     (0, (4, 2))),   # s_nat/delta = x * s_*/delta, s_*/delta = |m12/m11| on the leading solution
    (fT["kappa"], x_free_T * abs(np.divide(*hs.transfer(fT["kappa"], 0.0)[0, ::-1])), DELTA_FREE, st.BLUE,
     (0, (4, 2))),
]
final = []
for kap, s_d, dl, col, ls in curves:
    r = rms_path(kap, s_d, dl, tpos)
    b.plot(np.r_[-tpos[::-1], tpos[1:]], np.r_[r[::-1], r[1:]], color=col, ls=ls, lw=1.2)
    final.append(r[-1])
sigE, sigT, sigE_free, sigT_free = final
hs.check("sigma_E(f0=0.30) vs floor 0.1037", sigE / np.sqrt(1 + 4.85e-4), C153["B_next_order"]["next_order_floor"]["xi030_kappa19.3237"]["sigma_floor_exact"], 2e-4)
hs.check("sigma_E(free) vs C63 natural_rms", sigE_free, fE["natural_rms"], 2e-3)
hs.check("sigma_T2(f0=0.30) vs C76", sigT, C76["sigma_lin_analytic"]["angle"], 5e-3)
b.set_yscale("log")
b.set_xlim(-8, 8)
b.set_ylim(2e-3, 0.25)
b.set_xlabel(r"$ht$")
b.set_ylabel(r"rms shape $\Delta\beta$ per mode")
b.axvline(0, color=st.HAIR, lw=0.6)
b.text(7.7, sigE * 1.13, rf"aspect, $f_0=0.30$: {sigE:.3f}", ha="right", va="bottom", fontsize=7, color=st.INK2)
b.text(7.7, sigE_free * 0.88, rf"aspect, free fields: {sigE_free:.3f}", ha="right", va="top", fontsize=7,
       color=st.INK2)
b.text(7.7, sigT * 0.86, rf"angles: {sigT:.4f} ($f_0=0.30$), {sigT_free:.4f}", ha="right", va="top", fontsize=7,
       color=st.INK2)
h1 = b.plot([], [], color=st.INK2, ls="-", lw=1.2, label=r"SM, $f_0=0.30$")[0]
h2 = b.plot([], [], color=st.INK2, ls=(0, (4, 2)), lw=1.2, label="SM, free fields")[0]
b.legend(handles=[h1, h2], loc="center left", bbox_to_anchor=(0.02, 0.47), fontsize=7.5, handlelength=2.2,
         borderaxespad=0.3)
hs.plain_log(b, "y")
st.panel_label(b, "(b)", x=0.02, y=0.96)

# (c) outgoing variance vs turning-point width -----------------------------------------------------------------------
x = np.geomspace(0.03, 30, 400)
c.plot(x, 0.5 * (x + 1 / x), color=st.INK2, lw=1.0)
c.axhline(1, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
marks = [  # x = s_nat/s_*, colour, filled (open = free fields, drawn first so the filled f0 = 0.30 marks stay on top)
    (snat_E / sstar_E, st.ORANGE, True), (snat_T / sstar_T, st.BLUE, True),
    (x_free_E, st.ORANGE, False), (x_free_T, st.BLUE, False),
]
for xn, col, filled in sorted(marks, key=lambda m: m[2]):
    c.plot([xn], [0.5 * (xn + 1 / xn)], "s", ms=4.2 if filled else 6.0, mfc=col if filled else st.SURFACE,
           mec=col, mew=1.0, zorder=6 if filled else 5)
ex_E = 100 * (0.5 * (marks[0][0] + 1 / marks[0][0]) - 1)
ex_F = 100 * (fE["natural_over_floor_var"] - 1)
hs.check("excess f0=0.30 vs C137c", 1 + ex_E / 100, row0[(19.32, 0.0)]["adiabatic_over_floor"], 1e-6)
c.annotate(f"aspect: +{ex_E:.2f}%\n" + rf"($f_0=0.30$), +{ex_F:.2f}%" + "\n(free fields)", (1.04, 1.05),
           xytext=(1.5, 5.2), textcoords="data", fontsize=7, color=st.INK2, ha="center", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=10, shrinkB=3))
c.text(0.6, 1.62, "angle", fontsize=7, color=st.INK2, ha="center", va="bottom")
c.text(0.06, 12.5, "angle, free fields", fontsize=7, color=st.INK2, ha="left", va="bottom")
c.set_xscale("log")
c.set_yscale("log")
c.set_xlim(0.03, 30)
c.set_ylim(0.7, 30)
hs.plain_log(c, "x")
hs.plain_log(c, "y")
c.set_xlabel(r"turning-point width $s/s_*$")
c.set_ylabel(r"${\rm Var}\,\beta_+/(|T_{12}|\delta)$")
st.panel_label(c, "(c)", x=0.04, y=0.96)

# (d) G(|t_x|) ---------------------------------------------------------------------------------------------------------
# kappa_E = 19.32 as in D109/C137c and the text (the production nonlinear runs use 19.3237; G differs by < 0.07%)
rows_lo = sorted((r for r in C137["rows"] if r["kappa"] == 19.32 and r["t_x"] < 0), key=lambda r: -r["t_x"])
K_D = rows_lo[0]["kappa"]
for r in rows_lo:                                    # the curve must reproduce the archived C137c values
    hs.check(f"G({-r['t_x']})", hs.G_two_branch(K_D, -r["t_x"])[0], r["opt_over_floor"], 1e-5)
for r in GT["E_next18.0"]["rows"][:16]:
    hs.check(f"G_next({r['tau']:.3f})", hs.G_two_branch(K_N, r["tau"])[0], r["G_two_branch"], 1e-5)
XMAX = 2.5
tau = np.linspace(0.0, XMAX, 251)
G_lo = hs.G_two_branch(K_D, tau)
G_nx = hs.G_two_branch(K_N, tau)
tau_p = tau[tau <= 2.0]
G_p = hs.G_plus_single(K_D, tau_p)
fl = C153["B_next_order"]["floor_closed_form"]
Ginf_lo, Ginf_nx = fl["19.32"]["G_inf_closed_2calG"], fl["18.0"]["G_inf_closed_2calG"]
t_fr = 0.5 * np.arcsinh(np.sqrt(K_D))               # onset of freezing, h|t| = arsinh(sqrt(kappa))/2
G_fr = hs.G_two_branch(K_D, t_fr)[0]
# survival thresholds (two-branch, N = 2048): 99% from run02; 90% interpolated linearly between the run02 node
# h|t_x| = 0.35 (N = 2048) and the run01 node 0.40 (N = 512), as in D125 run02 table A3 note a
t99 = R02["lo_t"]["t_star_linear_interp"]["0.99"]["central"]
n35 = next(r for r in R02["lo_t"]["rows"] if abs(r["tau"] - 0.35) < 1e-9)["A"]["p"]
n40 = next(g for g in C153["A1_conditional_theorem"]["grid"] if abs(g["tau"] - 0.40) < 1e-9)["two_branch"]
t90 = 0.35 + (n35 - 0.9) / (n35 - n40) * 0.05
d.axvspan(0.0, t90, color=st.AQUA_FILL, lw=0, zorder=0)
for tv in (t99, t90):
    d.axvline(tv, color=st.AQUA, lw=0.8, zorder=1)
d.text(t99 - 0.025, 2.0e-3, f"{t99:.3f}", rotation=90, fontsize=7, color=st.INK2, ha="right", va="bottom")
d.text(t90 + 0.025, 2.0e-3, f"{t90:.3f}", rotation=90, fontsize=7, color=st.INK2, ha="left", va="bottom")
d.text(0.5 * t90, 150, "two-branch\nsurvival\n" + r"$\geq90\%$", fontsize=7, color=st.INK2, ha="center",
       va="center", linespacing=1.1, zorder=2,
       bbox=dict(boxstyle="square,pad=0.1", fc=st.AQUA_FILL, ec="none"))  # interrupts the t99 line behind the text
d.axvline(t_fr, color=st.MUTED, lw=0.6, ls=(0, (3, 2)))
d.text(t_fr + 0.03, 2.0e-3, "freezing", rotation=90, fontsize=7, color=st.MUTED, ha="left", va="bottom")
d.axhline(Ginf_lo, color=st.ORANGE, lw=0.6, ls=(0, (1, 1.5)))
d.axhline(Ginf_nx, color=st.ORANGE, lw=0.6, ls=(0, (1, 1.5)))
d.axhline(1.0, color=st.HAIR, lw=0.6)
d.plot(tau, G_lo, color=st.ORANGE, lw=1.3, zorder=4)
d.plot(tau, G_nx, color=st.ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=4)
d.plot(tau_p, G_p, color=st.ORANGE, lw=1.0, ls=(0, (1, 1.2)), zorder=4)
quoted = [r for r in rows_lo if -r["t_x"] in (0.1, 0.25, 0.5, 1.0)]
d.plot([-r["t_x"] for r in quoted] + [t_fr], [r["opt_over_floor"] for r in quoted] + [G_fr], "o", ms=3.6,
       color=st.ORANGE, mec=st.SURFACE, mew=0.6, zorder=5)
d.text(XMAX - 0.04, Ginf_lo * 1.15, rf"$2\mathcal{{G}}={Ginf_lo:.0f}$", ha="right", va="bottom", fontsize=7,
       color=st.INK2)
d.text(XMAX - 0.04, Ginf_nx * 0.66, rf"next order: {Ginf_nx:.0f}", ha="right", va="top", fontsize=7, color=st.INK2)
d.annotate(r"$G(|t_x|)$, $\kappa_E=$" + f"{K_D:.2f}", (0.66, float(hs.G_two_branch(K_D, 0.66)[0])),
           xytext=(0.60, 3.0), fontsize=7, color=st.INK2, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=2, shrinkB=2))
d.annotate(r"$\kappa_E=$" + f"{K_N:.1f}", (1.55, float(hs.G_two_branch(K_N, 1.55)[0])), xytext=(1.75, 60),
           fontsize=7, color=st.INK2, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=2, shrinkB=2))
d.annotate(r"single branch $G_+(t_x)$, $t_x>0$", (0.85, float(hs.G_plus_single(K_D, 0.85)[0])),
           xytext=(1.20, 0.08), fontsize=7, color=st.INK2, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=2, shrinkB=2))
d.set_yscale("log")
d.set_xlim(0.0, XMAX)
d.set_ylim(1.5e-3, 2e4)
hs.plain_log(d, "y")
d.set_xlabel(r"moment of reality $h|t_x|$")
d.set_ylabel("outgoing variance / floor")
st.panel_label(d, "(d)", x=0.015, y=0.96)

print(f"t99 = {t99:.4f}, t90 = {t90:.4f}, G(t99) = {hs.G_two_branch(K_D, t99)[0]:.2f}, "
      f"G(t90) = {hs.G_two_branch(K_D, t90)[0]:.2f}, freeze {t_fr:.4f} G = {G_fr:.2f}; sigma_E = {sigE:.5f}, "
      f"sigma_E free = {sigE_free:.5f}, sigma_T2 = {sigT:.5f}/{sigT_free:.5f}; 2Gain = {Ginf_lo:.2f}/{Ginf_nx:.2f}")
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_turning_state", width=st.DOUBLE, papers=("prd",)))
