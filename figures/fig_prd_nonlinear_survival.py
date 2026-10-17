"""PRD figure prd_nonlinear_survival (Sec. VIII B and D, label fig:s-survival; new in v12) - one conclusion: classical
nonlinearity ends the shape once the outgoing deformation is large, so 99% of the trajectories survive only up to
s_99 ~ 2.8-2.95 times the natural amplitude, and a state real at t_x keeps the shape on both branches with 99% (90%)
probability only for h|t_x| < 0.237 (0.355); the A_s-calibrated plateau history changes no fate.

(a) Survival vs the amplitude factor s of the natural turning-point state (f0 = 0.30): production runs C142 (N = 256)
    and C153 run02 (N = 2048), independent implementation R16/T19 (N = 2048), the same draws on the plateau history
    (C160, fates identical), next order (kappa_E = 18.0; C153), free fields (f0 = 0; C142 N = 128 and the 2048 v11
    draws at s = 1). Top axis: outgoing linear aspect amplitude sigma_lin = s * sigma_lin(1). Band: s_99 from run02
    (log-linear interpolation) to T19 (quantile of the critical amplification); tick: s_50 (T19).
(b) Two-branch survival S2 of the optimal real Gaussian vs the moment of reality h|t_x|: C153 run01 (N = 256-512) and
    run02 (N = 2048), the blind check R16/V4 (N = 2048), states real at t_x < 0 on the plateau history (C160), and the
    never-real mirror pairs (C153, 1.2 and 2 times the minimum width) at the right; thresholds t_* for 99%, 90%
    (interpolated as in D125 run02 note a) and 50%.
Error bars: Wilson 95% intervals as archived. Data: C142_summary.json, C153_summary.json, C153_run02_summary.json,
D114_own.json (T19), C160_summary.json, BLIND_RESULTS.json (V4), fossil_draws_R5_hi.csv.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

import style_v12 as st
import _helpers_shape as hs

st.use()
C142 = st.load_json("shape/C142_summary.json")
C153 = st.load_json("shape/C153_summary.json")
R02 = st.load_json("shape/C153_run02_summary.json")
T19 = st.load_json("shape/D114_own.json")
C160 = st.load_json("shape/C160_summary.json")
V4 = st.load_json("shape/BLIND_RESULTS.json")
HI = st.load_csv("shape/fossil_draws_R5_hi.csv")

fig, (a, b) = plt.subplots(2, 1, figsize=(st.SINGLE, 4.6), gridspec_kw={"hspace": 0.42})
EB = dict(fmt="none", elinewidth=0.6, capsize=0, zorder=3)

# (a) survival vs amplitude factor s --------------------------------------------------------------------------------
sets = C142["sets"]
s_pr = np.array([1, 2, 3, 4, 5, 7, 10, 20, 40])
pr = [sets[f"amp_s{s}"] for s in s_pr]
p_pr = np.array([r["survival"] for r in pr])
hs.check("C142 amp_threshold vs sets", float(np.sum(p_pr)), float(np.sum(C142["amp_threshold"]["survival"])), 1e-12)
lo_pr, hi_pr = np.array([r["survival_wilson95"] for r in pr]).T
a.plot(s_pr, p_pr, color=st.ORANGE, lw=1.1, zorder=2)
hs.errbar(a, s_pr, p_pr, lo_pr, hi_pr, ecolor=st.ORANGE, **EB)
a.plot(s_pr, p_pr, "o", ms=3.4, mfc=st.SURFACE, mec=st.ORANGE, mew=0.9, zorder=4)
r2 = R02["lo_amp"]["rows"]
s_r2 = np.array([r["s"] for r in r2])
p_r2 = np.array([r["p"] for r in r2])
lo_r2, hi_r2 = np.array([r["wilson95"] for r in r2]).T
hs.errbar(a, s_r2, p_r2, lo_r2, hi_r2, ecolor=st.ORANGE, **EB)
a.plot(s_r2, p_r2, "s", ms=3.4, color=st.ORANGE, mec=st.ORANGE, zorder=5)
tl = T19["turn_lead"]
s_t = np.array([1, 3, 5, 10, 20, 40])
p_t = np.array([tl[f"s={s}"]["frac"] for s in s_t])
lo_t, hi_t = np.array([tl[f"s={s}"]["wilson95"] for s in s_t]).T
hs.errbar(a, s_t, p_t, lo_t, hi_t, ecolor=st.INK, **EB)
a.plot(s_t, p_t, "D", ms=3.0, color=st.INK, mec=st.INK, zorder=5)
pl = C160["plateau"]
s_pl = np.array([1, 3, 5, 10])
for s in s_pl:
    assert pl[f"pl_amp_s{s}"]["n_flips"] == 0
a.plot(s_pl, [pl[f"pl_amp_s{s}"]["frac"] for s in s_pl], "+", ms=7, mew=1.0, color=st.INK2, zorder=6)
nx = C153["B_next_order"]["NL"]
s_nx = [1, 3, 5, 10]
nxa = {s: nx[f"nx_amp_s{s}"]["survival"] for s in s_nx}
for r in R02["nx_amp"]["rows"]:                       # N = 2048 replaces N = 256 where both exist (s = 3)
    nxa[r["s"]] = r["p"]
s_n = np.array(sorted(nxa))
a.plot(s_n, [nxa[s] for s in s_n], color=st.ORANGE, lw=0.9, ls=(0, (4, 2)), zorder=2)
fr = {1: sum(int(r["survivor"]) for r in HI) / len(HI)}
for s in (3, 5, 10):
    fr[s] = sets[f"var_r0_amp_s{s}"]["survival"]
s_f = np.array(sorted(fr))
a.plot(s_f, [fr[s] for s in s_f], color=st.MUTED, lw=0.9, ls=(0, (1, 1.3)), marker="o", ms=2.6, zorder=2)
s99_lo = R02["lo_amp"]["s99_loglinear"]["central"]
s99_hi = tl["s99_surv"]
s50 = tl["s50_surv"]
a.axvspan(s99_lo, s99_hi, color=st.AQUA_FILL, lw=0, zorder=0)
a.axhline(0.99, color=st.MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
a.axhline(0.5, color=st.MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
a.plot([s50, s50], [0.0, 0.5], color=st.MUTED, lw=0.6, zorder=1)
a.text(s99_hi * 1.04, 0.62, rf"$s_{{99}}\simeq{s99_lo:.1f}$" + "–" + f"{s99_hi:.2f}", fontsize=7,
       color=st.INK2, ha="left", va="center")
a.text(s50 * 1.05, 0.06, rf"$s_{{50}}={s50:.1f}$", fontsize=7, color=st.INK2, ha="left", va="bottom")
a.text(45, 0.975, "99%", fontsize=7, color=st.MUTED, ha="right", va="top")
a.text(45, 0.475, "50%", fontsize=7, color=st.MUTED, ha="right", va="top")
a.annotate("free fields", (10, fr[10]), xytext=(14, 0.84), fontsize=7, color=st.INK2, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=1, shrinkB=2))
s_mid = 8.0                                            # point on the next-order curve (log-linear between 5 and 10)
y_mid = nxa[5] + (nxa[10] - nxa[5]) * np.log(s_mid / 5) / np.log(2)
a.annotate("next order", (s_mid, y_mid), xytext=(12.5, 0.67), fontsize=7, color=st.INK2, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=1, shrinkB=2))
h = [a.plot([], [], "o", ms=3.4, mfc=st.SURFACE, mec=st.ORANGE, mew=0.9, label="256 draws")[0],
     a.plot([], [], "s", ms=3.4, color=st.ORANGE, label="2048 draws")[0],
     a.plot([], [], "D", ms=3.0, color=st.INK, label="independent\nimplementation")[0],
     a.plot([], [], "+", ms=7, mew=1.0, color=st.INK2, label="plateau background")[0]]
a.legend(handles=h, loc="lower left", fontsize=7, handlelength=1.0, handletextpad=0.4, borderaxespad=0.3,
         labelspacing=0.25, frameon=True, facecolor=st.SURFACE, edgecolor="none", framealpha=1.0, borderpad=0.2)
a.set_xscale("log")
a.set_xlim(0.9, 48)
a.set_ylim(-0.02, 1.04)
hs.plain_log(a, "x")
a.set_xlabel(r"amplitude factor $s$ of the natural state")
a.set_ylabel("surviving fraction")
sig1 = sets["amp_s1"]["rms_lin_all"]["aspect"]
top = a.secondary_xaxis("top", functions=(lambda s: s * sig1, lambda x: x / sig1))
top.set_xticks([0.2, 0.5, 1, 2, 4])
top.xaxis.set_major_formatter(lambda v, p: f"{v:g}")
top.xaxis.set_minor_formatter(lambda v, p: "")
top.set_xlabel(r"outgoing linear amplitude $\sigma_{\rm lin}$", labelpad=3)
st.panel_label(a, "(a)", x=0.03, y=0.82)

# (b) two-branch survival vs moment of reality ---------------------------------------------------------------------
grid = C153["A1_conditional_theorem"]["grid"]
t1 = np.array([g["tau"] for g in grid])
p1 = np.array([g["two_branch"] for g in grid])
lo1, hi1 = np.array([g["two_branch_wilson95"] for g in grid]).T
hs.errbar(b, t1, p1, lo1, hi1, ecolor=st.ORANGE, **EB)
b.plot(t1, p1, "o", ms=3.4, mfc=st.SURFACE, mec=st.ORANGE, mew=0.9, zorder=4)
rr = R02["lo_t"]["rows"]
assert R02["lo_t"]["B_failures_pooled"]["fail"] == 0      # mirror branch never fails: S2 = S_A to 4e-4
t2 = np.array([r["tau"] for r in rr])
p2 = np.array([r["A"]["p"] for r in rr])
lo2, hi2 = np.array([r["A"]["wilson95"] for r in rr]).T
hs.errbar(b, t2, p2, lo2, hi2, ecolor=st.ORANGE, **EB)
b.plot(t2, p2, "s", ms=3.4, color=st.ORANGE, zorder=5)
cv = V4["item2_nonlinear_survival"]["curves_fwd"]["N2048"]
tv = -np.array(cv["tx"])
keep = tv >= 0
o = np.argsort(tv[keep])
b.plot(tv[keep][o], np.array(cv["S"])[keep][o], color=st.INK, lw=0.8, zorder=3)
t_pl = {"pl_r1_tx-0.5": 0.5, "pl_r1_tx-1.0": 1.0, "pl_r1_txF": 0.5 * np.arcsinh(np.sqrt(C153["B_next_order"]
                                                                                     ["G_tables"]["E_xi030"]["kappa"]))}
for k, x in t_pl.items():
    assert pl[k]["n_flips"] == 0
    b.plot([x], [pl[k]["frac"]], "+", ms=7, mew=1.0, color=st.INK2, zorder=6)
# thresholds
t99 = R02["lo_t"]["t_star_linear_interp"]["0.99"]["central"]
n35 = next(r for r in rr if abs(r["tau"] - 0.35) < 1e-9)["A"]["p"]
n40 = next(g for g in grid if abs(g["tau"] - 0.40) < 1e-9)["two_branch"]
t90 = 0.35 + (n35 - 0.9) / (n35 - n40) * 0.05
t50 = C153["A1_conditional_theorem"]["t_star_two_branch"]["0.5"]["central"]
for tv_, lev in ((t99, 0.99), (t90, 0.9), (t50, 0.5)):
    b.plot([tv_, tv_], [-0.02, lev], color=st.AQUA, lw=0.8, zorder=1)
    b.plot([0, tv_], [lev, lev], color=st.AQUA, lw=0.6, ls=(0, (3, 2)), zorder=1)
b.text(t99 - 0.012, 0.05, f"{t99:.3f}", rotation=90, fontsize=7, color=st.INK2, ha="right", va="bottom")
b.text(t90 + 0.012, 0.05, f"{t90:.3f}", rotation=90, fontsize=7, color=st.INK2, ha="left", va="bottom")
b.text(t50 + 0.012, 0.05, f"{t50:.3f}", rotation=90, fontsize=7, color=st.INK2, ha="left", va="bottom")
# never-real mirror pairs (the |t_x| -> infinity limit), drawn at a separate abscissa
XNR = 1.27
for k, r in enumerate(C153["A2_never_real"]["never_real_pair_NL"]):
    tb = r["two_branch"]
    x = XNR + (k - 0.5) * 0.035
    hs.errbar(b, [x], [tb["p"]], [tb["wilson95"][0]], [tb["wilson95"][1]], ecolor=st.INK, **EB)
    b.plot([x], [tb["p"]], "v", ms=3.6, color=st.INK, zorder=5)
b.text(XNR - 0.015, 0.13, "never\nreal", fontsize=7, color=st.INK2, ha="center", va="bottom", linespacing=1.0)
b.axvline(1.17, color=st.HAIR, lw=0.6, ls=(0, (2, 2)))
b.set_xlim(0.0, 1.33)
b.set_ylim(-0.02, 1.04)
b.set_xticks([0, 0.25, 0.5, 0.75, 1.0, XNR])
b.set_xticklabels(["0", "0.25", "0.5", "0.75", "1", r"$\infty$"])
b.set_xlabel(r"moment of reality $h|t_x|$")
b.set_ylabel(r"two-branch survival $S_2$")
h = [b.plot([], [], "o", ms=3.4, mfc=st.SURFACE, mec=st.ORANGE, mew=0.9, label="256–512 draws")[0],
     b.plot([], [], "s", ms=3.4, color=st.ORANGE, label="2048 draws")[0],
     b.plot([], [], color=st.INK, lw=0.8, label="independent\nimplementation")[0],
     b.plot([], [], "+", ms=7, mew=1.0, color=st.INK2, label="plateau background")[0]]
b.legend(handles=h, loc="upper right", bbox_to_anchor=(0.86, 1.0), fontsize=7, handlelength=1.3,
         handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
st.panel_label(b, "(b)", x=0.03, y=0.13)

print(f"s99 band {s99_lo:.3f}-{s99_hi:.3f}, s50 {s50:.2f}, sigma_lin(1) {sig1:.4f}; t99 {t99:.4f} t90 {t90:.4f} "
      f"t50 {t50:.4f}; free s=1 {fr[1]:.4f} ({len(HI)} draws)")
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_nonlinear_survival", width=st.SINGLE, papers=("prd",)))
