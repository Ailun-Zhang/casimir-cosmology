"""PRL Fig. 2 / PRD 'prd_criterion' (two-column, with the (m_t, xi_H) plane as panel c): the survival criterion.

(a) P1(m_t) at alpha_s = 0.1171/0.1180/0.1189 (run05 claim-layer grid; 0.1171/0.1189 interpolated in ln(-ln P1)
    between the neighbouring grid rows), theory 1 sigma (model + SM mapping + M_h) on the central curve at
    P1 = 0.9/0.5/0.1, the direct measurement (band), sigma(tt) pole mass and ATLAS+CMS (bars), FCC-ee (inner bar:
    m_t measurement, pole equivalent; outer bar: total including alpha_s, SM mapping, M_h and model), top axis
    m-bar_t(m-bar_t) at alpha_s = 0.1180.
(b) (m_t, alpha_s) plane: critical lines P1 = 0.9/0.5/0.1 with theory 1 sigma, lambda(mu_b) = 0, the current
    68/95% ellipses (direct m_t x alpha_s = 0.1180 +- 0.0009) and the FCC-ee 68/95% ellipses (centred at the present
    values; covariance from sigma_meas, s_as and the m-bar -> pole slope kappa of the claim configuration).

usage: python -B fig_prl_fig2_criterion.py [prl|prd]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

ALPHAS = [("0.1171", S.RAMP[1], 1.0), ("0.1180", S.RAMP[2], 1.6), ("0.1189", S.RAMP[3], 1.0)]
LEVELS = [("0.9", S.RAMP[1]), ("0.5", S.RAMP[2]), ("0.1", S.RAMP[3])]


def contours():
    rows = H.C("claims_run05/fig2a_contours_mt_alphas.csv")
    a = H.col(rows, "alpha_s")
    return a, {p: (H.col(rows, f"mc_P{p}"), H.col(rows, f"mc_P{p}_th_lo1sig"), H.col(rows, f"mc_P{p}_th_hi1sig"))
               for p in ("0.9", "0.5", "0.1")}


def panel_p1(ax, wide=False):
    g = H.P1Grid()
    meas = H.measurements()
    fut = H.future("FCCee_base")
    fnow = H.future("now_direct")
    f, finv, dev = H.mbar_map()
    xlo, xhi = 171.35, 174.05
    d = meas["D1_direct_pole"]
    ax.axvspan(d["c"] - d["sl"], d["c"] + d["sr"], color=S.GREY_FILL, lw=0, zorder=0)
    ax.text(d["c"], 1.09, "direct", fontsize=7.5, color=S.INK2, ha="center", va="center")
    ax.axhline(1.0, color=S.HAIR, lw=0.5, zorder=0)
    for a, c, lw in ALPHAS:
        ax.plot(g.m, g.curve(float(a)), color=c, lw=lw, zorder=3, label=a)
    # theory 1 sigma on the central curve
    aa, cont = contours()
    i0 = int(np.argmin(np.abs(aa - 0.1180)))
    for p in ("0.9", "0.5", "0.1"):
        mc, lo, hi = cont[p][0][i0], cont[p][1][i0], cont[p][2][i0]
        ax.errorbar([mc], [float(p)], xerr=[[mc - lo], [hi - mc]], fmt="none", ecolor=S.INK, elinewidth=0.9,
                    capsize=1.8, capthick=0.9, zorder=5)
    # label above the curves' tail (at P1 = 0.1 it would sit on the alpha_s = 0.1189 curve), tied to the bar by a leader
    ax.annotate("theory", xy=(cont["0.1"][2][i0] + 0.01, 0.1), xytext=(cont["0.1"][2][i0] + 0.06, 0.25), fontsize=7.5,
                color=S.INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.5, shrinkA=1, shrinkB=1), zorder=5)
    # other determinations and FCC-ee (bars near the bottom left, labels above the bars)
    bars = [("D2_sigma_pole", r"$\sigma_{t\bar t}$ pole", 0.42, "above"), ("C1_ATLAS_CMS", "ATLAS+CMS", 0.27, "left")]
    for key, lab, y, where in bars:
        v = meas[key]
        ax.errorbar([v["c"]], [y], xerr=[[v["sl"]], [v["sr"]]], fmt="o", ms=2.6, color=S.INK2, elinewidth=1.1,
                    capsize=1.6, zorder=4)
        if where == "above":
            ax.text(v["c"] - v["sl"], y + 0.045, lab, fontsize=7.5, color=S.INK2, ha="left", va="bottom")
        else:
            ax.text(v["c"] - v["sl"] - 0.05, y, lab, fontsize=7.5, color=S.INK2, ha="right", va="center")
    yf = 0.11
    ax.plot([d["c"] - fut["sigma_tot"], d["c"] + fut["sigma_tot"]], [yf, yf], color=S.INK, lw=0.8, zorder=4)
    ax.plot([d["c"] - fut["sigma_meas"], d["c"] + fut["sigma_meas"]], [yf, yf], color=S.INK, lw=2.6, zorder=4,
            solid_capstyle="butt")
    for s in (-1, 1):
        ax.plot([d["c"] + s * fut["sigma_tot"]] * 2, [yf - 0.022, yf + 0.022], color=S.INK, lw=0.8, zorder=4)
    ax.text(d["c"] - fut["sigma_tot"] - 0.04, yf, "FCC-ee", fontsize=7.5, color=S.INK, ha="right", va="center")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.0, 1.17)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"$m_t$ (GeV)")
    ax.set_ylabel(r"$P_1$")
    leg = ax.legend(title=r"$\alpha_s$", loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=7.5,
                    title_fontsize=7.5, handlelength=1.6, borderaxespad=0.3, labelspacing=0.25)
    leg.set_zorder(6)
    top = ax.secondary_xaxis("top", functions=(f, finv))
    top.set_xticks([162.0, 162.5, 163.0, 163.5, 164.0])
    top.set_xticklabels(["162", "162.5", "163", "163.5", "164"])
    top.tick_params(direction="in", length=3, pad=1.5)
    top.set_xlabel(r"$\bar m_t(\bar m_t)$ (GeV)", labelpad=3)
    ax.tick_params(axis="x", which="both", top=False)
    chk = {a: {p: round(g.mcrit(float(p), float(a)) - H.critical(float(a))[p], 4) for p in ("0.9", "0.5", "0.1")}
           for a, _, _ in ALPHAS}
    return dict(mbar_map_dev=dev, crit_check=chk, fcc=fut, now_sigma_tot=fnow["sigma_tot"])


def panel_plane(ax, wide=False):
    aa, cont = contours()
    m0rows = H.C("generated/m0_lambda_mub_zero.csv")
    meas = H.measurements()
    fut = H.future("FCCee_base")
    xlo, xhi = 170.75, 174.25
    for p, c in LEVELS:
        mc, lo, hi = cont[p]
        ax.fill_betweenx(aa, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
        ax.plot(mc, aa, color=c, lw=1.4, zorder=3)
    # lambda(mu_b) = 0
    am, m0 = H.col(m0rows, "alpha_s"), H.col(m0rows, "m0_pole")
    ax.plot(m0, am, color=S.INK, lw=1.0, ls=(0, (4, 2)), zorder=3)
    # current ellipses
    e = H.C("claims_run05/fig2a_ellipses.csv")
    for cl, fc, ec in (("95", S.GREY_FILL2, S.MUTED), ("68", S.GREY_FILL, S.INK2)):
        sel = [r for r in e if r["measurement"] == "D1_direct_pole" and r["CL2D"] == cl]
        ax.fill([float(r["m_t_pole4L_equiv"]) for r in sel], [float(r["alpha_s"]) for r in sel], facecolor=fc,
                edgecolor=ec, lw=0.7, zorder=2 if cl == "95" else 2.1)
    d = meas["D1_direct_pole"]
    ax.plot([d["c"]], [H.A0], "+", color=S.INK, ms=5, mew=0.9, zorder=5)
    # FCC-ee ellipses: m_pole = m(m-bar) + kappa (alpha_s - alpha_0)
    s_a, kap, sm = fut["s_as"], fut["kappa"], fut["sigma_meas"]
    cov = [[sm ** 2 + (kap * s_a) ** 2, kap * s_a ** 2], [kap * s_a ** 2, s_a ** 2]]
    for cl, ls in (("68", "-"), ("95", (0, (2, 1.2)))):
        x, y = H.ellipse(d["c"], H.A0, cov, H.K2[cl])
        ax.plot(x, y, color=S.INK, lw=0.9, ls=ls, zorder=6)
    # labels along the critical lines (rotation given in data coordinates)
    wbox = dict(boxstyle="square,pad=0.06", fc=S.SURFACE, ec="none", alpha=0.9)
    for (p, c), a_lab in zip(LEVELS, (0.1159, 0.1159, 0.1159)):
        i1 = int(np.argmin(np.abs(aa - a_lab)))
        mc = cont[p][0]
        ang = np.degrees(np.arctan2(aa[i1 + 5] - aa[i1 - 5], mc[i1 + 5] - mc[i1 - 5]))
        ax.text(mc[i1], aa[i1], (r"$P_1=$" if p == "0.9" else "") + p, color=c, fontsize=7.5, ha="center",
                va="center", rotation=ang, rotation_mode="anchor", transform_rotates_text=True, bbox=wbox, zorder=6)
    j = int(np.argmin(np.abs(am - 0.1196)))
    ax.text(m0[j] - 0.08, am[j], r"$\lambda(\mu_b)=0$", fontsize=7.5, color=S.INK, ha="right", va="center")
    ax.text(d["c"] - 0.68, 0.11885, "direct", fontsize=7.5, color=S.INK2, ha="right", va="center")
    ax.annotate("FCC-ee", xy=(d["c"] + 0.07, H.A0 - 0.00028), xytext=(d["c"] + 0.95, 0.11645), fontsize=7.5,
                color=S.INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6, shrinkA=1, shrinkB=0.5), zorder=7)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.1150, 0.1210)
    ax.set_yticks([0.115, 0.116, 0.117, 0.118, 0.119, 0.120, 0.121])
    ax.set_yticklabels(["0.115", "0.116", "0.117", "0.118", "0.119", "0.120", "0.121"])
    ax.set_xlabel(r"$m_t$ (GeV)")
    ax.set_ylabel(r"$\alpha_s(M_Z)$")
    m0_118 = float(np.interp(H.A0, am, m0))
    return dict(m0_at_0118=m0_118, m0_archived=H.J("claims_run05/C154_summary.json")["relative_form"]["m0"],
                fcc_cov=cov)


def make(layout: str = "prl"):
    S.use()
    if layout == "prl":
        fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.3), gridspec_kw=dict(hspace=0.42, height_ratios=[1, 1.08]))
        ia = panel_p1(a)
        ib = panel_plane(b)
        S.panel_label(a, "(a)", x=0.03, y=0.975)
        S.panel_label(b, "(b)", x=0.03, y=0.965)
        paths, page = H.save(fig, "prl_fig2_criterion", S.SINGLE, papers=("prl",))
        print("prl_fig2_criterion", f"page {page:.3f} in", ia, ib)
    else:
        # two panels side by side; the (m_t, xi_H) plane is the separate single-column figure prl_fig3_xi, which is
        # also written to the PRD folder (a three-panel row was too cramped at 7.0 in)
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(wspace=0.24))
        ia = panel_p1(axs[0], wide=True)
        ib = panel_plane(axs[1], wide=True)
        S.panel_label(axs[0], "(a)", x=0.03, y=0.975)
        S.panel_label(axs[1], "(b)", x=0.03, y=0.965)
        paths, page = H.save(fig, "prd_criterion", S.DOUBLE, papers=("prd",))
        print("prd_criterion", f"page {page:.3f} in", ia, ib)
    return paths


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "prl")
