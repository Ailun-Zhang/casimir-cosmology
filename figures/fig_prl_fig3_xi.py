"""PRL Fig. 3 (also used in the PRD; panel (c) of 'prd_criterion'): the (m_t, xi_H(mu_b)) plane.

* zero-mode threshold xi_c(m_t): P_surv(xi_c) = 0.99 P_surv(1/6) (D111/D129 definition, mu_b reading). Curve = universal
  function xi_c(P1) (helpers.XiUniversal, fitted to all 26 directly computed points) evaluated on the run05 P1(m_t);
  markers = the run05 direct values with the prescription half-difference (claims_run05/xi_run05_summary.json);
  bands = the same function evaluated at m_t -+ sigma_SM (pole SM mapping, Table I) and at alpha_s = 0.1171/0.1189;
* all-mode inflation bound xi_inf(m_t) at mu_b (inflation_stability_run01/xi_inf_table.csv, alpha_s = 0.1180;
  PRD version adds alpha_s = 0.1171/0.1189);
* conformal value 1/6; xi = 3/8 (no tachyonic resonance above) and the upper bound xi <~ 1 (Herranen et al. 2015);
* allowed region shaded where xi_c is computed; top axis: P1 at alpha_s = 0.1180 (run05).

usage: python -B fig_prl_fig3_xi.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

XMIN, XMAX = 171.75, 173.6   # left edge = start of the directly computed xi_c (P1 = 0.9996)
YMIN, YMAX = 0.008, 1.7


def xi_inf_curves():
    rows = H.C("inflation_stability_run01/xi_inf_table.csv")
    out = {}
    for a in ("0.1171", "0.118", "0.1189"):
        sel = [r for r in rows if abs(float(r["als"]) - float(a)) < 1e-9]
        m = np.array([float(r["mt"]) for r in sel])
        y = np.array([float(r["xi_inf_mub"]) for r in sel])
        o = np.argsort(m)
        out[a] = PchipInterpolator(m[o], y[o])
        out[a].mrange = (m.min(), m.max())
    return out


def draw_xi(ax, wide=False, top_axis=True):
    g = H.P1Grid()
    X = H.XiUniversal()
    m = np.round(np.arange(XMIN, XMAX + 1e-9, 0.005), 3)
    P = g.at(m)
    xc = X(P)
    t1 = {r["source"]: r for r in H.C("claims_run05/table1_error_budget_run05.csv")}
    s_sm = float(t1["SM TOTAL pole formulation (4-loop truncated)"]["P0.5_plus"])
    sm_lo, sm_hi = X(g.at(m - s_sm)), X(g.at(m + s_sm))
    as_lo, as_hi = X(np.interp(m, g.m, g.curve(0.1189))), X(np.interp(m, g.m, g.curve(0.1171)))
    inf = xi_inf_curves()
    yinf = inf["0.118"](m)
    ok = np.isfinite(xc)
    # short continuation from the last direct point (P1 = 0.134) to P1 = 0.1 (straight line in ln(-ln P1))
    xe = X.extrapolated(P, 0.1)
    oke = np.isfinite(xe)
    m01 = g.mcrit(0.1)
    # allowed region (P1 >= 0.1): above max(xi_c, xi_inf), below xi ~ 1
    xall = np.where(ok, xc, xe)
    oka = np.isfinite(xall) & (m <= m01 + 1e-9)
    lower = np.maximum(xall, yinf)
    ax.fill_between(m[oka], lower[oka], 1.0, color=S.AQUA_FILL, lw=0, zorder=0)
    ax.plot([m01, m01], [float(np.interp(m01, m[oka], lower[oka])), 1.0], color=S.AQUA, lw=0.6, zorder=3)
    ax.text(m01 - 0.03, 0.72, r"$P_1=0.1$", fontsize=7.5, color=S.INK2, ha="right", va="center")
    xcross = float(m[ok][np.argmax(xc[ok] > yinf[ok])])
    # alpha_s band (light) and SM-mapping band (dark): filled where both edges are covered by direct points;
    # the alpha_s edges are drawn over their full range
    okb = np.isfinite(as_lo) & np.isfinite(as_hi)
    ax.fill_between(m[okb], as_lo[okb], as_hi[okb], color=S.GREY_FILL, alpha=0.75, lw=0, zorder=1)
    for e in (as_lo, as_hi):
        ax.plot(m, e, color=S.MUTED, lw=0.5, zorder=1)
    oks = np.isfinite(sm_lo) & np.isfinite(sm_hi)
    ax.fill_between(m[oks], sm_lo[oks], sm_hi[oks], color=S.MUTED, alpha=0.40, lw=0, zorder=2)
    ax.plot(m[ok], xc[ok], color=S.AQUA, lw=1.6, zorder=4)
    ax.plot(m[oke], xe[oke], color=S.AQUA, lw=1.2, ls=(0, (2, 1.2)), zorder=4)
    # inflation bound
    mi = m[(m >= inf["0.118"].mrange[0]) & (m <= inf["0.118"].mrange[1])]
    ax.plot(mi, inf["0.118"](mi), color=S.INK, lw=1.1, ls=(0, (4, 2)), zorder=4)
    if wide:
        for a in ("0.1171", "0.1189"):
            ax.plot(mi, inf[a](mi), color=S.INK, lw=0.6, ls=(0, (4, 2)), zorder=4)
    # run05 direct points
    xr = H.xi_run05()
    pts = [v for k, v in xr.items() if k != "172.96"]
    ax.errorbar([v["m_t"] for v in pts], [v["xi_c"] for v in pts], yerr=[v["half_diff"] for v in pts], fmt="o",
                ms=3.0, mfc=S.SURFACE, mec=S.AQUA, ecolor=S.AQUA, mew=1.0, elinewidth=0.9, capsize=1.4, zorder=6)
    # reference values
    ax.axhline(1.0, color=S.INK, lw=0.8, zorder=3)
    ax.fill_between([XMIN, XMAX], 1.0, YMAX, facecolor="none", edgecolor=S.HATCH_RED, hatch="////", lw=0, zorder=1)
    ax.axhline(1 / 6, color=S.INK2, lw=0.8, ls=(0, (1, 1.5)), zorder=3)
    ax.axhline(3 / 8, color=S.INK2, lw=0.6, ls=(0, (3, 2)), zorder=3)
    lab = dict(fontsize=7.5, color=S.INK2, ha="left", va="bottom")
    ax.text(XMIN + 0.04, 1 / 6 * 1.05, r"$1/6$ (conformal)", **lab)
    ax.text(XMIN + 0.04, 3 / 8 * 1.05, r"$3/8$", **lab)
    ax.text(XMIN + 0.04, 0.93, r"$\xi_H\lesssim1$", fontsize=7.5, color=S.INK, ha="left", va="top",
            bbox=dict(boxstyle="square,pad=0.1", fc=S.SURFACE, ec="none"), zorder=6)
    # curve labels
    i = np.argmin(np.abs(m - 173.08))
    ax.text(m[i] + 0.03, xc[i] * 0.64, r"zero mode $\xi_c$", color=S.AQUA, fontsize=7.5, ha="left", va="top",
            fontweight="bold")
    j = np.argmin(np.abs(m - 172.9))
    ax.text(m[j], yinf[j] * 0.80, r"inflation $\xi_{\rm inf}$", color=S.INK, fontsize=7.5, ha="left", va="top")
    ax.text(172.25, 0.46, "allowed", color=S.AQUA, fontsize=8, ha="center", va="center", fontweight="bold")
    v = xr["172.60"]
    ax.annotate(rf"$\xi_c={v['xi_c']:.3f}$", xy=(v["m_t"], v["xi_c"]), xytext=(v["m_t"] - 0.62, v["xi_c"] * 1.55),
                fontsize=7.5, color=S.INK, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.6, shrinkA=0.5, shrinkB=2.5), zorder=7)
    ax.set_yscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"$m_t$ (GeV)")
    ax.set_ylabel(r"$\xi_H(\mu_b)$")
    if top_axis:
        levels = [0.99, 0.9, 0.5, 0.1]
        top = ax.secondary_xaxis("top")
        top.set_xticks([g.mcrit(p) for p in levels])
        top.set_xticklabels([f"{p:g}" for p in levels])
        top.tick_params(direction="in", length=3, pad=1.5)
        top.set_xlabel(r"$P_1$", labelpad=2)
        ax.tick_params(axis="x", which="both", top=False)
    chk = {}
    for k, vv in xr.items():
        chk[k] = round(float(X(vv["P1_2D_run05"])) - vv["xi_c"], 5)
    # lower boundary of the shaded region = max(xi_c, xi_inf): verify at a few masses
    lb = {f"{mm:.2f}": dict(xi_c=round(float(np.interp(mm, m[oka], xall[oka])), 4),
                            xi_inf=round(float(inf['0.118'](mm)), 4),
                            shade_lower=round(float(np.interp(mm, m[oka], lower[oka])), 4))
          for mm in (171.8, 171.9, 172.0, 172.1, 172.2, 172.6, 173.0, 173.4)}
    return dict(fit_max_resid=float(np.max(np.abs(X.resid))), fit_n=X.n, univ_minus_direct=chk, s_sm=s_sm,
                m_range_direct=(float(m[ok].min()), float(m[ok].max())), m_P1_0p1=m01,
                extrap_xi_at_P1_0p1=float(X.extrapolated(np.array([0.1]))[0]), xi_c_equals_xi_inf_at=xcross,
                lower_boundary_check=lb)


def make():
    S.use()
    fig, ax = plt.subplots(figsize=(3.4, 2.62))
    info = draw_xi(ax)
    paths, page = H.save(fig, "prl_fig3_xi", S.SINGLE, papers=("prl", "prd"))
    print("prl_fig3_xi", f"page width {page:.3f} in", info)
    return paths


if __name__ == "__main__":
    make()
