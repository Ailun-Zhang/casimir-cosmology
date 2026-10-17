"""PRL Fig. 1 / PRD (two-column version 'prd_threshold'): the finite-volume threshold.

(a) Effective potential of the Higgs zero mode along r = |phi| in the reduced two-dimensional (r, s) model
    (D112 2D_ex: v11 couplings, MC+CW one-loop matching), for lambda = -0.010 and -0.013: the field-space potential
    U(r) = E_zp(r) + lambda r^4/4 + dV_1(r) of the four-dimensional radial problem, the zero-point energy E_zp of the
    gauge zero modes (the wall), the bare quartic (same origin), the Siegert resonance energy eps with its turning
    points and the full width Gamma (data: generated/potential_2Dex_lam010_013.csv, fulldim_dmc_run01/T14a_profile.json).
(b) SM running lambda(mu) from m_t to M-bar_P for m_t = 172.2/172.6/173.0 GeV (alpha_s = 0.1180; band alpha_s =
    0.1180 +- 0.0009 at m_t = 172.6), with the hazard-equivalent threshold band lambda_c(mu_b) for P1 = 0.9 ... 0.1 at
    mu_b = M-bar_P/722 and the SM point (error bar: SM-mapping uncertainty of the pole formulation)
    (data: generated/lambda_running.csv, generated/constants.json, holonomy_cross_run05/C149_run05_summary.json,
    claims_run05/table1_error_budget_run05.csv).

usage: python -B fig_prl_fig1_threshold.py [prl|prd]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

LAMS = [(-0.010, "m0p010", "-0.01", S.RAMP[1]), (-0.013, "m0p013", "-0.013", S.RAMP[3])]


def _crossings(r, U, e):
    """inner (r1) and outer (r2) classical turning points of the level e (U - e changes sign)."""
    s = np.sign(U - e)
    idx = np.where(np.diff(s) != 0)[0]
    xs = [r[i] + (e - U[i]) * (r[i + 1] - r[i]) / (U[i + 1] - U[i]) for i in idx]
    return xs[0], xs[1]


def panel_potential(ax, wide=False):
    rows = H.C("generated/potential_2Dex_lam010_013.csv")
    prof = H.J("fulldim_dmc_run01/T14a_profile.json")
    r = H.col(rows, "r")
    zp = H.col(rows, "zeropoint_m0p010")
    assert np.allclose(zp, H.col(rows, "zeropoint_m0p013")), "zero-point energy must not depend on lambda"
    ylo, yhi = 2.2, 5.65
    ax.plot(r, zp, color=S.INK2, ls=(0, (1, 1.6)), lw=1.0, zorder=2)
    ax.text(2.5, float(np.interp(2.5, r, zp)) + 0.1, "gauge zero-point\nenergy", color=S.INK2, fontsize=7.5,
            ha="right", va="bottom", linespacing=0.95)
    out = {}
    for k, (lam, tag, key, c) in enumerate(LAMS):
        Ug = H.col(rows, f"Ug_{tag}")
        q = H.col(rows, f"quartic_{tag}")
        ax.plot(r, Ug[0] + q, color=c, lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.plot(r, Ug, color=c, lw=1.5, zorder=3)
        eps, gam = prof[key]["Q_2D_ex_siegert"], prof[key]["gamma_2D_ex"]
        r1, r2 = _crossings(r, Ug, eps)
        out[key] = dict(eps=eps, gam=gam, r1=r1, r2=r2)
    # the two resonance energies differ by 0.009/L (not resolved here): one level, solid in the well, dotted under the
    # barrier; outer turning points of the two potentials marked on it
    e10, e13 = out["-0.01"], out["-0.013"]
    eps = 0.5 * (e10["eps"] + e13["eps"])
    ax.plot([0, e10["r1"]], [eps, eps], color=S.AQUA, lw=1.4, zorder=4, solid_capstyle="butt")
    ax.plot([e10["r1"], e10["r2"]], [eps, eps], color=S.AQUA, lw=1.0, ls=(0, (1.2, 1.2)), zorder=4)
    for key, (lam, tag, kk, c) in zip(("-0.01", "-0.013"), LAMS):
        ax.plot([out[key]["r2"]], [eps], "o", ms=3.2, mfc=c, mec="white", mew=0.5, zorder=6)
    ax.annotate("", xy=(e10["r2"] + 0.95, eps), xytext=(e10["r2"] + 0.12, eps),
                arrowprops=dict(arrowstyle="-|>", color=S.AQUA, lw=1.0, mutation_scale=7), zorder=5)
    ax.text(0.15, eps + 0.05, r"$\varepsilon$", color=S.AQUA, fontsize=8.5, ha="left", va="bottom")
    # curve labels with the widths
    lab = {"-0.01": (5.9, 4.72, "left"), "-0.013": (5.98, 2.82, "right")}
    for lam, tag, key, c in LAMS:
        x, y, ha = lab[key]
        ax.text(x, y, H.minus(rf"$\lambda={lam:.3f}$") + "\n" + rf"$\Gamma L={H.sci(out[key]['gam'])}$", color=c,
                fontsize=7.5, ha=ha, va="center", linespacing=1.1)
    ax.text(0.2, 2.55, r"bare $\lambda r^4/4$", color=S.INK2, fontsize=7.5, ha="left", va="top")
    ax.set_xlim(0, 8.6)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"Higgs zero mode $rL$")
    ax.set_ylabel(r"$U(r)\,L$")
    return out


def panel_running(ax, wide=False):
    d = H.C("generated/lambda_running.csv")
    k = H.J("generated/constants.json")
    mu = H.col(d, "mu_GeV")
    lam = {key: H.col(d, key) for key in d[0] if key.startswith("lam_")}
    mub, mp = k["MU_B_GeV"], k["MPBAR_GeV"]
    lc = H.lam_c()
    # SM-mapping error (pole formulation) converted to lambda(mu_b)
    t1 = {r["source"]: r for r in H.C("claims_run05/table1_error_budget_run05.csv")}
    s_sm = float(t1["SM TOTAL pole formulation (4-loop truncated)"]["P0.5_plus"])
    dlam = abs(k["dlam_mub_dmt_172p6"]) * s_sm
    styles = {"172.2": (0, (4, 2)), "172.6": "-", "173.0": (0, (5, 1.5, 1, 1.5))}
    lam_sm = k["lam_mu_b"]["172.6_0.1180"]
    zx0, zy0, zy1 = 1.0e13, -0.0178, -0.0046   # zoom window (inset)

    def curves(a, lw_main):
        a.axhline(0.0, color=S.HAIR, lw=0.6, zorder=1)
        a.fill_between(mu, lam["lam_mt172.6_as0.1171"], lam["lam_mt172.6_as0.1189"], color=S.GREY_FILL, lw=0, zorder=1)
        for mt, ls in styles.items():
            a.plot(mu, lam[f"lam_mt{mt}_as0.1180"], color=S.INK, lw=lw_main if mt == "172.6" else 0.85, ls=ls, zorder=3)

    curves(ax, 1.3)
    ax.add_patch(Rectangle((zx0, zy0), mp - zx0, zy1 - zy0, fill=False, edgecolor=S.MUTED, lw=0.6, zorder=4))
    ax.set_xscale("log")
    ax.set_xlim(150.0, mp)
    ax.set_ylim(-0.025, 0.145)
    ax.set_xlabel(r"$\mu$ (GeV)")
    ax.set_ylabel(r"$\lambda(\mu)$")
    ax.set_xticks([1e2, 1e6, 1e10, 1e14, 1e18])
    ax.minorticks_off()
    top = ax.secondary_xaxis("top")
    top.set_xticks([mub, mp])
    top.set_xticklabels([r"$\mu_b$", r"$\bar M_P$"])
    top.tick_params(direction="in", length=3, pad=1.5)
    ax.tick_params(axis="x", which="both", top=False)

    # inset: the threshold band at mu_b = M-bar_P/722
    ins = ax.inset_axes([0.40, 0.29, 0.585, 0.56])
    curves(ins, 1.2)
    w = 2.2
    x0, x1 = mub / w, mub * w
    ins.add_patch(Rectangle((x0, lc["0.1"]), x1 - x0, lc["0.9"] - lc["0.1"], facecolor=S.AQUA_FILL, edgecolor=S.AQUA,
                            lw=0.8, zorder=2, alpha=0.9))
    wbox = dict(boxstyle="square,pad=0.08", fc=S.SURFACE, ec="none", alpha=0.85)
    for p, v in lc.items():
        ins.plot([x0, x1], [v, v], color=S.AQUA, lw=0.9, zorder=2)
        ins.text(x0 / 1.25, v, (r"$P_1=$" if p == "0.1" else "") + p, color=S.AQUA, fontsize=7.5, ha="right",
                 va="center", bbox=wbox, zorder=7)
    ins.text(x1 * 1.25, lc["0.1"] + 0.0003, r"$\lambda_c(\mu_b)$", color=S.AQUA, fontsize=7.5, ha="left", va="center")
    ins.errorbar([mub], [lam_sm], yerr=[dlam], fmt="o", ms=3.0, color=S.INK, mfc=S.INK, elinewidth=1.0, capsize=1.5,
                 zorder=6)
    xl = 1.2e16
    for mt in styles:
        y = float(np.interp(np.log(xl), np.log(mu), lam[f"lam_mt{mt}_as0.1180"]))
        ins.text(xl, y + 0.0003, (r"$m_t=$" if mt == "172.2" else "") + mt + (" GeV" if mt == "172.2" else ""),
                 color=S.INK, fontsize=7.5, ha="left", va="bottom", zorder=7)
    ins.set_xscale("log")
    ins.set_xlim(zx0, mp)
    ins.set_ylim(zy0, zy1)
    ins.minorticks_off()
    ins.set_xticks([1e14, 1e16, 1e18])
    ins.set_yticks([-0.015, -0.010, -0.005])
    ins.tick_params(labelsize=7.5, pad=1.5, length=2.2)
    ins.set_facecolor(S.SURFACE)
    return dict(mub=mub, dlam=dlam, lam_sm=lam_sm)


def make(layout: str = "prl"):
    S.use()
    if layout == "prl":
        fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.25), gridspec_kw=dict(hspace=0.36))
        name, width, papers = "prl_fig1_threshold", S.SINGLE, ("prl",)
    else:
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(wspace=0.26))
        name, width, papers = "prd_threshold", S.DOUBLE, ("prd",)
    info_a = panel_potential(a, wide=layout != "prl")
    info_b = panel_running(b, wide=layout != "prl")
    S.panel_label(a, "(a)", x=0.915, y=0.96)
    S.panel_label(b, "(b)", x=0.915, y=0.96)
    paths, page = H.save(fig, name, width, papers=papers)
    print(name, f"page width {page:.3f} in", [str(p) for p in paths], info_a, info_b)
    return paths


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "prl")
