"""PRD (appendix on reduced dynamics and full-dimensional validation): resonance widths and their validation.

(a) Full width Gamma L of the zero-mode resonance vs |lambda|^{-1/2} (WKB: ln Gamma linear) for the reduced models of
    D112 (v11 couplings, MC+CW matching): 1D Gaussian (1D_G), 1D with the DMC gauge shape (1D_ex), 2D (r, s) Gaussian
    (2D_G) and 2D with the DMC shape (2D_ex); horizontal lines: static thresholds Gamma_c = -ln P / eta_inf for
    P = 0.9 and 0.5; ticks: the static lambda_c of each model (data: higgs_beyond_bo_run01/C140_summary.json
    gamma.*, trend_lam_to_0.*, lam_c.*, eta).
(b) Full-dimensional check of the 2D_ex width at lambda = -0.010 and -0.013: Gamma_full/Gamma_2D from four estimators
    (energy shift of the full-dimensional DMC converted with d ln Gamma/dQ; WKB along the full adiabatic channel; DMC
    tail amplitude; 3D collective-coordinate model with two zero-point prescriptions); band: +-20% acceptance
    (data: fulldim_dmc_run01/C151_summary.json verdict.gamma_ratios, lam_neg.*).
"""
from __future__ import annotations

import math
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

MODELS = [("1D_G", "1D Gaussian", S.RAMP[0], 1.0, "o"), ("1D_ex", "1D + DMC shape", S.RAMP[1], 1.0, "s"),
          ("2D_G", "2D Gaussian", S.RAMP[2], 1.0, "^"), ("2D_ex", "2D + DMC shape", S.RAMP[3], 1.6, "D")]


def panel_gamma(ax):
    c = H.J("higgs_beyond_bo_run01/C140_summary.json")
    eta = c["eta"]
    trend = c["trend_lam_to_0"]
    tkey = {"1D_G": "g_1DG", "1D_ex": "g_1Dex", "2D_ex": "g_2D"}
    out = {}
    for key, lab, col, lw, mk in MODELS:
        lam = [float(k) for k in c["gamma"][key]]
        g = [c["gamma"][key][k] for k in c["gamma"][key]]
        if key in tkey:
            for lk, v in trend.items():
                if float(lk) not in lam:
                    lam.append(float(lk))
                    g.append(v[tkey[key]])
        o = np.argsort(np.abs(lam))
        x = np.abs(np.array(lam)[o]) ** -0.5
        y = np.array(g)[o]
        ax.plot(x, y, color=col, lw=lw, marker=mk, ms=2.8, mfc=col, mec=col, zorder=3, label=lab)
        out[key] = dict(n=len(x))
    ax.legend(loc="lower left", fontsize=7.5, handlelength=1.6, borderaxespad=0.4, labelspacing=0.25)
    gc = {p: -math.log(p) / eta for p in (0.9, 0.5)}
    for p, gv in gc.items():
        ax.axhline(gv, color=S.AQUA, lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.text(15.5, gv * 1.12, rf"$\Gamma_c\ (P_1={p:g})$", color=S.AQUA, fontsize=7.5, ha="right", va="bottom")
        for key, lab, col, lw, mk in MODELS:
            lc = c["lam_c"][key][f"P1={p:g}"]
            ax.plot([abs(lc) ** -0.5] * 2, [gv / 1.35, gv * 1.35], color=col, lw=1.2, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(8.1, 15.6)
    ax.set_ylim(5e-7, 2e-2)
    ax.set_xlabel(r"$|\lambda|^{-1/2}$")
    ax.set_ylabel(r"$\Gamma L$")
    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    lt = [-0.013, -0.010, -0.008, -0.006, -0.005]
    top.set_xticks([abs(v) ** -0.5 for v in lt])
    top.set_xticklabels([H.minus(f"{v:.3f}") for v in lt], fontsize=7.5)
    top.set_xlabel(r"$\lambda(\mu_b)$", labelpad=2)
    top.tick_params(direction="in", length=3, pad=1.5)
    ax.tick_params(axis="x", which="both", top=False)
    out["Gamma_c"] = gc
    out["lam_c_2Dex"] = c["lam_c"]["2D_ex"]
    return out


def panel_fulldim(ax):
    c = H.J("fulldim_dmc_run01/C151_summary.json")
    v = c["verdict"]["gamma_ratios"]
    rows = [("conv", "DMC energy shift"), ("wkb_full_adiabatic", "WKB, full adiabatic"), ("tail_A", "DMC tail amplitude"),
            ("3D_zeta_ref", "3D collective (ref.)"), ("3D_zeta_local", "3D collective (local)")]
    lams = [("-0.01", "o", S.RAMP[1], -0.13), ("-0.013", "s", S.RAMP[3], 0.13)]
    ax.axvspan(0.8, 1.2, color=S.GREY_FILL2, lw=0, zorder=0)
    ax.axvline(1.0, color=S.INK2, lw=0.7, zorder=1)
    err = {}
    for i, (key, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        for k, (lk, mk, col, dy) in enumerate(lams):
            r = v[key][k]
            e = None
            ln = c["lam_neg"][lk]
            if key == "conv":
                e = r * ln["conversion"]["dlngamma_dQ"] * ln["se_dQ_ex_main"]
            elif key == "tail_A":
                e = ln["tail_fit"]["se_A"]
            err[f"{key}|{lk}"] = (r, e)
            ax.errorbar([r], [y + dy], xerr=None if e is None else [[e], [e]], fmt=mk, ms=3.6, color=col, mfc=col,
                        mec=col, elinewidth=0.9, capsize=1.5, zorder=3)
        ax.text(0.765, y + 0.2, lab, fontsize=7.5, color=S.INK, ha="left", va="bottom")
    for k, (lk, mk, col, dy) in enumerate(lams):
        ax.plot([], [], mk, color=col, ms=3.6, label=H.minus(rf"$\lambda={float(lk):.3f}$"))
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.3, borderaxespad=0.3, frameon=True,
              framealpha=0.9, edgecolor="none")
    # Delta m_c of the error budget: the width ratios converted with the final-model hazard slopes (deduplicated
    # Table I rebuild); the envelope must equal the budget row 'fulldim_T14a'
    t = H.J("claims_run05/table1_rebuild.json")["t14a"]
    dm, de = t["central"], t["envelope"]
    row = {r["source"]: r for r in H.C("claims_run05/table1_error_budget_run05.csv")}["fulldim_T14a"]
    for p in ("0.9", "0.5", "0.1"):
        assert abs(float(row[f"P{p}_plus"]) - de[p]) < 6e-5, (p, row[f"P{p}_plus"], de[p])

    def f3(x):
        return f"{round(x + 1e-12, 3):.3f}"
    ax.text(1.235, len(rows) - 0.55, r"$\Delta m_c$" + f" = +{f3(dm['0.9'])} / +{f3(dm['0.5'])} GeV\n" +
            f"(envelope +{f3(de['0.9'])} / +{f3(de['0.5'])};\n" + r"$P_1=0.9$ / 0.5)", fontsize=7.5, color=S.INK,
            ha="right", va="top", linespacing=1.15)
    ax.set_xlim(0.76, 1.24)
    ax.set_ylim(-0.6, len(rows) + 0.25)
    ax.set_yticks([])
    ax.set_xlabel(r"$\Gamma_{\rm full}/\Gamma_{\rm 2D}$")
    dq = {lk: (c["lam_neg"][lk]["dQ_ex_main"], c["lam_neg"][lk]["se_dQ_ex_main"]) for lk in ("-0.01", "-0.013")}
    return dict(ratios=err, dm_c_central=dm, dm_c_envelope=de, dQ=dq)


def make():
    S.use()
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.6), gridspec_kw=dict(hspace=0.42, height_ratios=[1.25, 1]))
    ia = panel_gamma(a)
    ib = panel_fulldim(b)
    S.panel_label(a, "(a)", x=0.925, y=0.965)
    S.panel_label(b, "(b)", x=0.925, y=0.965)
    paths, page = H.save(fig, "prd_gamma_validation", S.SINGLE, papers=("prd",))
    print("prd_gamma_validation", f"page {page:.3f} in", ia, ib)
    return paths


if __name__ == "__main__":
    make()
