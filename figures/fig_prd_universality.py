"""PRD Sec. V (universality of the threshold): WKB scaling of lambda_c with the turning duration and its coupling slopes.

(a) |lambda_c|^{-1/2} vs ln eta, eta = f eta_inf the conformal duration of the turning (f = 1/4 ... 4), for P1 = 0.9,
    0.5, 0.1: markers = exact hazard thresholds, lines = WKB prediction from the f = 1 point
    (data: universality_run01/C156_eta.json eta_scaling, wkb_vs_exact_duration, eta_inf; run02 centre).
(b) Logarithmic derivatives d ln|lambda_c| / d ln g for g = g2, gY, yt at mu_b (couplings varied by +-10%), bars for
    the three levels, diamonds = WKB estimate (data: universality_run02/C156_run02.json derivatives; run02 centre).
"""
from __future__ import annotations

import math
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

LEVELS = [("0.9", S.RAMP[1]), ("0.5", S.RAMP[2]), ("0.1", S.RAMP[3])]


def panel_eta(ax):
    d = H.J("universality_run01/C156_eta.json")
    eta0 = d["eta_inf"]
    fs = sorted(d["eta_scaling"], key=float)
    out = {}
    for p, col in LEVELS:
        f = np.array([float(k) for k in fs])
        x = np.log(f * eta0)
        y = np.array([abs(d["eta_scaling"][k][f"P1={p}"]["lam_c"]) ** -0.5 for k in fs])
        w = d["wkb_vs_exact_duration"][f"P1={p}"]
        yw = np.array([abs(w[k]["wkb"]) ** -0.5 for k in fs])
        ax.plot(x, yw, color=col, lw=1.0, zorder=2)
        ax.plot(x, y, "o", ms=3.4, mfc=S.SURFACE, mec=col, mew=1.1, zorder=3)
        ax.text(x[-1] + 0.08, y[-1], rf"$P_1={p}$" if p == "0.9" else p, color=col, fontsize=7.5, ha="left",
                va="center")
        slope = np.polyfit(x, y, 1)[0]
        out[p] = dict(slope_fit=float(slope), max_dev_wkb=float(np.max(np.abs(y - yw))))
    ax.axvline(math.log(eta0), color=S.INK2, lw=0.6, ls=(0, (1, 1.4)), zorder=1)
    ax.text(math.log(eta0) + 0.05, 7.15, rf"$\eta_\infty={eta0:.0f}$", color=S.INK2, fontsize=7.5, ha="left",
            va="bottom")
    ax.set_xlabel(r"$\ln\eta$")
    ax.set_ylabel(r"$|\lambda_c|^{-1/2}$")
    ax.set_xlim(4.15, 7.65)
    top = ax.secondary_xaxis("top", functions=(lambda x: np.exp(x) / eta0, lambda f: np.log(np.maximum(f, 1e-9) * eta0)))
    top.set_xscale("log")
    top.set_xticks([0.25, 0.5, 1, 2, 4])
    top.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    top.minorticks_off()
    top.set_xlabel(r"duration factor $f$", labelpad=2)
    top.tick_params(direction="in", length=3, pad=1.5)
    ax.tick_params(axis="x", which="both", top=False)
    return out


def panel_coup(ax):
    d = H.J("universality_run02/C156_run02.json")["derivatives"]
    gs = [("g2", r"$g_2$"), ("gY", r"$g_Y$"), ("yt", r"$y_t$")]
    wbar = 0.24
    out = {}
    for i, (g, lab) in enumerate(gs):
        for k, (p, col) in enumerate(LEVELS):
            v = d[g][f"P1={p}"]["dlnlamc_dlng"]
            vw = d[g][f"P1={p}"]["dlnlamc_dlng_wkb"]
            x = i + (k - 1) * wbar
            ax.bar([x], [v], width=wbar * 0.92, color=col, zorder=2, label=(rf"$P_1={p}$" if i == 0 else None))
            ax.plot([x], [vw], "D", ms=3.0, mfc=S.SURFACE, mec=S.INK, mew=0.9, zorder=4)
            out[f"{g}|{p}"] = (round(v, 4), round(vw, 4))
        v5 = d[g]["P1=0.5"]["dlnlamc_dlng"]
        ax.text(i, max(d[g][f"P1={p}"]["dlnlamc_dlng"] for p, _ in LEVELS) + 0.04, f"{v5:.2f}", fontsize=7.5,
                ha="center", va="bottom", color=S.INK)
    ax.plot([], [], "D", ms=3.0, mfc=S.SURFACE, mec=S.INK, mew=0.9, label="WKB")
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([lab for _, lab in gs])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel(r"$\partial\ln|\lambda_c|/\partial\ln g$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=2, handlelength=1.2,
              columnspacing=0.8, borderaxespad=0.3, labelspacing=0.25)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    return out


def make():
    S.use()
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 4.3), gridspec_kw=dict(hspace=0.48, height_ratios=[1.15, 1]))
    ia = panel_eta(a)
    ib = panel_coup(b)
    S.panel_label(a, "(a)", x=0.025, y=0.965)
    S.panel_label(b, "(b)", x=0.025, y=0.965)
    paths, page = H.save(fig, "prd_universality", S.SINGLE, papers=("prd",))
    print("prd_universality", f"page {page:.3f} in", ia, ib)
    return paths


if __name__ == "__main__":
    make()
