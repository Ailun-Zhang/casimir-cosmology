"""Report v1.1 figure: widths and thresholds of the static box (C170 S1, final model).

(a) ln(Gamma L) against x = |lambda|^(-1/2) for boxes E, C, B, A, D (mu = mu_b/1000 ... 10 mu_b; couplings of
    m_t = 172.60 GeV, alpha_s = 0.1180, M_h = 125.13 GeV at mu = 1/L; lambda varied). Filled: narrow resonances
    (status ok/ok_broad); open: level within Gamma/2 of the barrier top or above it. Line: fit ln(Gamma L) = a - b x of
    box A over the window lambda in [-0.018, -0.007] (shaded), dashed outside it. Horizontal ticks: Gamma = ln2/N.
(b) lambda_c(N) at P = 1/2 for the five boxes (interpolated thresholds of C170_lambda_c_N_boxes.csv), the P = 0.9
    and 0.1 thresholds of box A (thin), and the closed form -[b/(a + ln(N/ln2))]^2 with the window fit of box A.

Reads only data/higgs/static_box_c170 (copies of the archived C170 results; SHA-256 in SOURCES_c170.json).  Usage: python -B fig_rep_v11_static_box.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402

import style_v12 as S  # noqa: E402

D = Path(__file__).resolve().parent / "data" / "higgs" / "static_box_c170"
PAGE_W = 15.0 / 2.54          # report: 15 cm (included at 0.94\textwidth)
BOXES = ["E", "C", "B", "A", "D"]
COL = {"E": S.RAMP[0], "C": S.RAMP[1], "B": S.RAMP[2], "A": S.AQUA, "D": S.RAMP[3]}
LAB = {"E": r"E ($\mu_b/10^3$)", "C": r"C ($\mu_b/10^2$)", "B": r"B ($\mu_b/10$)", "A": r"A ($\mu_b$)",
       "D": r"D ($10\,\mu_b$)"}
MARK = {"E": "v", "C": "s", "B": "^", "A": "o", "D": "D"}


def rcsv(name):
    with open(D / name, encoding="utf-8") as fh:
        return list(csv.DictReader([ln for ln in fh if not ln.startswith("#")]))


def panel_widths(ax, fit):
    rows = rcsv("gamma_lambda_boxes.csv")
    a, b = fit["a"], fit["b"]
    x0, x1 = 0.018 ** -0.5, 0.007 ** -0.5
    ax.axvspan(x0, x1, color=S.GREY_FILL2, lw=0, zorder=0)
    for bx in BOXES:
        r = [q for q in rows if q["box"] == bx]
        x = np.array([float(q["x_abs_lambda_m1_2"]) for q in r])
        g = np.array([float(q["Gamma"]) for q in r])
        ok = np.array([q["status"] in ("ok", "ok_broad") for q in r])
        o = np.argsort(x)
        x, g, ok = x[o], g[o], ok[o]
        lw = 1.3 if bx == "A" else 0.8
        ax.plot(x[ok], np.log(g[ok]), color=COL[bx], lw=lw, zorder=3 if bx == "A" else 2)
        ax.plot(x[ok], np.log(g[ok]), MARK[bx], ms=2.4 if bx != "A" else 2.8, mfc=COL[bx], mec="white", mew=0.3,
                zorder=4, label=LAB[bx])
        ax.plot(x[~ok], np.log(g[~ok]), MARK[bx], ms=2.4, mfc="white", mec=COL[bx], mew=0.6, zorder=4)
    xx = np.linspace(x0, x1, 50)
    ax.plot(xx, a - b * xx, color=S.INK, lw=0.9, zorder=5)
    for seg in (np.linspace(5.0, x0, 20), np.linspace(x1, 19.2, 40)):
        ax.plot(seg, a - b * seg, color=S.INK, lw=0.7, ls=(0, (3, 2)), zorder=5)
    for N, lab in ((10, r"$N=10$"), (300, r"$300$"), (1e4, r"$10^4$")):
        y = math.log(math.log(2) / N)
        ax.plot([18.55, 19.2], [y, y], color=S.INK2, lw=0.8, zorder=5)
        ax.text(18.45, y, lab, fontsize=7, color=S.INK2, ha="right", va="center")
    ax.set_xlim(5.0, 19.2)
    ax.set_ylim(-21.0, 0.2)
    ax.set_xlabel(r"$|\lambda|^{-1/2}$")
    ax.set_ylabel(r"$\ln(\Gamma L)$")
    ax.legend(loc="lower left", fontsize=7, handlelength=1.6, labelspacing=0.22, borderaxespad=0.3, ncol=1)
    ax.text(0.5 * (x0 + x1), -1.45, "fit window", fontsize=7, color=S.INK2, ha="center", va="center")
    ax.text(12.6, a - b * 12.6 + 1.05, r"$a-b|\lambda|^{-1/2}$", fontsize=7.5, color=S.INK, ha="left",
            va="bottom", rotation=0)
    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    lt = [-0.02, -0.01, -0.006, -0.004, -0.003]
    top.set_xticks([abs(v) ** -0.5 for v in lt])
    top.set_xticklabels([f"{v:g}".replace("-", "−") for v in lt], fontsize=7)
    top.tick_params(direction="in", length=2.5, pad=1.5)
    top.xaxis.set_minor_locator(mt.NullLocator())
    top.set_xlabel(r"$\lambda$", fontsize=8, labelpad=2)
    ax.tick_params(axis="x", which="both", top=False)


def panel_thresholds(ax, fit):
    rows = rcsv("lambda_c_N_boxes.csv")
    a, b = fit["a"], fit["b"]
    for bx in BOXES:
        r = [q for q in rows if q["box"] == bx and q["P"] == "0.5" and q["status"] == "interpolated"]
        N = np.array([float(q["N"]) for q in r])
        lc = np.array([float(q["lambda_c"]) for q in r])
        o = np.argsort(N)
        ax.plot(N[o], lc[o], color=COL[bx], lw=1.3 if bx == "A" else 0.8, zorder=3 if bx == "A" else 2)
        ax.plot(N[o], lc[o], MARK[bx], ms=2.4 if bx != "A" else 2.8, mfc=COL[bx], mec="white", mew=0.3, zorder=4)
    for P, ls in (("0.9", (0, (1.2, 1.2))), ("0.1", (0, (1.2, 1.2)))):
        r = [q for q in rows if q["box"] == "A" and q["P"] == P and q["status"] == "interpolated"]
        N = np.array([float(q["N"]) for q in r])
        lc = np.array([float(q["lambda_c"]) for q in r])
        o = np.argsort(N)
        ax.plot(N[o], lc[o], color=S.AQUA, lw=0.8, ls=ls, zorder=3)
    NN = np.logspace(1, 6, 200)
    ax.plot(NN, -(b / (a + np.log(NN / math.log(2)))) ** 2, color=S.INK, lw=0.8, ls=(0, (3, 2)), zorder=5)
    lam_sm = -0.009774142178602658   # box A, m_t = 172.6 (C170 S1 lam_SM)
    ax.axhline(lam_sm, color=S.INK2, lw=0.6, ls=(0, (4, 1.5, 1, 1.5)), zorder=1)
    ax.text(1.25e1, lam_sm + 0.0004, r"$\lambda_{\rm SM}(\mu_b)$", fontsize=7, color=S.INK2, ha="left", va="bottom")
    ax.text(1.3e2, -0.0212, r"$P=0.1$", fontsize=7, color=S.AQUA, ha="left", va="center")
    ax.text(1.8e4, -0.0033, r"$P=0.9$", fontsize=7, color=S.AQUA, ha="left", va="center")
    ax.text(1.4e1, -0.0315, "closed form (dashed)", fontsize=7, color=S.INK, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(8, 1.25e6)
    ax.set_ylim(-0.0335, -0.0015)
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.xaxis.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xlabel(r"duration $N=T/L$")
    ax.set_ylabel(r"$\lambda_c(N)$")


def make():
    S.use()
    res = json.loads((D / "C170_results.json").read_text(encoding="utf-8"))
    fit = res["S1"]["boxes"]["A"]["fits"]["window_lam_m0p018_m0p007"]["two_param"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE_W, 2.65), gridspec_kw=dict(wspace=0.28))
    panel_widths(a, fit)
    panel_thresholds(b, fit)
    S.panel_label(a, "(a)", x=0.62, y=0.95)
    S.panel_label(b, "(b)", x=0.90, y=0.10)
    paths = S.save(fig, "rep_v11_static_box", width=PAGE_W - 0.04, papers=())
    print("rep_v11_static_box", [str(p) for p in paths])
    return paths


if __name__ == "__main__":
    make()
