"""Report v1.1 figure: the gauge zero-mode switch kappa (C170 S2; box-A couplings, two-dimensional model without
cross terms, Delta V at the physical couplings).

(a) Static threshold lambda_c(N = 300, P = 1/2) against kappa, with the DMC shape transported by the exact zero-mode
    scaling (2D_Gdmcs, main) and without it (2D_G). Shaded: no threshold for N = 300 (kappa < kappa_thr = 0.478);
    darker: no narrow resonance for any lambda <= 0 (kappa < kappa_min = 0.43).
(b) Widths Gamma L of the narrow resonances (2D_Gdmcs) against lambda for kappa = 0.475 ... 1.5; dashed: the width
    ln2/300 that defines lambda_c(300) (described in the caption, no in-plot label).

Reads only data/higgs/static_box_c170 (copies of the archived C170 results; SHA-256 in SOURCES_c170.json).  Usage: python -B fig_rep_v11_kappa.py
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
PAGE_W = 15.0 / 2.54
KAPPA_THR, KAPPA_MIN = 0.4777022775788982, 0.4298816433842907   # C170 S2 boundaries (2D_Gdmcs)


def rcsv(name):
    with open(D / name, encoding="utf-8") as fh:
        return list(csv.DictReader([ln for ln in fh if not ln.startswith("#")]))


def panel_threshold(ax):
    rows = rcsv("lambda_c_kappa.csv")
    ax.axvspan(0.2, KAPPA_MIN, color=S.GREY_FILL, lw=0, zorder=0)
    ax.axvspan(KAPPA_MIN, KAPPA_THR, color=S.GREY_FILL2, lw=0, zorder=0)
    st = {"2D_Gdmcs": (S.AQUA, "-", 1.4, "o", "DMC shape transported"),
          "2D_G": (S.RAMP[2], (0, (4, 1.6)), 1.0, "s", "Gaussian shape")}
    for m, (c, ls, lw, mk, lab) in st.items():
        r = [q for q in rows if q["model"] == m and q["lambda_c_N300_P05"]]
        k = np.array([float(q["kappa"]) for q in r])
        lc = np.array([float(q["lambda_c_N300_P05"]) for q in r])
        ax.plot(k, lc, color=c, ls=ls, lw=lw, marker=mk, ms=2.6, mec="white", mew=0.3, zorder=4, label=lab)
    ax.axhline(0.0, color=S.HAIR, lw=0.6, zorder=1)
    ax.axvline(1.0, color=S.HAIR, lw=0.6, zorder=1)
    ax.text(0.315, -0.019, "no narrow resonance", fontsize=7, color=S.INK2, rotation=90, ha="center", va="center")
    ax.set_xlim(0.2, 1.6)
    ax.set_ylim(-0.040, 0.002)
    ax.set_xlabel(r"gauge zero-mode coupling factor $\kappa$")
    ax.set_ylabel(r"$\lambda_c$ ($N=300$, $P=1/2$)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.20, 0.02), fontsize=7, handlelength=2.2, labelspacing=0.25,
              borderaxespad=0.3)


def panel_widths(ax):
    res = json.loads((D / "C170_results.json").read_text(encoding="utf-8"))
    ks = res["S2"]["kappa_scan"]
    sel = [("0.4750", S.RAMP[0]), ("0.5000", S.RAMP[1]), ("0.6000", S.RAMP[2]), ("0.7500", S.RAMP[3]),
           ("1.0000", S.AQUA), ("1.2500", S.INK2), ("1.5000", S.INK)]
    for k, c in sel:
        t = ks[f"S2_2D_Gdmcs_k{k}"]["table"]
        pts = sorted((r["lam"], r["Gamma"]) for r in t if r["status"] in ("ok", "ok_broad") and r.get("Gamma"))
        lam, g = map(np.array, zip(*pts))
        ax.plot(lam, np.log(g), color=c, lw=1.4 if k == "1.0000" else 0.9, marker="o", ms=2.3, mec="white",
                mew=0.3, zorder=4, label=rf"$\kappa={float(k):g}$")
    ax.legend(loc="lower left", fontsize=7, handlelength=1.6, labelspacing=0.2, borderaxespad=0.3)
    y = math.log(math.log(2) / 300)
    ax.axhline(y, color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(-0.041, 0.0015)
    ax.set_ylim(-18.0, 0.0)
    ax.xaxis.set_major_locator(mt.MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
    ax.set_xlabel(r"$\lambda(1/L)$")
    ax.set_ylabel(r"$\ln(\Gamma L)$")


def make():
    S.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE_W, 2.55), gridspec_kw=dict(wspace=0.28))
    panel_threshold(a)
    panel_widths(b)
    S.panel_label(a, "(a)", x=0.89, y=0.88)
    S.panel_label(b, "(b)", x=0.89, y=0.95)
    paths = S.save(fig, "rep_v11_kappa", width=PAGE_W - 0.04, papers=())
    print("rep_v11_kappa", [str(p) for p in paths])
    return paths


if __name__ == "__main__":
    make()
