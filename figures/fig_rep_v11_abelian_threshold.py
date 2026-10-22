"""Report v1.1, Sec. 18 (abelian Higgs model in a small box): thresholds Gamma N = ln 2.

|Lambda_c|^{1/2} ln N against log10 N for the exact two-dimensional widths (N = 10 ... 1e16), the plain
Born-Oppenheimer (BO) and BO + 3/(16 r^2) widths, and the continuation with the WKB action of the Langer BO barrier at
the computed (extrapolated) energy, anchored to the exact width at x = 12.4; dashed: the deep-well limit pi sqrt2.
Inputs: ./data/C171_thresholds.csv, ./data/C171_results.json (data/higgs/abelian_c171: copies of the archived C171 results; SHA-256 in SOURCES.json).
Output: ./out/rep_v11_abelian_threshold.pdf (+ .png). Usage: python -B fig_rep_v11_abelian_threshold.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

import style_v12 as st

B = math.pi * math.sqrt(2.0)
WIDTH = 0.62 * (210 / 25.4 - 2.0)   # included at 0.62\textwidth


def main():
    st.use()
    thr = st.load_csv("higgs/abelian_c171/C171_thresholds.csv")
    res = st.load_json("higgs/abelian_c171/C171_results.json")
    lgN = np.array([math.log10(float(t["N"])) for t in thr])
    lnN = lgN * math.log(10.0)
    y_ex = np.array([float(t["|Lambda_c|^1/2*lnN"]) for t in thr])

    def ybo(key):
        v = np.array([float(t[key]) if t[key] not in ("", "nan") else np.nan for t in thr])
        return np.sqrt(-v) * lnN

    y_bo, y_bod = ybo("Lambda_c_BO"), ybo("Lambda_c_BOD")
    asy = res["thresholds_asymptotic_BO_scaled"]
    lgN_a = np.array([math.log10(a["N"]) for a in asy])
    y_a = np.array([a["sqrtLam_c_lnN"] for a in asy])

    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    ax.axhline(B, color=st.INK2, lw=0.8, ls="--", zorder=1)
    ax.text(1.05, B - 0.07, r"$\pi\sqrt{2}$", color=st.INK2, fontsize=7.5, ha="left", va="top")
    ax.semilogx(lgN_a, y_a, "-", color=st.AQUA, lw=1.2, label=r"WKB at $\varepsilon$ (continuation)", zorder=2)
    ax.semilogx(lgN, y_bod, "s", color=st.ORANGE, ms=3.2, mfc="none", mew=0.8, label=r"BO $+\,3/(16r^2)$", zorder=3)
    ax.semilogx(lgN, y_bo, "x", color=st.BLUE, ms=3.6, mew=0.9, label="plain BO", zorder=3)
    ax.semilogx(lgN, y_ex, "o", color=st.INK, ms=3.0, mew=0, label="exact (2D)", zorder=4)
    ax.set_xlim(0.9, 400.0)
    ax.set_ylim(0.6, 4.75)
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\log_{10}N$")
    ax.set_ylabel(r"$|\Lambda_c|^{1/2}\,\ln N$")
    ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.5)
    st.save(fig, "rep_v11_abelian_threshold", width=WIDTH, papers=())


if __name__ == "__main__":
    main()
