"""Report v1.1, Sec. 18 (abelian Higgs model in a small box): width and local exponent.

(a) Gamma (units e^{2/3}/L) against x = |Lambda|^{-1/2}, Lambda = lambda/e^2: exact two-dimensional resonance (points),
    plain Born-Oppenheimer (BO) and BO with the diagonal term 3/(16 r^2) (lines), a reference line of slope -pi sqrt2
    through the last exact point, and the band where the level lies above the Langer-corrected BO barrier top.
(b) local exponent b = -d ln Gamma/dx: exact points, plain BO, the WKB action of the Langer BO barrier at the computed
    energy (continued beyond x = 12.4 with the extrapolated energy), the leading-order form pi sqrt2 - 1.26 eps x^{-2/3},
    and the deep-well value pi sqrt2.
Inputs: ./data/C171_gamma_table.csv, ./data/C171_results.json (data/higgs/abelian_c171: copies of the archived C171 results; SHA-256 in SOURCES.json).
Output: ./out/rep_v11_abelian_width.pdf (+ .png). Usage: python -B fig_rep_v11_abelian_width.py
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

import style_v12 as st

B = math.pi * math.sqrt(2.0)        # deep-well exponent in x = |Lambda|^{-1/2} (sigma_0 = 3e/2)
WIDTH = 210 / 25.4 - 2.0            # report \textwidth (a4paper, margin = 1 in)


def col(rows, key):
    return np.array([float(r[key]) for r in rows])


def main():
    st.use()
    rows = [r for r in st.load_csv("higgs/abelian_c171/C171_gamma_table.csv") if r["name"].startswith("j")]
    res = st.load_json("higgs/abelian_c171/C171_results.json")
    x = col(rows, "x=|Lambda|^-1/2")
    eps = col(rows, "eps_exact")
    G, GBO, GBOD = col(rows, "Gamma_exact"), col(rows, "Gamma_BO"), col(rows, "Gamma_BOD")
    b, bBO = col(rows, "b_exact"), col(rows, "b_BO")
    wkb = res["wkb_asymptotic"]
    xw, bw = np.array(wkb["x"]), np.array(wkb["b_WKB"])
    eps0 = wkb["eps0"]

    # level above the Langer-corrected BO barrier top U_top = (3/4) sigma0^{4/3} x^{2/3} (e = 1, sigma0 = 3/2)
    gap = 0.75 * 1.5 ** (4.0 / 3.0) * x ** (2.0 / 3.0) - eps
    i = int(np.where(gap > 0)[0][0])
    x_top = x[i - 1] - gap[i - 1] * (x[i] - x[i - 1]) / (gap[i] - gap[i - 1])

    fig, (a, c) = plt.subplots(1, 2, figsize=(WIDTH, 2.55), gridspec_kw=dict(wspace=0.30))

    # ---------------------------------------------------------------- (a) widths
    a.axvspan(0.8, x_top, color=st.GREY_FILL2, lw=0, zorder=0)
    a.semilogy(x, GBOD, "--", color=st.ORANGE, lw=1.1, label=r"BO $+\,3/(16r^2)$", zorder=2)
    a.semilogy(x, GBO, "-", color=st.BLUE, lw=1.1, label="plain BO", zorder=2)
    a.semilogy(x, G, "o", color=st.INK, ms=3.0, mfc=st.INK, mew=0, label="exact (2D)", zorder=3)
    xr = np.array([6.0, x[-1]])
    a.semilogy(xr, G[-1] * np.exp(-B * (xr - x[-1])), ":", color=st.INK2, lw=1.1, zorder=2,
               label=r"slope $-\pi\sqrt{2}$")
    a.text(1.32, 1e-16, "level above barrier top", color=st.INK2, fontsize=7.5, ha="center", va="bottom",
           rotation=90)
    a.set_xlim(0.8, 13.0)
    a.set_ylim(1e-19, 30)
    a.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(-18, 2, 3)]))
    a.yaxis.set_minor_locator(FixedLocator([10.0 ** k for k in range(-19, 2)]))
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel(r"$x=(e^2/|\lambda|)^{1/2}$")
    a.set_ylabel(r"$\Gamma\ \ [e^{2/3}/L]$")
    a.legend(loc="upper right", handlelength=2.0, borderaxespad=0.5)
    st.panel_label(a, "(a)", x=0.20, y=0.97)

    # ---------------------------------------------------------------- (b) local exponent
    c.axvspan(0.8, x_top, color=st.GREY_FILL2, lw=0, zorder=0)
    c.axhline(B, color=st.INK2, lw=0.8, ls="--", zorder=1)
    c.text(1.7, B - 0.08, r"$\pi\sqrt{2}=4.443$", color=st.INK2, fontsize=7.5, ha="left", va="top")
    xp = np.geomspace(4.0, 2000.0, 200)
    c.semilogx(xp, B - 1.2601 * eps0 * xp ** (-2.0 / 3.0), ":", color=st.MUTED, lw=1.2,
               label=r"$\pi\sqrt{2}-1.26\,\varepsilon\,x^{-2/3}$", zorder=2)
    c.semilogx(xw, bw, "-", color=st.AQUA, lw=1.2, label=r"WKB at $\varepsilon$", zorder=2)
    c.semilogx(x, bBO, "-", color=st.BLUE, lw=1.0, label="plain BO", zorder=2)
    c.semilogx(x, b, "o", color=st.INK, ms=3.0, mew=0, label="exact (2D)", zorder=3)
    c.set_xlim(0.8, 2000.0)
    c.set_ylim(1.6, 4.75)
    c.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    c.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1 * 10, numticks=20))
    c.xaxis.set_minor_formatter(NullFormatter())
    c.set_xlabel(r"$x=(e^2/|\lambda|)^{1/2}$")
    c.set_ylabel(r"$b=-d\ln\Gamma/dx$")
    c.legend(loc="lower right", handlelength=2.0, borderaxespad=0.5)
    st.panel_label(c, "(b)", x=0.20, y=0.97)

    st.save(fig, "rep_v11_abelian_width", width=WIDTH, papers=())


if __name__ == "__main__":
    main()
