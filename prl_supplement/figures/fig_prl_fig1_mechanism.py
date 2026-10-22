"""PRL Fig. 1 (restructured Letter, 2026-10-19): the finite-volume threshold as a field-theory effect.

(a) Born-Oppenheimer radial potential U(r) of the reduced zero-mode model (2D model with the DMC shape correction,
    one-loop matching, no cross terms) at the box-A couplings (SM, mu = 1/L = mu_b, m_t = 172.60 GeV) for
    lambda = -0.012, with the gauge zero-mode couplings scaled by kappa = 1, 0.5, 0 (Delta V unchanged); the
    resonance level eps and width Gamma for kappa = 1 (data: C170 S2).
(b) Static-box threshold lambda_c(mu; N) at P = 1/2 for N = 1e2, 3e2, 1e4, through the computed boxes
    mu/mu_b = 1e-3, 1e-2, 1e-1, 1, 10 (final run05 model, C170 S1), on the SM running lambda(mu) for
    m_t = 172.2/172.6/173.0 GeV (alpha_s = 0.1180, M_h = 125.13 GeV); diamond: the hazard-equivalent threshold of
    the expanding torus at mu_b (lambda_c = -0.0121, P1 = 1/2; bar: P1 = 0.9-0.1).
(c) lambda_c(N = 300, P = 1/2) of box A when the gauge zero-mode couplings are scaled by kappa (Delta V fixed;
    2D model without cross terms, C170 S2) or one of g2, gY, y_t by s in the whole model (final model, C170 S3).

Reads only ./data_c170 (written by prepare_data_c170.py); the hazard thresholds of the expanding torus are the
published values LAM_C_HAZARD below.
usage: python -B fig_prl_fig1_mechanism.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

D = Path(__file__).resolve().parent / "data_c170"
LAM_C_HAZARD = {"0.9": -0.0091339, "0.5": -0.0121415, "0.1": -0.0151849}  # run05 summary lam_c (H012; PRL text)


def rcsv(name):
    with open(D / name, encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def rjson(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ (a)
def panel_potential(ax):
    rows = rcsv("S2_potential_lam-0.012.csv")
    r = np.array([float(x["r"]) for x in rows])
    s2 = rjson("S2_summary.json")
    at = s2["S2_2D_Gdmcs_k1.0000"]["at"]["-0.012"]
    eps, gam = at["eps"], at["Gamma"]
    styles = {"1.00": (S.AQUA, "-", 1.6, r"$\kappa=1$ (SM)"), "0.50": (S.RAMP[2], (0, (4, 1.6)), 1.2, r"$\kappa=0.5$"),
              "0.00": (S.INK2, (0, (1.2, 1.4)), 1.2, r"$\kappa=0$")}
    curves = {}
    for k, (c, ls, lw, lab) in styles.items():
        U = np.array([float(x[f"U_with_k{k}"]) for x in rows])
        curves[k] = U
        ax.plot(r, U, color=c, ls=ls, lw=lw, zorder=3)
    # resonance level of kappa = 1: solid inside the well, dotted under the barrier, arrow beyond the outer turning point
    U1 = curves["1.00"]
    s = np.sign(U1 - eps)
    idx = np.where(np.diff(s) != 0)[0]
    xs = [r[i] + (eps - U1[i]) * (r[i + 1] - r[i]) / (U1[i + 1] - U1[i]) for i in idx]
    r1, r2, r3 = xs[0], xs[1], xs[2]
    ax.plot([r1, r2], [eps, eps], color=S.AQUA, lw=1.1, zorder=4, solid_capstyle="butt")
    ax.plot([r2, r3], [eps, eps], color=S.AQUA, lw=0.9, ls=(0, (1.2, 1.2)), zorder=4)
    ax.annotate("", xy=(r3 + 1.0, eps), xytext=(r3 + 0.12, eps),
                arrowprops=dict(arrowstyle="-|>", color=S.AQUA, lw=0.9, mutation_scale=7), zorder=5)
    ax.text(0.62, eps + 0.10, r"$\varepsilon$", color=S.AQUA, fontsize=8.5, ha="left", va="bottom")
    ax.text(4.1, 5.12, r"$\kappa=1$ (SM)", color=S.AQUA, fontsize=7.5, ha="center", va="bottom")
    ax.text(3.55, eps - 0.30, rf"$\Gamma L={H.sci(gam)}$", color=S.AQUA, fontsize=7.5, ha="center", va="top")
    ax.text(2.55, 2.42, r"$\kappa=0.5$", color=S.RAMP[2], fontsize=7.5, ha="center", va="bottom")
    ax.text(0.95, -0.55, r"$\kappa=0$", color=S.INK2, fontsize=7.5, ha="left", va="center")
    ax.set_xlim(0, 8.0)
    ax.set_ylim(-1.3, 5.9)
    ax.set_xlabel(r"Higgs zero mode $rL$")
    ax.set_ylabel(r"$U(r)\,L$")
    return dict(eps=eps, Gamma=gam, r1=r1, r2=r2, r3=r3)


# ------------------------------------------------------------------------------------------------ (b)
def panel_boxes(ax):
    lc = rcsv("S1_lambda_c_N.csv")
    run = rcsv("lambdaSM_running.csv")
    s1 = rjson("S1_summary.json")
    mub = s1["boxes"]["A"]["mu_GeV"]
    order = ["E", "C", "B", "A", "D"]
    mus = np.array([s1["boxes"][b]["mu_GeV"] for b in order])
    mu = np.array([float(x["mu_GeV"]) for x in run])
    styles = {"172.2": (0, (4, 2)), "172.6": "-", "173.0": (0, (5, 1.5, 1, 1.5))}
    sel = (mu >= 3.0e12) & (mu <= 8e16)
    for mt, ls in styles.items():
        ax.plot(mu[sel], np.array([float(x[f"lam_{mt}"]) for x in run])[sel], color=S.INK,
                lw=1.3 if mt == "172.6" else 0.85, ls=ls, zorder=3)
    ax.axhline(0.0, color=S.HAIR, lw=0.6, zorder=1)
    Nstyle = {"100": (S.RAMP[1], (0, (3, 1.5))), "300": (S.AQUA, "-"), "10000": (S.RAMP[3], (0, (1.2, 1.2)))}
    out = {}
    for N, (c, ls) in Nstyle.items():
        y = []
        for b in order:
            v = [float(x["lambda_c"]) for x in lc if x["box"] == b and x["P"] == "0.5" and x["N"] == N]
            y.append(v[0])
        y = np.array(y)
        lx = np.log10(mus)
        xx = np.linspace(lx[0], lx[-1], 200)
        yy = np.interp(xx, lx, y)
        ax.plot(10 ** xx, yy, color=c, ls=ls, lw=1.2, zorder=4)
        ax.plot(mus, y, "o", ms=2.6, mfc=c, mec="white", mew=0.4, zorder=5)
        out[N] = dict(zip(order, y.tolist()))
    # labels at the right edge of the threshold lines
    lab = {"100": r"$N=10^2$", "300": r"$3\times10^2$", "10000": r"$10^4$"}
    for N in Nstyle:
        ax.text(mus[0] / 1.15, out[N]["E"], lab[N], color=Nstyle[N][0], fontsize=7.5, ha="right", va="center")
    # hazard threshold of the expanding torus at mu_b
    ax.errorbar([mub], [LAM_C_HAZARD["0.5"]], yerr=[[LAM_C_HAZARD["0.5"] - LAM_C_HAZARD["0.1"]],
                                                   [LAM_C_HAZARD["0.9"] - LAM_C_HAZARD["0.5"]]],
                fmt="D", ms=3.6, mfc=S.SURFACE, mec=S.AQUA, ecolor=S.AQUA, elinewidth=0.8, capsize=1.6, zorder=6)
    # line styles of the three top masses are given in the caption (no in-plot labels: no free space next to the curves)
    ax.set_xscale("log")
    ax.set_xlim(1.0e11, 8e16)
    ax.set_ylim(-0.0205, 0.0015)
    ax.set_xticks([1e12, 1e13, 1e14, 1e15, 1e16])
    ax.minorticks_off()
    ax.set_xlabel(r"$\mu=1/L$ (GeV)")
    ax.set_ylabel(r"$\lambda(\mu)$, $\lambda_c$")
    top = ax.secondary_xaxis("top")
    top.set_xticks([mub])
    top.set_xticklabels([r"$\mu_b$"])
    top.tick_params(direction="in", length=3, pad=1.5)
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="x", which="both", top=False)
    return dict(lambda_c=out, mu_b=mub)


# ------------------------------------------------------------------------------------------------ (c)
def panel_scaling(ax):
    kap = [x for x in rcsv("S2_lambda_c_kappa.csv") if x["model"] == "2D_Gdmcs" and x["lambda_c_N300_P05"]]
    k = np.array([float(x["kappa"]) for x in kap])
    lk = np.array([float(x["lambda_c_N300_P05"]) for x in kap])
    kmin = 0.5
    ax.axvspan(0.35, 0.478, color=S.GREY_FILL2, lw=0, zorder=0)   # kappa_thr = 0.478 (C170 S2): no N = 300 threshold
    ax.text(0.414, -0.0175, "no threshold", color=S.INK2, fontsize=7, ha="center", va="center", rotation=90)
    ax.plot(k, lk, color=S.AQUA, lw=1.5, marker="o", ms=2.8, mec="white", mew=0.4, zorder=5)
    ax.text(0.60, -0.0152, "$\\kappa$: zero\nmodes only", color=S.AQUA, fontsize=7.5, ha="left", va="center",
            linespacing=1.0)
    out = {"kappa": dict(zip(k.tolist(), lk.tolist()))}
    p3 = D / "S3_coupling_scan.csv"
    if p3.exists():
        sc = rcsv("S3_coupling_scan.csv")
        st = {"g2": (S.RAMP[3], "-", r"$g_2$"), "gY": (S.RAMP[1], (0, (4, 1.6)), r"$g_Y$"),
              "yt": (S.INK, (0, (1.2, 1.2)), r"$y_t$")}
        for g, (c, ls, lab) in st.items():
            pts = sorted((float(x["s"]), float(x["lambda_c"])) for x in sc if x["coupling"] == g and x["P"] == "P=0.5")
            if len(pts) < 3:
                continue
            ss, ll = map(np.array, zip(*pts))
            ax.plot(ss, ll, color=c, ls=ls, lw=1.1, marker="s", ms=2.2, mec="white", mew=0.3, zorder=4,
                    label=lab + r" (all sectors)" if g == "g2" else lab)
            out[g] = dict(zip(ss.tolist(), ll.tolist()))
    if p3.exists():
        ax.legend(loc="lower left", bbox_to_anchor=(0.12, 0.03), fontsize=7.5, handlelength=2.0, borderaxespad=0.2,
                  labelspacing=0.25)
    ax.axvline(1.0, color=S.HAIR, lw=0.6, zorder=1)
    ax.axhline(0.0, color=S.HAIR, lw=0.6, zorder=1)
    ax.set_xlim(0.35, 1.62)
    ax.set_ylim(-0.040, 0.0025)
    ax.set_xlabel(r"coupling multiplier $\kappa$ or $s$")
    ax.set_ylabel(r"$\lambda_c$ ($N=300$)")
    return dict(kmin=kmin, **out)


def make():
    S.use()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.36))
    ia = panel_potential(a)
    ib = panel_boxes(b)
    ic = panel_scaling(c)
    S.panel_label(a, "(a)", x=0.88, y=0.96)
    S.panel_label(b, "(b)", x=0.88, y=0.12)   # top right is crossed by the line lambda = 0
    S.panel_label(c, "(c)", x=0.88, y=0.86)
    paths, page = H.save(fig, "prl_fig1_mechanism", S.DOUBLE, papers=("prl",))
    print("prl_fig1_mechanism", f"page width {page:.3f} in", [str(p) for p in paths])
    print(json.dumps({"a": ia, "b": ib, "c": ic}, indent=1, default=float))
    return paths


if __name__ == "__main__":
    make()
