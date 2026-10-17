"""PRD (hazard, real-time evolution and the xi_H condition): survival along the turning.

(a) P1(tau), tau = ht, from the hazard integral I(tau) = int_0^tau Gamma(tau') dtau' / (h L_b sqrt(cosh 2 tau'))
    (run05 node widths, alpha_s = 0.1180; ln Gamma interpolated by a cubic spline between the 11 nodes; the end
    values reproduce the archived P1 grid), with the norm of the one-dimensional real-time evolution (run05,
    xi_H = 1/6, open markers) and, for m_t = 172.6 GeV, xi_H = 0 (crosses; the final value includes the classical
    stage after the switch at tau = 6).
    Data: holonomy_cross_run05/X_main_u*.json (gamma), universality_run01/C156_eta.json (hLb),
    holonomy_cross_run05/C149_run05_summary.json (P1_grid), claims_run05/xi_run05/part2/p2_main_*.json (hist).
(b) Relative loss 1 - P_surv(xi_H)/P_surv(1/6): one-dimensional real-time (run05; solid lambda(1/L), dashed
    lambda(phi) prescription) and two-dimensional real-time TDSE on (r, s) (run02 centre, markers); the threshold
    0.01 defines xi_c (ticks: run05 xi_c of the two prescriptions).
    Data: claims_run05/xi_run05/C157_xi_quick.json (P), realtime_2d_run01/C157_twod.json (P_compare),
    claims_run05/xi_run05_summary.json (lam_1L, lam_phi).
(c) Universality: xi_c(mu_b) vs P1 for all direct computations (run05 and run02 centres; run02 SM-mapping and alpha_s
    variants) with the universal curve used in prl_fig3_xi (helpers.XiUniversal).
"""
from __future__ import annotations

import math
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

MT = [("172.2", S.RAMP[0]), ("172.6", S.RAMP[1]), ("173.0", S.RAMP[2]), ("173.4", S.RAMP[3])]


def hazard_curves():
    hlb = H.J("universality_run01/C156_eta.json")["hLb"]
    nodes = [0.5 * k for k in range(11)]
    g = {}
    for u in nodes:
        for c in H.J(f"holonomy_cross_run05/X_main_u{u:.4f}.json")["cols"]:
            if abs(c["als"] - 0.118) < 1e-9:
                g.setdefault(round(c["mt"], 2), {})[u] = c["gamma"]
    uu = np.linspace(0, 5, 2001)
    out = {}
    grid = H.J("holonomy_cross_run05/C149_run05_summary.json")["P1_grid"]
    for m, _ in MT:
        gm = g[round(float(m), 2)]
        cs = CubicSpline(nodes, np.log([gm[u] for u in nodes]))
        f = np.exp(cs(uu)) / (hlb * np.sqrt(np.cosh(2 * uu)))
        integ = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(uu))])
        P = np.exp(-integ)
        out[m] = (uu, P, float(P[-1]) - grid[m])
    return out


def panel_time(ax):
    hz = hazard_curves()
    chk = {}
    for m, col in MT:
        u, P, d = hz[m]
        ax.plot(u, P, color=col, lw=1.3, zorder=3)
        y = float(np.interp(2.5, u, P))
        ax.text(2.5, y + 0.025, (r"$m_t=$" if m == "172.2" else "") + m + (" GeV" if m == "172.2" else ""),
                color=col, fontsize=7.5, ha="center", va="bottom")
        chk[m] = round(d, 5)
    for m, col in MT[:3]:
        h = H.J(f"claims_run05/xi_run05/part2/p2_main_mt{float(m):.2f}_xi0.1667.json")["hist"]
        ax.plot([e["u"] for e in h], [e["norm"] for e in h], "o", ms=3.0, mfc=S.SURFACE, mec=col, mew=0.9, zorder=4)
    h0 = H.J("claims_run05/xi_run05/part2/p2_main_mt172.60_xi0.0000.json")["hist"]
    ax.plot([e["u"] for e in h0], [e["norm"] for e in h0], "x", ms=3.6, color=S.HATCH_RED, mew=1.0, zorder=5)
    pf = h0[-1]["P_surv"]
    ax.annotate("", xy=(7.45, pf), xytext=(h0[-1]["u"], h0[-1]["norm"]),
                arrowprops=dict(arrowstyle="-|>", color=S.HATCH_RED, lw=0.8, ls=(0, (2, 1.5)), mutation_scale=7),
                zorder=5)
    ax.text(6.95, 0.29, rf"$\xi_H=0$" + "\n" + rf"$P={pf:.2f}$", fontsize=7.5, color=S.HATCH_RED, ha="right",
            va="center", linespacing=1.1, zorder=4,
            bbox=dict(boxstyle="square,pad=0.1", fc=S.SURFACE, ec="none"))  # masks the tau = 6 hairline
    ax.axvline(6.0, color=S.HAIR, lw=0.6, zorder=1)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"$\tau=ht$")
    ax.set_ylabel(r"survival $P_1(\tau)$")
    return dict(P5_minus_grid=chk, xi0_final=pf)


def panel_xi(ax):
    q = H.J("claims_run05/xi_run05/C157_xi_quick.json")["P"]
    tw = H.J("realtime_2d_run01/C157_twod.json")["P_compare"]
    xr = H.xi_run05()
    out = {}
    for (m, col), mk in zip(MT[:3], ("o", "s", "^")):
        key = f"{float(m):.2f}"
        for pres, ls in (("P_surv", "-"), ("P_surv_rg", (0, (3, 1.5)))):
            d = q[f"{pres}|mt={key}|u_sw=6.0"]
            xs = np.array([float(k) for k in d])
            P = np.array([d[k] for k in d])
            ref = d["0.1667"]
            sel = xs < 0.16
            ax.plot(xs[sel], 1 - P[sel] / ref, color=col, lw=1.1, ls=ls, zorder=3)
        pts = {float(k.split("xi=")[1]): v["P2D_1L"] for k, v in tw.items() if k.startswith(f"mt={key}|")}
        if 0.1667 in pts:
            xx = sorted(x for x in pts if x < 0.16)
            ax.plot(xx, [1 - pts[x] / pts[0.1667] for x in xx], mk, ms=3.2, mfc=col, mec=S.INK, mew=0.4, zorder=5)
            out[key] = len(xx)
        v = xr[key]
        for xc in (v["lam_1L"], v["lam_phi"]):
            ax.plot([xc, xc], [0.0065, 0.0155], color=col, lw=1.0, zorder=4)
        ax.plot([], [], color=col, lw=1.1, label=m + (" GeV" if m == "173.0" else ""))
    ax.legend(loc="lower left", fontsize=7.5, handlelength=1.4, borderaxespad=0.3, labelspacing=0.2)
    ax.axhline(0.01, color=S.AQUA, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(0.1605, 0.0085, "0.01", color=S.AQUA, fontsize=7.5, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_xlim(0, 0.165)
    ax.set_ylim(2e-4, 1.2)
    ax.set_xlabel(r"$\xi_H(\mu_b)$")
    ax.set_ylabel(r"$1-P_{\rm surv}(\xi_H)/P_{\rm surv}(1/6)$")
    return dict(twoD_points=out)


def panel_univ(ax):
    X = H.XiUniversal()
    pts = H.xi_points()
    sty = {"run05": ("o", S.AQUA, S.AQUA), "run02 ": ("o", S.SURFACE, S.AQUA), "lam": ("s", S.MUTED, S.INK2),
           "as1": ("^", S.MUTED, S.INK2)}
    for P, xi, src in pts:
        if src.startswith("run05"):
            k = "run05"
        elif src.startswith("run02 variant"):
            k = "lam" if "|" in src and src.split("variant ")[1].startswith("lam") else "as1"
        else:
            k = "run02 "
        mk, fc, ec = sty[k]
        ax.plot([-math.log(P)], [xi], mk, ms=3.4 if k != "run05" else 3.0, mfc=fc, mec=ec, mew=0.8,
                zorder=5 if k == "run05" else 4)
    x = np.logspace(math.log10(-math.log(X.Pmax)), math.log10(-math.log(X.Pmin)), 300)
    ax.plot(x, X(np.exp(-x)), color=S.AQUA, lw=1.2, zorder=3)
    xe = np.logspace(math.log10(-math.log(X.Pmin)), math.log10(-math.log(0.1)), 30)
    ax.plot(xe, X.extrapolated(np.exp(-xe), 0.1), color=S.AQUA, lw=1.0, ls=(0, (2, 1.2)), zorder=3)
    for lab, mk, fc, ec in (("final model", "o", S.AQUA, S.AQUA), ("earlier model", "o", S.SURFACE, S.AQUA),
                            ("SM-mapping var.", "s", S.MUTED, S.INK2), (r"$\alpha_s$ var.", "^", S.MUTED, S.INK2)):
        ax.plot([], [], mk, ms=3.4, mfc=fc, mec=ec, mew=0.8, label=lab)
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.2)
    ax.set_xscale("log")
    ax.set_xlim(2.5e-4, 3.0)
    ax.set_ylim(0, 0.165)
    ax.set_xlabel(r"$-\ln P_1$")
    ax.set_ylabel(r"$\xi_c(\mu_b)$")
    top = ax.secondary_xaxis("top")
    pt = [0.999, 0.99, 0.9, 0.5, 0.1]
    top.set_xticks([-math.log(p) for p in pt])
    top.set_xticklabels([f"{p:g}" for p in pt])
    top.minorticks_off()
    top.set_xlabel(r"$P_1$", labelpad=2)
    top.tick_params(direction="in", length=3, pad=1.5)
    ax.tick_params(axis="x", which="both", top=False)
    return dict(n=X.n, max_resid=float(np.max(np.abs(X.resid))))


def make():
    S.use()
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw=dict(wspace=0.42))
    ia = panel_time(axs[0])
    ib = panel_xi(axs[1])
    ic = panel_univ(axs[2])
    for ax, t, x in zip(axs, ("(a)", "(b)", "(c)"), (0.03, 0.88, 0.88)):
        S.panel_label(ax, t, x=x, y=0.965 if t != "(c)" else 0.18)
    paths, page = H.save(fig, "prd_hazard_realtime", S.DOUBLE, papers=("prd",))
    print("prd_hazard_realtime", f"page {page:.3f} in", ia, ib, ic)
    return paths


if __name__ == "__main__":
    make()
