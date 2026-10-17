"""PRD (Sec. V, reduced dynamics): Higgs zero mode r and hypercharge zero mode s = |w| in the two-dimensional model.

Filled contours: probability density |u(r, s)|^2 (u = r^{3/2} s psi, i.e. density per dr ds including the measure) of
the lowest eigenstate of the D112 2D_ex operator, normalised to its maximum (0.1, 0.3, 0.5, 0.7, 0.9);
lines: field-space potential V(r, s) = lambda r^4/4 + dV_1 + E_x(r) + Om_eff(r)^2 s^2/2 (E_x = SU(2) zero-point part);
thick line V = E (classical boundary at the eigenvalue). Top: lambda = 0, bound ground state (wall at r = 9/L, far
outside). Bottom: lambda = -0.013, quasi-bound state with the Dirichlet wall r_w = 4.4/L just outside the barrier top
(T14a main wall); beyond the barrier V drops below E (escape along r).
Data: generated/wavefunction_2Dex_lam0.npz, generated/wavefunction_2Dex_lamm0p013.npz (prepare_data_higgs.py
gen_wavefunction; T14a same-wall FD solver, h = 0.04/L), higgs_beyond_bo_run01/C140_summary.json (reference energies).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

CM = LinearSegmentedColormap.from_list("aqua", ["#ffffff", S.AQUA_FILL, S.AQUA, "#0f6e4c"])


def panel(ax, tag, title):
    z = np.load(H.HD / "generated" / f"wavefunction_2Dex_{tag}.npz")
    r, s, dens, E = z["r"], z["s"], z["dens"], float(z["E"][0])
    rd, sd, V = z["rd"], z["sd"], z["V"]
    lam, rw = float(z["lam"]), float(z["rw"])
    ax.contourf(r, s, dens.T, levels=[0.1, 0.3, 0.5, 0.7, 0.9, 1.01], cmap=CM, vmin=0.0, vmax=1.15, zorder=1)
    lv = [E - 1.0, E - 0.5, E + 0.5, E + 1.0, E + 1.5]
    cs = ax.contour(rd, sd, V.T, levels=sorted(lv), colors=S.MUTED, linewidths=0.6, zorder=2)
    ax.contour(rd, sd, V.T, levels=[E], colors=S.INK, linewidths=1.1, zorder=3)
    info = dict(E=E, lam=lam, rw=rw)
    # transverse zero-point width <s^2>^{1/2} = (3 / 2 Om_eff)^{1/2}: squeezed as the gauge masses grow with r
    srms = np.sqrt(1.5 / np.sqrt(z["om2_rd"]))
    ax.plot(rd, srms, color=S.INK2, lw=0.9, ls=(0, (1, 1.4)), zorder=3)
    bb = dict(boxstyle="square,pad=0.08", fc=S.SURFACE, ec="none", alpha=0.85)
    Uad = z["Uad_rd"]
    i_top = int(np.argmax(np.where(rd > 2.0, Uad, -1e9)))
    if lam < 0:
        ax.axvline(rw, color=S.INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
        ax.text(rw + 0.08, 2.72, r"$r_w$", color=S.INK2, fontsize=7.5, ha="left", va="center", bbox=bb, zorder=5)
        # escape: the adiabatic channel U_ad = V(r, s) + (3/2) Om_eff drops below E beyond its outer turning point
        out = np.where((rd > rd[i_top]) & (Uad < E))[0]
        r2 = float(rd[out[0]])
        ax.annotate("", xy=(7.35, 0.3), xytext=(r2, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=S.HATCH_RED, lw=1.1, mutation_scale=8), zorder=4)
        ax.text(0.5 * (r2 + 7.35), 0.42, "escape", color=S.HATCH_RED, fontsize=7.5, ha="center", va="bottom")
        ax.axvline(rd[i_top], color=S.INK, lw=0.6, ls=(0, (1, 1.2)), zorder=3)
        ax.text(rd[i_top] - 0.08, 2.72, r"top of $U_{\rm ad}$", color=S.INK, fontsize=7.5, ha="right", va="center",
                bbox=bb, zorder=5)
        info.update(r_top_Uad=float(rd[i_top]), r2_Uad=r2)
    else:
        ax.text(4.65, float(np.interp(4.65, rd, srms)) + 0.12, r"$\langle s^2\rangle^{1/2}$", color=S.INK2,
                fontsize=7.5, ha="left", va="bottom", bbox=bb, zorder=5)
        ax.text(3.3, 0.3, r"$V=E$", color=S.INK, fontsize=7.5, ha="left", va="center", bbox=bb, zorder=5)
        info.update(r_top_Uad=None)
    ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=8, ha="left", va="top", color=S.INK, bbox=bb, zorder=6)
    # (b): a little lower, clear of the potential contours that leave through the top edge
    ax.text(0.98, 0.96 if lam >= 0 else 0.89, rf"$E L={E:.3f}$", transform=ax.transAxes, fontsize=7.5, ha="right",
            va="top", color=S.INK, bbox=bb, zorder=6)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 3.0)
    ax.set_ylabel(r"$sL=|\mathbf{w}|L$")
    return info


def make():
    S.use()
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.7), sharex=True, gridspec_kw=dict(hspace=0.12))
    ia = panel(a, "lam0", r"(a) $\lambda=0$")
    ib = panel(b, "lamm0p013", H.minus(r"(b) $\lambda=-0.013$"))
    b.set_xlabel(r"Higgs zero mode $rL$")
    c140 = H.J("higgs_beyond_bo_run01/C140_summary.json")
    ref = {"lam0_Q_2D_ex": c140["lam0"]["Q_2D_ex"], "lamm0p013_Q_siegert": c140["Q"]["2D_ex"]["-0.013"],
           "lamm0p013_Gamma": c140["gamma"]["2D_ex"]["-0.013"]}
    paths, page = H.save(fig, "prd_higgs_wavefunction", S.SINGLE, papers=("prd",))
    print("prd_higgs_wavefunction", f"page {page:.3f} in", ia, ib, ref)
    return paths


if __name__ == "__main__":
    make()
