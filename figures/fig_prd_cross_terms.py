"""PRD (cross-term appendix and error budget): zero-mode cross terms and the model-refinement ladder.

(a) Shift of the critical m_t by the one-loop zero-mode/non-zero-mode cross terms (alpha_s = 0.1180; bars P1 = 0.5,
    open circles P1 = 0.9, triangles P1 = 0.1; blue stabilizing, red destabilizing):
    run05 by zero-mode gauge factor, removed in the order SU(3)-top, then U(1)_Y + SU(2) jointly (telescoping, sums to
    the total); the new-vertex groups (transverse vev mixing + Coulomb mismatch; exact top chiral bubble) that are part
    of the total; and, for orientation only, the run01 ratio-method values of U(1)_Y, SU(2) and SU(3)-top taken one at a
    time with v11 couplings (not additive; the run05 split of U(1)_Y and SU(2) did not converge).
    Data: claims_run05/cross_decomp_summary.json (crit.*, order_a_*, one_loop_groups, total_full_minus_none);
    holonomy_cross_run01/C149_summary.json dmc_components_alone.
(b) Critical m_t (P1 = 0.5) along the refinement ladder: rows of the D122 convergence table (twod_direct_sm_run01
    C150_summary.json convergence_table), + NNLO SM mapping (twod_direct_sm_run02 main critical_mt), + cross terms
    (holonomy_cross_run02 crit.with_cross), run04 (holonomy_cross_run04 A.crit) and run05 (holonomy_cross_run05 crit).
    The first steps add different physics (matching, gauge shape, second dimension); they are not a convergence series.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style_v12 as S  # noqa: E402
import _helpers_higgs as H  # noqa: E402

P = ("0.9", "0.5", "0.1")


def decomposition():
    d = H.J("claims_run05/cross_decomp_summary.json")
    cr = d["crit"]
    su3 = d["order_a_Y_SU2_SU3"]["SU3top_given_noY_noSU2"]
    ysu2 = {p: cr["full"][f"P1={p}"] - cr["noY_noSU2"][f"P1={p}"] for p in P}
    tot = d["total_full_minus_none"]
    g = d["one_loop_groups"]
    r1 = H.J("holonomy_cross_run01/C149_summary.json")["dmc_components_alone"]
    rows = [
        ("SU(3)$-$top", {p: su3[f"P1={p}"] for p in P}, "run05"),
        (r"U(1)$_Y$ + SU(2)", ysu2, "run05"),
        ("total", {p: tot[f"P1={p}"] for p in P}, "total"),
        ("vev mixing + Coulomb", {p: g["vev_mixing_plus_Coulomb"][f"P1={p}"] for p in P}, "part"),
        ("top chiral", {p: g["top_chiral"][f"P1={p}"] for p in P}, "part"),
        (r"U(1)$_Y$ alone", {p: r1["U1_Y"][f"P1={p}"] for p in P}, "run01"),
        ("SU(2) alone", {p: r1["SU2"][f"P1={p}"] for p in P}, "run01"),
        ("SU(3)$-$top alone", {p: r1["SU3-top"][f"P1={p}"] for p in P}, "run01"),
    ]
    tele = {p: rows[0][1][p] + rows[1][1][p] - rows[2][1][p] for p in P}
    return rows, tele


def panel_decomp(ax):
    rows, tele = decomposition()
    ypos = [7.6, 6.6, 5.6, 3.9, 2.9, 1.2, 0.2, -0.8]
    for (lab, v, kind), y in zip(rows, ypos):
        x = v["0.5"]
        col = S.DIV_NEG if x > 0 else S.DIV_POS
        if kind == "run01":
            ax.barh([y], [x], height=0.62, facecolor="none", edgecolor=col, hatch="////", lw=0.8, zorder=2)
        else:
            ax.barh([y], [x], height=0.62, color=col, alpha=1.0 if kind != "part" else 0.6, lw=0, zorder=2)
        ax.plot([v["0.9"]], [y], "o", ms=3.2, mfc=S.SURFACE, mec=S.INK, mew=0.8, zorder=4)
        ax.plot([v["0.1"]], [y], "^", ms=3.4, mfc=S.INK, mec=S.INK, zorder=4)
        left = x > 0 or lab == "top chiral"          # label on the far side of the bar (keeps the legend area free)
        tx = (min(v.values()) - 0.012 if x < 0 else -0.012) if left else 0.012
        ax.text(tx, y, lab, fontsize=7.5, ha="right" if left else "left", va="center", color=S.INK,
                fontweight="bold" if kind == "total" else "normal")
    ax.axvline(0, color=S.INK2, lw=0.7, zorder=3)
    ax.axhline(4.75, color=S.HAIR, lw=0.6)
    ax.axhline(2.05, color=S.HAIR, lw=0.6)
    hdr = dict(fontsize=7, color=S.INK2, style="italic", ha="left", va="center")
    ax.text(-0.265, 4.45, "of which (new vertices):", **hdr)
    ax.text(-0.265, 1.75, "first estimate, one factor at a time:", **hdr)
    ax.plot([], [], "o", ms=3.2, mfc=S.SURFACE, mec=S.INK, mew=0.8, label=r"$P_1=0.9$")
    ax.plot([], [], "s", ms=4, color=S.INK2, label=r"$P_1=0.5$ (bar)")
    ax.plot([], [], "^", ms=3.4, color=S.INK, label=r"$P_1=0.1$")
    ax.legend(loc="center right", bbox_to_anchor=(0.215, 3.35), bbox_transform=ax.transData, fontsize=7.5,
              handletextpad=0.3, borderaxespad=0.0, labelspacing=0.2)
    ax.set_xlim(-0.27, 0.215)
    ax.set_ylim(-1.45, 8.25)
    ax.set_yticks([])
    ax.set_xlabel(r"$\Delta m_c$ (GeV)")
    return dict(rows={lab: v for lab, v, _ in rows}, telescoping_residual=tele)


def ladder():
    c1 = H.J("twod_direct_sm_run01/C150_summary.json")["convergence_table"]["rows"]
    # reader-facing stage names (no internal run or note numbers in the figure)
    short = {"MU (D108/R14)": "unitary gauge", "MU_CW": "+ CW potential",
             "MC+CW (D110)": "Coulomb gauge", "1D_ex (+DMC gauge shape)": "1D + DMC shape",
             "2D_G (2D, no DMC)": "2D Gaussian", "2D_ex (D112 ratio translation)": "2D + DMC (rescaled)",
             "U_eff hazard (C144)": "calibrated 1D", "2D direct (this work, C150)": "2D direct"}
    st = []
    for r in c1:
        if r["stage"] not in short:
            continue
        v = r["recomputed_this_pipeline"] or r["published"]
        st.append((short[r["stage"]], v["P1=0.5"], "published" if not r["recomputed_this_pipeline"] else "recomputed"))
    c2 = H.J("twod_direct_sm_run02/C150_summary.json")["main_2D_direct_as0.1180"]["critical_mt"]
    st.append(("NNLO SM mapping", c2["P1=0.5"], "recomputed"))
    st.append(("+ cross terms", H.J("holonomy_cross_run02/C149_run02_summary.json")["crit"]["0.1180"]["with_cross"]["P1=0.5"],
               "recomputed"))
    st.append(("+ vev, top vertices", H.J("holonomy_cross_run04/C149_run04_summary.json")["A"]["crit"]["0.1180"]["P1=0.5"],
               "recomputed"))
    st.append(("+ Coulomb (final)", H.critical(0.1180)["0.5"], "recomputed"))
    return st


def panel_ladder(ax):
    st = ladder()
    ylo = 172.50
    x = np.arange(len(st))
    prev = None
    for i, (lab, v, kind) in enumerate(st):
        if i == 0:
            ax.bar([i], [v - ylo], bottom=ylo, width=0.62, color=S.GREY_FILL, edgecolor=S.INK2, lw=0.6, zorder=2)
        elif i == len(st) - 1:
            ax.bar([i], [v - ylo], bottom=ylo, width=0.62, color=S.AQUA_FILL, edgecolor=S.AQUA, lw=0.8, zorder=2)
            d = v - prev
            ax.plot([i - 0.31, i + 0.31], [prev, prev], color=S.MUTED, lw=0.6, ls=(0, (2, 1)), zorder=3)
        else:
            d = v - prev
            ax.bar([i], [d], bottom=prev, width=0.62, color=S.DIV_NEG if d > 0 else S.DIV_POS,
                   hatch="////" if kind == "published" else None, edgecolor="white" if kind == "published" else None,
                   lw=0, zorder=2)
        if i > 0:
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=S.MUTED, lw=0.6, zorder=1)
            dv = v - prev
            ax.text(i, max(v, prev) + 0.008, f"{dv:+.3f}", fontsize=7.5, ha="center", va="bottom", color=S.INK2,
                    rotation=90)
        prev = v
    ax.text(len(st) - 1, st[-1][1] - 0.05, f"{st[-1][1]:.3f}", fontsize=7.5, ha="center", va="top", color=S.INK,
            rotation=90)
    ax.text(0, st[0][1] + 0.008, f"{st[0][1]:.3f}", fontsize=7.5, ha="center", va="bottom", color=S.INK, rotation=90)
    ax.axvspan(7.5, 8.5, color=S.GREY_FILL2, lw=0, zorder=0)
    ax.axvspan(8.5, len(st) - 0.5, color="#eef7f3", lw=0, zorder=0)
    grp = dict(fontsize=7.5, color=S.INK2, ha="center", va="top", style="italic")
    ax.text(3.6, 173.205, "matching and reduced dynamics (earlier SM mapping)", **grp)
    ax.text(8.0, 173.245, "SM", **grp)
    ax.text(10.0, 173.245, "cross terms", **grp)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in st], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(st) - 0.4)
    ax.set_ylim(ylo, 173.26)
    ax.set_ylabel(r"$m_c(P_1=0.5)$ (GeV)")
    ax.tick_params(axis="x", which="both", top=False, length=0)
    return [(s[0], round(s[1], 4), s[2]) for s in st]


def make():
    S.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(wspace=0.2, width_ratios=[1.0, 1.55]))
    ia = panel_decomp(a)
    ib = panel_ladder(b)
    S.panel_label(a, "(a)", x=0.03, y=0.975)
    S.panel_label(b, "(b)", x=0.015, y=0.975)
    paths, page = H.save(fig, "prd_cross_terms", S.DOUBLE, papers=("prd",))
    print("prd_cross_terms", f"page {page:.3f} in", ia, ib)
    return paths


if __name__ == "__main__":
    make()
