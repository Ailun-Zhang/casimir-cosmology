# PROVENANCE: copied from 06_paper_draft/figures/v11/fig5_fossil_shape.py on 2026-10-15 by fig-shape; reads the draws
# from data/shape (condensed CSV), adds the next-order fossil (sigma_E lower by the factor 0.900) as circles in (a) and
# as a linear-Gaussian curve in (b), vectorized geometry, style_v12.
"""PRD figure prd_fossil_shape (Sec. VIII C, label fig:s-fossil) - one conclusion: the frozen fundamental domain of the
Standard Model turning point is an almost rectangular box; with the zero-mode energy its sides typically differ by
about 18% (16% at next order, 10% for free fields), its angles stay within about a degree of 90 degrees, and the full
nonlinear evolution leaves the linear Gaussian fossil intact.

Data (leading background, natural turning-point state, full nonlinear (L, G) equations with twisted Ewald forces):
- f0 = 0.2999: the 2048 independent-seed draws of C76 (data/shape/fossil_draws_C76_ind.csv); linear Gaussian
  reference with the C76 analytic widths (sigma_E = 0.104, sigma_T2 = 0.0065);
- next order: linear Gaussian with sigma_E multiplied by the next-order floor ratio 0.900 (C153), sigma_T2 unchanged;
- free fields: the 2048 draws of C60 (data/shape/fossil_draws_R5_hi.csv); linear reference from the C58 natural
  variances.
(a) outgoing aspect coordinates (beta_d, beta_d2); circles: 1, 2, 3 sigma_E at leading order (solid) and next order
    (dashed); grey lines: shapes with two equal sides.
(b) distribution of ln(L_max/L_min); (c) distribution of the largest deviation of an edge angle from 90 degrees.
    Steps: nonlinear draws; lines: linear Gaussian (2e5 samples, fixed seed).
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

import style_v12 as st
import _helpers_shape as hs

st.use()
C58 = st.load_json("shape/C58_results.json")["SM_AAA"]
C76 = st.load_json("shape/C76_summary.json")
NXT = st.load_json("shape/C153_summary.json")["B_next_order"]["next_order_floor"]["xi028_kappa18.0(next)"]

S2, S6 = np.sqrt(2.0), np.sqrt(6.0)
X = [np.diag([1, -1, 0]) / S2, np.diag([1, 1, -2]) / S6]
for (i, j) in ((0, 1), (0, 2), (1, 2)):
    E = np.zeros((3, 3))
    E[i, j] = E[j, i] = 1 / S2
    X.append(E)
X = np.array(X)


def geometry_many(B):
    """ln(L_max/L_min) and the largest |90 deg - edge angle| for G = exp(2 sum_A beta_A X_A), vectorized."""
    M = 2 * np.tensordot(B, X, axes=1)                    # (N,3,3) symmetric
    w, V = np.linalg.eigh(M)
    G = np.einsum("nij,nj,nkj->nik", V, np.exp(w), V)
    s = np.sqrt(np.einsum("nii->ni", G))
    dev = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        dev.append(np.degrees(np.abs(np.arcsin(G[:, i, j] / (s[:, i] * s[:, j])))))
    return np.log(s.max(1) / s.min(1)), np.max(dev, axis=0)


def load_draws(rel):
    rows = st.load_csv(rel)
    assert len(rows) == 2048 and all(r["survivor"] == "1" and r["gates_pass"] == "1" for r in rows)
    B = np.array([[float(r[k]) for k in ("beta_plus_d", "beta_plus_d2", "beta_plus_12", "beta_plus_13",
                                         "beta_plus_23")] for r in rows])
    lnl = np.array([float(r["ln_Lmax_over_Lmin"]) for r in rows])
    ang = np.array([float(r["max_angle_deviation_deg"]) for r in rows])
    chk = geometry_many(B[:4])
    assert np.allclose(chk[0], lnl[:4], atol=1e-10) and np.allclose(chk[1], ang[:4], atol=1e-8), chk
    return B, lnl, ang


B_zm, lnl_zm, ang_zm = load_draws("shape/fossil_draws_C76_ind.csv")
B_fr, lnl_fr, ang_fr = load_draws("shape/fossil_draws_R5_hi.csv")

sigE_zm, sigT_zm = C76["sigma_lin_analytic"]["aspect"], C76["sigma_lin_analytic"]["angle"]
ratio_nx = NXT["ratio_to_xi030"]
sigE_nx = sigE_zm * ratio_nx
sigE_fr = np.sqrt(C58["aspect"]["var_beta_plus_natural"])
sigT_fr = np.sqrt(C58["angle"]["var_beta_plus_natural"])
rng = np.random.default_rng(20261014)
NREF = 200_000
lnl_ref_zm, ang_ref_zm = geometry_many(rng.standard_normal((NREF, 5)) * np.array([sigE_zm] * 2 + [sigT_zm] * 3))
lnl_ref_fr, ang_ref_fr = geometry_many(rng.standard_normal((NREF, 5)) * np.array([sigE_fr] * 2 + [sigT_fr] * 3))
lnl_ref_nx, _ = geometry_many(rng.standard_normal((NREF, 5)) * np.array([sigE_nx] * 2 + [sigT_zm] * 3))

fig = plt.figure(figsize=(st.DOUBLE, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.1], wspace=0.40)
a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

# (a) aspect plane -------------------------------------------------------------------------------------------------------
R = 0.42
for ang_deg in (90, 30, -30):
    t = np.radians(ang_deg)
    a.plot([-R * np.cos(t), R * np.cos(t)], [-R * np.sin(t), R * np.sin(t)], color=st.HAIR, lw=0.7, zorder=1)
th = np.linspace(0, 2 * np.pi, 361)
for k in (1, 2, 3):
    a.plot(k * sigE_zm * np.cos(th), k * sigE_zm * np.sin(th), color=st.INK2, lw=0.8, zorder=4)
    a.plot(k * sigE_nx * np.cos(th), k * sigE_nx * np.sin(th), color=st.INK2, lw=0.7, ls=(0, (2.5, 1.5)), zorder=4)
a.scatter(B_zm[:, 0], B_zm[:, 1], s=1.6, color=st.ORANGE, alpha=0.55, lw=0, zorder=2)
a.scatter(B_fr[:, 0], B_fr[:, 1], s=1.2, color=st.MUTED, alpha=0.55, lw=0, zorder=3)
a.plot([0], [0], marker="+", color=st.INK, ms=6, mew=0.9, zorder=5)
a.text(0.018, 0.40, r"$L_1{=}L_2$", fontsize=7, color=st.MUTED, ha="left", va="top")
a.text(0.330, 0.1905 + 0.016, r"$L_2{=}L_3$", fontsize=7, color=st.MUTED, ha="center", va="bottom", rotation=30)
a.text(0.330, -0.1905 - 0.016, r"$L_1{=}L_3$", fontsize=7, color=st.MUTED, ha="center", va="top", rotation=-30)
for k in (1, 2, 3):
    a.text(sigE_zm * k * np.cos(np.radians(-120)) - 0.004, sigE_zm * k * np.sin(np.radians(-120)) - 0.004,
           r"$%d\sigma_E$" % k if k > 1 else r"$\sigma_E$", fontsize=7, color=st.INK2, ha="right", va="top", zorder=6,
           bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
a.set_xlim(-R, R)
a.set_ylim(-R, R)
a.set_aspect("equal")
for ax_ in (a.xaxis, a.yaxis):
    ax_.set_major_formatter(lambda v, p: hs.fmt(v, 1) if abs(v) > 1e-9 else "0")
a.set_xlabel(r"outgoing $\beta_d$")
a.set_ylabel(r"outgoing $\beta_{d2}$")
st.panel_label(a, "(a)", x=0.04, y=0.96)

# (b) side-length contrast ----------------------------------------------------------------------------------------------
XB = 0.56
bins = np.linspace(0, XB, 43)
hi_fr = np.histogram(lnl_fr, bins=bins, density=True)[0]
b.hist(lnl_fr, bins=bins, density=True, histtype="step", color=st.MUTED, lw=1.0, label="free fields")
b.hist(lnl_zm, bins=bins, density=True, histtype="step", color=st.ORANGE, lw=1.3, label=r"$f_0=0.30$")
fine = np.linspace(0, XB, 169)
hr_zm, e = np.histogram(lnl_ref_zm, bins=fine, density=True)
hr_fr, _ = np.histogram(lnl_ref_fr, bins=fine, density=True)
hr_nx, _ = np.histogram(lnl_ref_nx, bins=fine, density=True)
mid = 0.5 * (e[1:] + e[:-1])
b.plot(mid, hr_zm, color=st.INK, lw=0.9, label="linear Gaussian")
b.plot(mid, hr_fr, color=st.INK, lw=0.9)
b.plot(mid, hr_nx, color=st.INK, lw=0.8, ls=(0, (2.5, 1.5)), label="next order")
top = 1.15 * max(hr_fr.max(), hi_fr.max())
b.set_ylim(0, top)
med_zm, med_fr, med_nx = np.median(lnl_zm), np.median(lnl_fr), np.median(lnl_ref_nx)
for v, col, ytop in ((med_zm, st.ORANGE, 0.86), (med_fr, st.MUTED, 1.0)):
    b.axvline(v, color=col, lw=0.7, ls=(0, (3, 2)), ymax=ytop)
b.text(med_fr + 0.01, 0.955 * top, f"median {med_fr:.3f} ({100 * np.expm1(med_fr):.0f}%)", fontsize=7,
       color=st.INK2, ha="left", va="top")
b.annotate(f"median {med_zm:.3f} ({100 * np.expm1(med_zm):.0f}%)", (med_zm, 0.40 * top), xytext=(0.29, 0.40 * top),
           fontsize=7, color=st.INK2, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK2, shrinkA=1, shrinkB=0))
b.text(0.32, 0.31 * top, f"next order: {100 * np.expm1(med_nx):.0f}%", fontsize=7, color=st.INK2, ha="left",
       va="center")
b.set_xlim(0, XB)
b.xaxis.set_major_formatter(lambda v, p: hs.fmt(v, 1) if abs(v) > 1e-9 else "0")
b.set_xlabel(r"$\ln(L_{\max}/L_{\min})$")
b.set_ylabel("probability density")
b.legend(loc="upper right", fontsize=7, handlelength=1.8, borderaxespad=0.3, bbox_to_anchor=(1.0, 0.86),
         labelspacing=0.3)
st.panel_label(b, "(b)", x=0.04, y=0.96)

# (c) angle deviation ---------------------------------------------------------------------------------------------------
bins = np.linspace(0, 2.4, 33)
c.hist(ang_fr, bins=bins, density=True, histtype="step", color=st.MUTED, lw=1.0, label="free fields")
c.hist(ang_zm, bins=bins, density=True, histtype="step", color=st.BLUE, lw=1.3, label=r"$f_0=0.30$")
hr, e = np.histogram(ang_ref_zm, bins=np.linspace(0, 2.4, 129), density=True)
c.plot(0.5 * (e[1:] + e[:-1]), hr, color=st.INK, lw=0.9, label="linear Gaussian")
top = 1.17 * max(hr.max(), np.histogram(ang_zm, bins=bins, density=True)[0].max(),
                 np.histogram(ang_fr, bins=bins, density=True)[0].max())
c.set_ylim(0, top)
c.axvline(np.median(ang_zm), color=st.BLUE, lw=0.7, ls=(0, (3, 2)))
c.text(np.median(ang_zm) + 0.04, 0.955 * top, f"median {np.median(ang_zm):.2f}" + r"$^\circ$", fontsize=7,
       color=st.INK2, ha="left", va="top")
c.set_xlim(0, 2.4)
c.set_xlabel(r"largest $|90^\circ-\theta_{ij}|$ (deg)")
c.set_ylabel("probability density")
c.legend(loc="upper right", fontsize=7, handlelength=1.8, borderaxespad=0.3, bbox_to_anchor=(1.0, 0.80),
         labelspacing=0.3)
st.panel_label(c, "(c)", x=0.04, y=0.96)

summary = {
    "n_draws": [len(lnl_zm), len(lnl_fr)],
    "sigma_E_linear": {"zero_mode": sigE_zm, "next_order": sigE_nx, "free": float(sigE_fr)},
    "next_order_ratio": ratio_nx,
    "lnL_median_NL": {"zero_mode": float(med_zm), "free": float(med_fr)},
    "lnL_median_ref": {"zero_mode": float(np.median(lnl_ref_zm)), "next_order": float(med_nx),
                       "free": float(np.median(lnl_ref_fr))},
    "side_ratio_median": {"zero_mode_NL": float(np.exp(med_zm)), "next_order_lin": float(np.exp(med_nx)),
                          "free_NL": float(np.exp(med_fr))},
    "ang_median_NL": {"zero_mode": float(np.median(ang_zm)), "free": float(np.median(ang_fr))},
    "aspect_rms_NL": {"zero_mode": float(np.sqrt(np.mean(B_zm[:, :2] ** 2))),
                      "free": float(np.sqrt(np.mean(B_fr[:, :2] ** 2)))},
}
print(json.dumps(summary, indent=1))
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(st.save(fig, "prd_fossil_shape", width=st.DOUBLE, papers=("prd",)))
    (st.OUT / "prd_fossil_shape_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
