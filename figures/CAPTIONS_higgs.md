# Figure captions, H group (fig-higgs draft, 2026-10-15)

Drafts for the PRL and PRD authors. Notation as in WRITING_SPEC §2 (Γ width, τ = ht, f₀, r_T, ξ_H at μ_b). First
sentence is a noun phrase; PRL 30–60 words, PRD 50–90 words. The "Data" line is for the authors and the code package,
not part of the caption. All plotted numbers come from `figures/data/higgs` (provenance in `figures/data/SOURCES_higgs.json`).

---

## prl_fig1_threshold (PRL Fig. 1; single column, two panels)

**Caption (PRL, ≈55 words).** Finite-volume threshold of the Higgs zero mode. (a) Effective potential along $r$ for
$\lambda=-0.010$ and $-0.013$: the zero-point energy of the gauge zero modes (dotted) walls in the negative quartic
(dashed), leaving a resonance at $\varepsilon$ with width $\Gamma$. (b) SM running $\lambda(\mu)$ for
$m_t=172.2$, 172.6, 173.0 GeV (band: $\alpha_s\pm0.0009$); inset: $\lambda_c(\mu_b)$ for $P_1=0.9$–0.1 and the SM point.

Data: (a) `generated/potential_2Dex_lam010_013.csv` (D112 2D_ex reduced model, v11 couplings, MC+CW matching;
U = E_zp + λr⁴/4 + δV₁), ε and Γ from `fulldim_dmc_run01/T14a_profile.json` (`Q_2D_ex_siegert`, `gamma_2D_ex`):
εL = 4.287/4.278, ΓL = 4.2×10⁻⁴/2.2×10⁻³; the two ε differ by 0.009/L and are drawn as one level; dots = outer turning
points. (b) `generated/lambda_running.csv` (SM provider, NNLO central; reproduces C148 λ(μ_b) = −0.0097741 exactly);
λ_c = −0.0091/−0.0121/−0.0152 from `holonomy_cross_run05/C149_run05_summary.json` `lam_c`; SM-point bar = SM-mapping
uncertainty of the pole formulation (0.102 GeV × |dλ(μ_b)/dm_t| = 0.00067).

## prd_threshold (PRD; two columns, same panels side by side)

**Caption (PRD, ≈80 words).** Finite-volume threshold of the Higgs zero mode. (a) Effective potential along $r$ in the
reduced two-dimensional model for $\lambda=-0.010$ and $-0.013$ (units of $1/L$): the zero-point energy of the gauge zero
modes (dotted) rises linearly and walls in the negative quartic (dashed, same origin), so the state is a resonance at
$\varepsilon$ with width $\Gamma$ (dots: outer turning points). (b) SM running $\lambda(\mu)$ for $m_t=172.2$, 172.6 and
173.0 GeV (band: $\alpha_s=0.1180\pm0.0009$); inset: the hazard-equivalent threshold $\lambda_c(\mu_b)$ for
$P_1=0.9$, 0.5, 0.1 at $\mu_b=\bar M_P/722$ and the SM value with its SM-mapping uncertainty.

---

## prl_fig2_criterion (PRL Fig. 2; single column, two panels)

**Caption (PRL, ≈60 words).** Survival criterion in the top-mass plane. (a) $P_1(m_t)$ for $\alpha_s=0.1171$, 0.1180,
0.1189 with theory $1\sigma$ bars; grey band: direct $m_t$; bars: $\sigma_{t\bar t}$ pole mass, ATLAS+CMS, and FCC-ee
(thick: $m_t$; thin: total). Top axis: $\bar m_t(\bar m_t)$. (b) Critical lines $P_1=0.9$, 0.5, 0.1 (theory $1\sigma$),
$\lambda(\mu_b)=0$, and 68/95% ellipses for present data and FCC-ee.

Data: `claims_run05/fig2a_P1_grid_mt_alphas.csv` (run05 centre; α_s = 0.1171/0.1189 interpolated in ln(−ln P₁) between
grid rows, critical masses reproduced to ≤1.1 MeV), `fig2a_contours_mt_alphas.csv` (theory 1σ = model ⊕ SM pole ⊕ M_h),
`fig2a_ellipses.csv` (D1_direct_pole), measurements from `C154_summary.json` `marginal.*` (172.60 ± 0.27; σ-pole
172.1 ± 0.6; ATLAS+CMS 172.52 ± 0.33), FCC-ee from `future_reach.csv` `FCCee_base` (σ_meas 0.052, σ_tot 0.131; ellipse
centred at the present values with s_as = 0.0002 and κ = 100 GeV from `C154_inputs_v2_run05.json`), top axis = linear
pole → m̄_t map through the QCD four-loop points (`msbar_points`; agrees with `R3_mbar_run05.json` to 0.7 MeV);
λ(μ_b) = 0 line `generated/m0_lambda_mub_zero.csv` (171.092 GeV at α_s = 0.1180, = C154 `relative_form.m0`).

## prd_criterion (PRD; two columns, panels side by side)

**Caption (PRD, ≈85 words).** Survival criterion in the top-mass plane. (a) Single-branch survival probability
$P_1(m_t)$ for $\alpha_s=0.1171$, 0.1180 and 0.1189, with the theory $1\sigma$ (model, SM mapping, $M_h$) on the central
curve at $P_1=0.9$, 0.5, 0.1. The grey band is the direct measurement; bars show the $\sigma_{t\bar t}$ pole mass,
ATLAS+CMS, and the FCC-ee projection (thick: $m_t$; thin: total). Top axis: $\bar m_t(\bar m_t)$ at $\alpha_s=0.1180$.
(b) Critical lines in the $(m_t,\alpha_s)$ plane with their theory bands, the line $\lambda(\mu_b)=0$, and the 68% and
95% regions of the present data and of FCC-ee.

Note: the (m_t, ξ_H) plane is not repeated here; the PRD uses `prl_fig3_xi` (written to both figure folders).

---

## prl_fig3_xi (PRL Fig. 3; single column; also in the PRD)

**Caption (PRL, ≈60 words).** Non-minimal coupling required at $\mu_b$. The zero-mode threshold $\xi_c(m_t)$ (line;
points: direct calculations; dark and light bands: SM mapping and $\alpha_s\pm0.0009$) and the all-mode inflation bound
$\xi_{\rm inf}$ (dashed) bound $\xi_H$ from below, and $\xi_H\lesssim1$ from above. Survival requires $\xi_H$ in the
shaded region ($P_1\ge0.1$; dashed segment: extrapolation below $P_1=0.13$).

Data: run05 direct ξ_c (`claims_run05/xi_run05_summary.json` `run05.*`: 0.0346/0.0737/0.1195/0.1248 at
172.2/172.6/172.95/173.0, bars = prescription half-difference). Curve = universal ξ_c(P₁): monotone PCHIP in
ln(−ln P₁) through all 26 direct points (5 run05 + 5 run02 centre + 16 run02 SM/α_s variants of C157), max residual
0.0010, evaluated on the run05 P₁(m_t); covers P₁ = 0.9996–0.134 (m_t = 171.755–173.33 GeV); the dashed continuation to
P₁ = 0.1 (173.418 GeV) is a straight line in ln(−ln P₁) through the last two points (ξ_c(0.1) ≈ 0.150, estimate). Bands
= same function at m_t ∓ 0.102 GeV (SM pole) and at α_s = 0.1189/0.1171; they reproduce the direct C157 run02 bands to
≤0.0015. ξ_inf(μ_b) from `inflation_stability_run01/xi_inf_table.csv` (α_s = 0.1180). Shaded lower edge =
max(ξ_c, ξ_inf) (checked numerically; the two cross at m_t = 172.105 GeV); upper edge ξ_H = 1; 3/8 and 1/6 marked.
The `xi_c_*` columns of `fig2b_xi_curves.csv` (run02 model) and `xi_plane_mt_xiH.csv` are not used.

---

## prd_higgs_wavefunction (PRD; single column, two panels)

**Caption (PRD, ≈85 words).** Higgs and hypercharge zero modes in the two-dimensional model. Filled contours show the
probability density of the lowest state in the $(r,s=|\mathbf w|)$ plane (0.1–0.9 of the maximum); lines are
equipotentials of $V(r,s)$, the thick one at the eigenvalue $E$; the dotted curve is the transverse zero-point width
$\langle s^2\rangle^{1/2}$. (a) $\lambda=0$: bound ground state. (b) $\lambda=-0.013$: quasi-bound state. $V$ stays open
at small $s$; the barrier of the adiabatic channel comes from the zero-point energy of the gauge zero modes, and beyond
its outer turning point the state escapes along $r$.

Data: `generated/wavefunction_2Dex_lam0.npz`, `..._lamm0p013.npz` (prepare_data_higgs.py gen_wavefunction: T14a
same-wall FD solver on the D112 2D_ex operator, v11 couplings, h = 0.04/L; Dirichlet walls r_w = 9/L and 4.4/L, the
latter the T14a main wall). EL = 4.450 (C140 Numerov 4.451) and 4.286 (Siegert Re E 4.278; the wall raises it by
0.008, Γ L = 2.2×10⁻³). Top of U_ad at r = 4.04/L, outer turning point 5.2/L. Units 1/L.

## prd_gamma_validation (PRD; single column, two panels)

**Caption (PRD, ≈75 words).** Resonance widths of the reduced models and their full-dimensional check. (a) $\Gamma L$
against $|\lambda|^{-1/2}$, linear on the logarithmic scale as in WKB, for one- and two-dimensional models with Gaussian
or DMC gauge shape; dashed lines are the static thresholds $\Gamma_c=-\ln P_1/\eta_\infty$, and ticks mark each model's
static $\lambda_c$. (b) $\Gamma_{\rm full}/\Gamma_{\rm 2D}$ at $\lambda=-0.010$ and $-0.013$ from four estimators; all
lie within 20% (band) and shift $m_c$ by $+0.004$/$+0.007$ GeV.

Data: (a) `higgs_beyond_bo_run01/C140_summary.json` (`gamma.*`, `trend_lam_to_0.*`, `lam_c.*`, `eta` = 299.84; v11
couplings, MC+CW; static readout, not the hazard-equivalent λ_c of the PRL). (b) `fulldim_dmc_run01/C151_summary.json`
(`verdict.gamma_ratios`; DMC-shift bar = ratio × dlnΓ/dQ × se(ΔQ); tail-amplitude bar = se_A). Δm_c annotation (since
2026-10-16): final-model conversion of the deduplicated budget, `claims_run05/table1_rebuild.json` `t14a.central`
0.0046/0.0075 and `t14a.envelope` 0.0123/0.0202 GeV (= `table1_error_budget_run05.csv` row `fulldim_T14a`, checked in
the script); the earlier conversion 0.0043/0.0070 and 0.011/0.018 is no longer shown.

## prd_hazard_realtime (PRD; two columns, three panels)

**Caption (PRD, ≈85 words).** Survival along the turning and the role of $\xi_H$. (a) Hazard survival $P_1(\tau)$ for
$m_t=172.2$–173.4 GeV with the norm of the real-time evolution at $\xi_H=1/6$ (circles) and $\xi_H=0$ (crosses; arrow:
final value after the classical stage). (b) Relative loss $1-P_{\rm surv}(\xi_H)/P_{\rm surv}(1/6)$ from one-dimensional
(lines; solid and dashed: two $\lambda$ prescriptions) and two-dimensional (markers) real-time evolution; the 1% line
defines $\xi_c$ (ticks). (c) $\xi_c$ depends on $m_t$, $\alpha_s$ and the SM mapping only through $P_1$.

Data: (a) run05 node widths `holonomy_cross_run05/X_main_u*.json` (α_s = 0.1180), hL_b = 0.00437243 from
`universality_run01/C156_eta.json`; ln Γ cubic spline over 11 nodes; P₁(τ = 5) reproduces `C149_run05_summary.json`
`P1_grid` to <10⁻⁵; real-time norms `claims_run05/xi_run05/part2/p2_main_*.json` (run05, 1D); ξ_H = 0 final
P_surv = 0.150 at 172.6 GeV. (b) lines `claims_run05/xi_run05/C157_xi_quick.json` (run05, u_sw = 6), markers
`realtime_2d_run01/C157_twod.json` `P_compare.*.P2D_1L` (2D TDSE, **run02 centre**; ξ_c shift run02 → run05 ≈ −0.0004,
not visible); ticks = run05 `lam_1L`/`lam_phi`. (c) 26 direct points (5 run05, 5 run02, 16 run02 variants of C157)
and the curve of prl_fig3_xi (max residual 0.0010; dashed: extrapolation P₁ = 0.134 → 0.1).

## prd_cross_terms (PRD; two columns, two panels)

**Caption (PRD, ≈80 words).** Zero-mode cross terms and the refinement ladder of the critical mass. (a) Shift of $m_c$ by
the one-loop cross terms (bars: $P_1=0.5$; circles: 0.9; triangles: 0.1; blue stabilizing, red destabilizing): run05
split by gauge factor, the new-vertex groups contained in the total, and run01 single-factor values (v11 couplings, not
additive). (b) $m_c(P_1=0.5)$ along the model refinements; the early steps add distinct physics and do not form a
convergence sequence. The final value is 172.96 GeV.

Data: (a) `claims_run05/cross_decomp_summary.json` (D132 §6): SU(3)–top = `order_a_Y_SU2_SU3.SU3top_given_noY_noSU2`
(−0.032/−0.040/−0.048), U(1)_Y+SU(2) = `crit.full − crit.noY_noSU2` (−0.048/−0.078/−0.112; the two telescope exactly
to `total_full_minus_none` −0.080/−0.118/−0.160), `one_loop_groups` (+0.021/+0.028/+0.034; −0.017/−0.022/−0.027);
run01 ratio method `holonomy_cross_run01/C149_summary.json` `dmc_components_alone` (U(1)_Y −0.137/−0.195/−0.258,
SU(2) +0.078/+0.104/+0.134, SU(3)–top −0.033/−0.042/−0.052; P₁ = 0.1 extrapolated in the source). The run05 split of
U(1)_Y and SU(2) separately did not converge (D132 §6) and is not shown. (b) `twod_direct_sm_run01/C150_summary.json`
`convergence_table.rows` (recomputed values; "2D ex (ratio)" hatched = published D112 value only),
`twod_direct_sm_run02` main critical m_t (+NNLO, 173.077), `holonomy_cross_run02` with_cross (172.954), run04 `A.crit`
(172.950), run05 `crit` (172.960). Stages before "+ NNLO" use R14 couplings.

## prd_universality (PRD; single column, two panels)

**Caption (PRD, ≈70 words).** Universality of the finite-volume threshold. (a) $|\lambda_c|^{-1/2}$ grows linearly with
the logarithm of the conformal duration $\eta=f\eta_\infty$ for $P_1=0.9$, 0.5, 0.1 (circles: hazard thresholds; lines:
WKB from $f=1$). (b) Logarithmic derivatives of $|\lambda_c|$ with respect to $g_2$, $g_Y$ and $y_t$ at $\mu_b$ (bars)
and their WKB estimates (diamonds): the threshold is set mainly by $g_2$ and $y_t$, nearly independently of $P_1$.

Data: (a) `universality_run01/C156_eta.json` (`eta_scaling`, `wkb_vs_exact_duration`, η_∞ = 299.84); max |exact − WKB|
in |λ_c|^{−1/2} = 0.02/0.05/0.06. (b) `universality_run02/C156_run02.json` `derivatives.*` (0.663/0.163/0.627 at
P₁ = 0.5). Both on the **run02 centre** (not recomputed at run05; λ_c(f = 1) differs from run05 by ≤ 4×10⁻⁵).

---

## Notes for the authors

- PRL uses three single-column figures; `prl_fig3_shape` was not made (brief update 15:20Z). `prd_criterion` has two
  panels; the PRD uses `prl_fig3_xi` (also in `PRD/figures/`) for the (m_t, ξ_H) plane instead of a third panel.
- All text is ≥ 7.5 pt; mathtext sub/superscripts (e.g. tick exponents 10⁻⁴) render at 5.2–5.9 pt, as usual.
- Panels on the run02 centre (stated in each Data line): 2D real-time markers in prd_hazard_realtime (b), C157 variants
  inside the universal ξ_c curve, prd_universality, and the v11-coupling reduced-model panels (prl_fig1 a,
  prd_higgs_wavefunction, prd_gamma_validation, cross-term run01 rows, ladder stages before + NNLO).

---

## 2026-10-16 revision: internal labels removed from the figures; caption sentences to update in the .tex

The figures no longer contain internal run, note or code numbers (checked by text extraction of all ten PDFs). Changed
figure text: prd_cross_terms (a) "run01, one at a time (v11)" → "first estimate, one factor at a time"; (b) stage names
"MU, MU+CW, MC+CW, 1D ex, 2D G, 2D ex (ratio), U_eff hazard, 2D direct, + NNLO SM, + cross terms, run04, run05" →
"unitary gauge, + CW potential, Coulomb gauge, 1D + DMC shape, 2D Gaussian, 2D + DMC (rescaled), calibrated 1D,
2D direct, NNLO SM mapping, + cross terms, + vev, top vertices, + Coulomb (final)"; header "(R14 couplings)" →
"(earlier SM mapping)". prd_hazard_realtime (c) legend "run05 / run02" → "final model / earlier model".
prd_gamma_validation (b) annotation now "Δm_c = +0.005 / +0.008 GeV (envelope +0.012 / +0.020; P₁ = 0.9 / 0.5)".

Caption sentences that are no longer accurate (old → new):

1. prd_gamma_validation: "…all lie within 20% (band) and shift $m_c$ by $+0.004$/$+0.007$ GeV." →
   "…all lie within 20% (band) and shift $m_c$ by $+0.005$/$+0.008$ GeV (envelope $+0.012$/$+0.020$ GeV)."
2. prd_cross_terms: "…run05 split by gauge factor, the new-vertex groups contained in the total, and run01 single-factor
   values (v11 couplings, not additive)." → "…the final model split by gauge factor, the new-vertex groups contained in
   the total, and a first estimate taking one gauge factor at a time with the earlier couplings (not additive)."
3. prd_hazard_realtime, prd_cross_terms (b): any .tex caption wording "run02/run05" or "run04" for the plotted points
   should read "earlier model / final model" and "+ vev mixing and top chiral terms / + Coulomb mismatch (final)".
