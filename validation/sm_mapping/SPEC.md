# V1 — Standard Model mapping: specification given to the blind reimplementation

Summary of the written specification of 2026-10-14, reduced to definitions, inputs and pass criteria. Report: Sec. 19.4.

**Blindness.** No access to the derivations, code, results or literature downloads of the main calculation, or to the
records of other checks; public literature and public codes (downloaded by the check itself) allowed. Results were
written to `BLIND_RESULTS.json` (time stamp, code SHA-256) before the comparison.

**Inputs (PDG 2026).** M_t = 172.60 GeV interpreted as the pole mass (also a grid 171.8–173.8 GeV in steps of 0.2),
α_s(M_Z) = 0.1180 (also 0.1171, 0.1189), M_h = 125.13 GeV, M_W, M_Z, G_F (or α(M_Z)) from PDG 2026;
μ_b = M̄_P/722 with M̄_P = 2.435×10¹⁸ GeV; hazard nodes μ(u) = μ_b/√cosh 2u, u = 0, 0.5, …, 5.

**To compute.** (1) MS-bar λ, y_t, g₁ (g_Y = √(3/5) g₁), g₂, g₃ at μ = M_t with full two-loop threshold corrections
(including O(α α_s), O(α²); y_t with four-loop QCD pole↔MS-bar), run with three-loop SM RGEs (plus known higher QCD
terms) to μ_b and the nodes; normalization V = λ(|H|² − v²/2)². (2) Theory errors, all recomputed: matching scale
μ₀ ∈ [M_t/2, 2M_t] with the thresholds evaluated at μ₀; truncation (RGE loops, four/five-loop QCD, pole-mass series);
scheme (tadpoles, input set); expressed as changes of λ(μ_b), y_t(μ_b) (optionally as M_t equivalents).
(3) m̄_t(m̄_t) (four-loop QCD) for the input M_t.

**Independence requirement.** At least one matching implementation other than SMDR (e.g. mr of Kniehl, Pikelner and
Veretin, or the published two-loop threshold formulae); SMDR may serve as a second implementation; the three-loop β
functions checked term by term against a published source; at least two literature benchmarks reproduced (e.g. the
Planck-scale λ of Buttazzo et al. Eq. (61), the critical M_t of Bednyakov et al. 2015, or SMDR reference points).

**Required outputs.** λ, y_t, g₂, g_Y, g₃ at μ_b for (172.60, 0.1180, 125.13); dλ(μ_b)/dM_t, dλ(μ_b)/dα_s; the
node tables; the error items; the benchmarks; the m̄_t correspondence.

**Pass criterion.** Agreement of λ(μ_b) with the main mapping within the Standard Model error of the papers
(±0.10 GeV in the pole mass, i.e. about ±7×10⁻⁴ in λ(μ_b)); differences to be traced to their origin.

Note: `code/drivers/c148_fit_copy.c` (a copy of the package driver `higgs/01_sm_mapping/c148_fit.c`, which contains
SMDR code, GPL-2.0-or-later) and `results/DIAGNOSIS.json` belong to the diagnosis made **after** the comparison
(the matching-scale variation reproduced bit for bit); see `third_party/README.md` for the SMDR licence.
