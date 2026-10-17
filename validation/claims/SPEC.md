# V3 — claims layer and robustness of the inputs: specification given to the blind reimplementation

Summary of the written specification of 2026-10-15, reduced to definitions, inputs and pass criteria. Report: Sec. 19.5.

**Blindness.** No access to the derivations, code or results of the claims layer and of the robustness calculation of
the main work. Allowed as input data: the table P₁(m_t, α_s) of the third-stage centre model (the statistics layer
downstream of it is what is checked, not P₁ itself), the independent two-dimensional implementation of the keystone
check (`../keystone/code/`), the earlier draft, the literature. Results in `BLIND_RESULTS.json` before the comparison.

**To compute.** (1) Marginalization P̄₁ = ∫ P₁(m_t, α_s) p(m_t) p(α_s) dm_t dα_s for three mass definitions: direct
(Monte Carlo) 172.60 ± 0.27 GeV (and a variant with a 0.5 GeV MC–pole interpretation error), σ-pole 172.1 ± 0.6 GeV,
MSR identification (m_MC ≡ m_MSR(1.3 GeV), m_pole − m_MSR = 0.36 ± 0.25 GeV); α_s = 0.1180 ± 0.0009; SM-mapping
error ±0.086 GeV (pole, as specified then); the model-error budget supplied with the specification; P̄₁ with 68 % and
95 % ranges. (2) Bayes factors of "the universe began at a Casimir turning point and the Higgs zero mode survived"
against the same without the survival condition, flat priors on [170, 176] and [150, 200] GeV, existence as data
and as condition. (3) m_t at P₁ = 0.1 for α_s = 0.1180 ± 1σ; the measured centre needed for a 95 % exclusion at present
precision; the separation of true values 172.6 and 173.3 GeV with the HL-LHC and FCC-ee precisions read from the
original reports (with page numbers). (4) Input sensitivity with an independent implementation: hL_b × 0.9 and × 1.1;
the change of (L_b, h) and Δm_c for a zero-mode share 0.28 instead of 0.2999. (5) The preparation weight w₀ of the
time-symmetric minimum-energy-variance state in the new model.

**Pass criterion.** Agreement with the main claims layer within the stated precision (P̄₁ to 0.005, thresholds to
0.01 GeV); every difference traced (the four corrections it caused are listed in the report).
