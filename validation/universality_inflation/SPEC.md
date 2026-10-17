# V5 — universality of λ_c and the bound during inflation: specification given to the blind reimplementation

Summary of the written specification of 2026-10-15, reduced to definitions, inputs and pass criteria. Report: Sec. 19.7.

**Blindness.** No access to the derivations, code or results of the inflation-stability and universality calculations
of the main work. Allowed: the earlier draft, the literature (Mantziris, Markkanen and Rajantie 2021, 2022; Herranen
et al. 2014, 2015), and the independent two-dimensional implementation of the keystone check (`../keystone/code/`;
copied to `code/kb_copy/`). Results in `BLIND_RESULTS.json` before the comparison.

**Part 1: lower bound on ξ_H during inflation.** Plateau inflation of the draft, V(φ) = 3M_P²h²(1 − e^{−√(2/3)φ/M_P})²,
h fixed by A_s (h and N_post of the draft); non-minimal coupling ξ_H defined at the electroweak scale and at
μ_b = M̄_P/722 (both readings); own two- or three-loop SM running (PDG 2026 inputs, M_t = 172.2/172.6/173.0 GeV,
α_s = 0.1180, M_h = 125.13 GeV). Criterion: expected number of nucleation events in the past light cone ⟨N⟩ < 1
(method and curved-space RG-improved potential of Mantziris et al. 2021). First reproduce a published value of that
paper, then give ξ_inf(m_t).

**Part 2: scaling of λ_c.** With the keystone implementation or an own one: λ_c at P₁ = 0.5 for hL_b × 0.5, 1, 2 and
for g₂, y_t changed by ±10 %; dλ_c/d ln η_∞ and the coupling derivatives.

**Pass criterion.** ξ_inf within 0.002 of the main calculation after the reproduction offset is accounted for;
derivatives consistent with the model difference (static second-stage model vs final hazard).
