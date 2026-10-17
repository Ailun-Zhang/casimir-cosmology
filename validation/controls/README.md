# validation/controls/ — numerical controls quoted from the derivation notes

Added for R19 (N4): the numerical controls that the papers and the report quote from the derivation notes are archived
in the package. Code and outputs are byte-identical copies (SHA-256, sizes and times in `../SOURCES.json`); they are
archived outputs and are not rerun by `validation/run.py`.

| control | quoted in | archived here | result |
|---|---|---|---|
| Exterior complex scaling (ECS) vs the Siegert solver of the resonances, same inner wall (D108 sec. 2.2) | PRL End Matter A; report Secs. 7 and 20 ("Γ to 3×10⁻⁷, ε to 3×10⁻⁹") | `D108_ecs/code/C136_stage1.py` (phases `ecs`, `ecsfix`) with `C136_core.py`; outputs `D108_ecs/results/C136_s1_ecs*.json` | `C136_s1_ecsfix.json` → `same_inner_wall/*`: max \|ΔE\| = 3.0×10⁻⁹, max \|Δγ/γ\| = 3.1×10⁻⁷ (4 cases); ECS invariant under θ = 0.25/0.40 and R₀ = 6.5/8 to 10⁻¹⁰ (`C136_s1_ecs.json`) |
| Inversion of the soft hypercharge direction with Higgs fluctuations only (D112 sec. 6.1, quoted in D121) | PRD Sec. V C; report Sec. 8 ("r < 0.776") | `../higgs/03_zero_mode_2d/expected/r2/higgs_beyond_bo/run01/C140_cross_note.json` (with `C140_crossX.json`) | `r_X_inverted_below` = 0.774 (λ = −0.008), 0.776 (−0.010, −0.0143); the blind check V2 finds the same with all fields removed but the Higgs ("unstable for r ≲ 0.78", `../keystone/results_v2/BLIND_RESULTS_V2.json` → `stability`) |
| Width of the one-dimensional BO reduction (lowest order) against the two-dimensional model in the final model (D129 sec. 0.1) | PRD Sec. V B; report Sec. 7 ("1.0–1.8 %" below, ratio 1.010–1.018) | already archived: `../higgs/07_realtime_xi/expected/r3/realtime_2d/run01/C157_calib.json` | `A0/<m_t>|<u>/rel` = −0.0104 … −0.0177 over the 55 nodes |
