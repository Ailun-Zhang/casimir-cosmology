# 05_moyal_wdw — SM Moyal correction, shear criterion with a quantum volume, B1 Wheeler–DeWitt test, B3 width lemma

**Physics.** (i) The leading quantum-minus-truncated-Wigner (Moyal) correction to the fossil variance of the SM shape
model is −2.03e-4 (aspect) and −1.08e-3 (angle) relative for the state real at the turn (−1.79e-4 / −9.2e-4 at next
order; −1.0 % in the toy window of v11). (ii) A quantum volume smooths the shear criterion ρ < 3^{−3/2} = 0.19245 only
by Δρ = 7.8e-5. (iii) In a minisuperspace WDW model with one aspect mode the turning-point state reconstructed from the
expanding branch is real to D_re = 2.4e-6 at X = 2282.49 for barrier-decaying (Ai-type) data, scaling ≈ X^{−2.4}; the
R15 counterexample fails exactly as Born–Oppenheimer predicts. (iv) B3: among centred Gaussians of the inverted
oscillator the energy variance is minimal (k/2m) for the real state of natural width 1/(2m|ω|).

**Method.** `C160_moyal.py` (D131 §1; C40 toy regression; `1d`: exact one-mode quantum vs TWA), `C160_shear_wdw.py`
(§3), `C160_b1_wdw.py` (§4; `zs0` = data at the turning point) and `b1_cases.py` (new driver: the valid z_s = 2.5 cases
one at a time with the production functions — the production z_s = 2.5 run was stopped in the contaminated case B4;
expected values = archived runs re-keyed in `expected/.../b1_cases_rekeyed.json`), `C160_summary.py`,
`C141_minvar_lemma.py` (SymPy; D113 §3). Status: Moyal and B1 numerical in the stated models; B3 proved for the
centred Gaussian family.

**Run.** `--quick` (~10 min): C141 (+ text comparison), Moyal (~110 s), shear (~240 s), B1 cases at X = 100 (three
cases) and the physical X = 2282.49 Ai-type case (~220 s), summary. `--full`: plus Moyal 1D (1057 s originally), the
z_s = 0 B1 set (1106 s) and all 15 valid z_s = 2.5 cases.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:16:10Z | 9/9 | 0 | C141 minvar lemma 4; C141 text check 0; C160 Moyal (SM) 101; C160 shear criterion with quantum volume 209; B1 WDW selected cases 195; C160 summary 0 |
| full | 2026-10-15T23:17:28Z | 23/23 | 0 | C141 minvar lemma 2; C141 text check 0; C160 Moyal (SM) 54; C160 shear criterion with quantum volume 112; C160 Moyal exact 1D check 563; C160 B1 WDW z_s = 0 set 998; B1 WDW z_s = 2.5 cases 753; C160 summary 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | number (id) | output : key | archived source |
|---|---|---|---|
| PRD Sec. VII E (corrections) | Moyal −2.03e-4 / −1.08e-3 (ξ = 0.30); next order −1.79e-4 / −9.2e-4 (FOS-08..10) | `r3/shape_tier2/run01/moyal/C160_moyal.json` : `SM/xi030/rows/0/rel/a_Heff/{aspect,angle}`, `SM/next_order/...` | `04_calculations/results/20261014_claude_v12_r3/shape_tier2/run01/moyal/C160_moyal.json` |
| PRD App. E 2 / PRD Sec. VII E | exact 1D quantum − TWA (FOS-13 analogue) | `.../moyal/C160_moyal_1d.json` : `checks[*]` | `.../moyal/C160_moyal_1d.json` |
| PRD App. E 2 (shear criterion) | Δρ = 7.8e-5 (NL-17) | `r3/shape_tier2/run01/shear/C160_shear_wdw.json` : `physical`, `fits` | `.../shear/C160_shear_wdw.json` |
| PRD Sec. VII D (`sec:s-equivalent`, Eq. `eq:s-WDW`) | D_re = 2.388e-6 at X = 2282.49; 3.1e-2 / 4.3e-4 / 1.5e-5 at X = 100 / 300 / 1000 (TS-01, 02) | `r3/shape_tier2/run01/b1/b1_cases.json` : `cases/A_Ai_real@<X>/rows/0/D_re` | `.../b1/C160_b1_wdw.json` (`runs[k]`) |
| PRD App. E 2 | R15 counterexample D_re 0.969 (ε = 1), 0.161 (ε = 0.3) (TS-03); (Ai+0.7Bi)⊗χ₀ 5.0e-4 (TS-04) | `b1_cases.json` / `.../b1/C160_b1_wdw_zs0.json` : `runs[*]` | `.../b1/` |
| PRD Sec. VII D; App. E 2 | B3: Var H_min = k/2m at width 1/(2m\|ω\|) (TS-05) | stdout `logs/05_moyal_wdw/C141_minvar_lemma.out.txt` | `.../20261013_claude_v12_r2/time_symmetry_origin/run01/C141_minvar_lemma_output.txt` |
| number keys of the papers | all parts | `r3/shape_tier2/run01/C160_summary.json` | `.../shape_tier2/run01/C160_summary.json` |
