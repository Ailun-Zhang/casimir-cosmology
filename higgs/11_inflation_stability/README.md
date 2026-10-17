# 11_inflation_stability — all-mode Higgs stability during inflation (D127 / C155)

**Physics.** The v11 plateau is a minimally coupled scalar with a Starobinsky-shaped potential (no R² term), so the
field-theory bound of Mantziris–Markkanen–Rajantie 2021 applies (reproduced: ξ_EW = 0.05946 vs 0.05938); the MMR 2022
R² bound (ξ_EW ≳ 0.1) does not. RG-improved potential V = ½ξ(μ*)Rh² + ¼λ(μ*)h⁴ (+ αR²/144), Hawking–Moss rate
Γ = (R/12)² e^{−B}, B = 384π²ΔV/R², past-light-cone count ⟨N⟩ = 1; plateau h/M_P = 6.116×10⁻⁶. Result:
ξ_inf(EW) = 0.04877/0.05512/0.06044 and ξ(μ_b) = 0.03030/0.03781/0.04413 at m_t = 172.2/172.6/173.0 (α_s 0.1180).
Compared at the same scale, the zero-mode condition (07) is stronger for m_t ≳ 172.2 GeV.

**Method.** `infl_core.py` (potential, running ξ with 16π² d ln Ξ/d ln μ = 12λ + 6y_t² − 3/2 g′² − 9/2 g², HM
exponent, light-cone counting; SM couplings from `../zmlib/sm_provider.py`), `run_repro.py` (MMR 2021/2022
reproduction), `run_main.py` (95-job scan over m_t, α_s), `build_plane.py` (ξ_inf table, C155 summary).

**Caveat (INV contradiction 19).** `xi_plane_mt_xiH.csv` uses `inputs/xi_c_zero_mode.json` = round-2 zero-mode values
(header "R3 values pending"); do not plot it — the current ξ_c(m_t) is in `07_realtime_xi` / `09_claims` fig2b data.

**Run.**
```
python run.py --quick                 # ~1 min: reproduction + table/summary from the archived scan shards
python run.py --full --workers 15     # + the 95-job scan
```
Original code: `04_calculations/code/20261014_claude_v12_r3/inflation_stability/`; results `…/results/
20261014_claude_v12_r3/inflation_stability/run01`; derivation `04_calculations/derivations/20261014_claude_v12_r3/
inflation_stability/D127_INFLATION_HIGGS_STABILITY_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:03:35Z | 5/5 | 0 | run_repro.py 92; build_plane.py 0 |
| full | 2026-10-15T19:40:12Z | 6/6 | 0 | run_repro.py 110; run_main.py 15 165; build_plane.py 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL one sentence; PRD Sec. V H | ξ_inf(EW) 0.04877/0.05512/0.06044 (α_s 0.118) | `r3/inflation_stability/run01/xi_inf_table.csv` : column `xi_inf_ew`, rows `mt`, `als` | same archived file |
| PRD Sec. V H | ξ_inf(μ_b) 0.03030/0.03781/0.04413 | same : `xi_inf_mub` | same |
| PRD Sec. V H | α_s = 0.1171/0.1189 rows | same | same |
| PRD Sec. V H | MMR 2021 reproduction 0.05946 | `r3/inflation_stability/run01/repro/repro_step1.json`; `C155_summary.json` : `reproduction` | same |
| PRD Sec. V H | all rows | `C155_summary.json` : `xi_inf_rows` | same |
