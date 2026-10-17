# 10_universality — how universal is λ_c? (D128 / C156)

**Physics.** Near threshold ln γ = a − b|λ|^{−1/2} (linear binding + quartic runaway, WKB), so |λ_c|^{−1/2} is linear
in ln η and the transit duration enters only logarithmically: |λ_c(η)|^{−1/2} = |λ_c(η₀)|^{−1/2} + ln(η/η₀)/b_eff,
b_eff = 1.3992/1.2991/1.1893 (P₁ = 0.9/0.5/0.1). Coupling derivatives ∂ln|λ_c|/∂ln(g₂, g_Y, y_t) = 0.66/0.16/0.63
(P₁ = 0.5): local power law |λ_c| ≈ 0.0121 (η/300)^{−0.17} (g₂/0.528)^{0.66} (g_Y/0.443)^{0.16} (y_t/0.430)^{0.63}.
Spin structures: the PPA/PAA zero-mode vacuum is the SU(3) centre holonomy (N_C 21.07/23.67); 't Hooft flux sectors
need |Φ_Y| ≥ 2π (Z₆), classical flux energy 100.7/L = 2.36 × the AAA Casimir energy — they cannot turn.
⇒ the existence and size (≈ −0.01) of λ_c are universal; the exact value depends on the sector and SM inputs (±10–20%).

**Method.** `U156_eta.py` (η scaling from the run02 node widths), `U156_coup.py` (2D direct hazard with g₂, g_Y, y_t
× (1 ± 0.1) at fixed λ(μ) trajectory; 39 node jobs), `U156_twist.py`, `U156_hol2.py`, `U156_centre.py` (sectors;
closed forms checked in Wolfram, receipts archived), `U156_summary.py`, `U156_fix2.py` (run02: large-r repair of the
cross-term functional for g_Y/g₂ ≲ 0.75; derivatives change ≤ 0.2%). Base: run02 pipeline (as in D128).

**Status.** Scaling form, coupling derivatives, twist sums, Z₆ flux bound: proven / numerically checked within the
model; PPA/PAA λ_c not computed (two-dimensional model invalid there; indicative Δm_c ≈ +0.08/+0.02). Blind check V5
PASS (round-2 static version differs by 10–20%, as expected).

**Run.**
```
python run.py --quick                  # ~1.5 min: eta, twist, hol2, centre recomputed; summary + run02 post
python run.py --full --workers 15      # + U156_coup run/post (39 node jobs, at most 3 workers inside) + U156_fix2 run
```
Original code: `04_calculations/code/20261014_claude_v12_r3/universality/`; results `…/results/20261014_claude_v12_r3/
universality/run01, run02` (profiles 40 MB not archived); derivation `04_calculations/derivations/
20261014_claude_v12_r3/universality/D128_UNIVERSALITY_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:02:02Z | 7/7 | 0 | U156_eta.py 1; U156_twist.py 94; U156_hol2.py 11; U156_centre.py 9; U156_summary.py 2; U156_fix2.py post 5 |
| full | 2026-10-15T20:56:00Z | 8/8 | 0 | U156_eta.py 1; U156_twist.py 73; U156_hol2.py 9; U156_centre.py 7; U156_coup.py run 15 2461; U156_coup.py post 6; U156_fix2.py run 982; U156_summary.py 1; U156_fix2.py post 3 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRD Sec. V G | b_eff 1.3992/1.2991/1.1893; dλ_c/d ln η 0.00124/0.00206/0.00315; dm_c/d ln η −0.1898/−0.3113/−0.4732 | `r3/universality/run01/C156_summary.json` : `eta` | same archived file |
| PRD Sec. V G | ∂ln\|λ_c\|/∂ln(g₂, g_Y, y_t) 0.663/0.163/0.627 | same : `couplings` | same |
| PRD Sec. V G | N_C 13.124/20.871 → 21.068/23.669 | `C156_centre.json` | same |
| PRD Sec. V G | flux sector 100.719/L, ratio 2.3565 | `C156_twist.json` | same |
| report Sec. 12 | run02 repair | `r3/universality/run02/C156_run02.json` | same |
