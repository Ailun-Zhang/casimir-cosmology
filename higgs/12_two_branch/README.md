# 12_two_branch — survival of both branches through the turn (D130 / C159)

**Physics.** The turn is time-symmetric; the Gell-Mann–Hartle two-time history set (in/out at −T and +T, conditioned
on the turnaround state ψ₀) shows maximal interference (ε = 0.9948–0.99997): "both branches survive" has no decohering
history probability, only Fréchet bounds max(0, 2P₁ − 1) ≤ P₂ ≤ P₁. The unique time-symmetric linear (Goldstein–Page
/ Margenau–Hill) value is P₂^lin = P₁²/w₀; if adopted, the 50% (90%) critical masses move down by 0.19 (0.12) GeV:
172.376/172.759/173.124 for P₂ = 0.9/0.5/0.1 (run02 base; run05: 172.380/172.764/173.131, see `09_claims`).
The GH diagonal D(in, in) = P₁³/w₀² is not a probability.

**Method.** `tb_tdse.py` (TDSE through the turn on the D111 potential tables; GH functional, linear two-time values,
several initial states and u_T), `tb_static.py` (static-spectrum propagation cross-check), `tb_post.py` (C159
summary, critical-mass tables). The D111 tables (`r2/higgs_realtime/run01/tables/U_table_MU_mt*.npz`, 3 × 1.7 MB)
are rebuilt by `../zmlib/hr_potential.py build` (≈ 12 s); note they use the round-1 MU potential, as in the
original C159 run (the two-branch laws are model-independent identities checked numerically).

**Status.** Two-branch lemmas (time reflection, [E₊, E₋] ≠ 0, uniqueness of the linear measure) proven within the
model + numerics; only production-internal TDSE vs static-spectrum comparison (no blind check).

**Run.**
```
python run.py --quick              # ~30 s: tables + static check + summary from the archived TDSE shards
python run.py --full               # + tb_tdse.py full (2 workers inside)
```
Original code: `04_calculations/code/20261014_claude_v12_r3/two_branch/`; results `…/results/20261014_claude_v12_r3/
two_branch/run01`; derivation `04_calculations/derivations/20261014_claude_v12_r3/two_branch/D130_TWO_BRANCH_CN.md`.

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T02:04:35Z | 3/3 | 0 | D111 potential tables 17; tb_static.py 42; tb_post.py 1 |
| full | 2026-10-15T19:40:02Z | 4/4 | 0 | D111 potential tables 15; tb_static.py 36; tb_tdse.py full 358; tb_post.py 0 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL one sentence; PRD Sec. V D, App. D 2 | P₂ = P₁² critical 172.376/172.759/173.124 (run02 base) | `r3/two_branch/run01/C159_summary.json` : `headline/crit_as0p1180` | same archived file |
| PRD Sec. V D, App. D 2 | GH interference ε 0.9948–0.99997 | same : `headline/GH_interference_eps` | same |
| PRD Sec. V D, App. D 2 | Fréchet bands [172.372, 172.498], [172.713, 172.954] | same : `crit` | same |
| PRD Sec. V D, App. D 2 | static-spectrum check | `r3/two_branch/run01/static_check.json` : `runs` | same |
