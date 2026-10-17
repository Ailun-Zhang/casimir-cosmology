# 04_cross_terms — zero-mode / non-zero-mode cross terms and the run05 node widths (D121 / C149, run02 → run05)

**Physics.** With the periodic holonomy dependence of all SM fields (AAA fermions) the net effect of the cross terms
is destabilising: the critical mass moves by −0.080/−0.118/−0.160 GeV (P₁ = 0.9/0.5/0.1) relative to the 2D direct +
NNLO baseline (03). Mechanism: the r-independent hypercharge curvature (light fermions) times the w-width that
narrows with r lifts the well more than the barrier; SU(3)–top destabilises; SU(2) stabilises. In the broken
direction the one-loop r²c² set is complete in run05: background-covariant Coulomb-gauge vev mixing (transverse log
= 4/3 × required) + instantaneous Coulomb mismatch (−1/3) + the exact top chiral bubble; the logs close the RG
(0.994/0.973/0.993/0.990 at r = 2/3/4/5), so μ = 1/L is a consistent matching scale.

**Method.** At each of the 11 nodes and each (m_t, α_s) column: one-loop Hessians H₁₁, H₃₃, H₃w, H_ww(r) per field
(`../zmlib/cross_kappa.py`), SU(2)/U(1) variational re-minimisation (`../zmlib/cross_var.py`), exact light-fermion
twist V_L(r, s) (`../zmlib/cross_2d.py`), SU(3)–top variational energy, then the 2D Siegert roots (`../zmlib/
C149_direct.py`, run02). `C149_top.py` (exact top chiral bubble → `run04/C149_top_chiral.json`), `C149_direct4.py`
(run04: vev mixing + top chiral), `C149_coulomb.py` (run05: covariant vertex + Coulomb mismatch; RG closure),
`C149_direct5.py` (run05 = PRL centre: `run main|als`, `post`, `final`).

**Status.** Image sums, curvature, AAA pinning, Ω_X² > 0, top-chiral log B/(q_Φ² m_t²) = 1.000 ± 0.001, transverse
4/3: proven. CCA, conditional Gaussian, W-direction Coulomb via the custodial ratio (±25%), neutral Coulomb
coefficient fit (6%): conditional. Model-internal error ±0.008/±0.010/±0.013 GeV (D121 run05 §3). Blind check V2
covers the CCA part (run02 level); the run03–05 additions (net +0.005/+0.006/+0.007 GeV) have no blind check.
`cross_decomp.py` (C161 step 6, added for R18; path block adapted) splits the run05 shift by zero-mode field
(U(1)_Y, SU(2), SU(3)–top in two removal orders; vev mixing + Coulomb; top chiral): `run` computes 66 node jobs
(archived in `expected/w/claims_run05/run01/cross_decomp/shards/`, not rerun by `run.py`), `post` the split, which
`run.py` recomputes in quick and full mode from those shards and the run05 summaries of the run (check
`cross_decomp_post`, bit for bit).

**Run.**
```
python run.py --quick                  # ~1.5 min: top chiral + RG closure + one node job (u = 0) + post/final
python run.py --full --workers 11      # ~12 min: all 22 run05 node jobs from scratch (incl. gauge profiles)
```
Original runtime: run05 main 8 min + als 9 min with 3 workers (Intel Core Ultra X9 378H). Re-computation in this
package (2026-10-15, 11 workers, profiles recomputed): 701 s, all 22 shards bit-identical, critical masses identical.
Original code: `04_calculations/code/20261014_claude_v12_r3/holonomy_cross/`; results `…/results/
20261014_claude_v12_r3/holonomy_cross/run02, run04, run05`; derivation `04_calculations/derivations/
20261014_claude_v12_r3/holonomy_cross/D121_HOLONOMY_CROSS_TERM_CN.md` (appendices run02–run05). Superseded and not
packaged: run01 ratio method (C149_report/final, stage A–D), run03 first vev-mixing run (factor-4 error).

## Verification
<!-- VERIFY-BLOCK-START -->
| mode | date (UTC) | checks passed | skipped | timings (s) |
|---|---|---|---|---|
| quick | 2026-10-17T01:54:03Z | 15/15 | 0 | C149_top.py 13; C149_coulomb.py 4; C149_direct5.py test 24; C149_direct5.py post 4; C149_direct5.py final 4; cross_decomp.py post 4 |
| full | 2026-10-15T14:57:04Z | 16/16 | 0 | C149_top.py 15; C149_coulomb.py 9; C149_direct5.py test 46; C149_direct5.py run main 353; C149_direct5.py run als 272; C149_direct5.py post 4; C149_direct5.py final 3 |

Machine of record: Intel64 Family 6 Model 204 Stepping 2, GenuineIntel, 16 CPUs, Python 3.12.13, numpy 2.5.1, scipy 1.17.1.
<!-- VERIFY-BLOCK-END -->

## Results → paper
| Paper location | Number | Package output (work tree) : key | Archived source |
|---|---|---|---|
| PRL Table I; PRD Sec. V E–F | m_c^pole 172.5024/172.9595/173.4184 (α_s 0.1180) | `r3/holonomy_cross/run05/C149_run05_summary.json` : `crit/0.1180/P1=*` | `r3/holonomy_cross/run05/C149_run05_summary.json` : `crit.0.1180` |
| PRL Fig. 2(a); Table I | α_s 0.1171: 172.156/172.611/173.069; 0.1189: 172.849/173.308/173.768 | same : `crit/0.1171`, `crit/0.1189` | same |
| PRL text | P₁(172.60) = 0.8426; P₁(172.52) = 0.8909 | same : `P1/172.6`, `P1/172.52` | same |
| PRL Fig. 2(a) | P₁ grid 172.0…173.8 | same : `P1_grid` | same |
| PRL headline | λ_c(μ_b) −0.0091339/−0.0121415/−0.0151849 | same : `lam_c` | same |
| PRL Table I (cross-term row) | −0.0802/−0.1180/−0.1600 vs no cross term | same : `shift_vs_T14b/0.1180` | same |
| PRD Sec. V C, App. C 4 | RG closure 0.994/0.973/0.993/0.990 | `r3/holonomy_cross/run05/C149_coulomb_check.json` : `<r>/closure_Z` | same archived file |
| PRD Sec. V C, App. C 4 | top chiral G(m_t²), B/(q_Φ² m²) | `r3/holonomy_cross/run04/C149_top_chiral.json` | same archived file |
| PRD Sec. V C–D, App. C 4 | node widths γ(m_t, u) with cross terms | `r3/holonomy_cross/run05/shards/X_{main,als}_u*.json` : `cols/*/gamma` | same archived shards |
| PRL main text and End Matter A (field-by-field split); PRD Table V, row "Further cross terms"; report Sec. 8.6 | split of the run05 cross-term shift: total −0.0802/−0.1180/−0.1600 GeV; SU(3)–top −0.0403, vev mixing + Coulomb +0.0275, top chiral −0.0219 GeV at P₁ = 1/2 (R5-049…053) | `w/claims_run05/run01/cross_decomp/cross_decomp_summary.json` (`cross_decomp.py post`) | `expected/w/claims_run05/run01/cross_decomp/` (+ 66 node shards; C161 step 6) |
