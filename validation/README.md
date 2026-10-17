# validation/ — independent reimplementations, report checks and their comparison

The PRL, the PRD and the technical report state that every number they quote is reproduced by the scripts of this
package or contained in its archived outputs, **including those of the independent reimplementations**. This folder
holds those reimplementations: their specifications, code (byte-identical, with the SHA-256 recorded before the
comparison), results, and a script that compares them with the main calculation. Added for the R18 review (R1, MH-02).

```
python validation/run.py --quick      # about 5 s: comparison (22 items + 1 anchor regression) + rerun of the report checks; exit 1 on failure
```
`run.py` runs `compare_blind.py` (each blind number is looked up in the archived blind results, each main number in
the archived package outputs (all 22 main values are package outputs; the r = 0 holonomy curvatures, which the V2
specification supplied as the target to reproduce, are computed from the archived Epstein constants with PRD
Eq. h-kappas and reported as an anchor regression), and the difference is checked against the pass criterion;
table: `COMPARISON.md`, details with the location of every number: `output/validation/comparison.json`) and reruns
the report's own checks in `report_checks/` against their archived output. `run_all.py` calls it. The blind codes
themselves are **archived outputs**: they are not rerun by the package (they need their own builds of mr, SMDR and
TSIL, the original Wolfram/Python environment and hours of DMC and hazard runs).

## Levels of independence (report Sec. 19.1)
- **Blind** (keystone, V1–V6): a written specification with the model definitions, the inputs and the pass criteria,
  an earlier draft of the paper as external anchor, and the literature; no access to the derivations, code and
  results of the main calculation or to the other checks; results recorded with a time stamp and the SHA-256 of the
  code before any main number was received. `*/SPEC.md` summarizes each specification (definitions, inputs, pass
  criteria; internal paths and workflow instructions removed).
- **Compute first** (R14, R15): own numbers computed before opening the main results; some reused earlier independent
  code that the main calculation also uses, and one (R15, two-dimensional model) saw result tables first (partly
  independent).
- **Semi-blind** (T19): claims and non-numerical parts of the derivations only, own results saved, then compared; one
  accidental exposure of a few table rows is recorded in its report.

## Contents and where the papers use them
| folder | check (level) | what it computed | quoted in | reproducible here |
|---|---|---|---|---|
| `keystone/` (`code/`, `results/`) | Keystone (blind) | reduced Higgs chain of the second stage: matching, 2D (r, s) model, resonances, hazard, own DMC shape; m_c = 172.521/173.017/173.520 GeV vs 172.518/173.014/173.515 | PRL main text and End Matter A ("critical masses without cross terms to 0.005 GeV"); PRD Table V, row "Full chain, no cross terms" (footnote a); report Sec. 19.2, Table 30 | comparison (11 items) from the archived results |
| `keystone/` (`code_v2/`, `results_v2/`) | V2 (blind) | cross terms (twisted sums), curvatures at r = 0, width ratios, Δm_c = −0.081/−0.117/−0.157 GeV vs −0.0848/−0.1238/−0.1667 | PRL ("cross terms to 0.01 GeV"); PRD Table V, row "Cross terms, charged part"; report Sec. 19.3 | comparison (2 items) |
| `sm_mapping/` | V1 (blind) | SM mapping with mr and SMDR: λ(μ_b) to 4×10⁻⁶ (SMDR), scheme difference 2.8×10⁻⁴ for mr against the main mapping (0.043 GeV; 0.042 GeV between SMDR and mr inside V1), scale variation +0.033/−0.019 GeV | PRL End Matter A ("the SM mapping"); PRD Table V, row "Standard Model mapping"; App. D 3; report Sec. 19.4 | comparison (3 items) |
| `claims/` | V3 (blind) | marginalization, Bayes factors, thresholds, reach, hL_b sensitivity, w₀ = 0.991–0.996 | PRL End Matter A ("the marginalization"); PRD Table V, rows "Marginalization, reach" and "Survival measure, preparation weight"; report Sec. 19.5 | comparison (2 items) |
| `shape/` | V4 (blind) | floor, G(\|t_x\|), nonlinear survival t_*(90 %/99 %), emptiness budget, preparation window | PRD Sec. VII D–F (statements of the theorem and of t_*); report Sec. 19.6 | comparison (4 items) |
| `controls/` | numerical controls (main calculation) | ECS vs Siegert resonances (D108); Higgs-only inversion radius (D112/D121); BO-lo width deficit (D129) | PRL End Matter A; PRD Secs. V B–C; report Secs. 7, 8, 20 | archived outputs (`controls/README.md`) |
| `universality_inflation/` | V5 (blind) | ξ_inf (EW and μ_b) by Hawking–Moss counting, dλ_c/d ln η, coupling derivatives | PRD Table V, row "Duration, couplings, inflationary bound"; PRD Sec. V G–H; report Sec. 19.7 | comparison (2 items) |
| `T19_semiblind/` | T19 (semi-blind) | open items: KL values, F(19.32), s₉₉, σ(99 %), photon-pair range | report Sec. 19.9 (Table 31, row S); its D114 run is also `figures/data/shape/D114_own.json` | archived outputs |
| `early_rounds/R14/`, `early_rounds/R15/` | R14, R15 (compute first) | Higgs survival and state selection (R14); matching, real time, 2D model, time symmetry, separate line, CMB ensemble (R15) | report Sec. 19.10; PRD Table V, row "Survival measure, preparation weight" (P = w₀P_haz reproduced) | archived outputs |
| `report_checks/` | report's own checks (rep-A) | intermediate steps added to report chapters 1–5 and 13–17 (Bianchi I invariants, durations, ...) | report, `% [num:]` comments citing `rep-A_checks.py` | **rerun** by `run.py` and compared with `expected/*.out.txt` |
| `../higgs/04_cross_terms/cross_decomp.py` | main calculation, C161 step 6 | split of the run05 cross-term shift by zero-mode field (U(1)_Y, SU(2), SU(3)–top in two orders; vev mixing + Coulomb; top chiral) | PRL main text and End Matter A (field-by-field split); PRD Table V, row "Further cross terms" ("groups add to 2×10⁻⁴ GeV"); report Sec. 8.6; registry R5-049…053 | **rerun** (`post`) by `higgs/04_cross_terms/run.py` from the 66 archived node shards (check `cross_decomp_post`, bit for bit) |

Not archived: V6 (literature questions; no numbers) and the read-only consistency audit (it compared quoted values
with their source files).

## Files and provenance
`SOURCES.json` lists every file of this folder with its original path in the research archive, SHA-256, size and
modification time, and for code whether its hash equals the one stored in the blind record before the comparison
(57 code files match; 11 helper scripts were not recorded). The archived code is kept byte-identical and therefore
still contains the absolute paths of the machine it ran on; it is not executed by the package (except
`report_checks/`, which needs no paths). Left out on purpose: third-party sources and literature downloads of the
checks (mr, SMDR, TSIL, 3VIL, PDFs; their `SOURCES.json` records with URLs and hashes are in
`sm_mapping/inputs_sources/`), text extracts of copyrighted reports, build trees, caches (110 MB for V2), the raw
hazard node shards of the keystone check and the 1328 trajectory shards of V4 (their summaries are in `results/`).
`sm_mapping/code/drivers/c148_fit_copy.c` and `smdr_driver.c` derive from SMDR 1.3 (GPL-2.0-or-later, see
`third_party/README.md`); `c148_fit_copy.c` and `sm_mapping/results/DIAGNOSIS.json` belong to the diagnosis made after
the comparison. `universality_inflation/code/kb_copy/` is the keystone code that V5 reused, as its specification allowed.

Note for the report: Table 30 quotes the blind 2D_G ground-state energy at λ = 0 as 4.50770; the archived record has
4.507692 (4.50769). The comparison uses the archived value.
