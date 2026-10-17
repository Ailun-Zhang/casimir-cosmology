# numbers/ — single source of the claim-layer numbers

The PRL, the PRD and the companion technical report (*Casimir Cosmology: Derivations and Numerical Validation*,
`docs/technical_report/`) typeset their claim-layer numbers through the same LaTeX macros (`\PbarOne`,
`\mExclTheory`, `\xiHcentral`, ...). This directory is the single source of those macros.

| file | role |
|---|---|
| `numbers_v12.json` | the source: for each of the 51 macros the value(s), the displayed precision, the body template, the comment of `numbers.tex` (meaning, unrounded value, registry id), the **source** of every value (package file + JSON pointer) and the last check result; `layout` keeps the order, comments and line ends of `numbers.tex` |
| `make_numbers_tex.py` | generates `numbers.tex` from the JSON (`--out PATH ...` writes copies into the document sources, `--check` / `--compare PATH` verify that a copy is up to date) |
| `numbers.tex` | generated file, `\input{numbers}` in the three documents; at import (2026-10-16) byte-identical to `06_paper_draft/versions/v12/PRD/numbers.tex` and its PRL copy (SHA-256 in `numbers_v12.json` → `meta.imported_sha256`); two macros added for R18 (`\PoneAtExclFCC`, `\BexFCC`) and the regenerated file copied to the PRL, PRD and report sources (byte-identical) |
| `derive_numbers.py` | computes the numbers that are simple functions of archived outputs and not stored by a production script (`\xiHminEW`, `\lamcErrUp`, `\lamcErrDn`) from the package files it names; writes `derived_numbers.json` (`--check`: exit 1 if that file is not up to date) |
| `derived_numbers.json` | output of `derive_numbers.py`: value, formula and inputs (file, JSON pointer, value) of each derived number; no time stamp, so it is byte-stable |
| `check_numbers.py` | checks every number against the archived outputs of this package (`--write` stores the result in the JSON, `--registry` also traces every row of the run05 registry) |

**Source of a number.** Each entry of `numbers_v12.json` has a field `source`: a list of
`{"file": <package path>, "pointer": <JSON pointer, RFC 6901>}`, one per value of the macro (or one for all its
values). The pointer must end at a single number (key-exact). Optional `"sign": -1` and `"magnitude": true` mark the
six lower errors (`\modelErrDn...`): the result file stores the size of the error under `minus`, the papers typeset
`-\modelErrDn`. Example: `\xiHmin` → `higgs/07_realtime_xi/expected/w/claims_run05/run01/xi_run05_summary.json`
`#/run05/172.60/spread_all/1` (0.0772, the upper end of the spread of ξ_c at m_t = 172.60 GeV).

Workflow for a change: edit `values` (and if needed `precision`, `source` or the comment) in `numbers_v12.json` →
`python numbers/make_numbers_tex.py --out <PRL>/numbers.tex <PRD>/numbers.tex <report>/numbers.tex` →
`python numbers/derive_numbers.py` (if an input of a derived number changed) → `python numbers/check_numbers.py --write`
→ `python tools/sha256sums.py manifest` and `write`. Never edit `numbers.tex` by hand.

## Check status (2026-10-16, after the R18 revision)
**51 of 51 numbers verified, 0 mismatch, 0 unresolved** (`numbers_v12.json` → `meta.check_summary`; details per macro
in `numbers.<Macro>.check`). The two macros added for R18, `\PoneAtExclFCC` = 0.023 and `\BexFCC` = 0.033 (central
P₁ and exclusion factor B_ex at the FCC-ee exclusion threshold), point to
`higgs/09_claims/expected/w/claims_run05/run01/fcc_exclusion.json#/headline/...` (`higgs/09_claims/fcc_exclusion.py`).
Verified means:
- every shown value equals the value at its source pointer rounded to the shown precision, **sign included**;
- every unrounded value quoted in the comment agrees with the source to its last digit (a value written with an
  explicit sign is compared with the signed quantity, an unsigned one with what the macro shows; for a derived number
  every decimal of the comment must be its value or one of its inputs);
- every registry row named in the comment (R5-xxx, `higgs/09_claims/expected/w/claims_run05/run01/numbers_run05.csv`,
  the archived C161 registry) holds the number (or one of its inputs) and agrees with its own result file;
- `derived_numbers.json` is up to date.

Sources: the run05 claims files in `higgs/09_claims/expected/w/claims_run05/run01/` (`C154_summary.json`,
`C154_correction_summary.json`, `claims_compare.json`, `table1_rebuild.json`),
`higgs/07_realtime_xi/expected/w/claims_run05/run01/xi_run05_summary.json`, the D129 bands of ξ_c
(`higgs/07_realtime_xi/expected/r3/realtime_2d/run02/C157_run02_summary.json` → `bands/172.60`; run02 model, not
recomputed at run05) and `derived_numbers.json`:
- `\xiHminEW` = 1/6 + (0.0772 − 1/6)/1.1552 = 0.0892 (electroweak image of the conservative ξ_H(μ_b) = `\xiHmin`,
  with the one-loop map of `higgs/07_realtime_xi/xi_compare.py`);
- `\lamcErrUp`, `\lamcErrDn` = 0.0842 GeV and 0.0721 GeV (lower and upper model error of m_c at P₁ = 1/2, Table 1
  composite) × |dλ(μ_b)/dm_t| = margin_λ / margin_m = 0.0023674/0.35945 = 0.00659 GeV⁻¹.

The 10 macros that the first release check left "unresolved" (no file:key named in their comment) now have sources:
`\PbarBKPV` (`C154_correction_summary.json#/marginal/{continuous,equal_mass}/D1_direct_pole/Pbar_BKPV_FJ`),
`\sigHLLHCmin`, `\sigHLLHCmax` (`…#/HLLHC/HLLHC_0.17_as0.0009|0.00045/separation_172p6_173p3`), `\xiHmin`,
`\marginGeV`, `\marginLam`, `\mcMinusMzero` (`C154_summary.json#/relative_form/...`), `\xiHminEW`, `\xiHbandSM`,
`\xiHbandAs`.

Registry trace (`python numbers/check_numbers.py --registry`): all 53 rows of `numbers_run05.csv` are found in their
result files (the key column uses `|`, `;`, `*`, `{a,b}` and `a/b`; see the docstring of `check_numbers.py`); for
R5-032 the second triple (w₀ = 1.002) is stored at the neighbouring key `mt_P2_w0_1p002`, which the key column does
not name.
