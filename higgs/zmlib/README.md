# higgs/zmlib — shared modules of the Higgs zero-mode chain

One canonical copy of every production module that more than one result directory imports. The research tree held
several byte-identical copies of these files (one per calculation directory, each with a provenance header); before
packaging all copies were compared after removing the provenance lines: they are identical, except
`sm_provider.py` (the `sm_mapping` copy adds the run02 compiled-SMDR refit variants `sfit_*`; that superset is used
here — all other variants behave identically) and the T14a copy of `C140_beyond_bo.py` (outputs redirected; it
lives in `../05_fulldim_validation/C140_beyond_bo_t14a.py`).

| module | content | paper |
|---|---|---|
| `_zpaths.py` | path shim (`_zp.w("r3/<calc>/<run>")` = work tree, `_zp.ZMLIB`, `_zp.smdr()`) | — |
| `R14_fv.py` | finite-volume sums on T³ (P/A), D_θ(m²), T_B/2, T_F/2, c₂ | PRD Sec. V A, App. C 1–2 |
| `R14_res.py` | conditional-Gaussian gauge profile, 1D Siegert resonance solver | PRD App. C 1–3 |
| `R14_rge.py` | two-loop RGE + three-loop calibration (rounds 1–2 mapping; convergence table) | report Sec. 11.1; PRD App. D 3 |
| `C138_core.py` | consistent one-loop finite-volume matching dV_MC (Coulomb gauge, MS-bar FV sums) | PRD Sec. V A, App. C 1–2 |
| `C140_beyond_bo.py` | 2D (r, \|w\|) model, 2D Siegert solver | PRD Sec. V B, App. C 3 |
| `C150_twod.py` | direct 2D hazard at SM running couplings (T14b) | PRD Sec. V B–D |
| `sm_provider.py`, `sm_rge.py`, `qcd.py`, `_smdr_betas.py`, `c150_nnlo_central.py` | NNLO SM mapping (D120) | PRD App. D 3 |
| `cross_core.py`, `cross_kappa.py`, `cross_2d.py`, `cross_var.py`, `C149_direct.py` | cross terms, node-direct hazard (D121 run02) | PRD Sec. V C, App. C 4 |
| `hr_potential.py`, `hr_core.py`, `hr_part2.py`, `hr_post.py` | real-time machinery (D111) | PRD Sec. V D, App. D 2 |

**Code changes made for the package** (complete list with unified diffs in `PACKAGE_PATCHES.json`; source and
destination SHA-256 for every file):
1. absolute project paths → `_zp.*` (mechanical, by token-level string-literal rewriting; e.g.
   `os.path.join(ROOT, r"04_calculations\results\20261014_claude_v12_r3\holonomy_cross\run05")` →
   `os.path.join(ROOT, _zp.w("r3/holonomy_cross/run05"))`); a session scratch directory → `_zp.w("scratch")`;
2. a 6-line bootstrap after each module docstring (adds `zmlib` to `sys.path`, imports `_zpaths as _zp`) and a
   one-paragraph `[v12 package]` docstring prefix (physics, units, paper location);
3. worker caps `min(nw, 3)` → `nw` (env `DNEC_WORKERS`) in `C149_direct.run`, `C150_twod.two/oned` (parallelism only);
4. `sm_rge.build`: if the SMDR source `betas.c` is absent, the shipped generated `_smdr_betas.py` is used and its
   recorded source hash returned (regeneration still needs `third_party/smdr`);
5. `C149_direct`: the research-time `setup` step (copying the T14b modules next to the file) removed;
6. check-mode writes of R14_* (`results.json`, `STATUS.json` next to the code) → `_zp.w("r1/R14_checks/…")`.
No numerical constant, formula or algorithm was changed; the package reproduces the archived node shards and
critical masses bit for bit (see the VERIFY.json files of 03, 04).
