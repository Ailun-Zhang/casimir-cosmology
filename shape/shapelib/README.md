# shapelib — shared modules of the nonlinear shape model

| module | role | original |
|---|---|---|
| `C60_twisted_ewald.py` | SM twisted Ewald Casimir energy Z(G), gradients, Hessians on unimodular G | `04_calculations/code/20261009_claude_R5/nonlinear_SM/` |
| `C60_nonlinear_integrator.py` | volume + five shape moduli, constraint, DOP853 (rtol 1e-11), fates | same |
| `C76_nonlinear_zero_mode.py` | zero-mode energy model (ξ = 0.299897, κ_E = 19.32) + fu/C76 ensembles | `.../20261011_claude_followup/` |
| `C142_shape_nonlinear_threshold.py` | D114 amplitude/R1 runs, class NLR, analysis helpers | `.../20261013_claude_v12_r2/shape_nonlinear_threshold/` |
| `C153_nl.py` | D125 run01 (conditional theorem, families, next order, width) + NLR2 | `.../20261014_claude_v12_r3/shape_hardening/` |
| `C153_run02.py` | D125 run02 (N = 2048: t_*, s₉₉) | same |
| `redraw_check.py` (new) | bit-level re-integration of archived trajectories | — |
| `textcheck.py` (new) | compares a text output with its archived copy | — |
The worker caps of the research runs (4–6 processes on a shared machine) are lifted: `--procs` defaults to env
`DNEC_WORKERS`; each trajectory is independent and deterministic, so results do not depend on the worker count.
