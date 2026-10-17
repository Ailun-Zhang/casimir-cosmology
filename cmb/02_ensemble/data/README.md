# cmb/02_ensemble/data/ — the l ≤ 80 CAMB transfer table

`C59_transfer.npz` (85 MB) is the T and E transfer table of the frozen snapshot P5_CMB used by `cmb/02 --full`
(`run.py --full` sets `DNEC_CMB_TRANSFER` to it); its SHA-256 equals the one recorded in `../ens_core.py`.
`high_l_transfer.py` (generator), `C59_transfer.json` (metadata, normalization check, low-k patch record) and
`transfer.log` are byte-identical copies of the snapshot files (`SOURCES.json`: original paths, SHA-256, settings).

Notes on the generator, which is kept unchanged so that its hash stays that of the snapshot:
- its docstring is inherited from `cmb/01_fisher_kl/C59_transfer.py` and says l = 2..40; the code computes
  l = 2..`LMAX_T` = 100 (CAMB `set_for_lmax(100)`), and `cmb/02` uses the rows l ≤ 80;
- it needs CAMB 2.0.4 (`../../requirements-optional.txt`) and writes its outputs next to itself (`OUT = HERE`), so
  regenerate the table by copying the script into an empty folder, running it there and comparing the result with
  this `C59_transfer.npz`; running it in place would overwrite the shipped table.
