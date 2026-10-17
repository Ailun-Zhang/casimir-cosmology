# dnec_common — shared helpers of the v12 code/data package

Import as a package from the package root (`sys.path.insert(0, <pkg root>)`, then `from dnec_common import paths`).
Do not put `dnec_common/` itself on `sys.path` (the module `io.py` would shadow the standard library).

## `paths.py` (stable interface)
| function | meaning | override |
|---|---|---|
| `PKG_ROOT` | package root (directory containing `dnec_common/`) | — |
| `pkg(*parts)` | file shipped with the package | — |
| `out_root()`, `out(*parts, mkdir=False)` | writable output tree, default `<pkg>/output` | env `DNEC_OUT` |
| `third_party(*parts)` | third-party tree, default `<pkg>/third_party` | env `DNEC_THIRD_PARTY` |
| `smdr_dir()` | SMDR 1.3 source/build tree, default `<third_party>/smdr/build/SMDR-1.3` | env `DNEC_SMDR` |
| `rel(path)` | path relative to the package root (for records) | — |

## `io.py`
`clean(obj)` (numpy/complex/non-finite → JSON-safe), `write_json(path, obj, sha=True)` (atomic UTF-8 JSON +
`<path>.sha256`), `load_json(path)` (tolerates a UTF-8 BOM), `sha256_file(path)`, `get_key(obj, "a/b/0")` (or a list
of keys when a key contains '/').

## `sm_inputs.py`
PDG 2026 inputs and pipeline constants used by the Higgs chain (m_t 172.60 ± 0.27, α_s 0.1180 ± 0.0009,
M_h 125.13 ± 0.11, M̄_P = 2.435×10¹⁸ GeV, μ_b = M̄_P/722.0, hL_b = 0.00437243, other m_t determinations).
The production modules keep their own literal copies (bit-for-bit reproduction); `check_consistency()` asserts that
they agree.

## `runner.py`
`Step` = common driver of every `run.py`: `--quick | --full | --reduced`, `--workers N`, `--work DIR`, `--fresh`,
`--no-verify`; work tree `<out>/<chain>/<mode>` (env `DNEC_<CHAIN>_WORK`, e.g. `DNEC_HIGGS_WORK`); `stage([...])`
copies archived inputs (mirror paths such as `r3/holonomy_cross/run05/shards`) from `<chain>/data/` or any
`<chain>/*/expected/` into the work tree without overwriting fresh results; `py(script, *args, env=…)` runs a step
script with logs in `<work>/logs/<step>/` and records the wall time; `verify("checks.json")` writes `VERIFY.json`
via `tools/verify_expected.py`.
