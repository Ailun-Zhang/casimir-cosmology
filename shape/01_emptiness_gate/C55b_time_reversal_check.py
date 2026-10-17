# PROVENANCE: 04_calculations/code/20261009_claude_R5/C55b_time_reversal_check.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Mode-by-mode time-reversal cross-check of C55 against the archived round-1 C49 pilot (input inputs/C49_pilot_results.json). Paper: PRD App. app:gate (two codes agree).

C55b: mode-by-mode time-reversal cross-check of C55 against the archived C49 pilot.

C49 (expanding branch): vacuum at phi=1.51 (window closed) -> occupation at the j-th
scalar turning point after inflation, relative to the instantaneous oscillator vacuum.
C55 (contracting branch, independent code): instantaneous oscillator vacuum at the
mirror turning point -> exact occupation after the window closes.
Time-reversal invariance of the free mode equation on the mirrored background implies
equal |beta|^2. Agreement validates both codes and the C55 readout.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C55_contracting_window_gate as C55

sys.path.insert(0, str(HERE.parent))  # package: chain root (shape/)
import _shape_paths as WP  # noqa: E402
PILOT = WP.step_dir('01_emptiness_gate', 'inputs', 'C49_pilot_results.json')


def main():
    pilot = json.loads(PILOT.read_text())
    bg = C55._bg()
    rows = []
    for mode in pilot['modes']:
        q = mode['q']
        if q > 10.0:
            continue
        for ck in mode['checkpoints'][1:4]:          # first turning points after t_end
            t = ck['t']
            j = int(np.argmin(np.abs(np.array(bg.turns) - t)))
            if abs(bg.turns[j] - t) > 1e-6:
                continue
            r = C55.run_mode((q, -bg.turns[j], -bg.t_on, 1e-12, 'instantaneous'))
            rows.append({'q': q, 'turn_index': j + 1, 't': t, 'n_C49_expanding': ck['occupation'],
                         'n_C55_contracting': r['n'],
                         'rel_diff': abs(r['n'] - ck['occupation']) / max(abs(ck['occupation']), 1e-300)})
            print(f"q={q:7.3f} turn={j+1}  C49 {ck['occupation']:.6e}  C55 {r['n']:.6e}  rel {rows[-1]['rel_diff']:.2e}", flush=True)
    out = WP.wfile('r5/C55_window_gate/C55b_time_reversal_check.json')
    out.write_text(json.dumps({'rows': rows, 'scope': 'mode-by-mode time-reversal comparison with archived C49 pilot'}, indent=1))


if __name__ == '__main__':
    main()
