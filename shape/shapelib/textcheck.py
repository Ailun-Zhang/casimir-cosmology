# PROVENANCE: new helper written 2026-10-15 by code-shape-cmb (v12 code/data package).
"""Compare a text output produced in the work tree with its archived copy (line endings / trailing blanks normalised).

    python textcheck.py <produced, work-tree relative> <archived mirror path> <report, work-tree relative>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402


def lines(p):
    return [x.rstrip() for x in Path(p).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").split("\n")
            if x.strip()]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    prod, arch, out = sys.argv[1:4]
    a, b = lines(WP.work_root() / prod), lines(WP.find_archived(arch))
    nd = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    rep = {"produced": prod, "archived": arch, "n_lines_produced": len(a), "n_lines_archived": len(b), "n_diff_lines": nd}
    WP.wfile(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))
