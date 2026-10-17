# -*- coding: utf-8 -*-
"""R15 r15-beyond-bo: inspect structure of allowed input JSONs (read-only)."""
import json, sys
ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
files = [
    ROOT + r"\04_calculations\results\20261013_claude_v12\higgs_survival\run02_stage3\S3_prop.json",
    ROOT + r"\04_calculations\results\20261013_claude_v12_r2\higgs_matching\run01\C138_summary.json",
    ROOT + r"\04_calculations\results\20261013_claude_v12_r2\higgs_matching\run01\C138_static.json",
    ROOT + r"\04_calculations\results\20261013_claude_v12_r2\higgs_matching\run01\C138_hazard.json",
]

def show(o, pre="", depth=0, maxd=4):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                tag = f"dict[{len(v)}]" if isinstance(v, dict) else f"list[{len(v)}]"
                print(f"{pre}{k}: {tag}")
                if depth < maxd:
                    show(v, pre + "  ", depth + 1, maxd)
            else:
                s = repr(v)
                print(f"{pre}{k}: {s[:110]}")
    elif isinstance(o, list):
        if len(o) == 0:
            return
        if all(not isinstance(x, (dict, list)) for x in o):
            print(f"{pre}[nums] {repr(o[:6])[:150]} ... last={repr(o[-1])[:40]}")
        else:
            print(f"{pre}[0]:")
            show(o[0], pre + "  ", depth + 1, maxd)
            if len(o) > 1 and isinstance(o[0], (int, float, str)) is False:
                pass

for f in files:
    print("=" * 30, f.split("\\")[-1])
    with open(f, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    show(d, maxd=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
