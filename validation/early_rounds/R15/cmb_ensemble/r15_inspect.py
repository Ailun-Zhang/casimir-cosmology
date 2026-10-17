# -*- coding: utf-8 -*-
"""R15: structure inspection of producer data files (read-only)."""
import json

R = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\04_calculations\results\20261013_argon_codex_intake\cmb_ensemble\run01\results"


def show(o, depth=0, maxd=4, key="root"):
    pad = "  " * depth
    if isinstance(o, dict):
        ks = list(o.keys())
        print(f"{pad}{key}: dict[{len(ks)}] keys[:12]={ks[:12]}")
        if depth < maxd:
            for k in ks[:4]:
                show(o[k], depth + 1, maxd, k)
    elif isinstance(o, list):
        print(f"{pad}{key}: list[{len(o)}]")
        if o and depth < maxd:
            show(o[0], depth + 1, maxd, "[0]")
    else:
        s = repr(o)
        print(f"{pad}{key}: {s[:160]}")


for fn in ["kl_table.json", "S3_postcheck.json"]:
    print("=" * 30, fn)
    with open(R + "\\" + fn, encoding="utf-8") as f:
        d = json.load(f)
    show(d)
