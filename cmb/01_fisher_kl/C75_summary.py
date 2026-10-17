# PROVENANCE: 04_calculations/code/20261011_claude_followup/C75/C75_summary.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Compact window summary from C75_results.json (no covariance recomputation).

C75_summary: compact window summary from C75_results.json (no new covariances, no KL recomputation).

Adds: D on the task grid L0/d_LSS in {1.07,...,1.20}; contiguity of every window (all allowed grid points below the edge
readable); profiled minima near the upper edges; D at fixed shortest cycle l_min = s e^{-sigma} (log-linear in s);
comparison with the free fossil; robustness (x_max = 240, l <= 40, orientation inside E) with the implied edge shift
delta_s = ln(D_new/D_old) / |d ln D/ds| (local slope of the l<=30, x_max=160 curve at the edge, ratio assumed constant
across the edge neighbourhood).

    python C75_summary.py   -> results/20261011_claude_followup/C75/C75_summary.json (+ printed table)
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C75_common import RES, SIGMAS, THR, CIRCLE, s_circle  # noqa: E402

TASK_GRID = [1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.14, 1.16, 1.20]
LMIN_GRID = [0.99, 1.00, 1.01, 1.02, 1.03]


def loglin(xs, ys, x):
    xs = np.asarray(xs, float); ys = np.log(np.asarray(ys, float))
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.exp(np.interp(x, xs, ys)))


def local_slope(pts, Ds, s0):
    """d ln D / ds from the two grid points bracketing s0."""
    i = int(np.searchsorted(pts, s0)) - 1
    i = min(max(i, 0), len(pts) - 2)
    return (math.log(Ds[i + 1]) - math.log(Ds[i])) / (pts[i + 1] - pts[i])


def main():
    R = json.loads((RES / "C75_results.json").read_text(encoding="utf-8"))
    out = {"source": str(RES / "C75_results.json"), "threshold": "2D >= 9 (D >= 4.5)", "per_sigma": {}}
    for b, sig in SIGMAS.items():
        Dss = R["D_same_size"][b]
        pts = sorted(float(k) for k in Dss)
        sc = s_circle(sig)
        allowed = [s for s in pts if s >= sc - 1e-12]
        W = R["windows"][b]
        e = dict(sigma_E=sig, r=W["r"], amp_beta_d=W["amp_beta_d"], s_circle=sc,
                 D_at_task_grid={f"{s:.2f}": dict(TE=Dss[f"{s:.5f}"]["TE"], T=Dss[f"{s:.5f}"]["T"],
                                                  allowed_by_circles=s >= sc - 1e-12,
                                                  shortest_cycle=s * math.exp(-sig))
                                 for s in TASK_GRID if f"{s:.5f}" in Dss})
        for fl in ("TE", "T"):
            w = W[fl]
            up = w["upper_edge_loglin"]
            ys = [Dss[f"{s:.5f}"][fl] for s in allowed]
            ent = dict(D_at_circle_bound=w["D_at_circle_bound"], upper_edge=up, upper_edge_pchip=w["upper_edge_pchip"],
                       window=w["window"], width=w["width"], width_efolds=w["width_efolds"],
                       shortest_cycle_range=w["shortest_cycle_range"],
                       monotone_decreasing_above_circle=w["monotone_decreasing_above_circle"])
            if isinstance(up, float):
                ent["contiguous"] = bool(all(y >= THR for s, y in zip(allowed, ys) if s <= up))
                ent["max_D_allowed"] = float(max(ys))
                ent["s_of_max_D_allowed"] = float(allowed[int(np.argmax(ys))])
                ent["dlnD_ds_at_edge"] = local_slope(allowed, ys, up)
            pr = w["profiled"]
            prow = pr["rows"]
            dmin = [min(p["D_min_grid"], p["D_min_parabola"] if p["D_min_parabola"] is not None else np.inf)
                    for p in prow]
            pe = dict(upper_edge=pr["upper_edge_loglin"], window=pr["window"],
                      shortest_cycle_range=pr.get("shortest_cycle_range"), D_at_circle_bound=pr["D_at_circle_bound"],
                      upper_edge_shift_vs_same_size=pr.get("upper_edge_shift"),
                      any_minimum_at_grid_edge=bool(any(p["at_grid_edge"] for p in prow)))
            if isinstance(pr["upper_edge_loglin"], float):
                pe["contiguous"] = bool(all(d >= THR for p, d in zip(prow, dmin) if p["s"] <= pr["upper_edge_loglin"]))
            ref = up if isinstance(up, float) else (pr["upper_edge_loglin"] if isinstance(pr["upper_edge_loglin"], float)
                                                    else sc)
            near = sorted(prow, key=lambda p: abs(p["s"] - ref))[:3]
            pe["near_upper_edge"] = [dict(s=p["s"], D_same=p["D_same"], D_min=min(p["D_min_grid"], p["D_min_parabola"]
                                                                                  if p["D_min_parabola"] is not None
                                                                                  else np.inf),
                                          s_prime_min=p["s_prime_parabola"] if p["s_prime_parabola"] is not None
                                          else p["s_prime_min"], reduction=p["reduction"])
                                     for p in sorted(near, key=lambda p: p["s"])]
            ent["profiled"] = pe
            # D at fixed shortest cycle (all grid points, allowed or not, are physical covariances; circles aside)
            lm = [s * math.exp(-sig) for s in pts]
            ent["D_at_shortest_cycle"] = {f"{l:.2f}": loglin(lm, [Dss[f"{s:.5f}"][fl] for s in pts], l)
                                          for l in LMIN_GRID}
            e[fl] = ent
        out["per_sigma"][b] = e
    # comparison with the free fossil (T+E, same-size)
    f = out["per_sigma"]["free"]["TE"]
    comp = {}
    for b in ("r25", "r30"):
        g = out["per_sigma"][b]["TE"]
        comp[b] = dict(lower_edge_shift=g["window"][0] - f["window"][0], upper_edge_shift=g["window"][1] - f["window"][1],
                       width_ratio=g["width"] / f["width"], width_efolds_ratio=g["width_efolds"] / f["width_efolds"],
                       overlaps_free_window=bool(g["window"][0] < f["window"][1]),
                       gap_to_free_window=g["window"][0] - f["window"][1],
                       shortest_cycle_upper_edge=g["shortest_cycle_range"][1],
                       free_shortest_cycle_upper_edge=f["shortest_cycle_range"][1])
    out["comparison_TE_vs_free"] = comp
    # derived diagnostics quoted in D57
    diag = {}
    for b, e in out["per_sigma"].items():
        d = {}
        for fl in ("TE", "T"):
            up, upp = e[fl]["upper_edge"], e[fl]["upper_edge_pchip"]
            d[f"pchip_minus_loglin_{fl}"] = (upp - up) if isinstance(up, float) and isinstance(upp, float) else None
        d["T_width_over_TE_width"] = e["T"]["width"] / e["TE"]["width"] if e["TE"]["width"] else None
        d["max_D_T_allowed"] = e["T"].get("max_D_allowed"); d["s_of_max_D_T_allowed"] = e["T"].get("s_of_max_D_allowed")
        diag[b] = d
    out["derived_diagnostics"] = diag
    out["ratio_to_free_selected"] = {b: {k: v for k, v in R["ratio_to_free"][b].items()
                                         if k in ("sigma_ratio_squared", "1.070", "1.080", "1.090", "1.100", "1.140",
                                                  "1.160", "1.200")} for b in ("r25", "r30")}
    # robustness extras
    ex = R.get("extras", {})
    rob = {"xconv": {}, "l40": {}, "dirs": {}}
    lab = {"free": "free", "r25": "r25", "r30": "r30"}
    for k, v in ex.get("xconv", {}).items():
        b = k.split("_")[0]
        rob["xconv"][k] = {}
        for fl in ("TE", "T"):
            ratio = v[fl]["x240"] / v[fl]["x160"]
            sl = out["per_sigma"][lab[b]][fl].get("dlnD_ds_at_edge")
            rob["xconv"][k][fl] = dict(x160=v[fl]["x160"], x240=v[fl]["x240"], rel_change=ratio - 1.0,
                                       implied_edge_shift=(math.log(ratio) / abs(sl)) if sl else None)
    for k, v in ex.get("l40", {}).items():
        b = k.split("_")[0]
        rob["l40"][k] = {}
        for fl in ("TE", "T"):
            ratio = v[fl]["l40"] / v[fl]["l30_from_l30_files"]
            sl = out["per_sigma"][lab[b]][fl].get("dlnD_ds_at_edge")
            rob["l40"][k][fl] = dict(l30=v[fl]["l30_from_l30_files"], l30_from_l40_files=v[fl]["l30_from_l40_files"],
                                     l30_subblock_rel_diff=abs(v[fl]["l30_from_l40_files"] / v[fl]["l30_from_l30_files"]
                                                               - 1.0),
                                     l40=v[fl]["l40"], ratio_l40_l30=ratio,
                                     implied_edge_shift=(math.log(ratio) / abs(sl)) if sl else None)
    rob["dirs"] = ex.get("dirs", {})
    out["robustness"] = rob
    out["orientation"] = R.get("orientation")
    out["validation_digest"] = dict(
        C59_cache_D_TE={k: v["TE"]["D"] for k, v in R["validation"]["from_C59_cache"].items()},
        C59_cache_D_T={k: v["T"]["D"] for k, v in R["validation"]["from_C59_cache"].items()},
        C59_json_D_TE={k: v["TE"]["C59_json"] for k, v in R["validation"]["from_C59_cache"].items()},
        paper_D_TE={k: v["TE"]["paper"] for k, v in R["validation"]["from_C59_cache"].items()},
        fresh_1p07=R["validation"]["fresh_vs_cached"]["s1.07_free_fossil"],
        determinism_cubes=R["validation"]["determinism_cubes"],
        kl_methods=R["validation"]["kl_methods"], kl_eig_vs_logdet=R["kl_eig_vs_logdet_max_rel_diff"],
        free_edge_paper_coarse_grid=R["free_fossil_edge_on_paper_coarse_grid"])
    (RES / "C75_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # printed digest
    for b, e in out["per_sigma"].items():
        print(f"== {b}: sigma_E={e['sigma_E']}, s_c={e['s_circle']:.5f}")
        for fl in ("TE", "T"):
            x = e[fl]
            print(f"  {fl}: window {x['window']} contiguous={x.get('contiguous')} width={x['width']:.4f} "
                  f"lmin-range {x['shortest_cycle_range']} | profiled {x['profiled']['window']} "
                  f"contig={x['profiled'].get('contiguous')} gridedge={x['profiled']['any_minimum_at_grid_edge']}")
            print(f"      D at l_min: {x['D_at_shortest_cycle']}")
            print(f"      near edge (profiled): {x['profiled']['near_upper_edge']}")
        print("  task grid:", {k: (round(v['TE'], 3), round(v['T'], 3), v['allowed_by_circles'])
                               for k, v in e["D_at_task_grid"].items()})
    print("comparison:", json.dumps(comp, indent=0))
    print("derived:", json.dumps(diag, indent=0))
    print("l40 subblock rel diff max:", max(x[fl]["l30_subblock_rel_diff"] for x in rob["l40"].values()
                                            for fl in ("TE", "T")) if rob["l40"] else None)
    print("robustness:", json.dumps({k: v for k, v in rob.items() if k != "dirs"}, indent=0))
    print("dirs:", json.dumps(rob["dirs"], indent=0))


if __name__ == "__main__":
    main()
