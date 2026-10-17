# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/table1_rebuild.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new script by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.
r"""[v12 package, 2026-10-15] PRL Table 1 rebuilt at run05 without double counting (C161 step 3): matching item from the run05 matching-scale scan, D112/T14a overlap removed, w0 by D124 bins -> table1_error_budget_run05.csv, C154_inputs_v2_run05_t1new.json.

Step 3: Table 1 (error budget on m_c, GeV) rebuilt on the run05 centre without double counting.
 (i)  one-loop matching: consistent matching-scale scan of the run05 node pipeline (mu_scan_summary.json): envelope over
      c in {1/2, 2, 2 pi} (D110 adopted range [1/2, 2 pi]) and the top threshold ct in {1/2, 2}; the two are NOT added in
      quadrature when the scan shows they probe the same y_t(mu) ambiguity (see 'matching_combination'); + BO gauge split
      +-0.002 (D110 Sec. 5, not recomputed).
 (ii) D112 'in-model' sub-items DMC +-1 sigma and B_x are superseded by the full-dimensional T14a result (D123 Sec. 0.5,
      3.4); 'static vs hazard conversion' and 'coupling transfer of the ratio' no longer exist in the node-direct hazard
      (D122/D121 run02-05); only dE_DMC with v11-coupling spline remains (+-0.002, D122 Sec. 3 estimate).
      T14a: gamma_full/gamma_2D = 0.976 (central) / 0.937 (envelope, WKB incl. B_w) from D123 Sec. 3.3, converted with the
      run05 d ln I/dm_t at each m_c (D123 used D112's 5.717/3.513), so P1 = 0.1 is computed, not extrapolated.
 (iii) w0: B3 rule w0(m_t) on run05 (crit_variants.json): one-sided, level dependent.
 (iv) P1 = 0.1 entries: (i) direct from the scan, (ii) from the run05 slope; nothing linearly extrapolated.
Writes table1_error_budget_run05.csv, table1_rebuild.json and C154_inputs_v2_run05_t1new.json (model composite of the new
table, for the claim-layer sensitivity run into run01/t1new/).
Run: python -B table1_rebuild.py"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import csv
import hashlib

import numpy as np
from scipy.interpolate import CubicSpline

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _zp.WORK
NEW = os.path.join(ROOT, _zp.w("w/claims_run05/run01"))
OLDT = os.path.join(ROOT, _zp.w("r3/claim_positioning/run02/table1_error_budget_final.csv"))
R5 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run05/C149_run05_summary.json"))
LK = ("0.9", "0.5", "0.1")
RHO_C, RHO_ENV = 0.976, 0.937          # D123 Sec. 3.3 table (gamma_full/gamma_2D; envelope = WKB incl. B_w, one-sided upper)


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    MS = jl(os.path.join(NEW, "mu_scan", "mu_scan_summary.json"))
    CV = jl(os.path.join(NEW, "crit_variants.json"))
    CS = jl(os.path.join(NEW, "C154_summary.json"))
    d5 = jl(R5)
    old = {r[0]: r for r in csv.reader(open(OLDT, encoding="utf-8"))}
    # run05 hazard slopes d ln I / dm_t at the critical masses
    g = d5["P1_grid"]
    m = np.array(sorted(float(k) for k in g))
    y = CubicSpline(m, np.log(-np.log(np.array([g["%.1f" % x] for x in m]))))
    crit = d5["crit"]["0.1180"]
    slope = {L: float(y(crit["P1=%s" % L], 1)) for L in LK}
    t14a_c = {L: -math.log(RHO_C) / slope[L] for L in LK}
    t14a_e = {L: -math.log(RHO_ENV) / slope[L] for L in LK}
    t14a_d123 = {"0.9": (0.0043, 0.011), "0.5": (0.0070, 0.018)}
    # (i) matching
    env = MS["mu_envelope"]
    top = MS.get("top_threshold", {})
    dmc = MS["dmc_vs_c1"]
    match = {}
    for L in LK:
        k = "P1=%s" % L
        mu_p, mu_m = env[k]["plus_half_2pi"], -env[k]["minus_half_2pi"]
        t5, t2 = top.get(k, {}).get("ct0p5", 0.0), top.get(k, {}).get("ct2", 0.0)
        top_p, top_m = max(0.0, t5, t2), -min(0.0, t5, t2)
        gs = 0.002
        match[L] = dict(mu_plus=mu_p, mu_minus=mu_m, mu_plus_half_2=env[k]["plus_half_2"], mu_minus_half_2=-env[k]["minus_half_2"],
                        top_ct0p5=t5, top_ct2=t2, gauge_split=gs,
                        quad_plus=math.sqrt(mu_p ** 2 + top_p ** 2 + gs ** 2), quad_minus=math.sqrt(mu_m ** 2 + top_m ** 2 + gs ** 2),
                        nodup_plus=math.sqrt(max(mu_p, top_p) ** 2 + gs ** 2), nodup_minus=math.sqrt(max(mu_m, top_m) ** 2 + gs ** 2))
    # adopted matching item: no quadrature of the mu scan with the top threshold (same y_t(mu) source, D110 Sec. 3.4:
    # 'the c = 2 change comes almost entirely from this term'); envelope of both + gauge split
    items = []

    def add(iid, typ, vals, src, files, change):
        items.append(dict(id=iid, type=typ, levels={L: list(vals[L]) for L in LK}, src=src, files=files, change=change))
    add("D112_residual_dE_DMC_v11", "sym", {L: (0.002, 0.002) for L in LK},
        "D112 Sec.4.4 sub-items DMC+-1sigma (+-0.004/0.006) and B_x (-0.007/-0.011) superseded by T14a (D123 Sec.0.5/3.4); "
        "'static-hazard conversion' (+-0.007/0.009) and ratio 'coupling transfer' (+-0.003/0.005) absent in the node-direct hazard "
        "(D122, D121 run02-05); remaining: dE_DMC spline with v11 couplings <= 0.002 (D122 Sec.3, estimate)",
        "D112 Sec.4.4; D123 Sec.0.5, 3.4; D122 Sec.3", "replaces D112_inmodel (+0.009/-0.012 | +0.012/-0.017 | +0.015/-0.022*)")
    add("matching_run05_scan", "asym", {L: (match[L]["nodup_plus"], match[L]["nodup_minus"]) for L in LK},
        "consistent matching-scale scan in the run05 node pipeline, mu = c/L, c in {1/2,1,2,2pi} (envelope, D110 range) with the "
        "top threshold ct in {1/2,2} taken as the same y_t(mu) source (envelope, not quadrature) + BO gauge split 0.002 (D110)",
        "run01/mu_scan/mu_scan_summary.json: mu_envelope, top_threshold, dmc_vs_c1", "replaces D110_matching (2nd-round 1D scan, P1=0.1 extrapolated)")
    add("cross_T2", "sym", {"0.9": (0.008, 0.008), "0.5": (0.010, 0.010), "0.1": (0.013, 0.013)},
        "D121 run05 Sec.3 (model-internal, incl. RG closure residual and custodial ratio)", "D121 appendix run05 Sec.3 table", "unchanged")
    add("cross_V2_minus_T2", "sym", {"0.9": (0.004, 0.004), "0.5": (0.007, 0.007), "0.1": (0.010, 0.010)},
        "V2 blind vs T2 difference (run02-level CCA); kept (possible partial overlap with D121 'first-order/variational' not resolved)",
        "R16 keystone_blind V2; C154 inputs", "unchanged")
    add("fulldim_T14a", "one_sided_plus", {L: (t14a_e[L], 0.0) for L in LK},
        "D123: gamma_full/gamma_2D = 0.976 central, 0.937 envelope (WKB incl. B_w), converted with run05 dlnI/dm_t "
        "(%.3f/%.3f/%.3f per GeV); central correction +%.4f/+%.4f/+%.4f" % (slope["0.9"], slope["0.5"], slope["0.1"],
                                                                            t14a_c["0.9"], t14a_c["0.5"], t14a_c["0.1"]),
        "D123 Sec.3.3; C149_run05_summary.json P1_grid (slopes)", "0.9/0.5 re-converted with run05 slope; 0.1 computed (was +0.025 extrapolated)")
    add("numerics", "sym", {L: (0.002, 0.002) for L in LK}, "D116", "D116", "unchanged")
    add("inflation_scale_vi", "sym", {"0.9": (0.035, 0.035), "0.5": (0.067, 0.067), "0.1": (0.112, 0.112)},
        "D124 (vi), r in [1e-3, 1e-2] (computed on the run02 centre; not recomputed)", "C152_input_dmc.json vi_inflation_scale", "unchanged")
    w0dm = {L: -min(CV["run05"]["B3_as0.1180"][t]["dmc"][L] for t in ("wall1.1", "wall1.2", "wall1.4", "CAPlike_mean")) for L in LK}
    add("w0_B3_rule", "one_sided_minus", {L: (0.0, w0dm[L]) for L in LK},
        "B3 rule w0(m_t) (walls 1.1/1.2/1.4/CAP mean) on run05, largest |Delta m_c| per level (D124 tiers [-0.016,0]/[-0.006,0]/[-0.004,0])",
        "run01/crit_variants.json B3_as0.1180.*.dmc", "level-dependent (was -0.016 at all levels)")

    def comp(drop=()):
        sp = {L: math.sqrt(sum(it["levels"][L][0] ** 2 for it in items if it["id"] not in drop)) for L in LK}
        sm = {L: math.sqrt(sum(it["levels"][L][1] ** 2 for it in items if it["id"] not in drop)) for L in LK}
        return sp, sm
    SP, SM = comp()
    SPn, SMn = comp(("inflation_scale_vi",))
    oldtot = old["MODEL TOTAL (quadrature per side)"]
    oldnovi = old["model total without (vi)"]
    rows = []
    for it in items:
        rows.append([it["id"], it["type"]] + [v for L in LK for v in it["levels"][L]] + [it["src"], it["files"], it["change"]])
    rows.append(["MODEL TOTAL (quadrature per side)", "composite"] + [v for L in LK for v in (SP[L], SM[L])] + ["this work (C161)", "", "was " + "/".join(oldtot[2:8])])
    rows.append(["model total without (vi)", "composite"] + [v for L in LK for v in (SPn[L], SMn[L])] + ["this work (C161)", "", "was " + "/".join(oldnovi[2:8])])
    da = CS["dmc_dalpha_s"]
    keep = ["SM: matching-scale envelope (m-bar)", "SM: scheme spread mr vs SMDR (incl. 0.04 implementation)", "SM: RGE / alpha_s order / W,Z pole convention",
            "SM TOTAL m-bar formulation", "SM: pole series O(alpha_s^5) (pole formulation only)", "SM TOTAL pole formulation (4-loop truncated)",
            "SM: all-orders pole offset (one-sided, criterion up)", "SM: renormalon ambiguity", "SM: BKPV/FJ scale variation (robustness)",
            "parametric M_h +-0.11 (PDG 2026)", "measurement m_t direct (PDG 2026)", "MC-pole interpretation"]
    for k in keep:
        r = old[k]
        rows.append(r[:9] + ["run02 table1_error_budget_final.csv (SM/measurement rows not model dependent)", "unchanged"])
    rows.append(["parametric alpha_s +-0.0009", "param"] + [0.0009 * da["P1=%s" % L] for L in LK for _ in (0, 1)] +
                ["run05 crit (C154_summary dmc_dalpha_s)", "run01/C154_summary.json dmc_dalpha_s", "was " + "/".join(old["parametric alpha_s +-0.0009"][2:8])])
    hdr = ["source", "type", "P0.9_plus", "P0.9_minus", "P0.5_plus", "P0.5_minus", "P0.1_plus", "P0.1_minus", "ref", "source_files", "change_vs_run02"]
    with open(os.path.join(NEW, "table1_error_budget_run05.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for r in rows:
            w.writerow([("%.4f" % x) if isinstance(x, float) else x for x in r])
    out = dict(task="C161 step 3: Table 1 rebuild (run05)", items=items, composite=dict(plus=SP, minus=SM, plus_no_vi=SPn, minus_no_vi=SMn),
               old_composite=dict(total=oldtot[2:8], no_vi=oldnovi[2:8]), matching_detail=match, scan_dmc_vs_c1=dmc,
               scan_c1_minus_run05=MS.get("c1_minus_run05"), t14a=dict(slope_dlnI_dm=slope, central=t14a_c, envelope=t14a_e,
                                                                       D123_original_0p9_0p5=t14a_d123),
               sha256={os.path.basename(__file__): sha(os.path.abspath(__file__)), "mu_scan_summary.json": sha(os.path.join(NEW, "mu_scan", "mu_scan_summary.json")),
                       "crit_variants.json": sha(os.path.join(NEW, "crit_variants.json")), "old_table": sha(OLDT)})
    json.dump(out, open(os.path.join(NEW, "table1_rebuild.json"), "w", encoding="utf-8"), indent=1)
    # input file for the claim-layer sensitivity run with the new model composite
    cfg = jl(os.path.join(HERE, "C154_inputs_v2_run05.json"))
    cfg["_doc"] += " | t1new: model_components replaced by the C161 Table-1 rebuild (table1_rebuild.json)."
    cfg["out_dir"] = cfg["out_dir"] + r"\t1new"
    cfg["model_components"] = [dict(id=it["id"], type=it["type"], levels=it["levels"], src=it["src"]) for it in items]
    for c in cfg["model_components"]:
        if c["id"] == "fulldim_T14a":
            c["central_correction"] = [t14a_c[L] for L in LK]
        if c["id"] == "inflation_scale_vi":
            c["r1e-4_one_sided"] = [-0.089, -0.1725, -0.2795]
    json.dump(cfg, open(_zp.w("w/claims_run05/run01/C154_inputs_v2_run05_t1new.json"), "w", encoding="utf-8"), indent=1)
    os.makedirs(os.path.join(NEW, "t1new"), exist_ok=True)
    print("slopes", slope, "T14a central", t14a_c, "envelope", t14a_e)
    print("matching", json.dumps(match, indent=1))
    for L in LK:
        print("P1=%s  new total +%.4f/-%.4f (no vi +%.4f/-%.4f)" % (L, SP[L], SM[L], SPn[L], SMn[L]))
    print("old total", oldtot[2:8], "old no vi", oldnovi[2:8])


if __name__ == "__main__":
    main()
