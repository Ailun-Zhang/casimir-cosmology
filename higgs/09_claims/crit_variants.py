# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/crit_variants.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new script by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.  Re-implements (does not
#   import) the critical-value steps of 04_calculations/code/20261014_claude_v12_r3/two_branch/tb_post.py (crit_tables) and
#   .../higgs_robustness/C152_account.py (with_B3_w0_of_mt); validated by reproducing their run02-based outputs.
r"""[v12 package, 2026-10-15] Two-branch P2 = P1^2/w0 and B3-rule w0 critical masses at the run05 centre (methods reproduce C159 and C152).  Paper: PRD H3.

Step 5: two-branch and B3-rule w0 critical m_t at the run05 centre.
(A) two-branch (D130): P2 = P1^2 / w0 (w0 = 1 central; 1.002 = D111 upper end); y = ln(-ln P1) cubic spline on the 10-point
    grid (alpha_s = 0.1180); m(P2 = p) = m_of(sqrt(p w0)); alpha_s = 0.1171/0.1189 by adding the offset m_of(sqrt p) - m_of(p)
    to crit(alpha_s, p) (D130 procedure); Frechet range m_of((1+p)/2) .. m_of(p).
(B) B3 rule (D124): P = w0(m) P1(m), w0(m) = linear fit of C152_mc_w0.json with_B3_w0_of_mt.w0[tag] (m = 172.2..173.4);
    solve ln w0(m) - exp(y(m)) = ln p; Delta m_c = m(B3) - m(P1 = p) is the Table-1 w0 item (one-sided).
Run: python -B crit_variants.py"""
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
from scipy.optimize import brentq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = _zp.WORK
CR3 = os.path.join(ROOT, _zp.w("r3"))
P02 = os.path.join(CR3, "holonomy_cross/run02/C149_run02_summary.json")
P05 = os.path.join(CR3, "holonomy_cross/run05/C149_run05_summary.json")
C159 = os.path.join(CR3, "two_branch/run01/C159_summary.json")
C152W = os.path.join(CR3, "higgs_robustness/run01/C152_mc_w0.json")
NEW = os.path.join(ROOT, _zp.w("w/claims_run05/run01"))
LEV = (0.9, 0.5, 0.1)
TAGS = ("wall1.1", "wall1.2", "wall1.4", "CAPlike_mean")


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def crit_of(d, a):
    c = d["crit"][a]
    c = c.get("with_cross", c)
    return {p: c["P1=%s" % p] for p in LEV}


def analyse(d, W0):
    m = np.array(sorted(float(k) for k in d["P1_grid"]))
    P = np.array([d["P1_grid"]["%.1f" % x] for x in m])
    L = CubicSpline(m, np.log(-np.log(P)))

    def m_of(P1):
        t = math.log(-math.log(P1))
        return brentq(lambda x: float(L(x)) - t, m[0], m[-1], xtol=1e-12)
    out = dict(P1_crit_spline={p: m_of(p) for p in LEV}, P1_crit_published=crit_of(d, "0.1180"))
    tb = {}
    for p in LEV:
        tb[p] = dict(mt_P2=m_of(math.sqrt(p)), mt_P2_w0_1p002=m_of(math.sqrt(p * 1.002)), offset_vs_P1crit=m_of(math.sqrt(p)) - m_of(p),
                     frechet_mt_range=[m_of((1 + p) / 2), m_of(p)])
    out["two_branch_as0.1180"] = tb
    out["two_branch_other_alpha_s"] = {a: {p: crit_of(d, a)[p] + tb[p]["offset_vs_P1crit"] for p in LEV} for a in ("0.1171", "0.1189")}
    b3 = {}
    cm = W0["with_B3_w0_of_mt"]["mt_nodes"]
    for tag in TAGS:
        wf = np.poly1d(np.polyfit(cm, W0["with_B3_w0_of_mt"]["w0"][tag], 1))
        cr = {}
        for p in LEV:
            cr[p] = brentq(lambda z: math.log(wf(z)) - math.exp(float(L(z))) - math.log(p), m[0], m[-1], xtol=1e-12)
        b3[tag] = dict(crit=cr, dmc={p: cr[p] - m_of(p) for p in LEV}, w0_at_crit={p: float(wf(cr[p])) for p in LEV})
    out["B3_as0.1180"] = b3
    out["B3_other_alpha_s_wall1.2"] = {a: {p: crit_of(d, a)[p] + b3["wall1.2"]["dmc"][p] for p in LEV} for a in ("0.1171", "0.1189")}
    out["B3_dmc_range"] = {p: [min(b3[t]["dmc"][p] for t in TAGS), max(b3[t]["dmc"][p] for t in TAGS)] for p in LEV}
    return out


def main():
    W0 = jl(C152W)
    o2, o5 = analyse(jl(P02), W0), analyse(jl(P05), W0)
    s159 = jl(C159)
    h = s159["headline"]["crit_as0p1180"]
    val = dict(two_branch_run02_vs_C159={p: o2["two_branch_as0.1180"][p]["mt_P2"] - h["P2_0p%s" % str(p)[2]] for p in LEV},
               B3_run02_vs_C152={t: {p: o2["B3_as0.1180"][t]["crit"][p] - W0["with_B3_w0_of_mt"]["crit"][t]["crit"]["P1=%s" % p] for p in LEV}
                                 for t in TAGS},
               spline_vs_published_P1crit_run05={p: o5["P1_crit_spline"][p] - o5["P1_crit_published"][p] for p in LEV})
    oth = s159.get("crit", {}).get("targets_other_alpha_s_by_offset")
    if oth:
        val["two_branch_other_alpha_run02_vs_C159"] = {a: {p: o2["two_branch_other_alpha_s"][a][p] - oth[a]["P2=%s" % p] for p in LEV} for a in oth}
    res = dict(task="C161 step 5: two-branch and B3 w0 critical m_t (run05)", agent="claims-run05", validation=val, run02=o2, run05=o5,
               shift_run05_minus_run02=dict(two_branch={p: o5["two_branch_as0.1180"][p]["mt_P2"] - o2["two_branch_as0.1180"][p]["mt_P2"] for p in LEV},
                                            B3_wall1p2={p: o5["B3_as0.1180"]["wall1.2"]["crit"][p] - o2["B3_as0.1180"]["wall1.2"]["crit"][p] for p in LEV}),
               sha256={os.path.basename(__file__): sha(os.path.abspath(__file__)), "C149_run02_summary.json": sha(P02),
                       "C149_run05_summary.json": sha(P05), "C159_summary.json": sha(C159), "C152_mc_w0.json": sha(C152W)})
    json.dump(res, open(os.path.join(NEW, "crit_variants.json"), "w", encoding="utf-8"), indent=1, default=float)
    with open(os.path.join(NEW, "crit_variants.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "alpha_s", "P_level", "run02", "run05", "run05_minus_run02"])
        for p in LEV:
            w.writerow(["P1 critical (published)", "0.1180", p] + ["%.4f" % v for v in (o2["P1_crit_published"][p], o5["P1_crit_published"][p],
                                                                                          o5["P1_crit_published"][p] - o2["P1_crit_published"][p])])
            for k, lab in (("mt_P2", "P2=P1^2 (w0=1) critical"), ("mt_P2_w0_1p002", "P2=P1^2/w0, w0=1.002")):
                a2, a5 = o2["two_branch_as0.1180"][p][k], o5["two_branch_as0.1180"][p][k]
                w.writerow([lab, "0.1180", p, "%.4f" % a2, "%.4f" % a5, "%.4f" % (a5 - a2)])
            for a in ("0.1171", "0.1189"):
                a2, a5 = o2["two_branch_other_alpha_s"][a][p], o5["two_branch_other_alpha_s"][a][p]
                w.writerow(["P2=P1^2 critical (offset method)", a, p, "%.4f" % a2, "%.4f" % a5, "%.4f" % (a5 - a2)])
            for t in TAGS:
                a2, a5 = o2["B3_as0.1180"][t]["crit"][p], o5["B3_as0.1180"][t]["crit"][p]
                w.writerow(["B3 w0(m_t) critical [%s]" % t, "0.1180", p, "%.4f" % a2, "%.4f" % a5, "%.4f" % (a5 - a2)])
                w.writerow(["B3 Delta m_c vs P1 critical [%s]" % t, "0.1180", p, "%.4f" % o2["B3_as0.1180"][t]["dmc"][p],
                            "%.4f" % o5["B3_as0.1180"][t]["dmc"][p], "%.4f" % (o5["B3_as0.1180"][t]["dmc"][p] - o2["B3_as0.1180"][t]["dmc"][p])])
            for a in ("0.1171", "0.1189"):
                a2, a5 = o2["B3_other_alpha_s_wall1.2"][a][p], o5["B3_other_alpha_s_wall1.2"][a][p]
                w.writerow(["B3 w0(m_t) critical [wall1.2, offset]", a, p, "%.4f" % a2, "%.4f" % a5, "%.4f" % (a5 - a2)])
    print(json.dumps(val, indent=1, default=float))
    for p in LEV:
        print("P=%.1f  P1 %.4f->%.4f | P2 %.4f->%.4f | B3(w1.2) %.4f->%.4f dmc %.4f | B3 range %s" % (
            p, o2["P1_crit_published"][p], o5["P1_crit_published"][p], o2["two_branch_as0.1180"][p]["mt_P2"], o5["two_branch_as0.1180"][p]["mt_P2"],
            o2["B3_as0.1180"]["wall1.2"]["crit"][p], o5["B3_as0.1180"]["wall1.2"]["crit"][p], o5["B3_as0.1180"]["wall1.2"]["dmc"][p],
            ["%.4f" % x for x in o5["B3_dmc_range"][p]]))


if __name__ == "__main__":
    main()
