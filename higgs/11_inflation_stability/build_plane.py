# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/inflation_stability/build_plane.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Collect the shards -> xi_inf table (xi_inf_table.csv: xi_EW 0.04877/0.05512/0.06044 and xi(mu_b) 0.03030/0.03781/0.04413 at m_t = 172.2/172.6/173.0), C155_summary.json; the (m_t, xi_H) plane uses inputs/xi_c_zero_mode.json (round-2 values in the archived run - superseded, do not plot; see 07_realtime_xi).  Paper: PRD Sec. VII.

C155 step 3: collect shards -> xi_inf table, (m_t, xi_H) survival plane, C155_summary.json.
Zero-mode xi_c(m_t) is a replaceable input: run01/inputs/xi_c_zero_mode.json (version tag carried into all outputs)."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import glob
import math
import hashlib
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402

ROOT = _zp.WORK
RUN = os.path.join(ROOT, _zp.w("r3/inflation_stability/run01"))
CODE = os.path.join(ROOT, _zp.ZMLIB)
SH = os.path.join(RUN, "shards")
XIC = os.path.join(RUN, "inputs", "xi_c_zero_mode.json")
if not os.path.exists(XIC):
    os.makedirs(os.path.dirname(XIC), exist_ok=True)
    json.dump(dict(version="R2: D116 Sec.0.3 / Sec.3 item 5 (final zero-mode model, P_surv(xi)=0.99 P_surv(1/6)); "
                           "R3 values pending (NNLO mapping + cross term)",
                   reading="xi_H = coefficient of R h^2/2 in the zero-mode Hamiltonian at the turn, with lambda taken at "
                           "mu_b = MPbar/722; natural reading: xi_H = xi(mu_b)",
                   mt=[172.2, 172.6, 173.0], xi_c=[0.036, 0.072, 0.120], err=[0.003, 0.003, 0.001]),
              open(XIC, "w", encoding="utf-8"), indent=1)
ZM = json.load(open(XIC, encoding="utf-8"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def xi_c(mt):
    m, x = np.array(ZM["mt"]), np.array(ZM["xi_c"])
    if mt < m.min() - 1e-9 or mt > m.max() + 1e-9:
        return None
    c = np.polyfit(m, np.log(x), min(2, len(m) - 1))
    return float(np.exp(np.polyval(c, mt)))


S = {}
for p in glob.glob(os.path.join(SH, "*.json")):
    d = json.load(open(p, encoding="utf-8"))
    S[d["cfg"]["key"]] = d
print("shards", len(S))


def bnd(key, lab="eps=1/4", field="xi_ew"):
    d = S.get(key)
    if d is None or lab not in d["bounds"]:
        return None
    return d["bounds"][lab].get(field)


MTS = [round(171.4 + 0.2 * k, 1) for k in range(12)]
ALS = (0.1171, 0.1180, 0.1189)
rows = []
for als in ALS:
    for mt in MTS:
        k = "A_mt%.1f_as%.4f" % (mt, als)
        if k not in S:
            continue
        d = S[k]
        b4, b1 = d["bounds"].get("eps=1/4", {}), d["bounds"].get("eps=1", {})
        r = dict(mt=mt, als=als, xi_inf_ew=b4.get("xi_ew"), xi_inf_ew_eps1=b1.get("xi_ew"),
                 xi_inf_mub=b4.get("xi_at_mu_b"), xi_inf_Hpeak=b4.get("xi_at_H_peak"), N_peak=b4.get("N_peak"),
                 H_peak=b4.get("H_peak"), mu_bar_peak=b4.get("mu_bar_peak"), lam_mustar_peak=b4.get("lam_mustar_peak"),
                 Lambda_I=d.get("Lambda_I"), lam_mu_b=d.get("lam_mu_b"), Xi_mu_b=d.get("Xi_mu_b"))
        if "extra" in d:
            r.update(d["extra"])
        rows.append(r)

# variants (alpha_s = 0.1180)
VAR = {}
for tag in ("her", "tree", "her10", "her01", "zeta0", "xiyukall", "kramers", "gauss"):
    for mt in (171.8, 172.2, 172.6, 173.0, 173.4):
        v = bnd("B_%s_mt%.1f" % (tag, mt))
        if v is not None:
            VAR.setdefault(tag, {})["%.1f" % mt] = v
for tag in ("BKPV15", "B13", "xi2", "loops3"):
    for mt in (172.2, 172.6, 173.0):
        v = bnd("C_%s_mt%.1f" % (tag, mt))
        if v is not None:
            VAR.setdefault("sm_" + tag, {})["%.1f" % mt] = v

# table
cols = ["mt", "als", "xi_inf_ew", "xi_inf_ew_eps1", "xi_inf_mub", "xi_inf_Hpeak", "N_peak", "H_peak", "mu_bar_peak",
        "lam_mustar_peak", "Lambda_I", "lam_mu_b", "Xi_mu_b", "xi_classical_ew", "xi_crude_Vmax_gt_H_ew"]
with open(os.path.join(RUN, "xi_inf_table.csv"), "w", encoding="utf-8") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join("" if r.get(c) is None else ("%.6g" % r[c] if isinstance(r.get(c), float) else str(r[c]))
                         for c in cols) + "\n")

# plane (alpha_s central for the survival flags; alpha_s band carried as columns)
cen = {r["mt"]: r for r in rows if r["als"] == 0.1180}
lo_ = {r["mt"]: r for r in rows if r["als"] == 0.1171}
hi_ = {r["mt"]: r for r in rows if r["als"] == 0.1189}
XIG = np.round(np.arange(0.0, 0.4001, 0.002), 4)
plane = []
cmp_rows = []
for mt in MTS:
    if mt not in cen:
        continue
    r = cen[mt]
    xc = xi_c(mt)
    # reading (b): xi_H = xi(mu_b);   reading (a): xi_H = xi_EW (= xi(m_t))
    xinf_b, xinf_a = r["xi_inf_mub"], r["xi_inf_ew"]
    Xi_b = r["Xi_mu_b"]
    for reading, xinf in (("b_xi_at_mu_b", xinf_b), ("a_xi_EW", xinf_a)):
        dom = None
        if xc is not None and xinf is not None:
            dom = "inflation" if xinf > xc else "zero_mode"
        cmp_rows.append(dict(mt=mt, reading=reading, xi_inf=xinf, xi_c=xc,
                             xi_inf_as_lo=(lo_.get(mt, {}).get("xi_inf_mub" if reading[0] == "b" else "xi_inf_ew")),
                             xi_inf_as_hi=(hi_.get(mt, {}).get("xi_inf_mub" if reading[0] == "b" else "xi_inf_ew")),
                             xi_min=(max(xinf, xc) if (xc is not None and xinf is not None) else None), stronger=dom))
        for x in XIG:
            inf_ok = (xinf is not None) and (x >= xinf)
            zero_ok = None if xc is None else bool(x >= xc)
            plane.append((mt, reading, float(x), int(inf_ok), "" if zero_ok is None else int(zero_ok),
                          int(x < 0.375), int(x < 1.0),
                          "" if zero_ok is None else int(inf_ok and zero_ok and x < 0.375)))
with open(os.path.join(RUN, "xi_plane_mt_xiH.csv"), "w", encoding="utf-8") as f:
    f.write("# (m_t, xi_H) survival plane, alpha_s=0.1180, Mh=125.13, sm_provider central; xi_c version: %s\n" % ZM["version"])
    f.write("# inf_ok: xi_H >= xi_inf (MMR criterion <N><1, HM, mu* RGI, N_end at Hdot/H^2=-1/4); zero_ok: xi_H >= xi_c "
            "(empty = zero-mode not computed at this m_t); up_res: xi_H < 3/8 (no tachyonic resonance, Herranen 2015); "
            "up_H15: xi_H < 1 (Herranen 2015 fatal window 1..1e5 for H > Lambda_I); survive = inf_ok*zero_ok*up_res\n")
    f.write("mt,reading,xi_H,inf_ok,zero_ok,up_res,up_H15,survive\n")
    for t in plane:
        f.write("%.1f,%s,%.3f,%d,%s,%d,%d,%s\n" % t)
with open(os.path.join(RUN, "xi_compare.json"), "w", encoding="utf-8") as f:
    json.dump(dict(xi_c_input=ZM, rows=cmp_rows), f, indent=1)

# reproduction
rep = json.load(open(os.path.join(RUN, "repro", "repro_step1.json"), encoding="utf-8"))
D = {k: S[k]["bounds"] for k in S if k.startswith("D_")}
summ = dict(
    task="T21/C155/D127", agent="inflation-stability",
    sm_provider=dict(path="higgs/zmlib/sm_provider.py", variant="central",
                     Mh=125.13, sha256=sha(os.path.join(ROOT, os.path.join(_zp.ZMLIB, "sm_provider.py"))),
                     note="T1 run02 SMDR refit pending (<=0.005 GeV-equivalent); not used here"),
    code_sha256={os.path.basename(p): sha(p) for p in glob.glob(os.path.join(CODE, "*.py"))},
    xi_c_input=ZM,
    reproduction=dict(
        MMR2021_starobinsky_fieldtheory=dict(target=0.05938, ours_MMRtable_provider_g3=rep["MMR2021_star_fieldtheory_MMRtab"]["bound"],
                                             ours_MMRtable_g3_B13=D.get("D_MMRtab_g3B13"),
                                             ours_provider_couplings=rep["MMR2021_star_fieldtheory_provider"]["bound"]),
        MMR2021_mt_dependence_provider={k: v for k, v in D.items() if k.startswith("D_provider")},
        MMR2021_mt_dependence_target="+0.007/-0.009 for m_t = 172.76 +- 0.6 GeV (Eq. 4.3)",
        MMR2022_R2=rep["MMR2022_conventions"]),
    xi_inf_rows=rows, variants_as01180_eps_quarter=VAR, compare=cmp_rows)
# validity flag: HM bound coincides with the xi(mu*)=0 line (MMR terminate curves there) -> only an upper estimate
cl = {r["mt"]: r.get("xi_classical_ew") for r in rows if r["als"] == 0.1180}
for r in rows:
    c0 = cl.get(r["mt"]) or 0.0249
    r["validity"] = "xi(mu*)~0 limited" if (r["xi_inf_ew"] is not None and r["xi_inf_ew"] < c0 + 0.002) else "ok"
# alpha_s collapse test: xi_inf as a function of lambda(mu_b)
cr = sorted([r for r in rows if r["als"] == 0.1180 and r["validity"] == "ok"], key=lambda r: r["lam_mu_b"])
lb, xb = np.array([r["lam_mu_b"] for r in cr]), np.array([r["xi_inf_ew"] for r in cr])
coll = []
for r in rows:
    if r["als"] != 0.1180 and r["validity"] == "ok" and lb.min() <= r["lam_mu_b"] <= lb.max():
        coll.append(dict(mt=r["mt"], als=r["als"], lam_mu_b=r["lam_mu_b"], xi_inf=r["xi_inf_ew"],
                         xi_inf_central_at_same_lam_b=float(np.interp(r["lam_mu_b"], lb, xb))))
summ["alpha_s_collapse_in_lambda_mu_b"] = coll
summ["R2_counterfactual_project_background"] = {k: S[k]["bounds"] for k in S if k.startswith("E_")}
summ["upper_bound_Herranen2015"] = dict(
    window="tachyonic/resonant amplification only for xi > 3/8; fatal for 1 <~ xi <~ 1e5 when H >~ Lambda_I "
           "(arXiv:1506.04065 Eq. 22, p.4 Fig.1, conclusion p.4-5)",
    H_end_eps1_GeV=0.484 * 6.116e-6 * 2.435e18,
    Lambda_I_by_mt={"%.1f" % r["mt"]: r["Lambda_I"] for r in rows if r["als"] == 0.1180},
    applies_for="m_t >~ 171.7 GeV (H_end > Lambda_I); conformal xi = 1/6 < 3/8 is inside the allowed window")
summ["conclusion"] = (
    "Project plateau = minimally coupled scalar with Starobinsky-shaped potential (v11 A4): the MMR 2021 field-theory "
    "criterion applies; MMR 2022 R^2 cross terms are absent. xi_inf(m_t) [xi_EW at mu=m_t] = 0.049/0.055/0.060 at "
    "m_t = 172.2/172.6/173.0 (alpha_s 0.1180). In the high-scale reading (xi at mu_b) 0.030/0.038/0.044 < xi_c(R2) "
    "0.036/0.072/0.120: zero-mode condition is the stronger one for m_t >= 172.2 (crossover ~172.1 by extrapolation); "
    "in the EW reading inflation is stronger below ~172.4 GeV. Herranen 2015 upper window xi <~ 1 (3/8 robust) applies; "
    "xi_H = 1/6 allowed by all.")
with open(os.path.join(RUN, "C155_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1, default=float)
print("collapse:", [(c["mt"], c["als"], round(c["xi_inf"], 4), round(c["xi_inf_central_at_same_lam_b"], 4)) for c in coll][:12])
print("validity-limited rows:", [(r["mt"], r["als"]) for r in rows if r["validity"] != "ok"])
with open(os.path.join(RUN, "C155_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1, default=float)

print("mt   xi_inf_EW(1/4)  (eps=1)   xi_inf(mu_b)  as-  as+   xi_c   N_peak  Lambda_I  classical  crude")
for mt in MTS:
    if mt not in cen:
        continue
    r = cen[mt]
    f = lambda v: ("%.4f" % v) if isinstance(v, (int, float)) else str(v)  # noqa: E731
    print(mt, f(r["xi_inf_ew"]), f(r["xi_inf_ew_eps1"]), f(r["xi_inf_mub"]), f(lo_.get(mt, {}).get("xi_inf_ew")),
          f(hi_.get(mt, {}).get("xi_inf_ew")), f(xi_c(mt)), f(r["N_peak"]), "%.2e" % (r["Lambda_I"] or 0),
          f(r.get("xi_classical_ew")), f(r.get("xi_crude_Vmax_gt_H_ew")))
print("variants (xi_EW, eps=1/4):")
for k, v in VAR.items():
    print(" ", k, {m: round(x, 4) for m, x in v.items()})
print("D:", {k: {lab: round(b["xi_ew"], 5) for lab, b in v.items() if b.get("xi_ew")} for k, v in D.items()})
