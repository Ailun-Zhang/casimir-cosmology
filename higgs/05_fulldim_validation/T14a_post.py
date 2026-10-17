# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_post.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Summary of the full-dimensional validation (C151_summary.json): Delta Q, four gamma estimates, Delta m_c = +0.0043(10)/+0.0070(17) GeV.  Paper: PRD App. H2; PRL Table 1 row 'full-dimensional validation'.

T14a step 4: C151_summary.json -- Delta Q (same wall), d ln gamma/dQ conversion, wall-dependence (tail-amplitude)
estimator of gamma_full/gamma_2D, WKB cross-check, Delta m_c.   python -B T14a_post.py
Conventions: Delta Q = Q_full(r_w) - Q_2D(r_w) (same Dirichlet wall at r = |phi| = r_w);
rho = gamma_full/gamma_2D;  Delta m_c = -ln(rho(lambda_b(m_c))) / (d ln I/dm_t)  with d ln I/dm_t = 5.717 | 3.513 GeV^-1
(P1 = 0.9 | 0.5) and lambda_b(m_c) = -0.00855 | -0.01130 (D112 Sec. 3.5).
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import numpy as np
import T14a_common as T

sys.stdout.reconfigure(encoding="utf-8")
LAMB = {"0.9": -0.00855, "0.5": -0.01130}
MAIN_WALLS = {-0.010: (4.80, 5.12, 5.44), -0.013: (4.40, 4.64, 4.96)}   # between barrier top and outer turning point


def wmean(vals, ses):
    w = 1 / np.asarray(ses) ** 2
    m = float((w * np.asarray(vals)).sum() / w.sum())
    se = float(1 / np.sqrt(w.sum()))
    chi2 = float((w * (np.asarray(vals) - m) ** 2).sum())
    return m, se, chi2, len(vals) - 1


def main():
    old = T.jload(T.OLDSUM)
    comb = T.jload(os.path.join(T.RES, "T14a_dmc_combined.json"))
    wall = T.jload(os.path.join(T.RES, "T14a_2d_wall.json"))
    capf = os.path.join(T.RES, "T14a_cap_dlng.json")
    cap = T.jload(capf) if os.path.exists(capf) else None
    wkbf = os.path.join(T.RES, "T14a_wkb.json")
    wkb = T.jload(wkbf) if os.path.exists(wkbf) else None
    S = dict(task="T14a/C151/D123 fulldim_dmc", agent="fulldim-dmc", units="L=1, gamma = -2 Im E (full width)",
             couplings=T.V11, matching="MC+CW (D110/C138 dV, mu=1/L)")
    # ---------------- lambda = 0 regression
    l0 = comb.get("lam0_rw9")
    w0 = wall["2D_ex|0.0|8.96"]["Q"]; w0g = wall["2D_G|0.0|8.96"]["Q"]
    if l0:
        e0, s0 = l0["extrap"]["E0"], l0["extrap"]["se_E0"]
        cm, cs, _, _ = wmean([e0, T.C72_LAM0], [s0, T.C72_SE])
        S["lam0_regression"] = dict(
            Q_full_DMC_T14a=e0, se=s0, slope=l0["extrap"]["slope"], chi2=l0["extrap"]["chi2"],
            C72=T.C72_LAM0, C72_se=T.C72_SE, T14a_minus_C72=e0 - T.C72_LAM0,
            T14a_minus_C72_sigma=(e0 - T.C72_LAM0) / np.hypot(s0, T.C72_SE),
            Q_full_combined_T14a_C72=cm, se_combined=cs,
            Q_2D_ex_FD_wall8p96=w0, Q_2D_G_FD_wall8p96=w0g, D112_2D_ex_numerov=T.Q2D_LAM0,
            dQ_full_minus_2Dex_T14a=e0 - w0, dQ_full_minus_2Dex_combined=cm - w0, dQ_full_minus_2DG=e0 - w0g)
    # ---------------- lambda < 0 same-wall comparison
    dl = {}
    if cap:
        for o in cap["tasks"]:
            dl.setdefault((o["var"], o["lam"]), []).append(o["dlng_dQ"])
    res = {}
    for lam, tag in ((-0.010, "m010"), (-0.013, "m013")):
        Qs, gs = old["Q"]["2D_ex"][f"{lam}"], old["gamma"]["2D_ex"][f"{lam}"]
        rows = []
        for k, v in sorted(comb.items()):
            if not k.startswith(tag) or v["extrap"].get("E0") is None:
                continue
            rw = v["case"]["rw"]
            a = wall.get(f"2D_ex|{lam}|{rw:.2f}"); b = wall.get(f"2D_G|{lam}|{rw:.2f}")
            if a is None:
                continue
            e, se = v["extrap"]["E0"], v["extrap"]["se_E0"]
            rows.append(dict(rw=rw, main=rw in MAIN_WALLS[lam], Q_full=e, se=se, slope=v["extrap"]["slope"],
                             chi2=v["extrap"]["chi2"], Q_2D_ex=a["Q"], Q_2D_G=b["Q"] if b else None,
                             dQ_ex=e - a["Q"], dQ_G=(e - b["Q"]) if b else None,
                             wall_shift_2D_ex=a["Q"] - Qs, r_max=max(p["r_max"] for p in v["per_tau"]),
                             frac_wall_rejected=float(np.mean([p["frac_wall_rejected"] for p in v["per_tau"]]))))
        main = [r for r in rows if r["main"]]
        rec = dict(Q_2D_ex_siegert=Qs, gamma_2D_ex_siegert=gs, walls=rows)
        if main:
            m, se, chi2, dof = wmean([r["dQ_ex"] for r in main], [r["se"] for r in main])
            mg, seg, _, _ = wmean([r["dQ_G"] for r in main], [r["se"] for r in main])
            rec.update(dQ_ex_main=m, se_dQ_ex_main=se, chi2_walls=chi2, dof_walls=dof, dQ_G_main=mg, se_dQ_G_main=seg,
                       pass_0p01=abs(m) + 2 * se <= 0.01, pass_target_0p005=abs(m) + 2 * se <= 0.005)
        # wall-dependence (tail amplitude) estimator: Q_full(rw) = Qinf + A * wall_shift_2D(rw)
        if len(rows) >= 3:
            x = np.array([r["wall_shift_2D_ex"] for r in rows]); y = np.array([r["Q_full"] for r in rows])
            s = np.array([r["se"] for r in rows])
            Am = np.vstack([np.ones_like(x), x]).T / s[:, None]
            coef, *_ = np.linalg.lstsq(Am, y / s, rcond=None)
            cov = np.linalg.inv(Am.T @ Am)
            chi2 = float((((Am @ coef) - y / s) ** 2).sum())
            rec["tail_fit"] = dict(Qinf_full=float(coef[0]), se_Qinf=float(np.sqrt(cov[0, 0])), A=float(coef[1]),
                                   se_A=float(np.sqrt(cov[1, 1])), corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                                   chi2=chi2, dof=len(rows) - 2, dQinf_vs_siegert=float(coef[0] - Qs))
        # conversion with the 2D model's own d ln gamma/dQ
        dd = dl.get(("2D_ex", lam), [])
        if dd and main:
            d0 = float(np.mean(dd)); dsys = float(np.ptp(dd) / 2) if len(dd) > 1 else 0.0
            dlg = d0 * rec["dQ_ex_main"]
            rec["conversion"] = dict(dlngamma_dQ=d0, dlngamma_dQ_shape_spread=dsys, dln_gamma=dlg,
                                     se_dln_gamma=float(np.hypot(d0 * rec["se_dQ_ex_main"], dsys * rec["dQ_ex_main"])),
                                     rho=float(np.exp(dlg)))
        if wkb and f"{lam}" in wkb:
            wk = wkb[f"{lam}"]
            rec["wkb"] = dict(ratio_full_adiabatic_at_Qfull=wk["full_adiabatic_at_Qfull"]["ratio_to_2D_wkb"],
                              ratio_2D_channel_at_Qfull=wk["2D_channel_at_Qfull"]["ratio_to_2D_wkb"],
                              ratio_full_adiabatic_at_Q2D=wk["full_adiabatic_at_Q2D"]["ratio_to_2D_wkb"],
                              ratio_2D_plus_Bw=wk["2D_channel_plus_Bw"]["ratio_to_2D_wkb"],
                              wkb_over_siegert_2D=wk["wkb_over_siegert_2D"], dSdE_2D=wk["2D_lowest_channel"]["dSdE"],
                              S_2D=wk["2D_lowest_channel"]["S"])
        res[f"{lam}"] = rec
    S["lam_neg"] = res
    # ---------------- Delta m_c
    dm = {}
    lamv = [-0.010, -0.013]

    def interp(vals):
        return {p: float(np.interp(LAMB[p], lamv[::-1], vals[::-1], left=None, right=None)
                         if lamv[1] <= LAMB[p] <= lamv[0] else
                         vals[0] + (vals[1] - vals[0]) * (LAMB[p] - lamv[0]) / (lamv[1] - lamv[0])) for p in LAMB}
    try:
        lr = [res[f"{l}"]["conversion"]["dln_gamma"] for l in lamv]
        ls = [res[f"{l}"]["conversion"]["se_dln_gamma"] for l in lamv]
        li, si = interp(lr), interp(ls)
        dm["via_dQ_and_dlngdQ"] = {p: dict(lambda_b=LAMB[p], dln_gamma=li[p], se=si[p],
                                           dm_c_GeV=-li[p] / T.DLNI_DMT[p], se_dm_c_GeV=si[p] / T.DLNI_DMT[p])
                                   for p in LAMB}
    except (KeyError, TypeError):
        pass
    try:
        la = [np.log(max(res[f"{l}"]["tail_fit"]["A"], 1e-3)) for l in lamv]
        sa = [res[f"{l}"]["tail_fit"]["se_A"] / max(res[f"{l}"]["tail_fit"]["A"], 1e-3) for l in lamv]
        li, si = interp(la), interp(sa)
        dm["via_tail_amplitude_A"] = {p: dict(lambda_b=LAMB[p], ln_A=li[p], se=si[p], dm_c_GeV=-li[p] / T.DLNI_DMT[p],
                                              se_dm_c_GeV=si[p] / T.DLNI_DMT[p]) for p in LAMB}
    except (KeyError, TypeError):
        pass
    try:
        lw = [np.log(res[f"{l}"]["wkb"]["ratio_full_adiabatic_at_Qfull"]) for l in lamv]
        li = interp(lw)
        dm["via_wkb_full_adiabatic"] = {p: dict(lambda_b=LAMB[p], ln_rho=li[p], dm_c_GeV=-li[p] / T.DLNI_DMT[p])
                                        for p in LAMB}
    except (KeyError, TypeError):
        pass
    # ---------------- 3D collective-coordinate model (item 3(b))
    f3 = os.path.join(T.RES, "T14a_3d.json")
    if os.path.exists(f3):
        d3 = T.jload(f3)
        S["model_3D_zeta"] = {k: {kk: v.get(kk) for kk in ("Q3", "Q2", "dQ_3D_minus_2D", "ratio_3D_2D", "gamma2", "gamma3",
                                                          "dQ", "eta_opt", "zpe", "Nz", "Ns", "h", "time_s")}
                              for k, v in d3.items()}
        try:
            for zp in ("ref", "local"):
                lr = [np.log(d3[f"cap_{l}_{zp}"]["ratio_3D_2D"]) for l in lamv]
                li = interp(lr)
                dm[f"via_3D_zeta_{zp}"] = {p: dict(lambda_b=LAMB[p], ln_rho=li[p], dm_c_GeV=-li[p] / T.DLNI_DMT[p])
                                           for p in LAMB}
        except KeyError:
            pass
    S["delta_m_c"] = dm
    # ---------------- verdict against the brief's pass criteria
    try:
        dqs = {l: (res[f"{l}"]["dQ_ex_main"], res[f"{l}"]["se_dQ_ex_main"]) for l in lamv}
        rhos = {"conv": [res[f"{l}"]["conversion"]["rho"] for l in lamv],
                "wkb_full_adiabatic": [res[f"{l}"]["wkb"]["ratio_full_adiabatic_at_Qfull"] for l in lamv],
                "tail_A": [res[f"{l}"]["tail_fit"]["A"] for l in lamv]}
        if "model_3D_zeta" in S:
            rhos["3D_zeta_ref"] = [S["model_3D_zeta"][f"cap_{l}_ref"]["ratio_3D_2D"] for l in lamv]
            rhos["3D_zeta_local"] = [S["model_3D_zeta"][f"cap_{l}_local"]["ratio_3D_2D"] for l in lamv]
        S["verdict"] = dict(
            dQ=dqs, pass_abs_dQ_le_0p01=all(abs(m) + 2 * s <= 0.01 for m, s in dqs.values()),
            central_abs_dQ_le_0p005=all(abs(m) <= 0.005 for m, _ in dqs.values()),
            gamma_ratios=rhos, gamma_within_20pct=all(abs(x - 1) <= 0.2 for v in rhos.values() for x in v),
            dm_c_central_GeV={p: dm["via_dQ_and_dlngdQ"][p]["dm_c_GeV"] for p in LAMB},
            dm_c_envelope_GeV={p: max(abs(v[p]["dm_c_GeV"]) for k, v in dm.items() if k != "via_tail_amplitude_A")
                               for p in LAMB},
            reduced_model_error_le_0p1GeV=bool(max(abs(v[p]["dm_c_GeV"]) + 2 * v[p].get("se_dm_c_GeV", 0.0)
                                                   for k, v in dm.items() if k != "via_tail_amplitude_A"
                                                   for p in LAMB) <= 0.1),
            note="central = Delta Q x (d ln gamma/dQ)_2D; envelope over conversion/WKB(a)/3D (tail-A estimator has +-0.04 GeV stat)")
    except (KeyError, TypeError) as e:
        S["verdict"] = dict(error=repr(e))
    # ---------------- provenance hashes
    code = os.path.dirname(os.path.abspath(__file__))
    S["sha256"] = {fn: T.sha256(os.path.join(code, fn)) for fn in sorted(os.listdir(code)) if fn.endswith(".py")}
    S["sha256_inputs"] = {p: T.sha256(p) for p in (T.OLDSUM, T.B.PROFILE, os.path.join(T.B.S3DIR, "S3_prop.json"))}
    out = os.path.join(T.RES, "C151_summary.json")
    T.wjson(out, S)
    print(json.dumps(T.clean(S), indent=1)[:6000])


if __name__ == "__main__":
    main()
