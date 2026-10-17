# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_matching/C138_report.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] D110 report tables from the C138 JSON outputs.

C138 report: error budget on the critical m_t, regression table, Wolfram receipts, STATUS (D110)."""
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = _zp.w("r2/higgs_matching/run01")


def J(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    H = J("C138_hazard.json")["sets"]
    S = J("C138_static.json")
    X = J("C138_cross.json")
    W = J("wolfram_results.json")
    K = J("C138_checks.json")
    cr = lambda k: H[k]["crit"]  # noqa: E731
    base = "MC|c=1|gphi=None|ct=None|as=0.1180|main"
    c0 = cr(base)
    d = lambda k, t: cr(k)[t] - c0[t]  # noqa: E731
    keys = {t: k for k in H for t in [k]}
    del keys
    out = {"central_MC+CW_mu=1/L": c0}
    rows = {}
    for t in ("P1=0.9", "P1=0.5"):
        mu = [d("MC|c=%s|gphi=C|ct=None|as=0.1180|mu" % c, t) for c in ("0.5", "2", "2pi", "4pi")]
        top = [d("MC|c=1|gphi=None|ct=%s(yt only)|as=0.1180|top" % c, t) for c in ("0.5", "2.0")]
        top_bad = [d("MC|c=1|gphi=None|ct=%s|as=0.1180|var" % c, t) for c in ("0.5", "2.0")]
        alt = {m: d("%s|c=1|gphi=None|ct=None|as=0.1180|var" % m, t) for m in ("MU", "ML", "MU_CW", "ML_CW", "MUPS")}
        dl = lambda tag, mod: X["SM"]["lambda_c"][mod][t] - X["SM"]["lambda_c"]["MC"][t]  # noqa: E731
        dlam_dmt = -0.006564
        cross = [dl("SM", "MCX_rs1.0") / dlam_dmt, dl("SM", "MCX_rs1.5") / dlam_dmt]
        rows[t] = {
            "gauge_split_within_MC (BO diagonal + induced mass, estimate)": 0.002,
            "mu_variation c in [1/2,2] (gamma_phi^C)": [min(mu[:2] + [0]), max(mu[:2] + [0])],
            "mu_variation c in [1/2,2pi] (conservative)": [min(mu[:3] + [0]), max(mu[:3] + [0])],
            "mu_variation c=4pi (info)": mu[3],
            "CW_constant (MS-bar consistent)": 0.0,
            "CW_constant_if_3/2_for_vectors (inconsistent, info)": d("MC_v32|c=1|gphi=None|ct=None|as=0.1180|var", t),
            "top_threshold y_t scale c_t in [1/2,2] (RG-consistent)": top,
            "top_scale_with_log (RG-inconsistent, info)": top_bad,
            "Goldstone_first_order_minus_exact": d("MC1|c=1|gphi=None|ct=None|as=0.1180|var", t),
            "numerics: 11 vs 29 nodes": d("MC|c=1|gphi=None|ct=None|as=0.1180|var", t),
            "NOT INCLUDED zero-mode/nonzero-mode cross term (Higgs part, estimate)": cross,
            "superseded_alternatives_minus_MC": alt,
        }
        q = [0.002, max(abs(x) for x in mu[:3]), max(abs(x) for x in top),
             abs(rows[t]["Goldstone_first_order_minus_exact"]), 0.002]
        rows[t]["quadrature_matching_total (conservative mu)"] = math.sqrt(sum(x * x for x in q))
    out["error_budget_GeV"] = rows
    s1, s2 = cr("MC|c=1|gphi=None|ct=None|as=0.1171|main"), cr("MC|c=1|gphi=None|ct=None|as=0.1189|main")
    out["dmc_dalpha_s"] = {t: (s2[t] - s1[t]) / 0.0018 for t in c0}
    out["mc_alpha_s_0.1171_0.1189"] = {t: [s1[t], s2[t]] for t in c0}
    reg = {r["mt"]: r["P1"] for r in H["MU|c=1|gphi=None|ct=None|as=0.1180|main"]["rows"]}
    out["regression"] = dict(
        P1_MU={"172.0": [reg[172.0], 0.978], "172.2": [reg[172.2], 0.906], "172.6": [reg[172.6], 0.477]},
        gamma_static_v11={m: [S["table"]["v11"][m]["-0.01"]["gamma"], S["regression_D108"]["%s -0.010" % m]]
                          for m in ("MU", "MU_CW", "ML")})
    out["P1_MC_vs_mt"] = {r["mt"]: r["P1"] for r in H[base]["rows"]}
    out["lambda_b_vs_mt"] = {r["mt"]: r["lam_b"] for r in H[base]["rows"]}
    out["static_MC"] = dict(gamma_v11={k: v["gamma"] for k, v in S["table"]["v11"]["MC"].items()},
                            gamma_SM={k: v["gamma"] for k, v in S["table"]["SM"]["MC"].items()},
                            lambda_c=S["lambda_c"])
    py = S["table"]["v11"]["MC"]["-0.01"]
    wg = float(W["MC_v11_lam-0.010_gamma"])
    out["wolfram_gamma_check"] = dict(wolfram=wg, python=py["gamma"], rel=wg / py["gamma"] - 1,
                                      wolfram_Q=float(W["MC_v11_lam-0.010_Q"]), python_Q=py["Q"])
    with open(os.path.join(RES, "C138_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    code = open(os.path.join(HERE, "C138_wolfram.wl"), encoding="utf-8").read()
    rec = dict(engine="wolframscript -file (Mathematica 15.0; env WOLFRAMSCRIPT)", script="C138_wolfram.wl", code=code,
               stdout=open(os.path.join(RES, "wolfram_stdout.txt"), encoding="utf-8", errors="replace").read(),
               stderr=open(os.path.join(RES, "wolfram_stderr.txt"), encoding="utf-8", errors="replace").read(),
               results=W, python_comparison=out["wolfram_gamma_check"],
               notes=["NDSolve::ndsz / dmval warnings occurred for intermediate FindRoot iterates (far from the pole);"
                      " the converged root agrees with the Python Numerov/Siegert value (see python_comparison).",
                      "transverse/coulomb_I outputs are identically zero after Log[m2/mu^2] = Log[m2] - 2 Log[mu]."])
    with open(os.path.join(RES, "wolfram_receipts.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("central_MC+CW_mu=1/L", "dmc_dalpha_s", "wolfram_gamma_check",
                                          "regression")}, indent=1))
    print(json.dumps(out["error_budget_GeV"], indent=1))
    print("C(r):", {k: round(v["rC"], 4) for k, v in K["C_shift"]["SM_mub"].items()})


if __name__ == "__main__":
    main()
