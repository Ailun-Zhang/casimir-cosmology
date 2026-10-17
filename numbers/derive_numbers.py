# PROVENANCE: new tool written 2026-10-16 by rev-code (R18 revision, MH-01/R2) for the v12 code/data package.
"""Claim-layer numbers that are simple functions of archived package outputs and are not stored by any production
script.  They are computed here from the package files named below and written to numbers/derived_numbers.json,
which numbers_v12.json names as their source (checked by check_numbers.py).

    python numbers/derive_numbers.py            # write numbers/derived_numbers.json
    python numbers/derive_numbers.py --check    # exit 1 if numbers/derived_numbers.json is not up to date

xiHminEW      electroweak-scale image of the conservative xi_H(mu_b) at m_t = 172.60 GeV (upper end of the spread of
              xi_c, spread_all[1]), with the one-loop map that higgs/07_realtime_xi/xi_compare.py uses for xi_c_EW:
              xi_EW = 1/6 + (xi(mu_b) - 1/6) / Xi(mu_b)
abs_dlam_dmt  |d lambda(mu_b)/d m_t| from the two safety margins at P1 = 1/2: margin_lambda / margin_mc
              (registry R5-030 / R5-031)
lamcErrUp     model error of lambda_c at P1 = 1/2, upper = (lower model error of m_c, Table 1 composite) x abs_dlam_dmt
lamcErrDn     model error of lambda_c at P1 = 1/2, lower = (upper model error of m_c) x abs_dlam_dmt
A_eff         prefactor of the closed form of the threshold, ln2 exp(b_eff |lambda_c|^(-1/2)) / eta_inf (higgs/10 C156_eta)
(lambda_c falls as m_c rises, so the upper error of lambda_c comes from the lower error of m_c.)
The output holds no time stamp, so it is byte-stable; every input is given as file#JSON-pointer with its value.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(HERE, "derived_numbers.json")
XI = "higgs/07_realtime_xi/expected/w/claims_run05/run01/xi_run05_summary.json"
C154 = "higgs/09_claims/expected/w/claims_run05/run01/C154_summary.json"
T1 = "higgs/09_claims/expected/w/claims_run05/run01/table1_rebuild.json"
UNIV = "higgs/10_universality/expected/r3/universality/run01/C156_eta.json"
SH2 = "shape/02_realtime_theorem/expected/r3/shape_hardening/run02/C153_run02_summary.json"
SH1 = "shape/02_realtime_theorem/expected/r3/shape_hardening/run01/C153_nl_summary.json"


def tstar_nodes():
    """Survival nodes of the leading-order theorem: run02 (tau <= 0.35, N = 2048; 0.15 with N = 1024) and the run01
    node tau = 0.40 (N = 512), as in the report (c15) and D125 (note a)."""
    rows = json.load(open(os.path.join(PKG, *SH2.split("/")), encoding="utf-8"))["lo_t"]["rows"]
    grid = json.load(open(os.path.join(PKG, *SH1.split("/")), encoding="utf-8"))["grid"]
    nodes = [dict(tau=r["tau"], n=r["A"]["n"], k=r["A"]["k"], file=SH2, pointer="/lo_t/rows/%d/A" % i)
             for i, r in enumerate(rows)]
    i40 = next(i for i, g in enumerate(grid) if abs(g["tau"] - 0.40) < 1e-9)
    g = grid[i40]["two_branch"]
    nodes.append(dict(tau=0.40, n=g["n"], k=g["k"], file=SH1, pointer="/grid/%d/two_branch" % i40))
    return sorted(nodes, key=lambda x: x["tau"])


def tstar(level, nodes, draws=20000, seed=20261017):
    """Threshold t_* where the survival fraction falls through `level`: linear interpolation between neighbouring
    nodes; errors from Monte Carlo draws of the Jeffreys posteriors Beta(k + 1/2, n - k + 1/2) of the node fractions
    (report Sec. 15, method of the thresholds)."""
    import numpy as np
    tau = np.array([x["tau"] for x in nodes])
    p0 = np.array([x["k"] / x["n"] for x in nodes])

    def cross(p):
        for i in range(len(p) - 1):
            if p[i] >= level > p[i + 1]:
                return tau[i] + (p[i] - level) / (p[i] - p[i + 1]) * (tau[i + 1] - tau[i]), i
        return None, None
    c0, i0 = cross(p0)
    rng = np.random.default_rng(seed)
    P = np.stack([rng.beta(x["k"] + 0.5, x["n"] - x["k"] + 0.5, draws) for x in nodes], axis=1)
    ts, seg = [], {}
    for row in P:
        c, i = cross(row)
        if c is not None:
            ts.append(c)
            key = "%.2f-%.2f" % (tau[i], tau[i + 1])
            seg[key] = seg.get(key, 0) + 1
    ts = np.array(ts)
    q = np.quantile(ts, [0.025, 0.16, 0.5, 0.84, 0.975])
    return {"central": float(c0), "central_segment": "%.2f-%.2f" % (tau[i0], tau[i0 + 1]), "draws": draws,
            "draws_with_crossing": int(len(ts)), "crossing_segments": seg, "median": float(q[2]),
            "p16": float(q[1]), "p84": float(q[3]), "half_width_68": float((q[3] - q[1]) / 2),
            "p2.5": float(q[0]), "p97.5": float(q[4])}


def tstar_block():
    nodes = tstar_nodes()
    t90, t99 = tstar(0.9, nodes), tstar(0.99, nodes)
    arch = json.load(open(os.path.join(PKG, *SH2.split("/")), encoding="utf-8"))["lo_t"]["t_star_linear_interp"]["0.99"]
    return {"tstar90": dict(t90, value=t90["central"], level=0.9,
                            method="linear interpolation between neighbouring nodes; 68% interval from 2e4 draws of the "
                                   "Jeffreys posteriors Beta(k+1/2, n-k+1/2) of the node survival fractions (fixed seed)",
                            nodes=[{k: v for k, v in x.items()} for x in nodes]),
            "tstar99_check": dict(t99, level=0.99, archived_central=arch["central"],
                                  archived_mc_p2_5=arch["mc_binomial"]["p2.5"], archived_mc_p97_5=arch["mc_binomial"]["p97.5"],
                                  note="cross-check of the method against the archived t_*(99%) = 0.237 (95% 0.225-0.252)")}

def value(file, pointer):
    """The number at a JSON pointer (RFC 6901) of a package file; key-exact."""
    with open(os.path.join(PKG, *file.split("/")), encoding="utf-8-sig") as f:
        cur = json.load(f)
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        raise ValueError("%s#%s is not a number" % (file, pointer))
    return float(cur)


def inp(file, pointer):
    return {"file": file, "pointer": pointer, "value": value(file, pointer)}


def compute():
    xi_cons, xi_map = inp(XI, "/run05/172.60/spread_all/1"), inp(XI, "/run05/172.60/Xi_mu_b")
    m_lam, m_mc = inp(C154, "/relative_form/margin_lambda"), inp(C154, "/relative_form/margin_mc05_minus_172p60")
    e_lo, e_hi = inp(T1, "/composite/minus/0.5"), inp(T1, "/composite/plus/0.5")
    slope = m_lam["value"] / m_mc["value"]
    b_eff, lam_c, eta = inp(UNIV, "/b_eff_hazard/P1=0.5/b_eff"), inp(UNIV, "/b_eff_hazard/P1=0.5/lam_c"), inp(UNIV, "/eta_inf")
    return {
        "meta": {"generator": "numbers/derive_numbers.py",
                 "note": "derived claim-layer numbers (R18 MH-01/R2); sources of xiHminEW, lamcErrUp, lamcErrDn in "
                         "numbers_v12.json; A_eff for the report and PRD (R18 A15); regenerate with python numbers/derive_numbers.py"},
        "xiHminEW": {"value": 1 / 6 + (xi_cons["value"] - 1 / 6) / xi_map["value"],
                     "formula": "1/6 + (xi_cons - 1/6) / Xi_mu_b",
                     "meaning": "electroweak-scale image of the conservative xi_H(mu_b) at m_t = 172.60 GeV",
                     "inputs": {"xi_cons": xi_cons, "Xi_mu_b": xi_map}},
        "abs_dlam_dmt": {"value": slope, "formula": "margin_lambda / margin_mc",
                         "meaning": "|d lambda(mu_b)/d m_t| in 1/GeV, from the safety margins at P1 = 1/2",
                         "inputs": {"margin_lambda": m_lam, "margin_mc": m_mc}},
        "lamcErrUp": {"value": e_lo["value"] * slope, "formula": "err_mc_lower x abs_dlam_dmt",
                      "meaning": "model error of lambda_c at P1 = 1/2, upper",
                      "inputs": {"err_mc_lower": e_lo, "abs_dlam_dmt": {"file": "numbers/derived_numbers.json",
                                                                        "pointer": "/abs_dlam_dmt/value",
                                                                        "value": slope}}},
        "lamcErrDn": {"value": e_hi["value"] * slope, "formula": "err_mc_upper x abs_dlam_dmt",
                      "meaning": "model error of lambda_c at P1 = 1/2, lower",
                      "inputs": {"err_mc_upper": e_hi, "abs_dlam_dmt": {"file": "numbers/derived_numbers.json",
                                                                        "pointer": "/abs_dlam_dmt/value",
                                                                        "value": slope}}},
        "A_eff": {"value": math.log(2) * math.exp(b_eff["value"] / math.sqrt(abs(lam_c["value"]))) / eta["value"],
                  "formula": "ln2 * exp(b_eff |lambda_c|^(-1/2)) / eta_inf",
                  "meaning": "effective prefactor of the closed form |lambda_c|^(1/2) = b_eff / ln(A_eff eta_inf / ln 2) at "
                             "P1 = 1/2 (D128 model of higgs/10), fixed by the computed threshold (report Sec. 7.6)",
                  "inputs": {"b_eff": b_eff, "lambda_c": lam_c, "eta_inf": eta}},
    } | tstar_block()


def render():
    return json.dumps(compute(), indent=1, ensure_ascii=False) + "\n"


def up_to_date():
    return os.path.exists(OUT) and open(OUT, encoding="utf-8").read() == render()


def main():
    if "--check" in sys.argv:
        ok = up_to_date()
        print("%s %s" % ("identical" if ok else "DIFFERS  ", os.path.relpath(OUT, PKG)))
        return 0 if ok else 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(render())
    d = compute()
    print("wrote", os.path.relpath(OUT, PKG))
    for k in ("xiHminEW", "abs_dlam_dmt", "lamcErrUp", "lamcErrDn", "A_eff"):
        print("  %-13s %.6g   (%s)" % (k, d[k]["value"], d[k]["formula"]))
    for k in ("tstar90", "tstar99_check"):
        x = d[k]
        print("  %-13s central %.5f median %.5f 68%% [%.5f, %.5f] half %.5f 95%% [%.5f, %.5f] segments %s" % (
            k, x["central"], x["median"], x["p16"], x["p84"], x["half_width_68"], x["p2.5"], x["p97.5"], x["crossing_segments"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
