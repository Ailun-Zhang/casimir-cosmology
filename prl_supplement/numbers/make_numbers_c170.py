"""Generate numbers_c170.tex for the restructured PRL from the C170 and C171 result files (single source of the new
numbers; the published numbers.tex of the code release v1.0.0 is not touched).

Every macro is computed from the archived outputs in results/ and printed
with the source and the unrounded value as a comment.  Usage:
    python -I -B make_numbers_c170.py [--out <PRL folder>]      (default: print only)
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"  # [supplement] results of this supplement
C170 = RES / "C170"
C171 = RES / "C171"
# code release v1.0.0 (read only): the Table I composite used for the model error of the static threshold
REL = Path(os.environ.get("DNEC_RELEASE", str(HERE.parents[2] / "08_release" / "v12_code_data")))


def pow10_bound(x):
    """smallest power of ten above |x|, as a TeX string (e.g. 6.9e-12 -> 10^{-11})"""
    return "10^{%d}" % math.ceil(math.log10(abs(x)))


def rcsv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader([ln for ln in fh if not ln.startswith("#")]))


def rjson(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(x, nd):
    s = f"{x:.{nd}f}"
    return s.replace("-", "-")


def build():
    m = {}  # name -> (tex value, unrounded, source)
    res = rjson(C170 / "C170_results.json")
    s1 = rjson(C170 / "S1" / "C170_S1_summary.json")
    A = s1["boxes"]["A"]
    fw = A["fits"]["window_lam_m0p018_m0p007"]["two_param"]
    a, b, rms = fw["a"], fw["b"], fw["rms"]
    lo, hi = -0.018, -0.007
    m["aFitA"] = (fmt(a, 3), a, "C170 S1 box A fit window [-0.018,-0.007] two_param.a")
    m["bFitA"] = (fmt(b, 3), b, "C170 S1 box A fit window two_param.b")
    m["resFitA"] = (fmt(rms, 2), rms, "C170 S1 box A fit window two_param.rms (ln Gamma)")
    m["lamFitLo"] = (fmt(lo, 3), lo, "fit window lower end (lambda)")
    m["lamFitHi"] = (fmt(hi, 3), hi, "fit window upper end (lambda)")
    m["sigEffA"] = (fmt(b / (2 * math.pi * math.sqrt(2) / 3), 2), b / (2 * math.pi * math.sqrt(2) / 3),
                    "b / (2 pi sqrt2 / 3)")
    lc = rcsv(C170 / "S1" / "C170_S1_lambda_c_N.csv")

    def lcv(box, P, N):
        return float([r for r in lc if r["box"] == box and r["P"] == P and r["N"] == N][0]["lambda_c"])
    dev = {}
    for N in ("100", "300", "1000", "10000", "100000"):
        cf = -(b / (a + math.log(float(N) / math.log(2)))) ** 2
        dev[N] = cf / lcv("A", "0.5", N) - 1
    small = max(abs(dev[N]) for N in ("100", "300", "1000", "10000"))
    m["closedAccSmall"] = (f"{math.ceil(small * 1000 - 1e-9) / 10:.1f}\\%", small,
                           "max |closed/direct-1|, N=1e2..1e4, box A, P=1/2 (rounded up)")
    m["closedAccLarge"] = (f"{math.ceil(abs(dev['100000']) * 1000 - 1e-9) / 10:.1f}\\%", dev["100000"],
                           "|closed/direct-1| at N=1e5 (rounded up)")
    for name, box, N in (("lcAhundred", "A", "100"), ("lcAthree", "A", "300"), ("lcAtenk", "A", "10000"),
                         ("lcBthree", "B", "300"), ("lcCthree", "C", "300"), ("lcDthree", "D", "300"),
                         ("lcEthree", "E", "300")):
        v = lcv(box, "0.5", N)
        m[name] = (fmt(v, 4), v, f"C170 S1 lambda_c box {box}, P=1/2, N={N}")
    m["lamTop"] = (fmt(A["narrow_range"]["lam"][0], 3), A["narrow_range"]["lam"][0],
                   "C170 S1 box A narrow-resonance range, lower end")
    # model error of the static threshold: zero-mode-model row of Table I (no inflation scale, no w0), converted with
    # |d lambda(mu_b)/d m_t| = 0.00659/GeV (the factor of lamcErrUp/Dn in numbers.tex)
    t1 = rjson(REL / "higgs" / "09_claims" / "expected" / "w" / "claims_run05" / "run01" / "table1_rebuild.json")
    up_gev, dn_gev = t1["composite"]["plus_no_vi"]["0.5"], t1["composite"]["minus_no_vi"]["0.5"]
    dl = 0.00659
    m["lcAthreeErrUp"] = (fmt(dn_gev * dl, 4), dn_gev * dl, "table1_rebuild composite minus_no_vi(0.5) x 0.00659 (m_c down = lambda_c up)")
    m["lcAthreeErrDn"] = (fmt(up_gev * dl, 4), up_gev * dl, "table1_rebuild composite plus_no_vi(0.5) x 0.00659")
    win = [p for p in A["table"] if -0.018 - 1e-9 <= p["lam"] <= -0.007 + 1e-9 and p["status"] in ("ok", "ok_broad")]
    e_lo, e_hi = min(-p["eps_minus_Utop"] for p in win), max(-p["eps_minus_Utop"] for p in win)
    s_lo, s_hi = min(p["S_wkb"] for p in win), max(p["S_wkb"] for p in win)
    m["topGapLo"] = (fmt(e_lo, 1), e_lo, "C170 S1 box A: U_top - eps, min over the fit window")
    m["topGapHi"] = (fmt(e_hi, 1), e_hi, "C170 S1 box A: U_top - eps, max over the fit window")
    m["SwkbLo"] = (f"{math.floor(s_lo):d}", s_lo, "C170 S1 box A: WKB exponent across the barrier, min over the fit window")
    m["SwkbHi"] = (f"{math.ceil(s_hi):d}", s_hi, "C170 S1 box A: WKB exponent, max over the fit window")
    ne = A["N_eff"]["P=0.5"]
    m["NeffHalf"] = (f"{round(ne['N_eff'], -1):.0f}", ne["N_eff"], "C170 S1 box A N_eff at P=1/2 (hazard lam_c)")
    m["NeffRatio"] = (fmt(ne["N_eff_over_eta"], 2), ne["N_eff_over_eta"], "N_eff / eta_inf")
    nsr = {r["box"]: float(r["Nstar"]) for r in rcsv(C170 / "S1" / "C170_S1_lambdaSM_Nstar.csv") if r["m_t"] == "172.6"}
    m["NstarA"] = (f"{nsr['A']:.0f}", nsr["A"], "C170 S1 N* = ln2/Gamma(lambda_SM), box A, m_t 172.6")
    m["NstarD"] = (f"{round(nsr['D'], -1):.0f}", nsr["D"], "C170 S1 N*, box D (10 mu_b)")
    m["NstarE"] = (f"{nsr['E'] / 1e5:.1f}\\times10^5", nsr["E"], "C170 S1 N*, box E (mu_b/1000)")
    cv = rjson(C170 / "S2" / "C170_S2_curves.json")
    zps = [v["dzp_dr"] for v in cv["slopes"]["1.00"].values()]
    red = [-v["ddV_dr"] / v["dzp_dr"] for v in cv["slopes"]["1.00"].values()]
    m["zpSlopeLo"] = (fmt(min(zps), 1), min(zps), "C170 S2 d(zero-point)/dr at kappa=1, r=1..5, min")
    m["zpSlopeHi"] = (fmt(max(zps), 1), max(zps), "C170 S2 d(zero-point)/dr at kappa=1, r=1..5, max")
    m["dVredLo"] = (f"{round(min(red) * 100):d}\\%", min(red), "C170 S2 slope reduction by dV at r=1..5, min")
    m["dVredHi"] = (f"{round(max(red) * 100):d}\\%", max(red), "C170 S2 slope reduction by dV at r=1..5, max")
    kv = res["S2"]["kappa1_vs_final_model"]["2D_Gdmcs_kappa1"]["dlam_c_vs_final"]
    m["crossShiftLc"] = (fmt(-kv, 4), -kv, "C170 S2: lambda_c(300) final model minus 2D_Gdmcs kappa=1 (cross terms)")
    s0reg = res["S0"]["regression"]
    gen = max(abs(v["generic_seed"]["rel_gamma"]) for v in s0reg.values()
              if isinstance(v, dict) and isinstance(v.get("generic_seed"), dict) and "rel_gamma" in v["generic_seed"])
    m["SzeroGeneric"] = (pow10_bound(gen), gen, "C170 S0: max relative width difference with generic seeds (bound)")
    # static vs hazard threshold at mu_b, N = eta_inf (archived run05 u = 0 widths)
    rows = res["S0"]["static_threshold_archived_u0"]["rows"]
    sh = [rows[k]["shallower_by"] for k in ("P1=0.9", "P1=0.5", "P1=0.1")]
    m["staticHazardDiff"] = (f"{min(sh):.4f}\\text{{--}}{max(sh):.4f}".replace("0.0003\\text{--}0.0004", "0.0003\\text{--}0.0004"),
                             sh, "C170 S0 static threshold shallower than hazard threshold (P1 = 0.9/0.5/0.1)")
    # kappa scan (2D model with the transported DMC shape, no cross terms)
    bd = res["S2"]["boundaries"]["2D_Gdmcs"]
    k0 = bd["kappa_thr_N300_P05"]["kappa"]
    m["kappaZero"] = (fmt(k0, 2), k0, "C170 S2 kappa at which lambda_c(N=300, P=1/2) -> 0- (2D_Gdmcs)")
    st = bd["lambda0_status"]
    ka, kb = bd["kappa_min_bracket"]
    da = st[f"{ka:.4f}"]["eps1D_minus_Utop"]
    db = st[f"{kb:.4f}"]["eps1D_minus_Utop"]
    kmin = ka + (kb - ka) * da / (da - db)
    m["kappaMin"] = (fmt(kmin, 2), kmin, "C170 S2 kappa below which no narrow resonance exists at any lambda <= 0")
    kap = [r for r in rcsv(C170 / "S2" / "C170_S2_lambda_c_kappa.csv") if r["model"] == "2D_Gdmcs" and r["lambda_c_N300_P05"]]
    pts = sorted((float(r["kappa"]), float(r["lambda_c_N300_P05"])) for r in kap)
    for kk in ("0.75", "1.0", "1.5"):
        v = [l for k, l in pts if abs(k - float(kk)) < 1e-9][0]
        m["lcKappa" + {"0.75": "ThreeQ", "1.0": "One", "1.5": "OneHalf"}[kk]] = (fmt(v, 4), v, f"C170 S2 lambda_c kappa={kk}")
    # coupling scans (final model, all sectors), P = 1/2, N = 300
    sc = rcsv(C170 / "S3" / "C170_S3_coupling_scan.csv")
    for g, name in (("g2", "lcSthirtyGtwo"), ("gY", "lcSthirtyGY"), ("yt", "lcSthirtyYt")):
        base = float([r for r in sc if r["coupling"] == g and r["P"] == "P=0.5" and float(r["s"]) == 1.0][0]["lambda_c"])
        devs = [abs(float(r["lambda_c"]) / base - 1) for r in sc if r["coupling"] == g and r["P"] == "P=0.5"]
        m[name] = (f"{round(max(devs) * 100):d}\\%", max(devs), f"C170 S3 max |lambda_c/lambda_c(1)-1| over s=0.7..1.3, {g}")
    # C171 abelian Higgs model (exact two-dimensional zero-mode problem)
    r171 = rjson(C171 / "C171_results.json")
    t = r171["table"]
    lo = min(t, key=lambda r: r["Lam"])
    hi = max(t, key=lambda r: r["Lam"])
    m["abelLamMin"] = (fmt(lo["Lam"], 0), lo["Lam"], "C171 most negative Lambda = lambda/e^2")
    m["abelLamMax"] = (fmt(hi["Lam"], 4), hi["Lam"], "C171 least negative Lambda")
    m["abelSlopeLo"] = (fmt(lo["b_exact"], 1), lo["b_exact"], "C171 local exponent b at the most negative Lambda")
    m["abelSlopeHi"] = (fmt(hi["b_exact"], 1), hi["b_exact"], "C171 local exponent b at the least negative Lambda")
    m["abelSlopeWKB"] = (fmt(math.pi * math.sqrt(2), 2), math.pi * math.sqrt(2), "deep-well WKB exponent 2.962 x 3/2")
    ratios = [1 / r["ratio_exact_BO"] for r in t]
    m["abelBOdev"] = (f"{math.ceil(max(abs(x - 1) for x in ratios) * 10 - 1e-9) * 10:d}\\%",
                      [min(ratios), max(ratios)], "C171 Gamma_BO/Gamma_exact range, max deviation (rounded up)")
    capb = [abs(c["rel_diff_G1"]) for c in r171["cap"] if c["name"].endswith("_b")]
    m["abelCAP"] = (pow10_bound(max(capb)), max(capb), "C171 CAP (setting b) vs ECS, max relative width difference (bound)")
    lc171 = [x for x in r171["thresholds"] if x["N"] == 300][0]
    m["abelLamcThree"] = (fmt(lc171["Lam_c_exact"], 3), lc171["Lam_c_exact"], "C171 Lambda_c(N=300) = lambda_c/e^2")
    # thresholds at the SM-like slope sigma_0 = 3e/2 = 1.31 (e = 0.873)
    ge = r171["thresholds_general_e"]
    row = None
    if isinstance(ge, dict):
        for k, v in ge.items():
            if isinstance(v, (list, dict)) and abs(float(k) - 0.8733) < 2e-3 if _isnum(k) else False:
                row = v
    if row is None and isinstance(ge, list):
        row = [x for x in ge if abs(x.get("e", 0) - 0.8733) < 2e-3]
    lam_sm300 = None
    for x in (row if isinstance(row, list) else (row.values() if isinstance(row, dict) else [])):
        if isinstance(x, dict) and x.get("N") == 300:
            lam_sm300 = x.get("lambda_c", x.get("lam_c"))
    if lam_sm300 is not None:
        m["abelLamcSM"] = (fmt(lam_sm300, 3), lam_sm300, "C171 lambda_c(N=300) at e = 0.873 (sigma_0 = 1.31)")
    hol = [r["holonomy"]["e=0.873333|ew|>pi"] for r in t if isinstance(r.get("holonomy"), dict)
           and "e=0.873333|ew|>pi" in r["holonomy"]]
    hmax = max(hol)
    m["abelHolonomy"] = (f"{math.ceil(hmax * 1e4 - 1e-9):d}\\times10^{{-4}}", hmax,
                         "C171 max probability in the well beyond |e w| = pi at e = 0.873 (rounded up)")
    return m, {}


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def tex(m):
    lines = ["% numbers_c170.tex -- generated by 04_calculations/code/20261019_prl_independence/make_numbers_c170.py",
             "% from 04_calculations/results/20261019_prl_independence/{C170,C171}; do not edit by hand.",
             "% Static box (no gravity), gauge zero-mode strength kappa, coupling scans; abelian Higgs check."]
    for k, (v, raw, src) in m.items():
        lines.append(f"\\providecommand{{\\{k}}}{{{v}}}% {src}; value {raw}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    m, bl = build()
    out = tex(m)
    print(out)
    if bl:
        print("% C171 local slopes available for", len(bl), "Lambda values")
    if args.out:
        (args.out / "numbers_c170.tex").write_text(out, encoding="utf-8")
        print("written", args.out / "numbers_c170.tex")


if __name__ == "__main__":
    main()
