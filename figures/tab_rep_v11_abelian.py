"""Print the LaTeX rows of the two tables of the report v1.1 abelian-Higgs subsection (Sec. 18) from ./data, with the
source tags used in the report, and the derived numbers quoted in the text.  Usage: python -B tab_rep_v11_abelian.py
"""
import csv
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "higgs" / "abelian_c171"
B = math.pi * math.sqrt(2.0)
ROWS = ["j00", "j04", "j06", "j08", "j10", "j12", "j14", "j16", "j19", "j21", "j23"]


def sci(v, nd):
    m, e = f"{v:.{nd}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def main():
    g = {r["name"]: r for r in csv.DictReader(open(DATA / "C171_gamma_table.csv", encoding="utf-8"))}
    print("% ---- Table v11-abelian-widths")
    for n in ROWS:
        r = g[n]
        lam, x, eps, G = float(r["Lambda"]), float(r["x=|Lambda|^-1/2"]), float(r["eps_exact"]), float(r["Gamma_exact"])
        rbo, rbod = float(r["Gamma_BO"]) / G, float(r["Gamma_BOD"]) / G
        b, aeff = float(r["b_exact"]), float(r["A_eff=Gamma*exp(S_asym)"])
        lam_s = f"{lam:.5f}" if abs(lam) < 0.1 else (f"{lam:.4f}" if abs(lam) < 0.5 else f"{lam:.1f}")
        gs = f"{G:.6f}" if G > 0.1 else f"{sci(G, 6)}"
        rbod_s = f"{rbod:.2f}" if rbod < 10 else (f"{rbod:.1f}" if rbod < 100 else f"{rbod:.0f}")
        aeff_s = f"{aeff:.0f}" if aeff < 100 else sci(aeff, 1)
        print(f"${lam_s}$ & ${x:.3f}$ & ${eps:.7f}$ & ${gs}$ & ${rbo:.3f}$ & ${rbod_s}$ & ${b:.3f}$ & ${aeff_s}$ \\\\ "
              f"% [num:C171 C171_gamma_table.csv {n}]")
    print("% ---- Table v11-abelian-thresholds")
    thr = list(csv.DictReader(open(DATA / "C171_thresholds.csv", encoding="utf-8")))
    res = json.load(open(DATA / "C171_results.json", encoding="utf-8"))
    ge = {int(t["N"]): t for t in res["thresholds_general_e"] if abs(t["e"] - 0.8733333) < 1e-4}
    for t in thr:
        N = float(t["N"])
        Ns = "10" if N == 10 else ("300" if N == 300 else f"10^{{{int(round(math.log10(N)))}}}")
        lc, le = float(t["Lambda_c_exact"]), float(t["Lambda_c_err"])
        bo = float(t["Lambda_c_BO"])
        bod = t["Lambda_c_BOD"]
        bod_s = "---" if bod in ("", "nan") else f"${float(bod):.4f}$"
        y = float(t["|Lambda_c|^1/2*lnN"])
        sl = res["thresholds"][thr.index(t)]["dsqrtLam_d_invlnN"]
        lce = ge.get(int(N))
        lce_s = f"${lce['lam_c']:.4f}$" if lce else "---"
        print(f"${Ns}$ & ${lc:.5f}$ & ${bo:.4f}$ & {bod_s} & ${y:.3f}$ & ${sl:.2f}$ & {lce_s} \\\\ "
              f"% [num:C171 C171_thresholds.csv N={t['N']}; thresholds_general_e]")
    print("% ---- derived numbers")
    J = [g[f"j{k:02d}"] for k in range(24)]
    frac = [(float(r["eps_exact"]) - float(r["eps_BO"])) / (float(r["eps_BOD"]) - float(r["eps_BO"])) for r in J]
    dBO = [float(r["eps_BO"]) - float(r["eps_exact"]) for r in J]
    dBOD = [float(r["eps_BOD"]) - float(r["eps_exact"]) for r in J]
    dbo = [float(r["b_BO"]) - float(r["b_exact"]) for r in J]
    dbod = [float(r["b_BOD"]) - float(r["b_exact"]) for r in J]
    dS = [(float(r["x=|Lambda|^-1/2"]), float(r["dS_BO(eps_exact)/dx"]) - float(r["b_exact"])) for r in J
          if r["dS_BO(eps_exact)/dx"] not in ("", "nan")]
    rw = [float(r["Gamma_exact/Gamma_WKB_BO_at_eps_exact"]) for r in J
          if r["Gamma_exact/Gamma_WKB_BO_at_eps_exact"] not in ("", "nan")]
    print("fraction of BO bracket:", min(frac), max(frac))
    print("eps_BO - eps:", min(dBO), max(dBO), " eps_BOD - eps:", min(dBOD), max(dBOD))
    print("b_BO - b:", min(dbo), max(dbo), " b_BOD - b:", min(dbod), max(dbod))
    print("max |dS/dx - b| for x >= 4:", max(abs(d) for xx, d in dS if xx >= 4),
          " for x >= 4.6:", max(abs(d) for xx, d in dS if xx >= 4.6))
    print("Gamma / Gamma_WKB(eps):", min(rw), max(rw))
    r21 = g["j21"]
    print("S_asym - S_BO(eps) at x=10:", float(r21["S_asym=4.443x"]) - float(r21["S_BO(eps_exact)"]))
    print("leading-order coefficient 3.78:", math.sqrt(2) / 3 * math.gamma(1 / 6) * math.gamma(0.5) / math.gamma(2 / 3)
          * (6.0 ** (1 / 3) / 1.5) ** 0.5)
    print("Langer BO barrier top at e=1: 0.75*1.5^(4/3) =", 0.75 * 1.5 ** (4 / 3))
    # thresholds: BO and BOD relative to exact
    rbo = [float(t["Lambda_c_BO"]) / float(t["Lambda_c_exact"]) - 1 for t in thr]
    rbod = [1 - float(t["Lambda_c_BOD"]) / float(t["Lambda_c_exact"]) for t in thr if t["Lambda_c_BOD"] not in ("", "nan")]
    print("Lambda_c_BO/Lambda_c - 1:", min(rbo), max(rbo), " shallower with BOD:", min(rbod), max(rbod))
    # Kemble width of the Langer BO model with its own Bohr-Sommerfeld energy, x >= 5
    kem = [(float(r["x=|Lambda|^-1/2"]), float(r["Gamma_WKB_BO_Kemble"]) / float(r["Gamma_exact"])) for r in J
           if float(r["x=|Lambda|^-1/2"]) >= 5]
    print("Gamma_Kemble(BS)/Gamma for x >= 5:", min(k for _, k in kem), max(k for _, k in kem))
    # barrier of U_BO above the level at the SM-like slope (e = 0.8733, lambda = -0.010)
    sm = res["SM_like"]["lam-0.010"]
    e, lam, eps_sm = sm["e"], sm["lam"], sm["E_phys"]
    U = lambda r: 1.5 * e * r - 1.0 / (8.0 * r * r) + lam / 4.0 * r ** 4
    rr = [0.5 + 0.0001 * k for k in range(200000)]
    umax = max(U(r) for r in rr)
    print(f"SM-like slope: e = {e:.4f}, lam = {lam}: U_BO max = {umax:.4f}, eps = {eps_sm:.4f}, barrier above level = "
          f"{umax - eps_sm:.4f}")
    print("abelian lambda_c(300) at e=0.8733 / SM static box -0.0117514921:",
          [t["lam_c"] for t in res["thresholds_general_e"] if t["N"] == 300 and t["e"] < 0.9][0] / -0.0117514921)


if __name__ == "__main__":
    main()
