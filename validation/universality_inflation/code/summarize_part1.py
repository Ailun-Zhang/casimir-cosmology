import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "part1")
for fn in sorted(glob.glob(os.path.join(OUT, "*.json"))):
    name = os.path.basename(fn)[:-5]
    if name == "checks":
        continue
    d = json.load(open(fn, encoding="utf-8"))
    if "error" in d:
        print(name, "ERROR", d["error"]); continue
    inf = d["info"]; pk = d["peak"]
    Npk = inf.get("N_peak", inf.get("N_before_end_peak"))
    Hpk = inf.get("H_peak", inf.get("H_peak_GeV"))
    print(f"{name:24s} xi_EW={d['xi_EW']:.5f} xi(mu_b)={d['xi_mu_b']:.5f} lnN={d['lnN_at_root']:+.2e} "
          f"Npk={Npk:.2f} Hpk={Hpk:.3e} Bpk={inf['B_peak']:.2f} xi*={pk['xi_mustar']:.5f} lam*={pk['lam_mustar']:.5f} "
          f"mu*={pk['mu_star']:.3e} hb/H={pk['h_bar']/Hpk:.2f} f10={inf.get('frac_first10', float('nan')):.1e} t={d['seconds']:.0f}s")
