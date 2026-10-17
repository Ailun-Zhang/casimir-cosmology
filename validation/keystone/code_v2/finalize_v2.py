# keystone-blind V2: assemble v2/BLIND_RESULTS_V2.json (timestamp + code SHA-256)
import os, sys, json, glob, hashlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
J = lambda f: json.load(open(os.path.join(HERE, f)))
S = J("static_v2.json"); H = J("hazard_v2_summary.json"); T = J("twisted_selftest.json"); ST = J("stability.json"); CK = J("checks_v2.json")
R = dict(agent="keystone-blind", task="V2 blind zero-mode/nonzero-mode cross term", timestamp_utc=datetime.datetime.utcnow().isoformat() + "Z",
         units="L=1; gamma=-2ImE; curvatures in coupling-absorbed units E=(1/2) kappa c^2 (c=g x, u=gY w)",
         scheme="twist approximation: constant Cartan background (c3,u) = momentum shift t3 c3 + y u per field component with T3 r-dependent masses; "
                "periodic bosons with n=0 slow modes excluded, AAA fermions chirality-wise; Coulomb modes carry Goldstone charges; direction-averaged (s-wave); "
                "SU(2) x-fluctuation curvature (kappa_1 ~= kappa_33) in the conditional Gaussian functional (re-minimised); U(1)_Y exact along (m(r) w, w); "
                "SU(3)-top via SU(3) Gaussian with r-dependent curvature (variational)")
R["anchors_r0"] = dict(kappa_P0_per_real_dof=T["kappa_P0"], kappa_A0_per_real_dof=T["kappa_A0"], SU3=T["SU3"], SU2=T["SU2"], UY=T["UY"],
                       dQ_with_v11_spreads=T["dQ"], v11=dict(SU3=-0.393, SU2=-0.242, UY=0.700, dQ=[-0.72, -0.17, 0.29]))
R["kappa_mass_dependence"] = "linear in m^2: dkappa/dm^2 = 1/(24 pi^2) per real dof (periodic) at m=0; Cartan kappa(r) profile in checks"
R["kappa_profile"] = CK["kappa_profile"]; R["su3t_profile"] = CK["su3t_profile"]
R["stability"] = dict(higgs_only=[dict(r=x["r"], w_curvature=x["total"]) for x in ST["higgs"]], all_fields=[dict(r=x["r"], w_curvature=x["total"]) for x in ST["all"]],
                      verdict="Higgs-only: soft w unstable for r < ~0.78 (U2 + kappa_mean/2 < 0; exact V_Y decreasing in s); all fields: stable for all r (kappa_mean/2 = +0.068 at small r, AAA fermions)")
st = {}
for lam in ["-0.008", "-0.01", "-0.0143"]:
    base = S[f"none_{lam}"]["2D"]["gamma"]; row = {}
    for cfg in ["none", "su3t", "su2", "uy", "all"]:
        d = S[f"{cfg}_{lam}"]
        row[cfg] = dict(Q_2D=d["2D"]["Q"], gamma_2D=d["2D"]["gamma"], ratio_2D=d["2D"]["gamma"]/base, Q_1D=d["V1D"]["Q"], gamma_1D=d["V1D"]["gamma"],
                        Q_BOlow=d["Vlow"]["Q"], gamma_BOlow=d["Vlow"]["gamma"], bracket_ok=d["2D"]["bracket_ok"])
    st[f"lambda_{lam}"] = row
st["local_lambda0_2D"] = dict(none=S["local0_none"], all=S["local0_all"])
R["static_v11_couplings"] = st
b = H["base_2D_G"]; hz = dict(base_2D_G={P: b[P]["mc"] for P in ["P0.9", "P0.5", "P0.1"]})
for cfg, res in H["cfgs"].items():
    hz[cfg] = {P: dict(mc=res["2D"][P]["mc"], shift=res["2D"][P]["mc"] - b[P]["mc"], d_quad=res["2D"][P]["d_quad"]) for P in ["P0.9", "P0.5", "P0.1"]}
    hz[cfg].update({k: v for k, v in res["2D"].items() if k.startswith("P1_")})
    hz[cfg]["V1D_mc"] = {P: res["V1D"][P]["mc"] for P in ["P0.9", "P0.5", "P0.1"]} if "V1D" in res else None
hz["sum_of_parts_shift"] = {P: sum(hz[c][P]["shift"] for c in ["su3t", "su2", "uy"]) for P in ["P0.9", "P0.5", "P0.1"]}
R["hazard_alpha_s_0.1180"] = hz
R["checks"] = dict(
    note="lambda=-0.010, v11 couplings, all parts", all_base=CK["all_base"], dr0_014=CK["all_dr0.014_Ns24_Om0.35"], Ns32=CK["all_dr0.02_Ns32_Om0.25"], Ns20=CK["all_dr0.02_Ns20_Om0.45"],
    cap_boxes={k: v for k, v in CK.items() if k.startswith("all_cap_")}, su2_kappa_massless_variant=dict(base=CK["su2_base"]["gamma"], variant=CK["su2_kappa_massless"]["gamma"]),
    su3_c2_gauss_with_kappa0=CK["su3_c2_gauss"], twisted_selftest=T)
R["code_sha256"] = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(os.path.join(HERE, "*.py")))}
R["t3_code_used_sha256"] = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(os.path.join(HERE, "..", "work", "*.py")))}
json.dump(R, open(os.path.join(HERE, "BLIND_RESULTS_V2.json"), "w"), indent=1, default=float)
for cfg in ["su3t", "su2", "uy", "all"]:
    print(cfg, {P: (round(hz[cfg][P]["mc"], 4), round(hz[cfg][P]["shift"], 4)) for P in ["P0.9", "P0.5", "P0.1"]}, {k: round(v, 4) for k, v in hz[cfg].items() if k.startswith("P1_") and v})
print("sum of parts", {k: round(v, 4) for k, v in hz["sum_of_parts_shift"].items()})
for lam in ["-0.008", "-0.01", "-0.0143"]:
    print(lam, {c: round(st[f"lambda_{lam}"][c]["ratio_2D"], 4) for c in ["su3t", "su2", "uy", "all"]})
print("written BLIND_RESULTS_V2.json")
