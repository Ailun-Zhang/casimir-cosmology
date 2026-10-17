"""Collect the data of the Higgs (H-group) figures into ./data/higgs and record their provenance.

Two kinds of entries are written to ./data/SOURCES_higgs.json:
  copy       archived production results, copied byte for byte (original path relative to the project root + SHA-256);
  generated  small tables evaluated with the production code (SM running couplings, the reduced-model potential and
             two-dimensional eigenfunctions); the entry lists the generator stage, the production modules it imports
             (path + SHA-256) and the checks against archived keys.
The figure scripts read only ./data/higgs. This script is needed only to rebuild that folder from the research tree
(stages 'copy' and 'gen_*' need the project tree with 04_calculations/; the released package ships the outputs).

usage: python -B prepare_data_higgs.py [copy] [gen_running] [gen_potential] [gen_wavefunction] [all]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the (read-only) production code folders
sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
HD = DATA / "higgs"
SOURCES = DATA / "SOURCES_higgs.json"
ROOT = HERE.parents[2]
R5 = "04_calculations/results/20261015_claude_v12_writing/claims_run05/run01"
CR3 = "04_calculations/results/20261014_claude_v12_r3"
CR2 = "04_calculations/results/20261013_claude_v12_r2"
CC3 = "04_calculations/code/20261014_claude_v12_r3"

# (destination relative to data/higgs, source relative to ROOT, purpose)
COPIES = [
    ("claims_run05/fig2a_P1_grid_mt_alphas.csv", f"{R5}/fig2a_P1_grid_mt_alphas.csv", "P1(m_t) at alpha_s grid (run05 centre): prl_fig2/prd_criterion panel a"),
    ("claims_run05/fig2a_contours_mt_alphas.csv", f"{R5}/fig2a_contours_mt_alphas.csv", "critical m_t(alpha_s) for P1 = 0.9/0.5/0.1 with theory 1 sigma: (m_t, alpha_s) plane"),
    ("claims_run05/fig2a_ellipses.csv", f"{R5}/fig2a_ellipses.csv", "current-measurement 68/95% ellipses in (m_t, alpha_s)"),
    ("claims_run05/fig2b_xi_curves.csv", f"{R5}/fig2b_xi_curves.csv", "xi_inf (mu_b and EW readings) on a fine m_t grid; its xi_c columns are the run02 model and are NOT used"),
    ("claims_run05/fig2b_points.json", f"{R5}/fig2b_points.json", "P1 = 0.9/0.5/0.1 verticals (run05), literature xi values"),
    ("claims_run05/xi_run05_summary.json", f"{R5}/xi_run05_summary.json", "xi_c(mu_b) at the run05 centre (172.20/172.60/172.95/172.96/173.00) and run02 points"),
    ("claims_run05/future_reach.csv", f"{R5}/future_reach.csv", "FCC-ee / HL-LHC sigma_meas and sigma_tot"),
    ("claims_run05/C154_summary.json", f"{R5}/C154_summary.json", "measurement definitions (marginal.*), m0, lambda_c, critical m_t, exclusion values"),
    ("claims_run05/table1_error_budget_run05.csv", f"{R5}/table1_error_budget_run05.csv", "SM-mapping error (pole/m-bar) for the SM point at mu_b"),
    ("claims_run05/table1_rebuild.json", f"{R5}/table1_rebuild.json",
     "full-dimensional width correction converted with the final-model slopes (t14a.central/envelope): prd_gamma_validation"),
    ("claims_run05/C154_inputs_v2_run05.json", "04_calculations/code/20261015_claude_v12_writing/claims_run05/C154_inputs_v2_run05.json",
     "claim-layer configuration: future scenarios (s_m, s_as, kappa), pole <-> m-bar points"),
    ("holonomy_cross_run05/C149_run05_summary.json", f"{CR3}/holonomy_cross/run05/C149_run05_summary.json", "run05 centre: lam_c, crit, P1 grid"),
    ("sm_mapping_run02/R3_mbar_run05.json", f"{CR3}/sm_mapping/run02/R3_mbar_run05.json", "critical m-bar_t(m-bar_t) at the run05 centre (top-axis check)"),
    ("sm_mapping_run01/C148_summary.json", f"{CR3}/sm_mapping/run01/C148_summary.json", "lambda(mu_b) at (172.60, 0.1180, 125.13): check of the generated running"),
    ("realtime_2d_run02/C157_run02_summary.json", f"{CR3}/realtime_2d/run02/C157_run02_summary.json", "xi_c SM/alpha_s variants and bands (run02 centre; direct band check)"),
    ("inflation_stability_run01/xi_inf_table.csv", f"{CR3}/inflation_stability/run01/xi_inf_table.csv", "all-mode inflation bound xi_inf(m_t) for three alpha_s"),
    ("fulldim_dmc_run01/T14a_profile.json", f"{CR3}/fulldim_dmc/run01/T14a_profile.json", "Siegert energies/widths and r_top, r1, r2 at lam = -0.010/-0.013 (2D_ex)"),
    # ---- PRD figures
    ("higgs_beyond_bo_run01/C140_summary.json", f"{CR2}/higgs_beyond_bo/run01/C140_summary.json", "Gamma(lambda) and Q for 1D_G/1D_ex/2D_G/2D_ex, static lambda_c, eta_inf: prd_gamma_validation"),
    ("fulldim_dmc_run01/C151_summary.json", f"{CR3}/fulldim_dmc/run01/C151_summary.json", "full-dimensional DMC Delta Q and the four gamma_full/gamma_2D estimates: prd_gamma_validation"),
    ("universality_run01/C156_eta.json", f"{CR3}/universality/run01/C156_eta.json", "lambda_c vs duration factor f (exact and WKB), hLb, eta_inf: prd_universality, prd_hazard_realtime"),
    ("universality_run02/C156_run02.json", f"{CR3}/universality/run02/C156_run02.json", "d ln|lambda_c|/d ln g for g2, gY, yt (exact and WKB): prd_universality"),
    ("realtime_2d_run01/C157_twod.json", f"{CR3}/realtime_2d/run01/C157_twod.json", "2D real-time (TDSE) vs 1D P_surv(xi_H) at the run02 centre: prd_hazard_realtime"),
    ("claims_run05/xi_run05/C157_xi_quick.json", f"{R5}/xi_run05/C157_xi_quick.json", "run05 1D real-time P_surv(xi_H) for both lambda prescriptions: prd_hazard_realtime"),
    ("claims_run05/cross_decomp_summary.json", f"{R5}/cross_decomp/cross_decomp_summary.json", "run05 cross-term decomposition (C161 step 6): prd_cross_terms"),
    ("holonomy_cross_run01/C149_summary.json", f"{CR3}/holonomy_cross/run01/C149_summary.json", "run01 ratio-method components U(1)_Y, SU(2), SU(3)-top (v11 couplings): prd_cross_terms"),
    ("holonomy_cross_run02/C149_run02_summary.json", f"{CR3}/holonomy_cross/run02/C149_run02_summary.json", "convergence ladder: + cross terms (run02)"),
    ("holonomy_cross_run04/C149_run04_summary.json", f"{CR3}/holonomy_cross/run04/C149_run04_summary.json", "convergence ladder: run04 (vev mixing, top chiral)"),
    ("twod_direct_sm_run01/C150_summary.json", f"{CR3}/twod_direct_sm/run01/C150_summary.json", "convergence ladder rows MU ... 2D direct (R14 couplings)"),
    ("twod_direct_sm_run02/C150_summary.json", f"{CR3}/twod_direct_sm/run02/C150_summary.json", "convergence ladder: + NNLO SM mapping (2D direct)"),
] + [(f"holonomy_cross_run05/X_main_u{u:.4f}.json", f"{CR3}/holonomy_cross/run05/shards/X_main_u{u:.4f}.json",
      "run05 node widths Gamma(m_t, u) at alpha_s = 0.1180: hazard along the turning (prd_hazard_realtime)")
     for u in [0.5 * k for k in range(11)]] + \
    [(f"claims_run05/xi_run05/part2/p2_main_mt{m}_xi{x}.json", f"{R5}/xi_run05/part2/p2_main_mt{m}_xi{x}.json",
      "run05 1D real-time history norm(u) and final P_surv: prd_hazard_realtime")
     for m, x in (("172.20", "0.1667"), ("172.60", "0.1667"), ("172.60", "0.0000"), ("173.00", "0.1667"))]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources() -> dict:
    if SOURCES.exists():
        return json.loads(SOURCES.read_text(encoding="utf-8"))
    return {"_doc": "Provenance of figures/data/higgs (fig-higgs). kind = copy | generated. Paths relative to the project root "
                    "(research tree) or to figures/data.", "files": {}}


def save_sources(src: dict) -> None:
    src["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    SOURCES.write_text(json.dumps(src, indent=1, ensure_ascii=False), encoding="utf-8")


def stage_copy(src: dict) -> None:
    for dest, orig, purpose in COPIES:
        s = ROOT / orig
        d = HD / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        h0, h1 = sha256(s), sha256(d)
        assert h0 == h1, dest
        src["files"][f"higgs/{dest}"] = dict(kind="copy", source=orig, sha256=h1, purpose=purpose)
        print(f"copy {dest}  {h1[:12]}")


def _code_entry(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}


def stage_running(src: dict) -> None:
    """lambda(mu) from mu = m_t to M-bar_P (SM provider of D120/C148, central NNLO variant)."""
    import numpy as np
    code = ROOT / CC3 / "sm_mapping"
    sys.path.insert(0, str(code))
    import sm_provider as SP  # noqa: E402  (read-only import of the production module)
    from scipy.optimize import brentq

    mu = np.logspace(np.log10(172.0), np.log10(SP.MPBAR), 400)
    cases = [(172.2, 0.1180), (172.6, 0.1180), (173.0, 0.1180), (172.6, 0.1171), (172.6, 0.1189)]
    cols, names = [mu], ["mu_GeV"]
    for mt, als in cases:
        c = SP.couplings(mt, als, 125.13, mu, variant="central")
        cols.append(c["lam"])
        names.append(f"lam_mt{mt:.1f}_as{als:.4f}")
    lam_b = {f"{mt:.1f}_{als:.4f}": SP.lam_b(mt, als, 125.13) for mt, als in cases}
    # derivative used to convert the SM-mapping error (GeV of m_t) into lambda(mu_b)
    dl = (SP.lam_b(172.7, 0.1180, 125.13) - SP.lam_b(172.5, 0.1180, 125.13)) / 0.2
    out = HD / "generated" / "lambda_running.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("# lambda(mu) MS-bar, SM provider central (NNLO matching at mu0 = M_t, SMDR25 inputs; M_h = 125.13). "
                f"mu_b = {SP.MU_B:.6e} GeV, M-bar_P = {SP.MPBAR:.4e} GeV\n")
        f.write(",".join(names) + "\n")
        for i in range(len(mu)):
            f.write(",".join(f"{c[i]:.10e}" for c in cols) + "\n")
    # lambda(mu_b) = 0 line in the (m_t, alpha_s) plane
    als_grid = np.round(np.arange(0.1150, 0.12101, 0.0005), 4)
    m0 = [brentq(lambda m: SP.lam_b(m, float(a), 125.13), 165.0, 178.0, xtol=1e-6) for a in als_grid]
    out2 = HD / "generated" / "m0_lambda_mub_zero.csv"
    with open(out2, "w", encoding="utf-8", newline="") as f:
        f.write("# m_t^pole at which lambda(mu_b) = 0 (SM provider central; M_h = 125.13)\nalpha_s,m0_pole\n")
        for a, m in zip(als_grid, m0):
            f.write(f"{a:.4f},{m:.6f}\n")
    ref = json.loads((HD / "sm_mapping_run01/C148_summary.json").read_text(encoding="utf-8"))
    lam_ref = ref["point_172p60_0p1180_125p13"]["lam_mu_b"]
    c154 = json.loads((HD / "claims_run05/C154_summary.json").read_text(encoding="utf-8"))
    m0_ref = c154["relative_form"]["m0"]
    m0_118 = float(np.interp(0.1180, als_grid, m0))
    checks = {"lam_mu_b_172.6_0.1180": [lam_b["172.6_0.1180"], lam_ref, abs(lam_b["172.6_0.1180"] - lam_ref)],
              "m0_at_0.1180": [m0_118, m0_ref, abs(m0_118 - m0_ref)], "dlam_mub_dmt_172.6": dl, "lam_mu_b": lam_b}
    print(json.dumps(checks, indent=1))
    assert checks["lam_mu_b_172.6_0.1180"][2] < 2e-6, "running does not reproduce C148 lambda(mu_b)"
    assert checks["m0_at_0.1180"][2] < 2e-3, "m0 does not reproduce C154 relative_form.m0"
    out3 = HD / "generated" / "constants.json"
    out3.write_text(json.dumps({"MPBAR_GeV": SP.MPBAR, "MU_B_GeV": SP.MU_B, "LbMPbar": SP.MPBAR / SP.MU_B,
                                "dlam_mub_dmt_172p6": dl, "lam_mu_b": lam_b,
                                "_src": "sm_provider.py constants MPBAR, MU_B = MPBAR/722; lam_b(); derivative by +-0.1 GeV"},
                               indent=1), encoding="utf-8")
    for o, purpose in ((out, "SM running lambda(mu), m_t = 172.2/172.6/173.0 (alpha_s 0.1180) and alpha_s = 0.1171/0.1189 (m_t 172.6): prl_fig1b"),
                       (out2, "lambda(mu_b) = 0 line m0(alpha_s): (m_t, alpha_s) plane"),
                       (out3, "M-bar_P, mu_b, dlambda(mu_b)/dm_t and lambda(mu_b) at the plotted points")):
        src["files"][f"higgs/generated/{o.name}"] = dict(
            kind="generated", generator="prepare_data_higgs.py gen_running", sha256=sha256(o), purpose=purpose,
            code=[_code_entry(code / n) for n in ("sm_provider.py", "sm_rge.py", "qcd.py", "_smdr_betas.py")], checks=checks)


def stage_potential(src: dict) -> None:
    """Lowest adiabatic channel of the 2D (r, s) model (D112 2D_ex, v11 couplings, MC+CW matching) and its parts."""
    import numpy as np
    code = ROOT / CC3 / "fulldim_dmc"
    sys.path.insert(0, str(code))
    import T14a_common as T  # noqa: E402  (read-only import; module reads archived inputs only)
    B = T.B
    M = T.model()
    r = np.round(np.arange(0.02, 9.0001, 0.01), 4)
    cols, names = [r], ["r"]
    rows_chk = {}
    prof = json.loads((HD / "fulldim_dmc_run01/T14a_profile.json").read_text(encoding="utf-8"))
    for lam in (-0.010, -0.013):
        U1, om2 = T.potentials("2D_ex", lam, r)
        Uad = U1 + 1.5 * np.sqrt(om2)
        cent = 3.0 / (8 * r * r)
        quart = lam * r ** 4 / 4
        dV = B.dVm("MC", r, lam)
        # field-space potential of the 4D radial problem (U_g of T14a_common.radial_guide): U_ad - 3/(8 r^2)
        Ug = Uad - cent
        zp = Ug - quart - dV          # zero-point energy of the gauge zero modes: C0 + B_x + dE_DMC + (3/2) Om_eff
        cols += [Uad, Ug, quart, dV, zp]
        tag = f"m{abs(lam):.3f}".replace(".", "p")
        names += [f"Uad_{tag}", f"Ug_{tag}", f"quartic_{tag}", f"dV_{tag}", f"zeropoint_{tag}"]
        # check against the archived profile (U_ad at r = 1, 3, 4.5, 6; r_top)
        p = prof[f"{lam}"]["2D_ex"]
        dev = max(abs(float(np.interp(float(x), r, Uad)) - p["U_at"][x]) for x in ("1.0", "3.0", "4.5", "6.0"))
        rows_chk[f"{lam}"] = {"max_abs_dev_U_at": dev, "r_top_here": float(r[np.argmax(np.where(r > 2, Uad, -1e9))]),
                              "r_top_archived": p["r_top"]}
        assert dev < 2e-3, (lam, dev)
    out = HD / "generated" / "potential_2Dex_lam010_013.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("# D112 2D_ex reduced model (v11 couplings g2 0.5268, gY 0.4415, yt 0.4585; MC+CW one-loop), units 1/L. "
                "Uad = U1 + (3/2) Om_eff (incl. 3/(8 r^2)); Ug = Uad - 3/(8 r^2); quartic = lam r^4/4; dV = one-loop MC "
                "finite-volume correction; zeropoint = Ug - quartic - dV\n")
        f.write(",".join(names) + "\n")
        for i in range(len(r)):
            f.write(",".join(f"{c[i]:.8e}" for c in cols) + "\n")
    print(json.dumps(rows_chk, indent=1))
    src["files"][f"higgs/generated/{out.name}"] = dict(
        kind="generated", generator="prepare_data_higgs.py gen_potential", sha256=sha256(out),
        purpose="effective potential along r (lam = -0.010/-0.013) and its parts: prl_fig1a / prd_threshold",
        code=[_code_entry(code / n) for n in ("T14a_common.py", "C140_beyond_bo.py", "C138_core.py")], checks=rows_chk)


def stage_wavefunction(src: dict) -> None:
    """Lowest 2D (r, s) eigenfunctions of the D112 2D_ex operator with Dirichlet walls (T14a same-wall FD solver):
    lambda = 0 (bound ground state, wall far out) and lambda = -0.013 (quasi-bound state, wall just outside the barrier,
    T14a main wall), and the field-space potential V(r, s) = U1 - 3/(8 r^2) + Om_eff^2 s^2 / 2 on a display grid."""
    import numpy as np
    code = ROOT / CC3 / "fulldim_dmc"
    sys.path.insert(0, str(code))
    import T14a_common as T  # noqa: E402
    c151 = json.loads((HD / "fulldim_dmc_run01/C151_summary.json").read_text(encoding="utf-8"))
    c140 = json.loads((HD / "higgs_beyond_bo_run01/C140_summary.json").read_text(encoding="utf-8"))
    h, Sw = 0.04, 8.0
    out = {}
    cases = [(0.0, 9.0), (-0.013, [w["rw"] for w in c151["lam_neg"]["-0.013"]["walls"] if w.get("main")][0])]
    rd = np.round(np.arange(0.04, 8.0001, 0.04), 4)
    sd = np.round(np.arange(0.0, 4.0001, 0.04), 4)
    for lam, rw in cases:
        rw = round(rw / h) * h
        vals, vecs, r, s = T.fd2d_wall("2D_ex", lam, rw, h, Sw=Sw, k=3)
        u = vecs[:, 0].reshape(len(r), len(s))
        dens = u ** 2 / np.max(u ** 2)          # probability density in (r, s) incl. the measure r^3 s^2
        U1, om2 = T.potentials("2D_ex", lam, rd)
        V = (U1 - 3.0 / (8 * rd * rd))[:, None] + 0.5 * om2[:, None] * sd[None, :] ** 2
        Uad = U1 + 1.5 * np.sqrt(om2)
        tag = "lam0" if lam == 0.0 else "lamm0p013"
        np.savez_compressed(HD / "generated" / f"wavefunction_2Dex_{tag}.npz", r=r, s=s, dens=dens.astype(np.float32),
                            E=vals, rd=rd, sd=sd, V=V, Uad_rd=Uad, om2_rd=om2, lam=lam, rw=rw, h=h, Sw=Sw)
        ref = c140["lam0"]["Q_2D_ex"] if lam == 0.0 else c140["Q"]["2D_ex"]["-0.013"]
        out[tag] = {"E0_wall": float(vals[0]), "E1_wall": float(vals[1]), "Q_ref_C140": ref, "rw": rw, "h": h,
                    "E0_minus_Qref": float(vals[0] - ref)}
        print(tag, out[tag])
    for tag in out:
        o = HD / "generated" / f"wavefunction_2Dex_{tag}.npz"
        src["files"][f"higgs/generated/{o.name}"] = dict(
            kind="generated", generator="prepare_data_higgs.py gen_wavefunction", sha256=sha256(o),
            purpose="2D (r, s) eigenfunction density and potential surface: prd_higgs_wavefunction",
            code=[_code_entry(code / n) for n in ("T14a_common.py", "C140_beyond_bo.py", "C138_core.py")],
            checks=out[tag])


if __name__ == "__main__":
    stages = sys.argv[1:] or ["copy"]
    if "all" in stages:
        stages = ["copy", "gen_running", "gen_potential"]
    src = load_sources()
    for st in stages:
        {"copy": stage_copy, "gen_running": stage_running, "gen_potential": stage_potential,
         "gen_wavefunction": stage_wavefunction}[st](src)
        save_sources(src)
    print("done:", stages)
