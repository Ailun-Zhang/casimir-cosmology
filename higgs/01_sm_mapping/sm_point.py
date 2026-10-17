# PROVENANCE: new verification script written 2026-10-15 by code-higgs (v12 package); it repeats the production calls
# of C148_post.py (point_172p60_0p1180_125p13) and C149_direct.post (lam_c_hazard_equiv) with the packaged sm_provider.
"""SM mapping at the bounce scale: lam, y_t, g2, gY, g3 at mu_b = Mbar_P/722.0 for (M_t, alpha_s, M_h) =
(172.60, 0.1180, 125.13), the r3 central SM mapping (SMDR25 NNLO matching at mu0 = M_t + 5-loop-QCD RGE), and
lam(mu_b) at the run05 critical pole masses (= the hazard-equivalent thresholds lam_c of the PRL).

Paper: PRL "lam(mu_b) = -0.0098" and lam_c = -0.0091/-0.0121/-0.0152; PRD App. H5.  Units: GeV, MS-bar couplings.
Output: <work>/r3/sm_mapping/package/sm_point.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "zmlib"))
import _zpaths as _zp  # noqa: E402
import sm_provider as SP  # noqa: E402


def main():
    c0 = SP.coup_point(172.6, 0.1180, 125.13, SP.MU_B)
    out = dict(mu_b_GeV=SP.MU_B, MPbar_GeV=SP.MPBAR, LbMPbar=SP.MPBAR / SP.MU_B,
               point_172p60_0p1180_125p13=dict(lam_mu_b=c0["lam"], yt_mu_b=c0["yt"], g2_mu_b=c0["g2"],
                                               gY_mu_b=c0["gY"], g3_mu_b=c0["g3"]))
    summ = os.path.join(_zp.WORK, "r3", "holonomy_cross", "run05", "C149_run05_summary.json")
    if os.path.exists(summ):
        crit = json.load(open(summ, encoding="utf-8"))["crit"]["0.1180"]
        out["lam_mu_b_at_run05_critical_mt"] = {k: SP.coup_point(v, 0.1180, 125.13, SP.MU_B)["lam"] for k, v in crit.items()}
        out["run05_critical_mt_0p1180"] = crit
    # numerical slope dlam(mu_b)/dM_t at 172.6 (central difference, +-0.05 GeV)
    lp = SP.coup_point(172.65, 0.1180, 125.13, SP.MU_B)["lam"]
    lm = SP.coup_point(172.55, 0.1180, 125.13, SP.MU_B)["lam"]
    out["dlam_mu_b_dMt_172p6"] = (lp - lm) / 0.1
    p = _zp.w("r3/sm_mapping/package/sm_point.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
