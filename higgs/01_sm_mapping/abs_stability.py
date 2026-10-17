# PROVENANCE: new script written 2026-10-16 by rep-B (v12 companion report, Sec. 10); it only calls the production SM
# mapping (higgs/zmlib/sm_provider.py = 04_calculations/code/20261014_claude_v12_r3/sm_mapping/sm_provider.py plus the
# package path bootstrap) and changes no other file of the package.
"""Absolute-stability line of the Standard Model mapping used for the Higgs zero-mode criterion.

For the inputs of the papers, alpha_s(M_Z) = 0.1180 and M_h = 125.13 GeV, with the production mapping
(sm_provider variant 'central': SMDR-1.3-type NNLO matching at mu0 = M_t, SMDR-port RGE), we compute
  m_0     : the top pole mass at which lambda(mu_b) = 0, mu_b = Mbar_P/722 (regression: 171.0921 GeV, R5-029);
  m_stab  : the top pole mass at which min_{mu <= Mbar_P} lambda(mu) = 0 (absolute stability up to the reduced
            Planck mass); the minimum over mu is located on a logarithmic grid and refined with a bounded
            one-dimensional minimisation in log10(mu);
and, for orientation, the same quantities at alpha_s = 0.1180 -+ 0.0009, the cut mu <= M_Pl (non-reduced) instead of
mu <= Mbar_P, the MS-bar masses mbar_t(mbar_t) of m_0 and m_stab (four-loop QCD, as for the criterion), and the
bound with this mapping at the inputs of Bednyakov et al. (M_h = 125.7 GeV, alpha_s = 0.1185).

Units: GeV; MS-bar couplings.  Report: Ref. [v12Report], Sec. 10 (the companion report is the only document that
quotes m_stab; the PRL and PRD state only that the absolute-stability bound lies just below m_0).

Usage (from anywhere; Python 3.12, numpy, scipy):
  python -B higgs/01_sm_mapping/abs_stability.py                    # packaged mapping -> abs_stability.json here
  python -B higgs/01_sm_mapping/abs_stability.py --crosscheck ROOT  # also rerun with the research copy under ROOT
  python -B higgs/01_sm_mapping/abs_stability.py --source research --research-root ROOT --json-stdout
Set DNEC_HIGGS_WORK to choose the work tree that the packaged path shim creates on import (default output/higgs/manual).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))   # [code-finalize 2026-10-16] paths in the output are relative


def _rel_path(path, base):
    """Path relative to the package root (package copy) or to the research root (research copy)."""
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:
        return os.path.basename(path)
ZMLIB = os.path.join(os.path.dirname(HERE), "zmlib")
RESEARCH_REL = os.path.join("04_calculations", "code", "20261014_claude_v12_r3", "sm_mapping")

ALS0, MH0 = 0.1180, 125.13
DALS = 0.0009


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_modules(source, research_root):
    if source == "package":
        sys.path.insert(1, ZMLIB)
        src_dir = ZMLIB
    else:
        src_dir = os.path.join(research_root, RESEARCH_REL)
        if not os.path.isdir(src_dir):
            raise SystemExit(f"research copy not found: {src_dir}")
        sys.path.insert(0, src_dir)
    import sm_provider as SP  # noqa: E402
    import qcd  # noqa: E402
    return SP, qcd, os.path.join(src_dir, "sm_provider.py")


def compute(SP, qcd):
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar

    mpbar, mpl, mub = SP.MPBAR, SP.MPL, SP.MU_B

    def lam_min(mt, als=ALS0, mh=MH0, mu_hi=None):
        """min of lambda(mu) over 1e8 GeV <= mu <= mu_hi (the minimum lies far above 1e8 GeV for these inputs)."""
        mu_hi = mpbar if mu_hi is None else mu_hi
        lg = np.linspace(8.0, np.log10(mu_hi), 241)
        lam = np.asarray(SP.couplings(mt, als, mh, 10.0 ** lg)["lam"], dtype=float)
        i = int(np.argmin(lam))
        if i == len(lg) - 1:  # minimum at the upper end of the range
            return float(lam[i]), float(10.0 ** lg[i]), True
        lo, hi = lg[max(i - 2, 0)], lg[min(i + 2, len(lg) - 1)]
        f = lambda x: float(SP.coup_point(mt, als, mh, 10.0 ** x)["lam"])  # noqa: E731
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6))
        return float(r.fun), float(10.0 ** r.x), False

    def m_zero(als=ALS0, mh=MH0):
        return brentq(lambda m: SP.lam_b(m, als, mh), 169.0, 174.0, xtol=1e-6)

    def m_stab(als=ALS0, mh=MH0, mu_hi=None):
        return brentq(lambda m: lam_min(m, als, mh, mu_hi)[0], 169.0, 174.0, xtol=1e-6)

    def mbar(mt, als=ALS0):
        return float(qcd.mbar_mbar(mt, qcd.as6_at(mt, als), N=4))

    out = {}
    lam_b_central = float(SP.lam_b(172.60, ALS0, MH0))
    m0 = m_zero()
    ms = m_stab()
    lmin, mu_min, at_edge = lam_min(ms)
    lp, _, _ = lam_min(ms + 0.01)
    lm, _, _ = lam_min(ms - 0.01)
    out["central"] = dict(
        alpha_s=ALS0, M_h=MH0, mu_b_GeV=mub, MPbar_GeV=mpbar, mu_upper_GeV=mpbar,
        lambda_mu_b_at_172p60=lam_b_central,
        m0_pole=m0, m0_mbar=mbar(m0),
        m_stab_pole=ms, m_stab_mbar=mbar(ms),
        m0_minus_m_stab=m0 - ms,
        mu_of_lambda_min_at_m_stab_GeV=mu_min, lambda_min_at_m_stab=lmin, minimum_at_upper_edge=at_edge,
        dlambda_min_dmt_per_GeV=(lp - lm) / 0.02,
        lambda_mu_b_at_m_stab=float(SP.lam_b(ms, ALS0, MH0)),
    )
    out["mu_upper_M_Pl_nonreduced"] = dict(mu_upper_GeV=mpl, m_stab_pole=m_stab(mu_hi=mpl))
    out["alpha_s_variants"] = {}
    for als in (ALS0 - DALS, ALS0 + DALS):
        a0, a1 = m_zero(als), m_stab(als)
        out["alpha_s_variants"][f"{als:.4f}"] = dict(m0_pole=a0, m_stab_pole=a1, m0_minus_m_stab=a0 - a1)
    out["BKPV_inputs_this_mapping"] = dict(M_h=125.7, alpha_s=0.1185, m_stab_pole=m_stab(0.1185, 125.7),
                                           note="our SMDR25 matching at the inputs of Bednyakov et al. 2015 "
                                                "(their own matching gives 171.44; reproduced as 171.453 in D120)")
    out["regression"] = dict(m0_expected_R5_029=171.0921, m0_diff=m0 - 171.0921,
                             lambda_mu_b_expected_H014=-0.0097741, lambda_mu_b_diff=lam_b_central + 0.0097741)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("package", "research"), default="package")
    ap.add_argument("--research-root", default=None, help="project root that contains " + RESEARCH_REL)
    ap.add_argument("--crosscheck", default=None, metavar="ROOT",
                    help="also run the research copy under ROOT (read-only import) and record the comparison")
    ap.add_argument("--out", default=os.path.join(HERE, "abs_stability.json"))
    ap.add_argument("--json-stdout", action="store_true", help="print the result JSON only (no file written)")
    args = ap.parse_args()

    t0 = time.time()
    SP, qcd, sp_path = load_modules(args.source, args.research_root)
    res = compute(SP, qcd)
    res["meta"] = dict(script="higgs/01_sm_mapping/abs_stability.py", source=args.source,
                       sm_provider=_rel_path(sp_path, args.research_root if args.source == "research" else PKG_ROOT), sm_provider_sha256=sha256(sp_path), variant="central",
                       definition="m_stab: min over 1e8 GeV <= mu <= Mbar_P of lambda(mu) = 0; "
                                  "m0: lambda(mu_b) = 0, mu_b = Mbar_P/722",
                       python=platform.python_version(), seconds=round(time.time() - t0, 1),
                       written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if args.json_stdout:
        print(json.dumps(res))
        return
    if args.crosscheck:
        cmd = [sys.executable, "-B", os.path.abspath(__file__), "--source", "research",
               "--research-root", args.crosscheck, "--json-stdout"]
        txt = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout
        rc = json.loads(txt.strip().splitlines()[-1])
        res["crosscheck_research_copy"] = dict(
            sm_provider=rc["meta"]["sm_provider"], sm_provider_sha256=rc["meta"]["sm_provider_sha256"],
            m0_pole=rc["central"]["m0_pole"], m_stab_pole=rc["central"]["m_stab_pole"],
            max_abs_diff_GeV=max(abs(rc["central"]["m0_pole"] - res["central"]["m0_pole"]),
                                 abs(rc["central"]["m_stab_pole"] - res["central"]["m_stab_pole"])))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    c = res["central"]
    print(f"m0 = {c['m0_pole']:.4f} GeV (mbar {c['m0_mbar']:.3f}); m_stab = {c['m_stab_pole']:.4f} GeV "
          f"(mbar {c['m_stab_mbar']:.3f}); m0 - m_stab = {c['m0_minus_m_stab']:.4f} GeV; "
          f"lambda_min at mu = {c['mu_of_lambda_min_at_m_stab_GeV']:.3e} GeV")
    print("written", args.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
