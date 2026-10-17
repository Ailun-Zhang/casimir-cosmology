# PROVENANCE: 04_calculations/code/20261011_claude_followup/C81_higgs_zero_mode_mass.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Induced mass of the Higgs zero mode from finite-volume tadpoles (D59 Sec. 2); completes C78: r_first_order = 0.2826 (Landau, 3 polarisations). Input: inputs/rge_couplings.json (literature work file L25). Paper: PRD Sec. IV C, App. app:zeromodes. Output <work>/fu/C81/C81_results.json.

C81: first-order estimate of the induced mass of the Higgs zero mode (R10 review item DA-9; completes C78).

At the same order as the holonomy curvature of C78, the nonzero modes give the constant Higgs mode a mass. For a field
with field-dependent mass^2 M^2(phi) the one-loop curvature at phi = 0 is (1/2) n (d^2 M^2/d phi^2) T for bosons and
-(1/2) n (d^2 m^2/d phi^2) T for fermions (n real degrees of freedom), with the finite-volume tadpoles

    T_B = (1/L^3) sum'_k 1/(2|k|) = Z_1(1;0)/(4 pi^2 L^2),      T_F = Z_1(1;AAA)/(4 pi^2 L^2)

(zeta-regularized; the flat-space part is the m = 0 image term; Z(1/2) = Z(1)/pi by the functional equation; the k = 0
modes are excluded because the zero modes are treated explicitly). Mapping the standard thermal-mass coefficients of the
SM Higgs (V = lambda (H^dag H)^2), m^2(T) = (T^2/12)(9g2^2/4 + 3gY^2/4 + 3yt^2 + 6 lambda), onto these tadpoles
(bosons T^2/12 -> T_B, top T^2/12 per 3 yt^2 -> -6 yt^2 T_F) gives

    m^2 L^2 = (9g2^2/4 + 3gY^2/4 + 6 lambda) Z(1;0)/(4 pi^2) - 6 yt^2 Z(1;AAA)/(4 pi^2).

The gauge part counts three polarizations per massive vector (Landau gauge); a four-polarization count is reported as a
bracket. First-order shift: L dE = m^2 L^2 <Phi^dag Phi>, with <Phi^dag Phi> = 1/z from the optimal Gaussian of the coupled
SU(2) x U(1) x Higgs model at one-loop couplings (C78: z = 0.77029).

    python -B C81_higgs_zero_mode_mass.py
Output: results/20261011_claude_followup/C81/C81_results.json
"""
import json
import math
from pathlib import Path

import sys  # package: work-tree path shim of the casimir chain (see casimir/_casimir_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _casimir_paths as WP  # noqa: E402
OUT = WP.wdir("fu/C81")
C78 = WP.wfile("fu/C78/C78_results.json")
RGE = WP.step_dir("02_zero_mode_energy", "inputs", "rge_couplings.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c78 = json.loads(C78.read_text())
    Z1, ZA = c78["zeta"]["Z1"], c78["zeta"]["Z_AAA1"]
    g2, gY, lam, C = c78["couplings"]["g2"], c78["couplings"]["gY"], c78["couplings"]["lam"], c78["C_free"]
    z = c78["gaussian_EW"]["z"]
    rge = json.loads(RGE.read_text())
    yt_3e15 = rge["at_scales"]["3e15"]["two_loop_from_Mt"]["yt"]
    TB, TF = Z1 / (4 * math.pi ** 2), ZA / (4 * math.pi ** 2)
    out = dict(T_B=TB, T_F=TF, z=z, PhiPhi=1 / z, yt=yt_3e15, couplings=dict(g2=g2, gY=gY, lam=lam), runs={})
    for label, gauge_pol in (("landau_3pol", 1.0), ("four_pol_bracket", 4.0 / 3.0)):
        gauge = gauge_pol * (9 * g2 ** 2 / 4 + 3 * gY ** 2 / 4) * TB
        quart = 6 * lam * TB
        top = -6 * yt_3e15 ** 2 * TF
        m2 = gauge + quart + top
        dE = m2 / z
        out["runs"][label] = dict(m2L2_gauge=gauge, m2L2_quartic=quart, m2L2_top=top, m2L2=m2, dQ=dE, dr=dE / C)
    hol = c78["shifts"]["gauss_coupled_gauss"]["dr"]
    out["holonomy_dr_C78"] = hol
    out["total_first_order_dr"] = {k: hol + v["dr"] for k, v in out["runs"].items()}
    out["r_LO"] = c78["leading_order_r"]
    out["r_first_order"] = {k: c78["leading_order_r"] + v for k, v in out["total_first_order_dr"].items()}
    (OUT / "C81_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
