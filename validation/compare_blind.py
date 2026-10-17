# PROVENANCE: new script written 2026-10-16 by rev-code (R18 revision, R1/MH-02) for the v12 code/data package.
"""Comparison of the archived blind reimplementations (validation/*/results*) with the main calculation.

For every item it (1) finds each quoted blind number among the numbers of the archived blind result files of that
check (to the quoted digits, sign included) and records where it is, (2) finds each main-calculation number in the
named archived package file (or takes it from the report table named, when the main value is not a package output),
and (3) checks |blind - main| <= tolerance (the pass criterion of the specification / of report Sec. 19).

    python validation/compare_blind.py                 # table to stdout; output/validation/comparison.json; exit 1 on failure
    python validation/compare_blind.py --write-md      # also rewrite validation/COMPARISON.md
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from dnec_common import paths as P  # noqa: E402

H01 = "higgs/01_sm_mapping/expected/r3/sm_mapping/run02/C148r2_summary.json"
H01B = "higgs/01_sm_mapping/expected/r3/sm_mapping/run01/C148_bench.json"
H02S = "higgs/02_finite_volume_matching/expected/r2/higgs_matching/run01/C138_static.json"
H03L = "higgs/03_zero_mode_2d/expected/r2/higgs_beyond_bo/run01/C140_lam0.json"
H03S = "higgs/03_zero_mode_2d/expected/r2/higgs_beyond_bo/run01/C140_summary.json"
H03 = "higgs/03_zero_mode_2d/expected/r3/twod_direct_sm/run01/C150_summary.json"
H04R2 = "higgs/04_cross_terms/expected/r3/holonomy_cross/run02/C149_run02_summary.json"
H09R2 = "higgs/09_claims/expected/r3/claim_positioning/run02/C154_correction_summary.json"
H09 = "higgs/09_claims/expected/w/claims_run05/run01/C154_summary.json"
H11 = "higgs/11_inflation_stability/expected/r3/inflation_stability/run01/C155_summary.json"
S01 = "shape/01_emptiness_gate/expected/r3/shape_hardening/run01/emptiness/C153_emptiness.json"
S02 = "shape/02_realtime_theorem/expected/r1/state_selection/run03/C137c_results.json"
S02S = "shape/02_realtime_theorem/expected/r3/shape_hardening/run01/C153_summary.json"
S02R2 = "shape/02_realtime_theorem/expected/r3/shape_hardening/run02/C153_run02_summary.json"
DER = "numbers/derived_numbers.json"
C78 = "casimir/02_zero_mode_energy/expected/fu/C78/C78_results.json"
KAPPAS = "kappas"   # computed: PRD Eq. (h-kappas) from the Epstein constants Z_P, Z_A of C78
# (check, quantity, blind numbers, main numbers, main source(s), tolerance, kind: abs | rel | mag (compare magnitudes);
#  an item whose source is KAPPAS is the anchor the V2 specification supplied, reported as a regression, not counted
#  among the comparisons with the main calculation)
ITEMS = [
    ("keystone", "lambda(mu_b), earlier SM mapping", ["-0.0101929"], ["-0.010193"], H01B, 2e-6, "abs"),
    ("keystone", "y_t(mu_b)", ["0.428359"], ["0.42836"], H01B, 2e-5, "abs"),
    ("keystone", "L eps0(lambda=0), 2D_G / 1D", ["4.50769", "4.59855"], ["4.507701", "4.598545"], H03L, 2e-5, "abs"),
    ("keystone", "L eps0(lambda=0), with DMC shape", ["4.45078"], ["4.45086"], H03L, 1e-4, "abs"),
    ("keystone", "L Gamma(lambda=-0.010), 1D (v11 couplings)", ["0.00098711"], ["0.00098711"], H02S, 1e-3, "rel"),
    ("keystone", "L Gamma(lambda=-0.010), 2D_G", ["0.00053183"], ["0.0005318"], H03S, 1e-3, "rel"),
    ("keystone", "static lambda_c (P=0.9/0.5), 2D_G", ["-0.009424", "-0.012613"], ["-0.009424", "-0.012613"], H03S, 1e-6, "abs"),
    ("keystone", "P1(172.2/172.6/173.0), 2D_G hazard", ["0.9792", "0.8210", "0.4371"], ["0.97920", "0.82098", "0.43738"], H03, 5e-4, "abs"),
    ("keystone", "m_c 2D_G (GeV, P1=0.9/0.5)", ["172.467", "172.941"], ["172.4671", "172.9412"], H03, 5e-3, "abs"),
    ("keystone", "m_c 2D with own DMC shape (GeV, P1=0.9/0.5/0.1)", ["172.521", "173.017", "173.520"],
     ["172.5185", "173.0140", "173.5155"], H03, 5e-3, "abs"),
    ("V2 cross terms", "holonomy curvatures at r=0 (SU(3), SU(2), Y): anchor supplied by the specification",
     ["-0.39261", "-0.24209", "0.70036"], None, KAPPAS, 1e-4, "abs"),
    ("V2 cross terms", "Delta m_c from the cross terms (GeV, P1=0.9/0.5/0.1)", ["-0.081", "-0.117", "-0.157"],
     ["-0.0848", "-0.1238", "-0.1667"], H04R2, 1.5e-2, "abs"),
    ("V1 SM mapping", "lambda(mu_b), SMDR matching + own running", ["-0.009778"], ["-0.0097741"], H09, 1e-5, "abs"),
    ("V1 SM mapping", "lambda(mu_b), mr matching (scheme difference)", ["-0.010055"], ["-0.0097741"], H09, 7e-4, "abs"),
    ("V1 SM mapping", "matching-scale variation, SMDR (GeV)", ["0.033", "-0.019"], ["0.0331", "-0.0186"], H01, 1e-3, "abs"),
    ("V3 claims", "Pbar_1, direct mass (third-stage model)", ["0.725"], ["0.7271"], H09R2, 5e-3, "abs"),
    ("V3 claims", "68% range of P1, direct mass", ["0.380", "0.989"], ["0.384", "0.989"], H09R2, 5e-3, "abs"),
    ("V4 shape", "sigma_floor", ["0.10406"], ["0.1037"], S02S, 5e-4, "abs"),
    ("V4 shape", "G(|t_x|) at h|t_x| = 0.1/0.5/1", ["2.3955", "45.1258", "314.090"], ["2.3952", "45.097", "313.79"], S02, 3e-3, "rel"),
    ("V4 shape", "t_*(90%), t_*(99%) (h|t_x|)", ["-0.360", "-0.243"], ["0.355", "0.237"], [DER, S02R2], 8e-3, "mag"),
    ("V4 shape", "emptiness budget (quanta) at shares 0.2999/0.28/0.258", ["4.7616", "4.8976", "5.0473"],
     ["4.762", "4.898", "5.047"], S01, 1e-3, "abs"),
    ("V5 universality/inflation", "xi_inf(EW) after the reproduction offset, m_t = 172.2/172.6/173.0",
     ["0.0487", "0.0547", "0.0599"], ["0.04877", "0.05512", "0.06044"], H11, 2e-3, "abs"),
    ("V5 universality/inflation", "xi_inf(mu_b), m_t = 172.2/172.6/173.0", ["0.03224", "0.03940", "0.04550"],
     ["0.03030", "0.03781", "0.04413"], H11, 2e-3, "abs"),
]
DIRS = {"keystone": "keystone", "V2 cross terms": "keystone", "V1 SM mapping": "sm_mapping", "V3 claims": "claims",
        "V4 shape": "shape", "V5 universality/inflation": "universality_inflation"}


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (i,))
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def find(s, files):
    """Pointers (file#/path) of the numbers that equal s to its quoted digits (sign included)."""
    x = float(s)
    dec = len(s.split(".")[1]) if "." in s else 0
    tol = 0.5 * 10.0 ** (-dec) + 1e-15
    hits = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
        except (ValueError, UnicodeDecodeError):
            continue
        for p, v in leaves(d):
            if abs(v - x) <= tol:
                hits.append("%s#/%s" % (os.path.relpath(f, PKG).replace(os.sep, "/"), "/".join(map(str, p))))
    return hits


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows, ok_all = [], True
    for check, qty, blind, main, src, tol, kind in ITEMS:
        bfiles = sorted(glob.glob(os.path.join(HERE, DIRS[check], "results*", "**", "*.json"), recursive=True))
        bhits = [find(b, bfiles) for b in blind]
        anchor = src == KAPPAS
        if anchor:
            z = json.load(open(os.path.join(PKG, *C78.split("/")), encoding="utf-8"))["zeta"]
            zp, za = z["Z1"], z["Z_AAA1"]
            k = [(zp - 2 * za) / math.pi ** 2, (5 * zp - 12 * za) / (6 * math.pi ** 2), (zp - 20 * za) / (6 * math.pi ** 2)]
            main = ["%.5f" % v for v in k]
            mhits = [["PRD Eq. (h-kappas) with %s#/zeta/Z1 = %.6f, #/zeta/Z_AAA1 = %.6f" % (C78, zp, za)]] * 3
            label = "PRD Eq. (h-kappas) from C78_results.json (Z_P, Z_A)"
        else:
            files = [os.path.join(PKG, *s.split("/")) for s in (src if isinstance(src, list) else [src])]
            mhits = [find(m, files) for m in main]
            label = " + ".join(s.split("/")[-1] for s in (src if isinstance(src, list) else [src]))
        f = (lambda v: abs(float(v))) if kind == "mag" else (lambda v: float(v))
        diffs = [abs(f(b) - f(m)) / (abs(f(m)) if kind == "rel" else 1.0) for b, m in zip(blind, main)]
        traced = all(bhits) and all(mhits)
        passed = traced and max(diffs) <= tol
        ok_all &= passed
        rows.append(dict(check=check, quantity=qty, blind=blind, main=main, main_source=label, anchor=anchor, tolerance=tol,
                         tolerance_kind=kind, max_diff=max(diffs), blind_found_at=[h[:2] for h in bhits],
                         main_found_at=[h[:2] for h in mhits], traced=traced, passed=passed))
        print("%-4s %-26s %-55s blind %s | main %s | max diff %.3g (%s tol %.3g)%s%s" % (
            "PASS" if passed else "FAIL", check, qty[:55], "/".join(blind), "/".join(main), max(diffs), kind, tol,
            "" if traced else "  NOT TRACED", "  [anchor regression]" if anchor else ""))
    out = dict(task="comparison of the blind reimplementations with the main calculation (validation/)", n_items=len(rows),
               n_pass=sum(r["passed"] for r in rows), n_main_comparisons=sum(not r["anchor"] for r in rows),
               n_anchor_regressions=sum(r["anchor"] for r in rows), all_pass=ok_all, items=rows)
    p = P.out("validation", "comparison.json", mkdir=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("%d/%d items traced and within tolerance (%d comparisons with the main calculation, %d anchor regression) -> %s"
          % (out["n_pass"], len(rows), out["n_main_comparisons"], out["n_anchor_regressions"], p))
    if "--write-md" in sys.argv:
        L = ["# Blind reimplementations vs main calculation (generated by `compare_blind.py --write-md`)", "",
             "Blind numbers are found in the archived blind results of this folder; main numbers in the archived package "
             "file named (the r = 0 curvatures are computed from the archived Epstein constants with PRD Eq. h-kappas; "
             "they are the anchor that the V2 specification supplied, so that item is a regression, not one of the 22 "
             "comparisons with the main calculation). Tolerance = pass criterion (abs; rel = relative; mag = magnitudes).", "",
             "| check | quantity | blind | main | main source | tolerance | max diff | result |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append("| %s | %s | %s | %s | `%s` | %g (%s) | %.3g | %s |" % (
                r["check"], r["quantity"], " / ".join(r["blind"]), " / ".join(r["main"]),
                r["main_source"] + (" (anchor regression)" if r["anchor"] else ""),
                r["tolerance"], r["tolerance_kind"], r["max_diff"], "pass" if r["passed"] else "FAIL"))
        L += ["", "Where each blind number sits: `output/validation/comparison.json` → `items[*].blind_found_at`.", ""]
        with open(os.path.join(HERE, "COMPARISON.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(L))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
