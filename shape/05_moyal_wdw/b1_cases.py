# PROVENANCE: new driver written 2026-10-15 by code-shape-cmb (v12 code/data package); mirrors the case() closure of
# C160_b1_wdw.main() (research tree 04_calculations/code/20261014_claude_v12_r3/shape_tier2/C160_b1_wdw.py) line by line.
"""Selected cases of the B1 Wheeler-DeWitt test (D131 Sec. 4) one at a time, keyed '<case>@<X>'.

The production script runs all cases in one main() (the z_s = 2.5 set was stopped during case B4, which D131 Sec. 4.2(d)
declares contaminated; the z_s = 0 set is complete).  This driver calls the production functions of C160_b1_wdw
(Grid, wdw, metrics, evolve, gauss, hermite1, linear_var_b) with exactly the arguments of main() for the valid cases,
so that single cases (e.g. the physical X = 2282.49 Ai-type case, D_re = 2.4e-6) can be recomputed and compared with
the archived run entries (expected/.../b1_cases_rekeyed.json = archived C160_b1_wdw.json re-keyed, no new numbers).

    python b1_cases.py A_Ai_real@100 A_Ai_real@2282.49 --out r3/shape_tier2/run01/b1/b1_cases.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C160_b1_wdw as B  # noqa: E402  (production module; also puts the chain root on sys.path)
import _shape_paths as WP  # noqa: E402


def table():
    var0 = B.MU * 2 ** 1.5 / (6 * np.sqrt(B.KAP))
    nat = lambda g: B.gauss(g, var0).astype(complex)                       # noqa: E731
    zero = lambda g: np.zeros(len(g.b), complex)                           # noqa: E731
    h1 = lambda g: B.hermite1(g, var0)                                     # noqa: E731
    return nat, {
        "A_Ai_real": (nat, zero, nat, {}, 0.0),
        "A_sep_control": (nat, zero, nat, {"sep": True}, 0.0),
        "B1_Ai_chi0_plus_Bi_h1_eps1": (nat, h1, None, {"sep": True}, 1.0),
        "A_Ai_real_zs5": (nat, zero, nat, {"zs": 5.0}, 0.0),
        "A_Ai_real_nb512_bmax16": (nat, zero, nat, {"nb": 512, "bmax": 16.0}, 0.0),
        "A_Ai_real_nb512_bmax12": (nat, zero, nat, {"nb": 512, "bmax": 12.0}, 0.0),
        "A_Ai_real_quadratic": (nat, zero, nat, {"cubic": False}, 0.0),
        "B2_eps0.3": (nat, lambda g: 0.3 * B.hermite1(g, var0), None, {"sep": True}, 2 * 0.09 / 1.09),
        "B3_Ai_plus_0.7Bi_same_chi0": (nat, lambda g: 0.7 * B.gauss(g, var0).astype(complex), nat, {"sep": True}, 0.0),
    }


def run_case(X, name):
    nat, T = table()
    A, Bf, chi_in, kw, predicted = T[name]
    zs, cubic, nb, bmax, sep = kw.get("zs", 2.5), kw.get("cubic", True), kw.get("nb", 384), kw.get("bmax", 12.0), kw.get("sep", False)
    t_list = (0.3, 0.45, 0.6)
    g = B.Grid(X, bmax=bmax, nb=nb, cubic=cubic)
    fa, fb = A(g), Bf(g)
    xl = [0.5 * np.log(np.cosh(2 * t)) for t in t_list]
    sols, nfev, xs = B.wdw(g, fa, fb, zs, xl, sep=sep)
    cin = chi_in(g) if chi_in else None
    rows = []
    for t, x in zip(t_list, xl):
        key = min(sols, key=lambda k: abs(k - x))
        m = B.metrics(g, *sols[key], x, t, cin)
        if cin is not None:
            m["var_b_linear_from_input"] = B.linear_var_b(g, cin, t)
        rows.append(m)
    chk = B.evolve(g, nat(g), 0.0, max(t_list))
    kk = np.fft.fftfreq(len(g.b), g.db) * 2 * np.pi
    ph = np.abs(np.fft.fft(chk)) ** 2
    grid_check = {"k_edge": float(ph[np.abs(kk) > 0.5 * np.abs(kk).max()].sum() / ph.sum()),
                  "b_edge": float(np.sum(np.abs(chk[np.abs(g.b) > 0.8 * g.b.max()]) ** 2) / np.sum(np.abs(chk) ** 2))}
    return {"grid_check_linear_natural_tmax": grid_check, "b_cap": g.b_cap, "eps_c": g.eps_c, "X": X, "case": name, "zs": zs,
            "sep": sep, "cubic": cubic, "nb": nb, "bmax": bmax, "x_start": xs, "nfev": nfev, "predicted_D_re": predicted,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", help="<case>@<X>")
    ap.add_argument("--out", required=True, help="output path relative to the work tree")
    a = ap.parse_args()
    out, t0 = {}, time.time()
    for c in a.cases:
        name, X = c.split("@")
        t1 = time.time()
        out["%s@%g" % (name, float(X))] = run_case(float(X), name)
        print(c, "D_re", [round(r["D_re"], 10) for r in out["%s@%g" % (name, float(X))]["rows"]], "%.1f s" % (time.time() - t1),
              flush=True)
    p = WP.wfile(a.out)
    p.write_text(json.dumps({"cases": out, "runtime_s": time.time() - t0}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
