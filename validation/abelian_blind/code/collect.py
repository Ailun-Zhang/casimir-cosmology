"""Collect Method A / Method B runs into blind_results.json and print the summary tables."""
import os, json
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
mp.mp.dps = 30

KEYS = ["e=1,lam=-1", "e=1,lam=-0.11174", "e=1,lam=-0.03", "e=1,lam=-0.01003",
        "e=0.873,lam=-0.02286387", "e=1,lam=-0.05", "e=1,lam=-0.02"]


def load(name):
    p = os.path.join(ROOT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def summarize_A(runs, key):
    rr = [r for r in runs if r["key"] == key]
    if not rr:
        return None
    def gflux(r):
        return float(np.mean([f["Gamma_flux"] for f in r["flux"]]))
    byh = {}
    for r in rr:
        if r["tag"].startswith("h=") and "," not in r["tag"]:
            byh[r["h"]] = r
    hs = sorted(byh)
    finest = byh[hs[0]]
    second = byh[hs[1]] if len(hs) > 1 else None
    var = [r for r in rr if "," in r["tag"]]  # h=0.04 variants
    ref04 = byh.get(0.04)
    table = []
    for h in sorted(byh, reverse=True):
        r = byh[h]
        table.append(dict(h=h, n=r["n"], eps=r["eps"], Gamma_eig=r["Gamma_eig"], Gamma_flux=gflux(r),
                          flux_R=[f["R"] for f in r["flux"]],
                          flux_spread_rel=float(np.ptp([f["Gamma_flux"] for f in r["flux"]]) / gflux(r))))
    vt = []
    for r in var:
        vt.append(dict(tag=r["tag"], eps=r["eps"], Gamma_eig=r["Gamma_eig"], Gamma_flux=gflux(r)))
    G = gflux(finest)
    e_ref = finest["eps"]
    d_eps_h = abs(finest["eps"] - second["eps"]) if second else float("nan")
    d_G_h = abs(gflux(finest) - gflux(second)) / G if second else float("nan")
    d_eps_v = max([abs(v["eps"] - ref04["eps"]) for v in vt], default=0.0) if ref04 else 0.0
    d_G_v = max([abs(v["Gamma_flux"] - gflux(ref04)) / G for v in vt], default=0.0) if ref04 else 0.0
    return dict(best_h=hs[0], eps=e_ref, Gamma_flux=G, Gamma_eig=finest["Gamma_eig"],
                eps_abs_err=max(d_eps_h, d_eps_v), Gamma_rel_err=max(d_G_h, d_G_v),
                refinement=table, variants_h004=vt,
                params=dict(R0=finest["R0"], w=finest["w"], theta=finest["theta"], Rmax=finest["Rmax"],
                            Smax=finest["Smax"], p=finest["p"], mask_c=finest["mask_c"], switch="erf, a=6"))


def main():
    A = load("methodA_results.json")
    B = load("methodB_results.json")
    S = load("slope_results.json")
    out = {"problem": "H = -1/2(d_r^2 + r^-1 d_r) - 1/2(d_s^2 + 2 s^-1 d_s) + e^2 r^2 s^2/2 + lam r^4/4, "
                      "measure r dr s^2 ds, L = 1; resonance E = eps - i Gamma/2",
           "results": [], "scaling_check": None, "optional_slope": None}
    rows = []
    for key in KEYS:
        b = B["summary"].get(key) if B else None
        a = summarize_A(A["runs"], key) if A else None
        if b is None:
            continue
        epsB, GB = mp.mpf(b["eps"]), mp.mpf(b["Gamma"])
        rec = dict(key=key, e=b["e"], lam=b["lam"], eps=b["eps"], Gamma=b["Gamma"])
        rec["methodB"] = dict(eps=b["eps"], Gamma=b["Gamma"], eps_abs_err=b["eps_abs_err"],
                              Gamma_rel_err=b["Gamma_rel_err"],
                              discretization="Nr=150 (2D-HO m=0, beta=1) x Ns=70 (3D-HO l=0, gamma=1), "
                                             "theta=pi/8; variations: N, theta in {pi/12,pi/8,pi/6}, "
                                             "phi=-pi/16, beta,gamma in {0.7,1,1.5}",
                              basis_sequence=b["basis_sequence"])
        if a is not None:
            rec["methodA"] = a
            dAe = abs(a["eps"] - float(epsB))
            dAG = abs(a["Gamma_flux"] - float(GB)) / float(GB)
            dAGe = abs(a["Gamma_eig"] - float(GB)) / float(GB)
            rec["A_minus_B"] = dict(eps_abs=dAe, Gamma_flux_rel=dAG, Gamma_eig_rel=dAGe)
            eps_err = max(b["eps_abs_err"], dAe)
            G_err = max(b["Gamma_rel_err"], dAG)
        else:
            eps_err, G_err = b["eps_abs_err"], b["Gamma_rel_err"]
        rec["eps_rel_err_conservative"] = eps_err / float(epsB)
        rec["Gamma_rel_err_conservative"] = G_err
        rec["eps_rel_err_methodB_internal"] = b["eps_abs_err"] / float(epsB)
        rec["Gamma_rel_err_methodB_internal"] = b["Gamma_rel_err"]
        rec["methods"] = ["B: HO product basis + complex rotation (eigenvalue, 40-digit Rayleigh quotient)",
                          "A: smooth exterior complex scaling on 2D FD grid (eigenvalue and flux formula J/N)"]
        out["results"].append(rec)
        rows.append(rec)
    # scaling check
    if B:
        b1 = B["summary"]["e=1,lam=-0.03"]
        b2 = B["summary"]["e=0.873,lam=-0.02286387"]
        f = mp.mpf("0.873") ** (mp.mpf(2) / 3)
        sc = dict(statement="H(e,lam) is unitarily equivalent to e^{2/3} H(1, lam/e^2) under the dilation "
                            "(r,s) -> e^{-1/3}(r,s) (kinetic ~ length^-2, e^2 r^2 s^2 and lam r^4 ~ length^4), "
                            "hence eps(e,lam) = e^{2/3} eps(1,lam/e^2), Gamma(e,lam) = e^{2/3} Gamma(1,lam/e^2)",
                  factor_e23=mp.nstr(f, 15))
        for tag, d1, d2 in [("methodB", b1, b2)]:
            e1, g1 = mp.mpf(d1["eps"]), mp.mpf(d1["Gamma"])
            e2, g2 = mp.mpf(d2["eps"]), mp.mpf(d2["Gamma"])
            sc[tag] = dict(eps_0873=mp.nstr(e2, 18), e23_eps_1=mp.nstr(f * e1, 18),
                           eps_rel_dev=float((e2 - f * e1) / e2),
                           Gamma_0873=mp.nstr(g2, 15), e23_Gamma_1=mp.nstr(f * g1, 15),
                           Gamma_rel_dev=float((g2 - f * g1) / g2))
        if A:
            a1 = summarize_A(A["runs"], "e=1,lam=-0.03")
            a2 = summarize_A(A["runs"], "e=0.873,lam=-0.02286387")
            if a1 and a2:
                ff = float(f)
                sc["methodA"] = dict(eps_0873=a2["eps"], e23_eps_1=ff * a1["eps"],
                                     eps_rel_dev=(a2["eps"] - ff * a1["eps"]) / a2["eps"],
                                     Gamma_0873=a2["Gamma_flux"], e23_Gamma_1=ff * a1["Gamma_flux"],
                                     Gamma_rel_dev=(a2["Gamma_flux"] - ff * a1["Gamma_flux"]) / a2["Gamma_flux"])
        out["scaling_check"] = sc
    if S:
        out["optional_slope"] = S["slopes"]
        out["optional_dense_GammaB"] = S["dense"]
    json.dump(out, open(os.path.join(ROOT, "blind_results.json"), "w"), indent=1, default=str)
    # print table
    print(f"{'case':26s} {'eps':>22s} {'Gamma':>24s} {'rel.err eps':>11s} {'rel.err Gamma':>13s} "
          f"{'|A-B| eps':>10s} {'|A-B|/G flux':>12s}")
    for r in rows:
        ab = r.get("A_minus_B", {})
        print(f"{r['key']:26s} {r['eps']:>22s} {r['Gamma']:>24s} {r['eps_rel_err_conservative']:11.1e} "
              f"{r['Gamma_rel_err_conservative']:13.1e} {ab.get('eps_abs', float('nan')):10.1e} "
              f"{ab.get('Gamma_flux_rel', float('nan')):12.1e}")
    if out["scaling_check"]:
        print(json.dumps(out["scaling_check"], indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
