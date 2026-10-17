# keystone-blind (T3): hazard collection -> P1(M_t), critical M_t, lambda(mu_b) at m_c, numerical error estimates
import os, sys, json, glob
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
HLB = 0.00437243
tag = sys.argv[1] if len(sys.argv) > 1 else ""          # "" -> 2D_G shards ; "ex" -> 2D_ex' shards
shdir = os.path.join(RES, "hazard_shards" + ("_" + tag if tag else ""))
C = json.load(open(os.path.join(RES, "couplings.json")))

def load():
    D = {}
    for f in glob.glob(os.path.join(shdir, "h_*.json")):
        d = json.load(open(f))
        D.setdefault((round(d["als"], 4), round(d["Mt"], 2)), {})[d["inode"]] = d
    return D

def hazard_integral(us, g, kind="cubic", umax=5.0):
    g = np.maximum(np.asarray(g, float), 1e-300); lg = np.log(g)
    if kind == "cubic": f = CubicSpline(us, lg)
    elif kind == "pchip": f = PchipInterpolator(us, lg)
    else: f = lambda u: np.interp(u, us, lg)
    w = lambda u: np.exp(f(u))/(HLB*np.sqrt(np.cosh(2*u)))
    val, err = quad(w, 0.0, umax, limit=400, epsabs=0, epsrel=1e-11, points=list(us))
    return val

GMIN_2D = 1e-6      # below this the 2D CAP width is not resolved -> fallback gamma_low(Siegert) * ratio
def pick_1d(d):
    """Siegert is the accurate method for narrow resonances; accept it when its Q agrees with CAP Q."""
    if abs(d["Q_sieg"] - d["Q_cap"]) < 2e-4 and d["gamma_sieg"] > 0:
        return d["gamma_sieg"], d["Q_sieg"], "sieg"
    return d["gamma_cap"], d["Q_cap"], "cap(!)"

def gammas(nd, model, ratio_fallback):
    g, how = [], []
    for i in sorted(nd):
        if model in ("V1D", "Vlow"):
            gg, qq, h = pick_1d(nd[i][model]); g.append(gg); how.append(h)
        else:
            gl, ql, hl = pick_1d(nd[i]["Vlow"]); g1, q1, h1 = pick_1d(nd[i]["V1D"])
            d = nd[i]["2D"]
            ok = d["gamma"] > GMIN_2D and gl <= d["gamma"] <= g1 and ql <= d["Q"] <= q1
            if ok: g.append(d["gamma"]); how.append("cap2d")
            else: g.append(gl*ratio_fallback); how.append("fallback")
    return np.array(g), how

def main():
    D = load()
    out = dict(models={}, flags=[], diag=[])
    keys = sorted(D)
    # 2D/low ratio from resolved nodes (for fallback)
    rat = [nd_i["2D"]["gamma"]/pick_1d(nd_i["Vlow"])[0] for nd in D.values() for nd_i in nd.values()
           if nd_i["2D"]["gamma"] > 1e-5 and nd_i["2D"].get("bracket_ok")]
    ratio_fb = float(np.median(rat)) if rat else 1.07
    out["ratio_2D_over_low"] = dict(median=ratio_fb, min=float(np.min(rat)) if rat else None, max=float(np.max(rat)) if rat else None, n=len(rat))
    print("2D/low ratio median %.4f range [%.4f, %.4f] n=%d" % (ratio_fb, np.min(rat), np.max(rat), len(rat)))
    for model in ["2D", "V1D", "Vlow"]:
        rows = {}
        for (als, Mt) in keys:
            nd = D[(als, Mt)]
            if len(nd) < 13: continue
            us = np.array([nd[i]["u"] for i in sorted(nd)])
            g, how = gammas(nd, model, ratio_fb)
            for i, h in zip(sorted(nd), how):
                if h not in ("sieg", "cap2d"):
                    out["flags"].append([model, als, Mt, i, h, float(g[i])])
            if model == "2D":
                gfb_lo, _ = gammas(nd, model, 1.0); gfb_hi, _ = gammas(nd, model, 1.15)
            else:
                gfb_lo = gfb_hi = g
            I = hazard_integral(us, g)
            I_lin = hazard_integral(us, g, "linear"); I_pch = hazard_integral(us, g, "pchip")
            sub = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12]       # drop nodes 0.25,0.75,1.25
            I_sub = hazard_integral(us[sub], g[sub])
            I_fb = [hazard_integral(us, gfb_lo), hazard_integral(us, gfb_hi)]
            rows[f"{als:.4f}_{Mt:.2f}"] = dict(als=als, Mt=Mt, I=I, P1=float(np.exp(-I)), I_lin=I_lin, I_pchip=I_pch, I_subnodes=I_sub, I_fallback_range=I_fb,
                                              n_fallback=int(sum(h == "fallback" for h in how)),
                                              gamma_u0=float(g[0]), lam_mub=C["grid"][f"{als:.4f}_{Mt:.2f}"][0]["lam"],
                                              Q_u0=float(nd[0][model]["Q"]), gammas=g.tolist())
        out["models"][model] = rows
    # critical masses per alpha_s
    crit = {}
    for model, rows in out["models"].items():
        for als in sorted(set(v["als"] for v in rows.values())):
            rr = sorted([v for v in rows.values() if v["als"] == als], key=lambda v: v["Mt"])
            # drop leading points that break monotonicity (P1~1 region where widths <1e-8 are not resolved)
            while len(rr) > 4 and rr[0]["I"] >= rr[1]["I"]:
                out.setdefault("dropped", []).append([model, als, rr[0]["Mt"], rr[0]["I"]]); rr = rr[1:]
            if len(rr) < 4: continue
            M = np.array([v["Mt"] for v in rr]); lI = np.log([v["I"] for v in rr])
            lIl = np.log([v["I_lin"] for v in rr]); lIs = np.log([v["I_subnodes"] for v in rr])
            sp = CubicSpline(M, lI); spl = CubicSpline(M, lIl); sps = CubicSpline(M, lIs)
            spf0 = CubicSpline(M, np.log([v["I_fallback_range"][0] for v in rr])); spf1 = CubicSpline(M, np.log([v["I_fallback_range"][1] for v in rr]))
            lamsp = CubicSpline(M, [v["lam_mub"] for v in rr])
            if not np.all(np.diff(lI) > 0): print("WARNING non-monotone ln I for", model, als, np.round(lI, 3))
            res = {}
            for P in [0.9, 0.5, 0.1]:
                tgt = np.log(-np.log(P)); fine = np.linspace(M[0], M[-1], 200001)
                def root(s):
                    v = s(fine) - tgt; j = np.where(np.diff(np.sign(v)) != 0)[0]
                    return float(fine[j[0]]) if len(j) else None
                mc = root(sp); mc_lin = root(spl); mc_sub = root(sps)
                mc_l = float(np.interp(tgt, lI, M)) if np.all(np.diff(lI) > 0) else None
                mf0, mf1 = root(spf0), root(spf1)
                res[f"P{P}"] = dict(mc=mc, lam_mub=float(lamsp(mc)) if mc else None,
                                    d_fallback=[(mf0 - mc) if (mc and mf0) else None, (mf1 - mc) if (mc and mf1) else None],
                                    d_quad_u=(mc_lin - mc) if (mc and mc_lin) else None,
                                    d_nodes=(mc_sub - mc) if (mc and mc_sub) else None,
                                    d_Mt_interp=(mc_l - mc) if (mc and mc_l) else None)
            for Mq in [172.2, 172.6, 173.0]:
                res[f"P1_{Mq}"] = float(np.exp(-np.exp(sp(Mq))))
            crit[f"{model}_{als:.4f}"] = res
    out["critical"] = crit
    json.dump(out, open(os.path.join(RES, f"hazard_summary{('_' + tag) if tag else ''}.json"), "w"), indent=1)
    for k, v in crit.items():
        print(k, {kk: (round(vv["mc"], 4) if isinstance(vv, dict) and vv["mc"] else vv) for kk, vv in v.items()})
        print("    lam_mub at mc:", {kk: (round(vv["lam_mub"], 6) if isinstance(vv, dict) and vv["lam_mub"] else None) for kk, vv in v.items() if isinstance(vv, dict)})
        print("    errs:", {kk: (vv["d_quad_u"], vv["d_nodes"], vv["d_Mt_interp"]) for kk, vv in v.items() if isinstance(vv, dict)})
    for model in ["2D", "V1D"]:
        for k in sorted(out["models"][model]):
            v = out["models"][model][k]
            print(model, k, "P1=%.5f I=%.5f g(u=0)=%.4e lam_mub=%.6f" % (v["P1"], v["I"], v["gamma_u0"], v["lam_mub"]))
    print("flags:", out["flags"][:20], len(out["flags"]))

if __name__ == "__main__":
    main()
