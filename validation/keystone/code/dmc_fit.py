# keystone-blind (T3): time-step extrapolation of fixed-r DMC energies and dE(r) = E_exact(r) - E_Gauss(r)
import os, sys, json, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
import model

def extrap(rows):
    t = np.array([x["tau"] for x in rows]); E = np.array([x["E"] for x in rows]); s = np.array([x["err"] for x in rows])
    A = np.vstack([np.ones_like(t), t]).T; W = np.diag(1/s**2)
    cov = np.linalg.inv(A.T @ W @ A); p = cov @ A.T @ W @ E
    chi2 = float(((A @ p - E)**2/s**2).sum())
    out = dict(E0=float(p[0]), err=float(np.sqrt(cov[0, 0])), slope=float(p[1]), chi2=chi2, raw=[[float(a), float(b), float(c)] for a, b, c in zip(t, E, s)])
    if len(t) >= 3:
        A2 = np.vstack([np.ones_like(t), t, t*t]).T; cov2 = np.linalg.inv(A2.T @ W @ A2); p2 = cov2 @ A2.T @ W @ E
        out.update(E0_quad=float(p2[0]), err_quad=float(np.sqrt(cov2[0, 0])))
    return out

pure = json.load(open(sorted(glob.glob(os.path.join(RES, "dmc_pure_*.json")))[-1]))
fixr = json.load(open(sorted(glob.glob(os.path.join(RES, "dmc_fixr_*.json")))[-1]))
g, gY = 0.5268, 0.4415
ep = extrap(sorted(pure, key=lambda x: x["tau"]))
eps2_exact = 4.116720 * g**(2/3)
EG0 = 27/16*(16*g*g)**(1/3)
print("pure SU(2): DMC E0 = %.5f +- %.5f (quad %.5f); Luscher eps2 g^2/3 = %.5f; Gaussian %.5f" % (ep["E0"], ep["err"], ep.get("E0_quad", np.nan), eps2_exact, EG0))
G = model.Gauss(g, gY)
res = dict(pure=ep, eps2_g23=eps2_exact, EG_pure=EG0, r=[0.0], dE=[eps2_exact - EG0], dE_err=[0.0], fixr={})
for r in sorted(set(x["r"] for x in fixr)):
    rows = sorted([x for x in fixr if x["r"] == r], key=lambda x: x["tau"])
    e = extrap(rows)
    EG = float(G.terms(np.array([r]))["EG"][0])
    e["EG"] = EG; e["dE"] = e["E0"] - EG
    res["fixr"][str(r)] = e
    res["r"].append(r); res["dE"].append(e["dE"]); res["dE_err"].append(e["err"])
    print("r=%.2f  E0=%.5f +- %.5f (quad %.5f +- %.5f) chi2=%.2f  EG=%.5f  dE=%.5f" % (r, e["E0"], e["err"], e.get("E0_quad", np.nan), e.get("err_quad", np.nan), e["chi2"], EG, e["dE"]))
d4, d5 = res["dE"][-2], res["dE"][-1]
p = float(np.log(d4/d5)/np.log(5/4)) if (d4 < 0 and d5 < 0 and d4 < d5) else 4.0
res["tail_power"] = min(max(p, 2.0), 6.0)
print("tail power", res["tail_power"])
json.dump(res, open(os.path.join(RES, "dE_fit.json"), "w"), indent=1)
