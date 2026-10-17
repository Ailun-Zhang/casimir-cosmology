# keystone-blind V2: convergence and systematics checks for the cross-term resonances (v11 couplings, lambda=-0.010)
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "work")); sys.path.insert(0, HERE)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.sparse.linalg as spla
import xmodel as X, run_v2 as R, oneloop, twisted as tw
from production import landmarks, outer_turn, ETAS
sys.stdout.reconfigure(encoding="utf-8")
V = R.V11; lam = -0.010; out = {}
XG = X.XGauss(V["g2"], V["gY"], V["yt"], lam, V["g3"], R.CFG["all"])
base = R.solve_x(XG, lam, V["yt"], V["g2"], V["gY"]); out["all_base"] = base["2D"]
print("base", base["2D"]["Q"], base["2D"]["gamma"], flush=True)
for (dr2, Ns, Om0) in [(0.014, 24, 0.35), (0.02, 32, 0.25), (0.02, 20, 0.45)]:
    r = R.solve_x(XG, lam, V["yt"], V["g2"], V["gY"], Ns=Ns, Om0=Om0, dr2=dr2)
    out[f"all_dr{dr2}_Ns{Ns}_Om{Om0}"] = r["2D"]; print(dr2, Ns, Om0, r["2D"]["Q"], r["2D"]["gamma"], flush=True)
# CAP box variation
rf = XG.r; P = XG._pot
rc = np.arange(0.01, 21.0, 0.02); v1 = oneloop.V1loop(rc, lam, V["g2"], V["gY"], V["yt"])
V1 = CubicSpline(np.concatenate([[0.0], rc]), np.concatenate([[0.0], v1]))(np.minimum(rf, 20.99)); q = lam*rf**4/4 + V1
Vr = CubicSpline(rf, P["U0"] + q); U2 = CubicSpline(rf, P["U2"]); VY = X.VYfun(XG)
for (Rc, Rmax) in [(base["2D"]["Rc"] + 1.0, base["2D"]["Rc"] + 7.0), (base["2D"]["Rc"] - 0.7, base["2D"]["Rc"] + 4.0)]:
    c = X.cap2d_x(Vr, U2, VY, Rc, Rmax, 0.02, 24, 0.35, base["2D"]["Ew"], ETAS)
    out[f"all_cap_{Rc:.2f}_{Rmax:.2f}"] = dict(Q=c["Ecorr"].real, gamma=-2*c["Ecorr"].imag); print("cap", Rc, Rmax, c["Ecorr"], flush=True)
# SU(3)-top: variational vs first order (Gaussian <c^2>) vs virial <c^2>=3.77 at selected r
k0 = X.su3_kappa(0.0, V["yt"]); E0, c20 = X.su3_energy(k0, V["g3"]); rows = []
for r in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
    k = X.su3_kappa(r, V["yt"]); E, c2 = X.su3_energy(k, V["g3"])
    rows.append(dict(r=r, dkappa=k - k0, var=E - E0, first=0.5*(k - k0)*c20, virial=0.5*(k - k0)*3.77))
    print("SU3t r=%.1f dkappa=%.5f var=%.5f first=%.5f virial=%.5f" % (r, k - k0, E - E0, 0.5*(k - k0)*c20, 0.5*(k - k0)*3.77), flush=True)
out["su3t_profile"] = rows; out["su3_kappa0"] = k0; out["su3_c2_gauss"] = c20
# kappa profiles (Cartan) along r for the report
kr = []
for r in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
    K = X.cartan_kappa(max(r, 1e-3), V["g2"], V["gY"], V["yt"], lam); kr.append(dict(r=r, k33=K[0, 0], k3w=K[0, 1], kww=K[1, 1]))
    print("kappa r=%.1f k33=%.5f k3w=%.5f kww=%.5f" % (r, K[0, 0], K[0, 1], K[1, 1]), flush=True)
out["kappa_profile"] = kr
# SU(2) variant: W-direction curvature kept massless (kappa_1 = kappa(0)) -> bound on the kappa_1 ~= kappa_33 approximation
class XG1(X.XGauss):
    pass
k00 = X.cartan_kappa(1e-3, V["g2"], V["gY"], V["yt"], lam)[0, 0]
XGs = X.XGauss(V["g2"], V["gY"], V["yt"], lam, V["g3"], ["su2"])
rs = R.solve_x(XGs, lam, V["yt"], V["g2"], V["gY"]); out["su2_base"] = rs["2D"]
XGs.k33 = np.full_like(XGs.k33, k00)        # massless curvature for all x-fluctuations (affects terms(), re-opt not redone)
XGs.__dict__.pop("_pot", None)
rs0 = R.solve_x(XGs, lam, V["yt"], V["g2"], V["gY"]); out["su2_kappa_massless"] = rs0["2D"]
print("su2 base", rs["2D"]["gamma"], " massless-kappa variant", rs0["2D"]["gamma"], flush=True)
json.dump(out, open(os.path.join(HERE, "checks_v2.json"), "w"), indent=1, default=float)
