# keystone-blind: validate resonance methods at lambda=-0.010 (v11 couplings, with matching)
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline
sys.stdout.reconfigure(encoding="utf-8")
import model, solvers, oneloop

g2, gY, yt = 0.5268, 0.4415, 0.4585
lam = float(sys.argv[1]) if len(sys.argv) > 1 else -0.010
G = model.Gauss(g2, gY)
rf = G.r; P = G.potentials(rf)
rc = np.arange(0.01, 27.0, 0.02)
v1 = oneloop.V1loop(rc, lam, g2, gY, yt)
V1 = CubicSpline(np.concatenate([[0.0], rc]), np.concatenate([[0.0], v1]))(rf)
q = lam*rf**4/4 + V1
V1D = CubicSpline(rf, P["V1D"] + q); Vlow = CubicSpline(rf, P["Vlow"] + q)
Vr = CubicSpline(rf, P["U0"] + q); U2 = CubicSpline(rf, P["U2"])
etas = [1, 2, 4, 8, 16, 32, 64, 128, 256]
out = {}
for name, Vs in [("1D", V1D), ("low", Vlow)]:
    vals, vecs, r = solvers.eig1d(Vs, 5.2 if lam < -0.012 else 5.6, 0.02, 4.4, k=2)
    E0 = np.sort(vals.real)[0]
    for (Rc, Rmax, dr) in [(7.5, 11.5, 0.02), (8.5, 13.5, 0.02), (8.5, 13.5, 0.01)]:
        t = time.time()
        c = solvers.cap1d(Vs, Rc, Rmax, dr, E0, etas)
        print(f"{name} CAP Rc={Rc} Rmax={Rmax} dr={dr}: E={c['E']:.8f} Ecorr={c['Ecorr']:.8f} eta*={c['eta']} ({time.time()-t:.1f}s)")
        for tr in c["traj"]: print("     eta=%6g E=%.8f%+.8fj corr=%.8f%+.8fj |eta dE|=%.2e" % (tr[0], tr[1].real, tr[1].imag, tr[2].real, tr[2].imag, tr[3]))
    for rm in [9.0, 11.0, 13.0]:
        t = time.time()
        Es, it = solvers.siegert1d(Vs, c["Ecorr"], rm)
        print(f"{name} Siegert rm={rm}: E={Es:.10f} Q={Es.real:.8f} gamma={-2*Es.imag:.6e} it={it} ({time.time()-t:.1f}s)")
        out[f"{name}_siegert_rm{rm}"] = [Es.real, -2*Es.imag]
# 2D
vals, vecs, r, sig = solvers.eig2d(Vr, U2, 5.6, 0.02, 24, 0.35, 4.4, k=3)
E0 = np.sort(vals.real)[0]; print("2D wall estimate", E0)
for (Rc, Rmax, dr, Ns, Om0) in [(7.5, 11.5, 0.02, 24, 0.35), (8.5, 13.5, 0.02, 24, 0.35), (8.5, 13.5, 0.02, 32, 0.25), (8.5, 13.5, 0.014, 24, 0.35)]:
    t = time.time()
    c = solvers.cap2d(Vr, U2, Rc, Rmax, dr, Ns, Om0, E0, etas)
    print(f"2D CAP Rc={Rc} Rmax={Rmax} dr={dr} Ns={Ns} Om0={Om0}: E={c['E']:.8f} Ecorr={c['Ecorr']:.8f} eta*={c['eta']} ({time.time()-t:.1f}s)")
    for tr in c["traj"]: print("     eta=%6g E=%.8f%+.8fj corr=%.8f%+.8fj |eta dE|=%.2e" % (tr[0], tr[1].real, tr[1].imag, tr[2].real, tr[2].imag, tr[3]))
    out[f"2D_cap_{Rc}_{Rmax}_{dr}_{Ns}"] = [c["Ecorr"].real, -2*c["Ecorr"].imag]
Ec = c["Ecorr"]
for rm in [9.0, 11.0]:
    t = time.time()
    Es, it = solvers.siegert2d(Vr, U2, 24, 0.35, Ec, rm)
    print(f"2D Siegert rm={rm}: E={Es:.10f} Q={Es.real:.8f} gamma={-2*Es.imag:.6e} it={it} ({time.time()-t:.1f}s)")
    out[f"2D_siegert_rm{rm}"] = [Es.real, -2*Es.imag]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"validate_res_{lam}.json"), "w"), indent=1)
