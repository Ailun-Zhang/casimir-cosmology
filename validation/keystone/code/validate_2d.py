# keystone-blind: 2D cross-check CAP vs multichannel Siegert (log-derivative shooting + adiabatic-channel outgoing WKB)
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
Vr = CubicSpline(rf, P["U0"] + q); U2 = CubicSpline(rf, P["U2"])
etas = [8, 16, 32, 64, 128]
out = {}
vals, vecs, r, sig = solvers.eig2d(Vr, U2, 5.6 if lam > -0.012 else 5.0, 0.02, 24, 0.35, 4.4, k=3)
E0 = np.sort(vals.real)[0]
for (Rc, Rmax, dr, Ns, Om0) in [(8.5, 13.5, 0.014, 24, 0.35), (9.5, 15.5, 0.014, 20, 0.45)]:
    t = time.time()
    c = solvers.cap2d(Vr, U2, Rc, Rmax, dr, Ns, Om0, E0, etas)
    print(f"2D CAP Rc={Rc} Rmax={Rmax} dr={dr} Ns={Ns} Om0={Om0}: Ecorr={c['Ecorr']:.9f} gamma={-2*c['Ecorr'].imag:.6e} eta*={c['eta']} ({time.time()-t:.1f}s)", flush=True)
    out[f"cap_{Rc}_{Rmax}_{dr}_{Ns}_{Om0}"] = [c["Ecorr"].real, -2*c["Ecorr"].imag]
Ec = c["Ecorr"]
for (rm, Ns, Om0) in [(9.0, 16, 0.35), (11.0, 16, 0.35), (11.0, 24, 0.35)]:
    t = time.time()
    Es, it = solvers.siegert2d(Vr, U2, Ns, Om0, Ec, rm)
    print(f"2D Siegert rm={rm} Ns={Ns}: Q={Es.real:.9f} gamma={-2*Es.imag:.6e} it={it} ({time.time()-t:.1f}s)", flush=True)
    out[f"siegert_rm{rm}_Ns{Ns}"] = [Es.real, -2*Es.imag]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"validate_2d_{lam}.json"), "w"), indent=1)
