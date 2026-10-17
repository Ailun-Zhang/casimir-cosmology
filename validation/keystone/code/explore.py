# keystone-blind: exploratory checks (lambda=0 local bound states; potentials at lambda<0; one resonance by all methods)
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.sparse.linalg as spla
sys.stdout.reconfigure(encoding="utf-8")
import model, solvers, oneloop

g2, gY, yt = 0.5268, 0.4415, 0.4585
t0 = time.time()
G = model.Gauss(g2, gY)
rf = G.r
P = G.potentials(rf)
print("gauss done", time.time() - t0)

def mk(lam, match=True):
    rc = np.arange(0.01, 27.0, 0.02)
    v1 = oneloop.V1loop(rc, lam, g2, gY, yt) if match else 0*rc
    v1s = CubicSpline(np.concatenate([[0.0], rc]), np.concatenate([[0.0], v1]))
    V1 = v1s(rf)
    q = lam*rf**4/4 + V1
    return dict(V1D=CubicSpline(rf, P["V1D"] + q), Vlow=CubicSpline(rf, P["Vlow"] + q),
                Vr=CubicSpline(rf, P["U0"] + q), U2=CubicSpline(rf, P["U2"]), V1=V1)

# lambda = 0 local
M0 = mk(0.0, match=False)
for name in ["V1D", "Vlow"]:
    for dr in [0.02, 0.01]:
        vals, vecs, r = solvers.eig1d(M0[name], 16.0, dr, 3.0, k=3)
        print(f"lam0 local {name} dr={dr}: E0={np.sort(vals.real)[:3]}")
for Ns, Om0 in [(24, 0.35), (32, 0.35), (32, 0.25), (40, 0.25)]:
    for dr in [0.02]:
        t1 = time.time()
        vals, vecs, r, sig = solvers.eig2d(M0["Vr"], M0["U2"], 16.0, dr, Ns, Om0, 4.3, k=4)
        print(f"lam0 local 2D Ns={Ns} Om0={Om0} dr={dr}: E={np.sort(vals.real)[:4]}  smax={np.sqrt(sig.max()):.1f}  t={time.time()-t1:.1f}s")
# potentials at lambda<0 with matching
for lam in [-0.008, -0.010, -0.0143]:
    M = mk(lam)
    rr = np.arange(1, 16.1, 1.0)
    print("lam", lam, "V1D:", np.array2string(M["V1D"](rr), precision=3, max_line_width=250))
    rg = np.linspace(1, 15, 2801); v = M["V1D"](rg); i = np.argmax(v)
    print("   barrier top r=%.3f V=%.4f" % (rg[i], v[i]))
print("time", time.time() - t0)
