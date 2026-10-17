# keystone-blind (T3): fixed-r ground energy of the local zero-mode problem by unguided DMC (psi_T = 1, mixed estimator = <V>).
# Lab frame: x (3 spatial x 3 colour), w (3), Higgs unit doublet n on S^3 with |Phi| = r/sqrt2 fixed.
#   H_r = -1/2 nabla_x^2 - 1/2 nabla_w^2 - 1/(2 r^2) Delta_{S^3} + V_YM(x) + sum_i |M_i Phi|^2,
#   |M_i Phi|^2 = r^2/8 [g^2 |x_i|^2 + 2 g gY w_i (x_i . v) + gY^2 w_i^2],  v^a = n^dag tau^a n.
# Equivalent to the body-frame fixed-r problem with (2/r^2) J^2 (gauge-invariant ground state is unique and positive).
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np

def pot(X, W, n, r, g, gY, pure):
    c01 = np.cross(X[:, 0], X[:, 1]); c02 = np.cross(X[:, 0], X[:, 2]); c12 = np.cross(X[:, 1], X[:, 2])
    V = 0.5*g*g*((c01**2).sum(1) + (c02**2).sum(1) + (c12**2).sum(1))
    if pure: return V
    z1 = n[:, 0] + 1j*n[:, 1]; z2 = n[:, 2] + 1j*n[:, 3]
    p = np.conj(z1)*z2
    v = np.stack([2*p.real, 2*p.imag, np.abs(z1)**2 - np.abs(z2)**2], 1)
    xv = np.einsum("wia,wa->wi", X, v)
    V += r*r/8*(g*g*(X**2).sum((1, 2)) + 2*g*gY*(W*xv).sum(1) + gY*gY*(W**2).sum(1))
    return V

def run(r, g, gY, tau, Nw, T_eq, T_run, seed, pure=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(Nw, 3, 3))*0.55
    W = rng.normal(size=(Nw, 3))*(1.0/np.sqrt(max(0.15*r, 0.05)))*0.7
    n = rng.normal(size=(Nw, 4)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    V = pot(X, W, n, r, g, gY, pure)
    Eref = V.mean(); st = np.sqrt(tau)
    nsteps = int((T_eq + T_run)/tau); neq = int(T_eq/tau)
    Es, Eg = [], []
    blk, blkg, nb = 0.0, 0.0, 0
    for it in range(nsteps):
        X = X + st*rng.normal(size=X.shape)
        if not pure:
            W = W + st*rng.normal(size=W.shape)
            n = n + (st/r)*rng.normal(size=n.shape); n /= np.linalg.norm(n, axis=1, keepdims=True)
        Vn = pot(X, W, n, r, g, gY, pure)
        w = np.exp(-tau*(0.5*(V + Vn) - Eref))
        mw = w.mean()
        Emix = (w*Vn).sum()/w.sum()
        Egr = Eref - np.log(mw)/tau
        # systematic resampling to fixed population
        c = np.cumsum(w); c /= c[-1]
        u = (rng.random() + np.arange(Nw))/Nw
        idx = np.searchsorted(c, u)
        X, Vn = X[idx], Vn[idx]
        if not pure: W, n = W[idx], n[idx]
        V = Vn
        Eref = 0.99*Eref + 0.01*Emix
        if it >= neq:
            blk += Emix; blkg += Egr; nb += 1
            if nb*tau >= 1.0:
                Es.append(blk/nb); Eg.append(blkg/nb); blk = blkg = 0.0; nb = 0
    Es, Eg = np.array(Es), np.array(Eg)
    return dict(r=r, tau=tau, Nw=Nw, E=float(Es.mean()), err=float(Es.std(ddof=1)/np.sqrt(len(Es))),
                Eg=float(Eg.mean()), errg=float(Eg.std(ddof=1)/np.sqrt(len(Eg))), nblk=len(Es))

def task(a):
    return run(*a)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from multiprocessing import Pool
    g, gY = 0.5268, 0.4415
    mode = sys.argv[1]
    nw = int(os.environ.get("NWORK", "4"))
    out = []
    if mode == "pure":
        tasks = [(0.0, g, gY, tau, 8000, 20.0, 200.0, 11 + k, True) for k, tau in enumerate([0.04, 0.02, 0.01])]
    else:
        rs = [float(x) for x in sys.argv[2].split(",")]
        taus = [float(x) for x in sys.argv[3].split(",")]
        T_run = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
        tasks = [(r, g, gY, tau, 8000, 60.0, T_run, 1000 + 17*k, False) for k, (r, tau) in enumerate([(r, t) for r in rs for t in taus])]
    t0 = time.time()
    with Pool(nw) as pool:
        for res in pool.imap_unordered(task, tasks):
            print(json.dumps(res), round(time.time() - t0, 1), flush=True)
            out.append(res)
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"dmc_{mode}_{int(time.time())}.json")
    json.dump(out, open(fn, "w"), indent=1)
