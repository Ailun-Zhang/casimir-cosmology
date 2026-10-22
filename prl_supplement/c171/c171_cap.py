"""C171 method B -- independent cross-check of the 2D resonance by a complex absorbing potential (CAP).

Independent of c171_core's 2D machinery (separately written):
  r : fourth-order central finite differences of -1/2 (psi'' + psi'/r) on cell-centred nodes r_i = (i-1/2) h,
      even reflection (psi(-r) = psi(r)) at the origin, psi = 0 beyond the last node; non-symmetric matrix.
  s : spectral basis of odd 1D harmonic-oscillator functions phi_{2j+1}(s; w0) (= 3D s-wave oscillator states),
      exact Galerkin matrices of -1/2 d^2/ds^2 and s^2.
  CAP: H(eta) = H - i eta W(r),  W = ((r - Rc)/(rmax - Rc))^p for r > Rc.  The eigenvalue is followed along a
      logarithmic eta trajectory; the estimate is taken where |eta dE/deta| is minimal (zeroth order) and for the
      first-order corrected E1 = E - eta dE/deta where |eta dE1/deta| is minimal (Riss-Meyer).
Units: e = 1, Lambda = lambda/e^2.  Usage: python -B c171_cap.py [n_workers<=4]
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import json
import math
import time
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from multiprocessing import Pool

warnings.filterwarnings("ignore")
# [supplement] path block adapted for standalone use; physics unchanged
RES = os.environ.get("C171_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rerun", "C171"))
RAW = os.path.join(RES, "raw")


def fd4_radial(N, h):
    r = (np.arange(N) + 0.5) * h
    d2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)
    d1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
    rows, cols, vals = [], [], []
    for i in range(N):
        for k, off in enumerate(range(-2, 3)):
            j = i + off
            if j < 0:
                j = -j - 1                     # mirror node (even extension)
            if j >= N:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(-0.5 * (d2[k] + d1[k] / r[i]))
    return r, sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def ho_odd(Nj, w0):
    n = 2 * np.arange(Nj) + 1
    T = np.diag(w0 / 4.0 * (2 * n + 1.0))
    S2 = np.diag((2 * n + 1.0) / (2.0 * w0))
    off = np.sqrt((n[:-1] + 1.0) * (n[:-1] + 2.0))
    T += np.diag(-w0 / 4.0 * off, 1) + np.diag(-w0 / 4.0 * off, -1)
    S2 += np.diag(off / (2.0 * w0), 1) + np.diag(off / (2.0 * w0), -1)
    return T, S2


def build(Lam, h, rmax, Nj, w0):
    N = int(round(rmax / h))
    r, L = fd4_radial(N, h)
    T, S2 = ho_odd(Nj, w0)
    Ij, Ir = sp.identity(Nj, format="csr"), sp.identity(N, format="csr")
    H0 = (sp.kron(L, Ij) + sp.kron(Ir, sp.csr_matrix(T)) + sp.kron(sp.diags(0.5 * r ** 2), sp.csr_matrix(S2))
          + sp.kron(sp.diags(Lam / 4.0 * r ** 4), Ij)).tocsc()
    return r, H0, N


def turning_points(Lam, E):
    U = lambda x: 1.5 * x + 1.0 / (16.0 * x * x) + Lam / 4.0 * x ** 4
    xs = np.linspace(0.3, 4.0 * (1.5 / abs(Lam)) ** (1 / 3), 20000)
    u = U(xs)
    ib = np.argmax(u)
    rb, Umax = xs[ib], u[ib]
    if E < Umax:
        k = ib + np.argmax(u[ib:] < E)
        r3 = xs[k]
    else:
        r3 = rb
    return rb, Umax, r3


def cap_trajectory(task):
    name = task["name"]
    path = os.path.join(RAW, name + ".json")
    if os.path.exists(path):
        return name, "skip"
    t0 = time.time()
    Lam, Eg, h, Nj, w0, p, Lcap, dRc = (task[k] for k in ("Lam", "Eg", "h", "Nj", "w0", "p", "Lcap", "dRc"))
    rb, Umax, r3 = turning_points(Lam, Eg)
    Rc = (r3 if Eg < Umax else rb + 0.5) + dRc
    rmax = Rc + Lcap
    r, H0, N = build(Lam, h, rmax, Nj, w0)
    W = np.where(r > Rc, ((r - Rc) / (rmax - Rc)) ** p, 0.0)
    Wk = sp.kron(sp.diags(W), sp.identity(Nj)).tocsc()
    etas = np.geomspace(task["eta_min"], task["eta_max"], task["n_eta"])
    i0 = int(np.argmin(abs(np.log(etas / task["eta_start"]))))
    well = (r < rb)

    def frac_well(v):
        V = np.abs(v.reshape(N, Nj)) ** 2
        w = V.sum(axis=1)
        return w[well].sum() / w.sum()

    def eig_near(eta, sigma, k):
        H = (H0 - 1j * eta * Wk).tocsc()
        vals, vecs = spla.eigs(H, k=k, sigma=sigma, which="LM", tol=1e-14, ncv=max(2 * k + 2, 20))
        return vals, vecs

    # start: localized, lowest-Re eigenvalue near the guess
    vals, vecs = eig_near(etas[i0], Eg, 10)
    fr = np.array([frac_well(vecs[:, j]) for j in range(len(vals))])
    cand = np.where(fr > 0.5)[0]
    j0 = cand[np.argmin(vals[cand].real)] if len(cand) else int(np.argmax(fr))
    E = np.full(len(etas), np.nan + 0j)
    E[i0] = vals[j0]
    for direction in (+1, -1):
        prev = [E[i0]]
        i = i0 + direction
        while 0 <= i < len(etas):
            guess = prev[-1] if len(prev) < 2 else prev[-1] + (prev[-1] - prev[-2]) * 0.5
            vals, vecs = eig_near(etas[i], guess, 4)
            j = int(np.argmin(abs(vals - prev[-1])))
            E[i] = vals[j]
            prev.append(vals[j])
            i += direction
    lne = np.log(etas)
    dE = np.gradient(E, lne)
    E1 = E - dE
    dE1 = np.gradient(E1, lne)
    # exclude the weak-CAP regime where Im E grows ~ eta (box closed); search where dln|Im E|/dln eta < 0.5
    slope = np.gradient(np.log(np.abs(E.imag)), lne)
    ok = np.where((slope < 0.5) & (np.arange(len(etas)) >= 2) & (np.arange(len(etas)) <= len(etas) - 3))[0]
    if len(ok) == 0:
        ok = np.arange(2, len(etas) - 2)
    io0 = int(ok[np.argmin(abs(dE[ok]))])
    io1 = int(ok[np.argmin(abs(dE1[ok]))])
    out = dict(name=name, Lam=Lam, h=h, Nj=Nj, w0=w0, p=p, Rc=Rc, rmax=rmax, N=N, dim=N * Nj,
               etas=etas.tolist(), E_re=E.real.tolist(), E_im=E.imag.tolist(),
               E0=[E[io0].real, E[io0].imag], eta_opt0=etas[io0], err0=float(abs(dE[io0])),
               E1=[E1[io1].real, E1[io1].imag], eta_opt1=etas[io1], err1=float(abs(dE1[io1])),
               Gamma0=float(-2 * E[io0].imag), Gamma1=float(-2 * E1[io1].imag),
               Gamma_err0=float(2 * abs(dE[io0].imag)), Gamma_err1=float(2 * abs(dE1[io1].imag)),
               wall=time.time() - t0)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(path + ".tmp", path)
    return name, f"done {time.time()-t0:.0f}s G0={out['Gamma0']:.6e} G1={out['Gamma1']:.6e} err1={out['Gamma_err1']:.1e}"


def main():
    nw = min(int(sys.argv[1]) if len(sys.argv) > 1 else 4, 4)
    oned = json.load(open(os.path.join(RAW, "oneD_all.json"), encoding="utf-8"))
    tasks = []
    for jn in ["j04", "j10", "j16", "j19"]:
        Lam = oned[jn]["Lam"]
        Eg = oned[jn]["res_BO"]["E_re"] + 0.1          # guess from the 1D BO model only
        for tag, h, Nj, w0, p, Lcap, dRc in (("a", 0.02, 50, 2.5, 3, 8.0, 1.0), ("b", 0.015, 64, 2.5, 4, 10.0, 1.5)):
            tasks.append(dict(name=f"cap_{jn}_{tag}", Lam=Lam, Eg=Eg, h=h, Nj=Nj, w0=w0, p=p, Lcap=Lcap, dRc=dRc,
                              eta_min=1e-1, eta_max=1e5, n_eta=49, eta_start=100.0))
    todo = [t for t in tasks if not os.path.exists(os.path.join(RAW, t["name"] + ".json"))]
    print(f"{len(tasks)} CAP tasks, {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(nw) as pool:
        for name, msg in pool.imap_unordered(cap_trajectory, todo):
            print(f"[{time.time()-t0:7.1f}s] {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
