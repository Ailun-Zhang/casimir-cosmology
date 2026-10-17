# PROVENANCE: copied from D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\keystone_blind\work\solvers.py on 2026-10-15 by ui-blind (unmodified below this line)
# keystone-blind (T3): resonance / bound-state solvers for the reduced models.
# u(r) = r^{3/2} chi(r) (4D radial), u(r,s) = r^{3/2} s chi(r,s) (s=|w|, s-wave).
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

# ---------------- FD operators ----------------
def kin_r(N, dr):
    """-(1/2) d^2/dr^2, 4th order, u(0)=0 with odd ghost; Dirichlet + odd ghost at r_{N+1}."""
    c = -1.0/(2*12*dr*dr)
    main = np.full(N, -30.0); off1 = np.full(N-1, 16.0); off2 = np.full(N-2, -1.0)
    main[0] = -29.0; main[-1] = -29.0
    return sp.diags([off2, off1, main, off1, off2], [-2, -1, 0, 1, 2], format="csr") * c

def sdvr(Ns, Om0):
    """odd-HO basis on s in (0,inf) (u(0)=0); returns DVR points s_j^2 and kinetic matrix."""
    n = np.arange(Ns)
    s2 = np.diag((4*n+3)/(2*Om0)) + np.diag(np.sqrt((2*n[:-1]+2)*(2*n[:-1]+3))/(2*Om0), 1) + np.diag(np.sqrt((2*n[:-1]+2)*(2*n[:-1]+3))/(2*Om0), -1)
    T = np.diag(Om0*(4*n+3)/4) - np.diag(Om0/4*np.sqrt((2*n[:-1]+2)*(2*n[:-1]+3)), 1) - np.diag(Om0/4*np.sqrt((2*n[:-1]+2)*(2*n[:-1]+3)), -1)
    sig, U = np.linalg.eigh(s2)
    return sig, U.T @ T @ U

# ---------------- 1D ----------------
def cap_W(r, Rc, Rmax):
    x = np.clip((r - Rc)/(Rmax - Rc), 0, None)
    return x*x

def eig1d(Vfun, Rmax, dr, sigma, k=4, eta=0.0, Rc=None):
    N = int(round(Rmax/dr)) - 1
    r = dr*np.arange(1, N+1)
    H = kin_r(N, dr) + sp.diags(Vfun(r) + 3/(8*r*r))
    if eta:
        H = H.astype(complex) - 1j*eta*sp.diags(cap_W(r, Rc, Rmax))
    vals, vecs = spla.eigs(H.tocsc(), k=k, sigma=sigma)
    return vals, vecs, r

def cap1d(Vfun, Rc, Rmax, dr, E0, etas):
    """CAP trajectory; returns dict with best (min |eta dE/deta|) raw and first-order corrected values."""
    traj = []
    for eta in etas:
        vals, vecs, r = eig1d(Vfun, Rmax, dr, E0, k=6, eta=eta, Rc=Rc)
        # pick eigenvalue closest to current estimate
        i = np.argmin(np.abs(vals - E0)); E = vals[i]; v = vecs[:, i]
        W = cap_W(r, Rc, Rmax)
        dEdeta = -1j*np.sum(v*v*W)/np.sum(v*v)        # c-product first-order: dE/deta = -i <v|W|v>_c
        traj.append((eta, E, E - eta*dEdeta, abs(eta*dEdeta)))
        E0 = E
    j = int(np.argmin([t[3] for t in traj]))
    return dict(traj=traj, E=traj[j][1], Ecorr=traj[j][2], eta=traj[j][0])

def wkb_out(E, r, Vs):
    """outgoing log-derivative (2nd-order WKB) of u'' = -Pi u, Pi = 2(E - Veff)."""
    V, V1, V2 = Vs(r), Vs(r, 1), Vs(r, 2)
    Pi = 2*(E - V - 3/(8*r*r)); dPi = -2*(V1 - 3/(4*r**3)); d2Pi = -2*(V2 + 9/(4*r**4))
    p = np.sqrt(Pi + 0j)
    dp = dPi/(2*p); d2p = (d2Pi - 2*dp*dp)/(2*p)
    a = -dp/(2*p)
    return 1j*p + a + 1j/(2*p)*(0.75*(dp/p)**2 - d2p/(2*p))

def shoot1d(Vs, E, rm, r0=2e-3):
    """regular solution to rm; returns u'/u."""
    V0 = float(Vs(r0))
    u0 = r0**1.5*(1 + (V0 - E)*r0*r0/4); du0 = 1.5*r0**0.5 + 3.5*(V0 - E)*r0**2.5/4
    def f(r, y):
        return [y[1], 2*(Vs(r) + 3/(8*r*r) - E)*y[0]]
    sol = solve_ivp(f, (r0, rm), [complex(u0), complex(du0)], method="DOP853", rtol=1e-11, atol=1e-30)
    u, du = sol.y[0, -1], sol.y[1, -1]
    return du/u

def siegert1d(Vs, E0, rm, tol=1e-12, maxit=60):
    f = lambda E: shoot1d(Vs, E, rm) - wkb_out(E, rm, Vs)
    E1 = E0; E2 = E0*(1 + 1e-7) - 1e-9j
    f1, f2 = f(E1), f(E2)
    for it in range(maxit):
        E3 = E2 - f2*(E2 - E1)/(f2 - f1)
        E1, f1 = E2, f2; E2 = E3; f2 = f(E2)
        if abs(E2 - E1) < tol*max(1, abs(E2)): return E2, it
    return E2, -1

# ---------------- 2D ----------------
def build2d(Vr, U2, Rmax, dr, Ns, Om0, eta=0.0, Rc=None):
    N = int(round(Rmax/dr)) - 1
    r = dr*np.arange(1, N+1)
    sig, Ts = sdvr(Ns, Om0)
    Kr = kin_r(N, dr)
    H = sp.kron(Kr, sp.identity(Ns)) + sp.kron(sp.identity(N), sp.csr_matrix(Ts))
    diag = (Vr(r) + 3/(8*r*r))[:, None] + U2(r)[:, None]*sig[None, :]
    H = H + sp.diags(diag.ravel())
    if eta:
        H = H.astype(complex) - 1j*eta*sp.diags(np.repeat(cap_W(r, Rc, Rmax), Ns))
    return H.tocsc(), r, sig

def eig2d(Vr, U2, Rmax, dr, Ns, Om0, sigma, k=6, eta=0.0, Rc=None):
    H, r, sig = build2d(Vr, U2, Rmax, dr, Ns, Om0, eta, Rc)
    vals, vecs = spla.eigs(H, k=k, sigma=sigma)
    return vals, vecs, r, sig

def cap2d(Vr, U2, Rc, Rmax, dr, Ns, Om0, E0, etas, k=6):
    traj = []
    for eta in etas:
        vals, vecs, r, sig = eig2d(Vr, U2, Rmax, dr, Ns, Om0, E0, k=k, eta=eta, Rc=Rc)
        i = np.argmin(np.abs(vals - E0)); E = vals[i]; v = vecs[:, i]
        W = np.repeat(cap_W(r, Rc, Rmax), len(sig))
        dEdeta = -1j*np.sum(v*v*W)/np.sum(v*v)
        traj.append((eta, E, E - eta*dEdeta, abs(eta*dEdeta)))
        E0 = E
    j = int(np.argmin([t[3] for t in traj]))
    return dict(traj=traj, E=traj[j][1], Ecorr=traj[j][2], eta=traj[j][0])

def shoot2d_Y(Vr_s, U2_s, Ns, Om0, E, rm, r0=5e-3, seg=0.25):
    """multichannel regular solutions (s-DVR channels), QR-stabilised; returns Y=U'U^-1 at rm."""
    sig, Ts = sdvr(Ns, Om0)
    I = np.eye(Ns)
    def rhs(r, y):
        U = y[:Ns*Ns].reshape(Ns, Ns); dU = y[Ns*Ns:].reshape(Ns, Ns)
        W = 2*((Vr_s(r) + 3/(8*r*r) - E)*I + Ts + U2_s(r)*np.diag(sig))
        return np.concatenate([dU.ravel(), (W @ U).ravel()])
    U = r0**1.5*I.astype(complex); dU = 1.5*r0**0.5*I.astype(complex)
    edges = np.arange(r0, rm, seg); edges = np.append(edges, rm)
    for a, b in zip(edges[:-1], edges[1:]):
        sol = solve_ivp(rhs, (a, b), np.concatenate([U.ravel(), dU.ravel()]), method="DOP853", rtol=1e-10, atol=1e-40)
        U = sol.y[:Ns*Ns, -1].reshape(Ns, Ns); dU = sol.y[Ns*Ns:, -1].reshape(Ns, Ns)
        Q, R = np.linalg.qr(U)
        Ri = np.linalg.inv(R); U = U @ Ri; dU = dU @ Ri
    return dU @ np.linalg.inv(U), sig, Ts

def yout2d(E, rm, Vr_s, U2_s, sig, Ts, h=1e-3):
    def chan(r):
        w, Phi = np.linalg.eigh(Ts + U2_s(r)*np.diag(sig))
        return w, Phi
    w0, Phi = chan(rm); wm, _ = chan(rm - h); wp, _ = chan(rm + h)
    Vm, V0, Vp = Vr_s(rm - h), Vr_s(rm), Vr_s(rm + h)
    ys = []
    for n in range(len(w0)):
        Pf = lambda rr, wv, Vv: 2*(E - Vv - 3/(8*rr*rr) - wv)
        Pim, Pi0, Pip = Pf(rm-h, wm[n], Vm), Pf(rm, w0[n], V0), Pf(rm+h, wp[n], Vp)
        dPi = (Pip - Pim)/(2*h); d2Pi = (Pip - 2*Pi0 + Pim)/h**2
        # outgoing for open channels (Re p>0), decaying for closed channels (Im p>0)
        p = np.sqrt(Pi0 + 0j) if np.real(Pi0) > 0 else 1j*np.sqrt(-Pi0 + 0j)
        dp = dPi/(2*p); d2p = (d2Pi - 2*dp*dp)/(2*p)
        ys.append(1j*p - dp/(2*p) + 1j/(2*p)*(0.75*(dp/p)**2 - d2p/(2*p)))
    return Phi @ np.diag(ys) @ Phi.T

def siegert2d(Vr_s, U2_s, Ns, Om0, E0, rm, tol=1e-10, maxit=40):
    def f(E):
        Y, sig, Ts = shoot2d_Y(Vr_s, U2_s, Ns, Om0, E, rm)
        M = Y - yout2d(E, rm, Vr_s, U2_s, sig, Ts)
        ev = np.linalg.eigvals(M)
        return ev
    prev = None
    def g(E, ref):
        ev = f(E)
        return ev[np.argmin(np.abs(ev - ref))] if ref is not None else ev[np.argmin(np.abs(ev))]
    E1 = E0; E2 = E0 + 1e-6*(1 - 1j)
    f1 = g(E1, None); f2 = g(E2, f1)
    for it in range(maxit):
        E3 = E2 - f2*(E2 - E1)/(f2 - f1)
        E1, f1 = E2, f2; E2 = E3; f2 = g(E2, f1)
        if abs(E2 - E1) < tol*max(1, abs(E2)): return E2, it
    return E2, -1
