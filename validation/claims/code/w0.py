# claims-blind (V3, R16): preparation weight w0 = |<phi_res|psi0>|^2 at the turning point (u = 0, mu = mu_b).
# psi0: real (time-symmetric) central Gaussian of the 4 Higgs zero-mode components, chi = exp(-r^2/(4 s)), with s chosen
#       to minimise the instantaneous energy variance <H^2> - <H>^2 of the 1D effective Hamiltonian.
# H (u = r^{3/2} chi):  -1/2 d^2/dr^2 + 3/(8 r^2) + V(r);  for the Gaussian, H chi = W_s chi with W_s = 1/s - r^2/(8 s^2) + V(r) (exact).
# V(r): 1D Born-Oppenheimer effective potential of the keystone-blind models (base 2D_G-derived V1D, and V2 'all' cross-term V1D)
#       + lam r^4/4 + one-loop finite-volume matching, node-0 couplings from keystone-blind couplings.json (read-only reuse).
# phi_res: (i) box state with a wall 1 unit beyond the barrier top (real, L2-normalised);
#          (ii) CAP eigenvector at the optimal eta, c-normalised (sum v^2 dr = 1), overlap by the c-product;
#          (iii) CAP eigenvector truncated at the barrier top / outer turning point, L2-normalised there.
import os, sys, json, time, pickle
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
KB = ROOT + r"\07_reviews\R16_v12_round3\keystone_blind"
OUTD = ROOT + r"\07_reviews\R16_v12_round3\claims_robustness_blind\results"
sys.path.insert(0, KB + r"\work"); sys.path.insert(0, KB + r"\v2")
import solvers, oneloop, model as MD, production as PR
import xmodel as X   # noqa: F401 (needed to unpickle XGauss)
CPL = json.load(open(KB + r"\results\couplings.json"))
ref0 = CPL["ref_172.6_0.1180"][0]

def Vtotal(rf, Vg, lam, yt, g2l, gYl):
    rc = np.arange(0.01, 21.0, 0.02); v1 = oneloop.V1loop(rc, lam, g2l, gYl, yt)
    V1 = CubicSpline(np.concatenate([[0.0], rc]), np.concatenate([[0.0], v1]))(np.minimum(rf, 20.99))
    return CubicSpline(rf, Vg + lam*rf**4/4 + V1)

def gauss_u(r, s):
    u = r**1.5*np.exp(-r*r/(4*s)); return u

def moments(r, dr, Vr, s):
    u = gauss_u(r, s); n = np.sum(u*u)*dr; w = u*u/n
    W = 1/s - r*r/(8*s*s) + Vr
    E = np.sum(W*w)*dr; E2 = np.sum(W*W*w)*dr
    return E, E2 - E*E

def analyze(Vs, label, dr=0.005):
    out = dict(model=label)
    rtop, vtop = PR.landmarks(Vs); out.update(rtop=float(rtop), vtop=float(vtop))
    # (i) wall state
    vals, vecs, r = solvers.eig1d(Vs, rtop + 1.0, dr, 4.4, k=3); i = int(np.argmin(vals.real))
    Ew = float(vals[i].real); uw = np.real(vecs[:, i]); uw /= np.sqrt(np.sum(uw**2)*dr); uw *= np.sign(uw[np.argmax(np.abs(uw))])
    out["E_wall"] = Ew
    # (ii) CAP (keystone-blind protocol), then eigenvector at the optimal eta
    rout = PR.outer_turn(Vs, rtop, min(Ew, vtop - 1e-3)); Rc = min(rout + 2.5, 13.0); Rmax = Rc + 5.0
    c = solvers.cap1d(Vs, Rc, Rmax, dr, Ew, PR.ETAS)
    vals2, vecs2, r2 = solvers.eig1d(Vs, Rmax, dr, c["E"], k=6, eta=c["eta"], Rc=Rc); j = int(np.argmin(np.abs(vals2 - c["E"])))
    v = vecs2[:, j]; v = v/np.sqrt(np.sum(v*v)*dr); v *= np.sign(np.real(v[np.argmax(np.abs(v))]))
    out.update(E_cap=complex(c["Ecorr"]), Q=float(c["Ecorr"].real), gamma=float(-2*c["Ecorr"].imag), eta=float(c["eta"]), Rc=float(Rc), rout=float(rout))
    rm = min(max(rout + 4.0, 10.0), 17.5)
    try:
        Es, it = solvers.siegert1d(Vs, c["Ecorr"], rm); out.update(Q_sieg=float(Es.real), gamma_sieg=float(-2*Es.imag))
    except Exception as e:
        out["sieg_error"] = str(e)
    # norm diagnostics of the c-normalised CAP state
    out["cap_state_L2norm_total"] = float(np.sum(np.abs(v)**2)*dr); out["cap_state_imag_frac_inner"] = float(np.sqrt(np.sum(np.imag(v[r2 < rtop])**2)/np.sum(np.abs(v[r2 < rtop])**2)))
    out["cap_state_L2_beyond_rtop"] = float(np.sum(np.abs(v[r2 > rtop])**2)*dr)
    # Gaussian family on the CAP grid
    Vr2 = Vs(r2)
    ss = np.exp(np.linspace(np.log(0.03), np.log(30.0), 241))
    mom = np.array([moments(r2, dr, Vr2, s) for s in ss])
    out["scan"] = dict(s=ss[::20].tolist(), E=mom[::20, 0].tolist(), var=mom[::20, 1].tolist())
    iv = int(np.argmin(mom[:, 1]))
    # <H> is unbounded below for lam < 0 (s -> infinity samples the -|lam| r^4 region): take the first interior local minimum (metastable)
    loc = np.where((mom[1:-1, 0] < mom[:-2, 0]) & (mom[1:-1, 0] < mom[2:, 0]))[0] + 1
    ie = int(loc[0]) if len(loc) else iv
    out["E_local_min_found"] = bool(len(loc)); out["E_at_s_max"] = float(mom[-1, 0])
    fV = lambda ls: moments(r2, dr, Vr2, np.exp(ls))[1]; fE = lambda ls: moments(r2, dr, Vr2, np.exp(ls))[0]
    sV = float(np.exp(minimize_scalar(fV, bracket=(np.log(ss[max(iv-2, 0)]), np.log(ss[iv]), np.log(ss[min(iv+2, 240)])), tol=1e-10).x))
    sE = float(np.exp(minimize_scalar(fE, bracket=(np.log(ss[max(ie-2, 0)]), np.log(ss[ie]), np.log(ss[min(ie+2, 240)])), tol=1e-10).x))
    nloc = int(np.sum((mom[1:-1, 1] < mom[:-2, 1]) & (mom[1:-1, 1] < mom[2:, 1])))
    out["n_local_minima_var"] = nloc
    for tag, s in [("varmin", sV), ("Emin", sE)]:
        E, var = moments(r2, dr, Vr2, s); ug = gauss_u(r2, s); ug /= np.sqrt(np.sum(ug*ug)*dr)
        ugw = gauss_u(r, s); ugw /= np.sqrt(np.sum(ugw*ugw)*dr)
        w_wall = float((np.sum(uw*ugw)*dr)**2)
        w_c = float(abs(np.sum(v*ug)*dr)**2)
        trunc = {}
        for tn, rt in [("rtop", rtop), ("rout", rout)]:
            k = r2 < rt; nv = np.sum(np.abs(v[k])**2)*dr; ng = np.sum(ug[k]**2)*dr
            trunc[tn] = float(abs(np.sum(np.conj(v[k])*ug[k])*dr)**2/(nv*ng))
        out[tag] = dict(s=s, E=E, dE=float(np.sqrt(var)), rms_r=float(np.sqrt(4*s)), w0_wall=w_wall, w0_cprod=w_c, w0_trunc=trunc,
                        E_minus_Q=E - out["Q"])
    # weight of the wall state's own energy spread (diagnostic): <H> of the box state equals E_wall
    return out

def main():
    t0 = time.time(); R = dict(agent="claims-blind", task="V3 w0")
    G = PR.GaussCached(ref0["g2"], ref0["gY"], "node00") if os.path.exists(os.path.join(KB, "work", "cache", "gauss_node00.npz")) else MD.Gauss(ref0["g2"], ref0["gY"])
    Pb = G.potentials(G.r)
    XG = pickle.load(open(os.path.join(KB, "v2", "cache", "xg_all_node00.pkl"), "rb")); Px = XG._pot
    R["node0"] = dict(ref0)
    for Mt in [172.6, 173.0, 173.4]:
        own = CPL["grid"][f"0.1180_{Mt:.2f}"][0]
        for label, rf, Vg in [("base_V1D", G.r, Pb["V1D"]), ("v2_all_V1D", XG.r, Px["V1D"])]:
            Vs = Vtotal(rf, Vg, own["lam"], own["yt"], ref0["g2"], ref0["gY"])
            res = analyze(Vs, label); res.update(Mt=Mt, lam=own["lam"], yt=own["yt"])
            R[f"{label}|{Mt}"] = res
            print(label, Mt, {k: res[k] for k in ["Q", "gamma", "E_wall", "rtop", "vtop"]}, "varmin", {k: res["varmin"][k] for k in ["s", "E", "dE", "w0_wall", "w0_cprod"]}, res["varmin"]["w0_trunc"],
                  "Emin", {k: res["Emin"][k] for k in ["s", "E", "w0_wall"]}, "nloc", res["n_local_minima_var"], "%.0fs" % (time.time() - t0), flush=True)
            json.dump(R, open(os.path.join(OUTD, "w0_results.json"), "w"), indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o))
    # grid check at one point
    own = CPL["grid"]["0.1180_173.00"][0]; Vs = Vtotal(XG.r, Px["V1D"], own["lam"], own["yt"], ref0["g2"], ref0["gY"])
    res = analyze(Vs, "v2_all_V1D_dr0.0025", dr=0.0025); R["gridcheck_v2_all_173.0_dr0.0025"] = res
    print("gridcheck", res["varmin"]["w0_wall"], res["varmin"]["w0_cprod"], res["Q"], res["gamma"], flush=True)
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(OUTD, "w0_results.json"), "w"), indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o))
    print("done", flush=True)

if __name__ == "__main__":
    main()
