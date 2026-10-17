# keystone-blind V2 driver. modes: stability | static | build <cfg> | hazard <cfg> | collect
import os, sys, json, time, glob, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "..", "work"); sys.path.insert(0, W); sys.path.insert(0, HERE)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.sparse.linalg as spla
import xmodel as X, solvers, oneloop, model
from production import landmarks, outer_turn, ETAS, ETA_TOT
RES3 = os.path.join(HERE, "..", "results"); CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
CFG = {"none": [], "su3t": ["su3t"], "su2": ["su2"], "uy": ["uy"], "all": ["su3t", "su2", "uy"]}
V11 = dict(g2=0.5268, gY=0.4415, yt=0.4585, g3=0.5396)

def solve_x(XG, lam, yt, g2l, gYl, Ns=24, Om0=0.35, dr2=0.02):
    rf = XG.r
    if not hasattr(XG, "_pot"):
        XG._pot = XG.potentials(rf); XG._vlow = XG.vlow(rf)
    P = XG._pot
    rc = np.arange(0.01, 21.0, 0.02); v1 = oneloop.V1loop(rc, lam, g2l, gYl, yt)
    V1 = CubicSpline(np.concatenate([[0.0], rc]), np.concatenate([[0.0], v1]))(np.minimum(rf, 20.99))
    q = lam*rf**4/4 + V1
    M = dict(V1D=CubicSpline(rf, P["V1D"] + q), Vlow=CubicSpline(rf, XG._vlow + q), Vr=CubicSpline(rf, P["U0"] + q), U2=CubicSpline(rf, P["U2"]))
    out = dict(lam=lam, yt=yt)
    for name in ["V1D", "Vlow"]:
        Vs = M[name]; rtop, vtop = landmarks(Vs)
        vals, _, _ = solvers.eig1d(Vs, rtop + 1.0, 0.01, 4.4, k=2); Ew = float(np.sort(vals.real)[0])
        rout = outer_turn(Vs, rtop, min(Ew, vtop - 1e-3)); Rc = min(rout + 2.5, 13.0)
        c = solvers.cap1d(Vs, Rc, Rc + 5.0, 0.01, Ew, ETAS); Ecap = c["Ecorr"]
        rm = min(max(rout + 4.0, 10.0), 17.5)
        Es, it = solvers.siegert1d(Vs, Ecap, rm); Es2, it2 = solvers.siegert1d(Vs, Es, rm + 1.5)
        ok = abs(Es2.real - Ecap.real) < 2e-4 and Es2.imag < 0
        Eb = Es2 if ok else Ecap
        out[name] = dict(Q=Eb.real, gamma=-2*Eb.imag, Q_cap=Ecap.real, gamma_cap=-2*Ecap.imag, Q_sieg=Es2.real, gamma_sieg=-2*Es2.imag, rtop=rtop, vtop=vtop, flag="ok" if ok else "cap")
    VY = X.VYfun(XG)
    rtop = out["V1D"]["rtop"]
    H, r, sig = X.build2d_x(M["Vr"], M["U2"], VY, rtop + 1.0, dr2, Ns, Om0)
    vals = spla.eigs(H, k=3, sigma=out["Vlow"]["Q"], return_eigenvectors=False); Ew = float(np.sort(vals.real)[0])
    rout = outer_turn(M["Vlow"], landmarks(M["Vlow"])[0], Ew); Rc = min(rout + 2.5, 14.0)
    c = X.cap2d_x(M["Vr"], M["U2"], VY, Rc, Rc + 5.0, dr2, Ns, Om0, Ew, ETAS); E = c["Ecorr"]
    out["2D"] = dict(Q=E.real, gamma=-2*E.imag, eta=c["eta"], Rc=Rc, Ew=Ew,
                     bracket_ok=bool(out["Vlow"]["Q"] <= E.real <= out["V1D"]["Q"] and out["Vlow"]["gamma"] <= -2*E.imag <= out["V1D"]["gamma"]))
    return out

def stability():
    g2, gY, yt, lam = V11["g2"], V11["gY"], V11["yt"], -0.010
    G = model.Gauss(g2, gY); res = {}
    for which in ["higgs", "all"]:
        rows = []
        for r in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
            t = G.terms(np.array([r])); m = float(t["m"][0]); U2 = float(t["U2n"][0])
            K = X.cartan_kappa(r, g2, gY, yt, lam, which)
            km = K[0, 0]*g2*g2*m*m + 2*K[0, 1]*g2*gY*m + K[1, 1]*gY*gY
            rows.append(dict(r=r, m=m, U2_tree_gauss=U2, half_kappa_mean=0.5*km, total=U2 + 0.5*km, kappa=K.tolist()))
            print(which, "r=%.2f m=%.4f U2=%.5f 0.5*kappa_mean=%.5f total=%.5f" % (r, m, U2, 0.5*km, U2 + 0.5*km), flush=True)
        res[which] = rows
        # exact s-profile of V_Y at small r
        prof = {}
        for r in [0.3, 0.5, 1.0]:
            t = G.terms(np.array([r])); m = float(t["m"][0]); s = np.linspace(0, 14, 29); V = np.zeros_like(s)
            for f in X.fields(r, g2, gY, yt, lam, which):
                q = abs(g2*f[5]*m + gY*f[6])
                if q > 0: V += f[2]*f[1]*X.dfield(f, q*s/(2*np.pi))
            prof[str(r)] = dict(s=s.tolist(), VY=V.tolist(), VY_plus_U2s2=(V + float(t["U2n"][0])*s*s).tolist())
            print(which, "r=%.1f V_Y(s=2,4,8,12) =" % r, np.round(np.interp([2, 4, 8, 12], s, V), 4), " +U2 s^2:", np.round(np.interp([2, 4, 8, 12], s, V + float(t["U2n"][0])*s*s), 4), flush=True)
        res[which + "_profiles"] = prof
    json.dump(res, open(os.path.join(HERE, "stability.json"), "w"), indent=1)

def static():
    out = {}; t0 = time.time()
    for lam in [-0.008, -0.010, -0.0143]:
        for cfg in ["none", "su3t", "su2", "uy", "all"]:
            XG = X.XGauss(V11["g2"], V11["gY"], V11["yt"], lam, V11["g3"], CFG[cfg])
            r = solve_x(XG, lam, V11["yt"], V11["g2"], V11["gY"])
            out[f"{cfg}_{lam}"] = r
            print(cfg, lam, {k: (round(v["Q"], 6), "%.5e" % v["gamma"]) for k, v in r.items() if isinstance(v, dict)}, r["2D"]["bracket_ok"], "%.0fs" % (time.time() - t0), flush=True)
            json.dump(out, open(os.path.join(HERE, "static_v2.json"), "w"), indent=1, default=float)
    # local lambda=0 ground states (no matching) for none / all
    for cfg in ["none", "all"]:
        XG = X.XGauss(V11["g2"], V11["gY"], V11["yt"], 0.0, V11["g3"], CFG[cfg]); rf = XG.r; P = XG.potentials(rf)
        Vr = CubicSpline(rf, P["U0"]); U2 = CubicSpline(rf, P["U2"])
        H, r, sig = X.build2d_x(Vr, U2, X.VYfun(XG), 16.0, 0.02, 24, 0.35)
        vals = spla.eigs(H, k=3, sigma=4.5, return_eigenvectors=False); out[f"local0_{cfg}"] = float(np.sort(vals.real)[0])
        print("local lambda=0", cfg, out[f"local0_{cfg}"], flush=True)
    json.dump(out, open(os.path.join(HERE, "static_v2.json"), "w"), indent=1, default=float)

# ---------------- hazard ----------------
def _xg_path(cfg, inode): return os.path.join(CACHE, f"xg_{cfg}_node{inode:02d}.pkl")
def _build(args):
    cfg, inode = args; f = _xg_path(cfg, inode)
    if os.path.exists(f): return f
    C = json.load(open(os.path.join(RES3, "couplings.json"))); ref = C["ref_172.6_0.1180"][inode]
    XG = X.XGauss(ref["g2"], ref["gY"], ref["yt"], ref["lam"], ref["g3"], CFG[cfg])
    XG._pot = XG.potentials(XG.r); XG._vlow = XG.vlow(XG.r)
    pickle.dump(XG, open(f, "wb")); return f

def _task(args):
    cfg, Mt, inode = args
    d = os.path.join(HERE, f"hazard_shards_{cfg}"); f = os.path.join(d, f"h_0.1180_{Mt:.2f}_{inode:02d}.json")
    if os.path.exists(f): return f
    C = json.load(open(os.path.join(RES3, "couplings.json"))); ref = C["ref_172.6_0.1180"][inode]; own = C["grid"][f"0.1180_{Mt:.2f}"][inode]
    XG = pickle.load(open(_xg_path(cfg, inode), "rb"))
    r = solve_x(XG, own["lam"], own["yt"], ref["g2"], ref["gY"])
    r.update(als=0.1180, Mt=Mt, inode=inode, u=own["u"], mu=own["mu"])
    json.dump(r, open(f, "w"), indent=1, default=float); return f

def hazard(cfgs, Mts):
    from multiprocessing import Pool
    nw = int(os.environ.get("NWORK", "4"))
    for c in cfgs: os.makedirs(os.path.join(HERE, f"hazard_shards_{c}"), exist_ok=True)
    with Pool(nw) as pool:
        for f in pool.imap_unordered(_build, [(c, i) for c in cfgs for i in range(13)]): pass
        print("builds done", flush=True)
        tasks = [(c, m, i) for c in cfgs for m in Mts for i in range(13)]
        for k, f in enumerate(pool.imap_unordered(_task, tasks)):
            if k % 20 == 0: print(k, len(tasks), os.path.basename(f), flush=True)

def collect():
    import collect as C3
    base = json.load(open(os.path.join(RES3, "hazard_summary.json")))
    out = dict(base_2D_G=base["critical"]["2D_0.1180"], cfgs={})
    for d in sorted(glob.glob(os.path.join(HERE, "hazard_shards_*"))):
        cfg = d.split("hazard_shards_")[1]; D = {}
        for f in glob.glob(os.path.join(d, "h_*.json")):
            x = json.load(open(f)); D.setdefault(round(x["Mt"], 2), {})[x["inode"]] = x
        rows = []
        for Mt in sorted(D):
            nd = D[Mt]
            if len(nd) < 13: continue
            us = np.array([nd[i]["u"] for i in sorted(nd)])
            for mdl in ["2D", "V1D", "Vlow"]:
                g, how = C3.gammas(nd, mdl, 1.065)
                rows.append(dict(model=mdl, Mt=Mt, I=C3.hazard_integral(us, g), I_lin=C3.hazard_integral(us, g, "linear"), nfb=int(sum(h == "fallback" for h in how))))
        res = {}
        for mdl in ["2D", "V1D", "Vlow"]:
            rr = sorted([x for x in rows if x["model"] == mdl], key=lambda x: x["Mt"])
            while len(rr) > 4 and rr[0]["I"] >= rr[1]["I"]: rr = rr[1:]
            if len(rr) < 4: continue
            M = np.array([x["Mt"] for x in rr]); sp_ = CubicSpline(M, np.log([x["I"] for x in rr])); spl = CubicSpline(M, np.log([x["I_lin"] for x in rr]))
            fine = np.linspace(M[0], M[-1], 200001); r = {}
            for P in [0.9, 0.5, 0.1]:
                tg = np.log(-np.log(P))
                def root(s):
                    v = s(fine) - tg; j = np.where(np.diff(np.sign(v)) != 0)[0]; return float(fine[j[0]]) if len(j) else None
                r[f"P{P}"] = dict(mc=root(sp_), d_quad=(root(spl) - root(sp_)) if (root(spl) and root(sp_)) else None)
            for Mq in [172.2, 172.6, 173.0]:
                r[f"P1_{Mq}"] = float(np.exp(-np.exp(sp_(Mq)))) if M[0] <= Mq <= M[-1] else None
            res[mdl] = r
        out["cfgs"][cfg] = res
        b = out["base_2D_G"]
        if "2D" in res:
            print(cfg, {P: (round(res["2D"][P]["mc"], 4), round(res["2D"][P]["mc"] - b[P]["mc"], 4)) for P in ["P0.9", "P0.5", "P0.1"] if res["2D"][P]["mc"]},
                  {k: round(v, 4) for k, v in res["2D"].items() if k.startswith("P1_") and v})
    json.dump(out, open(os.path.join(HERE, "hazard_v2_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1]
    if mode == "stability": stability()
    elif mode == "static": static()
    elif mode == "hazard":
        cfgs = sys.argv[2].split(",")
        Mts = [round(x, 2) for x in np.arange(float(sys.argv[3]), float(sys.argv[4]) + 1e-9, float(sys.argv[5]))]
        hazard(cfgs, Mts)
    elif mode == "collect": collect()
