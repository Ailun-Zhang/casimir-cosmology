# PROVENANCE: copied from D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\keystone_blind\work\production.py on 2026-10-15 by ui-blind (unmodified below this line)
# keystone-blind (T3): production driver. Modes: static | hazard <als> | collect
import os, sys, time, json, glob
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.interpolate import CubicSpline
import model, solvers, oneloop

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
HLB = 0.00437243
ETA_TOT = 1.3110287771461 / HLB      # int_0^inf du/sqrt(cosh 2u) = (sqrt(pi)/4) Gamma(1/4)/Gamma(3/4)
ETAS = [8, 16, 32, 64, 128]

class GaussCached:
    def __init__(self, g2, gY, tag):
        f = os.path.join(CACHE, f"gauss_{tag}.npz")
        if os.path.exists(f):
            d = np.load(f); self.g, self.gY = float(d["g"]), float(d["gY"])
            self.r, self.P, self.E = d["r"], d["P"], d["E"]
            self.sp = [CubicSpline(self.r, self.P[:, j]) for j in range(3)]
        else:
            G = model.Gauss(g2, gY); self.__dict__.update(G.__dict__)
            np.savez(f, g=g2, gY=gY, r=G.r, P=G.P, E=G.E)
    terms = model.Gauss.terms
    potentials = model.Gauss.potentials

def build(G, lam, yt, g2l, gYl, matching=True, dE=None):
    rf = G.r
    if not hasattr(G, "_pot"): G._pot = G.potentials(rf)
    P = G._pot
    if matching:
        rc = np.arange(0.01, 21.0, 0.02)
        v1 = oneloop.V1loop(rc, lam, g2l, gYl, yt)
        V1 = CubicSpline(np.concatenate([[0.0], rc]), np.concatenate([[0.0], v1]))(np.minimum(rf, 20.99))
    else:
        V1 = 0*rf
    q = lam*rf**4/4 + V1
    if dE is not None: q = q + dE(rf)
    return dict(V1D=CubicSpline(rf, P["V1D"] + q), Vlow=CubicSpline(rf, P["Vlow"] + q),
                Vr=CubicSpline(rf, P["U0"] + q), U2=CubicSpline(rf, P["U2"]))

def landmarks(Vs, Eguess=None):
    rg = np.linspace(1.0, 20.0, 9501); v = Vs(rg); i = int(np.argmax(v))
    rtop, vtop = rg[i], v[i]
    return rtop, vtop

def outer_turn(Vs, rtop, E):
    rg = np.linspace(rtop, 20.0, 19001); v = Vs(rg)
    j = np.where(v < E)[0]
    return rg[j[0]] if len(j) else 20.0

def solve_case(G, lam, yt, g2l, gYl, matching=True, do2d=True, dE=None, Ns=24, Om0=0.35, dr2=0.02):
    M = build(G, lam, yt, g2l, gYl, matching, dE)
    out = dict(lam=lam, yt=yt)
    # 1D models: wall estimate then Siegert (ODE + outgoing WKB)
    for name in ["V1D", "Vlow"]:
        Vs = M[name]
        rtop, vtop = landmarks(Vs)
        vals, _, _ = solvers.eig1d(Vs, rtop + 1.0, 0.01, 4.4, k=2)
        Ew = float(np.sort(vals.real)[0])
        flag = "above_barrier" if Ew > vtop else "ok"
        rout = outer_turn(Vs, rtop, min(Ew, vtop - 1e-3))
        Rc = min(rout + 2.5, 13.0); Rmax = Rc + 5.0
        c = solvers.cap1d(Vs, Rc, Rmax, 0.01, Ew, ETAS)
        Ecap = c["Ecorr"]
        rm = min(max(rout + 4.0, 10.0), 17.5)
        Es, it = solvers.siegert1d(Vs, Ecap, rm)
        Es2, it2 = solvers.siegert1d(Vs, Es, rm + 1.5)
        agree = bool(abs(Es2.real - Ecap.real) < 2e-5 and abs(Es2.imag/Ecap.imag - 1) < 0.02)
        if not agree:
            flag += "|siegert_cap_mismatch"
        Eb = Es2 if agree else Ecap
        out[name] = dict(Q=Eb.real, gamma=-2*Eb.imag, Q_cap=Ecap.real, gamma_cap=-2*Ecap.imag, Q_sieg=Es2.real, gamma_sieg=-2*Es2.imag,
                         gamma_rm_diff=float(-2*(Es2.imag - Es.imag)), rtop=rtop, vtop=vtop, rout=rout, rm=rm, it=[it, it2], flag=flag, eta=c["eta"])
    if do2d:
        Vr, U2 = M["Vr"], M["U2"]
        # 2D wall estimate
        rtop = out["V1D"].get("rtop", landmarks(M["Vlow"])[0]) if isinstance(out.get("V1D"), dict) else 5.0
        vals, _, _, _ = solvers.eig2d(Vr, U2, rtop + 1.0, dr2, Ns, Om0, out.get("Vlow", {}).get("Q", 4.3), k=3)
        Ew = float(np.sort(vals.real)[0])
        rout = outer_turn(M["Vlow"], landmarks(M["Vlow"])[0], Ew)
        Rc = min(rout + 2.5, 14.0); Rmax = Rc + 5.0
        c = solvers.cap2d(Vr, U2, Rc, Rmax, dr2, Ns, Om0, Ew, ETAS)
        E = c["Ecorr"]
        out["2D"] = dict(Q=E.real, gamma=-2*E.imag, Q_raw=c["E"].real, gamma_raw=-2*c["E"].imag, eta=c["eta"], Rc=Rc, Rmax=Rmax, Ew=Ew,
                         traj=[[t[0], t[2].real, -2*t[2].imag, t[3]] for t in c["traj"]])
        # bracket check
        try:
            out["2D"]["bracket_ok"] = bool(out["Vlow"]["Q"] <= E.real <= out["V1D"]["Q"] and out["Vlow"]["gamma"] <= -2*E.imag <= out["V1D"]["gamma"])
        except KeyError:
            out["2D"]["bracket_ok"] = None
    return out

def lam_crit(G, yt, g2l, gYl, gtarget, model_key, l0, l1, do2d):
    """secant on ln gamma(lambda) = ln gtarget."""
    f = lambda l: np.log(solve_case(G, l, yt, g2l, gYl, True, do2d)[model_key]["gamma"]) - np.log(gtarget)
    f0, f1 = f(l0), f(l1)
    for it in range(12):
        l2 = l1 - f1*(l1 - l0)/(f1 - f0)
        l0, f0 = l1, f1; l1 = l2; f1 = f(l1)
        if abs(l1 - l0) < 2e-7: break
    return l1

def static():
    g2, gY, yt = 0.5268, 0.4415, 0.4585
    G = GaussCached(g2, gY, "v11")
    res = {}
    t = time.time()
    # (i) lambda=0 local model bound states (no matching)
    M = build(G, 0.0, yt, g2, gY, matching=False)
    loc = {}
    for name in ["V1D", "Vlow"]:
        e = []
        for dr in [0.02, 0.01, 0.005]:
            vals, _, _ = solvers.eig1d(M[name], 16.0, dr, 4.5, k=2); e.append(float(np.sort(vals.real)[0]))
        loc[name] = dict(E_dr=e, E=e[-1], err=abs(e[-1] - e[-2]))
    e2 = []
    for (dr, Ns, Om0) in [(0.02, 24, 0.35), (0.014, 24, 0.35), (0.01, 24, 0.35), (0.01, 32, 0.25)]:
        vals, _, _, _ = solvers.eig2d(M["Vr"], M["U2"], 16.0, dr, Ns, Om0, 4.5, k=2); e2.append(float(np.sort(vals.real)[0]))
    loc["2D"] = dict(E_list=e2, E=e2[2], err=max(abs(e2[2] - e2[1]), abs(e2[3] - e2[2])))
    res["lambda0_local"] = loc
    print("local", loc, time.time() - t, flush=True)
    # (ii) resonances
    for lam in [-0.008, -0.009, -0.010, -0.011, -0.012, -0.0143]:
        r = solve_case(G, lam, yt, g2, gY)
        if lam in (-0.008, -0.010, -0.0143):
            r2 = solve_case(G, lam, yt, g2, gY, dr2=0.014)
            r["2D_dr0.014"] = dict(Q=r2["2D"]["Q"], gamma=r2["2D"]["gamma"])
            r3 = solve_case(G, lam, yt, g2, gY, Ns=32, Om0=0.25)
            r["2D_Ns32"] = dict(Q=r3["2D"]["Q"], gamma=r3["2D"]["gamma"])
        res[f"lam_{lam}"] = r
        print(lam, {k: (v["Q"], v["gamma"]) for k, v in r.items() if isinstance(v, dict) and "Q" in v}, time.time() - t, flush=True)
        json.dump(res, open(os.path.join(RES, "static_raw.json"), "w"), indent=1, default=float)
    # (iii) static lambda_c
    gt = {"P0.9": -np.log(0.9)/ETA_TOT, "P0.5": -np.log(0.5)/ETA_TOT}
    lc = {}
    for key, g in gt.items():
        for mk, d2 in [("2D", True), ("V1D", False), ("Vlow", False)]:
            lc[f"{mk}_{key}"] = float(lam_crit(G, yt, g2, gY, g, mk, -0.009, -0.011, d2))
            print("lambda_c", mk, key, lc[f"{mk}_{key}"], time.time() - t, flush=True)
    res["lambda_c"] = lc; res["gamma_targets"] = gt; res["eta_tot"] = ETA_TOT
    json.dump(res, open(os.path.join(RES, "static_raw.json"), "w"), indent=1, default=float)

# ---------------- hazard ----------------
def _hazard_task(args):
    als, Mt, inode = args
    f = os.path.join(RES, "hazard_shards", f"h_{als:.4f}_{Mt:.2f}_{inode:02d}.json")
    if os.path.exists(f): return f
    C = json.load(open(os.path.join(RES, "couplings.json")))
    ref = C["ref_172.6_0.1180"][inode]
    own = C["grid"][f"{als:.4f}_{Mt:.2f}"][inode]
    G = GaussCached(ref["g2"], ref["gY"], f"node{inode:02d}")
    r = solve_case(G, own["lam"], own["yt"], ref["g2"], ref["gY"])
    r.update(als=als, Mt=Mt, inode=inode, u=own["u"], mu=own["mu"], g2_ref=ref["g2"], gY_ref=ref["gY"])
    json.dump(r, open(f, "w"), indent=1, default=float)
    return f

def hazard(als_list, Mts):
    os.makedirs(os.path.join(RES, "hazard_shards"), exist_ok=True)
    C = json.load(open(os.path.join(RES, "couplings.json")))
    nodes = list(range(len(C["ref_172.6_0.1180"])))
    for inode in nodes:   # pre-build gauss caches sequentially
        ref = C["ref_172.6_0.1180"][inode]; GaussCached(ref["g2"], ref["gY"], f"node{inode:02d}")
    tasks = [(a, m, i) for a in als_list for m in Mts for i in nodes]
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NWORK", "3"))) as pool:
        for k, f in enumerate(pool.imap_unordered(_hazard_task, tasks)):
            if k % 10 == 0: print(k, len(tasks), f, flush=True)

def load_dE():
    from scipy.interpolate import PchipInterpolator
    d = json.load(open(os.path.join(RES, os.environ.get("DE_FILE", "dE_fit.json"))))
    rk, dk = np.array(d["r"]), np.array(d["dE"])*float(os.environ.get("DE_SCALE", "1.0")); p = d.get("tail_power", 4.0)
    f = PchipInterpolator(rk, dk)
    return lambda r: np.where(r <= rk[-1], f(np.minimum(r, rk[-1])), dk[-1]*(rk[-1]/np.maximum(r, 1e-9))**p)

def _hazard_task_ex(args):
    als, Mt, inode = args
    tg = os.environ.get("DE_TAG", "")
    f = os.path.join(RES, "hazard_shards_ex" + (("_" + tg) if tg else ""), f"h_{als:.4f}_{Mt:.2f}_{inode:02d}.json")
    if os.path.exists(f): return f
    C = json.load(open(os.path.join(RES, "couplings.json")))
    ref = C["ref_172.6_0.1180"][inode]; own = C["grid"][f"{als:.4f}_{Mt:.2f}"][inode]
    G = GaussCached(ref["g2"], ref["gY"], f"node{inode:02d}")
    r = solve_case(G, own["lam"], own["yt"], ref["g2"], ref["gY"], dE=load_dE())
    r.update(als=als, Mt=Mt, inode=inode, u=own["u"], mu=own["mu"])
    json.dump(r, open(f, "w"), indent=1, default=float)
    return f

def static_ex():
    g2, gY, yt = 0.5268, 0.4415, 0.4585
    G = GaussCached(g2, gY, "v11"); dE = load_dE(); res = {}
    M = build(G, 0.0, yt, g2, gY, matching=False, dE=dE)
    vals, _, _, _ = solvers.eig2d(M["Vr"], M["U2"], 16.0, 0.01, 24, 0.35, 4.45, k=2)
    res["lambda0_local_2Dex"] = float(np.sort(vals.real)[0])
    vals, _, _ = solvers.eig1d(M["V1D"], 16.0, 0.005, 4.5, k=2); res["lambda0_local_1Dex"] = float(np.sort(vals.real)[0])
    for lam in [-0.008, -0.010, -0.0143]:
        res[f"lam_{lam}"] = solve_case(G, lam, yt, g2, gY, dE=dE)
        print(lam, {k: (v["Q"], v["gamma"]) for k, v in res[f"lam_{lam}"].items() if isinstance(v, dict) and "Q" in v}, flush=True)
    gt = {"P0.9": -np.log(0.9)/ETA_TOT, "P0.5": -np.log(0.5)/ETA_TOT}
    def lc(g, mk, d2):
        f = lambda l: np.log(solve_case(G, l, yt, g2, gY, True, d2, dE=dE)[mk]["gamma"]) - np.log(g)
        l0, l1 = -0.009, -0.011; f0, f1 = f(l0), f(l1)
        for it in range(12):
            l2 = l1 - f1*(l1 - l0)/(f1 - f0); l0, f0 = l1, f1; l1 = l2; f1 = f(l1)
            if abs(l1 - l0) < 2e-7: break
        return float(l1)
    res["lambda_c"] = {f"2Dex_{k}": lc(g, "2D", True) for k, g in gt.items()}
    print(res["lambda0_local_2Dex"], res["lambda_c"], flush=True)
    sc = os.environ.get("DE_SCALE"); tg = os.environ.get("DE_TAG", "")
    name = "static_ex_raw" + (("_" + tg) if tg else "") + (f"_scale{sc}" if sc else "") + ".json"
    json.dump(res, open(os.path.join(RES, name), "w"), indent=1, default=float)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1]
    if mode == "static":
        static()
    elif mode == "static_ex":
        static_ex()
    elif mode == "hazard_ex":
        tg = os.environ.get("DE_TAG", "")
        os.makedirs(os.path.join(RES, "hazard_shards_ex" + (("_" + tg) if tg else "")), exist_ok=True)
        als = [float(a) for a in sys.argv[2].split(",")]
        Mts = [round(x, 2) for x in np.arange(float(sys.argv[3]), float(sys.argv[4]) + 1e-9, float(sys.argv[5]))]
        tasks = [(a, m, i) for a in als for m in Mts for i in range(13)]
        from multiprocessing import Pool
        with Pool(int(os.environ.get("NWORK", "3"))) as pool:
            for k, f in enumerate(pool.imap_unordered(_hazard_task_ex, tasks)):
                if k % 10 == 0: print(k, len(tasks), f, flush=True)
    elif mode == "hazard":
        als = [float(a) for a in sys.argv[2].split(",")]
        Mts = [round(x, 2) for x in np.arange(float(sys.argv[3]), float(sys.argv[4]) + 1e-9, float(sys.argv[5]))]
        hazard(als, Mts)
