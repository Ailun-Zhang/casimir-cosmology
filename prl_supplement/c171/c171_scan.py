"""C171 driver: 1D BO models + WKB for all Lambda, 2D exact resonance (h-ladder, Richardson), convergence study,
direct e != 1 scaling check.  Resumable: every task writes raw/<name>.json and is skipped if present.
Usage: python -B c171_scan.py [n_workers<=4]"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import json
import math
import time
import warnings
import numpy as np
from multiprocessing import Pool

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c171_core as C

# [supplement] path block adapted for standalone use; physics unchanged
RES = os.environ.get("C171_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rerun", "C171"))
RAW = os.path.join(RES, "raw")
os.makedirs(RAW, exist_ok=True)

X_MAX = 1.0 / math.sqrt(0.03)
E_SM = 2.0 * 1.31 / 3.0                      # e with sigma0 = 3e/2 = 1.31
LAM_SM = {"SM_lam-0.010": -0.010 / E_SM ** 2, "SM_lam-0.013": -0.013 / E_SM ** 2}
BASE = dict(theta=0.45, delta=0.6, S=10.0, ds=0.14, dR=1.2, D=40.0, extra_ext=0.0)
HS_MAIN = [0.02, 0.01, 0.005]
HS_CONV = [0.01, 0.005]


def lam_list():
    out = {}
    for j in range(0, 24):
        x = X_MAX ** (j / 16.0)
        out[f"j{j:02d}"] = -1.0 / x ** 2
    out.update(LAM_SM)
    return out


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        raise TypeError(type(o))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)
    os.replace(tmp, path)


def design_grid(Lam, Eg, variant, sigma0=C.SIGMA0, e=1.0, scale=1.0):
    """Physical grid design independent of h: R0 and rmax are multiples of 0.04*scale."""
    kw = dict(BASE)
    kw.update(variant)
    q = 0.04 * scale
    lm = C.bo_landmarks(Lam, Eg, sigma0)
    r3 = lm["r3"] if not lm["above"] else lm["rb"] + 0.5 * scale
    R0 = q * round((r3 + kw["dR"] * scale) / q)
    delta = kw["delta"] * scale
    rend = C.exterior_end(Lam, Eg, R0, kw["theta"], delta, kw["D"], sigma0) + (0.3 + kw["extra_ext"]) * scale
    rmax = q * math.ceil(rend / q)
    rabs = abs(R0 + delta / 2 + np.exp(1j * kw["theta"]) * (rmax - R0 - delta / 2))
    S = kw["S"] * scale
    ds = kw["ds"] * scale if kw.get("ds") else min(0.16, 0.5 / math.sqrt(rabs))
    Ns = int(round(S / ds)) - 1
    return dict(Lam=Lam, rmax=rmax, R0=R0, theta=kw["theta"], delta=delta, S=S, Ns=Ns, e=e), lm


def richardson(hs, vals):
    v = np.asarray(vals, float)
    out = {}
    if len(v) >= 2:
        r1 = (4 * v[-1] - v[-2]) / 3.0          # assumes halving
        out["R1"] = r1
        out["R1_err_est"] = abs(r1 - v[-1])
    if len(v) >= 3:
        r1a = (4 * v[-2] - v[-3]) / 3.0
        r2 = (16 * r1 - r1a) / 15.0
        out["R2"] = r2
        out["ratio"] = (v[-3] - v[-2]) / (v[-2] - v[-1]) if v[-2] != v[-1] else float("nan")
        out["R1_vs_R2"] = abs(r1 - r2)
    return out


def task_2d(task):
    name = task["name"]
    path = os.path.join(RAW, name + ".json")
    if os.path.exists(path):
        return name, "skip"
    t0 = time.time()
    Lam, Eg, variant, hs = task["Lam"], task["Eg"], task.get("variant", {}), task["hs"]
    e = task.get("e", 1.0)
    sigma0 = 1.5 * e
    scale = e ** (-1.0 / 3.0)
    g0, lm = design_grid(Lam, Eg, variant, sigma0=sigma0, e=e, scale=scale)
    runs = []
    Eshift = Eg
    for h in hs:
        g = dict(g0)
        g["h"] = h * scale
        r = C.solve_2d(g, Eshift, k=8, r_well=lm["rb"], r3=(lm["r3"] if not lm["above"] else None),
                       hol_e=(1.0, E_SM, 0.5), frac_min=task.get("frac_min", 0.5))
        r["h_scaled"] = h
        runs.append(r)
        Eshift = r["E_re"]
    G = [r["Gamma_flux"] if r.get("Gamma_flux") else r["Gamma_rq"] for r in runs]
    out = dict(name=name, Lam=Lam, e=e, variant=variant, hs=hs, grid=g0, landmarks=lm, runs=runs,
               rich_E=richardson(hs, [r["E_re"] for r in runs]), rich_G=richardson(hs, G),
               rich_lnG=richardson(hs, [math.log(x) for x in G]), wall=time.time() - t0)
    jdump(out, path)
    return name, f"done {time.time()-t0:.1f}s G={G[-1]:.6e}"


def one_d_all(lams):
    path = os.path.join(RAW, "oneD_all.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    out = {}
    prev = {}
    for name, Lam in sorted(lams.items(), key=lambda kv: -kv[1]):     # small |Lam| first
        rec = dict(Lam=Lam)
        for c2, lab, c2L in ((-0.125, "BO", 0.0), (1.0 / 16.0, "BOD", 3.0 / 16.0)):
            w = C.wkb_bo(Lam, c2L)
            rec["wkb_" + lab] = w
            if math.isnan(w["E_BS"]):
                E0 = prev.get(lab, complex(2.0, -0.1))
            else:
                E0 = complex(w["E_BS"], -0.5 * w["Gamma_Kemble"])
            r = C.resonance_1d(Lam, c2, complex(E0))
            r2 = C.resonance_1d(Lam, c2, complex(r["E_re"], r["E_im"]), phi=0.35, dR=1.6, rtol=1e-13)
            r["Gamma_alt"] = r2["Gamma"]
            r["E_alt"] = r2["E_re"]
            r["rel_diff_alt"] = abs(r2["Gamma"] / r["Gamma"] - 1.0)
            rec["res_" + lab] = r
            prev[lab] = complex(r["E_re"], r["E_im"])
            rec["S_at_E_" + lab] = C.barrier_action(Lam, r["E_re"], c2L)
        out[name] = rec
        print(f"1D {name} Lam={Lam:.5f} BO E={rec['res_BO']['E_re']:.6f} G={rec['res_BO']['Gamma']:.4e} | "
              f"BOD E={rec['res_BOD']['E_re']:.6f} G={rec['res_BOD']['Gamma']:.4e}", flush=True)
    jdump(out, path)
    return out


def build_tasks(lams, oned):
    tasks = []
    for name, Lam in lams.items():
        Eg = oned[name]["res_BO"]["E_re"] + 0.1
        tasks.append(dict(name=f"main_{name}", Lam=Lam, Eg=Eg, hs=HS_MAIN))
    conv_set = ["j00", "j08", "j12", "j16", "j21", "j23"]
    variants = {"th035": dict(theta=0.35), "th055": dict(theta=0.55), "dR20": dict(dR=2.0), "del10": dict(delta=1.0),
                "D55": dict(D=55.0), "ds010": dict(ds=0.10), "S12": dict(S=12.0), "base": {}}
    for jn in conv_set:
        Lam = lams[jn]
        Eg = oned[jn]["res_BO"]["E_re"] + 0.1
        for vn, v in variants.items():
            # j00 is above the barrier: with R0 moved out (dR20) the broad ground resonance has well fraction < 0.5
            tasks.append(dict(name=f"conv_{jn}_{vn}", Lam=Lam, Eg=Eg, variant=v, hs=HS_CONV,
                              frac_min=0.3 if jn == "j00" else 0.5))
    for jn in ["j08", "j16"]:
        tasks.append(dict(name=f"fine_{jn}", Lam=lams[jn], Eg=oned[jn]["res_BO"]["E_re"] + 0.1,
                          hs=[0.01, 0.005, 0.0025]))
    # direct e != 1 checks: Problem2D takes Lam = physical lambda and e
    for lab, lam_phys in (("lam-0.010", -0.010), ("lam-0.013", -0.013)):
        nm = "SM_" + lab
        Eg = (oned[nm]["res_BO"]["E_re"] + 0.1) * E_SM ** (2.0 / 3.0)
        tasks.append(dict(name=f"direct_e{E_SM:.4f}_{lab}", Lam=lam_phys, Eg=Eg, e=E_SM, hs=HS_MAIN))
    tasks.append(dict(name="direct_e0.5_j12", Lam=lams["j12"] * 0.25,
                      Eg=(oned["j12"]["res_BO"]["E_re"] + 0.1) * 0.5 ** (2.0 / 3.0), e=0.5, hs=HS_MAIN))
    return tasks


def main():
    nw = min(int(sys.argv[1]) if len(sys.argv) > 1 else 4, 4)
    lams = lam_list()
    oned = one_d_all(lams)
    tasks = build_tasks(lams, oned)
    todo = [t for t in tasks if not os.path.exists(os.path.join(RAW, t["name"] + ".json"))]
    print(f"{len(tasks)} tasks, {len(todo)} to do, workers={nw}", flush=True)
    t0 = time.time()
    with Pool(nw) as pool:
        for name, msg in pool.imap_unordered(task_2d, todo):
            print(f"[{time.time()-t0:7.1f}s] {name}: {msg}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
