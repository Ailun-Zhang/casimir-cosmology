# part2_lamc.py -- ui-blind (V5) part 2: static lambda_c (P1 = exp(-gamma eta_inf) = 0.5) of the 2D_G
# (and 1D BO) zero-mode model of keystone-blind (round-2 model: no cross term, two-loop map), as a function
# of the turning duration eta_inf = 1.31103/(hL_b) (hL_b x0.5, x1, x2) and of g2, y_t (+-10%).
# The keystone-blind modules are used unmodified from work/kb_copy (PROVENANCE headers there).
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kb_copy"))
import numpy as np
import production as P

OUT = os.path.join(os.path.dirname(HERE), "results", "part2")
os.makedirs(OUT, exist_ok=True)
G2, GY, YT = 0.5268, 0.4415, 0.4585          # v11 couplings at 1/L_b, as in keystone-blind static runs
ETA0 = P.ETA_TOT                              # 1.3110287771461/0.00437243 = 299.837

CASES = {
    "base": dict(eta=ETA0, g2=G2, yt=YT),
    "hLb_x0.5": dict(eta=2 * ETA0, g2=G2, yt=YT),
    "hLb_x2": dict(eta=0.5 * ETA0, g2=G2, yt=YT),
    "g2_x1.1": dict(eta=ETA0, g2=1.1 * G2, yt=YT),
    "g2_x0.9": dict(eta=ETA0, g2=0.9 * G2, yt=YT),
    "yt_x1.1": dict(eta=ETA0, g2=G2, yt=1.1 * YT),
    "yt_x0.9": dict(eta=ETA0, g2=G2, yt=0.9 * YT),
}


def gauss_for(g2):
    tag = "v11" if abs(g2 - G2) < 1e-12 else f"g2_{g2:.6f}"
    return P.GaussCached(g2, GY, tag)


def secant_lamc(G, c, key, do2d, l0, l1, log):
    gt = np.log(2.0) / c["eta"]

    def f(l):
        r = P.solve_case(G, l, c["yt"], c["g2"], GY, True, do2d)
        g = r[key]["gamma"]
        log.append(dict(lam=l, gamma=g, Q=r[key]["Q"], flag=r[key].get("flag", ""),
                        bracket_ok=r.get("2D", {}).get("bracket_ok") if do2d else None))
        return np.log(g) - np.log(gt)
    f0, f1 = f(l0), f(l1)
    for it in range(14):
        l2 = l1 - f1 * (l1 - l0) / (f1 - f0)
        l0, f0 = l1, f1; l1 = l2; f1 = f(l1)
        if abs(l1 - l0) < 2e-7:
            break
    # local slope d ln gamma / d lambda from the last two iterates
    slope = (f1 - f0) / (l1 - l0)
    return float(l1), float(slope), float(gt)


def run_case(name):
    fn = os.path.join(OUT, f"{name}.json")
    if os.path.exists(fn):
        return name, "cached"
    c = CASES[name]
    t0 = time.time()
    G = gauss_for(c["g2"])
    guess = {"base": (-0.0120, -0.0130), "hLb_x0.5": (-0.0108, -0.0118), "hLb_x2": (-0.0135, -0.0145)}
    l0, l1 = guess.get(name, (-0.0115, -0.0135))
    res = dict(case=c, name=name, model="keystone-blind 2D_G / 1D BO, round-2 (no cross term, two-loop map)")
    log2 = []
    lc2, s2, gt = secant_lamc(G, c, "2D", True, l0, l1, log2)
    res["2D"] = dict(lambda_c=lc2, dlngamma_dlam=s2, gamma_target=gt, iterates=log2)
    log1 = []
    lc1, s1, _ = secant_lamc(G, c, "V1D", False, l0 + 0.0008, l1 + 0.0008, log1)
    res["1D_BO"] = dict(lambda_c=lc1, dlngamma_dlam=s1, iterates=log1)
    res["seconds"] = time.time() - t0
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    return name, f"ok {time.time()-t0:.0f}s lc2D={lc2:.6f} lc1D={lc1:.6f}"


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CASES)
    # build the extra Gaussian caches first (sequentially, they are shared)
    for n in names:
        gauss_for(CASES[n]["g2"])
    print(time.strftime("%H:%M:%S"), "gauss caches ready", flush=True)
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NWORK", "3"))) as pool:
        for name, st in pool.imap_unordered(run_case, names):
            print(time.strftime("%H:%M:%S"), name, st, flush=True)
