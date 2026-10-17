# run_part1.py -- ui-blind (V5): checks, MMR 2021 reproduction, v11 xi_inf(M_t). Shards -> results/part1/
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
import smhiggs as S
import inflation_bound as IB

OUT = os.path.join(os.path.dirname(HERE), "results", "part1")
os.makedirs(OUT, exist_ok=True)
MP = IB.MP_GEV
MU_B = MP / 722.0
MPL_NONRED = 1.2209e19

MMR_TAB1 = dict(g2=0.648382, gY=0.358729, yt=0.934843, lam=0.126249)   # MMR 2021 Table 1, mu_EW = m_t


def mmr_couplings(mt=172.76, alphas=0.1179):
    """MMR Table 1 at m_t=172.76; other m_t: Buttazzo (55),(57)-(59) slopes applied to the Table-1 values."""
    c = dict(MMR_TAB1)
    dm = mt - 172.76
    c["yt"] += 0.00556 * dm; c["lam"] += -0.00004 * dm; c["g2"] += 0.00004 * dm; c["gY"] += 0.00011 * dm
    c["g3"] = 1.1666 + 0.00314 * (alphas - 0.1184) / 0.0007 - 0.00046 * (mt - 173.34)
    return c


def solve_xi(lnN_of_xi, lo=0.0, hi=0.2, tol=2e-6):
    flo, fhi = lnN_of_xi(lo), lnN_of_xi(hi)
    if not (flo > 0 > fhi):
        raise RuntimeError(f"bracket fail {flo} {fhi}")
    return brentq(lnN_of_xi, lo, hi, xtol=tol)


def task_checks():
    res = {}
    # 1) RGE vs Buttazzo (61a-e) at M_Pl (central inputs of Buttazzo, MW=80.384)
    for nl in (2, 3):
        c0 = S.buttazzo_matching(173.34, 125.15, 0.1184, MW=80.384)
        run = S.SMRun(173.34, c0, nloop=nl)
        s = 2 * np.log(MPL_NONRED)
        G1, G2, G3, Yt, L = run.coup(s)
        res[f"buttazzo_central_MPl_nloop{nl}"] = dict(g1=np.sqrt(G1), g2=np.sqrt(G2), g3=np.sqrt(G3),
                                                       yt=np.sqrt(Yt), lam=L)
    res["buttazzo_61_ref"] = dict(g1=0.6154, g2=0.5055, g3=0.4873, yt=0.3825, lam=-0.0143)
    # 2) couplings at mu_b for (172.6, 125.13, 0.1180, MW 80.3692): compare keystone-blind numbers
    for nl in (2, 3):
        c0 = S.buttazzo_matching(172.6, 125.13, 0.1180)
        run = S.SMRun(172.6, c0, nloop=nl)
        G1, G2, G3, Yt, L = run.coup(2 * np.log(MU_B))
        res[f"mu_b_172.6_nloop{nl}"] = dict(mu_b=MU_B, g2=np.sqrt(G2), gY=np.sqrt(0.6 * G1), g3=np.sqrt(G3),
                                             yt=np.sqrt(Yt), lam=L, K1=run.K(2 * np.log(MU_B), 1),
                                             K3=run.K(2 * np.log(MU_B), 3))
    res["keystone_mu_b_ref"] = dict(lam_2loop=-0.0110527, lam_cal=-0.0101929, yt=0.428359, g2=0.527719,
                                    gY=0.442397)
    # 3) table consistency: sum(n theta^2 + n') at h=0 vs 2*(288 xi^2 - 96 xi - 1751/30)
    for xi in (0.0, 0.05, 0.1667, 0.3):
        M2 = S.masses2(0.0, 1.0, 0.4, 0.2, 0.2, -0.01, xi)
        lhs = float(np.sum(S._n * M2 * M2 + S._np))
        res[f"table_sum_xi{xi}"] = dict(lhs=lhs, rhs=2 * (288 * xi * xi - 96 * xi - 1751 / 30))
    # 4) tree-level limit of B at one R: compare with 96 pi^2 xi(mu*)^2/|lam(mu*)|
    c0 = mmr_couplings()
    run = S.SMRun(172.76, c0, nloop=3)
    ve = S.Veff(run, 0.06 - 1 / 6, xiloop=1)
    R = 12 * (1e13)**2
    t0 = time.time()
    B, hb, mus = ve.B_HM(R)
    dt = time.time() - t0
    s = 2 * np.log(mus)
    xi_s = run.xi(s, ve.c); lam_s = run.coup(s)[4]
    res["tree_check_H1e13_xiEW0.06"] = dict(B=B, h_bar_over_H=hb / 1e13, mu_star=mus, xi_mustar=xi_s,
                                             lam_mustar=lam_s, B_tree=96 * np.pi**2 * xi_s**2 / abs(lam_s),
                                             seconds=dt)
    # 5) backgrounds
    v = IB.V11Background()
    res["v11_background"] = dict(Npost_raw=v.Npost_raw, Lb_eff=v.Lb, constraint_max=v.constraint_max,
                                 H_end_over_h=float(v.H[-1] / 6.116e-6), R_turn_over_h2=float(v.R[0] / 6.116e-6**2),
                                 R_end_over_Hend2=float(v.R[-1] / v.H[-1]**2), frac_end=v.frac_end,
                                 LeHe=float(v.Lb * np.exp(v.lna[-1]) * v.H[-1]))
    for kind in ("starobinsky", "quadratic"):
        b = IB.MMRBackground(kind)
        res[f"mmr_bg_{kind}"] = dict(par=float(b.par), alpha=float(getattr(b, "alpha", np.nan)),
                                     m_GeV=float(getattr(b, "m", np.nan) * MP),
                                     H60=float(np.interp(60, b.N, b.H)), H0=float(b.H[0]),
                                     lnrho_end_H=float(np.log(b.rho[0] * b.H[0])))
    return res


def make_case(case):
    """case: dict(kind='mmr'|'v11', ...)"""
    nloop = case.get("nloop", 3); xiloop = case.get("xiloop", 1); zeta = case.get("zeta", 1.0)
    nodes = case.get("nodes", 36)
    if case["kind"] == "mmr":
        c0 = mmr_couplings(case.get("mt", 172.76), case.get("alphas", 0.1179))
        mu0 = case.get("mt", 172.76)
        bg = IB.MMRBackground(case["model"])
    else:
        if case.get("coup") == "mmr":
            c0 = mmr_couplings(case["Mt"], 0.1179)
        else:
            c0 = S.buttazzo_matching(case["Mt"], case.get("Mh", 125.13), case.get("alphas", 0.1180),
                                     MW=case.get("MW", 80.3692))
        mu0 = case["Mt"]
        bg = IB.V11Background(torus=case.get("torus", True), dlnLe=case.get("dlnLe", 0.0),
                              r_after_over_L=case.get("r_after_over_L", 3.21 / 6.0))
    run = S.SMRun(mu0, c0, nloop=nloop)
    Rmin, Rmax = bg.Rrange
    cache = {}
    Kref = run.K(2 * np.log(case["xi_ref_mu"]), xiloop) if "xi_ref_mu" in case else 1.0
    vopt = dict(noalpha=case.get("noalpha", False), light_fields=case.get("light_fields", True))

    def lnN(xiEW):
        ve = S.Veff(run, (xiEW - 1 / 6) / Kref, xiloop=xiloop, zeta=zeta, **vopt)
        bt = IB.BTable(ve, Rmin, Rmax, nodes=nodes)
        val, info = bg.lnN(bt)
        cache[xiEW] = (val, info, bt)
        return val
    t0 = time.time()
    xi = solve_xi(lnN)
    val, info, bt = cache[xi] if xi in cache else (lnN(xi), *cache[xi][1:])
    sb = 2 * np.log(MU_B)
    c = (xi - 1 / 6) / Kref
    # xi at mu* (barrier) at the peak
    ve = S.Veff(run, c, xiloop=xiloop, zeta=zeta, **vopt)
    Bp, hb, mus = ve.B_HM(info["R_peak"])
    out = dict(case=case, xi_EW=xi, xi_Mt=1 / 6 + c, xi_mu_b=1 / 6 + c * run.K(sb, xiloop), lnN_at_root=val, info=info,
               peak=dict(B=Bp, h_bar=hb, mu_star=mus, xi_mustar=run.xi(2 * np.log(mus), c, xiloop),
                         lam_mustar=run.coup(2 * np.log(mus))[4]),
               Btable=dict(lnR=bt.lR.tolist(), B=bt.B.tolist(), hb=bt.hb.tolist()),
               lam_mu_b=run.coup(sb)[4], K_mu_b=run.K(sb, xiloop), seconds=time.time() - t0)
    if case["kind"] == "v11":
        out["Npost_raw"] = bg.Npost_raw; out["frac_end"] = bg.frac_end
    return out


def run_task(name_case):
    name, case = name_case
    fn = os.path.join(OUT, name + ".json")
    if os.path.exists(fn):
        return name, "cached"
    t0 = time.time()
    try:
        res = task_checks() if name == "checks" else make_case(case)
        status = "ok"
    except Exception as e:
        import traceback
        res = dict(error=repr(e), tb=traceback.format_exc(), case=case)
        status = "error"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    return name, f"{status} {time.time()-t0:.1f}s"


def all_tasks():
    T = [("checks", None),
         ("mmr_star_172.76", dict(kind="mmr", model="starobinsky")),
         ("mmr_quad_172.76", dict(kind="mmr", model="quadratic")),
         ("mmr_star_172.16", dict(kind="mmr", model="starobinsky", mt=172.16)),
         ("mmr_star_173.36", dict(kind="mmr", model="starobinsky", mt=173.36))]
    for Mt in (172.2, 172.6, 173.0):
        T.append((f"v11_{Mt}_base", dict(kind="v11", Mt=Mt)))
        T.append((f"v11_{Mt}_xi3", dict(kind="v11", Mt=Mt, xiloop=3)))
        T.append((f"v11_{Mt}_rge2", dict(kind="v11", Mt=Mt, nloop=2)))
        T.append((f"v11_{Mt}_notorus", dict(kind="v11", Mt=Mt, torus=False)))
        T.append((f"v11_{Mt}_zeta0", dict(kind="v11", Mt=Mt, zeta=0.0)))
    T.append(("v11_172.6_nodes60", dict(kind="v11", Mt=172.6, nodes=60)))
    T.append(("v11_172.6_L0_2dLSS", dict(kind="v11", Mt=172.6, dlnLe=float(np.log(2 / 0.95)),
                                         r_after_over_L=3.21 / 6.0 * 0.95 / 2)))
    # second batch: corrected MMR m_t variation + diagnostics of the reproduction residual
    for mt in (172.16, 173.36, 171.5):
        T.append((f"mmrB_star_{mt}", dict(kind="mmr", model="starobinsky", mt=mt)))
    for tag, opt in (("rge2", dict(nloop=2)), ("zeta0", dict(zeta=0.0)), ("noalpha", dict(noalpha=True)),
                     ("xi3", dict(xiloop=3)), ("nolight", dict(light_fields=False)),
                     ("muEW246", dict(xi_ref_mu=246.22)), ("muEWMZ", dict(xi_ref_mu=91.1876))):
        T.append((f"mmrD_star_{tag}", dict(kind="mmr", model="starobinsky", **opt)))
    T.append(("v11geom_mmrcoup_172.76", dict(kind="v11", Mt=172.76, coup="mmr")))
    T.append(("v11_172.6_noalpha", dict(kind="v11", Mt=172.6, noalpha=True)))
    T.append(("v11_172.6_muEW246", dict(kind="v11", Mt=172.6, xi_ref_mu=246.22)))
    return T


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "test":
        r = task_checks()
        print(json.dumps(r, indent=1, default=float))
        sys.exit(0)
    from multiprocessing import Pool
    T = all_tasks()
    if mode != "all":
        T = [t for t in T if t[0].startswith(mode)]
    with Pool(int(os.environ.get("NWORK", "3"))) as pool:
        for name, st in pool.imap_unordered(run_task, T):
            print(time.strftime("%H:%M:%S"), name, st, flush=True)
