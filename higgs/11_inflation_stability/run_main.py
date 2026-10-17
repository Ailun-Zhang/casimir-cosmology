# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/inflation_stability/run_main.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] xi_inf(m_t, alpha_s) scan (95 jobs, resumable shards).  Paper: PRD Sec. VII.

C155 step 2: xi_inf(m_t) on the project's plateau background (+ variants, cross-checks, extra reproduction jobs).
Resumable: one JSON shard per job in run01/shards/. Usage: python run_main.py [nworkers<=3] [group ...]"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import time
import math
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import infl_core as IC  # noqa: E402

RUN = os.path.join(IC.ROOT, _zp.w("r3/inflation_stability/run01"))
SH = os.path.join(RUN, "shards")
# project plateau (v11 Sec. III/VII): h/MP = 6.116e-6, phi_b = 5.5794 MP, N* = 55.722 (instant reheating), L0 = 6/H0
H_OURS, PHI_B, NSTAR = 6.116e-6, 5.5794, 55.722
H0_OURS = 67.4 * 1e3 / 3.08568e22 * 6.582119569e-25  # GeV (h = 0.674)
G3_B13 = 1.1666 + 0.00314 * (0.1179 - 0.1184) / 0.0007 - 0.00046 * (172.76 - 173.15)  # Buttazzo Eq.(60)

_BG = {}


def background(kind):
    if kind in _BG:
        return _BG[kind]
    if kind == "ours":
        bg = IC.plateau_background(3 * H_OURS ** 2, PHI_B)
        ne = bg["n_ev"]
        n_end = ne[1.0]
        a_end = IC.a_inf_pivot(bg, n_end, NSTAR, n_end)
        fT = IC.ball_cube_fraction(3.21, 6.0)
        cnt = IC.Count(bg, n_end, a_end, H0_OURS, n_end, [n_end - ne[0.25], 0.0], fT=fT)
        Hs = float(bg["state"](n_end - NSTAR)[3]) * IC.MPBAR
        meta = dict(kind=kind, h_over_MP=H_OURS, phi_b=PHI_B, Nstar=NSTAR, N_start_turn=n_end, a_end=a_end, H0=H0_OURS,
                    torus_fraction=fT, H_pivot=Hs, H_plateau=H_OURS * IC.MPBAR, N_end_eps_quarter=n_end - ne[0.25],
                    labels={"%.6f" % (n_end - ne[0.25]): "eps=1/4", "%.6f" % 0.0: "eps=1"})
    else:  # MMR 2021 reproduction background
        bg = IC.plateau_background(0.75 * (1.1e-5) ** 2, 6.0)
        ne = bg["n_ev"]
        n0 = ne[1.0]
        a0 = IC.a_inf_mmr(bg, n0)
        cnt = IC.Count(bg, n0, a0, 1.5e-42, 60.0, [0.0])
        meta = dict(kind=kind, a_inf=a0, labels={"%.6f" % 0.0: "eps=1 (MMR2021)"})
    _BG[kind] = (bg, cnt, meta)
    return _BG[kind]


def lam_cross(sm):
    lam = sm.lam_raw
    k = np.where(lam < 0)[0]
    if len(k) == 0:
        return None
    k0 = k[0]
    if k0 == 0:
        return float(np.exp(sm.t[0]))
    t1, t2, l1, l2 = sm.t[k0 - 1], sm.t[k0], lam[k0 - 1], lam[k0]
    return float(np.exp(t1 - l1 * (t2 - t1) / (l2 - l1)))


def classical_and_crude(sm, cnt):
    """(c) V''(0)>0 at all R of the grid (xi(mu*) > 0 branch); (c') Herranen 2014 Eq.14 V_max^{1/4} >= H (B >= 8pi^2/3)."""
    Rlo, Rhi = float(cnt.HR2.min()), float(cnt.HR2.max())

    def fv_ok(xi):
        return all(sm.barrier(R, xi - 1 / 6)["flag"] != "no_fv" for R in (Rlo, Rhi))
    a, b = 0.0, 0.12
    for _ in range(22):
        m = 0.5 * (a + b)
        a, b = (a, m) if fv_ok(m) else (m, b)
    xi_cl = b
    Rg = np.exp(np.linspace(math.log(Rlo), math.log(Rhi), 9))

    def g(xi):
        return min(sm.barrier(R, xi - 1 / 6)["B"] for R in Rg) - 8 * math.pi ** 2 / 3
    try:
        from scipy.optimize import brentq
        xi_cr = brentq(g, xi_cl + 1e-4, 0.16, xtol=1e-5)
    except Exception as e:  # noqa: BLE001
        xi_cr = "fail:%s" % e
    return dict(xi_classical_ew=xi_cl, xi_crude_Vmax_gt_H_ew=xi_cr)


def job(cfg):
    path = os.path.join(SH, cfg["key"] + ".json")
    if os.path.exists(path):
        return cfg["key"], "skip"
    t0 = time.time()
    ov = cfg.get("override")
    sm = IC.SMCouplings(cfg["Mt"], cfg["als"], cfg.get("Mh", 125.13), variant=cfg.get("variant", "central"),
                        override=ov, loops=cfg.get("loops", 3), xi_yuk=cfg.get("xi_yuk", "top"))
    bg, cnt, meta = background(cfg.get("bg", "ours"))
    mode = cfg.get("mode", "hm")
    r = IC.xi_bound(sm, cnt, cfg.get("presc", "mmr"), cfg.get("zeta", 1.0), r2M=cfg.get("r2M"),
                    kramers=(mode == "kramers"), gauss=(mode == "gauss"), hi=(0.165 if cfg.get("r2M") else 0.16))
    out = dict(cfg=cfg, meta={k: v for k, v in meta.items()}, bounds={})
    for Ne, v in r.items():
        lab = meta["labels"].get("%.6f" % Ne, str(Ne))
        d = dict(v)
        x = v.get("xi_ew")
        if x is not None:
            Np = v.get("N_peak", Ne)
            p, q, eps, H = bg["state"](cnt.n_ref - Np)
            HR2p = float(6 * (H * IC.MPBAR) ** 2 * (2 - eps) / 12)
            bb = sm.barrier(HR2p, x - 1 / 6, cfg.get("presc", "mmr"), cfg.get("zeta", 1.0))
            d.update(H_peak=math.sqrt(HR2p), B_peak=bb.get("B"), hbar_peak=bb.get("hbar"),
                     mu_bar_peak=math.exp(bb["t_bar"]) if "t_bar" in bb else None, xi_mustar_peak=bb.get("xi_bar"),
                     lam_mustar_peak=bb.get("lam_bar"), xi_at_mu_b=sm.xi_at(x, IC.MU_B),
                     xi_at_H_peak=sm.xi_at(x, math.sqrt(HR2p)))
        out["bounds"][lab] = d
    out["lam_mu_b"] = float(sm.at([IC.MU_B])["lam"][0])
    out["Xi_mu_b"] = float(sm.at([IC.MU_B])["Xi"][0])
    out["Lambda_I"] = lam_cross(sm)
    if cfg.get("extra"):
        out["extra"] = classical_and_crude(sm, cnt)
    out["secs"] = time.time() - t0
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    return cfg["key"], "%.1fs" % out["secs"]


def configs(groups):
    C = []
    mts = [round(171.4 + 0.2 * k, 1) for k in range(12)]
    if "A" in groups:  # main: alpha_s central and +-1 sigma (PDG 0.1180 +- 0.0009), MMR prescription, HM, zeta=1
        for als in (0.1180, 0.1171, 0.1189):
            for mt in mts:
                C.append(dict(key="A_mt%.1f_as%.4f" % (mt, als), Mt=mt, als=als, extra=(als == 0.1180)))
    if "B" in groups:  # prescription / gauge / xi-running / stochastic variants at alpha_s = 0.1180
        for mt in (171.8, 172.2, 172.6, 173.0, 173.4):
            for tag, extra in (("her", dict(presc="her")), ("tree", dict(presc="tree")), ("her10", dict(presc="her10")),
                               ("her01", dict(presc="her01")), ("zeta0", dict(zeta=0.0)),
                               ("xiyukall", dict(xi_yuk="all")), ("kramers", dict(mode="kramers")),
                               ("gauss", dict(mode="gauss"))):
                c = dict(key="B_%s_mt%.1f" % (tag, mt), Mt=mt, als=0.1180)
                c.update(extra)
                C.append(c)
    if "C" in groups:  # SM-mapping variants of sm_provider
        for mt in (172.2, 172.6, 173.0):
            for v in ("BKPV15", "B13", "xi2", "loops3"):
                C.append(dict(key="C_%s_mt%.1f" % (v, mt), Mt=mt, als=0.1180, variant=v))
    if "E" in groups:  # counterfactual: the same plateau realised as R^2 gravity (M = 2h/MP), MMR 2022 extras
        for mt in (172.2, 172.6, 173.0):
            C.append(dict(key="E_R2_mt%.1f" % mt, Mt=mt, als=0.1180, r2M=2 * H_OURS))
    if "D" in groups:  # extra reproduction jobs on the MMR 2021 background
        C.append(dict(key="D_MMRtab_g3B13", Mt=172.76, als=0.1179, Mh=125.10, bg="mmr",
                      override=dict(IC.MMR_TABLE1, g3=G3_B13)))
        for mt in (172.16, 172.76, 173.36):
            C.append(dict(key="D_provider_mt%.2f" % mt, Mt=mt, als=0.1179, Mh=125.10, bg="mmr"))
    return C


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(SH, exist_ok=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # [v12-package] was capped at 3
    groups = sys.argv[2:] or ["D", "A", "B", "C"]
    C = configs(groups)
    print("jobs", len(C), "workers", nw, flush=True)
    T0 = time.time()
    with Pool(nw) as pool:
        for k, s in pool.imap_unordered(job, C):
            print(k, s, "t=%.0fs" % (time.time() - T0), flush=True)
    print("all done %.0fs" % (time.time() - T0), flush=True)
