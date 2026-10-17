# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/fulldim_dmc/T14a_dmc.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
r"""[v12 package, 2026-10-15] Walled full-dimensional diffusion Monte Carlo (12 dims; the wall is the only exact node), 11 cases x 3 time steps x 2 shards with SHA-256-derived seeds; 'combine' extrapolates tau -> 0.  Paper: PRD App. H2 (Delta Q = -0.0032(8)). Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

T14a / C151: walled full-dimensional electroweak zero-mode DMC (fulldim-dmc, 2026-10-14).

Algorithm adapted from C72_dmc.run_case (copied with PROVENANCE in this directory), changes:
  * potential V(y) = g2^2 v_su2(x) + Higgs gradient(x,w,phi) + lam r^4/4 + dV_MC(r)    (C72 v_ew + D110 MC+CW dV(r))
  * Dirichlet wall at r = |phi| = r_w (gauge invariant); moves with r' >= r_w are rejected (psi_T(y') = 0)
  * guide psi_T = exp(-1/2 a_x |x|^2 - 1/2 a_w |w|^2) f(r): a_x, a_w = C72 optimal Gaussian (lam = 0), f = ground state
    of the 1D radial problem with the same wall (T14a_common.radial_guide); f(r_w) = 0
  * UNR drift limiting on the Higgs 4-block (forward and reverse Green functions use the same limited drift)
  * exact local energy of the spline guide: E_L = 1/2 sum a - 1/2 sum a^2 y^2 - 1/2 (f''+3f'/r)/f + V(y)
Shards: independent runs (own equilibration) written to run01/shards/<case>_tau<..>_k<..>.json; resumable.
    python -B T14a_dmc.py test           (timing)
    python -B T14a_dmc.py run <set> [nw] (set = lam0 | m010 | m013 | all)
    python -B T14a_dmc.py combine
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import json
import time
import hashlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import T14a_common as T  # noqa: E402
from C72_common import v_su2, v_higgs_gradient, minimize_ew  # noqa: E402

G2, GY = T.V11["g2"], T.V11["gY"]
UNR_A = 0.5


def case_list():
    cs = [dict(name="lam0_rw9", lam=0.0, rw=9.0)]
    cs += [dict(name=f"m010_rw{rw:.2f}", lam=-0.010, rw=rw) for rw in (4.80, 5.12, 5.44, 4.32, 4.08)]
    cs += [dict(name=f"m013_rw{rw:.2f}", lam=-0.013, rw=rw) for rw in (4.40, 4.64, 4.96, 4.08, 3.84)]
    return cs


TAUS = (0.04, 0.02, 0.01)
NSHARD = 2
W, T_EQ, T_PROD = 10000, 10.0, 40.0


def seed_of(name, tau, k):
    return int(hashlib.sha256(f"T14a|{name}|{tau}|{k}".encode()).hexdigest()[:8], 16)


def shard_path(name, tau, k):
    return os.path.join(T.SHARDS, f"{name}_tau{tau:.3f}_k{k}.json")


def run_shard(spec):
    name, lam, rw, tau, k = spec["name"], spec["lam"], spec["rw"], spec["tau"], spec["k"]
    Wn, t_eq, t_pr = spec.get("W", W), spec.get("T_eq", T_EQ), spec.get("T_prod", T_PROD)
    seed = seed_of(name, tau, k)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    m = minimize_ew(G2, GY, 0.0)
    ax, aw = m["a"] * G2 ** 2, m["b"] * GY ** 2
    al = np.concatenate([np.full(9, ax), np.full(3, aw)])
    g = T.radial_guide(lam, rw)
    Gd = T.Guide(g)
    dvs = T.dv_table(lam, rw + 0.1)

    def pot(y, r):
        x = y[:, :9].reshape(-1, 3, 3)
        return (G2 ** 2 * v_su2(x) + v_higgs_gradient(x, y[:, 9:12], y[:, 12:16], G2, GY)
                + 0.25 * lam * r ** 4 + dvs(r))

    def state(y):
        r = np.sqrt((y[:, 12:16] ** 2).sum(1))
        ok, lnf, dlf, elr = Gd.eval(r)
        el = 0.5 * al.sum() - 0.5 * ((al * y[:, :12]) ** 2).sum(1) + elr + pot(y, r)
        ln2 = -(al * y[:, :12] ** 2).sum(1) + 2 * lnf                       # ln psi_T^2
        v = np.empty_like(y)
        v[:, :12] = -al * y[:, :12]
        vh = (dlf / np.maximum(r, 1e-12))[:, None] * y[:, 12:16]
        v2 = (vh * vh).sum(1)
        fac = np.where(v2 * tau > 1e-12, (-1 + np.sqrt(1 + 2 * UNR_A * v2 * tau)) / (UNR_A * v2 * tau + 1e-300), 1.0)
        v[:, 12:16] = vh * fac[:, None]
        return ok, el, ln2, v, r

    # initial walkers: x, w from |G|^2; phi with radial density r^3 f^2 and uniform direction
    y = np.empty((Wn, 16))
    y[:, :12] = rng.normal(size=(Wn, 12)) / np.sqrt(2 * al)
    rg, fg = g["r"], g["f"]
    cdf = np.cumsum(rg ** 3 * fg ** 2)
    cdf /= cdf[-1]
    rr0 = np.interp(rng.random(Wn), cdf, rg)
    dirs = rng.normal(size=(Wn, 4))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    y[:, 12:16] = dirs * rr0[:, None]
    ok, E, ln2, v, r = state(y)
    Eref = float(E.mean())
    n_eq, n_pr = int(round(t_eq / tau)), int(round(t_pr / tau))
    sqt = np.sqrt(tau)
    ecut = 10.0 / sqt
    Emix = np.empty(n_pr); Egr = np.empty(n_pr); accs = np.empty(n_pr); taus = np.empty(n_pr)
    rmean = np.empty(n_pr); rmax = 0.0; n_wall = 0; n_try = 0
    for step in range(n_eq + n_pr):
        yp = y + tau * v + sqt * rng.normal(size=y.shape)
        okp, Ep, ln2p, vp, rp = state(yp)
        ln_gf = -((yp - y - tau * v) ** 2).sum(1) / (2 * tau)
        ln_gb = -((y - yp - tau * vp) ** 2).sum(1) / (2 * tau)
        arg = np.where(okp, ln2p - ln2 + ln_gb - ln_gf, -np.inf)
        acc = np.log(rng.random(Wn)) < arg
        dr2 = ((yp - y) ** 2).sum(1)
        tau_eff = tau * float((dr2 * acc).sum() / dr2.sum())
        Enew = np.where(acc, Ep, E)
        y = np.where(acc[:, None], yp, y)
        v = np.where(acc[:, None], vp, v)
        ln2 = np.where(acc, ln2p, ln2)
        r = np.where(acc, rp, r)
        S = 0.5 * (np.clip(E - Eref, -ecut, ecut) + np.clip(Enew - Eref, -ecut, ecut))
        wgt = np.exp(-tau_eff * S)
        wsum = wgt.sum()
        e_mix = float((wgt * Enew).sum() / wsum)
        e_gr = Eref - np.log(wsum / Wn) / tau_eff
        c = np.cumsum(wgt); c /= c[-1]
        idx = np.minimum(np.searchsorted(c, (rng.random() + np.arange(Wn)) / Wn), Wn - 1)
        y, E, v, ln2, r = y[idx], Enew[idx], v[idx], ln2[idx], r[idx]
        if step >= n_eq:
            kk = step - n_eq
            Emix[kk], Egr[kk], accs[kk], taus[kk] = e_mix, e_gr, acc.mean(), tau_eff
            rmean[kk] = float(r.mean())
            rmax = max(rmax, float(r.max()))
            n_wall += int((~okp).sum()); n_try += Wn
        Eref = 0.98 * Eref + 0.02 * e_mix
    out = dict(spec)
    out.update(seed=seed, W=Wn, T_eq=t_eq, T_prod=t_pr, guide_eps=g["eps"], guide_ax=ax, guide_aw=aw)
    for lab, series in (("mixed", Emix), ("growth", Egr)):
        ses = []
        bms = {}
        for T_blk in (1.0, 2.0, 4.0):
            bl = max(1, int(round(T_blk / tau)))
            nb = len(series) // bl
            bm = series[:nb * bl].reshape(nb, bl).mean(1)
            ses.append(float(bm.std(ddof=1) / np.sqrt(nb)))
            bms[f"{T_blk}"] = bm.tolist()
        out[f"E_{lab}"] = float(series.mean())
        out[f"se_{lab}"] = max(ses)
        out[f"se_{lab}_blocks_1_2_4"] = ses
        if lab == "mixed":
            out["block_means_T4"] = bms["4.0"]
    out.update(acceptance=float(accs.mean()), tau_eff_mean=float(taus.mean()), r_mean=float(rmean.mean()),
               r_max=rmax, frac_wall_rejected=n_wall / max(n_try, 1), wall_s=round(time.time() - t0, 1))
    T.wjson(shard_path(name, tau, k), out)
    return out


def specs_for(sel, nshard=None):
    nshard = NSHARD if nshard is None else nshard
    out = []
    for c in case_list():
        if sel != "all" and not c["name"].startswith(sel):
            continue
        for tau in TAUS:
            for k in range(nshard):
                out.append(dict(c, tau=tau, k=k))
    return out


def extrapolate(pts):
    t = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts]); s = np.array([p[2] for p in pts])
    A = np.vstack([np.ones_like(t), t]).T / s[:, None]
    coef, *_ = np.linalg.lstsq(A, e / s, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    chi2 = float((((A @ coef) - e / s) ** 2).sum())
    return dict(E0=float(coef[0]), se_E0=float(np.sqrt(cov[0, 0])), slope=float(coef[1]), chi2=chi2, dof=len(t) - 2)


def combine():
    res = {}
    for c in case_list():
        pts, rows = [], []
        for tau in TAUS:
            sh = [T.jload(shard_path(c["name"], tau, k)) for k in range(8) if os.path.exists(shard_path(c["name"], tau, k))]
            if not sh:
                continue
            e = np.array([s["E_mixed"] for s in sh]); se = np.array([s["se_mixed"] for s in sh])
            te = float(np.mean([s["tau_eff_mean"] for s in sh]))
            wgt = 1 / se ** 2
            em = float((wgt * e).sum() / wgt.sum())
            sem = float(1 / np.sqrt(wgt.sum()))
            if len(sh) > 1:          # inflate by shard scatter if larger
                chi = float(((e - em) ** 2 * wgt).sum() / (len(sh) - 1))
                sem *= max(1.0, np.sqrt(chi))
            pts.append((te, em, sem))
            rows.append(dict(tau=tau, tau_eff=te, E=em, se=sem, n_shards=len(sh),
                             E_growth=float(np.mean([s["E_growth"] for s in sh])),
                             acc=float(np.mean([s["acceptance"] for s in sh])),
                             r_mean=float(np.mean([s["r_mean"] for s in sh])), r_max=max(s["r_max"] for s in sh),
                             frac_wall_rejected=float(np.mean([s["frac_wall_rejected"] for s in sh]))))
        if len(pts) >= 3:
            ex = extrapolate(pts)
        elif pts:
            ex = dict(E0=None, note="fewer than 3 time steps")
        else:
            continue
        res[c["name"]] = dict(case=c, per_tau=rows, extrap=ex)
        print(c["name"], json.dumps(T.clean(ex)), flush=True)
    T.wjson(os.path.join(T.RES, "T14a_dmc_combined.json"), res)
    return res


def main():
    os.makedirs(T.SHARDS, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        for lam, rw in ((0.0, 9.0), (-0.010, 5.12)):
            o = run_shard(dict(name=f"TEST_{lam}", lam=lam, rw=rw, tau=0.02, k=0, W=2000, T_eq=1.0, T_prod=2.0))
            print(json.dumps({k: o[k] for k in ("E_mixed", "se_mixed", "E_growth", "acceptance", "tau_eff_mean", "r_mean",
                                                "r_max", "frac_wall_rejected", "wall_s", "guide_eps")}), flush=True)
        return
    if mode == "run":
        sel = sys.argv[2]
        nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        nsh = int(sys.argv[4]) if len(sys.argv) > 4 else None
        todo = [s for s in specs_for(sel, nsh) if not os.path.exists(shard_path(s["name"], s["tau"], s["k"]))]
        todo.sort(key=lambda s: (s["k"], -1.0 / s["tau"]))
        print(f"{len(todo)} shards to run with {nw} workers", flush=True)
        with Pool(nw) as pool:
            for o in pool.imap_unordered(run_shard, todo):
                print(f"{o['name']} tau={o['tau']} k={o['k']} E={o['E_mixed']:.5f}+-{o['se_mixed']:.5f} "
                      f"Egr={o['E_growth']:.5f} acc={o['acceptance']:.4f} rmax={o['r_max']:.2f} {o['wall_s']}s", flush=True)
        combine()
        return
    if mode == "combine":
        combine()


if __name__ == "__main__":
    main()
