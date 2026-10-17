# PROVENANCE: 04_calculations/code/20261011_claude_followup/C72/C72_dmc.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Diffusion Monte Carlo ground energies of the local constant-mode Hamiltonians (quartic validation 2.1407, van Baal eps_2 = 4.116720, Pavel eps_3 = 12.5868, electroweak block at several lambda). The exact-local energies give the zero-mode fraction xi = Q/C_free = 0.2999 (lambda = 0.01). Paper: PRD Sec. IV C, App. app:zeromodes (DMC paragraph). Output <work>/fu/C72/C72_dmc_results.json (--test: C72_dmc_test.json). Pool size = env DNEC_WORKERS (default 8); every case has its own seed, so results do not depend on the worker count.

C72 DMC: exact ground energies of the LOCAL constant-mode Hamiltonians (domain R^n, no fast-mode matching).

The unconstrained Hamiltonian (1/2) p^2 + V on R^n with V >= 0 has a unique positive ground state; it is invariant
under the compact connected gauge group (SU(2) x U(1) acts orthogonally on (x, w, phi)), hence it satisfies the Gauss
law. Bosonic DMC without nodes therefore gives the physical (gauge-invariant) ground energy, up to time-step,
population and statistical errors. Importance sampling with the optimal Gaussian guide psi_T = exp(-sum alpha_k y_k^2/2),
Metropolis accept/reject (Reynolds-Umrigar), effective time step, fixed population with systematic resampling.

Cases (unit-mass coordinates):
  quartic4 : (1/2)p^2 + (lam/4)(phi^2)^2 on R^4, lam = 1      -> validation against the certified 2.1407
  su2      : (1/2)p^2 + (1/4)F^2, F = eps x x, R^9             -> van Baal eps = 4.116720 (normalisation test)
  su3      : (1/2)p^2 + (1/4)F^2, f of SU(3), R^24             -> Pavel 12.5868 (variational)
  ew_*     : SU(2) x U(1)_Y x Higgs, L*H in C72_common docstring, at (g2, gY, lam)

    python C72_dmc.py            (full run, ~10 min on 8 workers)
    python C72_dmc.py --test     (timing test)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C72_common import (RES, C_FREE, structure_constants, v_su2, v_sun, v_ew, minimize_ew, couplings_one_loop,  # noqa: E402
                        mu_bounce_gev, v_higgs_gradient, v_higgs_gradient_matrix)


def build(case):
    kind = case["kind"]
    if kind == "quartic4":
        lam = case["lam"]
        alphas = np.full(4, (3 * lam) ** (1 / 3))
        def pot(y):
            r2 = (y * y).sum(1)
            return 0.25 * lam * r2 * r2
    elif kind == "su2":
        alphas = np.full(9, 2 ** (1 / 3))
        pot = lambda y: v_su2(y.reshape(-1, 3, 3))
    elif kind == "su3":
        f = structure_constants(3)
        alphas = np.full(24, 3 ** (1 / 3))
        pot = lambda y: v_sun(y.reshape(-1, 8, 3), f)
    elif kind == "ew":
        g2, gY, lam = case["g2"], case["gY"], case["lam"]
        m = minimize_ew(g2, gY, lam)
        alphas = np.concatenate([np.full(9, m["a"] * g2 ** 2), np.full(3, m["b"] * gY ** 2), np.full(4, m["z"])])
        pot = lambda y: v_ew(y, g2, gY, lam)
    else:
        raise ValueError(kind)
    return pot, alphas


def run_case(case):
    t0 = time.time()
    pot, alphas = build(case)
    tau, W = case["tau"], case["W"]
    rng = np.random.default_rng(case["seed"])
    n = alphas.size
    y = rng.normal(size=(W, n)) / np.sqrt(2 * alphas)
    half_tr = 0.5 * alphas.sum()
    el = lambda yy, vv: half_tr - 0.5 * ((alphas * yy) ** 2).sum(1) + vv
    E = el(y, pot(y))
    Eref = float(E.mean())
    n_eq, n_pr = int(round(case["T_eq"] / tau)), int(round(case["T_prod"] / tau))
    sqt = np.sqrt(tau)
    ecut = 10.0 / sqt
    Emix = np.empty(n_pr); Egr = np.empty(n_pr); accs = np.empty(n_pr); taus = np.empty(n_pr)
    for step in range(n_eq + n_pr):
        drift = -alphas * y
        yp = y + tau * drift + sqt * rng.normal(size=y.shape)
        Ep = el(yp, pot(yp))
        driftp = -alphas * yp
        ln_ratio = -(alphas * (yp * yp - y * y)).sum(1)                       # ln psi_T(y')^2 / psi_T(y)^2
        ln_gf = -((yp - y - tau * drift) ** 2).sum(1) / (2 * tau)
        ln_gb = -((y - yp - tau * driftp) ** 2).sum(1) / (2 * tau)
        acc = np.log(rng.random(W)) < (ln_ratio + ln_gb - ln_gf)
        dr2 = ((yp - y) ** 2).sum(1)
        tau_eff = tau * float((dr2 * acc).sum() / dr2.sum())
        Enew = np.where(acc, Ep, E)
        y = np.where(acc[:, None], yp, y)
        S = 0.5 * (np.clip(E - Eref, -ecut, ecut) + np.clip(Enew - Eref, -ecut, ecut))
        w = np.exp(-tau_eff * S)
        wsum = w.sum()
        e_mix = float((w * Enew).sum() / wsum)
        e_gr = Eref - np.log(wsum / W) / tau_eff
        c = np.cumsum(w); c /= c[-1]
        idx = np.minimum(np.searchsorted(c, (rng.random() + np.arange(W)) / W), W - 1)
        y, E = y[idx], Enew[idx]
        if step >= n_eq:
            k = step - n_eq
            Emix[k], Egr[k], accs[k], taus[k] = e_mix, e_gr, acc.mean(), tau_eff
        Eref = 0.98 * Eref + 0.02 * e_mix
    out = dict(case)
    for lab, series in (("mixed", Emix), ("growth", Egr)):
        ses = []
        for T_blk in (1.0, 2.0, 4.0):
            bl = max(1, int(round(T_blk / tau)))
            nb = len(series) // bl
            bm = series[:nb * bl].reshape(nb, bl).mean(1)
            ses.append(float(bm.std(ddof=1) / np.sqrt(nb)))
        out[f"E_{lab}"] = float(series.mean())
        out[f"se_{lab}"] = max(ses)
        out[f"se_{lab}_blocks_1_2_4"] = ses
    out.update(acceptance=float(accs.mean()), tau_eff_mean=float(taus.mean()), n_steps=n_eq + n_pr,
               wall_s=round(time.time() - t0, 1), guide_alphas=sorted(set(np.round(alphas, 10).tolist())))
    return out


def extrapolate(rows):
    """Weighted linear fit E(tau_eff) = E0 + k tau_eff over the time steps of one case (mixed estimator)."""
    t = np.array([r["tau_eff_mean"] for r in rows]); e = np.array([r["E_mixed"] for r in rows])
    s = np.array([r["se_mixed"] for r in rows])
    A = np.vstack([np.ones_like(t), t]).T / s[:, None]
    coef, *_ = np.linalg.lstsq(A, e / s, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    chi2 = float((((A @ coef) - e / s) ** 2).sum())
    i_min = int(np.argmin(t))
    return dict(E0=float(coef[0]), se_E0=float(np.sqrt(cov[0, 0])), slope=float(coef[1]), chi2=chi2, dof=len(t) - 2,
                E_smallest_tau=float(e[i_min]), se_smallest_tau=float(s[i_min]),
                syst_extrap_minus_smallest=float(coef[0] - e[i_min]))


def self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 3, 3)); w = rng.normal(size=(50, 3)); phi = rng.normal(size=(50, 4))
    f2 = structure_constants(2)
    d1 = float(np.abs(v_sun(x, f2) - v_su2(x)).max())
    d2 = float(np.abs(v_higgs_gradient(x, w, phi, 0.5, 0.46) - v_higgs_gradient_matrix(x, w, phi, 0.5, 0.46)).max())
    return dict(su2_f_vs_cross=d1, higgs_identity_vs_matrix=d2)


def cases(test=False):
    mu = mu_bounce_gev(C_FREE)
    run = couplings_one_loop(mu)
    W, T_eq, T_prod = (2000, 0.5, 2.0) if test else (10000, 4.0, 40.0)
    taus_val = (0.04, 0.02, 0.01, 0.005)
    taus_ew = (0.04, 0.02, 0.01)
    base = [dict(name="quartic4_lam1", kind="quartic4", lam=1.0, taus=taus_val),
            dict(name="su2", kind="su2", taus=taus_val),
            dict(name="su3", kind="su3", taus=taus_val),
            dict(name="ew_g05_gY046_lam0.01", kind="ew", g2=0.5, gY=0.46, lam=0.01, taus=taus_ew),
            dict(name="ew_g05_gY046_lam0", kind="ew", g2=0.5, gY=0.46, lam=0.0, taus=taus_ew),
            dict(name="ew_g05_gY046_lam0.1", kind="ew", g2=0.5, gY=0.46, lam=0.1, taus=taus_ew),
            dict(name="ew_g05_gY046_lam0.001", kind="ew", g2=0.5, gY=0.46, lam=0.001, taus=taus_ew),
            dict(name="ew_g05_gY046_lam0.05", kind="ew", g2=0.5, gY=0.46, lam=0.05, taus=taus_ew),
            dict(name="ew_running_lam0.01", kind="ew", g2=run["g2"], gY=run["gY_SM"], lam=0.01, taus=taus_ew),
            dict(name="ew_running_lam0", kind="ew", g2=run["g2"], gY=run["gY_SM"], lam=0.0, taus=taus_ew),
            dict(name="ew_running_lam0.1", kind="ew", g2=run["g2"], gY=run["gY_SM"], lam=0.1, taus=taus_ew)]
    if test:
        base = [b for b in base if b["name"] in ("su3", "ew_g05_gY046_lam0.01")]
    out = []
    seed = 1000
    for b in base:
        for tau in (b["taus"][:1] if test else b["taus"]):
            seed += 1
            c = {k: v for k, v in b.items() if k != "taus"}
            c.update(tau=tau, W=W, T_eq=T_eq, T_prod=T_prod, seed=seed)
            out.append(c)
    return out, run


def main():
    test = "--test" in sys.argv
    RES.mkdir(parents=True, exist_ok=True)
    st = self_test()
    print("self-test:", st, flush=True)
    cs, run = cases(test)
    cs.sort(key=lambda c: -(c["T_eq"] + c["T_prod"]) / c["tau"] * (24 if c["kind"] == "su3" else 16 if c["kind"] == "ew" else 9))
    with Pool(min(int(os.environ.get("DNEC_WORKERS", "8")), len(cs))) as pool:  # package: --workers
        rows = []
        for r in pool.imap_unordered(run_case, cs):
            print(f"{r['name']:24s} tau={r['tau']:.3f} E_mix={r['E_mixed']:.5f}+-{r['se_mixed']:.5f} "
                  f"E_gr={r['E_growth']:.5f}+-{r['se_growth']:.5f} acc={r['acceptance']:.4f} {r['wall_s']}s", flush=True)
            rows.append(r)
    summary = {}
    for name in sorted({r["name"] for r in rows}):
        rr = sorted([r for r in rows if r["name"] == name], key=lambda r: r["tau"])
        summary[name] = extrapolate(rr) if len(rr) >= 3 else dict(E_smallest_tau=rr[0]["E_mixed"], se_smallest_tau=rr[0]["se_mixed"])
        print(name, summary[name], flush=True)
    fn = RES / ("C72_dmc_test.json" if test else "C72_dmc_results.json")
    fn.write_text(json.dumps(dict(self_test=st, running_couplings=run, rows=rows, extrapolated=summary), indent=1),
                  encoding="utf-8")
    print("wrote", fn)


if __name__ == "__main__":
    main()
