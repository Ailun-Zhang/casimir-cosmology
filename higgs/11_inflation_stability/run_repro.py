# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/inflation_stability/run_repro.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Literature reproduction: Mantziris-Markkanen-Rajantie 2021 field-theory Starobinsky bound xi_EW = 0.05946 (target 0.05938) and MMR 2022 R^2 counterfactual.  Paper: PRD Sec. VII.

C155 step 1: implementation checks + literature reproduction (MMR 2021 Starobinsky-like field theory; MMR 2022 R^2)."""
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
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import infl_core as IC  # noqa: E402

OUT = os.path.join(IC.ROOT, _zp.w("r3/inflation_stability/run01/repro"))
os.makedirs(OUT, exist_ok=True)
res = {}
T0 = time.time()

# --- SM inputs as in MMR 2021 Table 1 (mu_EW = m_t = 172.76), 3-loop RGE; g3, yb, ytau from sm_provider (als 0.1179)
sm = IC.SMCouplings(172.76, 0.1179, 125.10, override=IC.MMR_TABLE1, loops=3)
smP = IC.SMCouplings(172.76, 0.1179, 125.10)  # same physical inputs through sm_provider NNLO matching (5-loop RGE)
for name, s in (("MMRtab_3loop", sm), ("provider_central", smP)):
    mus = [172.76, 1e10, 1e12, 1e13, 1e14, 1e16, 1e18]
    c = s.at(mus)
    res["couplings_" + name] = {"%.3g" % m: dict(lam=float(c["lam"][k]), yt=float(c["yt"][k]), Xi=float(c["Xi"][k]))
                                for k, m in enumerate(mus)}
print("couplings", json.dumps(res, indent=0)[:1500], flush=True)

# --- single-point barrier checks at H = 1e13 GeV
for xi in (0.04, 0.06, 0.10):
    for presc in ("mmr", "her", "tree"):
        b = sm.barrier(1e26, xi - 1 / 6, presc, want_curv=True)
        res["barrier_H1e13_xi%.2f_%s" % (xi, presc)] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                         for k, v in b.items()}
        print(xi, presc, {k: b[k] for k in ("B", "hbar", "flag") if k in b},
              "mu*bar=%.3g" % math.exp(b.get("t_bar", 0)), "xi*=%.4f lam*=%.5f" % (b.get("xi_bar", 0), b.get("lam_bar", 0)),
              flush=True)
t0 = sm.t_star(np.array([0.0, 1e13, 1e14]), 1e26, 0.06 - 1 / 6)
print("mu*(h=0,1e13,1e14; H=1e13) =", np.exp(t0), flush=True)
res["mu_star_H1e13"] = [float(x) for x in np.exp(t0)]
print("t=%.1fs" % (time.time() - T0), flush=True)

# --- MMR 2021 Starobinsky-like (field theory): N=0 at a''=0, N_start=60, a_inf Eq.(4.2) with H0=1.5e-42 GeV
bg = IC.plateau_background(0.75 * (1.1e-5) ** 2, 6.0)  # ~100 e-folds of slow roll (MMR start at N=106; irrelevant)
ne = bg["n_ev"]
print("events", ne, flush=True)
res["bg_mmr"] = dict(n_ev={str(k): v for k, v in ne.items()})
n_acc0, n_R0 = ne[1.0], ne[2.0]
aA = IC.a_inf_mmr(bg, n_acc0)
aR = IC.a_inf_mmr(bg, n_R0)
pR = bg["state"](n_R0)
res["bg_mmr"].update(a_inf_acc0=aA, a_inf_R0=aR, phi_R0=float(pR[0]), H_R0_GeV=float(pR[3]) * IC.MPBAR,
                     dN_acc0_R0=n_R0 - n_acc0)
print("a_inf(a''=0)=%.4g a_inf(R=0)=%.4g phi(R=0)=%.3f H(R=0)=%.3g dN=%.4f" % (
    aA, aR, pR[0], pR[3] * IC.MPBAR, n_R0 - n_acc0), flush=True)

cntA = IC.Count(bg, n_acc0, aA, 1.5e-42, 60.0, [0.0])
t1 = time.time()
r = IC.xi_bound(sm, cntA, "mmr")
res["MMR2021_star_fieldtheory_MMRtab"] = dict(bound=r[0.0], target=0.05938, secs=time.time() - t1)
print("MMR2021 Starobinsky field theory (target 0.05938):", r, "%.1fs" % (time.time() - t1), flush=True)

# --- MMR 2022 conventions: N=0 at R=0; N_end at Hdot/H^2 = -1/4 (eps=1/4) and -1 (eps=1); field theory and R^2
NeQ, NeA = n_R0 - ne[0.25], n_R0 - ne[1.0]
cntR = IC.Count(bg, n_R0, aR, 1.5e-42, 60.0, [NeQ, NeA])
t1 = time.time()
rf = IC.xi_bound(sm, cntR, "mmr")
print("field theory, R=0 ref:", rf, "%.1fs" % (time.time() - t1), flush=True)
t1 = time.time()
rr = IC.xi_bound(sm, cntR, "mmr", r2M=1.1e-5, hi=0.165)
print("R^2 (target ~0.1 at eps=1/4):", rr, "%.1fs" % (time.time() - t1), flush=True)
res["MMR2022_conventions"] = dict(N_end_eps_quarter=NeQ, N_end_eps_1=NeA,
                                  fieldtheory={("eps=1/4" if k == NeQ else "eps=1"): v for k, v in rf.items()},
                                  R2={("eps=1/4" if k == NeQ else "eps=1"): v for k, v in rr.items()},
                                  target_R2_eps_quarter="~0.1 (MMR 2022 Sec.5, Fig.5)")
# provider couplings, same physics inputs (shows matching systematics)
t1 = time.time()
rp = IC.xi_bound(smP, cntA, "mmr")
res["MMR2021_star_fieldtheory_provider"] = dict(bound=rp[0.0], secs=time.time() - t1)
print("provider couplings (172.76, 0.1179, 125.10):", rp, flush=True)
res["total_secs"] = time.time() - T0
with open(os.path.join(OUT, "repro_step1.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, default=str)
print("done %.1fs" % (time.time() - T0))
