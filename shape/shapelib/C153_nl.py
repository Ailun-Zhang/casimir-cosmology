# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_nl.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] D125 nonlinear runs (run01): A1 conditional theorem grid (real optimal-width Gaussians at t_x = -tau and +tau, paired, tau = 0...1.1/h), A2 never-real / phase / adiabatic families, B next order (kappa_E = 18.0), width scan; analysis -> C153_nl_summary.json (two-branch survival S2, t_*). Shared by shape/02 (t_*) and shape/04 (next-order amplitude threshold). Paper: PRD Sec. VIII B (sec:s-conditional, fig:s-turning), App. app:nonlinear (tab:s-S2).

C153_nl (shape-hardening, DNEC v12 round 3; derivation D125): nonlinear shape survival for
  A1  real (C = 0) optimal-width Gaussians at t_x = -tau and +tau (tau grid 0..1.1/h), paired runs
        A: forward from -tau with (beta, Pi)  [future branch of t_x=-tau; time mirror of the past branch of t_x=+tau]
        B: forward from +tau with (beta, -Pi) [mirror of the past branch of t_x=-tau; future branch of t_x=+tau]
      (real-state Wigner functions are Pi-symmetric, so (beta,-Pi) is an equally valid draw);
  A2  never-real / complex states: phase family at the turn (phi = C/delta on E modes), never-real pair at the
      turn (outgoing C(-inf)>0 and its mirror, incoming C(+inf)<0), first-order adiabatic states prepared at t_in<0;
  B   next order (kappa_E = 18.0 by the rescaled-force completion r_f = 1 - 13.528581/18, delta from xi = 0.28):
      turning-point amplitude scan s = 1, 3, 5, 10 (C76 hi seed 20261011, first 256) and R1 t_x = -0.5, -1, F.
Model = C142/C76/C60 copies in this directory (unchanged apart from the PROVENANCE line): full nonlinear volume +
five shape moduli, SM twisted Ewald Casimir force, zero-mode energy, leading cosh background, DOP853 rtol 1e-11.
New here: class NLR2 (start at t0 > 0 on the expanding branch, H0 = +sqrt(constraint); fate without bounce).
    python C153_nl.py smoke
    python C153_nl.py run --groups grid next phase never adiab --procs 4      (resumable, one JSON per draw)
    python C153_nl.py analyze
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json, time, hashlib, argparse
from collections import Counter
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C142_shape_nonlinear_threshold as C142            # noqa: E402  (copy)
import C76_nonlinear_zero_mode as C76                     # noqa: E402  (copy)
from C60_nonlinear_integrator import pack, unpack, C32    # noqa: E402  (copy)
from C60_twisted_ewald import ControlExit, beta_from_shape  # noqa: E402  (copy)

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_hardening/run01")
DRAWS = OUT / "draws"
TYPE = C76.TYPE
ASP = TYPE == 0
KF = C76.KAPPA
MODELS = {"xi030": {"r": C76.R_ZM, "delta": C76.DELTA0 * (1 - C76.R_ZM) ** -0.75},
          "next": {"r": 1 - KF[0] / 18.0, "delta": C76.DELTA0 * (1 - 0.28) ** -0.75}}
TAUS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
N_TAU = {t: (512 if 0.15 <= t <= 0.4 + 1e-9 else 256) for t in TAUS}
PHIS = (0.5, 1.0, 2.0, 4.0)
NEVER_S = (1.2, 2.0)                       # Var beta_E(0) in units of the never-real threshold 2|m12 m22| delta
ADIAB_TIN = (-0.25, -0.5, -1.0)
NX_AMP = (1, 3, 5, 10)
NX_TX = ("-0.5", "-1.0", "F")
SEED_GRID, SEED_PHASE, SEED_NEVER, SEED_ADIAB, SEED_NX = 20261401, 20261501, 20261551, 20261601, 20261701
N_PH, N_NEVER, N_AD, N_NX = 256, 256, 256, 256


def kap_of(model):
    return KF / (1 - MODELS[model]["r"])


def opt_widths(model, t0):
    """Per-mode optimal real width s_opt = |M12/M11| delta for the transfer t0 -> +inf (C137c / C142 r1_info)."""
    kap, d = kap_of(model), MODELS[model]["delta"]
    return np.array([abs(C142.lin_M(k, t0)[0, 1] / C142.lin_M(k, t0)[0, 0]) * d for k in kap]), d, kap


def nat_widths(model):
    kap, d = kap_of(model), MODELS[model]["delta"]
    return (3 / (2 * np.sqrt(2))) * d / np.sqrt(np.abs(kap)), d, kap


def tau_key(t):
    return "%.3f" % t


class NLR2(C142.NLR):
    """C142 NLR + start on the expanding branch (t0 > 0): no bounce required for survival."""

    def run_two(self, beta0, Pi0, t0):
        if t0 <= 0.0:
            return self.run_one(beta0, Pi0, t0)
        t_start = time.perf_counter()
        self.nfev, self.maxH, self.last = 0, -np.inf, None
        rec = {"t0": float(t0), "T": C142.T_END, "rtol": self.rtol, "atol": self.atol, "cut": self.cas.cut,
               "eig_stop": self.eig_stop, "branch": "expanding_start"}
        try:
            G0, M0, s0, zeta = self.shape_state(beta0, Pi0)
            ev0 = np.linalg.eigvalsh(G0)
            rec.update(s0=s0, zeta0=zeta, lnaniso0=float(0.5 * np.log(ev0[-1] / ev0[0])))
            lr0 = 0.5 * float(np.log(np.cosh(2 * t0)))
            q = 1.0 - zeta * np.exp(-4 * lr0) + s0 * np.exp(-6 * lr0) / 6.0
            rec.update(lr0=lr0, S0=float(s0 * np.exp(-6 * lr0)), H0_background=float(np.tanh(2 * t0)))
            if q < 0:
                rec.update(status="forbidden_start", fate="other", bounced=False, survivor=False, survivor_C60=False)
                return self._fin(rec, t_start)
            y0 = pack(lr0, +np.sqrt(q), G0, M0)
            rec.update(H0=float(y0[1]), constraint0=self.constraint(y0))
            sol = self._integrate(y0, t0)
            self._classify(sol, rec)
            cr = rec.get("H_crossings", [])
            ok = bool(rec["status"] == "complete" and rec["H_end"] > 0.9 and rec["min_eigG_path"] > self.eig_stop and not cr)
            rec["survivor"], rec["survivor_C60"] = ok, bool(ok and rec["min_eigG_path"] > 1e-3)
            rec["fate"] = "survive" if ok else ("pinch_expanding" if rec["status"] == "pinch" else
                                                 ("recollapse" if (cr or rec["status"] == "collapse") else "other"))
        except (ControlExit, C142.NfevLimit) as exc:
            rec.update(status="control_exit" if isinstance(exc, ControlExit) else "nfev_limit", reason=str(exc),
                       bounced=False, survivor=False, survivor_C60=False,
                       fate="pinch_expanding" if isinstance(exc, ControlExit) else "stalled")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rec.update(status="numerical_error", reason=str(exc), survivor=False, survivor_C60=False, fate="other")
        except Exception as exc:
            rec.update(status="error", reason=repr(exc), survivor=False, survivor_C60=False, fate="other")
        return self._fin(rec, t_start)

    def backward_beta(self, beta0, Pi0, t0, t1):
        """Direct backward integration from t0 > 0 (expanding start) to t1 < t0 without events (mirror check)."""
        G0, M0, s0, zeta = self.shape_state(beta0, Pi0)
        lr0 = 0.5 * float(np.log(np.cosh(2 * t0)))
        q = 1.0 - zeta * np.exp(-4 * lr0) + s0 * np.exp(-6 * lr0) / 6.0
        y0 = pack(lr0, +np.sqrt(q), G0, M0)
        self.nfev = 0
        try:
            sol = solve_ivp(self.rhs, (t0, t1), y0, method="DOP853", rtol=self.rtol, atol=self.atol, max_step=self.max_step)
        except (ControlExit, C142.NfevLimit) as exc:            # backward run pinched (Ewald box limit)
            return None, "pinched: %s" % exc
        G = unpack(sol.y[:, -1])[2]
        return beta_from_shape(G / np.linalg.det(G) ** (1 / 3)).tolist(), bool(sol.success)


# ------------------------------------------------------------------ specs
def mk(set_, i, model, t0, b, p, group, meta=None):
    return dict(set=set_, i=int(i), model=model, t0=float(t0), beta0=[float(x) for x in b], Pi0=[float(x) for x in p],
                group=group, meta=meta or {})


def grid_specs():
    S = []
    for k, tau in enumerate(TAUS):
        n = N_TAU[tau]
        s, d, _ = opt_widths("xi030", -tau)
        z = np.random.default_rng(SEED_GRID + 10 * k).standard_normal((n, 10))
        B, P = np.sqrt(s) * z[:, :5], d / np.sqrt(s) * z[:, 5:]
        for i in range(n):
            S.append(mk("g_tau%s_A" % tau_key(tau), i, "xi030", -tau, B[i], P[i], "grid", {"tau": tau}))
            S.append(mk("g_tau%s_B" % tau_key(tau), i, "xi030", +tau, B[i], -P[i], "grid", {"tau": tau}))
    return S


def corr_draws(rng, n, var_b, cov, var_p):
    """Draws from per-mode 2x2 Gaussians: beta = sqrt(vb) z1, Pi = (c/vb) beta + sqrt(vp - c^2/vb) z2."""
    z = rng.standard_normal((n, 10))
    B = np.sqrt(var_b) * z[:, :5]
    P = (cov / var_b) * B + np.sqrt(var_p - cov ** 2 / var_b) * z[:, 5:]
    return B, P


def phase_specs():
    S = []
    s, d, _ = opt_widths("xi030", 0.0)
    for k, ph in enumerate(PHIS):
        c = np.where(ASP, ph * d, 0.0)
        B, P = corr_draws(np.random.default_rng(SEED_PHASE + k), N_PH, s, c, (d ** 2 + c ** 2) / s)
        for i in range(N_PH):
            S.append(mk("ph%.1f_A" % ph, i, "xi030", 0.0, B[i], P[i], "phase", {"phi": ph}))
            S.append(mk("ph%.1f_B" % ph, i, "xi030", 0.0, B[i], -P[i], "phase", {"phi": -ph}))
    return S


def never_params(mult):
    s, d, kap = opt_widths("xi030", 0.0)
    M = C142.lin_M(kap[0], 0.0)
    m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    thr = 2 * abs(m12 * m22) * d
    sE = mult * thr
    cE = (m11 * m22 + m12 * m21) * sE / (2 * m12 * m22)          # maximises C(-inf): outgoing never-real state
    vb = np.where(ASP, sE, s)
    c = np.where(ASP, cE, 0.0)
    return vb, c, (d ** 2 + c ** 2) / vb, {"threshold_var": thr, "sE": sE, "cE_over_delta": cE / d,
                                         "beta_rms_E": float(np.sqrt(sE)), "m": [m11, m12, m21, m22]}


def never_specs():
    S = []
    for k, mult in enumerate(NEVER_S):
        vb, c, vp, info = never_params(mult)
        B, P = corr_draws(np.random.default_rng(SEED_NEVER + k), N_NEVER, vb, c, vp)
        for i in range(N_NEVER):
            S.append(mk("nv%.1f_out" % mult, i, "xi030", 0.0, B[i], P[i], "never", {"mult": mult, "cls": "outgoing"}))
            S.append(mk("nv%.1f_in" % mult, i, "xi030", 0.0, B[i], -P[i], "never", {"mult": mult, "cls": "incoming"}))
    return S


def adiab_params(t_in):
    kap, d = kap_of("xi030"), MODELS["xi030"]["delta"]
    c2 = np.cosh(2 * t_in)
    m, H = c2 ** 1.5, np.tanh(2 * t_in)
    vb, vp, cv = np.empty(5), np.empty(5), np.empty(5)
    for a, k in enumerate(kap):
        wl = np.sqrt(abs(k)) / c2
        W, dlnW = (wl, -2 * H) if wl >= abs(H) else (abs(H), 2 / c2 ** 2 / H)
        g = 3 * H + dlnW
        vb[a], vp[a], cv[a] = d * C32 / (m * W), d * m * (W ** 2 + g ** 2 / 4) / (C32 * W), -d * g / (2 * W)
    return vb, cv, vp


def adiab_specs():
    S = []
    for k, t_in in enumerate(ADIAB_TIN):
        vb, cv, vp = adiab_params(t_in)
        B, P = corr_draws(np.random.default_rng(SEED_ADIAB + k), N_AD, vb, cv, vp)
        for i in range(N_AD):
            S.append(mk("ad_tin%.2f" % t_in, i, "xi030", t_in, B[i], P[i], "adiab", {"t_in": t_in}))
    return S


def nx_tx_value(tx):
    return -0.5 * float(np.arcsinh(np.sqrt(18.0))) if tx == "F" else float(tx)


def next_specs():
    S = []
    s, d, _ = nat_widths("next")
    z = np.random.default_rng(C76.SEED_HI).standard_normal((2048, 10))[:N_NX]
    for amp in NX_AMP:
        B, P = amp * np.sqrt(s) * z[:, :5], amp * d / np.sqrt(s) * z[:, 5:]
        for i in range(N_NX):
            S.append(mk("nx_amp_s%d" % amp, i, "next", 0.0, B[i], P[i], "next", {"s": amp}))
    for k, tx in enumerate(NX_TX):
        t0 = nx_tx_value(tx)
        so, d, _ = opt_widths("next", t0)
        zz = np.random.default_rng(SEED_NX + k).standard_normal((N_NX, 10))
        B, P = np.sqrt(so) * zz[:, :5], d / np.sqrt(so) * zz[:, 5:]
        for i in range(N_NX):
            S.append(mk("nx_r1_tx%s" % tx, i, "next", t0, B[i], P[i], "next", {"tx": tx}))
    return S


WIDTH_TAU, WIDTH_X, N_W = (0.3, 0.4), (0.5, 2.0), 256


def width_specs():
    """Real Gaussians at t_x = -tau with non-optimal E widths x*s_opt (T2 at optimum); same z as the grid (paired)."""
    S = []
    for tau in WIDTH_TAU:
        k = TAUS.index(tau)
        s, d, _ = opt_widths("xi030", -tau)
        z = np.random.default_rng(SEED_GRID + 10 * k).standard_normal((N_TAU[tau], 10))[:N_W]
        for x in WIDTH_X:
            sx = np.where(ASP, x * s, s)
            B, P = np.sqrt(sx) * z[:, :5], d / np.sqrt(sx) * z[:, 5:]
            for i in range(N_W):
                S.append(mk("w_tau%s_x%.1f" % (tau_key(tau), x), i, "xi030", -tau, B[i], P[i], "width", {"tau": tau, "x": x}))
    return S


BUILD = {"grid": grid_specs, "phase": phase_specs, "never": never_specs, "adiab": adiab_specs, "next": next_specs,
         "width": width_specs}
RANK = {"grid": 0, "next": 1, "phase": 2, "never": 3, "adiab": 4, "width": 5}

_M = {}


def model_for(model, cut=42.0):
    key = (model, cut)
    if key not in _M:
        _M[key] = NLR2(MODELS[model]["r"], cut=cut)
    return _M[key]


def work(spec):
    path = DRAWS / spec["set"] / ("%04d.json" % spec["i"])
    if path.exists():
        return 0
    m = model_for(spec["model"], spec.get("cut", 42.0))
    rec = m.run_two(spec["beta0"], spec["Pi0"], spec["t0"])
    rec.update(set=spec["set"], i=spec["i"], r=MODELS[spec["model"]]["r"], model=spec["model"], group=spec["group"],
               meta=spec["meta"], beta0=spec["beta0"], Pi0=spec["Pi0"], delta=MODELS[spec["model"]]["delta"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return 1


def run(groups, procs):
    S = []
    for g in groups:
        S += BUILD[g]()
    S.sort(key=lambda s: (s["i"] // 128, RANK[s["group"]], s["set"], s["i"]))
    todo = [s for s in S if not (DRAWS / s["set"] / ("%04d.json" % s["i"])).exists()]
    print("groups %s: specs %d, todo %d" % (groups, len(S), len(todo)), flush=True)
    t0, done = time.time(), 0
    if todo:
        with Pool(procs) as pool:
            for k in pool.imap_unordered(work, todo, chunksize=1):
                done += k
                if done % 200 == 0:
                    print("  done %d/%d  %.0f s" % (done, len(todo), time.time() - t0), flush=True)
    print("run finished: ran %d in %.0f s" % (done, time.time() - t0), flush=True)
    (OUT / ("run_done_%s.flag" % "_".join(groups))).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


# ------------------------------------------------------------------ smoke
def smoke():
    t_s = time.time()
    rep = {"runs": [], "mirror_check": []}
    m = model_for("xi030")
    picks = []
    for g in ("grid", "phase", "never", "adiab", "next"):
        S = BUILD[g]()
        seen = {}
        for s in S:
            if seen.get(s["set"], 0) < 1 and (g != "grid" or s["meta"]["tau"] in (0.0, 0.3, 1.1)):
                picks.append(s)
                seen[s["set"]] = seen.get(s["set"], 0) + 1
    for s in picks:
        q = model_for(s["model"]).run_two(s["beta0"], s["Pi0"], s["t0"])
        row = {"set": s["set"], "t0": s["t0"], **{k: q.get(k) for k in ("fate", "status", "runtime_s", "nfev", "H0", "constraint0",
                                                                        "t_bounce", "t_stop", "min_eigG_path")}}
        rep["runs"].append(row)
        print("%-18s t0=%+.4f fate=%-18s st=%-10s rt=%.2fs nfev=%d H0=%s c0=%s" % (
            s["set"], s["t0"], q.get("fate"), q.get("status"), q["runtime_s"], q["nfev"], C142.fm(q.get("H0")),
            C142.fm(q.get("constraint0"))), flush=True)
    # mirror check: direct backward integration from +tau vs forward run from -tau with (beta, -Pi)
    tau = 0.5
    s, d, _ = opt_widths("xi030", -tau)
    z = np.random.default_rng(99).standard_normal((8, 10))
    for i in range(8):
        b, p = np.sqrt(s) * z[i, :5], d / np.sqrt(s) * z[i, 5:]
        f = m.run_two(b.tolist(), (-p).tolist(), -tau)
        bb, ok = m.backward_beta(b.tolist(), p.tolist(), tau, -C142.T_END)
        dmax = float(np.max(np.abs(np.array(bb) - np.array(f["beta_plus"])))) if (f.get("beta_plus") and bb) else None
        rep["mirror_check"].append({"i": i, "fate_forward_mirror": f.get("fate"), "backward_ok": ok, "max_abs_dbeta": dmax})
        print("mirror i=%d fate=%s backward_ok=%s max|dbeta|=%s" % (i, f.get("fate"), ok, C142.fm(dmax)), flush=True)
    rep["runtime_s"] = time.time() - t_s
    (OUT / "smoke.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print("smoke done %.1f s" % rep["runtime_s"])


# ------------------------------------------------------------------ analysis
def load(name):
    d = DRAWS / name
    if not d.exists():
        return {}
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return {q["i"]: q for q in recs}


def seg_ok(q):
    """Forward run from t0 < 0: no pinch / collapse before t = 0 (segment [t0, 0])."""
    if q.get("survivor"):
        return True
    ts = q.get("t_stop")
    return bool(ts is not None and ts > 0)


def frac(k, n):
    return {"k": int(k), "n": int(n), "p": k / n if n else None, "wilson95": C142.wilson(k, n) if n else None}


def lin_rms(recs, model):
    kap = kap_of(model)
    t0 = float(next(iter(recs.values()))["t0"])
    MA = C142.lin_M(kap[0], t0)
    B0 = np.array([q["beta0"] for q in recs.values()])[:, ASP]
    P0 = np.array([q["Pi0"] for q in recs.values()])[:, ASP]
    return float(np.sqrt(np.mean((MA[0, 0] * B0 + MA[0, 1] * P0) ** 2)))


def crossing(xs, ps, level):
    """First x where p drops below level (linear interpolation between grid points); None if never."""
    for (x1, p1), (x2, p2) in zip(zip(xs, ps), zip(xs[1:], ps[1:])):
        if p1 >= level > p2:
            return float(x1 + (x2 - x1) * (p1 - level) / (p1 - p2))
    return None


def analyze():
    rng = np.random.default_rng(7)
    out = {"code": Path(__file__).name, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "models": MODELS, "sets": {}, "grid": [], "phase": [], "never": [], "adiab": [], "next": {}}
    names = sorted(p.name for p in DRAWS.iterdir() if p.is_dir()) if DRAWS.exists() else []
    cache = {nm: load(nm) for nm in names}
    for nm, recs in cache.items():
        if recs:
            o = C142.set_summary(list(recs.values()), rng)
            o["fates"] = dict(Counter(q.get("fate") for q in recs.values()))
            out["sets"][nm] = o
    # A1 grid
    for tau in TAUS:
        A, B = cache.get("g_tau%s_A" % tau_key(tau), {}), cache.get("g_tau%s_B" % tau_key(tau), {})
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        a = np.array([bool(A[i]["survivor"]) for i in ids])
        b = np.array([bool(B[i]["survivor"]) for i in ids])
        sa = np.array([seg_ok(A[i]) for i in ids])
        row = {"tau": tau, "n": len(ids), "two_branch": frac((a & b).sum(), len(ids)),
               "A_future_of_minus_tau": frac(a.sum(), len(ids)), "B_future_of_plus_tau": frac(b.sum(), len(ids)),
               "single_branch_plus_tau(B and seg[-tau,0] of A)": frac((b & sa).sum(), len(ids)),
               "single_branch_minus_tau(A)": frac(a.sum(), len(ids)),
               "sigma_lin_A": lin_rms({i: A[i] for i in ids}, "xi030"), "sigma_lin_B": lin_rms({i: B[i] for i in ids}, "xi030"),
               "fates_A": dict(Counter(A[i].get("fate") for i in ids)), "fates_B": dict(Counter(B[i].get("fate") for i in ids)),
               "rms_NL_A_survivors": (out["sets"]["g_tau%s_A" % tau_key(tau)].get("rms_NL") or {}).get("aspect")}
        out["grid"].append(row)
    if out["grid"]:
        xs = [r["tau"] for r in out["grid"]]
        for key in ("two_branch", "A_future_of_minus_tau"):
            ps = [r[key]["p"] for r in out["grid"]]
            lo = [r[key]["wilson95"][0] for r in out["grid"]]
            hi = [r[key]["wilson95"][1] for r in out["grid"]]
            out["t_star_" + key] = {str(L): {"central": crossing(xs, ps, L), "ci_from_wilson": [crossing(xs, lo, L), crossing(xs, hi, L)]}
                                    for L in (0.99, 0.9, 0.5)}
    # A2 phase family: A = +phi future, B = -phi future (mirror of the +phi past)
    for ph in PHIS:
        A, B = cache.get("ph%.1f_A" % ph, {}), cache.get("ph%.1f_B" % ph, {})
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        a = np.array([bool(A[i]["survivor"]) for i in ids])
        b = np.array([bool(B[i]["survivor"]) for i in ids])
        out["phase"].append({"phi": ph, "n": len(ids), "two_branch(+-phi)": frac((a & b).sum(), len(ids)),
                             "single_branch_future(+phi)": frac(a.sum(), len(ids)), "single_branch_future(-phi)": frac(b.sum(), len(ids)),
                             "sigma_lin_future(+phi)": lin_rms({i: A[i] for i in ids}, "xi030"),
                             "sigma_lin_future(-phi)": lin_rms({i: B[i] for i in ids}, "xi030"),
                             "fates_A": dict(Counter(A[i].get("fate") for i in ids)), "fates_B": dict(Counter(B[i].get("fate") for i in ids))})
    for mult in NEVER_S:
        A, B = cache.get("nv%.1f_out" % mult, {}), cache.get("nv%.1f_in" % mult, {})
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        a = np.array([bool(A[i]["survivor"]) for i in ids])
        b = np.array([bool(B[i]["survivor"]) for i in ids])
        out["never"].append({"mult": mult, "params": never_params(mult)[3], "n": len(ids), "two_branch": frac((a & b).sum(), len(ids)),
                             "future_outgoing": frac(a.sum(), len(ids)), "future_incoming": frac(b.sum(), len(ids)),
                             "sigma_lin_out": lin_rms({i: A[i] for i in ids}, "xi030"), "sigma_lin_in": lin_rms({i: B[i] for i in ids}, "xi030"),
                             "fates_out": dict(Counter(A[i].get("fate") for i in ids)), "fates_in": dict(Counter(B[i].get("fate") for i in ids)),
                             "rms_NL_in_survivors": (out["sets"].get("nv%.1f_in" % mult, {}).get("rms_NL") or {}).get("aspect")})
    for t_in in ADIAB_TIN:
        A = cache.get("ad_tin%.2f" % t_in, {})
        if not A:
            continue
        a = np.array([bool(q["survivor"]) for q in A.values()])
        out["adiab"].append({"t_in": t_in, "n": len(A), "future": frac(a.sum(), len(A)), "sigma_lin": lin_rms(A, "xi030"),
                             "fates": dict(Counter(q.get("fate") for q in A.values())),
                             "rms_NL_survivors": (out["sets"].get("ad_tin%.2f" % t_in, {}).get("rms_NL") or {}).get("aspect")})
    out["width"] = []
    for tau in WIDTH_TAU:
        G = cache.get("g_tau%s_A" % tau_key(tau), {})
        ids0 = [i for i in range(N_W) if i in G]
        base = frac(sum(bool(G[i]["survivor"]) for i in ids0), len(ids0)) if ids0 else None
        for x in WIDTH_X:
            W = cache.get("w_tau%s_x%.1f" % (tau_key(tau), x), {})
            if not W:
                continue
            out["width"].append({"tau": tau, "x": x, "n": len(W), "future": frac(sum(bool(q["survivor"]) for q in W.values()), len(W)),
                                 "grid_optimum_same_z_first256": base, "sigma_lin": lin_rms(W, "xi030"),
                                 "fates": dict(Counter(q.get("fate") for q in W.values()))})
    for nm in [n for n in names if n.startswith("nx_")]:
        o = out["sets"][nm]
        out["next"][nm] = {"n": o["N"], "survival": o["survival"], "wilson95": o["survival_wilson95"], "fates": o["fates"],
                           "sigma_lin": o["rms_lin_all"]["aspect"], "rms_NL_survivors": (o.get("rms_NL") or {}).get("aspect")}
    sv = [s for s in NX_AMP if "nx_amp_s%d" % s in out["next"]]
    fr = [out["next"]["nx_amp_s%d" % s]["survival"] for s in sv]
    out["next_amp_threshold"] = {"s": sv, "survival": fr, "s90": C142.threshold(sv, fr, 0.9), "s99": C142.threshold(sv, fr, 0.99),
                                 "s50": C142.threshold(sv, fr, 0.5)}
    # next-order t_* from the xi=0.30 survival-vs-sigma_lin curve (D114 universality), sigma_lin = sqrt(G_18(tau)) floor_next
    try:
        lin = json.loads((OUT / "linear" / "C153_linear.json").read_text(encoding="utf-8"))
        fl_nx = lin["next_order_floor"]["xi028_kappa18.0(next)"]["sigma_floor_C76conv"]
        rows = [r for r in lin["G_tables"]["E_next18.0"]["rows"] if r["tau"] <= 1.1 + 1e-9]
        xs_ = np.array([r["sigma_lin_A"] for r in out["grid"]])
        ps_ = np.array([r["two_branch"]["p"] for r in out["grid"]])
        o_ = np.argsort(xs_)
        est = [{"tau": 0.0, "sigma_lin": fl_nx, "S2_est": float(np.interp(fl_nx, xs_[o_], ps_[o_]))}]
        est += [{"tau": r["tau"], "sigma_lin": float(np.sqrt(r["G_two_branch"]) * fl_nx),
                 "S2_est": float(np.interp(np.sqrt(r["G_two_branch"]) * fl_nx, xs_[o_], ps_[o_]))} for r in rows]
        est.sort(key=lambda r: r["tau"])
        out["next_t_star_from_universal_curve"] = {"floor_next_C76conv": fl_nx, "rows": est,
                                                    "t99": crossing([r["tau"] for r in est], [r["S2_est"] for r in est], 0.99),
                                                    "t90": crossing([r["tau"] for r in est], [r["S2_est"] for r in est], 0.9),
                                                    "t50": crossing([r["tau"] for r in est], [r["S2_est"] for r in est], 0.5)}
        print("NEXT t_* (universal-curve estimate):", json.dumps({k: v for k, v in out["next_t_star_from_universal_curve"].items() if k != "rows"}))
    except Exception as exc:                                  # analysis must not fail on this optional block
        out["next_t_star_from_universal_curve"] = {"error": repr(exc)}
    (OUT / "C153_nl_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for r in out["grid"]:
        print("tau=%.3f n=%d two=%.4f %s A=%.4f B=%.4f single+=%.4f sigA=%.4f sigB=%.4f rmsNL_A=%s fatesA=%s fatesB=%s" % (
            r["tau"], r["n"], r["two_branch"]["p"], [round(x, 3) for x in r["two_branch"]["wilson95"]], r["A_future_of_minus_tau"]["p"],
            r["B_future_of_plus_tau"]["p"], r["single_branch_plus_tau(B and seg[-tau,0] of A)"]["p"], r["sigma_lin_A"], r["sigma_lin_B"],
            C142.fm(r["rms_NL_A_survivors"]), r["fates_A"], r["fates_B"]))
    for k in ("t_star_two_branch", "t_star_A_future_of_minus_tau"):
        if k in out:
            print(k, json.dumps(out[k]))
    for key in ("phase", "never", "adiab", "width"):
        for r in out[key]:
            print(key.upper(), json.dumps({k: v for k, v in r.items() if k != "params"}, default=float))
    for nm, o in out["next"].items():
        print("NEXT %-14s n=%d surv=%.4f %s sig_lin=%.4f rmsNL=%s fates=%s" % (nm, o["n"], o["survival"], [round(x, 3) for x in o["wilson95"]],
                                                                         o["sigma_lin"], C142.fm(o["rms_NL_survivors"]), o["fates"]))
    print("NEXT amp threshold", json.dumps(out["next_amp_threshold"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run", "analyze"])
    ap.add_argument("--groups", nargs="+", default=["grid", "next", "phase", "never", "adiab"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("DNEC_WORKERS", "4")))  # package: --workers
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "smoke":
        smoke()
    elif a.mode == "run":
        run(a.groups, a.procs)  # package: no cap (each draw is independent; results do not depend on the worker count)
    else:
        analyze()


if __name__ == "__main__":
    main()
