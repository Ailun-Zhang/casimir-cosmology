# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/shape_nonlinear_threshold/C142_shape_nonlinear_threshold.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] D114 nonlinear survival threshold: turning-point amplitude scan s = 1...40 of the natural state, R1 real states at t_x < 0, convergence controls, regression against C76; class NLR (C76 model + fates). Driver of r2/shape_nonlinear_threshold/run01 and shared library (imported by C153_nl, C160_plateau). Paper: PRD Sec. VIII E (sec:s-nonlinear), App. app:nonlinear (tab:s-amp). Header erratum: research tree 04_calculations/code/20261014_claude_v12_r3/cleanup_t12/C142_HEADER_ERRATUM.md (wording only).

C142 (Claude v12 round 2, direction D1 'shape-nonlinear'; derivation D114): nonlinear survival threshold of the
SM (xi = 0.30, i.e. C76's r = 0.299897) shape fossil with the full nonlinear volume + five-shape-moduli equations,
twisted Ewald Casimir force and v11's leading cosh background (units h = 1, L_b = 1, C60 normalisation).

Copied code (unchanged; sha256 in PROVENANCE.txt; imported from THIS directory only):
    C76_nonlinear_zero_mode.py   <- 04_calculations/code/20261011_claude_followup/
    C60_nonlinear_integrator.py  <- 04_calculations/code/20261009_claude_R5/nonlinear_SM/
    C60_twisted_ewald.py         <- 04_calculations/code/20261009_claude_R5/nonlinear_SM/
New here (class NLR(C76.NLQ)):
  * start at any t0 <= 0: G0, M0 from (beta, Pi) by C60's 'on' map; volume clock lr0 = (1/2) ln cosh 2t0 and
    H0 = -sqrt(1 - (Z/Z0) e^{-4 lr0} + S0/6) from the Hamiltonian constraint, S0 = e^{-6 lr0} tr((G0^-1 M0)^2);
  * turning-point data (t0 = 0): exact existence of the large-volume root, s0 <= 12 (zeta/3)^{3/2}; otherwise the
    draw cannot sit at a bounce ('no_turning_point': shear energy too large, the exact form of S_b < 12 h^2);
  * events: H = 0 crossings (bounce = crossing with dH/dt = 2 - S/6 > 0), collapse ln(L/L_b) < -1.5, pinch
    min eig G < 1e-6 (C60's 1e-3 criterion is kept as survivor_C60), nfev cap;
  * fates: survive | no_turning | no_bounce | pinch_after_bounce | recollapse | stalled | other.

    python C142_shape_nonlinear_threshold.py smoke
    python C142_shape_nonlinear_threshold.py run --group all --procs 5      (resumable: one JSON per draw)
    python C142_shape_nonlinear_threshold.py analyze
Outputs: results/20261013_claude_v12_r2/shape_nonlinear_threshold/run01/ (draws/<set>/NNNN.json, C142_summary.json).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import time
import hashlib
import argparse
from collections import Counter
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C76_nonlinear_zero_mode as C76                                    # noqa: E402  (copy, unchanged)
from C60_nonlinear_integrator import unpack, pack, geometry, C32         # noqa: E402  (copy, unchanged)
from C60_twisted_ewald import BASIS, ControlExit, beta_from_shape        # noqa: E402  (copy, unchanged)

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r2/shape_nonlinear_threshold/run01")
DRAWS = OUT / "draws"
C76_DRAWS = WP.wdir("fu/C76/draws")
R_SM = C76.R_ZM
TYPE = C76.TYPE
ASP, ANG = TYPE == 0, TYPE == 1
# pinch stop = C60's eig(G) < 1e-3; a deep subsample (eig < 1e-6) checks that these are finite-time runaways
# (smoke test: going from 1e-3 to 1e-6 costs ~35 s per draw because the Ewald boxes grow like eig^-1/2)
EIG_STOP, EIG_DEEP, LR_COLL, T_END, NFEV_MAX = 1e-3, 1e-6, -1.5, 12.0, 60000
AMP_S = (1, 2, 3, 4, 5, 7, 10, 20, 40)
N_AMP, N_R1, N_VAR, N_CONV, N_DEEP = 256, 512, 128, 48, 40
N_R1_TX = {"-0.5": 512, "-1.0": 512, "F": 512, "-0.25": 256, "-2.0": 256}
TX_LIST = ("-0.5", "-1.0", "F", "-0.25", "-2.0")          # "F" = freeze start -0.5 arsinh sqrt(kappa_E)
SEED_R1 = {"-0.25": 20261301, "-0.5": 20261302, "-1.0": 20261303, "F": 20261304, "-2.0": 20261305}
GRID = np.geomspace(0.01, 2.0, 40)


class NfevLimit(RuntimeError):
    pass


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else None


def fm(x):
    return "-" if x is None else "%.4g" % x


_LIN = {}


def lin_M(kappa, t0, tf=14.0):
    """Linear transfer (beta, Pi)(t0) -> (beta, Pi)(tf), leading cosh background, C60 normalisation Pi = L^3 beta'/C32."""
    key = (round(float(kappa), 12), round(float(t0), 12), tf)
    if key not in _LIN:
        def rhs(t, y):
            c = np.cosh(2 * t)
            return [C32 * y[1] / c ** 1.5, kappa * c ** -0.5 * y[0] / C32]
        cols = []
        for y0 in ([1.0, 0.0], [0.0, 1.0]):
            s = solve_ivp(rhs, (t0, tf), y0, method="DOP853", rtol=1e-13, atol=1e-15)
            cols.append(s.y[:, -1])
        _LIN[key] = np.array(cols).T
    return _LIN[key]


def tx_value(tx, r):
    if tx == "F":
        return -0.5 * float(np.arcsinh(np.sqrt(C76.widths(r)[0][0])))
    return float(tx)


def r1_info(r, t0):
    """Real (uncorrelated, minimum-uncertainty) Gaussian at t0 with the width minimising the outgoing variance:
    s_opt = |M12/M11| delta, Var_out = 2|M11 M12| delta; G = |M11 M12|(t0)/|M11 M12|(0) (D109 Prop. 1)."""
    kap, delta, _ = C76.widths(r)
    info = {"t0": t0, "delta": float(delta)}
    for name, k in (("aspect", kap[0]), ("angle", kap[2])):
        M, M0 = lin_M(k, t0), lin_M(k, 0.0)
        p, p0 = abs(M[0, 0] * M[0, 1]), abs(M0[0, 0] * M0[0, 1])
        so = abs(M[0, 1] / M[0, 0]) * delta
        info[name] = {"kappa": float(k), "M": M.tolist(), "s_opt": float(so), "beta_rms_opt": float(np.sqrt(so)),
                      "Pi_rms_opt": float(delta / np.sqrt(so)), "G": float(p / p0),
                      "sigma_lin_opt": float(np.sqrt(2 * p * delta)), "sigma_floor": float(np.sqrt(2 * p0 * delta))}
    return info


class NLR(C76.NLQ):
    """C76 model + start at arbitrary t0 <= 0, bounce/collapse events and fates."""

    def __init__(self, r, cut=42.0, rtol=1e-11, atol=1e-13, eig_stop=EIG_STOP):
        super().__init__(r, cut=cut, rtol=rtol, atol=atol)
        self.eig_stop = eig_stop
        self.maxH, self.last = -np.inf, None

    def rhs(self, t, y):
        if self.nfev > NFEV_MAX:
            raise NfevLimit("nfev > %d" % NFEV_MAX)
        self.last = (float(t), np.array(y))
        if y[1] > self.maxH:
            self.maxH = float(y[1])
        return super().rhs(t, y)

    def shape_state(self, beta0, Pi0):
        B = np.einsum("a,aij->ij", np.asarray(beta0, float), BASIS)
        G0, Gh = expm(2 * B), expm(B)
        M0 = Gh @ np.einsum("a,aij->ij", C32 * np.asarray(Pi0, float), BASIS) @ Gh
        M0 = 0.5 * (M0 + M0.T)
        Gi0 = np.linalg.inv(G0)
        s0 = float(np.trace((Gi0 @ M0) @ (Gi0 @ M0)))
        return G0, M0, s0, float(self.cas(G0)[0] / self.Z0)

    def run_one(self, beta0, Pi0, t0):
        t_start = time.perf_counter()
        self.nfev, self.maxH, self.last = 0, -np.inf, None
        rec = {"t0": float(t0), "T": T_END, "rtol": self.rtol, "atol": self.atol, "cut": self.cas.cut,
               "eig_stop": self.eig_stop}
        try:
            G0, M0, s0, zeta = self.shape_state(beta0, Pi0)
            ev0 = np.linalg.eigvalsh(G0)
            rec.update(s0=s0, zeta0=zeta, lnaniso0=float(0.5 * np.log(ev0[-1] / ev0[0])))
            if t0 == 0.0:
                smax = 12.0 * (zeta / 3.0) ** 1.5 if zeta > 0 else 0.0      # zeta <= 0: no turning point at all
                rec["s0_over_smax"] = s0 / smax if smax > 0 else float("inf")
                if zeta <= 0 or s0 > smax:
                    rec.update(status="no_turning_point", fate="no_turning", bounced=False, survivor=False,
                               survivor_C60=False)
                    return self._fin(rec, t_start)
                y0, info = self.initial_state(beta0, Pi0, "on")
                rec.update(info)
            else:
                lr0 = 0.5 * float(np.log(np.cosh(2 * t0)))
                q = 1.0 - zeta * np.exp(-4 * lr0) + s0 * np.exp(-6 * lr0) / 6.0
                rec.update(lr0=lr0, S0=float(s0 * np.exp(-6 * lr0)), H0_background=float(np.tanh(2 * t0)))
                if q < 0:
                    rec.update(status="forbidden_start", fate="other", bounced=False, survivor=False, survivor_C60=False)
                    return self._fin(rec, t_start)
                y0 = pack(lr0, -np.sqrt(q), G0, M0)
                rec.update(H0=float(y0[1]), constraint0=self.constraint(y0))
            sol = self._integrate(y0, t0)
            self._classify(sol, rec)
        except (ControlExit, NfevLimit) as exc:
            rec.update(status="control_exit" if isinstance(exc, ControlExit) else "nfev_limit", reason=str(exc))
            self._exc_fate(rec)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rec.update(status="numerical_error", reason=str(exc))
            self._exc_fate(rec)
        except Exception as exc:                                           # never let one draw kill the pool
            rec.update(status="error", reason=repr(exc))
            self._exc_fate(rec)
        return self._fin(rec, t_start)

    def _fin(self, rec, t_start):
        rec["nfev"] = int(self.nfev)
        rec["runtime_s"] = float(time.perf_counter() - t_start)
        return rec

    def _integrate(self, y0, t0):
        es = self.eig_stop

        def pinch(t, y):
            return float(np.linalg.eigvalsh(unpack(y)[2])[0]) - es
        pinch.terminal, pinch.direction = True, -1

        def coll(t, y):
            return y[0] - LR_COLL
        coll.terminal, coll.direction = True, -1

        def hzero(t, y):
            return y[1]
        hzero.terminal, hzero.direction = False, 0
        return solve_ivp(self.rhs, (t0, T_END), y0, method="DOP853", rtol=self.rtol, atol=self.atol,
                         max_step=self.max_step, events=(pinch, coll, hzero), dense_output=True)

    def _classify(self, sol, rec):
        rec["t_end"], rec["nsteps"] = float(sol.t[-1]), int(len(sol.t))
        ye = sol.y[:, -1]
        rec["H_end"], rec["lr_end"] = float(ye[1]), float(ye[0])
        cr = []
        for te, ys in zip(sol.t_events[2], sol.y_events[2]):
            lr, H, G, M = unpack(ys)
            S = self.pieces(lr, G, M)[0]
            cr.append({"t": float(te), "lr": float(lr), "S_over_12": float(S / 12.0), "Hdot": float(2 - 2 * H * H - S / 6),
                       "beta": beta_from_shape(G).tolist()})
        rec["H_crossings"] = cr[:6]
        bo = [c for c in cr if c["Hdot"] > 0]
        rec["bounced"] = bool(bo)
        if bo:
            rec.update(t_bounce=bo[0]["t"], lr_bounce=bo[0]["lr"], S_over_12_bounce=bo[0]["S_over_12"],
                       beta_bounce=bo[0]["beta"])
        mins, lnl, mb = [], [], []
        for k in range(sol.y.shape[1]):
            G = unpack(sol.y[:, k])[2]
            ev = np.linalg.eigvalsh(G)
            mins.append(float(ev[0]))
            if ev[0] > 0:
                lnl.append(float(0.5 * np.log(ev[-1] / ev[0])))
                mb.append(float(np.max(np.abs(beta_from_shape(G)))))
        rec["min_eigG_path"] = float(min(mins))
        rec["max_lnaniso_path"] = max(lnl) if lnl else None
        rec["max_abs_beta_path"] = max(mb) if mb else None
        if len(sol.t_events[0]):
            rec.update(status="pinch", t_stop=float(sol.t_events[0][0]))
        elif len(sol.t_events[1]):
            rec.update(status="collapse", t_stop=float(sol.t_events[1][0]))
        elif sol.status != 0 or abs(sol.t[-1] - T_END) > 1e-9:
            rec.update(status="solver_fail", reason=str(sol.message))
        else:
            rec["status"] = "complete"
        surv = False
        if rec["status"] == "complete":
            ys = sol.y
            cons = [abs(self.constraint(ys[:, k])) for k in range(ys.shape[1])]
            detd = [abs(np.linalg.det(unpack(ys[:, k])[2]) - 1.0) for k in range(ys.shape[1])]
            Graw = unpack(ye)[2]
            G10 = unpack(sol.sol(10.0))[2]
            Gp = Graw / np.linalg.det(Graw) ** (1 / 3)
            evp = np.linalg.eigvalsh(Gp)
            rec.update(gate_constraint_max=float(max(cons)), gate_detG_minus1_max=float(max(detd)),
                       gate_G_change_10_to_12=float(np.max(np.abs(Graw - G10))),
                       gate_G_change_10_to_12_rel=float(np.max(np.abs(Graw - G10)) / np.max(np.abs(Graw))),
                       G_plus=Gp.tolist(), beta_plus=beta_from_shape(Gp).tolist(), geometry_plus=geometry(Gp),
                       lnaniso_plus=float(0.5 * np.log(evp[-1] / evp[0])))
            surv = bool(rec["bounced"] and rec["H_end"] > 0.9 and rec["min_eigG_path"] > self.eig_stop)
        rec["survivor"] = surv
        rec["survivor_C60"] = bool(surv and rec["min_eigG_path"] > 1e-3)
        if surv:
            rec["fate"] = "survive"
        elif not rec["bounced"]:
            rec["fate"] = "no_bounce"
        elif rec["status"] == "pinch":
            rec["fate"] = "pinch_after_bounce"
        elif rec["status"] == "collapse":
            rec["fate"] = "recollapse"
        else:
            rec["fate"] = "other"

    def _exc_fate(self, rec):
        b = bool(rec.get("t0", 1.0) == 0.0 or self.maxH > 1e-6)
        rec.update(bounced=b, survivor=False, survivor_C60=False)
        if self.last is not None:
            t, y = self.last
            rec.update(t_stop=t, lr_last=float(y[0]), H_last=float(y[1]))
            try:
                rec["min_eig_last"] = float(np.linalg.eigvalsh(unpack(y)[2])[0])
            except Exception:
                pass
        if rec["status"] == "control_exit":
            rec["fate"] = "pinch_after_bounce" if b else "no_bounce"
        elif rec["status"] == "nfev_limit":
            rec["fate"] = "stalled"
        else:
            rec["fate"] = "other"


# ------------------------------------------------------------------ draws
def amp_specs(r, n, s, name, **kw):
    base = C76.specs("hi", r)[:n]
    return [dict(set=name, i=b["i"], r=r, t0=0.0, beta0=(s * np.asarray(b["beta0"])).tolist(),
                 Pi0=(s * np.asarray(b["Pi0"])).tolist(), meta={"s": s}, **kw) for b in base]


def r1_specs(r, tx, n, name, seed, **kw):
    t0 = tx_value(tx, r)
    info = r1_info(r, t0)
    _, delta, _ = C76.widths(r)
    s = np.array([info["aspect"]["s_opt"] if t == 0 else info["angle"]["s_opt"] for t in TYPE])
    z = np.random.default_rng(seed).standard_normal((max(n, N_R1), 10))[:n]
    return [dict(set=name, i=i, r=r, t0=t0, beta0=(np.sqrt(s) * z[i, :5]).tolist(),
                 Pi0=(delta / np.sqrt(s) * z[i, 5:]).tolist(), meta={"tx": tx}, **kw) for i in range(n)]


def grid_specs(r):
    S = []
    dirs = {"slab": np.array([0, 1.0, 0, 0, 0]), "rod": np.array([0, -1.0, 0, 0, 0]), "mid": np.array([1.0, 0, 0, 0, 0])}
    for kind in ("B", "P"):
        for dn, e in dirs.items():
            for i, a in enumerate(GRID):
                v, z0 = (a * e).tolist(), [0.0] * 5
                S.append(dict(set="grid%s_%s" % (kind, dn), i=i, r=r, t0=0.0, beta0=v if kind == "B" else z0,
                              Pi0=z0 if kind == "B" else v, meta={"a": float(a)}))
    return S


def specs_core():
    base = C76.specs("main", R_SM)
    S = [dict(set="reg_c76", i=b["i"], r=R_SM, t0=0.0, beta0=b["beta0"], Pi0=b["Pi0"], engine="c76") for b in base]
    S += [dict(set="reg_c142", i=b["i"], r=R_SM, t0=0.0, beta0=b["beta0"], Pi0=b["Pi0"]) for b in base]
    for tx in TX_LIST:
        S += r1_specs(R_SM, tx, N_R1_TX[tx], "r1_tx%s" % tx, SEED_R1[tx])
    for s in AMP_S:
        S += amp_specs(R_SM, N_AMP, s, "amp_s%d" % s)
    return S + grid_specs(R_SM)


def specs_extra():
    S = []
    kw = dict(eig_stop=EIG_DEEP)
    S += amp_specs(R_SM, N_DEEP, 5, "deep_amp_s5", **kw)
    S += r1_specs(R_SM, "-1.0", N_DEEP, "deep_r1_tx-1.0", SEED_R1["-1.0"], **kw)
    for lab, kw in (("cut60", dict(cut=60.0)), ("rtol9", dict(rtol=1e-9, atol=1e-11))):
        S += amp_specs(R_SM, N_CONV, 5, "conv_%s_amp_s5" % lab, **kw)
        S += r1_specs(R_SM, "-1.0", N_CONV, "conv_%s_r1_tx-1.0" % lab, SEED_R1["-1.0"], **kw)
    for r, lab in ((0.0, "r0"), (0.5, "r05")):
        for s in (3, 5, 10):
            S += amp_specs(r, N_VAR, s, "var_%s_amp_s%d" % (lab, s))
        for tx in ("-0.5", "-1.0", "F"):
            S += r1_specs(r, tx, N_VAR, "var_%s_r1_tx%s" % (lab, tx), SEED_R1[tx])
    return S


_MODELS = {}


def model_for(spec):
    eng = spec.get("engine", "c142")
    key = (eng, spec["r"], spec.get("cut", 42.0), spec.get("rtol", 1e-11), spec.get("atol", 1e-13),
           spec.get("eig_stop", EIG_STOP))
    if key not in _MODELS:
        _MODELS[key] = C76.NLQ(spec["r"]) if eng == "c76" else NLR(spec["r"], cut=key[2], rtol=key[3], atol=key[4],
                                                                     eig_stop=key[5])
    return _MODELS[key]


def work(spec):
    path = DRAWS / spec["set"] / ("%04d.json" % spec["i"])
    if path.exists():
        return 0
    m = model_for(spec)
    if spec.get("engine") == "c76":
        rec = m.forward(spec["beta0"], spec["Pi0"], ic="on", label="%s_%d" % (spec["set"], spec["i"]))
        rec["t0"] = 0.0
    else:
        rec = m.run_one(spec["beta0"], spec["Pi0"], spec["t0"])
    rec.update(set=spec["set"], i=spec["i"], r=spec["r"], meta=spec.get("meta", {}),
               beta0=[float(x) for x in spec["beta0"]], Pi0=[float(x) for x in spec["Pi0"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return 1


def run(group, procs):
    S = (specs_core() if group in ("core", "all") else []) + (specs_extra() if group in ("extra", "all") else [])
    for name in sorted(set(s["set"] for s in S)):
        (DRAWS / name).mkdir(parents=True, exist_ok=True)
    todo = [s for s in S if not (DRAWS / s["set"] / ("%04d.json" % s["i"])).exists()]
    print("group %s: specs %d, todo %d" % (group, len(S), len(todo)), flush=True)
    t0, done = time.time(), 0
    if todo:
        with Pool(procs) as pool:
            for k in pool.imap_unordered(work, todo, chunksize=2):
                done += k
                if done % 250 == 0:
                    print("  done %d/%d  %.0f s" % (done, len(todo), time.time() - t0), flush=True)
    print("run %s finished: ran %d in %.0f s" % (group, done, time.time() - t0), flush=True)


# ------------------------------------------------------------------ analysis
def load(name):
    d = DRAWS / name
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else None


def set_summary(recs, rng):
    r, t0 = recs[0]["r"], float(recs[0].get("t0", 0.0))
    kap, delta, _ = C76.widths(r)
    MA, MT = lin_M(kap[0], t0), lin_M(kap[2], t0)
    m11, m12 = np.where(ASP, MA[0, 0], MT[0, 0]), np.where(ASP, MA[0, 1], MT[0, 1])
    B0, P0 = np.array([q["beta0"] for q in recs]), np.array([q["Pi0"] for q in recs])
    BL = m11 * B0 + m12 * P0
    sv = np.array([bool(q.get("survivor")) for q in recs])
    n, k = len(recs), int(sv.sum())
    fates = Counter(q.get("fate", "survive" if q.get("survivor") else q["status"]) for q in recs)
    o = {"N": n, "r": r, "t0": t0, "kappa_E": float(kap[0]), "n_survivors": k, "survival": k / n,
         "survival_wilson95": wilson(k, n),
         "survival_C60": float(np.mean([bool(q.get("survivor_C60", q.get("survivor"))) for q in recs])),
         "fates": dict(fates), "status_counts": dict(Counter(q["status"] for q in recs)),
         "rms_lin_all": {"aspect": rms(BL[:, ASP]), "angle": rms(BL[:, ANG])},
         "runtime_mean_s": float(np.mean([q["runtime_s"] for q in recs])),
         "runtime_max_s": float(np.max([q["runtime_s"] for q in recs]))}
    if k:
        S = [q for q, f in zip(recs, sv) if f]
        BP, BLs = np.array([q["beta_plus"] for q in S]), BL[sv]
        num, den = (BP[:, ASP] ** 2).sum(1), (BLs[:, ASP] ** 2).sum(1)
        idx = rng.integers(0, k, (1000, k))
        vb = num[idx].sum(1) / den[idx].sum(1)
        lna = [q.get("lnaniso_plus", np.nan) for q in S]
        o.update({"rms_NL": {"aspect": rms(BP[:, ASP]), "angle": rms(BP[:, ANG])},
                  "rms_lin_survivors": {"aspect": rms(BLs[:, ASP]), "angle": rms(BLs[:, ANG])},
                  "var_ratio_aspect": float(num.sum() / den.sum()), "var_ratio_aspect_boot95": [pct(vb, 2.5), pct(vb, 97.5)],
                  "var_ratio_angle": float((BP[:, ANG] ** 2).sum() / (BLs[:, ANG] ** 2).sum()),
                  "paired_rel_rms": float(np.sqrt(((BP - BLs) ** 2).sum() / (BLs ** 2).sum())),
                  "lnaniso_plus": {"median": pct(lna, 50), "p90": pct(lna, 90), "max": pct(lna, 100)},
                  "ln_side_ratio_plus_median": pct([q["geometry_plus"]["ln_Lmax_over_Lmin"] for q in S], 50),
                  "gate_constraint_max": float(max(q.get("gate_constraint_max", 0.0) for q in S)),
                  "gate_G_change_10_to_12_rel_max": float(max(q.get("gate_G_change_10_to_12_rel", 0.0) for q in S)),
                  "min_eigG_path_min": float(min(q["min_eigG_path"] for q in S))})
    b = [q for q in recs if q.get("S_over_12_bounce") is not None]
    if b:
        o["bounce"] = {"n": len(b), "S_over_12_median": pct([q["S_over_12_bounce"] for q in b], 50),
                       "S_over_12_p99": pct([q["S_over_12_bounce"] for q in b], 99),
                       "S_over_12_max": pct([q["S_over_12_bounce"] for q in b], 100),
                       "t_bounce_median": pct([q["t_bounce"] for q in b], 50),
                       "lr_bounce_median": pct([q["lr_bounce"] for q in b], 50),
                       "lr_bounce_min": pct([q["lr_bounce"] for q in b], 0)}
    s0r = [q["s0_over_smax"] for q in recs if q.get("s0_over_smax") is not None]
    if s0r:
        o["s0_over_smax"] = {"median": pct(s0r, 50), "p90": pct(s0r, 90), "frac_gt1": float(np.mean(np.array(s0r) > 1))}
    sf = [q["S0"] / 6.0 / q["H0"] ** 2 for q in recs if q.get("S0") is not None and q.get("H0")]
    if sf:
        o["shear_energy_fraction_at_t0"] = {"median": pct(sf, 50), "max": pct(sf, 100)}
    mb = [q["max_abs_beta_path"] for q in recs if q.get("max_abs_beta_path") is not None]
    if mb:
        o["max_abs_beta_path"] = {"median": pct(mb, 50), "p90": pct(mb, 90)}
    tp = [q["t_stop"] for q in recs if q.get("fate") == "pinch_after_bounce" and q.get("t_stop") is not None]
    if tp:
        o["pinch_time"] = {"median": pct(tp, 50), "min": pct(tp, 0), "max": pct(tp, 100)}
    return o


def grid_summary(recs):
    rows = []
    for q in sorted(recs, key=lambda q: q["i"]):
        bp = q.get("beta_plus")
        rows.append({"a": q["meta"]["a"], "fate": q.get("fate"), "status": q["status"],
                     "beta_plus_norm": float(np.linalg.norm(bp)) if bp else None,
                     "lnaniso_plus": q.get("lnaniso_plus"), "S_over_12_bounce": q.get("S_over_12_bounce"),
                     "s0_over_smax": q.get("s0_over_smax"), "min_eigG_path": q.get("min_eigG_path")})
    return {"rows": rows, "first_fail_a": next((x["a"] for x in rows if x["fate"] != "survive"), None),
            "last_survive_a": max((x["a"] for x in rows if x["fate"] == "survive"), default=None)}


def regression(rng):
    rep = {}

    def stored(tag, i):
        p = C76_DRAWS / ("%s_%04d.json" % (tag, i))
        return json.loads(p.read_text()) if p.exists() else None

    def dmax(A, ref_fn):
        d = []
        for q in A:
            s = ref_fn(q["i"])
            if s is not None and q.get("beta_plus") and s.get("beta_plus"):
                d.append(float(np.max(np.abs(np.array(q["beta_plus"]) - np.array(s["beta_plus"])))))
        return (max(d) if d else None), len(d)
    a, b, c = load("reg_c76"), load("reg_c142"), load("amp_s1")
    amap = {q["i"]: q for q in a}
    rep["reg_c76_vs_stored_C76_main"] = dmax(a, lambda i: stored("main", i))
    rep["reg_c142_vs_reg_c76"] = dmax(b, lambda i: amap.get(i))
    rep["amp_s1_vs_stored_C76_hi"] = dmax(c, lambda i: stored("hi", i))
    if a:
        rep["reg_c76_summary"] = set_summary(a, rng)
    rep["C76_reference"] = {"main_var_ratio_aspect": 1.0513592097468514, "hi_var_ratio_aspect": 1.042557977826608,
                            "ind_var_ratio_aspect": 1.0451399158067842, "sigma_lin_analytic_aspect": 0.10406007919151665}
    return rep


def threshold(svals, fr, level):
    for (s1, f1), (s2, f2) in zip(zip(svals, fr), zip(svals[1:], fr[1:])):
        if f1 >= level > f2:
            return float(np.exp(np.log(s1) + (np.log(s2) - np.log(s1)) * (f1 - level) / (f1 - f2)))
    return None


def analyze():
    rng = np.random.default_rng(7)
    out = {"code": Path(__file__).name, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "settings": {"eig_stop": EIG_STOP, "lr_collapse": LR_COLL, "T_end": T_END, "nfev_max": NFEV_MAX, "r_SM": R_SM},
           "regression": regression(rng), "r1_linear": {}, "sets": {}}
    for tx in TX_LIST:
        out["r1_linear"][tx] = r1_info(R_SM, tx_value(tx, R_SM))
    for r, lab in ((0.0, "r0"), (0.5, "r05")):
        for tx in ("-1.0", "F"):
            out["r1_linear"]["%s_%s" % (lab, tx)] = r1_info(r, tx_value(tx, r))
    names = sorted(p.name for p in DRAWS.iterdir() if p.is_dir()) if DRAWS.exists() else []
    for nm in names:
        recs = load(nm)
        if recs:
            out["sets"][nm] = grid_summary(recs) if nm.startswith("grid") else set_summary(recs, rng)
    sv = [s for s in AMP_S if "amp_s%d" % s in out["sets"]]
    fr = [out["sets"]["amp_s%d" % s]["survival"] for s in sv]
    out["amp_threshold"] = {"s": sv, "survival": fr, "s50": threshold(sv, fr, 0.5), "s90": threshold(sv, fr, 0.9),
                            "s99": threshold(sv, fr, 0.99)}
    pairs = {"deep_amp_s5": "amp_s5", "deep_r1_tx-1.0": "r1_tx-1.0", "conv_cut60_amp_s5": "amp_s5",
             "conv_cut60_r1_tx-1.0": "r1_tx-1.0", "conv_rtol9_amp_s5": "amp_s5", "conv_rtol9_r1_tx-1.0": "r1_tx-1.0"}
    out["paired_checks"] = {}
    for a_, b_ in pairs.items():
        A, B = {q["i"]: q for q in load(a_)}, {q["i"]: q for q in load(b_)}
        common = sorted(set(A) & set(B))
        if not common:
            continue
        db = [float(np.max(np.abs(np.array(A[i]["beta_plus"]) - np.array(B[i]["beta_plus"])))) for i in common
              if A[i].get("beta_plus") and B[i].get("beta_plus")]
        dt = [A[i]["t_stop"] - B[i]["t_stop"] for i in common if A[i].get("fate") == "pinch_after_bounce"
              and A[i].get("t_stop") is not None and B[i].get("t_stop") is not None]
        out["paired_checks"][a_] = {
            "vs": b_, "n": len(common), "fate_agreement": float(np.mean([A[i].get("fate") == B[i].get("fate") for i in common])),
            "max_abs_dbeta_plus_common_survivors": max(db) if db else None,
            "pinch_time_shift_median": pct(dt, 50), "pinch_time_shift_max": pct(dt, 100),
            "mismatches": [(i, B[i].get("fate"), A[i].get("fate")) for i in common if A[i].get("fate") != B[i].get("fate")][:20]}
    print("PAIRED", json.dumps(out["paired_checks"]))
    (OUT / "C142_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    g = out["regression"]
    print("REG c76-copy vs stored main: max|dbeta+|=%s (n=%s); c142 vs c76: %s; amp_s1 vs stored hi: %s" % (
        g["reg_c76_vs_stored_C76_main"][0], g["reg_c76_vs_stored_C76_main"][1], g["reg_c142_vs_reg_c76"][0],
        g["amp_s1_vs_stored_C76_hi"][0]))
    if "reg_c76_summary" in g:
        print("REG var ratio aspect (main) = %.10f (C76: 1.0513592097)" % g["reg_c76_summary"].get("var_ratio_aspect", np.nan))
    for tx, I in out["r1_linear"].items():
        print("LIN %-9s t0=%.4f G_E=%.4g sig_lin_E=%.4f beta_rms_E=%.3g Pi_rms_E=%.3g | G_T=%.3g beta_rms_T=%.3g" % (
            tx, I["t0"], I["aspect"]["G"], I["aspect"]["sigma_lin_opt"], I["aspect"]["beta_rms_opt"],
            I["aspect"]["Pi_rms_opt"], I["angle"]["G"], I["angle"]["beta_rms_opt"]))
    print("%-24s %4s %6s %-62s %8s %8s %8s %7s %7s" % ("set", "N", "surv", "fates", "rmsNL_E", "rmsLinE", "ratioE", "lnA_med", "S12max"))
    for nm, o in out["sets"].items():
        if nm.startswith("grid"):
            continue
        print("%-24s %4d %6.3f %-62s %8s %8s %8s %7s %7s" % (
            nm, o["N"], o["survival"], json.dumps(o["fates"]), fm((o.get("rms_NL") or {}).get("aspect")),
            fm(o["rms_lin_all"]["aspect"]), fm(o.get("var_ratio_aspect")), fm((o.get("lnaniso_plus") or {}).get("median")),
            fm((o.get("bounce") or {}).get("S_over_12_max"))))
    for nm, o in out["sets"].items():
        if nm.startswith("grid"):
            print("GRID %-12s first_fail_a=%s last_survive_a=%s" % (nm, fm(o["first_fail_a"]), fm(o["last_survive_a"])))
    print("AMP threshold:", json.dumps(out["amp_threshold"]))


# ------------------------------------------------------------------ smoke test
def smoke():
    t_s = time.time()
    rep = {"r1_linear": {}, "runs": []}
    for tx in ("-0.5", "-1.0", "F", "-2.0"):
        t0 = tx_value(tx, R_SM)
        info = r1_info(R_SM, t0)
        rep["r1_linear"][tx] = info
        a, g = info["aspect"], info["angle"]
        print("tx=%-5s t0=%.5f | E: G=%.4g sig_lin=%.4f beta_rms=%.4g Pi_rms=%.4g | T2: G=%.4g sig_lin=%.4g beta_rms=%.4g Pi_rms=%.4g"
              % (tx, t0, a["G"], a["sigma_lin_opt"], a["beta_rms_opt"], a["Pi_rms_opt"], g["G"], g["sigma_lin_opt"],
                 g["beta_rms_opt"], g["Pi_rms_opt"]), flush=True)
    m = NLR(R_SM)

    def show(tag, q):
        bp = q.get("beta_plus")
        print("%-16s fate=%-18s st=%-16s s0/smax=%s S12b=%s tb=%s lrb=%s |bE+|=%s lnA+=%s minEig=%s rt=%.2fs nfev=%d" % (
            tag, q.get("fate"), q.get("status"), fm(q.get("s0_over_smax")), fm(q.get("S_over_12_bounce")),
            fm(q.get("t_bounce")), fm(q.get("lr_bounce")), fm(float(np.linalg.norm(np.array(bp)[ASP])) if bp else None),
            fm(q.get("lnaniso_plus")), fm(q.get("min_eigG_path")), q["runtime_s"], q["nfev"]), flush=True)
        rep["runs"].append({"tag": tag, **{k: q.get(k) for k in (
            "fate", "status", "reason", "s0_over_smax", "S_over_12_bounce", "t_bounce", "lr_bounce", "lnaniso_plus",
            "min_eigG_path", "runtime_s", "nfev", "beta_plus", "H0", "H0_background", "S0")}})
    for b in C76.specs("main", R_SM)[:3]:
        q = m.run_one(b["beta0"], b["Pi0"], 0.0)
        st = json.loads((C76_DRAWS / ("main_%04d.json" % b["i"])).read_text())
        print("  reg main_%04d max|dbeta+| vs stored C76 = %.2e" % (
            b["i"], np.max(np.abs(np.array(q["beta_plus"]) - np.array(st["beta_plus"])))))
        show("reg%d" % b["i"], q)
    for s in (10, 20, 40):
        for sp in amp_specs(R_SM, 3, s, "x"):
            show("amp s=%d i=%d" % (s, sp["i"]), m.run_one(sp["beta0"], sp["Pi0"], 0.0))
    for tx in ("-1.0", "F"):
        for sp in r1_specs(R_SM, tx, 3, "x", SEED_R1[tx]):
            q = m.run_one(sp["beta0"], sp["Pi0"], sp["t0"])
            show("r1 %s i=%d" % (tx, sp["i"]), q)
            print("     H0=%s (bg %s) S0=%s constraint0=%s" % (fm(q.get("H0")), fm(q.get("H0_background")), fm(q.get("S0")),
                                                              fm(q.get("constraint0"))), flush=True)
    pc = float(np.sqrt(12 / 3 ** 1.5) / C32)
    for p in (0.99 * pc, 1.01 * pc):
        show("gridP p=%.4f" % p, m.run_one([0.0] * 5, [0.0, p, 0.0, 0.0, 0.0], 0.0))
    rep["p_c_analytic"] = pc
    rep["runtime_s"] = time.time() - t_s
    (OUT / "smoke.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print("smoke done %.1f s (p_c analytic %.5f)" % (rep["runtime_s"], pc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run", "analyze"])
    ap.add_argument("--group", default="all", choices=["core", "extra", "all"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("DNEC_WORKERS", "5")))  # package: --workers
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "smoke":
        smoke()
    elif a.mode == "run":
        run(a.group, a.procs)  # package: no cap (each draw is independent; results do not depend on the worker count)
    else:
        analyze()


if __name__ == "__main__":
    main()
