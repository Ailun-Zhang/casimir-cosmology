# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_tier2/C160_plateau.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Nonlinear survival on the A_s-calibrated plateau history (D131 Sec. 2): 2560 trajectories paired with D114 (same z); 0/2560 fate changes. Paper: PRD Sec. VIII E (sec:s-nonlinear), App. app:nonlinear.

C160 part 2 (shape-tier2, D131 Sec. 2): nonlinear shape survival on the A_s-calibrated plateau history.

C142 model (copied unchanged: full nonlinear volume + five shape moduli, SM twisted Ewald Casimir force, zero-mode r,
DOP853 1e-11/1e-13, max_step 0.2, T = 12, pinch eig G < 1e-3) with the leading cosh background replaced by the
time-symmetric plateau history: homogeneous inflaton phi with
    Vhat(phi) = 3 [(1 - e^{-a phi}) / (1 - e^{-a phi_b})]^2,  a = sqrt(2/3),  phi_b = 5.5794 (v11, A_s/N_* calibrated),
in units M_P^2 H_eff^2 (H_eff = sqrt(V(phi_b)/3)/M_P is the time unit, as in C142).  Equations (y = [lr,H,G,M,phi,phid]):
    lr' = H,  H' = -2H^2 - phid^2/6 + 2 Vhat/3 - S/6,  phi'' = -3 H phid - Vhat'(phi),  shape equations as C60;
    constraint 3H^2 = phid^2/2 + Vhat - 3 (Z/Z0) e^{-4 lr} + S/2   (reduces to C60/C142 for Vhat = 3, phid = 0).
Turning point: H = 0, phi = phi_b, phid = 0 (time symmetry); t0 < 0 starts use the reflected plateau background
(lr, phi, -phid) at t0 and H0 from the constraint.  R1 widths use the plateau linear transfer.  Draws are paired with
D114 (same z): amplitude scan = C76 hi seed 20261011 first 256 x s; R1 seeds 20261302/03/04, N = 512.
    python -B C160_plateau.py smoke
    python -B C160_plateau.py run --procs 2 [--sets amp r1]
    python -B C160_plateau.py analyze
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
sys.path.insert(0, str(HERE.parent / "shapelib"))  # package: C142/C76/C60
import C142_shape_nonlinear_threshold as C142                          # noqa: E402 (copy, unchanged)
import C76_nonlinear_zero_mode as C76                                   # noqa: E402
from C60_nonlinear_integrator import unpack, pack, C32, IU              # noqa: E402
from C60_twisted_ewald import ControlExit                              # noqa: E402

sys.path.insert(0, str(HERE.parent))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("r3/shape_tier2/run01/plateau")
DRAWS = OUT / "draws"
D114 = WP.wdir("r2/shape_nonlinear_threshold/run01/draws")
A = np.sqrt(2.0 / 3.0)
PHI_B = 5.5794
R_SM = C142.R_SM
AMP = (1, 3, 5, 10)
TXS = ("-0.5", "-1.0", "F")
N_AMP, N_R1 = 256, 512


class Plateau:
    """Time-symmetric plateau background of the cube (no shear): t >= 0 integrated, t < 0 by reflection."""

    def __init__(self, phi_b=PHI_B, tmax=14.5):
        self.phi_b = phi_b
        self.eb = 1.0 - np.exp(-A * phi_b)

        def rhs(t, y):
            lr, H, p, pd = y
            return [H, -2 * H * H - pd * pd / 6 + 2 * self.V(p) / 3, pd, -3 * H * pd - self.Vp(p)]
        self.sol = solve_ivp(rhs, (0.0, tmax), [0.0, 0.0, phi_b, 0.0], method="DOP853", rtol=1e-13, atol=1e-15,
                             max_step=0.01, dense_output=True)

    def V(self, p):
        return 3.0 * ((1.0 - np.exp(-A * p)) / self.eb) ** 2

    def Vp(self, p):
        return 6.0 * A * np.exp(-A * p) * (1.0 - np.exp(-A * p)) / self.eb ** 2

    def at(self, t):
        y = np.array(self.sol.sol(abs(t)), float)
        if t < 0:
            y[[1, 3]] *= -1
        return y                                   # lr, H, phi, phid

    def lin_M(self, kappa, t0, tf=14.0):
        def rhs(t, y):
            lr = self.at(t)[0]
            return [C32 * y[1] * np.exp(-3 * lr), kappa * np.exp(-lr) * y[0] / C32]
        cols = []
        for y0 in ([1.0, 0.0], [0.0, 1.0]):
            s = solve_ivp(rhs, (t0, tf), y0, method="DOP853", rtol=1e-13, atol=1e-15, max_step=0.02)
            cols.append(s.y[:, -1])
        return np.array(cols).T


_BG = {}


def background(phi_b=PHI_B):
    if phi_b not in _BG:
        _BG[phi_b] = Plateau(phi_b)
    return _BG[phi_b]


class NLP(C142.NLR):
    """C142 NLR with the rolling plateau scalar (y has 16 components: C60's 14 + phi, phid)."""

    def __init__(self, r, phi_b=PHI_B, **kw):
        super().__init__(r, **kw)
        self.bg = background(phi_b)

    def rhs(self, t, y):
        if self.nfev > C142.NFEV_MAX:
            raise C142.NfevLimit("nfev > %d" % C142.NFEV_MAX)
        self.nfev += 1
        self.last = (float(t), np.array(y))
        if y[1] > self.maxH:
            self.maxH = float(y[1])
        lr, H, G, M = unpack(y)
        p, pd = y[14], y[15]
        S, Z, gd, md = self.pieces(lr, G, M)
        dH = -2 * H * H - pd * pd / 6 + 2 * self.bg.V(p) / 3 - S / 6
        return np.concatenate(([H, dH], gd[IU], (0.5 * (md + md.T))[IU], [pd, -3 * H * pd - self.bg.Vp(p)]))

    def constraint(self, y):
        lr, H, G, M = unpack(y)
        S, Z, gd, md = self.pieces(lr, G, M)
        p, pd = (y[14], y[15]) if len(y) > 15 else (self.bg.phi_b, 0.0)   # C60 initial_state passes the 14-vector
        return float(3 * H * H - (0.5 * pd * pd + self.bg.V(p)) + 3 * Z / self.Z0 * np.exp(-4 * lr) - S / 2)

    def run_one(self, beta0, Pi0, t0):
        t_start = time.perf_counter()
        self.nfev, self.maxH, self.last = 0, -np.inf, None
        rec = {"t0": float(t0), "T": C142.T_END, "rtol": self.rtol, "atol": self.atol, "cut": self.cas.cut,
               "eig_stop": self.eig_stop, "background": "plateau", "phi_b": self.bg.phi_b}
        try:
            G0, M0, s0, zeta = self.shape_state(beta0, Pi0)
            ev0 = np.linalg.eigvalsh(G0)
            rec.update(s0=s0, zeta0=zeta, lnaniso0=float(0.5 * np.log(ev0[-1] / ev0[0])))
            if t0 == 0.0:
                smax = 12.0 * (zeta / 3.0) ** 1.5 if zeta > 0 else 0.0
                rec["s0_over_smax"] = s0 / smax if smax > 0 else float("inf")
                if zeta <= 0 or s0 > smax:
                    rec.update(status="no_turning_point", fate="no_turning", bounced=False, survivor=False,
                               survivor_C60=False)
                    return self._fin(rec, t_start)
                y0, info = self.initial_state(beta0, Pi0, "on")       # identical turning-point data (Vhat(phi_b)=3)
                y0 = np.concatenate((y0, [self.bg.phi_b, 0.0]))
                rec.update({k: v for k, v in info.items() if k != "constraint0"})
                rec["constraint0"] = self.constraint(y0)
            else:
                lrb, Hb, pb, pdb = self.bg.at(t0)
                q = (0.5 * pdb * pdb + self.bg.V(pb) - 3 * zeta * np.exp(-4 * lrb) + 0.5 * s0 * np.exp(-6 * lrb)) / 3
                rec.update(lr0=float(lrb), S0=float(s0 * np.exp(-6 * lrb)), H0_background=float(Hb),
                           phi0=float(pb), phid0=float(pdb))
                if q < 0:
                    rec.update(status="forbidden_start", fate="other", bounced=False, survivor=False, survivor_C60=False)
                    return self._fin(rec, t_start)
                y0 = np.concatenate((pack(lrb, -np.sqrt(q), G0, M0), [pb, pdb]))
                rec.update(H0=float(y0[1]), constraint0=self.constraint(y0))
            sol = self._integrate(y0, t0)
            self._classify(sol, rec)
            ye = sol.y[:, -1]
            rec.update(phi_end=float(ye[14]), phid_end=float(ye[15]))
        except (ControlExit, C142.NfevLimit) as exc:
            rec.update(status="control_exit" if isinstance(exc, ControlExit) else "nfev_limit", reason=str(exc))
            self._exc_fate(rec)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rec.update(status="numerical_error", reason=str(exc))
            self._exc_fate(rec)
        except Exception as exc:
            rec.update(status="error", reason=repr(exc))
            self._exc_fate(rec)
        return self._fin(rec, t_start)


# ------------------------------------------------------------------ draws (paired with D114)
def r1_info_pl(r, t0, bg):
    kap, delta, _ = C76.widths(r)
    info = {"t0": t0, "delta": float(delta)}
    for name, k in (("aspect", kap[0]), ("angle", kap[2])):
        M, M0 = bg.lin_M(k, t0), bg.lin_M(k, 0.0)
        p, p0 = abs(M[0, 0] * M[0, 1]), abs(M0[0, 0] * M0[0, 1])
        so = abs(M[0, 1] / M[0, 0]) * delta
        info[name] = {"kappa": float(k), "M": M.tolist(), "s_opt": float(so), "G": float(p / p0),
                      "sigma_lin_opt": float(np.sqrt(2 * p * delta)), "sigma_floor": float(np.sqrt(2 * p0 * delta))}
    return info


def specs(sets=("amp", "r1")):
    S = []
    if "amp" in sets:
        base = C76.specs("hi", R_SM)[:N_AMP]
        for s in AMP:
            S += [dict(set="pl_amp_s%d" % s, ref="amp_s%d" % s, i=b["i"], t0=0.0,
                       beta0=(s * np.asarray(b["beta0"])).tolist(), Pi0=(s * np.asarray(b["Pi0"])).tolist(),
                       meta={"s": s}) for b in base]
    if "r1" in sets:
        bg = background()
        for tx in TXS:
            t0 = C142.tx_value(tx, R_SM)
            info = r1_info_pl(R_SM, t0, bg)
            _, delta, _ = C76.widths(R_SM)
            sv = np.array([info["aspect"]["s_opt"] if t == 0 else info["angle"]["s_opt"] for t in C142.TYPE])
            z = np.random.default_rng(C142.SEED_R1[tx]).standard_normal((max(N_R1, C142.N_R1), 10))[:N_R1]
            S += [dict(set="pl_r1_tx%s" % tx, ref="r1_tx%s" % tx, i=i, t0=t0, beta0=(np.sqrt(sv) * z[i, :5]).tolist(),
                       Pi0=(delta / np.sqrt(sv) * z[i, 5:]).tolist(), meta={"tx": tx}) for i in range(N_R1)]
    return S


_M = {}


def work(spec):
    path = DRAWS / spec["set"] / ("%04d.json" % spec["i"])
    if path.exists():
        return 0
    if "m" not in _M:
        _M["m"] = NLP(R_SM)
    rec = _M["m"].run_one(spec["beta0"], spec["Pi0"], spec["t0"])
    rec.update(set=spec["set"], ref=spec["ref"], i=spec["i"], meta=spec["meta"],
               beta0=[float(x) for x in spec["beta0"]], Pi0=[float(x) for x in spec["Pi0"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return 1


def run(sets, procs, only=None):
    S = specs(sets)
    if only:
        S = [s for s in S if s["set"] in only]
    todo = [s for s in S if not (DRAWS / s["set"] / ("%04d.json" % s["i"])).exists()]
    print("specs %d todo %d" % (len(S), len(todo)), flush=True)
    t0, done = time.time(), 0
    if todo:
        with Pool(procs) as pool:
            for k in pool.imap_unordered(work, todo, chunksize=2):
                done += k
                if done % 100 == 0:
                    print("  done %d/%d  %.0f s" % (done, len(todo), time.time() - t0), flush=True)
    print("finished: ran %d in %.0f s" % (done, time.time() - t0), flush=True)


# ------------------------------------------------------------------ smoke
def smoke():
    out = {}
    bg = background()
    tt = np.linspace(0, 3, 61)
    lr = np.array([bg.at(t)[0] for t in tt])
    H = np.array([bg.at(t)[1] for t in tt])
    out["bg_vs_cosh"] = {"max_dlr_0_3": float(np.max(np.abs(lr - 0.5 * np.log(np.cosh(2 * tt))))),
                         "max_dH_0_3": float(np.max(np.abs(H - np.tanh(2 * tt)))),
                         "dlr_t1": float(bg.at(1.0)[0] - 0.5 * np.log(np.cosh(2.0))),
                         "dlr_t12": float(bg.at(12.0)[0] - 0.5 * np.log(np.cosh(24.0))),
                         "phi_t1": float(bg.at(1.0)[2]), "phi_t12": float(bg.at(12.0)[2]), "H_t12": float(bg.at(12.0)[1]),
                         "epsilonV_phib": float(0.5 * (bg.Vp(PHI_B) / bg.V(PHI_B)) ** 2)}
    kap, delta, _ = C76.widths(R_SM)
    out["linear"] = {}
    for t0 in (0.0, -0.5, -1.0, C142.tx_value("F", R_SM)):
        Mp, Mc = bg.lin_M(kap[0], t0), C142.lin_M(kap[0], t0)
        out["linear"]["%.4f" % t0] = {"M_plateau": Mp.tolist(), "M_cosh": Mc.tolist(),
                                      "prod_ratio": float(abs(Mp[0, 0] * Mp[0, 1]) / abs(Mc[0, 0] * Mc[0, 1]))}
    # flat-limit regression (phi_b = 60 -> Vhat = 3 exactly) against C142 on identical draws
    flat = NLP(R_SM, phi_b=60.0)
    ref = C142.NLR(R_SM)
    pl = NLP(R_SM)
    comp = []
    S = specs(("amp", "r1"))
    picks = [s for s in S if s["set"] in ("pl_amp_s1", "pl_amp_s5") and s["i"] < 3] + \
            [s for s in S if s["set"] == "pl_r1_tx-1.0" and s["i"] < 2]
    for sp in picks:
        a = ref.run_one(sp["beta0"], sp["Pi0"], sp["t0"]) if sp["t0"] == 0.0 else None
        b = flat.run_one(sp["beta0"], sp["Pi0"], sp["t0"]) if sp["t0"] == 0.0 else None
        c = pl.run_one(sp["beta0"], sp["Pi0"], sp["t0"])
        dref = D114 / sp["ref"] / ("%04d.json" % sp["i"])
        d = json.loads(dref.read_text()) if dref.exists() else None
        row = {"set": sp["set"], "i": sp["i"], "plateau_fate": c.get("fate"), "plateau_beta_plus": c.get("beta_plus"),
               "plateau_constraint_max": c.get("gate_constraint_max"), "runtime_s": c.get("runtime_s"),
               "D114_fate": d.get("fate") if d else None, "D114_beta_plus": d.get("beta_plus") if d else None,
               "D114_beta0_match": bool(d and np.allclose(d["beta0"], sp["beta0"], rtol=0, atol=0)) if d else None}
        if a is not None:
            row.update(C142_fate=a.get("fate"), flat_fate=b.get("fate"))
            if a.get("beta_plus") and b.get("beta_plus"):
                row["flat_minus_C142_max_abs_dbeta"] = float(np.max(np.abs(np.array(a["beta_plus"]) - np.array(b["beta_plus"]))))
        comp.append(row)
        print(json.dumps(row, default=float), flush=True)
    out["draw_checks"] = comp
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("bg_vs_cosh",)}, default=float))


# ------------------------------------------------------------------ analysis
def wilson(k, n, z=1.96):
    return C142.wilson(k, n, z) if n else [None, None]


def analyze():
    bg = background()
    kap, delta, _ = C76.widths(R_SM)
    summ = {"sets": {}}
    for d in sorted(DRAWS.glob("pl_*")):
        recs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
        if not recs:
            continue
        ref = recs[0]["ref"]
        n = len(recs)
        k = sum(r["survivor"] for r in recs)
        fates = Counter(r["fate"] for r in recs)
        flips, both, dbeta, refn, refk, db0 = [], 0, [], 0, 0, []
        for r in recs:
            p = D114 / ref / ("%04d.json" % r["i"])
            if not p.exists():
                continue
            q = json.loads(p.read_text())
            # paired by draw index (same z); R1 widths differ by the plateau-vs-cosh s_opt (recorded, ~1e-4)
            db0.append(float(np.max(np.abs(np.array(q["beta0"]) - np.array(r["beta0"])) / (np.abs(np.array(q["beta0"])) + 1e-300))))
            refn += 1
            refk += int(q["survivor"])
            if q["survivor"] != r["survivor"]:
                flips.append({"i": r["i"], "D114": q["fate"], "plateau": r["fate"]})
            elif r["survivor"]:
                both += 1
                bp, bq = np.array(r["beta_plus"]), np.array(q["beta_plus"])
                dbeta.append(float(np.linalg.norm(bp - bq) / max(np.linalg.norm(bq), 1e-300)))
        surv = [r for r in recs if r["survivor"]]
        bp = np.array([r["beta_plus"] for r in surv]) if surv else np.zeros((0, 5))
        # paired linear prediction on the plateau background
        t0 = recs[0]["t0"]
        ME, MT = bg.lin_M(kap[0], t0), bg.lin_M(kap[2], t0)
        b0 = np.array([r["beta0"] for r in recs])
        P0 = np.array([r["Pi0"] for r in recs])
        blin = np.zeros_like(b0)
        for a in range(5):
            M = ME if C142.TYPE[a] == 0 else MT
            blin[:, a] = M[0, 0] * b0[:, a] + M[0, 1] * P0[:, a]
        sig_lin = float(np.sqrt(np.mean(blin[:, :2] ** 2)))
        cons = [r.get("gate_constraint_max") for r in surv if r.get("gate_constraint_max") is not None]
        row = {"n": n, "survive": k, "frac": k / n, "wilson95": wilson(k, n), "fates": dict(fates),
               "sigma_lin_aspect_plateau": sig_lin,
               "sigma_E_NL_survivors": float(np.sqrt(np.mean(bp[:, :2] ** 2))) if len(bp) else None,
               "paired_with_D114": refn, "D114_survive": refk, "D114_frac": refk / refn if refn else None,
               "max_rel_beta0_diff_vs_D114": float(max(db0)) if db0 else None,
               "paired_diff_frac": (k - refk) / n if refn == n else None,
               "fate_flips": flips, "n_flips": len(flips), "joint_survivors": both,
               "median_rel_dbeta_plus_joint": float(np.median(dbeta)) if dbeta else None,
               "max_rel_dbeta_plus_joint": float(np.max(dbeta)) if dbeta else None,
               "max_constraint_survivors": float(max(cons)) if cons else None,
               "max_runtime_s": float(max(r.get("runtime_s", 0) for r in recs))}
        if recs[0]["meta"].get("s") == 1 and len(surv) == n:
            vl = np.mean(blin[:, :2] ** 2)
            row["variance_ratio_NL_over_lin_aspect"] = float(np.mean(bp[:, :2] ** 2) / vl)
        summ["sets"][d.name] = row
        print(d.name, "n=%d surv=%.4f %s D114=%s flips=%d dbeta_med=%s" % (n, k / n, wilson(k, n), row["D114_frac"],
              len(flips), row["median_rel_dbeta_plus_joint"]), flush=True)
    summ["meta"] = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "phi_b": PHI_B,
                    "delta_C76": float(delta), "kappa": kap.tolist()}
    (OUT / "C160_plateau_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "analyze"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("DNEC_WORKERS", "2")))  # package: --workers
    ap.add_argument("--sets", nargs="+", default=["amp", "r1"])
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke()
    elif a.cmd == "run":
        run(tuple(a.sets), a.procs, a.only)
    else:
        analyze()


if __name__ == "__main__":
    main()
