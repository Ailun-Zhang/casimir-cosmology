# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_run02.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] D125 run02: N = 2048 recomputation of the 99%-level quantities: t_*(99%) = 0.237 and the 90% check (lo_t, nx_t), s99 of the amplitude scans (lo_amp 2.76 / logistic 2.86; nx_amp 3.16). Paper: PRD Sec. VIII B and E, App. app:nonlinear.

C153_run02 (shape-hardening; D125 "run02 appendix"; main-session request W3, 2026-10-15T04:52Z):
N = 2048 re-computation of the 99%-level quantities of D125 (run01 used N = 256-512 per point).
  lo_t : leading order (xi = 0.30), real optimal-width Gaussian at t_x = -tau, forward runs A (through the turn),
         tau in {0.15, 0.20, 0.25, 0.30, 0.35}, N = 2048 each; B (mirror of the past branch, (beta,-Pi) from +tau) on
         the first 256 draws of each point (run01: 0/5632 B failures);
  nx_t : next order (kappa_E = 18.0, delta(xi=0.28), rescaled-force completion), tau in {0.25, 0.30, 0.35}, same design;
  lo_amp / nx_amp : turning-point amplitude scans of the natural state, s in {2.5, 3.0, 3.5} (leading) and
         {3.0, 3.5, 4.0} (next order), N = 2048, one z set per order (paired across s).
Seeds (disjoint from run01): 20262001+10k (lo_t), 20262101+10k (nx_t), 20262201 (lo_amp), 20262301 (nx_amp).
Model, integrator and fates: exactly C153_nl.NLR2 / C142 (imported; run01 code unchanged).
    python C153_run02.py run --procs 4        (resumable: one JSON per draw in run02/draws/<set>/NNNN.json)
    python C153_run02.py analyze              -> run02/C153_run02_summary.json
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
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import C153_nl as NL                                      # noqa: E402  (run01 driver; its DRAWS path is not used)
C142, C76 = NL.C142, NL.C76
import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
RUN01 = WP.wdir("r3/shape_hardening/run01")
OUT = WP.wdir("r3/shape_hardening/run02")
DRAWS = OUT / "draws"
N2, NB = 2048, 256
LO_TAUS, NX_TAUS = (0.15, 0.20, 0.25, 0.30, 0.35), (0.25, 0.30, 0.35)
# 2026-10-15T05:28Z restructure (wall-time budget, machine shared): nx_amp s=4.0 replaced by s=2.5 (bracket from
# below; the partial s=4.0 set already on disk is reported separately, not used for s99); the two far-side tau points
# (lo_t 0.15, nx_t 0.35) are cut to N=1024 (their role is only to show the level is on one side).
LO_AMPS, NX_AMPS = (2.5, 3.0, 3.5), (2.5, 3.0, 3.5)
N_T = {("lo_t", 0.15): 1024, ("nx_t", 0.35): 1024}
SEED_LO_T, SEED_NX_T, SEED_LO_A, SEED_NX_A = 20262001, 20262101, 20262201, 20262301


def tgrid_specs(model, taus, seed0, tag):
    S = []
    for k, tau in enumerate(taus):
        s, d, _ = NL.opt_widths(model, -tau)
        z = np.random.default_rng(seed0 + 10 * k).standard_normal((N2, 10))
        B, P = np.sqrt(s) * z[:, :5], d / np.sqrt(s) * z[:, 5:]
        for i in range(N_T.get((tag, tau), N2)):
            S.append(NL.mk("%s_tau%s_A" % (tag, NL.tau_key(tau)), i, model, -tau, B[i], P[i], tag, {"tau": tau}))
            if i < NB:
                S.append(NL.mk("%s_tau%s_B" % (tag, NL.tau_key(tau)), i, model, +tau, B[i], -P[i], tag, {"tau": tau}))
    return S


def amp_specs(model, amps, seed, tag):
    s, d, _ = NL.nat_widths(model)
    z = np.random.default_rng(seed).standard_normal((N2, 10))
    S = []
    for a in amps:
        B, P = a * np.sqrt(s) * z[:, :5], a * d / np.sqrt(s) * z[:, 5:]
        for i in range(N2):
            S.append(NL.mk("%s_s%.1f" % (tag, a), i, model, 0.0, B[i], P[i], tag, {"s": a}))
    return S


def all_specs():
    return (tgrid_specs("xi030", LO_TAUS, SEED_LO_T, "lo_t") + tgrid_specs("next", NX_TAUS, SEED_NX_T, "nx_t")
            + amp_specs("xi030", LO_AMPS, SEED_LO_A, "lo_amp") + amp_specs("next", NX_AMPS, SEED_NX_A, "nx_amp"))


def work(spec):
    path = DRAWS / spec["set"] / ("%04d.json" % spec["i"])
    if path.exists():
        return 0
    m = NL.model_for(spec["model"])
    rec = m.run_two(spec["beta0"], spec["Pi0"], spec["t0"])
    rec.update(set=spec["set"], i=spec["i"], r=NL.MODELS[spec["model"]]["r"], model=spec["model"], group=spec["group"],
               meta=spec["meta"], beta0=spec["beta0"], Pi0=spec["Pi0"], delta=NL.MODELS[spec["model"]]["delta"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return 1


def run(procs):
    OUT.mkdir(parents=True, exist_ok=True)
    S = all_specs()
    S.sort(key=lambda s: (s["i"] // 256, s["set"], s["i"]))
    if not (OUT / "execution_contract.json").exists():
        (OUT / "execution_contract.json").write_text(json.dumps({
            "task": "D125 run02: N=2048 recomputation of 99%-level quantities (t_*(99%), t_*(90%) check, s99 leading/next order)",
            "agent": "shape-hardening", "request": "main session W3 2026-10-15T04:52Z", "code": Path(__file__).name,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": "C153_nl.NLR2 (= C142/C76/C60 copies, unchanged)", "N_per_point": N2, "B_subsample": NB,
            "sets": {"lo_t": LO_TAUS, "nx_t": NX_TAUS, "lo_amp": LO_AMPS, "nx_amp": NX_AMPS},
            "seeds": {"lo_t": "20262001+10k", "nx_t": "20262101+10k", "lo_amp": SEED_LO_A, "nx_amp": SEED_NX_A},
            "statistics": "Wilson 95%, binomial SE; t_* / s99 by linear (log s) interpolation with Jeffreys-Beta MC of node "
                          "uncertainties, and binomial logistic MLE with parametric bootstrap",
            "budget": {"tool_calls": 25, "workers": 4}, "outputs": "run02/C153_run02_summary.json, run02/draws/"}, indent=1),
            encoding="utf-8")
    todo = [s for s in S if not (DRAWS / s["set"] / ("%04d.json" % s["i"])).exists()]
    print("run02: specs %d, todo %d" % (len(S), len(todo)), flush=True)
    t0, done = time.time(), 0
    if todo:
        with Pool(procs) as pool:
            for k in pool.imap_unordered(work, todo, chunksize=4):
                done += k
                if done % 500 == 0:
                    print("  done %d/%d  %.0f s" % (done, len(todo), time.time() - t0), flush=True)
    print("run02 finished: ran %d in %.0f s" % (done, time.time() - t0), flush=True)
    (OUT / "run02_done.flag").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


# ------------------------------------------------------------------ analysis
def load(name):
    d = DRAWS / name
    if not d.exists():
        return {}
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return {q["i"]: q for q in recs}


def cross_lin(xs, ps, level, logx=False):
    xx = np.log(xs) if logx else np.asarray(xs, float)
    for i in range(len(xs) - 1):
        if ps[i] >= level > ps[i + 1]:
            x = xx[i] + (xx[i + 1] - xx[i]) * (ps[i] - level) / (ps[i] - ps[i + 1])
            return float(np.exp(x) if logx else x)
    return None


def mc_cross(xs, ks, ns, level, logx=False, nrep=20000, seed=11):
    rng = np.random.default_rng(seed)
    P = np.stack([rng.beta(k + 0.5, n - k + 0.5, nrep) for k, n in zip(ks, ns)], axis=1)
    vals, below, above = [], 0, 0
    for row in P:
        c = cross_lin(xs, row, level, logx)
        if c is None:
            if row[0] < level:
                below += 1
            else:
                above += 1
        else:
            vals.append(c)
    v = np.array(vals)
    q = (lambda a: float(np.percentile(v, a))) if v.size else (lambda a: None)
    return {"median": q(50), "p16": q(16), "p84": q(84), "p2.5": q(2.5), "p97.5": q(97.5),
            "frac_below_range": below / nrep, "frac_above_range": above / nrep}


def logit_fit(x, k, n):
    x, k, n = (np.asarray(v, float) for v in (x, k, n))

    def nll(p):
        eta = p[0] + p[1] * x
        return -(k * -np.logaddexp(0, -eta) + (n - k) * -np.logaddexp(0, eta)).sum()
    best = None
    for start in ([5.0, -10.0], [10.0, -30.0], [3.0, -3.0]):
        r = minimize(nll, start, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=40000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def logit_tstar(x, k, n, levels, nboot=400, seed=12):
    (a, b), nll0 = logit_fit(x, k, n)
    lg = lambda L: np.log(L / (1 - L))
    out = {"a": float(a), "b": float(b), "nll": float(nll0),
           "t_star": {str(L): float((lg(L) - a) / b) for L in levels}}
    rng = np.random.default_rng(seed)
    pfit = 1 / (1 + np.exp(-(a + b * np.asarray(x, float))))
    # goodness of fit: deviance
    kk, nn = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.nansum(np.where(kk > 0, kk * np.log(kk / (nn * pfit)), 0) +
                            np.where(nn - kk > 0, (nn - kk) * np.log((nn - kk) / (nn * (1 - pfit))), 0))
    out["deviance"], out["dof"] = float(dev), int(len(x) - 2)
    bs = {str(L): [] for L in levels}
    for _ in range(nboot):
        kb = rng.binomial(np.asarray(n, int), pfit)
        (ab, bb), _ = logit_fit(x, kb, n)
        for L in levels:
            bs[str(L)].append((lg(L) - ab) / bb)
    out["boot"] = {L: {"p2.5": float(np.percentile(v, 2.5)), "p16": float(np.percentile(v, 16)),
                       "p84": float(np.percentile(v, 84)), "p97.5": float(np.percentile(v, 97.5))} for L, v in bs.items()}
    return out


def node(recs, model):
    n = len(recs)
    k = sum(bool(q["survivor"]) for q in recs.values())
    p = k / n if n else None
    return {"n": n, "k": k, "p": p, "se": float(np.sqrt(p * (1 - p) / n)) if n else None,
            "wilson95": C142.wilson(k, n) if n else None, "sigma_lin": NL.lin_rms(recs, model) if n else None,
            "fates": dict(Counter(q.get("fate") for q in recs.values()))}


def analyze():
    out = {"code": Path(__file__).name, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "N_per_point": N2, "B_subsample": NB}
    r01 = json.loads((RUN01 / "C153_nl_summary.json").read_text(encoding="utf-8"))
    b01_fail = sum(r["n"] - r["B_future_of_plus_tau"]["k"] for r in r01["grid"])
    b01_n = sum(r["n"] for r in r01["grid"])
    for tag, model, taus in (("lo_t", "xi030", LO_TAUS), ("nx_t", "next", NX_TAUS)):
        rows, bf, bn = [], 0, 0
        for tau in taus:
            A = load("%s_tau%s_A" % (tag, NL.tau_key(tau)))
            B = load("%s_tau%s_B" % (tag, NL.tau_key(tau)))
            rA = node(A, model)
            nb, kb = len(B), sum(bool(q["survivor"]) for q in B.values())
            ids = sorted(set(A) & set(B))
            both = sum(bool(A[i]["survivor"] and B[i]["survivor"]) for i in ids)
            bf, bn = bf + (nb - kb), bn + nb
            rows.append({"tau": tau, "A": rA, "B_sub": {"n": nb, "k": kb}, "two_branch_on_B_subsample": {"n": len(ids), "k": both}})
        tot_f, tot_n = bf + (b01_fail if tag == "lo_t" else 0), bn + (b01_n if tag == "lo_t" else 0)
        eps = (3.0 / tot_n) if tot_f == 0 else float(C142.wilson(tot_f, tot_n)[1])
        xs = [r["tau"] for r in rows]
        ks = [r["A"]["k"] for r in rows]
        ns = [r["A"]["n"] for r in rows]
        res = {"rows": rows, "B_failures_pooled": {"fail": tot_f, "n": tot_n, "eps95_upper": eps,
                                                   "note": "S2 = P(A and B) in [S_A - eps, S_A]"}}
        if all(n_ > 0 for n_ in ns):
            res["t_star_linear_interp"] = {str(L): {"central": cross_lin(xs, [k_ / n_ for k_, n_ in zip(ks, ns)], L),
                                                    "mc_binomial": mc_cross(xs, ks, ns, L)} for L in (0.99, 0.9)}
            res["t_star_logistic"] = logit_tstar(xs, ks, ns, (0.99, 0.9))
        out[tag] = res
    for tag, model, amps in (("lo_amp", "xi030", LO_AMPS), ("nx_amp", "next", NX_AMPS)):
        rows = [{"s": a, **node(load("%s_s%.1f" % (tag, a)), model)} for a in amps]
        xs, ks, ns = [r["s"] for r in rows], [r["k"] for r in rows], [r["n"] for r in rows]
        res = {"rows": rows}
        if all(n_ > 0 for n_ in ns):
            res["s99_loglinear"] = {"central": cross_lin(xs, [k_ / n_ for k_, n_ in zip(ks, ns)], 0.99, logx=True),
                                    "mc_binomial": mc_cross(xs, ks, ns, 0.99, logx=True)}
            lt = logit_tstar(np.log(xs), ks, ns, (0.99,))
            lt["s99"] = float(np.exp(lt["t_star"]["0.99"]))
            lt["s99_boot"] = {q: float(np.exp(v)) for q, v in lt["boot"]["0.99"].items()}
            res["s99_logistic_in_ln_s"] = lt
            fl = (NL.MODELS[model]["delta"] * 2 * abs(np.prod(C142.lin_M(NL.kap_of(model)[0], 0.0)[0]))) ** 0.5
            res["sigma_floor_this_convention"] = float(fl)
        out[tag] = res
    (OUT / "C153_run02_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for tag in ("lo_t", "nx_t"):
        o = out[tag]
        for r in o["rows"]:
            print("%s tau=%.3f A %d/%d=%.4f se=%.4f W%s sig_lin=%.4f | B %d/%d | two(Bsub) %d/%d fates=%s" % (
                tag, r["tau"], r["A"]["k"], r["A"]["n"], r["A"]["p"] or 0, r["A"]["se"] or 0,
                [round(x, 4) for x in (r["A"]["wilson95"] or [0, 0])], r["A"]["sigma_lin"] or 0, r["B_sub"]["k"], r["B_sub"]["n"],
                r["two_branch_on_B_subsample"]["k"], r["two_branch_on_B_subsample"]["n"], r["A"]["fates"]))
        print(tag, "B pooled", json.dumps(o["B_failures_pooled"]))
        if "t_star_linear_interp" in o:
            print(tag, "t* lin", json.dumps(o["t_star_linear_interp"]))
            print(tag, "t* logit", json.dumps({k: v for k, v in o["t_star_logistic"].items()}))
    for tag in ("lo_amp", "nx_amp"):
        o = out[tag]
        for r in o["rows"]:
            print("%s s=%.1f %d/%d=%.4f se=%.4f W%s sig_lin=%.4f fates=%s" % (tag, r["s"], r["k"], r["n"], r["p"] or 0, r["se"] or 0,
                  [round(x, 4) for x in (r["wilson95"] or [0, 0])], r["sigma_lin"] or 0, r["fates"]))
        if "s99_loglinear" in o:
            print(tag, "s99 loglin", json.dumps(o["s99_loglinear"]), "| logit s99 %.4f boot %s floor %.5f" % (
                o["s99_logistic_in_ln_s"]["s99"], json.dumps(o["s99_logistic_in_ln_s"]["s99_boot"]), o["sigma_floor_this_convention"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("DNEC_WORKERS", "4")))  # package: --workers
    a = ap.parse_args()
    if a.mode == "run":
        run(a.procs)  # package: no cap (each draw is independent; results do not depend on the worker count)
    else:
        analyze()


if __name__ == "__main__":
    main()
