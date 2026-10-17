# PROVENANCE: 04_calculations/code/20261011_claude_followup/C76_nonlinear_zero_mode.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] Nonlinear shape model with the zero-mode energy (xi = r = 0.299897; in the code r is the zero-mode fraction, not the tensor ratio): Z0 -> (1-r) Z_free, kappa_E = 19.32, delta = 2.077e-4 (C60/C76 convention). Driver of the fu/C76 ensembles (main 128, hi 2048, ind 2048 draws) and shared library. Paper: PRD Sec. VIII D (sec:s-fossil: nonlinear/linear variance ratio 1.043), App. app:nonlinear.

C76: nonlinear survival and width of the SM quantum fossil WITH the gauge/Higgs zero-mode energy (follow-up).

The constant-mode Hamiltonian of the SM gauge/Higgs zero modes is shape independent at fixed volume (C71, C72/D54), so
its ground energy Q/L = r C_free(1)/L enters as a shape-independent positive constant: Z_eff(G) -> Z_eff(G) - r Z_eff(1)
in the volume constraint, while the shape force (gradient) is unchanged. In the C60 equations this is a wrapped Casimir
object returning (Z(G) - ZQ, D(G)) and Z0 -> (1 - r) Z_eff(1), which raises every kappa to kappa/(1 - r).

Draws follow the C60 protocol (predeclared, seeded, no rejection) with the natural widths and delta rescaled:
    delta_r = delta (1 - r)^(-3/4),  s_A = (3/(2 sqrt2)) delta_r / sqrt|kappa_A / (1 - r)|.
Each draw is evolved through the full nonlinear L + five-shape-moduli equations (C60_nonlinear_integrator, imported
read-only) and compared with the paired linear transfer at kappa/(1 - r).

    python C76_nonlinear_zero_mode.py run --tags main hi     (main: 128 draws, seed 20261010; hi: 2048, seed 20261011)
    python C76_nonlinear_zero_mode.py run --tags ind         (ind: 2048 draws, seed 20261012; added after R10 because
                                                              main/hi reuse the seeds of the free-field C60 draws)
    python C76_nonlinear_zero_mode.py analyze
Default r = 0.299897 (C72: DMC exact local ground energy, one-loop couplings at 1/L_b, lambda = 0.01).
Outputs: results/20261011_claude_followup/C76/ (draws/, C76_summary.json).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import time
import argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
NLDIR = HERE  # package: the C60 modules live in this directory (shape/shapelib)
sys.path.insert(0, str(NLDIR))
from C60_nonlinear_integrator import NLModel, linear_transfer          # noqa: E402  (read-only import)
from C60_twisted_ewald import CONTENT_SM                               # noqa: E402

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir("fu/C76")
DRAWS = OUT / "draws"
R_ZM = 0.299897                       # C72 DMC, one-loop couplings at 1/L_b, lambda = 0.01
DELTA0 = 1.5897991e-4                 # leading-background delta of the free benchmark (C60)
KAPPA = np.array([13.528581, 13.528581, -6.352387, -6.352387, -6.352387])
TYPE = np.array([0, 0, 1, 1, 1])
SEED, SEED_HI = 20261010, 20261011


def widths(r):
    kap = KAPPA / (1 - r)
    delta = DELTA0 * (1 - r) ** (-0.75)
    s = (3 / (2 * np.sqrt(2))) * delta / np.sqrt(np.abs(kap))
    return kap, delta, s


class CasQ:
    """Casimir object with a shape-independent zero-mode energy subtracted from Z (force unchanged)."""
    def __init__(self, base, zq):
        self.base, self.zq, self.cut = base, zq, base.cut

    def __call__(self, G):
        Z, D = self.base(G)
        return Z - self.zq, D


class NLQ(NLModel):
    def __init__(self, r, cut=42.0, rtol=1e-11, atol=1e-13):
        super().__init__(CONTENT_SM, cut=cut, rtol=rtol, atol=atol)
        zfree = self.Z0
        self.cas = CasQ(self.cas, r * zfree)
        self.Z0 = (1 - r) * zfree


_MODEL = {}


def work(spec):
    path = DRAWS / ("%s_%04d.json" % (spec["tag"], spec["i"]))
    if path.exists():
        return 0
    key = (spec["r"],)
    if key not in _MODEL:
        _MODEL[key] = NLQ(spec["r"])
    rec = _MODEL[key].forward(spec["beta0"], spec["Pi0"], ic="on", label="%s_%d" % (spec["tag"], spec["i"]))
    rec.update(tag=spec["tag"], i=spec["i"], r=spec["r"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    return 1


SEED_IND = 20261012       # independent set (R10 review AUDIT-2): main/hi reuse the C60 free-field seeds


def specs(tag, r):
    n, seed = {"main": (128, SEED), "hi": (2048, SEED_HI), "ind": (2048, SEED_IND)}[tag]
    z = np.random.default_rng(seed).standard_normal((n, 10))
    kap, delta, s = widths(r)
    return [dict(tag=tag, i=i, r=r, beta0=(np.sqrt(s) * z[i, :5]).tolist(), Pi0=(delta / np.sqrt(s) * z[i, 5:]).tolist())
            for i in range(n)]


def run(tags, r, procs):
    DRAWS.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        sp = specs(tag, r)
        todo = [s for s in sp if not (DRAWS / ("%s_%04d.json" % (s["tag"], s["i"]))).exists()]
        t0 = time.time()
        done = 0
        if todo:
            with Pool(procs) as pool:
                for k in pool.imap_unordered(work, todo, chunksize=4):
                    done += k
        print("%-5s total %d, ran %d, %.1f s" % (tag, len(sp), done, time.time() - t0), flush=True)


def geometry_stats(BP):
    """Median ln(Lmax/Lmin) and median maximal edge-angle deviation (deg) for outgoing beta vectors."""
    from scipy.linalg import expm
    from C60_twisted_ewald import BASIS
    lnl, ang = [], []
    for b in BP:
        G = expm(2 * np.einsum("a,aij->ij", b, BASIS))
        sd = np.sqrt(np.diag(G))
        lnl.append(np.log(sd.max() / sd.min()))
        ca = [G[i, j] / (sd[i] * sd[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        ang.append(np.max(np.abs(np.degrees(np.arcsin(np.clip(ca, -1, 1))))))
    return float(np.median(lnl)), float(np.median(ang))


def analyze(r):
    kap, delta, s = widths(r)
    MA = linear_transfer(kap[0])          # full 2x2 [[m11, m12], [m21, m22]]
    MT = linear_transfer(kap[2])
    m11 = np.array([MA[0][0] if t == 0 else MT[0][0] for t in TYPE])
    m12 = np.array([MA[0][1] if t == 0 else MT[0][1] for t in TYPE])
    var_ana = m11 ** 2 * s + m12 ** 2 * delta ** 2 / s
    out = {"r": r, "kappa": kap.tolist(), "delta": delta, "s_nat": s.tolist(),
           "sigma_lin_analytic": {"aspect": float(np.sqrt(var_ana[0])), "angle": float(np.sqrt(var_ana[2]))},
           "linear_transfer": {"aspect": np.asarray(MA).tolist(), "angle": np.asarray(MT).tolist()}, "sets": {}}
    rng = np.random.default_rng(7)
    # linear-Gaussian geometry reference (1e5 draws)
    zz = rng.standard_normal((100000, 5)) * np.sqrt(var_ana)
    out["linear_gaussian_geometry"] = dict(zip(("lnL_median", "angle_dev_deg_median"), geometry_stats(zz[:20000])))
    for tag in ("main", "hi", "ind"):
        recs = [json.loads(p.read_text()) for p in sorted(DRAWS.glob(tag + "_*.json"))]
        if not recs:
            continue
        surv = [q for q in recs if q.get("survivor")]
        B0 = np.array([q["beta0"] for q in surv])
        P0 = np.array([q["Pi0"] for q in surv])
        BP = np.array([q["beta_plus"] for q in surv])
        BL = m11 * B0 + m12 * P0
        sq_nl, sq_lin = BP ** 2, BL ** 2
        asp, ang = TYPE == 0, TYPE == 1
        idx = rng.integers(0, len(surv), (2000, len(surv)))

        def boot(num, den):
            v = num[idx].mean(1) / den[idx].mean(1)
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        D = BP - BL
        gnl = geometry_stats(BP)
        glin = geometry_stats(BL)
        done = [q for q in recs if q["status"] == "complete"]
        out["sets"][tag] = {
            "N": len(recs), "n_survivors": len(surv),
            "status_counts": {k: sum(1 for q in recs if q["status"] == k) for k in set(q["status"] for q in recs)},
            "max_constraint": max(q["gate_constraint_max"] for q in done) if done else None,
            "runtime_mean_s": float(np.mean([q["runtime_s"] for q in recs])),
            "var_ratio_NL_over_lin_paired": {
                "aspect": float(sq_nl[:, asp].sum() / sq_lin[:, asp].sum()), "aspect_boot95": boot(sq_nl[:, asp].sum(1), sq_lin[:, asp].sum(1)),
                "angle": float(sq_nl[:, ang].sum() / sq_lin[:, ang].sum()), "angle_boot95": boot(sq_nl[:, ang].sum(1), sq_lin[:, ang].sum(1))},
            "rms_NL": {"aspect": float(np.sqrt(sq_nl[:, asp].mean())), "angle": float(np.sqrt(sq_nl[:, ang].mean()))},
            "rms_lin_paired": {"aspect": float(np.sqrt(sq_lin[:, asp].mean())), "angle": float(np.sqrt(sq_lin[:, ang].mean()))},
            "paired_rel_rms": float(np.sqrt((D ** 2).sum() / (BL ** 2).sum())),
            "geometry_NL": {"lnL_median": gnl[0], "angle_dev_deg_median": gnl[1]},
            "geometry_lin_paired": {"lnL_median": glin[0], "angle_dev_deg_median": glin[1]}}
    (OUT / "C76_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--tags", nargs="+", default=["main", "hi"])
    ap.add_argument("--r", type=float, default=R_ZM)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("DNEC_WORKERS", "6")))  # package: --workers
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "run":
        run(a.tags, a.r, a.procs)
    else:
        analyze(a.r)


if __name__ == "__main__":
    main()
