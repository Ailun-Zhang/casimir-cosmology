# PROVENANCE: 04_calculations/code/20261009_claude_R5/nonlinear_SM/C60_twisted_ewald.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Twisted Ewald Casimir energy Z(G) of the SM content (15 periodic bosons + 45 AAA Weyl fermions, static graviton) on a unimodular torus metric G = exp(2 sum beta_A X_A), analytic gradients and Hessians (C60, D49b). Shared library of the nonlinear shape model. The stand-alone validation (__main__) compares with the round-1 C40 kernel and needs the research tree (not shipped); its archived output is shape/03_fossil/expected/r5/nonlinear_SM/C60_validation_ewald.json.

C60_twisted_ewald: twisted det-one Ewald evaluation of the Casimir shape function
Z_eff(G) = sum_f w_f Z2(G, theta_f) and of its symmetric gradient D (round R5, WP R5-NL).

    Z2(G, theta) = sum'_n cos(2 pi theta.n) (n^T G n)^-2           (G symmetric positive, det G = 1)

Ewald split at t = 1 (derivation: Gamma(2) (pi Q)^-2 = int_0^oo t e^{-pi Q t} dt, Poisson resummation of the
t < 1 piece with the Fourier phase e^{2 pi i theta.n}):

    Z2 = pi^2 [ -1/2 + 2 [theta in Z^3] + sum'_n c_n f(x_n) + sum_v' j(y_v) ],
    x_n = pi n^T G n,                       f(x) = e^{-x} (1+x) / x^2,
    y_v = pi v^T G^-1 v,  v = m - theta,    j(y) = 2 e^{-y} - 2 sqrt(pi y) erfc(sqrt y),
    (v = 0 excluded from the v-sum when theta in Z^3; its contribution is the "+2").

Gradient of the det-one extension (the det G^-1/2 prefactor is set to 1; its derivative is a pure G^-1 trace
term that the trace-free force (D G)_TF removes):

    D = pi^3 [ sum_n c_n f'(x_n) n n^T - sum_v j'(y_v) (G^-1 v)(G^-1 v)^T ],
    f'(x) = -e^{-x} (1/x + 2/x^2 + 2/x^3),   j'(y) = -sqrt(pi/y) erfc(sqrt y).

For theta = 0 this is exactly C40 ewald(G, cut) (same boxes, same summands).  Direct sums over several twists
share one lattice box with the combined weight W_n = sum_f w_f cos(2 pi theta_f.n).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import json
import time
import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import erfc
from scipy.linalg import expm, expm_frechet

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py; this file lives in shape/shapelib)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
ROOT = None  # package: the research tree is not shipped
CODE_ROOT = None  # package: load_C40() (validation against the round-1 C40 kernel) needs the research tree
OUT = WP.wdir("r5/nonlinear_SM")

# orthonormal basis of symmetric traceless 3x3 matrices (identical to C25 BASIS):
# E (aspect, two), T2 (angle, three)
XD = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
XD2 = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
XA = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float) / np.sqrt(2.0)
XA2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], float) / np.sqrt(2.0)
XA3 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], float) / np.sqrt(2.0)
BASIS = np.array([XD, XD2, XA, XA2, XA3])
I3 = np.eye(3)

TH0 = (0.0, 0.0, 0.0)
THAAA = (0.5, 0.5, 0.5)
# Standard Model, all-antiperiodic fermions: 12 gauge vectors + 4 real Higgs (1/2 each) + static graviton = 15
# boson weight (Maxwell units), 45 Weyl fermions.
CONTENT_SM = [(15.0, TH0), (-45.0, THAAA)]
CONTENT_MAXWELL = [(1.0, TH0)]


class ControlExit(RuntimeError):
    pass


def is_int_vec(th, tol=1e-12):
    th = np.asarray(th, float)
    return bool(np.all(np.abs(th - np.round(th)) < tol))


class Casimir:
    """Z_eff(G) and D(G) for weights [(w, theta)], twisted det-one Ewald sums."""

    def __init__(self, content, cut=42.0, max_points=1_500_000):
        self.content = [(float(w), np.asarray(th, float)) for w, th in content]
        self.cut = float(cut)
        self.max_points = int(max_points)
        # reciprocal groups: merge equal twists modulo the lattice
        groups = {}
        for w, th in self.content:
            key = tuple(np.round(th - np.floor(th), 12))
            groups[key] = groups.get(key, 0.0) + w
        self.recip = [(np.array(k), w) for k, w in groups.items() if w != 0.0]
        self.const = sum(w * (-0.5 + (2.0 if is_int_vec(th) else 0.0)) for w, th in self.content)
        self._dbox = {}
        self._rbox = {}
        self.calls = 0

    # ---------- lattice boxes (cached) ----------
    def _direct_box(self, rd):
        b = self._dbox.get(rd)
        if b is None:
            if int(np.prod(2 * np.array(rd) + 1)) > self.max_points:
                raise ControlExit("adaptive Ewald box exceeds %d lattice points" % self.max_points)
            grids = [np.arange(-r, r + 1, dtype=float) for r in rd]
            nn = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
            nn = nn[np.any(nn != 0, axis=1)]
            W = np.zeros(len(nn))
            for w, th in self.content:
                W += w * np.cos(2 * np.pi * (nn @ th))
            if len(self._dbox) > 64:
                self._dbox.clear()
            b = (nn, W)
            self._dbox[rd] = b
        return b

    def _recip_box(self, rq, th):
        key = (rq, tuple(th))
        b = self._rbox.get(key)
        if b is None:
            if int(np.prod(2 * np.array(rq) + 1)) > self.max_points:
                raise ControlExit("adaptive Ewald box exceeds %d lattice points" % self.max_points)
            axes = []
            for r, t in zip(rq, th):
                k = np.arange(int(np.ceil(t - r)), int(np.floor(t + r)) + 1, dtype=float)
                axes.append(k - t)
            vv = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            if is_int_vec(th):
                vv = vv[np.any(np.abs(vv) > 1e-9, axis=1)]
            if len(self._rbox) > 64:
                self._rbox.clear()
            b = vv
            self._rbox[key] = b
        return b

    # ---------- main evaluation ----------
    def __call__(self, G):
        G = np.asarray(G, float)
        self.calls += 1
        if not np.all(np.isfinite(G)):
            raise ControlExit("non-finite shape")
        eig = np.linalg.eigvalsh(G)
        if eig[0] <= 0:
            raise ControlExit("shape lost positive definiteness")
        Gi = np.linalg.inv(G)
        sd = np.sqrt(self.cut * np.diag(Gi) / np.pi)
        sq = np.sqrt(self.cut * np.diag(G) / np.pi)
        rd = tuple((np.ceil(sd).astype(int) + 1).tolist())
        rq = tuple((np.ceil(sq).astype(int) + 1).tolist())
        n, W = self._direct_box(rd)
        x = np.pi * np.einsum("ni,ij,nj->n", n, G, n)
        ex = np.exp(-x)
        f = ex * (1 + x) / x ** 2
        fp = -ex * (1 / x + 2 / x ** 2 + 2 / x ** 3)
        Zs = float(np.sum(W * f))
        wfp = W * fp
        Dm = np.einsum("n,ni,nj->ij", wfp, n, n)
        for th, w in self.recip:
            v = self._recip_box(rq, tuple(th))
            gv = v @ Gi
            y = np.pi * np.einsum("ni,ni->n", gv, v)
            sy = np.sqrt(y)
            ec = erfc(sy)
            j = 2 * np.exp(-y) - 2 * np.sqrt(np.pi * y) * ec
            jp = -np.sqrt(np.pi / y) * ec
            Zs += w * float(np.sum(j))
            Dm -= w * np.einsum("n,ni,nj->ij", jp, gv, gv)
        Z = np.pi ** 2 * (self.const + Zs)
        D = np.pi ** 3 * Dm
        return float(Z), 0.5 * (D + D.T)

    def Z(self, G):
        return self(G)[0]


# ------------------------------------------------------------------ helpers
def shape_from_beta(beta):
    """G = exp(2 sum_A beta_A X_A)."""
    return expm(2.0 * np.einsum("a,aij->ij", np.asarray(beta, float), BASIS))


def beta_from_shape(G):
    """beta_A = tr(X_A (1/2) log G), G normalised to det one."""
    vv, qq = np.linalg.eigh(G)
    if vv[0] <= 0:
        raise ControlExit("log of non-SPD shape")
    lg = np.log(vv) - np.mean(np.log(vv))
    B = 0.5 * (qq * lg) @ qq.T
    return np.einsum("aij,ji->a", BASIS, B)


def grad_beta(cas, beta):
    """dZ/dbeta_A at G = exp(2B(beta)) from D: tr(D dG/dbeta_A), with the Frechet derivative of expm."""
    B = np.einsum("a,aij->ij", np.asarray(beta, float), BASIS)
    G = expm(2 * B)
    Z, D = cas(G)
    g = np.empty(5)
    for a in range(5):
        _, dG = expm_frechet(2 * B, 2 * BASIS[a])
        g[a] = float(np.sum(D * dG))
    return Z, g


def hessian_from_gradient(cas, h=1e-3):
    """Hessian of Z along beta at the cube by 5-point centred differences of the analytic gradient."""
    H = np.zeros((5, 5))
    for b in range(5):
        def gb(s):
            e = np.zeros(5)
            e[b] = s
            return grad_beta(cas, e)[1]
        H[:, b] = (-gb(2 * h) + 8 * gb(h) - 8 * gb(-h) + gb(-2 * h)) / (12 * h)
    return H


def load_C40():
    """Import C40_full_matrix_kernel by path (its import side effects: sys.path, C32/C25 imports only)."""
    p = CODE_ROOT / "20261007_074655UTC" / "model_completion" / "C40_full_matrix_kernel.py"
    spec = importlib.util.spec_from_file_location("C40_full_matrix_kernel_ref", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["C40_full_matrix_kernel_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def z2_direct_cubebox(G, theta, R=60, chunk=8):
    """Brute-force Z2(G,theta) over the box |n_i| <= R (absolutely convergent only for theta != 0 in a
    conditional sense; used for the alternating AAA twist where the cubic-box sum converges fast)."""
    r = np.arange(-R, R + 1)
    n2, n3 = np.meshgrid(r, r, indexing="ij")
    n2 = n2.ravel().astype(float)
    n3 = n3.ravel().astype(float)
    th = np.asarray(theta, float)
    tot = 0.0
    for lo in range(-R, R + 1, chunk):
        n1 = np.arange(lo, min(lo + chunk, R + 1), dtype=float)
        N1 = np.repeat(n1, len(n2))
        N2 = np.tile(n2, len(n1))
        N3 = np.tile(n3, len(n1))
        Q = (G[0, 0] * N1 * N1 + G[1, 1] * N2 * N2 + G[2, 2] * N3 * N3
             + 2 * G[0, 1] * N1 * N2 + 2 * G[0, 2] * N1 * N3 + 2 * G[1, 2] * N2 * N3)
        c = np.cos(2 * np.pi * (th[0] * N1 + th[1] * N2 + th[2] * N3))
        zero = (N1 == 0) & (N2 == 0) & (N3 == 0)
        Q[zero] = np.inf
        tot += float(np.sum(c / Q ** 2))
    return tot


def richardson_fd(f, h):
    """5-point centred derivative of a scalar function f(s) at s=0."""
    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)


# ------------------------------------------------------------------ validation (task 1)
SHAPES = {
    "diagonal": [0.15, -0.08, 0, 0, 0],
    "noncommuting": [-0.07, 0.10, 0.20, -0.06, 0.05],
    "strong_noncommuting": [0.70, -0.30, 0.20, 0.10, -0.10],
}


def validate():
    t0 = time.time()
    rep = {}
    # ---- (a) weights [(1,0)] reproduce C40 ewald() ----
    C40 = load_C40()
    cm = Casimir(CONTENT_MAXWELL)
    rows = []
    shapes_a = dict(SHAPES)
    shapes_a["cube"] = [0] * 5
    shapes_a["aniso_mild"] = [0.9, -0.2, 0.4, 0.0, 0.3]
    for name, b in shapes_a.items():
        G = shape_from_beta(b)
        Zm, Dm = cm(G)
        Zr, Dr = C40.ewald(G)
        rows.append({"shape": name, "Z_mine": Zm, "Z_C40": Zr,
                     "Z_rel_diff": abs(Zm - Zr) / abs(Zr),
                     "D_rel_diff": float(np.max(np.abs(Dm - Dr)) / max(np.max(np.abs(Dr)), 1e-300)) if np.max(np.abs(Dr)) > 1e-8
                     else float(np.max(np.abs(Dm - Dr)))})
    rep["a_C40_match"] = {"rows": rows, "max_Z_rel_diff": max(r["Z_rel_diff"] for r in rows),
                          "max_D_diff": max(r["D_rel_diff"] for r in rows),
                          "BASIS_equals_C25": bool(np.max(np.abs(np.array(C40.BASIS) - BASIS)) == 0.0),
                          "pass_1e-12": bool(max(r["Z_rel_diff"] for r in rows) < 1e-12 and max(r["D_rel_diff"] for r in rows) < 1e-12)}
    # ---- (b) cube value and Hessian eigenvalues, (d) trace identity ----
    for tag, content in (("SM", CONTENT_SM), ("Maxwell", CONTENT_MAXWELL)):
        cas = Casimir(content)
        Zc, Dc = cas(I3)
        H1 = hessian_from_gradient(cas, 1e-3)
        H2 = hessian_from_gradient(cas, 2e-3)
        Hs = 0.5 * (H1 + H1.T)
        ev = np.linalg.eigvalsh(Hs)
        # Hessian from second differences of Z itself (independent of the gradient)
        def f2(a, b, h=2e-3):
            def f(x, y):
                return cas.Z(shape_from_beta(x * np.eye(5)[a] + y * np.eye(5)[b]))
            if a == b:
                return (-f(2 * h, 0) + 16 * f(h, 0) - 30 * Zc + 16 * f(-h, 0) - f(-2 * h, 0)) / (12 * h * h)
            return (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
        Hz = np.array([[f2(a, b) for b in range(5)] for a in range(5)])
        evz = np.linalg.eigvalsh(0.5 * (Hz + Hz.T))
        rep["b_cube_" + tag] = {
            "Zeff_cube": Zc, "D_cube_is_pure_trace_gauge": float(np.linalg.norm(Dc - np.trace(Dc) / 3 * I3)),
            "D_cube_trace_part_c": float(np.trace(Dc) / 3),
            "hess_from_gradient_h1e-3_eigs": ev.tolist(),
            "hess_asym_max": float(np.max(np.abs(H1 - H1.T))),
            "hess_h1e-3_vs_2e-3_max_abs_diff": float(np.max(np.abs(H1 - H2))),
            "hess_from_Z_second_diff_eigs": evz.tolist(),
            "trace_hess": float(np.trace(Hs)), "eight_thirds_Z": 8.0 / 3.0 * Zc,
            "d_trace_identity_rel": float(abs(np.trace(Hs) - 8.0 / 3.0 * Zc) / (8.0 / 3.0 * Zc)),
            "hess_matrix": Hs.tolist(),
        }
    sm = rep["b_cube_SM"]
    evs = np.array(sm["hess_from_gradient_h1e-3_eigs"])
    ref_T, ref_E = -893.2028574, 1902.2404336
    sm["eigs_E_mean"] = float(np.mean(evs[3:]))
    sm["eigs_T_mean"] = float(np.mean(evs[:3]))
    sm["E_split"] = float(np.ptp(evs[3:]))
    sm["T_split"] = float(np.ptp(evs[:3]))
    sm["rel_diff_E_vs_C57"] = float(abs(sm["eigs_E_mean"] - ref_E) / ref_E)
    sm["rel_diff_T_vs_C57"] = float(abs(sm["eigs_T_mean"] - ref_T) / abs(ref_T))
    sm["rel_diff_Z_vs_421.827111"] = float(abs(sm["Zeff_cube"] - 421.827111) / 421.827111)
    sm["kappa_E"] = 3 * sm["eigs_E_mean"] / sm["Zeff_cube"]
    sm["kappa_T"] = 3 * sm["eigs_T_mean"] / sm["Zeff_cube"]
    mx = rep["b_cube_Maxwell"]
    evm = np.array(mx["hess_from_gradient_h1e-3_eigs"])
    mx["eigs_E_mean"] = float(np.mean(evm[3:]))
    mx["eigs_T_mean"] = float(np.mean(evm[:3]))
    mx["kappa_E"] = 3 * mx["eigs_E_mean"] / mx["Zeff_cube"]
    mx["kappa_T"] = 3 * mx["eigs_T_mean"] / mx["Zeff_cube"]
    # ---- (c) gradient vs centred finite differences of Z at three non-cubic shapes ----
    rows = []
    for tag, content in (("SM", CONTENT_SM), ("Maxwell", CONTENT_MAXWELL)):
        cas = Casimir(content)
        for name, b in SHAPES.items():
            B = np.einsum("a,aij->ij", b, BASIS)
            Gh = expm(B)
            G = Gh @ Gh
            Z, D = cas(G)
            an = np.array([float(np.sum(D * (2 * Gh @ BASIS[a] @ Gh))) for a in range(5)])
            errs = {}
            for h in (1e-3, 2e-3):
                fd = np.array([richardson_fd(lambda s, a=a: cas.Z(Gh @ expm(2 * s * BASIS[a]) @ Gh), h) for a in range(5)])
                errs["h%g" % h] = float(np.max(np.abs(fd - an)) / np.max(np.abs(an)))
            for hp in (1e-4, 2e-5):      # plain 2-point centred differences (d/ds Z(Gh exp(2 s X) Gh), s = +-hp)
                fd2 = np.array([(cas.Z(Gh @ expm(2 * hp * BASIS[a]) @ Gh) - cas.Z(Gh @ expm(-2 * hp * BASIS[a]) @ Gh)) / (2 * hp)
                                for a in range(5)])
                errs["plain_centred_h%g" % hp] = float(np.max(np.abs(fd2 - an)) / np.max(np.abs(an)))
            # second tangent family: exp(2(B + eps X)) through the Frechet derivative
            _, g2 = grad_beta(cas, b)
            fd3 = np.array([richardson_fd(lambda s, a=a: cas.Z(shape_from_beta(np.asarray(b) + s * np.eye(5)[a])), 1e-3) for a in range(5)])
            errs["expchart_h1e-3"] = float(np.max(np.abs(fd3 - g2)) / np.max(np.abs(g2)))
            # trace-free force identity: tr(D G) part is pure gauge for det-one motion
            rows.append({"content": tag, "shape": name, "Z": Z, "grad_norm": float(np.linalg.norm(an)), **errs,
                         "max_rel_err": max(errs["h0.001"], errs["h0.002"], errs["expchart_h1e-3"], errs["plain_centred_h2e-05"])})
    rep["c_gradient_vs_FD"] = {"rows": rows, "max_rel_err": max(r["max_rel_err"] for r in rows),
                               "pass_1e-8": bool(max(r["max_rel_err"] for r in rows) < 1e-8)}
    # ---- cubic-group (signed permutation) covariance of Z_eff and D for the SM content ----
    cas_sm = Casimir(CONTENT_SM)
    rng = np.random.default_rng(3)
    worstZ, worstD = 0.0, 0.0
    for _ in range(6):
        G = shape_from_beta(rng.normal(scale=0.25, size=5))
        Z1, D1 = cas_sm(G)
        perm = rng.permutation(3)
        R = np.eye(3)[perm] * rng.choice([-1.0, 1.0], size=3)
        Z2, D2 = cas_sm(R @ G @ R.T)
        worstZ = max(worstZ, abs(Z2 - Z1) / abs(Z1))
        worstD = max(worstD, float(np.max(np.abs(D2 - R @ D1 @ R.T)) / np.max(np.abs(D1))))
    rep["cubic_group_covariance_SM"] = {"max_rel_Z_diff": worstZ, "max_rel_D_diff": worstD}
    # ---- independent check of the twisted Z itself: Ewald vs direct sum ----
    cas_aaa = Casimir([(1.0, THAAA)])
    rows = []
    for name, b in (("cube", [0] * 5), ("noncommuting", [-0.07, 0.10, 0.20, -0.06, 0.05])):
        G = shape_from_beta(b)
        ze = cas_aaa.Z(G)
        zd = z2_direct_cubebox(G, THAAA, R=60)
        rows.append({"shape": name, "Z_Ewald_AAA": ze, "Z_direct_R60": zd, "abs_diff": abs(ze - zd)})
    rep["Ewald_vs_direct_AAA"] = rows
    rep["runtime_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C60_validation_ewald.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    r = validate()
    a = r["a_C40_match"]
    print("(a) C40 match: max Z rel %.2e, max D %.2e, BASIS==C25 %s, pass %s" % (a["max_Z_rel_diff"], a["max_D_diff"], a["BASIS_equals_C25"], a["pass_1e-12"]))
    sm = r["b_cube_SM"]
    print("(b) SM cube Zeff=%.9f (rel diff vs 421.827111: %.1e), |trace-free part of D(cube)|=%.2e (D = c*1 gauge, c=%.3f)" % (
        sm["Zeff_cube"], sm["rel_diff_Z_vs_421.827111"], sm["D_cube_is_pure_trace_gauge"], sm["D_cube_trace_part_c"]))
    print("    Hessian E=%.6f (x2, split %.1e) rel %.1e ; T2=%.6f (x3, split %.1e) rel %.1e ; asym %.1e" % (
        sm["eigs_E_mean"], sm["E_split"], sm["rel_diff_E_vs_C57"], sm["eigs_T_mean"], sm["T_split"], sm["rel_diff_T_vs_C57"], sm["hess_asym_max"]))
    print("    kappa_E=%.6f kappa_T=%.6f ; Hess from Z-second-diff eigs: %s" % (sm["kappa_E"], sm["kappa_T"], np.round(sm["hess_from_Z_second_diff_eigs"], 4)))
    mx = r["b_cube_Maxwell"]
    print("    Maxwell Z=%.9f E=%.6f T2=%.6f kappa_E=%.6f kappa_T=%.6f" % (mx["Zeff_cube"], mx["eigs_E_mean"], mx["eigs_T_mean"], mx["kappa_E"], mx["kappa_T"]))
    c = r["c_gradient_vs_FD"]
    print("(c) gradient vs FD: max rel err %.2e pass %s | worst: Richardson h1e-3 %.1e, plain centred h2e-5 %.1e, plain h1e-4 %.1e, exp-chart %.1e" % (
        c["max_rel_err"], c["pass_1e-8"], max(x["h0.001"] for x in c["rows"]), max(x["plain_centred_h2e-05"] for x in c["rows"]),
        max(x["plain_centred_h0.0001"] for x in c["rows"]), max(x["expchart_h1e-3"] for x in c["rows"])))
    print("(d) tr Hess = 8/3 Z: SM rel %.2e, Maxwell rel %.2e" % (sm["d_trace_identity_rel"], mx["d_trace_identity_rel"]))
    cg = r["cubic_group_covariance_SM"]
    print("    cubic-group covariance (SM): Z %.1e, D %.1e" % (cg["max_rel_Z_diff"], cg["max_rel_D_diff"]))
    for row in r["Ewald_vs_direct_AAA"]:
        print("    AAA Ewald vs direct (%s): %.10f vs %.10f diff %.1e" % (row["shape"], row["Z_Ewald_AAA"], row["Z_direct_R60"], row["abs_diff"]))
    print("runtime %.1f s" % r["runtime_s"])
