# PROVENANCE: 04_calculations/code/20261009_claude_R5/nonlinear_SM/C60_nonlinear_integrator.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Nonlinear volume + five-shape-modulus integrator on the leading cosh background (C60, D49b): Hamiltonian constraint, DOP853, pinch/collapse classification; linear transfer. Shared library.

C60_nonlinear_integrator: forward integration of the full-matrix shape + volume system from a prepared
turning point (H = 0), for general twisted Casimir content (WP R5-NL).

Variables (units h = 1, L_b(cube) = 1 with Z0 := Z_eff(1)):  lr = ln(L/L_b), H, G (det 1), M = e^{3 lr} K,
K = Gdot/2.  Equations (identical in form to C40 Model.pieces/rhs, with Z0 -> Z_eff(1) and the twisted
Z_eff, D of C60_twisted_ewald):
    lr' = H,  H' = 2 - 2H^2 - S/6,  S = tr(sigma^2),  sigma = e^{-3 lr} G^-1 M,
    G' = 2 e^{-3 lr} M,  M' = 2 e^{-3 lr} M G^-1 M + e^{3 lr} G pi,  pi = (6 e^{-4 lr}/Z0) (D G)_TF,
    constraint  3H^2 = 3 - 3 (Z/Z0) e^{-4 lr} + S/2.
The constraint is conserved by the equations for any Z (dC/dt = 0 uses only dZ/dt = tr(D Gdot)); tr(G^-1 M) = 0
and det G = 1 are conserved too.

Turning-point data.  Shape coordinates beta_A:  G0 = exp(2 sum_A beta_A X_A).  Canonical momentum
Pi_A = P_A/p0 with  p0 = 3 M_P^2 h L_b^3 / 2^{3/2},  [beta, Pi] = 2 i delta.  At G = 1 (linear order)
M = C32 sum_A Pi_A X_A, C32 = 3/2^{3/2}.  At finite beta we use (ic = 'on', primary)
    M0 = G0^{1/2} (C32 sum_A Pi_A X_A) G0^{1/2},
i.e. the orthonormal-frame momentum m_on = G0^{-1/2} M0 G0^{-1/2} = C32 sum Pi_A X_A (this is the definition used by
C25 shape_data and satisfies tr(G0^-1 M0) = 0 exactly).  Variants for the sensitivity test:
  'canonical': Pi is the momentum conjugate to the exp-chart beta, p_can = f(ad_B) m_on, f = sinh(ad_B)/ad_B
               (differs from 'on' at O(beta^2) ~ 1e-4);
  'trfree'   : M0 = Mlin - [tr(G0^-1 Mlin)/3] G0 with Mlin = C32 sum Pi_A X_A (a deliberately cruder O(beta)
               change of the initial momentum).
lr0 solves the constraint at H = 0:  e^{4 lr0} + (s0/6) e^{-2 lr0} = Z(G0)/Z0,  s0 = tr((G0^-1 M0)^2).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from C60_twisted_ewald import (Casimir, BASIS, ControlExit, I3, CONTENT_SM, CONTENT_MAXWELL,
                               shape_from_beta, beta_from_shape)

C32 = 3.0 / 2 ** 1.5
IU = np.triu_indices(3)


def pack(lr, H, G, M):
    return np.concatenate(([lr, H], G[IU], M[IU]))


def unpack(y):
    G = np.zeros((3, 3))
    M = np.zeros((3, 3))
    G[IU] = y[2:8]
    M[IU] = y[8:14]
    G = G + np.triu(G, 1).T
    M = M + np.triu(M, 1).T
    return y[0], y[1], G, M


def geometry(G):
    """Side lengths sqrt(G_ii), ln(Lmax/Lmin), maximal angle deviation from 90 deg (C40 definitions)."""
    s = np.sqrt(np.diag(G))
    ca = np.array([G[i, j] / (s[i] * s[j]) for i, j in ((0, 1), (0, 2), (1, 2))])
    return {"sides": s.tolist(), "ln_Lmax_over_Lmin": float(np.log(s.max() / s.min())),
            "angles_deg": np.degrees(np.arccos(np.clip(ca, -1, 1))).tolist(),
            "max_angle_deviation_deg": float(np.max(np.abs(np.degrees(np.arcsin(np.clip(ca, -1, 1))))))}


def ad2_matrix(B):
    """Matrix of X -> [B,[B,X]] on the 5-dim space of symmetric traceless matrices, basis BASIS."""
    A = np.zeros((5, 5))
    for b in range(5):
        c1 = B @ BASIS[b] - BASIS[b] @ B
        c2 = B @ c1 - c1 @ B
        for a in range(5):
            A[a, b] = np.trace(BASIS[a] @ c2)
    return 0.5 * (A + A.T)


class NLModel:
    def __init__(self, content, cut=42.0, rtol=1e-11, atol=1e-13, max_step=0.2, eig_stop=1e-3):
        self.cas = Casimir(content, cut)
        self.Z0 = self.cas(I3)[0]
        self.rtol, self.atol, self.max_step, self.eig_stop = rtol, atol, max_step, eig_stop
        self.nfev = 0

    # ------------------------------------------------------------ equations of motion
    def pieces(self, lr, G, M):
        Gi = np.linalg.inv(G)
        e3 = np.exp(-3 * lr)
        sig = e3 * (Gi @ M)
        S = float(np.trace(sig @ sig))
        Z, D = self.cas(G)
        dg = D @ G
        pi = (6 * np.exp(-4 * lr) / self.Z0) * (dg - np.trace(dg) / 3 * I3)
        gd = 2 * e3 * M
        md = 2 * e3 * (M @ Gi @ M) + np.exp(3 * lr) * (G @ pi)
        return S, Z, gd, md

    def rhs(self, t, y):
        self.nfev += 1
        lr, H, G, M = unpack(y)
        S, Z, gd, md = self.pieces(lr, G, M)
        return np.concatenate(([H, 2 - 2 * H * H - S / 6], gd[IU], (0.5 * (md + md.T))[IU]))

    def constraint(self, y):
        lr, H, G, M = unpack(y)
        S, Z, gd, md = self.pieces(lr, G, M)
        return float(3 * H * H - 3 + 3 * Z / self.Z0 * np.exp(-4 * lr) - S / 2)

    # ------------------------------------------------------------ initial data at the turning point
    def initial_state(self, beta0, Pi0, ic="on"):
        beta0 = np.asarray(beta0, float)
        Pi0 = np.asarray(Pi0, float)
        B = np.einsum("a,aij->ij", beta0, BASIS)
        G0 = expm(2 * B)
        Gh = expm(B)
        mvec = C32 * Pi0
        if ic == "on":
            m_on = np.einsum("a,aij->ij", mvec, BASIS)
            M0 = Gh @ m_on @ Gh
        elif ic == "canonical":
            a2 = ad2_matrix(B)
            w, V = np.linalg.eigh(a2)
            w = np.clip(w, 0.0, None)
            sw = np.sqrt(w)
            finv = np.where(sw > 1e-12, sw / np.sinh(np.where(sw > 1e-12, sw, 1.0)), 1.0)   # 1/f(ad_B)
            mon_vec = V @ (finv * (V.T @ mvec))
            m_on = np.einsum("a,aij->ij", mon_vec, BASIS)
            M0 = Gh @ m_on @ Gh
        elif ic == "trfree":
            Mlin = np.einsum("a,aij->ij", mvec, BASIS)
            M0 = Mlin - np.trace(np.linalg.inv(G0) @ Mlin) / 3 * G0
        else:
            raise ValueError(ic)
        M0 = 0.5 * (M0 + M0.T)
        Gi0 = np.linalg.inv(G0)
        s0 = float(np.trace((Gi0 @ M0) @ (Gi0 @ M0)))
        zeta = self.cas(G0)[0] / self.Z0
        # u = e^{2 lr0}: u^2 + (s0/6)/u = zeta on the branch u^3 >= s0/12 (large-volume root)
        u = np.sqrt(max(zeta, 1e-12))
        for _ in range(100):
            f = u * u + (s0 / 6) / u - zeta
            fp = 2 * u - (s0 / 6) / u ** 2
            du = f / fp
            u -= du
            if abs(du) < 1e-15 * abs(u):
                break
        lr0 = 0.5 * float(np.log(u))
        y0 = pack(lr0, 0.0, G0, M0)
        return y0, {"lr0": lr0, "s0": s0, "Z_over_Z0": zeta,
                    "trace_GinvM0": float(np.trace(Gi0 @ M0)), "constraint0": self.constraint(y0)}

    # ------------------------------------------------------------ forward run
    def forward(self, beta0, Pi0, ic="on", T=12.0, label=""):
        t_start = time.perf_counter()
        self.nfev = 0
        rec = {"label": label, "ic": ic, "beta0": list(map(float, beta0)), "Pi0": list(map(float, Pi0)), "T": T,
               "rtol": self.rtol, "atol": self.atol, "cut": self.cas.cut}
        try:
            y0, info = self.initial_state(beta0, Pi0, ic)
            rec.update(info)

            def pinch(t, y):
                _, _, G, _ = unpack(y)
                return float(np.linalg.eigvalsh(G)[0]) - self.eig_stop
            pinch.terminal = True
            pinch.direction = -1

            def volume(t, y):
                return y[0] + 1.0
            volume.terminal = True
            volume.direction = -1

            sol = solve_ivp(self.rhs, (0.0, T), y0, method="DOP853", rtol=self.rtol, atol=self.atol,
                            max_step=self.max_step, events=(pinch, volume), dense_output=True)
            rec["t_end"] = float(sol.t[-1])
            rec["nsteps"] = int(len(sol.t))
            ye = sol.y[:, -1]
            rec["H_end"] = float(ye[1])
            rec["lr_end"] = float(ye[0])
            if len(sol.t_events[0]):
                rec.update({"status": "pinch", "reason": "min eig G reached %.0e at t=%.4f" % (self.eig_stop, sol.t_events[0][0])})
            elif len(sol.t_events[1]):
                rec.update({"status": "recollapse", "reason": "ln(L/L_b) < -1 at t=%.4f" % sol.t_events[1][0]})
            elif sol.status != 0 or abs(sol.t[-1] - T) > 1e-9:
                rec.update({"status": "solver_fail", "reason": str(sol.message)})
            else:
                rec["status"] = "complete"
            # trajectory diagnostics on the accepted steps
            if rec["status"] in ("complete", "pinch", "recollapse"):
                ts = sol.t
                mineig = []
                maxb = []
                for k in range(0, len(ts), max(1, len(ts) // 80)):
                    G = unpack(sol.y[:, k])[2]
                    mineig.append(float(np.linalg.eigvalsh(G)[0]))
                    if mineig[-1] > 1e-6:
                        maxb.append(float(np.max(np.abs(beta_from_shape(G / np.linalg.det(G) ** (1 / 3))))))
                rec["min_eigG_path"] = float(min(mineig))
                rec["max_abs_beta_path"] = float(max(maxb)) if maxb else None
            if rec["status"] == "complete":
                cons = [abs(self.constraint(sol.y[:, k])) for k in range(sol.y.shape[1])]
                detd = [abs(np.linalg.det(unpack(sol.y[:, k])[2]) - 1.0) for k in range(sol.y.shape[1])]
                trg = [abs(np.trace(np.linalg.inv(unpack(sol.y[:, k])[2]) @ unpack(sol.y[:, k])[3])) for k in range(sol.y.shape[1])]
                Graw = unpack(ye)[2]
                G10 = unpack(sol.sol(10.0))[2]
                rec["gate_constraint_max"] = float(max(cons))
                rec["gate_detG_minus1_max"] = float(max(detd))
                rec["gate_trGinvM_max"] = float(max(trg))
                rec["gate_G_change_10_to_12"] = float(np.max(np.abs(Graw - G10)))
                rec["gate_G_change_10_to_12_rel"] = float(np.max(np.abs(Graw - G10)) / np.max(np.abs(Graw)))
                Gp = Graw / np.linalg.det(Graw) ** (1 / 3)
                rec["G_plus"] = Gp.tolist()
                rec["beta_plus"] = beta_from_shape(Gp).tolist()
                rec["geometry_plus"] = geometry(Gp)
                rec["gates_pass"] = bool(rec["gate_constraint_max"] < 1e-8 and rec["gate_detG_minus1_max"] < 1e-8
                                         and rec["gate_G_change_10_to_12"] < 1e-8)
                rec["survivor"] = bool(rec["min_eigG_path"] > 1e-3 and rec["H_end"] > 0.9)
                if not rec["survivor"]:
                    rec["status"] = "no_exit" if rec["H_end"] <= 0.9 else "pinch"
            else:
                rec["survivor"] = False
        except ControlExit as exc:
            rec.update({"status": "control_exit", "reason": str(exc), "survivor": False})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rec.update({"status": "numerical_error", "reason": str(exc), "survivor": False})
        rec["nfev"] = int(self.nfev)
        rec["runtime_s"] = float(time.perf_counter() - t_start)
        return rec


# ------------------------------------------------------------------ linear transfer from the turning point
def linear_transfer(kappa, tf=14.0):
    """M = [[m11, m12], [m21, m22]]: (beta_0, Pi_0) at t=0 -> (beta_+, Pi_+); the leading cosh background,
    beta'' + 3 tanh(2t) beta' - kappa sech^2(2t) beta = 0, beta' = C32 Pi cosh^-3/2(2t) (cf. C58)."""
    def rhs(t, y):
        c = np.cosh(2 * t)
        return [C32 * y[1] / c ** 1.5, kappa * c ** -0.5 * y[0] / C32]
    cols = []
    for y0 in ([1.0, 0.0], [0.0, 1.0]):
        s = solve_ivp(rhs, (0.0, tf), y0, method="DOP853", rtol=1e-13, atol=1e-15)
        cols.append(s.y[:, -1])
    return np.array(cols).T


MODE_TYPE = ["aspect", "aspect", "angle", "angle", "angle"]
