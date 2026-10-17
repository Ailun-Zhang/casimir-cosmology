"""Helpers shared by the S-group (shape) figure scripts; style_v12 itself is imported unchanged.

Linear shape modes on the leading turning-point solution (PRD Sec. VII-VIII; v11 Eq. (shapeode)):
    L = L_b sqrt(cosh 2ht),   beta'' + 3 H beta' - kappa sech^2(2ht) beta = 0   (time in units of 1/h),
written as a first-order system for (beta, Pi) with Pi = L^3 beta' / C32, C32 = 3/2^(3/2). In this normalization
[beta_A, Pi_B] = 2i delta delta_AB, the transfer matrix from the turn to the far future is the m_ij of v11/D109
(m11 = 10.554, m12 = 2.468 at kappa_E = 19.3237; checked against C153 in `selftest`), the floor is
2|m11 m12| delta = |T12| delta, and a real Gaussian of width Var beta_0 = s has Var Pi_0 = delta^2/s.

G(|t_x|) (Theorem "real moment", D109): the larger outgoing variance of a state real at t_x is at least
    G(|t_x|) * floor,   G = |M11 M12|(-|t_x| -> inf) / |M11 M12|(0 -> inf);
the single-branch bound for t_x > 0 on the future branch is G_+(t_x) = |M11 M12|(t_x -> inf) / |M11 M12|(0 -> inf).

Agent fig-shape, 2026-10-15.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

import style_v12 as st

C32 = 3.0 / 2.0 ** 1.5
T_INF = 16.0          # h t at which the transfer is frozen to < 1e-12 relative (sech^2 ~ 4e-28)
MINUS = "−"


def _rhs(kappa):
    def f(t, y):
        L = math.sqrt(math.cosh(2.0 * t))
        L3 = L ** 3
        b1, q1, b2, q2 = y
        k = kappa / C32 / L
        return [C32 * q1 / L3, k * b1, C32 * q2 / L3, k * b2]
    return f


def transfer(kappa: float, t0: float, t1: float = T_INF) -> np.ndarray:
    """2x2 transfer matrix of (beta, Pi) from h t0 to h t1 on the leading solution."""
    s = solve_ivp(_rhs(kappa), (t0, t1), [1.0, 0.0, 0.0, 1.0], method="DOP853", rtol=1e-12, atol=1e-14)
    b1, q1, b2, q2 = s.y[:, -1]
    return np.array([[b1, b2], [q1, q2]])


def transfer_path(kappa: float, t0: float, ts: np.ndarray) -> np.ndarray:
    """Transfer matrices M(t0 -> t) for an increasing array ts (ts[0] >= t0); shape (len(ts), 2, 2)."""
    s = solve_ivp(_rhs(kappa), (t0, float(ts[-1])), [1.0, 0.0, 0.0, 1.0], t_eval=ts, method="DOP853",
                  rtol=1e-12, atol=1e-14)
    M = np.empty((len(ts), 2, 2))
    M[:, 0, 0], M[:, 1, 0], M[:, 0, 1], M[:, 1, 1] = s.y[0], s.y[1], s.y[2], s.y[3]
    return M


def G_two_branch(kappa: float, abs_tx) -> np.ndarray:
    """Lower bound G(|t_x|) on the larger outgoing variance (units of the floor) of a state real at t_x."""
    m = transfer(kappa, 0.0)
    f0 = abs(m[0, 0] * m[0, 1])
    out = []
    for a in np.atleast_1d(abs_tx):
        M = transfer(kappa, -float(a))
        out.append(abs(M[0, 0] * M[0, 1]) / f0)
    return np.array(out)


def G_plus_single(kappa: float, tx) -> np.ndarray:
    """Single-branch bound G_+(t_x) on the future-branch variance for a state real at t_x > 0."""
    m = transfer(kappa, 0.0)
    f0 = abs(m[0, 0] * m[0, 1])
    return np.array([abs(np.prod(transfer(kappa, float(a))[0, :])) / f0 for a in np.atleast_1d(tx)])


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def errbar(ax, x, p, lo, hi, **kw):
    """Asymmetric error bars from interval end points (Wilson intervals stored in the data files)."""
    p = np.asarray(p, float)
    yerr = np.vstack([p - np.asarray(lo, float), np.asarray(hi, float) - p])
    yerr = np.clip(yerr, 0.0, None)
    return ax.errorbar(x, p, yerr=yerr, **kw)


def plain_log(ax, axis: str = "y") -> None:
    """Decimal labels (0.01, 0.1, 1, 10, ...) on a log axis instead of 10^n, which keeps the exponent lettering
    from falling below the 7 pt floor; minor ticks stay unlabelled."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    f = FuncFormatter(lambda v, p: f"{v:g}".replace("-", MINUS))
    ax_ = ax.yaxis if axis == "y" else ax.xaxis
    ax_.set_major_formatter(f)
    ax_.set_minor_formatter(NullFormatter())


def fmt(v: float, nd: int) -> str:
    """Fixed-point number with a typographic minus sign."""
    return f"{v:.{nd}f}".replace("-", MINUS)


def check(name: str, value: float, expected: float, rtol: float) -> None:
    """Assert that a value recomputed in a figure script reproduces the archived value."""
    if not math.isclose(value, expected, rel_tol=rtol, abs_tol=0.0):
        raise AssertionError(f"{name}: {value!r} vs archived {expected!r} (rtol {rtol})")


def selftest() -> None:
    c153 = st.load_json("shape/C153_summary.json")["B_next_order"]
    for key in ("19.3237", "18.0"):
        ref = c153["floor_closed_form"][key]
        m = transfer(ref["kappa"], 0.0)
        check(f"m11({key})", m[0, 0], ref["m11"], 1e-7)
        check(f"m12({key})", m[0, 1], ref["m12"], 1e-7)
        check(f"m21({key})", m[1, 0], ref["m21"], 1e-7)
        check(f"m22({key})", m[1, 1], ref["m22"], 1e-7)
    tab = c153["G_tables"]["E_xi030"]
    for r in tab["rows"][:5]:
        check(f"G({r['tau']})", G_two_branch(tab["kappa"], r["tau"])[0], r["G_two_branch"], 1e-6)
        check(f"G+({r['tau']})", G_plus_single(tab["kappa"], r["tau"])[0], r["G_plus_single_branch"], 1e-6)
    print("selftest passed: transfer matrices and G tables reproduce C153")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    selftest()
