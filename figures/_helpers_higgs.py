"""Shared data access and small derived quantities for the Higgs (H-group) figures.

Everything is read from ./data/higgs (see ./data/SOURCES_higgs.json). Derived curves computed here:
  * P1(m_t) at an arbitrary alpha_s: linear interpolation of y = ln(-ln P1) between the two neighbouring alpha_s rows
    of the run05 claim-layer grid (rows every 0.0002; checked against the archived critical masses, Delta <= 2 MeV);
  * the pole -> m-bar_t(m-bar_t) map at alpha_s = 0.1180: straight line through the QCD four-loop points of the claim
    configuration (checked against the run05 critical m-bar values);
  * the universal zero-mode threshold xi_c(P1): least-squares polynomial in x = ln(-ln P1) through all directly
    computed (P1, xi_c) pairs (run05 centre, run02 centre, and the 16 run02 SM/alpha_s variants of C157); D129/D132
    found xi_c to depend on m_t, alpha_s and the SM mapping only through P1 (spread <= 0.002).
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np

import style_v12 as S

HD = S.DATA / "higgs"
A0 = 0.1180


def J(rel: str):
    return json.loads((HD / rel).read_text(encoding="utf-8"))


def C(rel: str) -> list[dict]:
    with open(HD / rel, newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows])


# ------------------------------------------------------------------------------------------------ P1(m_t, alpha_s)
class P1Grid:
    def __init__(self):
        rows = C("claims_run05/fig2a_P1_grid_mt_alphas.csv")
        self.m = np.unique(col(rows, "m_t"))
        self.a = np.unique(col(rows, "alpha_s"))
        P = np.full((len(self.a), len(self.m)), np.nan)
        Ps = np.full_like(P, np.nan)
        ia = {round(x, 5): i for i, x in enumerate(self.a)}
        im = {round(x, 4): i for i, x in enumerate(self.m)}
        for r in rows:
            i, j = ia[round(float(r["alpha_s"]), 5)], im[round(float(r["m_t"]), 4)]
            P[i, j] = float(r["P1_central"])
            Ps[i, j] = float(r["P1_theory_smeared"])
        self.P, self.Ps = P, Ps

    @staticmethod
    def _y(P):
        P = np.clip(P, 1e-300, 1 - 1e-15)
        return np.log(-np.log(P))

    def curve(self, alpha: float, smeared: bool = False) -> np.ndarray:
        Z = self.Ps if smeared else self.P
        k = int(np.searchsorted(self.a, alpha - 1e-12))
        if abs(self.a[min(k, len(self.a) - 1)] - alpha) < 1e-9:
            return Z[k].copy()
        k = min(max(k, 1), len(self.a) - 1)
        t = (alpha - self.a[k - 1]) / (self.a[k] - self.a[k - 1])
        y = (1 - t) * self._y(Z[k - 1]) + t * self._y(Z[k])
        return np.exp(-np.exp(y))

    def at(self, m, alpha: float = A0):
        return np.interp(m, self.m, self.curve(alpha))

    def mcrit(self, p: float, alpha: float = A0) -> float:
        y = self._y(self.curve(alpha))           # increasing in m_t
        return float(np.interp(math.log(-math.log(p)), y, self.m))


def critical(alpha: float = A0) -> dict:
    """archived critical pole masses (run05) for P1 = 0.9/0.5/0.1."""
    c = J("holonomy_cross_run05/C149_run05_summary.json")["crit"][f"{alpha:.4f}"]
    return {p: c[f"P1={p}"] for p in ("0.9", "0.5", "0.1")}


def lam_c() -> dict:
    c = J("holonomy_cross_run05/C149_run05_summary.json")["lam_c"]
    return {p: c[f"P1={p}"] for p in ("0.9", "0.5", "0.1")}


# ------------------------------------------------------------------------------------------------ m_t definitions
def mbar_map():
    """linear map pole -> m-bar_t(m-bar_t) at alpha_s = 0.1180 (QCD four loops); returns (f, f_inv, max check dev)."""
    pts = J("claims_run05/C154_inputs_v2_run05.json")["msbar_points"]
    b, a = np.polyfit(pts["pole"], pts["mbar"], 1)
    f = lambda m: a + b * np.asarray(m)          # noqa: E731
    finv = lambda mb: (np.asarray(mb) - a) / b    # noqa: E731
    nc = J("sm_mapping_run02/R3_mbar_run05.json")["new_central"]["0.1180"]
    dev = max(abs(float(f(v["pole"])) - v["mbar_mbar_QCD4L"]) for v in nc.values())
    return f, finv, dev


def measurements() -> dict:
    m = J("claims_run05/C154_summary.json")["marginal"]
    return {k: dict(c=v["center_pole4L_equiv"], sl=v["s_left"], sr=v["s_right"], label=v["label"]) for k, v in m.items()}


def future(scenario: str = "FCCee_base") -> dict:
    rows = [r for r in C("claims_run05/future_reach.csv") if r["scenario"] == scenario]
    cfg = {s["id"]: s for s in J("claims_run05/C154_inputs_v2_run05.json")["future"]}[scenario]
    return dict(sigma_meas=float(rows[0]["sigma_meas"]), sigma_tot=float(rows[0]["sigma_tot"]),
                s_as=cfg["s_as"], kappa=cfg["kappa"], excl=float(rows[0]["mhat_excl_P0p1_95"]),
                surv=float(rows[0]["mhat_surv95"]))


def ellipse(cx, cy, cov, k2, n=181):
    w, v = np.linalg.eigh(np.asarray(cov, float))
    t = np.linspace(0, 2 * np.pi, n)
    pts = (v @ np.diag(np.sqrt(w * k2)) @ np.vstack([np.cos(t), np.sin(t)]))
    return cx + pts[0], cy + pts[1]


K2 = {"68": 2.2789, "95": 5.9915}  # chi^2 quantiles for two parameters (same as the claim-layer ellipses)


# ------------------------------------------------------------------------------------------------ xi_c(P1)
def xi_points() -> list[tuple[float, float, str]]:
    pts = []
    xs = J("claims_run05/xi_run05_summary.json")
    for k, v in xs["run05"].items():
        pts.append((v["P1_2D_run05"], v["xi_c"], f"run05 {k}"))
    for k, v in xs["run02"].items():
        pts.append((v["P1"], v["xi_c"], f"run02 {k}"))
    c157 = J("realtime_2d_run02/C157_run02_summary.json")
    for k, v in c157["variants"].items():
        pts.append((c157["P1"][k], v["merged"], f"run02 variant {k}"))
    return pts


class XiUniversal:
    """xi_c(mu_b) as a function of P1: monotone cubic (PCHIP) in x = ln(-ln P1) through the direct points, after
    averaging points closer than `dx` in x (these differ by <= 0.0015 and would otherwise make the interpolant wiggle).
    Defined only inside the range of the direct points (NaN outside)."""

    def __init__(self, dx: float = 0.12):
        from scipy.interpolate import PchipInterpolator
        pts = xi_points()
        P = np.array([p[0] for p in pts])
        xi = np.array([p[1] for p in pts])
        x = np.log(-np.log(P))
        o = np.argsort(x)
        x, xi = x[o], xi[o]
        bx, bxi, cur = [], [], [0]
        for i in range(1, len(x) + 1):
            if i == len(x) or x[i] - x[cur[0]] > dx:
                bx.append(float(np.mean(x[cur])))
                bxi.append(float(np.mean(xi[cur])))
                cur = [i]
            else:
                cur.append(i)
        self.f = PchipInterpolator(bx, bxi)
        self.bins = list(zip(bx, bxi))
        self.xmin, self.xmax = float(x.min()), float(x.max())
        self.Pmin, self.Pmax = float(P.min()), float(P.max())
        self.resid = xi - self.f(np.clip(x, bx[0], bx[-1]))
        self.n = len(pts)

    def __call__(self, P):
        P = np.asarray(P, float)
        x = np.log(-np.log(np.clip(P, 1e-300, 1 - 1e-15)))
        lo, hi = self.bins[0][0], self.bins[-1][0]
        out = self.f(np.clip(x, lo, hi))
        return np.where((x >= lo - 1e-9) & (x <= hi + 1e-9), out, np.nan)

    def extrapolated(self, P, P_low: float = 0.1):
        """beyond the last direct point (P1 = Pmin) down to P_low: straight line in x through the last two bins
        (used only for the short dashed continuation to P1 = 0.1; NaN elsewhere outside the direct range)."""
        P = np.asarray(P, float)
        x = np.log(-np.log(np.clip(P, 1e-300, 1 - 1e-15)))
        (x1, y1), (x2, y2) = self.bins[-2], self.bins[-1]
        lin = y2 + (y2 - y1) / (x2 - x1) * (x - x2)
        xl = math.log(-math.log(P_low))
        return np.where((x > x2) & (x <= xl + 1e-9), lin, np.nan)


def xi_run05() -> dict:
    return J("claims_run05/xi_run05_summary.json")["run05"]


# ------------------------------------------------------------------------------------------------ output
def save(fig, name: str, target: float, papers=("prl",)):
    """style_v12.save with the tight-box width chosen so that the PDF page (tight box + 2 x 0.02 in padding) has the
    column width `target`; the page width is then verified with PyMuPDF and the call repeated once if needed."""
    import fitz
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    w = target - 0.04
    paths = S.save(fig, name, width=w, papers=papers)
    page = fitz.open(paths[0])[0].rect.width / 72
    if abs(page - target) > 0.008:
        FigureCanvasAgg(fig)  # style_v12.save closed the figure; re-attach a canvas
        w += target - page
        paths = S.save(fig, name, width=w, papers=papers)
        page = fitz.open(paths[0])[0].rect.width / 72
    return paths, page


# ------------------------------------------------------------------------------------------------ formatting
def sci(x: float, digits: int = 1) -> str:
    """mathtext scientific notation, e.g. 4.2\\times10^{-4}."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return rf"{m:.{digits}f}\times10^{{{e}}}"


def minus(s: str) -> str:
    return s.replace("-", "−")
