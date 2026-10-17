# -*- coding: utf-8 -*-
"""
R15 (r15-cmb) independent check of the C145 / D117 CMB shape-ensemble layer.   (v2)

Data input ONLY: producer fossil_table.csv (readout D for shapes (A, theta, s=L0/d_LSS)) and, in
"poolmin" mode, kl_table.json (55 fossils x 19 cubes KL) from which the pool-min profile is rebuilt.
Covariance / KL construction is NOT recomputed.

Ensemble: beta = A*sig_E*(cos th X_d + sin th X_d2), A ~ Rayleigh(1) (2D isotropic Gaussian,
per-component sd sig_E), th uniform; D3 fundamental domain th in [-pi/6, pi/6] (xi = sin 3th).
P_joint(s) = P[D >= 4.5  and  u >= 0.985],  u = s * min_i exp(beta_ii) (shortest side / d_LSS)
P_cond(s)  = P_joint / P[u >= 0.985]

Method A : per (A,xi) node, D=4.5 crossing u_hi in u (log-linear in u); u_hi interpolated in A (PCHIP)
           and xi (quadratic Lagrange through xi=-1,0,1); detection <=> 0.985 <= u <= u_hi(A,xi).
           Integrated on a fine (A,theta) midpoint grid (exact sort/cumsum in ln s).
Method B : direct interpolation of ln D over (A, xi) at fixed s (node ln D(s) log-linear in s).
MC       : full-plane Gaussian draws (no fundamental-domain reduction; xi from tr X^3 invariant,
           u from the actual min diagonal) pushed through the Method-A contour.
Usage: python r15_ensemble_check.py            -> table = fossil_table.csv (parabolic profile)
       python r15_ensemble_check.py poolmin    -> prof columns replaced by pool-min from kl_table.json
"""
import csv, json, math, sys, time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

T0 = time.time()
POOLMIN = "poolmin" in sys.argv
SUFFIX = "_poolmin" if POOLMIN else ""
ROOT = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology"
RUN = ROOT + r"\04_calculations\results\20261013_argon_codex_intake\cmb_ensemble\run01\results"
CSV = RUN + r"\fossil_table.csv"
OUT = ROOT + r"\07_reviews\R15_v12_round2\cmb_ensemble"

SIG_E, UCUT, DTH = 0.1041, 0.985, 4.5
LNDTH, LNUCUT = math.log(DTH), math.log(UCUT)
DATASETS = ["TE30", "TE40", "TE80", "T80"]
KINDS = ["prof", "fixed"]
A_NODES = np.array([0.25, 0.5, 1.0, 1.41421, 2.0])
LABS = ["obl", "tri", "pro"]                     # xi = -1, 0, +1
SQ2, SQ6 = math.sqrt(2.0), math.sqrt(6.0)
SPTS = [0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10,
        1.12, 1.13, 1.14, 1.16, 1.18, 1.20, 1.22, 1.25]


def diag_coeffs(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c / SQ2 + s / SQ6, -c / SQ2 + s / SQ6, -2 * s / SQ6], axis=-1)


def mmin(theta):
    return diag_coeffs(theta).min(axis=-1)


def jd(o):
    if isinstance(o, dict):
        return {str(k): jd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jd(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jd(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------- load
rows = []
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        rec = dict(name=r["name"], A=float(r["A"]), lab=r["lab"], theta=float(r["theta"]),
                   xi=float(r["xi"]), s=float(r["s"]), u=float(r["u"]), s_low=float(r["s_low"]))
        for d in DATASETS:
            for k in ("fixed", "prof", "sprof"):
                v = r[f"{d}_{k}"].strip()
                rec[f"{d}_{k}"] = float(v) if v else float("nan")
        rows.append(rec)
RES = {"mode": "poolmin" if POOLMIN else "parabolic_csv", "n_rows": len(rows),
       "setup": dict(sig_E=SIG_E, u_cut=UCUT, D_th=DTH)}

# ---------------------------------------------------------------- pool-min rebuild (poolmin mode)
if POOLMIN:
    with open(RUN + r"\kl_table.json", encoding="utf-8") as f:
        KL = json.load(f)
    cmp_, kl_dpi, nent = [], [], 0
    for r in rows:
        cubes = sorted((float(k), v) for k, v in KL[r["name"]].items())
        scs = np.array([sc for sc, _ in cubes])
        for sc, v in cubes:
            nent += 1
            x = {d: v[d][0] for d in DATASETS}
            for a, b_ in [("TE30", "TE40"), ("TE40", "TE80"), ("TE30", "TE80"), ("T80", "TE80")]:
                if x[a] > x[b_] * (1 + 1e-9):
                    kl_dpi.append(dict(row=r["name"], cube=sc, rule=f"{a}<={b_}", lhs=x[a], rhs=x[b_]))
        for d in DATASETS:
            vals = np.array([v[d][0] for _, v in cubes])
            i = int(np.argmin(vals)); pm = float(vals[i])
            fk = [v[d][0] for sc, v in cubes if abs(sc - r["s"]) < 1e-6]
            cmp_.append(dict(row=r["name"], A=r["A"], lab=r["lab"], s=r["s"], u=r["u"], ds=d,
                             parab=r[f"{d}_prof"], poolmin=pm, s_argmin=float(scs[i]),
                             argmin_edge=("low" if i == 0 else ("high" if i == len(vals) - 1 else "")),
                             rel_parab_minus_pool=(r[f"{d}_prof"] - pm) / pm,
                             fixed_csv=r[f"{d}_fixed"], fixed_kl=(fk[0] if fk else None)))
            r[f"{d}_prof"] = pm; r[f"{d}_sprof"] = float(scs[i])
    RES["poolmin"] = dict(
        n_kl_entries=nent, cube_list=[float(c) for c in scs], kl_dpi_violations=kl_dpi,
        parab_undershoot_gt1pct=[c for c in cmp_ if c["rel_parab_minus_pool"] < -0.01],
        parab_above_pool=[c for c in cmp_ if c["rel_parab_minus_pool"] > 1e-6],
        argmin_at_pool_edge=[c for c in cmp_ if c["argmin_edge"]],
        fixed_mismatch=[c for c in cmp_ if c["fixed_kl"] is not None and np.isfinite(c["fixed_csv"])
                        and abs(c["fixed_kl"] - c["fixed_csv"]) > 1e-4 * max(1.0, abs(c["fixed_csv"]))],
        n_fixed_compared=sum(1 for c in cmp_ if c["fixed_kl"] is not None and np.isfinite(c["fixed_csv"])),
        second_element_examples=[KL[rows[0]["name"]][k]["TE80"] for k in list(KL[rows[0]["name"]])[:3]])

# ---------------------------------------------------------------- 0. geometry + DPI checks
g = []
for r in rows:
    dc = diag_coeffs(r["theta"]); m = dc.min()
    g.append([abs(r["s"] * math.exp(r["A"] * SIG_E * m) - r["u"]),
              abs(UCUT / math.exp(r["A"] * SIG_E * m) - r["s_low"]),
              abs(math.sin(3 * r["theta"]) - r["xi"]),
              abs(SQ6 * np.sum(dc ** 3) - math.sin(3 * r["theta"])),
              abs(np.sum(dc ** 2) - 1.0), abs(np.sum(dc))])
g = np.array(g)
RES["geometry_maxdev"] = dict(u=g[:, 0].max(), s_low=g[:, 1].max(), xi=g[:, 2].max(),
                              sqrt6_trX3_minus_sin3th=g[:, 3].max(), trX2_minus_1=g[:, 4].max(),
                              trX=g[:, 5].max())
viol = []
for r in rows:
    for kind in KINDS:
        v = {d: r[f"{d}_{kind}"] for d in DATASETS}
        for a, b in [("TE30", "TE40"), ("TE40", "TE80"), ("TE30", "TE80"), ("T80", "TE80")]:
            if np.isfinite(v[a]) and np.isfinite(v[b]) and v[a] > v[b] * (1 + 1e-9):
                viol.append(dict(row=r["name"], kind=kind, rule=f"{a}<={b}", lhs=v[a], rhs=v[b]))
    for d in DATASETS:
        if np.isfinite(r[f"{d}_fixed"]) and r[f"{d}_prof"] > r[f"{d}_fixed"] * (1 + 1e-9):
            viol.append(dict(row=r["name"], kind="prof<=fixed", rule=d, lhs=r[f"{d}_prof"],
                             rhs=r[f"{d}_fixed"]))
RES["dpi_violations"] = viol


# ---------------------------------------------------------------- node data
def node_pts(col, override=None):
    nodes = {}
    for lab in LABS:
        for A in A_NODES:
            pts = []
            for r in rows:
                if r["lab"] != lab or abs(r["A"] - A) > 1e-4:
                    continue
                val = r[col]
                if override and col in override:
                    key = (lab, round(float(A), 5), round(r["s"], 5))
                    if key in override[col]:
                        val = override[col][key]
                if val is None or not np.isfinite(val):
                    continue
                pts.append((r["u"], val, r["s"]))
            pts.sort()
            nodes[(lab, float(A))] = np.array(pts)
    return nodes


def crossing(u, D, method="loglin"):
    u = np.asarray(u, float); D = np.asarray(D, float)
    lnD = np.log(D); y = lnD - LNDTH; n = len(u)
    down = [k for k in range(n - 1) if y[k] >= 0 and y[k + 1] < 0]
    mono = bool(np.all(np.diff(lnD) < 0))
    flag = [] if mono else ["nonmono"]
    if down:
        k = down[-1]
        if len(down) > 1:
            flag.append("multi")
    elif np.all(y >= 0):
        k = n - 2; flag.append("extrap_high")
    else:
        k = 0; flag.append("extrap_low")
    u0, u1 = u[k], u[k + 1]
    slope = (lnD[k] - lnD[k + 1]) / (u1 - u0)
    uh = u0 + (lnD[k] - LNDTH) / slope                    # log-linear chord (= convexity upper bound)
    extrap = any(fl.startswith("extrap") for fl in flag)
    if method == "linD" and not extrap:
        uh = u0 + (D[k] - DTH) * (u1 - u0) / (D[k] - D[k + 1])
    elif method == "pchip" and not extrap and n >= 3 and mono:
        f = PchipInterpolator(u, y)
        uh = brentq(lambda t: float(f(t)), u0, u1)
    elif method == "convex_lo" and not extrap and k >= 1:   # extend steeper previous segment
        sp = (lnD[k - 1] - lnD[k]) / (u[k] - u[k - 1])
        if sp > slope:
            uh = u0 + (lnD[k] - LNDTH) / sp
    elif method == "convex_hi" and "extrap_high" in flag:  # shallow slope 60 beyond last point
        uh = u1 + (lnD[k + 1] - LNDTH) / min(slope, 60.0)
    return float(uh), float(slope), ",".join(flag), float(u1 - u0)


class UhiModel:
    def __init__(self, col, cross="loglin", smallA="quad", largeA="flat", xinterp="quad",
                 Ainterp="pchip", override=None):
        self.col, self.smallA, self.largeA, self.xinterp = col, smallA, largeA, xinterp
        nodes = node_pts(col, override)
        self.uh, self.k, self.flags, self.gap, self.f = {}, {}, {}, {}, {}
        for lab in LABS:
            out = [crossing(nodes[(lab, float(A))][:, 0], nodes[(lab, float(A))][:, 1], cross)
                   for A in A_NODES]
            self.uh[lab] = np.array([o[0] for o in out])
            self.k[lab] = np.array([o[1] for o in out])
            self.flags[lab] = [o[2] for o in out]
            self.gap[lab] = [o[3] for o in out]
            y = np.log(self.uh[lab])
            if Ainterp == "pchip":
                self.f[lab] = PchipInterpolator(A_NODES, y)
            else:
                self.f[lab] = (lambda A, y=y: np.interp(A, A_NODES, y))

    def lab_curve(self, lab, A):
        A = np.asarray(A, float)
        y = np.log(self.uh[lab]); a0, a1 = A_NODES[0], A_NODES[-1]
        out = np.empty_like(A)
        mid = (A >= a0) & (A <= a1)
        out[mid] = self.f[lab](A[mid])
        lo = A < a0
        if self.smallA in ("quad", "linD"):           # D ~ A^p at fixed u below A=0.25
            p = 2.0 if self.smallA == "quad" else 1.0
            uu = self.uh[lab][0] - (p / self.k[lab][0]) * np.log(a0 / np.maximum(A[lo], 1e-300))
            out[lo] = np.log(np.maximum(uu, 1e-3))
        elif self.smallA == "flat":
            out[lo] = y[0]
        elif self.smallA == "linA":                     # producer: linear extrapolation in A
            sl = (y[1] - y[0]) / (A_NODES[1] - a0)
            out[lo] = y[0] + sl * (A[lo] - a0)
        else:                                           # "zero": no detection below A=0.25
            out[lo] = math.log(1e-3)
        hi = A > a1
        if self.largeA == "flat":
            out[hi] = y[-1]
        else:
            sl = (y[-1] - y[-2]) / (a1 - A_NODES[-2])
            out[hi] = y[-1] + sl * (np.minimum(A[hi], 3.0) - a1)
        return out

    def weights(self, xi, theta):
        if self.xinterp == "lin":
            t = xi
            return np.where(t < 0, -t, 0.0), np.where(t < 0, 1 + t, 1 - t), np.where(t > 0, t, 0.0)
        t = xi if self.xinterp == "quad" else theta / (math.pi / 6)
        return t * (t - 1) / 2, (1 - t) * (1 + t), t * (t + 1) / 2

    def ln_uhi_grid(self, A, xi, theta):
        f = {lab: self.lab_curve(lab, A) for lab in LABS}
        wm, w0, wp = self.weights(xi, theta)
        return f["obl"][:, None] * wm[None, :] + f["tri"][:, None] * w0[None, :] + f["pro"][:, None] * wp[None, :]

    def ln_uhi_pts(self, A, xi, theta):
        f = {lab: self.lab_curve(lab, A) for lab in LABS}
        wm, w0, wp = self.weights(xi, theta)
        return f["obl"] * wm + f["tri"] * w0 + f["pro"] * wp


# ---------------------------------------------------------------- quadrature machinery
NA, AMAX, NT = 4000, 8.0, 360
Ag = (np.arange(NA) + 0.5) * (AMAX / NA)
TH = -math.pi / 6 + (np.arange(NT) + 0.5) * (math.pi / 3 / NT)
XI = np.sin(3 * TH)
MM = mmin(TH)
LNUREL = Ag[:, None] * SIG_E * MM[None, :]
S = np.round(np.arange(0.980, 1.40001, 0.0005), 6)
X = np.log(S)
LAST = {}


def curves_from_t(t1, t2, w, x):
    t1 = np.ravel(t1); t2 = np.ravel(t2); w = np.ravel(w)
    o = np.argsort(t1); c = np.concatenate([[0.0], np.cumsum(w[o])])
    Pnc = c[np.searchsorted(t1[o], x, side="right")]
    ok = t2 >= t1
    a1, a2, ww = t1[ok], t2[ok], w[ok]
    LAST["E_lnwidth"] = float(np.sum(ww * (a2 - a1)))    # = E[ln(u_hi/0.985)_+] = int P_joint dln s
    o1 = np.argsort(a1); c1 = np.concatenate([[0.0], np.cumsum(ww[o1])])
    o2 = np.argsort(a2); c2 = np.concatenate([[0.0], np.cumsum(ww[o2])])
    Pj = c1[np.searchsorted(a1[o1], x, side="right")] - c2[np.searchsorted(a2[o2], x, side="left")]
    return np.clip(Pj, 0, None), Pnc


def grid_curves(mdl, sigR=1.0):
    lnuh = mdl.ln_uhi_grid(Ag, XI, TH)
    wA = Ag / sigR ** 2 * np.exp(-Ag ** 2 / (2 * sigR ** 2)) * (AMAX / NA)
    w = np.broadcast_to(wA[:, None] / NT, LNUREL.shape)
    return curves_from_t(LNUCUT - LNUREL, lnuh - LNUREL, w, X)


def intervals(s, P, p):
    above = P >= p; out = []; i = 0; n = len(s)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            lo = s[0] if i == 0 else s[i - 1] + (p - P[i - 1]) * (s[i] - s[i - 1]) / (P[i] - P[i - 1])
            hi = s[-1] if j == n - 1 else s[j] + (P[j] - p) * (s[j + 1] - s[j]) / (P[j] - P[j + 1])
            out.append([round(float(lo), 4), round(float(hi), 4)])
            i = j + 1
        else:
            i += 1
    return out


def summarize(s, Pj, Pnc):
    Pc = np.where(Pnc > 1e-4, Pj / np.maximum(Pnc, 1e-300), np.nan)
    Pc0 = np.nan_to_num(Pc, nan=0.0)
    i = int(np.argmax(Pj)); ic = int(np.nanargmax(Pc))
    return dict(Pj_max=float(Pj[i]), s_at_Pj_max=float(s[i]),
                Pc_max=float(Pc[ic]), s_at_Pc_max=float(s[ic]),
                Pj_win={str(p): intervals(s, Pj, p) for p in (0.5, 0.3, 0.2, 0.1, 0.05)},
                Pc_win={str(p): intervals(s, Pc0, p) for p in (0.5, 0.3, 0.2, 0.1)},
                Pj_at={f"{x:.2f}": float(np.interp(x, s, Pj)) for x in SPTS},
                Pc_at={f"{x:.2f}": float(np.interp(x, s, Pc0)) for x in SPTS},
                Pnc_at={f"{x:.2f}": float(np.interp(x, s, Pnc)) for x in SPTS},
                int_Pj_ds=float(np.sum(0.5 * (Pj[1:] + Pj[:-1]) * np.diff(s))),
                int_Pj_dlns=float(np.sum(0.5 * (Pj[1:] + Pj[:-1]) * np.diff(np.log(s)))))


def brief(sm):
    return dict(Pj_max=round(sm["Pj_max"], 4), s_at=round(sm["s_at_Pj_max"], 4),
                Pj_w03=sm["Pj_win"]["0.3"], Pj_w02=sm["Pj_win"]["0.2"], Pj_w01=sm["Pj_win"]["0.1"],
                Pc_w05=sm["Pc_win"]["0.5"], Pc_w03=sm["Pc_win"]["0.3"],
                Pj_104=round(sm["Pj_at"]["1.04"], 3), Pj_106=round(sm["Pj_at"]["1.06"], 3),
                Pc_100=round(sm["Pc_at"]["1.00"], 3), Pc_102=round(sm["Pc_at"]["1.02"], 3),
                Pc_104=round(sm["Pc_at"]["1.04"], 3), Pc_110=round(sm["Pc_at"]["1.10"], 3),
                intPj_dlns=round(sm["int_Pj_dlns"], 5))


# ---------------------------------------------------------------- MC sample (full plane)
rng = np.random.default_rng(20261014)
NMC = 2_000_000
b = rng.standard_normal((NMC, 2))
Amc = np.hypot(b[:, 0], b[:, 1])
dg = SIG_E * (b[:, [0]] * np.array([1.0, -1.0, 0.0]) / SQ2 + b[:, [1]] * np.array([1.0, 1.0, -2.0]) / SQ6)
lnurel_mc = dg.min(axis=1)
Xn = dg / (SIG_E * Amc[:, None])
xi_mc = np.clip(SQ6 * np.sum(Xn ** 3, axis=1), -1, 1)
th_mc = np.arcsin(xi_mc) / 3
sub = np.sort(xi_mc[:200000])
ecdf = (np.arange(sub.size) + 0.5) / sub.size
RES["measure_checks"] = dict(
    KS_xi_vs_arcsine=float(np.max(np.abs(ecdf - (0.5 + np.arcsin(sub) / math.pi)))),
    KS_95pct_crit=float(1.36 / math.sqrt(sub.size)),
    max_abs_minDiag_vs_formula=float(np.max(np.abs(lnurel_mc - Amc * SIG_E * mmin(th_mc)))),
    mean_A=float(Amc.mean()), mean_A2=float((Amc ** 2).mean()),
    rayleigh_mean_expected=math.sqrt(math.pi / 2))


def mc_curves(mdl):
    lnuh = mdl.ln_uhi_pts(Amc, xi_mc, th_mc)
    return curves_from_t(LNUCUT - lnurel_mc, lnuh - lnurel_mc, np.full(NMC, 1.0 / NMC), X)


# ---------------------------------------------------------------- Method B
NA_B, NT_B = 1600, 240
Ab = (np.arange(NA_B) + 0.5) * (AMAX / NA_B)
wAb = Ab * np.exp(-Ab ** 2 / 2) * (AMAX / NA_B)
THb = -math.pi / 6 + (np.arange(NT_B) + 0.5) * (math.pi / 3 / NT_B)
XIb = np.sin(3 * THb)
LNURELb = Ab[:, None] * SIG_E * mmin(THb)[None, :]
Wb = wAb[:, None] / NT_B * np.ones((1, NT_B))
WMb, W0b, WPb = XIb * (XIb - 1) / 2, (1 - XIb) * (1 + XIb), XIb * (XIb + 1) / 2
S_B = np.round(np.arange(0.985, 1.30001, 0.001), 6)


def lin_ie(x, xp, fp):
    if x <= xp[0]:
        return fp[0] + (fp[1] - fp[0]) * (x - xp[0]) / (xp[1] - xp[0])
    if x >= xp[-1]:
        return fp[-1] + (fp[-1] - fp[-2]) * (x - xp[-1]) / (xp[-1] - xp[-2])
    return float(np.interp(x, xp, fp))


def methodB(col, override=None):
    nodes = node_pts(col, override)
    ns = {k: (v[:, 2], np.log(v[:, 1])) for k, v in nodes.items()}
    lo = Ab < A_NODES[0]; hi = Ab > A_NODES[-1]; mid = ~lo & ~hi
    Pj, Pnc = [], []
    for s in S_B:
        f = {}
        for lab in LABS:
            vals = np.array([lin_ie(s, *ns[(lab, float(A))]) for A in A_NODES])
            gA = np.empty(NA_B)
            gA[mid] = PchipInterpolator(A_NODES, vals)(Ab[mid])
            gA[lo] = vals[0] + 2 * np.log(Ab[lo] / A_NODES[0])
            sl = (vals[-1] - vals[-2]) / (A_NODES[-1] - A_NODES[-2])
            gA[hi] = vals[-1] + max(sl, 0.0) * (Ab[hi] - A_NODES[-1])
            f[lab] = gA
        lnD = f["obl"][:, None] * WMb + f["tri"][:, None] * W0b + f["pro"][:, None] * WPb
        nc = (math.log(s) + LNURELb) >= LNUCUT
        Pj.append(float((Wb * (nc & (lnD >= LNDTH))).sum())); Pnc.append(float((Wb * nc).sum()))
    return np.array(Pj), np.array(Pnc)


# ---------------------------------------------------------------- Taylor-ansatz diagnostic
def taylor_fit(col, s0):
    pts = [r for r in rows if abs(r["s"] - s0) < 1e-6 and np.isfinite(r[col])]
    A = np.array([r["A"] for r in pts]); xi = np.array([r["xi"] for r in pts])
    D = np.array([r[col] for r in pts])
    M = np.stack([A ** 2, A ** 3 * xi, A ** 4], axis=1)
    c = np.linalg.lstsq(M, D, rcond=None)[0]
    Dg = c[0] * Ag[:, None] ** 2 + c[1] * Ag[:, None] ** 3 * XI[None, :] + c[2] * Ag[:, None] ** 4
    nc = (math.log(s0) + LNUREL) >= LNUCUT
    W = (Ag * np.exp(-Ag ** 2 / 2) * (AMAX / NA))[:, None] / NT
    rel = (M @ c - D) / D
    return dict(s=s0, n=len(pts), coef=c.tolist(), rel_resid_rms=float(np.sqrt(np.mean(rel ** 2))),
                n_negative_fit=int(np.sum(M @ c < 0)),
                pts=[dict(A=r["A"], lab=r["lab"], u=r["u"], D=r[col], Dfit=float(df))
                     for r, df in zip(pts, M @ c)],
                P_joint_fit=float((W * (nc & (Dg >= DTH))).sum()))


# ---------------------------------------------------------------- main runs
MAIN, CURV = {}, {"s": S}
for kind in KINDS:
    for d in DATASETS:
        col = f"{d}_{kind}"
        mdl = UhiModel(col)
        Pj, Pnc = grid_curves(mdl)
        sm = summarize(S, Pj, Pnc)
        sm["E_ln_uhi_over_ucut"] = LAST["E_lnwidth"]
        sm["node_uhi"] = {lab: np.round(mdl.uh[lab], 5) for lab in LABS}
        sm["node_slope_k"] = {lab: np.round(mdl.k[lab], 2) for lab in LABS}
        sm["node_gap_u"] = {lab: np.round(mdl.gap[lab], 4) for lab in LABS}
        sm["node_flags"] = mdl.flags
        for cm in ("convex_lo", "convex_hi"):
            mm_ = UhiModel(col, cross=cm)
            sm[f"node_uhi_{cm}"] = {lab: np.round(mm_.uh[lab], 5) for lab in LABS}
        rep = UhiModel(col).uh["tri"][3]
        sm["rep_window"] = [UCUT / math.exp(-1.41421 * SIG_E / SQ2), rep / math.exp(-1.41421 * SIG_E / SQ2)]
        sm["rep_lnwidth"] = math.log(sm["rep_window"][1] / sm["rep_window"][0])
        Pj_mc, Pnc_mc = mc_curves(mdl)
        sm["MC"] = brief(summarize(S, Pj_mc, Pnc_mc))
        sm["MC_maxabs_dPj"] = float(np.max(np.abs(Pj_mc - Pj)))
        PjB, PncB = methodB(col)
        sm["B"] = brief(summarize(S_B, PjB, PncB))
        sm["B_maxabs_dPj_vs_A"] = float(np.max(np.abs(PjB - np.interp(S_B, S, Pj))))
        MAIN[col] = sm
        CURV[col + "_Pj"] = Pj; CURV[col + "_Pnc"] = Pnc
        CURV[col + "_Pc"] = np.where(Pnc > 1e-4, Pj / np.maximum(Pnc, 1e-300), np.nan)
RES["main"] = MAIN

# variants
VAR = {}
vspec = {"cross_pchip": dict(cross="pchip"), "convex_lo": dict(cross="convex_lo"),
         "convex_hi": dict(cross="convex_hi"), "cross_linD(extreme)": dict(cross="linD"),
         "A_linear": dict(Ainterp="lin"), "producer_like(Alin+smallA_linA)": dict(Ainterp="lin", smallA="linA"),
         "xi_linear": dict(xinterp="lin"), "xi_quad_in_theta": dict(xinterp="quadtheta"),
         "smallA_zero": dict(smallA="zero"), "smallA_flat": dict(smallA="flat"),
         "smallA_D~A": dict(smallA="linD"), "smallA_linA": dict(smallA="linA"),
         "largeA_lin_to3": dict(largeA="lin")}
ANOM = ("pro", 0.5, 1.045)
aspec = {"anom_drop": {"TE40_prof": {ANOM: None}, "TE80_prof": {ANOM: None}},
         "anom_DPI_lower(=TE30_prof)": {"TE40_prof": {ANOM: 2.9302}, "TE80_prof": {ANOM: 2.9302}},
         "anom_fixed_upper": {"TE40_prof": {ANOM: 4.30232}, "TE80_prof": {ANOM: 6.95202}}}
vcols = ["TE30_prof", "TE80_prof", "TE30_fixed", "TE80_fixed"] + ([] if POOLMIN else ["TE40_prof"])
for col in vcols:
    VAR[col] = {}
    for nm, kw in vspec.items():
        VAR[col][nm] = brief(summarize(S, *grid_curves(UhiModel(col, **kw))))
    for sr in (0.9, 1.1):
        VAR[col][f"rayleigh_scale_{sr}"] = brief(summarize(S, *grid_curves(UhiModel(col), sigR=sr)))
    if col in ("TE40_prof", "TE80_prof") and not POOLMIN:
        for nm, ov in aspec.items():
            mdl = UhiModel(col, override=ov)
            VAR[col][nm] = brief(summarize(S, *grid_curves(mdl)))
            VAR[col][nm]["uhi_pro_A0.5"] = round(float(mdl.uh["pro"][1]), 5)
RES["variants"] = VAR

# ratio analysis (2.17x KL vs probability)
rat = {}
for kind in KINDS:
    a, c = MAIN[f"TE30_{kind}"], MAIN[f"TE80_{kind}"]
    rat[kind] = dict(Pjmax_TE80_over_TE30=c["Pj_max"] / a["Pj_max"],
                     intPj_dlns_TE80_over_TE30=c["int_Pj_dlns"] / a["int_Pj_dlns"],
                     Pc_at_1p10_ratio=c["Pc_at"]["1.10"] / max(a["Pc_at"]["1.10"], 1e-12),
                     rep_lnwidth_ratio=c["rep_lnwidth"] / a["rep_lnwidth"],
                     Pj_at_1p13_ratio=c["Pj_at"]["1.13"] / max(a["Pj_at"]["1.13"], 1e-12))
rep = []
for r in rows:
    if r["lab"] == "tri" and abs(r["A"] - 1.41421) < 1e-4:
        rep.append(dict(s=r["s"], u=r["u"], prof_TE80_TE30=r["TE80_prof"] / r["TE30_prof"],
                        fixed_TE80_TE30=(r["TE80_fixed"] / r["TE30_fixed"]) if np.isfinite(r["TE30_fixed"]) else None))
rat["rep_point_tri_Asqrt2_KL_ratios"] = rep
RES["ratios"] = rat
RES["taylor_ansatz"] = [taylor_fit(c_, s0) for c_ in ("TE30_prof", "TE80_prof") for s0 in (1.03, 1.14, 1.15)]
RES["runtime_s"] = time.time() - T0

with open(OUT + rf"\r15_ensemble_results{SUFFIX}.json", "w", encoding="utf-8") as f:
    json.dump(jd(RES), f, indent=1)
keys = [k for k in CURV if k != "s" and (k.endswith("_Pj") or k.endswith("_Pc"))]
with open(OUT + rf"\r15_curves_methodA{SUFFIX}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["s"] + keys)
    for i in range(0, len(S), 2):
        w.writerow([f"{S[i]:.4f}"] + [("" if not np.isfinite(CURV[k][i]) else f"{CURV[k][i]:.5f}") for k in keys])

# ---------------------------------------------------------------- print compact summary
print("MODE:", RES["mode"])
if POOLMIN:
    pm = RES["poolmin"]
    print("kl entries", pm["n_kl_entries"], "cubes", pm["cube_list"])
    print("KL-table DPI violations:", len(pm["kl_dpi_violations"]), pm["kl_dpi_violations"][:6])
    print("fixed compared", pm["n_fixed_compared"], "mismatch", len(pm["fixed_mismatch"]))
    print("2nd element examples (TE80):", pm["second_element_examples"])
    print("parab undershoot >1% vs pool-min:")
    for c in pm["parab_undershoot_gt1pct"]:
        print("   A=%.3f %s s=%.3f u=%.4f %s parab=%.4g pool=%.4g s*=%.4f rel=%.3f" % (
            c["A"], c["lab"], c["s"], c["u"], c["ds"], c["parab"], c["poolmin"], c["s_argmin"], c["rel_parab_minus_pool"]))
    print("parab above pool:", [(c["row"][:28], c["ds"], round(c["rel_parab_minus_pool"], 5)) for c in pm["parab_above_pool"]])
    print("argmin at pool edge:", [(c["A"], c["lab"], c["s"], c["ds"], c["argmin_edge"], c["s_argmin"]) for c in pm["argmin_at_pool_edge"]])
print("DPI violations (table used):", len(viol), viol[:4])
for col, sm in MAIN.items():
    print(f"\n== {col}  A:", brief(sm), " Pc_max=%.3f@%.4f" % (sm["Pc_max"], sm["s_at_Pc_max"]))
    print("   E[ln(uhi/.985)]=%.5f  rep_window=%s lnw=%.4f" % (sm["E_ln_uhi_over_ucut"], np.round(sm["rep_window"], 4), sm["rep_lnwidth"]))
    print("   MC:", {k: sm["MC"][k] for k in ("Pj_max", "s_at", "Pj_w03", "Pj_w01", "Pc_w05")}, " |dPj|max=%.4f" % sm["MC_maxabs_dPj"])
    print("   B :", {k: sm["B"][k] for k in ("Pj_max", "s_at", "Pj_w03", "Pj_w01", "Pc_w05")}, " |dPj|max vs A=%.4f" % sm["B_maxabs_dPj_vs_A"])
    print("   uhi:", jd(sm["node_uhi"]), "\n   lo :", jd(sm["node_uhi_convex_lo"]), "\n   hi :", jd(sm["node_uhi_convex_hi"]))
    print("   Pj_at:", {k: round(v, 3) for k, v in sm["Pj_at"].items()})
    print("   Pc_at:", {k: round(v, 3) for k, v in sm["Pc_at"].items()})
print("   Pnc_at:", {k: round(v, 3) for k, v in MAIN["TE80_prof"]["Pnc_at"].items()})
print("\n== variants")
for col, vv in VAR.items():
    for nm, b_ in vv.items():
        print(f"  {col:10s} {nm:32s}", {k: b_[k] for k in ("Pj_max", "s_at", "Pj_w03", "Pj_w01", "Pc_w05", "Pj_104", "Pc_100", "Pc_102", "Pc_104", "intPj_dlns")})
print("\n== ratios:", json.dumps(jd({k: v for k, v in rat.items() if k != "rep_point_tri_Asqrt2_KL_ratios"})))
for t in RES["taylor_ansatz"]:
    print("== taylor %s-ish s=%.3f n=%d rms_rel=%.2f n_neg=%d P_joint_fit=%.4f" % (
        "", t["s"], t["n"], t["rel_resid_rms"], t["n_negative_fit"], t["P_joint_fit"]))
print("runtime %.1f s" % (time.time() - T0))
