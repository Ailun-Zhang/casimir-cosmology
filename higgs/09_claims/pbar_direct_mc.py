# PROVENANCE: 04_calculations/code/20261015_claude_v12_writing/claims_run05/pbar_direct_mc.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
# PROVENANCE: new script by agent claims-run05 (C161/D132, v12 writing round) on 2026-10-15.  It does NOT import or copy
#   any C154 code; it reads only data: run05 C149 summary + shards, and the measurement/error definitions in
#   C154_inputs_v2_run05.json (identical to the run02 inputs except the central-P1 file).
r"""[v12 package, 2026-10-15] Independent direct Monte Carlo of P1bar (no C154 code reused): samples m_t, alpha_s, M_h, SM and model errors with the same conventions; agreement <= 0.003 required (C161 step 2).  Paper: PRD H4 (cross-check).

Step 2: independent direct Monte Carlo of Pbar1 on the run05 centre (check of step 1 to <= 0.003).

Surface (own construction, deliberately different from C154):
  P1(m, a) = exp(-exp(y)),  m = m_ref(y) + Delta(y, a)  solved for y by fixed-point iteration;
  m_ref(y): PCHIP of y_i = ln(-ln P1_i) on the run05 10-point grid (alpha_s = 0.1180), linear end extrapolation with the
            PCHIP end slopes (C154: cubic spline with 4-point end fits);
  Delta(y, a): quadratic in y through the P1 = 0.9/0.5/0.1 offsets crit(a) - crit(0.1180), CLAMPED beyond the end levels
            (C154: quadratic in m - m_c with LINEAR extrapolation), quadratic Lagrange in a through 0.1171/0.1180/0.1189,
            extrapolated (C154: clamped shape weights + quadratic m_c(a)).
  Variant 'uniform': Delta(y, a) = Delta(y_0.5, a) (pure alpha_s shift) -> interpolation sensitivity.
Sampling (D126 v2 + correction addendum conventions, equal-mass split for every asymmetric input):
  m ~ measurement; alpha_s ~ N(0.1180, 0.0009); M_h: N(0, 0.11) x (dlam/dM_h)/|dlam/dm_t|; SM ~ N(0, 0.102 pole | 0.070
  m-bar) [BKPV/FJ variant: +0.17|z| or -0.36|z| with probability 1/2 each]; model: Z ~ N(0,1), d = Z s_+(x) (Z > 0) or
  Z s_-(x) (Z < 0), s_+- = Table-1 composite (quadrature per side) interpolated linearly in x = m_eff - m_c,0.5(a) between
  the level offsets (clamped); P = P1(m_eff - d, a).  No w0 factor (central w0 = 1), as in the adopted P-bar.
Also: direct P1 at alpha_s = 0.1171 / 0.1189 from the run05 X_als node hazards (5 masses each) vs this surface.
Run: python -B pbar_direct_mc.py  [N per measurement, default 4e6]"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import json
import math
import glob
import csv
import time
import hashlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator, CubicSpline  # noqa: E402
from scipy.integrate import simpson  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _zp.WORK
R5 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run05"))
NEW = os.path.join(ROOT, _zp.w("w/claims_run05/run01"))
CFGP = os.path.join(HERE, "C154_inputs_v2_run05.json")
LEVELS = (0.9, 0.5, 0.1)
YL = np.array([math.log(-math.log(p)) for p in LEVELS])
AL = (0.1171, 0.1180, 0.1189)
HLB = 0.00437243
UN11 = np.round(np.arange(0.0, 5.0001, 0.5), 6)


def jl(p):
    return json.load(open(p, encoding="utf-8"))


class Surf:
    def __init__(self, d5, uniform=False):
        g = d5["P1_grid"]
        m = np.array(sorted(float(k) for k in g))
        y = np.array([math.log(-math.log(g[k])) for k in sorted(g, key=float)])
        self.pc = PchipInterpolator(m, y)
        self.m0, self.m1 = m[0], m[-1]
        self.y0, self.y1 = float(y[0]), float(y[-1])
        self.s0, self.s1 = float(self.pc.derivative()(m[0])), float(self.pc.derivative()(m[-1]))
        self.mf = np.linspace(m[0], m[-1], 40001)
        self.yf = self.pc(self.mf)
        cr = {a: np.array([d5["crit"]["%.4f" % a]["P1=%s" % p] for p in LEVELS]) for a in AL}
        self.cref = cr[0.1180]
        self.q = {}
        for a in (0.1171, 0.1189):
            dk = cr[a] - cr[0.1180]
            self.q[a] = np.full(3, dk[1]) if uniform else dk
        self.coef = {a: np.polyfit(YL, self.q[a], 2) for a in self.q}

    def yref(self, m):
        y = np.interp(m, self.mf, self.yf)
        y = np.where(m < self.m0, self.y0 + self.s0 * (m - self.m0), y)
        return np.where(m > self.m1, self.y1 + self.s1 * (m - self.m1), y)

    def delta(self, y, a):
        yc = np.clip(y, YL[0], YL[-1])
        dm, dp = np.polyval(self.coef[0.1171], yc), np.polyval(self.coef[0.1189], yc)
        a0, am, ap = 0.1180, 0.1171, 0.1189
        return dm * (a - a0) * (a - ap) / ((am - a0) * (am - ap)) + dp * (a - am) * (a - a0) / ((ap - am) * (ap - a0))

    def y_of(self, m, a, it=8):
        y = self.yref(m)
        for _ in range(it):
            yn = self.yref(m - self.delta(y, a))
            err = float(np.max(np.abs(yn - y)))
            y = yn
        return y, err

    def P1(self, m, a):
        y, _ = self.y_of(np.asarray(m, float), np.asarray(a, float))
        return np.exp(-np.exp(np.minimum(y, 50.0)))

    def mc05(self, a):
        y05 = YL[1]
        # m at y_0.5: m_ref(y_0.5) + Delta(y_0.5, a); m_ref(y_0.5) by inverting the fine table
        mref05 = float(np.interp(y05, self.yf, self.mf))
        return mref05 + self.delta(np.full_like(np.asarray(a, float), y05), a)


def hazard_I(gam):
    uf = np.linspace(0.0, 5.0, 20001)
    return float(simpson(np.exp(CubicSpline(UN11, np.log(np.maximum(gam, 1e-300)))(uf)) / (HLB * np.sqrt(np.cosh(2 * uf))), x=uf))


def direct_points():
    R = {}
    for f in glob.glob(os.path.join(R5, "shards", "X_*.json")):
        for x in jl(f)["cols"]:
            R.setdefault((round(x["als"], 4), round(x["mt"], 2)), {})[round(x["u"], 4)] = x["gamma"]
    return {k: math.exp(-hazard_I(np.array([g[round(float(u), 4)] for u in UN11]))) for k, g in R.items() if len(g) == 11}


def meas(ms, cfg):
    if ms.get("msbar_map"):
        P = np.array(cfg["msbar_points"]["pole"]); M = np.array(cfg["msbar_points"]["mbar"])
        b, a0 = np.polyfit(M, P, 1)
        return a0 + b * ms["m_msbar"], b * ms["s_minus"], b * ms["s_plus"]
    s = math.sqrt(ms["s"] ** 2 + sum(e * e for e in ms.get("extra", [])))
    return ms["m"] + ms.get("offset", 0.0), s, s


def two_piece(rng, n, sl, sr):
    z = np.abs(rng.standard_normal(n))
    return np.where(rng.random(n) < 0.5, -sl * z, sr * z)


def sample(S, cfg, sig, c, sl, sr, s_sm, n, seed, bkpv=False, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    kmh = cfg["Mh"]["dlam_dMh"] / abs(cfg["Mh"]["dlam_dMt"])
    xl = S.cref - S.cref[1]
    out = []
    for st in range(0, n, chunk):
        k = min(chunk, n - st)
        m = c + two_piece(rng, k, sl, sr)
        a = 0.1180 + 0.0009 * rng.standard_normal(k)
        dmh = kmh * 0.11 * rng.standard_normal(k)
        dsm = two_piece(rng, k, 0.36, 0.17) if bkpv else s_sm * rng.standard_normal(k)
        me = m - dmh - dsm
        x = me - S.mc05(a)
        Z = rng.standard_normal(k)
        d = np.where(Z > 0, Z * np.interp(x, xl, sig[0]), Z * np.interp(x, xl, sig[1]))
        out.append(S.P1(me - d, a))
    return np.concatenate(out)


def main():
    t0 = time.time()
    n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 4_000_000
    cfg = jl(CFGP)
    d5 = jl(os.path.join(R5, "C149_run05_summary.json"))
    S, Su = Surf(d5), Surf(d5, uniform=True)
    sp = np.zeros(3); sm = np.zeros(3)
    for comp in cfg["model_components"]:
        for i, L in enumerate(("0.9", "0.5", "0.1")):
            p, q = comp["levels"][L]
            sp[i] += p * p; sm[i] += q * q
    sig = (np.sqrt(sp), np.sqrt(sm))
    res = dict(task="C161 step 2: independent direct MC of Pbar1 (run05 centre)", agent="claims-run05", n_per_measurement=n,
               model_sigma_plus=sig[0].tolist(), model_sigma_minus=sig[1].tolist())
    # surface checks
    chk = {}
    for a in AL:
        for i, p in enumerate(LEVELS):
            y, _ = S.y_of(np.array([d5["crit"]["%.4f" % a]["P1=%s" % p]]), np.array([a]))
            chk["%.4f_P1=%s" % (a, p)] = float(np.exp(-np.exp(y[0])) - p)
    dp = direct_points()
    pts = []
    for (a, m), p in sorted(dp.items()):
        pts.append(dict(alpha_s=a, mt=m, P1_direct=p, P1_surface=float(S.P1(np.array([m]), np.array([a]))[0]),
                        P1_surface_uniform=float(Su.P1(np.array([m]), np.array([a]))[0])))
    res["surface_check"] = dict(P1_at_official_crit_minus_level=chk, direct_points=pts,
                                max_abs_vs_direct_other_alpha=max(abs(x["P1_surface"] - x["P1_direct"]) for x in pts if abs(x["alpha_s"] - 0.118) > 1e-9),
                                max_abs_vs_direct_0p1180=max(abs(x["P1_surface"] - x["P1_direct"]) for x in pts if abs(x["alpha_s"] - 0.118) < 1e-9),
                                fixed_point_residual=float(S.y_of(np.linspace(170, 176, 1001), np.full(1001, 0.1207))[1]),
                                P1_172p60=float(S.P1(np.array([172.60]), np.array([0.1180]))[0]),
                                P1_172p52=float(S.P1(np.array([172.52]), np.array([0.1180]))[0]))
    print(json.dumps({k: v for k, v in res["surface_check"].items() if k != "direct_points"}, indent=1))
    st1 = jl(os.path.join(NEW, "C154_correction_summary.json"))["marginal"]["equal_mass"]
    rows = []
    jobs = [(ms["id"], ms, False) for ms in cfg["measurements"]] + [("D1_direct_pole_BKPV_FJ", cfg["measurements"][0], True)]
    for k, (mid, ms, bk) in enumerate(jobs):
        c, sl, sr = meas(ms, cfg)
        s_sm = cfg["sm_mapping"]["pole"] if ms["scheme"] == "pole" else cfg["sm_mapping"]["msbar"]
        P = sample(S, cfg, sig, c, sl, sr, s_sm, n, seed=161000 + k, bkpv=bk)
        Pu = sample(Su, cfg, sig, c, sl, sr, s_sm, n // 4, seed=162000 + k, bkpv=bk)
        q = np.quantile(P, [0.025, 0.16, 0.5, 0.84, 0.975])
        ref = st1[ms["id"]]
        r1 = ref["Pbar_BKPV_FJ"] if bk else ref["Pbar"]
        e = dict(id=mid, center=c, s_left=sl, s_right=sr, s_SM=("BKPV +0.17/-0.36" if bk else s_sm), Pbar_direct=float(P.mean()),
                 se=float(P.std() / math.sqrt(len(P))), Pbar_uniform_shift_surface=float(Pu.mean()), prob_P_ge_0p5=float((P >= 0.5).mean()),
                 quantiles_025_16_50_84_975=q.tolist(), Pbar_step1=r1, diff_vs_step1=float(P.mean()) - r1,
                 prob_ge_0p5_step1=None if bk else ref["prob_P_ge_0p5"],
                 quantiles_step1=None if bk else ref["quantiles_025_16_50_84_975"])
        rows.append(e)
        print("%-26s direct=%.4f(%.4f) step1=%.4f diff=%+.4f uniform=%.4f P>=.5 %.4f/%s" % (
            mid, e["Pbar_direct"], e["se"], r1, e["diff_vs_step1"], e["Pbar_uniform_shift_surface"], e["prob_P_ge_0p5"],
            "%.4f" % e["prob_ge_0p5_step1"] if e["prob_ge_0p5_step1"] is not None else "-"), flush=True)
    res["results"] = rows
    res["max_abs_diff_vs_step1"] = max(abs(x["diff_vs_step1"]) for x in rows)
    res["pass_le_0p003"] = res["max_abs_diff_vs_step1"] <= 0.003
    res["sha256"] = {os.path.basename(__file__): hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                     "C149_run05_summary.json": hashlib.sha256(open(os.path.join(R5, "C149_run05_summary.json"), "rb").read()).hexdigest(),
                     "C154_inputs_v2_run05.json": hashlib.sha256(open(CFGP, "rb").read()).hexdigest()}
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(NEW, "pbar_direct_mc.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(NEW, "pbar_direct_mc.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "Pbar_direct_MC", "se", "Pbar_step1_C154", "diff", "Pbar_uniform_shift_surface", "P_ge_0p5_direct", "P_ge_0p5_step1",
                    "q16_direct", "q84_direct", "q025_direct", "q975_direct"])
        for e in rows:
            q = e["quantiles_025_16_50_84_975"]
            w.writerow([e["id"]] + ["%.5f" % v for v in (e["Pbar_direct"], e["se"], e["Pbar_step1"], e["diff_vs_step1"], e["Pbar_uniform_shift_surface"],
                                                         e["prob_P_ge_0p5"])] + ["%.5f" % e["prob_ge_0p5_step1"] if e["prob_ge_0p5_step1"] is not None else ""]
                       + ["%.4f" % v for v in (q[1], q[3], q[0], q[4])])
    print("max |diff| = %.4f  pass=%s  %.0fs" % (res["max_abs_diff_vs_step1"], res["pass_le_0p003"], res["runtime_s"]))


if __name__ == "__main__":
    main()
