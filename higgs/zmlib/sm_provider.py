# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/sm_provider.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] SM mapping (D120/T1): SMDR-1.3-type NNLO matching at mu0 = M_t (scheme 'SMDR25') + SMDR-port RGE (full 3-loop, 4-loop gauge, leading 4/5-loop QCD); couplings(M_t, alpha_s, M_h, mu) -> lam, y_t, g2, gY, g3 (MS-bar).  Includes the run02 compiled-SMDR refit variants 'sfit_*'.  The copies used by the Higgs runs (holonomy_cross etc.) are identical except for that additional sfit block.  Paper: PRD App. H5; PRL lam(mu_b) = -0.0098.  Units: GeV.

C148 / D120 (sm-mapping): MSbar coupling provider for the DNEC Higgs zero-mode criterion.

couplings(Mt, alphas, Mh, mu_array, variant="central", MW=80.3692) -> dict of arrays lam, gY(=g'), g2, g3, yt, yb, ytau
(normalisation as R14_rge: gY = sqrt(3/5) g1^GUT, V = lam (|H|^2 - v^2/2)^2).

Matching at mu0 = M_t (NNLO; "scheme" = which NNLO determination):
  SMDR25 : SMDR 1.3 ReferenceModel.dat (Q = Mt = 172.4, Mh 125.20, as 0.1180; inputs GF, alpha, MZ; tadpole-free
           MSbar; full 2-loop + leading 3-loop + 4-loop QCD), moved in (Mt, as, Mh) with Buttazzo (55)-(60) slopes
  SMDR19 : SMDR 1.0 reference model (Q = Mt = 173.1, Mh 125.10, as 0.1181), same slopes
  BKPV15 : Bednyakov-Kniehl-Pikelner-Veretin PRL 115 (2015) 201802 Table I (FJ tadpole scheme, GF, 4-loop QCD)
  B13    : Buttazzo et al. Eqs. (55)-(60) (3-loop QCD for yt)
  B13c   : B13 + exact 4-loop QCD pole-MSbar term (Marquard) on yt + own pure-QCD g3 (qcd.as6_at)
  SMDR25_MW : SMDR25 with g, g' shifted (Buttazzo MW slopes) to reproduce the measured MW (input-set variation)
RGE: SMDR-port (loops 1..5) or Buttazzo App. B ('b13' impl).
Matching-scale variation xi: fixed-order NNLO at mu0 = xi Mt reconstructed from the RG (non-QCD towers through
2 loops: t[b1+b2] + t^2/2 (grad b1 . beta1)), pure-QCD towers exactly at fixed order (yt: z_m through a^N;
g3: decoupling at mu0, N_dec loops), then evolved with the full RGE (see D120 Sec. 2.3).
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import sys
import math
import json
import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm_rge as RG  # noqa: E402
import qcd as Q  # noqa: E402

PI = math.pi
MPBAR = 2.435e18
MU_B = MPBAR / 722.0
MPL = 1.2209e19
MW_PDG = 80.3692
K = RG.K

SMDR25 = dict(Q=172.4, Mt=172.4, Mh=125.20, als=0.1180, g3=1.1635013021305831, g=0.6475934615902679,
              gp=0.3585947982153167, yt=0.9309715465595599, yb=0.0155080080991419, ytau=0.0099949323705356,
              lam=0.1262969232875777, v=246.60029586365829, MW_pred=80.34918252735155)
SMDR19 = dict(Q=173.1, Mt=173.1, Mh=125.10, als=0.1181, g3=1.1636240557157115, g=0.6476596104984968,
              gp=0.3585387714929383, yt=0.9348008219005578, yb=0.0154800969000490, ytau=0.0099944422053384,
              lam=0.1260384218104124, v=246.60109064536805, MW_pred=80.36037496647687)
BKPV = dict(mu_thr=173.21, Mt=173.21, dMt=0.87, Mh=125.7, dMh=0.4, als=0.1185, das=0.0006, MW=80.385, dMW=0.015,
            x0=dict(gp=0.35838, g=0.64812, g3=1.16540, yt=0.93517, yb=0.01706, lam=0.12714),
            coef=dict(gp=(-3.8e-6, -2.3e-4, -2.5e-6, 7.1e-5, 2.1e-3), g=(8.5e-7, 1.2e-4, -6.6e-7, -9.8e-6, -5.2e-3),
                      g3=(2.7e-3, 8.9e-8, 7.8e-8, -4.0e-5, -7.2e-2), yt=(-3.6e-4, -1.3e-7, -8.6e-6, 5.1e-3, -5.2e-2),
                      yb=(-5.7e-5, -5.1e-10, 1.3e-7, -2.4e-7, -9.2e-4), lam=(-6.2e-6, -4.2e-7, 8.2e-4, 6.4e-5, -2.0e-2)),
            dmu=dict(gp=8.5e-5, g=5.8e-5, g3=5.6e-5, yt=8.0e-4, yb=2.5e-4, lam=5.8e-4),
            dtru=dict(gp=6.4e-4, g=1.0e-3, yt=1.2e-3, yb=1.1e-3, lam=5.5e-4))
# Buttazzo slopes at mu = Mt (Eqs. 55-60); yt-Mh and lam-as slopes from BKPV Table I (tiny)
SL = dict(lam_Mh=0.00206, lam_Mt=-0.00004, lam_as=-6.2e-6 / 0.0006, yt_Mt=0.00556, yt_as=-0.00042 / 0.0007,
          yt_Mh=-8.6e-6 / 0.4, g3_as=0.00314 / 0.0007, g3_Mt=-0.00046, g_Mt=0.00004, g_MW=0.00011 / 0.014,
          gp_Mt=0.00011, gp_MW=-0.00020 / 0.014)

VARIANTS = {
    "central": dict(match="SMDR25"),
    "xi0.5": dict(match="SMDR25", xi=0.5), "xi2": dict(match="SMDR25", xi=2.0),
    "xi0.316": dict(match="SMDR25", xi=10 ** -0.5), "xi3.16": dict(match="SMDR25", xi=10 ** 0.5),
    "loops2": dict(match="SMDR25", loops=2), "loops3": dict(match="SMDR25", loops=3),
    "loops4": dict(match="SMDR25", loops=4), "rge_b13": dict(match="SMDR25", impl="b13", loops=4),
    "SMDR19": dict(match="SMDR19"), "BKPV15": dict(match="BKPV15"), "B13": dict(match="B13", qcdN=3, decN=3),
    "B13c": dict(match="B13c", qcdN=4, decN=3), "B13c_l3": dict(match="B13c", qcdN=4, decN=3, lam3=True),
    "MWinput": dict(match="SMDR25_MW"),
    "z4hi": dict(match="SMDR25", z4=Q.Z4_NL5 - Q.Z4_NL5_ERR), "z4lo": dict(match="SMDR25", z4=Q.Z4_NL5 + Q.Z4_NL5_ERR),
    "BKPV15_xi0.5": dict(match="BKPV15", xi=0.5), "BKPV15_xi2": dict(match="BKPV15", xi=2.0),
    "xi0.5b": dict(match="SMDR25", xi=0.5, mode="bwd"), "xi2b": dict(match="SMDR25", xi=2.0, mode="bwd"),
    "xi0.316b": dict(match="SMDR25", xi=10 ** -0.5, mode="bwd"), "xi3.16b": dict(match="SMDR25", xi=10 ** 0.5, mode="bwd"),
    "qcd5": dict(match="SMDR25", qcd5=True), "Mh125.20": dict(match="SMDR25", Mh_override=125.20),
}


FITS2 = os.path.join(RG.ROOT, _zp.w("r3/sm_mapping/run02/fits/fits.json"))
_SF = {}


def _sfit_poly(cfg):
    """run02: compiled SMDR 1.3 refits (c148_fit.exe) at (as 0.1180, Mh 125.13), quadratic in Mt (4 points)."""
    if cfg not in _SF:
        import glob as _g
        d = {}
        for fn in _g.glob(os.path.join(os.path.dirname(FITS2), "fit_%s_Mt*.txt" % cfg)):
            mt = os.path.basename(fn)[len("fit_%s_Mt" % cfg):-4]
            d[mt] = {ln.split()[0]: float(ln.split()[1]) for ln in open(fn, encoding="utf-8")
                     if ln.strip() and not ln.startswith("RG_mub")}
        mts = np.array(sorted(float(k) for k in d if d[k]))
        _SF[cfg] = {k: np.polyfit(mts - 172.6, [d["%.2f" % m][kk] for m in mts], min(2, len(mts) - 1))
                    for k, kk in (("g3", "g3"), ("g", "g"), ("gp", "gp"), ("yt", "yt"), ("yb", "yb"),
                                  ("ytau", "ytau"), ("lam", "lambda"), ("v", "v"))}
    return _SF[cfg]


def match_sfit(Mt, als, Mh, cfg, mode):
    if mode == "direct":
        co = _sfit_poly(cfg)
        x = {k: float(np.polyval(c, Mt - 172.6)) for k, c in co.items()}
        da, dh = als - 0.1180, Mh - 125.13
        x["g3"] += SL["g3_as"] * da
        x["yt"] += SL["yt_as"] * da + SL["yt_Mh"] * dh
        x["lam"] += SL["lam_Mh"] * dh + SL["lam_as"] * da
        return x
    x = match_Mt(Mt, als, Mh, MW_PDG, "SMDR25")
    c1, c0 = _sfit_poly(cfg), _sfit_poly("xi1")
    for k in ("g3", "g", "gp", "yt", "lam"):
        x[k] += float(np.polyval(c1[k], Mt - 172.6) - np.polyval(c0[k], Mt - 172.6))
    return x


for _c in ("xi0.25", "xi0.5", "xi2", "xi4", "t0.5", "t2", "h0.5", "h2", "Z0.5", "Z2", "G0.5", "G2", "d0.5", "d2"):
    VARIANTS["sfit_" + _c] = dict(match="SFIT", sfit_cfg=_c, sfit_mode="delta")
VARIANTS["sfit_xi1"] = dict(match="SFIT", sfit_cfg="xi1", sfit_mode="direct")


def spec(variant):
    s = dict(match="SMDR25", xi=1.0, loops=5, impl="smdr", qcdN=4, decN=4, lam3=False, z4=Q.Z4_NL5, mode="fwd",
             qcd5=False, Mh_override=None, sfit_cfg=None, sfit_mode="delta")
    s.update(VARIANTS[variant] if isinstance(variant, str) else variant)
    return s


def lam3_qcd(g3, Mt, GF=1.1663787e-5):
    """Buttazzo footnote 5 (PDF p.14): 3-loop pure-QCD lambda(mu=Mt), Mh,MW,MZ << Mt limit."""
    L2 = math.log(2.0)
    Li4h = 0.5174790616738994
    br = (176 * PI ** 4 + 240 * PI ** 2 * (3 + 4 * L2 ** 2 + 6 * L2)
          + 15 * (607 - 64 * L2 ** 4 - 1536 * Li4h + 576 * RG.ZETA3))
    return -8.0 / 135 * g3 ** 4 * GF ** 2 * Mt ** 4 / (4 * PI) ** 6 * br


def qcd5_rel(g3):
    """estimated relative 5-loop term of m(Mt)/M (geometric continuation t5 = t4^2/t3 of the z_m series at mu=Mt);
    used only as a truncation-error probe of the pole-mass relation."""
    t = Q.zm_terms(g3 ** 2 / (4 * PI ** 2), 0.0, 4)
    return t[4] ** 2 / t[3] / (1.0 + sum(t[1:]))


def _from_ref(R, Mt, als, Mh):
    dt, da, dh = Mt - R["Mt"], als - R["als"], Mh - R["Mh"]
    return dict(g3=R["g3"] + SL["g3_as"] * da + SL["g3_Mt"] * dt, g=R["g"] + SL["g_Mt"] * dt,
                gp=R["gp"] + SL["gp_Mt"] * dt,
                yt=R["yt"] + SL["yt_Mt"] * dt + SL["yt_as"] * da + SL["yt_Mh"] * dh,
                lam=R["lam"] + SL["lam_Mh"] * dh + SL["lam_Mt"] * dt + SL["lam_as"] * da,
                yb=R["yb"], ytau=R["ytau"], v=R["v"])


def buttazzo(Mt, als, Mh, MW):
    dt, da, dh, dw = Mt - 173.34, (als - 0.1184) / 0.0007, Mh - 125.15, (MW - 80.384) / 0.014
    return dict(lam=0.12604 + 0.00206 * dh - 0.00004 * dt, yt=0.93690 + 0.00556 * dt - 0.00042 * da,
                g=0.64779 + 0.00004 * dt + 0.00011 * dw, gp=0.35830 + 0.00011 * dt - 0.00020 * dw,
                g3=1.1666 + 0.00314 * da - 0.00046 * dt)


def match_Mt(Mt, als, Mh, MW=MW_PDG, scheme="SMDR25", z4=Q.Z4_NL5):
    """MSbar couplings at mu = Mt: dict g3,g,gp,yt,yb,ytau,lam,v."""
    if scheme in ("SMDR25", "SMDR25_MW"):
        x = _from_ref(SMDR25, Mt, als, Mh)
        if scheme == "SMDR25_MW":
            x["g"] += SL["g_MW"] * (MW - SMDR25["MW_pred"])
            x["gp"] += SL["gp_MW"] * (MW - SMDR25["MW_pred"])
    elif scheme == "SMDR19":
        x = _from_ref(SMDR19, Mt, als, Mh)
    elif scheme == "BKPV15":
        B = BKPV
        dd = ((als - B["als"]) / B["das"], (MW - B["MW"]) / B["dMW"], (Mh - B["Mh"]) / B["dMh"],
              (Mt - B["Mt"]) / B["dMt"], (Mt - B["mu_thr"]) / B["mu_thr"])
        x = {k: B["x0"][k] + sum(c * d for c, d in zip(B["coef"][k], dd)) for k in B["x0"]}
        x.update(ytau=SMDR25["ytau"], v=SMDR25["v"])
    elif scheme in ("B13", "B13c"):
        x = buttazzo(Mt, als, Mh, MW)
        x.update(yb=SMDR25["yb"], ytau=SMDR25["ytau"], v=SMDR25["v"])
        if scheme == "B13c":
            as6 = Q.as6_at(Mt, als, nrun=4, ndec=3)
            x["g3"] = math.sqrt(4 * PI * as6)
            a = as6 / PI
            x["yt"] *= Q.zm(a, 0.0, 4, z4) / Q.zm(a, 0.0, 3, z4)
    else:
        raise ValueError(scheme)
    if abs(z4 - Q.Z4_NL5) > 1e-9 and scheme != "B13c":   # z4 variation on schemes that contain the 4-loop QCD term
        a = x["g3"] ** 2 / (4 * PI ** 2)
        x["yt"] *= Q.zm(a, 0.0, 4, z4) / Q.zm(a, 0.0, 4, Q.Z4_NL5)
    return x


def vec(x):
    return np.array([x["g3"], x["g"], x["gp"], x["yt"], x["yb"], x["ytau"], x["lam"], x["v"]], float)


# ------------------------------------------------------------------------------------------------ mu0 variation
def _b1(y):
    o = RG.smdr_orders(y, 1)[0]
    return np.array([o[0] / y[0], o[1] / y[1], o[2] / y[2], o[3] / y[3], o[6]])


def taylor_rest(y, t0):
    """NNLO-truncated Taylor increment of (ln g3, ln g, ln g', ln yt, lam) over Delta ln mu = t0 at the point y,
    pure-QCD towers (g3^2n parts of b_g3, b_yt) removed."""
    o = RG.smdr_orders(y, 2)
    b1 = np.array([o[0][0] / y[0], o[0][1] / y[1], o[0][2] / y[2], o[0][3] / y[3], o[0][6]])
    b2 = np.array([o[1][0] / y[0], o[1][1] / y[1], o[1][2] / y[2], o[1][3] / y[3], o[1][6]])
    g34 = y[0] ** 4
    b1[0] -= -7 * K * y[0] ** 2
    b1[3] -= -8 * K * y[0] ** 2
    b2[0] -= -26 * K * K * g34
    b2[3] -= -108 * K * K * g34
    full1 = np.array(RG.smdr_orders(y, 1)[0])
    h = 1e-3
    D = (_b1(y + h * full1) - _b1(y - h * full1)) / (2 * h)
    D[0] -= 98 * K * K * g34
    D[3] -= 112 * K * K * g34
    return t0 * (b1 + b2) + 0.5 * t0 * t0 * D


def mu0_shift(y, Mt, als, xi, loops=5, qcdN=4, decN=4, z4=Q.Z4_NL5, detail=False, mode="fwd"):
    """fixed-order NNLO couplings at mu0 = xi*Mt reconstructed from y(Mt) (see module doc).
    mode 'fwd': truncated terms built from the couplings at Mt; 'bwd': from the couplings at mu0 (solved
    self-consistently: y(Mt) = truncated evolution of y_FO(mu0) back to Mt), as in a matching computed at mu0."""
    t0 = math.log(xi)
    mu0 = xi * Mt
    sol = RG.solve(y, Mt, mu0, loops=loops)
    yR = sol.y[:, -1].copy()
    T2 = taylor_rest(y, t0)
    X0 = np.array([math.log(y[0]), math.log(y[1]), math.log(y[2]), math.log(y[3]), y[6]])
    X1 = np.array([math.log(yR[0]), math.log(yR[1]), math.log(yR[2]), math.log(yR[3]), yR[6]])
    Iyt = quad(lambda t: RG.bqcd_yt(sol.sol(math.log(Mt) + t)[0], loops), 0.0, t0, epsabs=1e-14, epsrel=1e-12)[0]
    Ig3 = quad(lambda t: RG.bqcd_g3(sol.sol(math.log(Mt) + t)[0], loops), 0.0, t0, epsabs=1e-14, epsrel=1e-12)[0]
    rest_exact = X1 - X0
    rest_exact[0] -= Ig3
    rest_exact[3] -= Iyt
    Drest = T2 - rest_exact
    aM, a0 = y[0] ** 2 / (4 * PI ** 2), yR[0] ** 2 / (4 * PI ** 2)
    dq_yt = math.log(Q.zm(a0, 2 * t0, qcdN, z4) / Q.zm(aM, 0.0, qcdN, z4)) - Iyt
    A = Q.run_a(Q.as6_at(Mt, als, ndec=decN) / PI, Mt, mu0, 6, 5)
    B = Q.as6_at(Mt, als, mu_dec=mu0, mu_out=mu0, ndec=decN) / PI
    dq_g3 = 0.5 * math.log(B / A)

    def apply(Dr):
        yN = yR.copy()
        yN[0] *= math.exp(Dr[0] + dq_g3)
        yN[1] *= math.exp(Dr[1])
        yN[2] *= math.exp(Dr[2])
        yN[3] *= math.exp(Dr[3] + dq_yt)
        yN[6] += Dr[4]
        return yN
    yN = apply(Drest)
    if mode == "bwd":
        for _ in range(10):
            Drest = -taylor_rest(yN, -t0) - rest_exact
            yN = apply(Drest)
    if detail:
        return yN, dict(Drest=Drest.tolist(), dq_yt=dq_yt, dq_g3=dq_g3, mu0=mu0, mode=mode)
    return yN


# ------------------------------------------------------------------------------------------------ provider
_CACHE = {}


def _solution(Mt, als, Mh, MW, variant, mu_hi=MPL * 1.01):
    s = spec(variant)
    key = (round(Mt, 6), round(als, 7), round(Mh, 6), round(MW, 6), json.dumps(s, sort_keys=True))
    if key in _CACHE:
        return _CACHE[key]
    if s["Mh_override"]:
        Mh = s["Mh_override"]
    if s["match"] == "SFIT":
        x = match_sfit(Mt, als, Mh, s["sfit_cfg"], s["sfit_mode"])
    else:
        x = match_Mt(Mt, als, Mh, MW, s["match"], s["z4"])
    if s["lam3"]:
        x["lam"] += lam3_qcd(x["g3"], Mt)
    if s["qcd5"]:
        x["yt"] *= 1.0 + qcd5_rel(x["g3"])
    y = vec(x)
    mu0 = Mt
    if abs(s["xi"] - 1) > 1e-12:
        y = mu0_shift(y, Mt, als, s["xi"], s["loops"], s["qcdN"], s["decN"], s["z4"], mode=s["mode"])
        mu0 = s["xi"] * Mt
    sol = RG.solve(y, mu0, mu_hi, loops=s["loops"], impl=s["impl"])
    _CACHE[key] = (sol, y, mu0)
    if len(_CACHE) > 400:
        _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[key]


def couplings(Mt, alphas, Mh=125.13, mu_array=(MU_B,), variant="central", MW=MW_PDG):
    sol, _, _ = _solution(Mt, alphas, Mh, MW, variant)
    mu = np.atleast_1d(np.asarray(mu_array, float))
    Y = sol.sol(np.log(mu))
    return dict(mu=mu, g3=Y[0], g2=Y[1], gY=Y[2], yt=Y[3], yb=Y[4], ytau=Y[5], lam=Y[6], v=Y[7])


def coup_point(Mt, alphas, Mh, mu, variant="central", MW=MW_PDG):
    c = couplings(Mt, alphas, Mh, [mu], variant, MW)
    return {k: float(v[0]) for k, v in c.items() if k != "mu"}


def lam_b(Mt, alphas=0.1180, Mh=125.13, variant="central", MW=MW_PDG, mu=MU_B):
    return coup_point(Mt, alphas, Mh, mu, variant, MW)["lam"]


def rge_estimates(variant="central", Mt=172.6, als=0.1180, Mh=125.13):
    d = dict(variant=variant)
    d["lam_b"] = lam_b(Mt, als, Mh, variant)
    d["dlam_dMt"] = (lam_b(Mt + 0.1, als, Mh, variant) - lam_b(Mt - 0.1, als, Mh, variant)) / 0.2
    d["dlam_das"] = (lam_b(Mt, als + 0.0001, Mh, variant) - lam_b(Mt, als - 0.0001, Mh, variant)) / 0.0002
    d["dlam_dMh"] = (lam_b(Mt, als, Mh + 0.1, variant) - lam_b(Mt, als, Mh - 0.1, variant)) / 0.2
    d["dmc_das_linear"] = -d["dlam_das"] / d["dlam_dMt"]
    return d
