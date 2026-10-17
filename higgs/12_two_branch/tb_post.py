# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/two_branch/tb_post.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] C159 summary: critical masses for P2 = P1^2 (172.376/172.759/173.124, run02 base), GH interference eps 0.9948-0.99997, Frechet bands.  Paper: PRD App. H3; PRL one sentence.

C159 / D130 post: (i) collect TDSE shards + static check -> P2(P1) laws and decoherence diagnostics;
(ii) critical m_t for P2 = 0.5 / 0.9 (and 0.1) on the central C149 run02 P1 grid (alpha_s = 0.1180), spline of
ln(-ln P1) in m_t; offsets relative to the P1 critical values carried to alpha_s = 0.1171 / 0.1189.
Usage: python -B tb_post.py
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
sys.dont_write_bytecode = True
import glob
import json
import math
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = _zp.WORK
OUT = os.path.join(ROOT, _zp.w("r3/two_branch/run01"))
C149 = os.path.join(ROOT, _zp.w("r3/holonomy_cross/run02/C149_run02_summary.json"))
HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def crit_tables():
    d = json.load(open(C149, encoding="utf-8"))
    m = np.array(sorted(float(k) for k in d["P1_grid"]))
    P = np.array([d["P1_grid"]["%.1f" % x] for x in m])
    L = CubicSpline(m, np.log(-np.log(P)))
    Lin = lambda mm: float(np.interp(mm, m, np.log(-np.log(P))))  # noqa: E731

    def mt_of(P1, f=None):
        f = f or L
        t = math.log(-math.log(P1))
        return brentq(lambda x: float(f(x)) - t, m[0], m[-1])
    pub = d["crit"]["0.1180"]["with_cross"]
    check = {k: dict(spline=mt_of(float(k.split("=")[1])), published=v, diff=mt_of(float(k.split("=")[1])) - v,
                     linear_interp=mt_of(float(k.split("=")[1]), Lin)) for k, v in pub.items()}
    out = dict(validation_P1_crit=check)
    tg = {}
    for P2 in (0.9, 0.5, 0.1):
        P1_kd = math.sqrt(P2)                 # P2 = P1^2 / w0 with w0 = 1 (Gamow-like, central model)
        P1_kd_w = math.sqrt(P2 * 1.002)       # w0 = 1.002 (D111 upper end)
        P1_gh = P2 ** (1.0 / 3.0)             # GH diagonal P1^3 (NOT a probability; reference only)
        fr_lo, fr_hi = P2, (1 + P2) / 2       # Frechet: P2 <= P1 needs P1 >= P2; P2 >= 2P1-1 guaranteed if P1 >= (1+P2)/2
        tg["P2=%.1f" % P2] = dict(
            P1_needed_KD=P1_kd, mt_KD=mt_of(P1_kd), mt_KD_w0_1p002=mt_of(P1_kd_w), mt_KD_linear_interp=mt_of(P1_kd, Lin),
            mt_P1_equal_P2=mt_of(P2), offset_KD_vs_P1crit=mt_of(P1_kd) - mt_of(P2),
            frechet_P1_range=[fr_lo, fr_hi], frechet_mt_range=[mt_of(fr_hi), mt_of(fr_lo)],
            GH_diag_invalid=dict(P1_needed=P1_gh, mt=mt_of(P1_gh)))
    out["targets_as0.1180"] = tg
    # carry to other alpha_s by the offset (crit shifts are uniform: dm_c/dalpha_s = 385/387/388 GeV)
    oth = {}
    for a in ("0.1171", "0.1189"):
        c = d["crit"][a]["with_cross"]
        oth[a] = {"P2=0.9": c["P1=0.9"] + tg["P2=0.9"]["offset_KD_vs_P1crit"],
                  "P2=0.5": c["P1=0.5"] + tg["P2=0.5"]["offset_KD_vs_P1crit"],
                  "P2=0.1": c["P1=0.1"] + tg["P2=0.1"]["offset_KD_vs_P1crit"]}
    out["targets_other_alpha_s_by_offset"] = oth
    rows = []
    for x, p in zip(m, P):
        rows.append(dict(mt=float(x), P1=float(p), P2_KD=float(p * p), GH_diag_invalid=float(p ** 3),
                         frechet_lo=max(0.0, 2 * p - 1), frechet_hi=float(p)))
    out["grid_as0.1180"] = rows
    out["slope_dlnmlnP_dm_at_targets"] = {k: float(L(v["mt_KD"], 1)) for k, v in tg.items()}
    out["C149_sha256"] = sha(C149)
    return out


def collect():
    shards = sorted(glob.glob(os.path.join(OUT, "shards", "tdse_*.json")))
    S = dict(files=[os.path.basename(s) for s in shards], per_run={}, ranges={})
    allRK, allRG, alleps, allIm, allc2 = [], [], [], [], []
    for s in shards:
        r = json.load(open(s, encoding="utf-8"))
        if r["tag"] == "smoke":
            continue
        key = "%s|mt=%.1f" % (r["tag"], r["mt"])
        d = {}
        for nm, st in r["states"].items():
            recs = st["recs"]
            d[nm] = dict(w0=st["w0"], inf=st["inf"],
                         u_T=[q["u_T"] for q in recs], P1=[q["P1"] for q in recs],
                         R_K=[q["R_K"] for q in recs], R_G=[q["R_G"] for q in recs], eps=[q["eps"] for q in recs],
                         K_im_over_re=[q["K_im"] / q["K_re"] for q in recs], marg_viol=[q["marg_viol"] for q in recs],
                         GH_below_frechet=[q["GH_below_frechet"] for q in recs],
                         # 1 - |<A|X>|^2/(|A|^2|X|^2): the two (-)-alternatives give parallel (+)-survivor vectors
                         one_minus_cos2=[1 - (q["K_re"] ** 2 + q["K_im"] ** 2) / (q["P1"] * q["G_GH"]) for q in recs],
                         # |E+ xi|^2/P1^3, xi = E- psi0 - P1 psi0 (exact identity G - 2 P1 ReK + P1^3)
                         Exi2_over_P1cube=[(q["G_GH"] - 2 * q["P1"] * q["K_re"] + q["P1"] ** 3) / q["P1"] ** 3 for q in recs],
                         # linear (GP) four cells, all must be >= 0 (GP Eq. 12)
                         lin_cells_min=[min(q["K_re"], q["P1"] - q["K_re"], 1 - 2 * q["P1"] + q["K_re"]) for q in recs])
            for q in recs:
                if q["u_T"] >= 1.0:
                    allRK.append(q["R_K"])
                    allRG.append(q["R_G"])
                    alleps.append(q["eps"])
                    allIm.append(abs(q["K_im"] / q["K_re"]))
                    allc2.append(1 - (q["K_re"] ** 2 + q["K_im"] ** 2) / (q["P1"] * q["G_GH"]))
        S["per_run"][key] = dict(states=d, time_s=r["time_s"], table_sha256=r["table_sha256"],
                                 code_sha256=r["code_sha256"])
    if allRK:
        S["ranges"] = dict(R_K=[min(allRK), max(allRK)], R_G=[min(allRG), max(allRG)], eps=[min(alleps), max(alleps)],
                           abs_Kim_over_Kre_max=max(allIm), one_minus_cos2=[min(allc2), max(allc2)],
                           note="u_T >= 1, all states / m_t / numerical variants")
    st = os.path.join(OUT, "static_check.json")
    if os.path.exists(st):
        z = json.load(open(st, encoding="utf-8"))
        rk = [q["R_K"] for r in z["runs"] for s in r["states"].values() for q in s["recs"] if q["T"] >= 50]
        rg = [q["R_G"] for r in z["runs"] for s in r["states"].values() for q in s["recs"] if q["T"] >= 50]
        ep = [q["eps"] for r in z["runs"] for s in r["states"].values() for q in s["recs"] if q["T"] >= 100]
        cnd = []
        for r in z["runs"]:
            for s in r["states"].values():
                ref = {q["T"]: q for q in s["recs"]}
                for c in s["cn"]:
                    cnd.append(max(abs(c["K_re"] - ref[c["T"]]["K_re"]), abs(c["G"] - ref[c["T"]]["G"])))
        S["static"] = dict(R_K=[min(rk), max(rk)], R_G=[min(rg), max(rg)], eps_T100plus=[min(ep), max(ep)],
                           CN_vs_eig_maxabs=max(cnd))
    return S


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ct = crit_tables()
    co = collect()
    tg = ct["targets_as0.1180"]
    head = dict(
        definitions=dict(
            GH="time-ordered two-time set C_ab = P_b(+T) P_a(-T) conditioned on the turnaround state psi0; "
               "D(h,h') = <C_h' psi0|C_h psi0>",
            linear="q_ab = Re<psi0|E+_b E-_a|psi0> = Re Tr(C_ab rho0) (Goldstein-Page Eq. 10-12; Margenau-Hill)"),
        GH_decoheres=False,
        GH_interference_eps=co.get("ranges", {}).get("eps"),
        GH_diag_law="D(in,in) = P1^3/w0^2 (R_G range %s); below Frechet lower bound for P1 > 0.618" %
                    (co.get("ranges", {}).get("R_G"),),
        linear_law="P2_lin = P1^2/w0 (R_K range %s); inside Frechet band; all four cells >= 0" %
                   (co.get("ranges", {}).get("R_K"),),
        crit_as0p1180=dict(P2_0p9=tg["P2=0.9"]["mt_KD"], P2_0p5=tg["P2=0.5"]["mt_KD"], P2_0p1=tg["P2=0.1"]["mt_KD"],
                           offsets_vs_P1crit={k: v["offset_KD_vs_P1crit"] for k, v in tg.items()},
                           frechet_mt_ranges={k: v["frechet_mt_range"] for k, v in tg.items()},
                           offset_formula="Delta m = -ln2 / <d ln(-ln P1)/dm_t> (P2 = P1^2 doubles the integrated hazard)"),
        crit_other_alpha_s=ct["targets_other_alpha_s_by_offset"])
    summ = dict(task="T7 two_branch C159/D130", headline=head, crit=ct, tdse=co,
                code_sha256={f: sha(os.path.join(HERE if os.path.exists(os.path.join(HERE, f)) else _zp.ZMLIB, f)) for f in
                             ("tb_tdse.py", "tb_static.py", "tb_post.py", "hr_core.py", "hr_potential.py")})
    with open(os.path.join(OUT, "C159_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(ct["validation_P1_crit"], indent=0))
    print(json.dumps(ct["targets_as0.1180"], indent=0))
    print(json.dumps(ct["targets_other_alpha_s_by_offset"], indent=0))
    print(json.dumps(ct["slope_dlnmlnP_dm_at_targets"]))
    print(json.dumps(co.get("ranges"), indent=0))
    print(json.dumps(co.get("static"), indent=0))
    for k, v in co["per_run"].items():
        for nm, s in v["states"].items():
            print(k, nm, "w0=%.5f" % s["w0"], "inf:", None if s["inf"] is None else
                  "P1=%.5f KD=%.5f KD/P1^2=%.5f G/(P1^3/w0^2)=%.5f" % (s["inf"]["P1"], s["inf"]["P2_KD"],
                                                                     s["inf"]["KD_over_P1sq"], s["inf"]["G_over_P1cube_w0sq"]))
