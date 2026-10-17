r"""C136_stage1.py -- Stage 1 (a)-(e), branch higgs_survival (Claude v12, 2026-10-13).
    python -B C136_stage1.py <phase> [...]     phases: checks tables repro grid ecs uv sens
Each phase writes run01/C136_s1_<phase>.json as soon as it finishes (grid: after every variant);
gauge tables cached in run01/gauge_<tag>.npz.  Workers: <= 4, BLAS single-thread (set in C136_core).
"""
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                       # noqa: E402
import C136_core as C                    # noqa: E402
import C136_rge as R                     # noqa: E402

RUN = C.RES / "run01"
RGRID = np.concatenate([np.geomspace(0.004, 0.2, 160, endpoint=False), np.arange(0.2, 26.0 + 1e-9, 0.004)])
LAMS = [-0.003, -0.004, -0.005, -0.006, -0.007, -0.008, -0.009, -0.010, -0.011, -0.012, -0.0143, -0.016, -0.018, -0.020]
GC = {"one_0.9": -math.log(0.9) / C.ETA_ONE, "one_0.5": math.log(2) / C.ETA_ONE,
      "two_0.9": -math.log(0.9) / (2 * C.ETA_ONE), "two_0.5": math.log(2) / (2 * C.ETA_ONE)}
BETA_SM = -4.0e-4   # overwritten from the RGE (2-loop beta_lambda at 1/L_b) in main()
VAR = {"L0": dict(match="none"), "M0": dict(match="gt"), "MU": dict(match="gt", higgs="U"),
       "MF": dict(match="gt", higgs="F"), "ML": dict(match="gt", higgs="L"), "LQ": dict(match="gt", higgs="LQ"),
       "MU_CW": dict(match="gt", higgs="U", cw=True), "ML_CW": dict(match="gt", higgs="L", cw=True),
       "MU_RG": dict(match="gt", higgs="U", cw=True, rg_beta="SM"), "MU_hol": dict(match="gt", higgs="U", table="hol")}


def conv(o):
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def jdump(name, obj):
    (RUN / name).write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")


def status(update):
    f = RUN / "STATUS.json"
    s = json.loads(f.read_text(encoding="utf-8"))
    s.setdefault("key_numbers", {}).update(update.get("key_numbers", {}))
    s.setdefault("completed", []).extend(update.get("completed", []))
    for k in ("next_step", "updated"):
        if k in update:
            s[k] = update[k]
    f.write_text(json.dumps(s, indent=1, default=conv), encoding="utf-8")


def cp_of(tag="v11"):
    cp = dict(C.COUP_V11)
    if tag == "hol":
        cp.update(C.HOL_V11)
    elif tag == "g2p":
        cp["g2"] *= 1.01
    elif tag == "gYp":
        cp["gY"] *= 1.01
    return cp


def get_table(tag):
    f = RUN / f"gauge_{tag}.npz"
    if f.exists():
        d = np.load(f)
        return d["r"], d["G"], d["P"]
    t0 = time.time()
    t = C.gauge_table(RGRID, cp_of(tag))
    G = (t["E"] + t["Br"]).real
    np.savez(f, r=t["r"], G=G, P=t["P"], E=t["E"], Br=t["Br"], ang=t["parts"]["ang"], ym=t["parts"]["ym"],
             mass=t["parts"]["mass"], kin=t["parts"]["kin"], seconds=time.time() - t0)
    return t["r"], G, t["P"]


def opt_of(name, extra=None):
    o = dict(VAR[name])
    if o.get("rg_beta") == "SM":
        o["rg_beta"] = BETA_SM
    if extra:
        o.update(extra)
    return o


def potential(name, lam, cp=None, extra=None, tag=None):
    o = opt_of(name, extra)
    tag = tag or o.get("table", "v11")
    r, G, P = get_table(tag)
    cp = cp or cp_of(tag)
    rmax = C.higgs_rmax(lam, o)
    sel = r <= rmax
    X = C.extra_terms(r[sel], lam, cp, o)
    return r[sel], G[sel] + X.real, float(np.max(np.abs(X.imag)))


def solve_case(args):
    r, U, lam, label, rout, r0 = args
    rad = C.Radial(r, U)
    out = dict(label=label, lam=lam, beta_origin=rad.beta, rout=min(rout, r[-1] - 1.0))
    try:
        E0, lm, wk0 = rad.guess()
        out["landmarks"], out["guess"] = lm, E0
        g0 = max(-2 * E0.imag, 1e-13)
        best, attempts = None, []
        for scale in (0.5, 3.0, 20.0, 200.0):
            dE = complex(0.3, -0.5) * g0 * scale
            try:
                E, it = rad.pole(E0, r0=r0, rout=out["rout"], dE=dE)
            except Exception as e:      # noqa: BLE001
                attempts.append(dict(scale=scale, error=repr(e)))
                continue
            gam = -2 * E.imag
            ok = (abs(E.real - E0.real) < 0.02) and (0 < gam < 10 * g0 + 1e-9)
            attempts.append(dict(scale=scale, E=E, ok=bool(ok)))
            if ok:
                best = (E, it)
                break
        out["attempts"] = attempts
        if best is None:
            raise RuntimeError("no accepted pole")
        E, it = best
        out.update(Q=E.real, gamma=-2 * E.imag, iters=it)
        out["wkb"] = rad.wkb(E.real)
        out["resolved"] = bool(out["gamma"] > 1e-10)
    except Exception as e:      # noqa: BLE001
        out["error"] = repr(e)
        try:
            out["wkb_guess"] = rad.wkb(rad.guess()[0].real)
        except Exception:       # noqa: BLE001
            pass
    return out


def bound_energy(name, lam, r_wall=12.0, tag=None):
    r, U, _ = potential(name, lam, tag=tag)
    rad = C.Radial(r, U)
    e, levels = rad.dirichlet_rich(r_wall)
    return e, levels


def lam_crit(rows):
    pts = sorted((x["lam"], math.log(x["gamma"])) for x in rows if x.get("resolved") and x.get("gamma", 0) > 0)
    if len(pts) < 4:
        return {}
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    lam = np.array([p[0] for p in pts])
    lg = np.array([p[1] for p in pts])
    sp = CubicSpline(lam, lg)
    out = {}
    for k, g in GC.items():
        f = lambda x: sp(x) - math.log(g)
        if f(lam[0]) * f(lam[-1]) < 0:
            out[k] = float(brentq(f, lam[0], lam[-1]))
        else:
            out[k] = None
    return out


# ------------------------------------------------------------------------------------------------ phases
def ph_checks():
    out = {}
    eps = [1e-5, 1e-4, 1e-3]
    out["DP_over_mu2_small"] = {f"{e:g}": [float((C.DP(e)[0] / e).real), float((C.DP(-e, "neg")[0] / -e).real)] for e in eps}
    out["DA_over_mu2_small"] = {f"{e:g}": float((C.DA(e)[0] / e).real) for e in eps}
    out["ref_T_half"] = [C.TB_HALF_REF, C.TF_HALF_REF]
    xs = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    FP = C.bessel_F(xs, "P")
    FA = C.bessel_F(xs, "A")
    hp = C.DP(xs ** 2).real
    ha = C.DA(xs ** 2).real
    out["bessel_P"] = {f"{x}": [float(hp[i]), float(FP[i] + C.GAMMA_Z2 / 2 - x / 2)] for i, x in enumerate(xs)}
    out["bessel_A_differences"] = {f"{xs[i]}-{xs[i+1]}": [float(ha[i] - ha[i + 1]), float(FA[i] - FA[i + 1])] for i in range(4)}
    out["largex_P_DP(20^2)+10_vs_GAMMA/2"] = [float(C.DP(400.0)[0].real + 10), C.GAMMA_Z2 / 2]
    out["largex_A_minusFA0"] = float(C.DA(900.0)[0].real)
    h = {}
    for e in (1e-3, 2e-3, 4e-3):
        h[e] = float(((C.DP(e)[0] + C.DP(-e, "neg")[0]) / (2 * e * e)).real)
    out["log_coeff_est"] = [(h[2e-3] - h[1e-3]) / math.log(2), (h[4e-3] - h[2e-3]) / math.log(2), -1 / (64 * math.pi ** 2)]
    z0, dz = 1.0 + 0.5j, 1e-4
    dr = (C.DP(z0 + dz)[0] - C.DP(z0 - dz)[0]) / (2 * dz)
    di = (C.DP(z0 + 1j * dz)[0] - C.DP(z0 - 1j * dz)[0]) / (2j * dz)
    out["cauchy_riemann_DP"] = [dr, di, abs(dr - di)]
    z1 = -0.7 + 0.3j
    dr = (C.DP(z1 + dz, "neg")[0] - C.DP(z1 - dz, "neg")[0]) / (2 * dz)
    di = (C.DP(z1 + 1j * dz, "neg")[0] - C.DP(z1 - 1j * dz, "neg")[0]) / (2j * dz)
    out["cauchy_riemann_DP_neg"] = [dr, di, abs(dr - di)]
    jdump("C136_s1_checks.json", out)
    return out


def ph_tables():
    out = {}
    for tag in ("v11", "hol"):
        r, G, P = get_table(tag)
        cp = cp_of(tag)
        spot = {}
        for rr in (0.05, 0.5, 2.0, 4.0, 8.0, 15.0):
            i = int(np.argmin(np.abs(r - rr)))
            p, fmin, it = C.global_min(r[i], cp, starts=16)
            Eg = float(C.F_E(*C._cargs(p, r[i], cp)))
            Et = float(C.F_E(*C._cargs(P[i], r[i], cp)))
            spot[f"{r[i]:.4f}"] = dict(global_min=Eg, table=Et, diff=Eg - Et, p_table=P[i].tolist(), p_global=p.tolist())
        ag = 0.75 * (2 * cp["g2"] + math.sqrt(cp["g2"] ** 2 + cp["gY"] ** 2))
        asym = {f"{rr}": float(G[int(np.argmin(np.abs(r - rr)))] - ag * rr) for rr in (5, 10, 15, 20, 25)}
        out[tag] = dict(spot=spot, G_minus_ag_r=asym, a_g=ag, G_at_0p004=float(G[0]),
                        su2_gauss_g23=(27 / 8) * (2 * cp["g2"] ** 2) ** (1 / 3))
    jdump("C136_s1_tables.json", out)
    return out


def ph_repro():
    out = {"lam0_local_bound": bound_energy("L0", 0.0, r_wall=16.0)}
    rows = []
    for name, lams in (("L0", [-0.0143, -0.010, -0.005, -0.020]), ("M0", [-0.0143, -0.010, -0.005, -0.020]),
                       ("LQ", [-0.0143, -0.010])):
        for lam in lams:
            r, U, im = potential(name, lam)
            res = solve_case((r, U, lam, name, 14.0, 0.02))
            res["imag_extra_max"] = im
            ref = (C.CODEX["local"] if name == "L0" else C.CODEX["matched"] if name == "M0" else C.CODEX["LQ"]).get(lam)
            res["codex"] = ref
            res["ratio_to_codex"] = (res.get("gamma", float("nan")) / ref) if ref else None
            rows.append(res)
            jdump("C136_s1_repro.json", dict(out, rows=rows))
    conv_rows = []
    r, U, _ = potential("M0", -0.0143)
    for rout in (10.0, 12.0, 16.0, 18.0, 22.0):
        conv_rows.append(dict(rout=rout, **{k: v for k, v in solve_case((r, U, -0.0143, "M0", rout, 0.02)).items()
                                             if k in ("Q", "gamma", "error")}))
    for r0 in (0.01, 0.04):
        conv_rows.append(dict(r0=r0, **{k: v for k, v in solve_case((r, U, -0.0143, "M0", 14.0, r0)).items()
                                         if k in ("Q", "gamma", "error")}))
    rad = C.Radial(r, U)
    E, _ = rad.pole(complex(4.34, -0.0096), r0=0.02, rout=14.0, rtol=1e-13)
    conv_rows.append(dict(rtol=1e-13, Q=E.real, gamma=-2 * E.imag))
    out["convergence_M0_0143"] = conv_rows
    out["rows"] = rows
    jdump("C136_s1_repro.json", out)
    m0 = {x["lam"]: x.get("ratio_to_codex") for x in rows if x["label"] == "M0"}
    status({"completed": ["stage1a repro"], "key_numbers": {"repro_ratio_M0_vs_codex": m0,
                                                              "lam0_local_bound": out["lam0_local_bound"][0]}})
    return out


def ph_grid():
    tasks = []
    for name in VAR:
        lams = LAMS if name != "LQ" else [-0.006, -0.010, -0.0143, -0.020]
        for lam in lams:
            r, U, im = potential(name, lam)
            tasks.append((r, U, lam, name, 14.0, 0.02))
    res = {name: [] for name in VAR}
    with Pool(4) as pool:
        for k, x in enumerate(pool.imap_unordered(solve_case, tasks)):
            res[x["label"]].append(x)
            if k % 10 == 9 or k == len(tasks) - 1:
                jdump("C136_s1_grid.json", dict(rows=res))
    crit = {}
    table = {}
    for name, rows in res.items():
        rows.sort(key=lambda x: -x["lam"])
        crit[name] = lam_crit(rows)
        table[name] = {f"{x['lam']}": [x.get("Q"), x.get("gamma"), (x.get("wkb") or {}).get("gamma"), x.get("resolved")]
                       for x in rows}
    qb = {}
    for name in ("L0", "M0", "MU", "ML"):
        qb[name] = {f"{lam}": bound_energy(name, lam)[0] for lam in (0.01, 0.0)}
    out = dict(rows=res, gamma_table=table, lam_crit=crit, gamma_crit=GC, bound_Q=qb, beta_SM_used=BETA_SM)
    jdump("C136_s1_grid.json", out)
    status({"completed": ["stage1e grid"], "key_numbers": {"lam_crit": crit, "bound_Q_lam+0.01": {k: v["0.01"] for k, v in qb.items()}}})
    return out


def ecs_case(name, lam, thetas=(0.25, 0.40), R0s=(6.5, 8.0), rho_max=5.0, hf=0.001):
    o = opt_of(name)
    tag = o.get("table", "v11")
    cp = cp_of(tag)
    r, U, _ = potential(name, lam)
    rad = C.Radial(r, U)
    E0, lm, wk = rad.guess()
    Es, _ = rad.pole(E0, r0=0.02, rout=14.0)
    rt, G, P = get_table(tag)
    out = dict(siegert=Es, runs=[])
    for R0 in R0s:
        iR = int(np.argmin(np.abs(rt - R0)))
        R0n = float(rt[iR])
        nL = int(round((R0n - 0.02) / (4 * hf))) * 4
        s0 = R0n - nL * hf
        nR = int(round(rho_max / hf / 4)) * 4
        for th in thetas:
            rho = hf * np.arange(nR + 1)
            z, Gz, Pz = C.gauge_on_ray(P[iR], R0n, th, rho, cp)
            Uz = Gz + C.extra_terms(z, lam, cp, o)
            sL = s0 + hf * np.arange(nL + 1)
            zz = np.concatenate([sL.astype(complex), z[1:]])
            UU = np.concatenate([rad.Uf(sL).astype(complex), Uz[1:]])
            ev = []
            for m in (4, 2, 1):
                vals = C.ecs_eigs(UU[::m], zz[::m], Es, k=4)
                ev.append(vals[0])
            Er = C.richardson3(*ev)
            dz = 1e-4
            zc = R0n + 2.0 * np.exp(1j * th)
            zz3 = np.array([zc - dz, zc + dz, zc - 1j * dz, zc + 1j * dz])
            # CR check of the continued potential (gauge part via fresh continuation to these points)
            out["runs"].append(dict(R0=R0n, theta=th, levels=ev, richardson=Er, gamma=-2 * Er.imag, Q=Er.real,
                                    rel_diff_gamma_vs_siegert=(-2 * Er.imag) / (-2 * Es.imag) - 1))
            jdump(f"C136_s1_ecs_{name}_{lam}.json", out)
    # Cauchy-Riemann check on the continued gauge part at one interior ray point
    iR = int(np.argmin(np.abs(rt - R0s[0])))
    rho = hf * np.arange(2001)
    z, Gz, _ = C.gauge_on_ray(P[iR], float(rt[iR]), thetas[0], rho, cp)
    j = 1500
    dGdz_ray = (Gz[j + 1] - Gz[j - 1]) / (z[j + 1] - z[j - 1])
    z2_, G2_, _ = C.gauge_on_ray(P[iR], float(rt[iR]), 0.0, np.array([0.0]), cp)
    out["holomorphy"] = dict(dGdz_on_ray=dGdz_ray, note="compare with real-axis derivative continued; see D108")
    jdump(f"C136_s1_ecs_{name}_{lam}.json", out)
    return out


def ph_ecs():
    allout = {}
    for name, lam in (("M0", -0.0143), ("L0", -0.0143), ("MU", -0.010), ("MU", -0.0143)):
        try:
            allout[f"{name}_{lam}"] = ecs_case(name, lam)
        except Exception as e:      # noqa: BLE001
            allout[f"{name}_{lam}"] = dict(error=repr(e))
        jdump("C136_s1_ecs.json", allout)
    return allout


def ph_uv():
    out = []
    for name, lam in (("MU", -0.0143), ("MU", -0.010), ("M0", -0.0143)):
        r, U, _ = potential(name, lam)
        base = solve_case((r, U, lam, name, 14.0, 0.02))
        r2 = base["wkb"]["tp"][2]
        o = opt_of(name)
        cp = cp_of(o.get("table", "v11"))
        Mt = (C.extra_terms(r, lam, cp, o) - C.extra_terms(r, lam, cp, {"match": "none"})).real
        for fac in (1.5, 2.0):
            ruv = fac * r2
            S = 0.5 * (1 + np.tanh((r - ruv) / 0.5))
            iu = int(np.argmin(np.abs(r - ruv)))
            mods = {"lin": S * lam / 4 * (ruv ** 4 + 4 * ruv ** 3 * (r - ruv) - r ** 4),
                    "steep": S * lam / 4 * r ** 4,
                    "freeze": S * (Mt[iu] - Mt),
                    "smrun": S * BETA_SM / 4 * np.log(np.maximum(r, 1e-9) / ruv) * r ** 4}
            for kind, dU in mods.items():
                ro = min(ruv + 6.0, r[-1] - 1.0)
                x = solve_case((r, U + dU, lam, name, ro, 0.02))
                b2 = solve_case((r, U, lam, name, ro, 0.02))
                out.append(dict(name=name, lam=lam, r2=r2, r_uv=ruv, kind=kind, rout=ro, gamma=x.get("gamma"),
                                gamma_base_same_rout=b2.get("gamma"), Q=x.get("Q"),
                                rel_change=(x.get("gamma", np.nan) / b2.get("gamma", np.nan) - 1)))
                jdump("C136_s1_uv.json", dict(rows=out))
    return out


def ph_sens():
    out = {}
    for name in ("MU", "M0"):
        for lam in (-0.010, -0.0143):
            r, U, _ = potential(name, lam)
            g0 = solve_case((r, U, lam, name, 14.0, 0.02))["gamma"]
            d = {"gamma0": g0}
            for tag in ("g2p", "gYp"):
                rr, UU, _ = potential(name, lam, tag=tag)
                d[tag] = solve_case((rr, UU, lam, name, 14.0, 0.02))["gamma"]
            cpy = cp_of("v11")
            cpy["yt"] *= 1.01
            rr, UU, _ = potential(name, lam, cp=cpy)
            d["ytp"] = solve_case((rr, UU, lam, name, 14.0, 0.02))["gamma"]
            rr, UU, _ = potential(name, lam, extra={"mc2": 1e-3})
            d["mc2_1e-3"] = solve_case((rr, UU, lam, name, 14.0, 0.02))["gamma"]
            d["s_lng_lng2"] = math.log(d["g2p"] / g0) / math.log(1.01)
            d["s_lng_lngY"] = math.log(d["gYp"] / g0) / math.log(1.01)
            d["s_lng_lnyt"] = math.log(d["ytp"] / g0) / math.log(1.01)
            d["s_lng_mc2"] = math.log(d["mc2_1e-3"] / g0) / 1e-3
            out[f"{name}_{lam}"] = d
            jdump("C136_s1_sens.json", out)
    return out


def ph_fill():
    """Re-solve failed grid rows with the entire residual H = u' - w u (rerun only failed cases) + 2 controls."""
    d = json.loads((RUN / "C136_s1_grid.json").read_text(encoding="utf-8"))
    res = d["rows"]
    tasks = [(None, None, x["lam"], name, 14.0, 0.02) for name, rows in res.items() for x in rows if x.get("error")]
    tasks += [(None, None, -0.0143, "M0", 14.0, 0.02), (None, None, -0.010, "MU", 14.0, 0.02)]
    full = []
    for _, _, lam, name, ro, r0 in tasks:
        r, U, _ = potential(name, lam)
        full.append((r, U, lam, name, ro, r0))
    with Pool(4) as pool:
        new = pool.map(solve_case, full)
    controls = []
    for x in new:
        rows = res[x["label"]]
        for i, y in enumerate(rows):
            if y["lam"] == x["lam"]:
                if y.get("error"):
                    rows[i] = x
                else:
                    controls.append(dict(label=x["label"], lam=x["lam"], old=y.get("gamma"), new=x.get("gamma"),
                                         oldQ=y.get("Q"), newQ=x.get("Q")))
    crit, table = {}, {}
    for name, rows in res.items():
        rows.sort(key=lambda x: -x["lam"])
        crit[name] = lam_crit(rows)
        table[name] = {f"{x['lam']}": [x.get("Q"), x.get("gamma"), (x.get("wkb") or (x.get("wkb_guess") or {})).get("gamma"),
                                       x.get("resolved"), x.get("error")] for x in rows}
    d.update(rows=res, gamma_table=table, lam_crit=crit, fill_controls=controls,
             fill_note="failed rows re-solved with residual H=u'-w u; controls compare old (F) and new (H) residuals")
    jdump("C136_s1_grid.json", d)
    status({"completed": ["stage1e grid fill"], "key_numbers": {"lam_crit": crit}})
    return d


def ecs_single(name, lam, R0, th, s0_target, hf, rho_max=5.0):
    o = opt_of(name)
    tag = o.get("table", "v11")
    cp = cp_of(tag)
    r, U, _ = potential(name, lam)
    rad = C.Radial(r, U)
    rt, G, P = get_table(tag)
    iR = int(np.argmin(np.abs(rt - R0)))
    R0n = float(rt[iR])
    nL = int(round((R0n - s0_target) / (4 * hf))) * 4
    s0 = R0n - nL * hf
    nR = int(round(rho_max / hf / 4)) * 4
    rho = hf * np.arange(nR + 1)
    z, Gz, _ = C.gauge_on_ray(P[iR], R0n, th, rho, cp)
    Uz = Gz + C.extra_terms(z, lam, cp, o)
    sL = s0 + hf * np.arange(nL + 1)
    zz = np.concatenate([sL.astype(complex), z[1:]])
    UU = np.concatenate([rad.Uf(sL).astype(complex), Uz[1:]])
    E0 = rad.pole(rad.guess()[0], r0=0.02, rout=14.0)[0]
    ev = [C.ecs_eigs(UU[::m], zz[::m], E0, k=4)[0] for m in (4, 2, 1)]
    Edir = rad.pole(E0, r0=s0, rout=14.0, start="dir")[0]
    return dict(s0=s0, levels=ev, ecs=C.richardson3(*ev), siegert_dirichlet_same_s0=Edir, siegert_frobenius=E0)


def ph_ecsfix():
    out = {"same_inner_wall": [], "s0_extrapolation": None}
    for name, lam in (("M0", -0.0143), ("L0", -0.0143), ("MU", -0.010), ("MU", -0.0143)):
        x = ecs_single(name, lam, 6.5, 0.3, 0.02, 0.001)
        x.update(name=name, lam=lam, dE_ecs_minus_siegertdir=x["ecs"] - x["siegert_dirichlet_same_s0"],
                 rel_dgamma=(x["ecs"].imag / x["siegert_dirichlet_same_s0"].imag - 1))
        out["same_inner_wall"].append(x)
        jdump("C136_s1_ecsfix.json", out)
    ex = []
    for s0 in (0.02, 0.01, 0.005):
        ex.append(ecs_single("M0", -0.0143, 6.5, 0.3, s0, 0.0005))
        jdump("C136_s1_ecsfix.json", dict(out, s0_extrapolation=ex))
    p = 2 * (0.5 + math.sqrt(0.25 + 2 * 0.5625)) - 1
    f = 2 ** p
    Eex = ex[2]["ecs"] + (ex[2]["ecs"] - ex[1]["ecs"]) / (f - 1)
    out["s0_extrapolation"] = dict(rows=ex, power=p, extrapolated=Eex, siegert_frobenius=ex[0]["siegert_frobenius"],
                                   rel_dgamma=Eex.imag / ex[0]["siegert_frobenius"].imag - 1,
                                   dQ=Eex.real - ex[0]["siegert_frobenius"].real)
    jdump("C136_s1_ecsfix.json", out)
    return out


def main():
    global BETA_SM
    RUN.mkdir(parents=True, exist_ok=True)
    add = {"launched": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv,
           "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(C.CODE.glob("C136_*.py"))}}
    with open(RUN / "contract_addendum.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(add) + "\n")
    try:
        yb = R.run(R.inputs_at_Mt(R.PDG["Mt"], R.PDG["als"], R.PDG["Mh"], R.PDG["MW"]), R.PDG["Mt"], [R.MU_B], 2)[0]
        BETA_SM = float(R.beta(0, yb)[4])
    except Exception:            # noqa: BLE001
        pass
    phases = sys.argv[1:] or ["checks", "tables", "repro", "grid", "ecs", "uv", "sens"]
    for ph in phases:
        t0 = time.time()
        globals()["ph_" + ph]()
        print(f"phase {ph} done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
