# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_ensemble.py (copied 2026-10-15 by code-shape-cmb)
# PROVENANCE: edge/ensemble functions copied from 04_calculations/code/20261013_argon_codex_intake/cmb_ensemble/postcheck_ens.py (sha256 04dd92047de876a8bc7cafed86b20f6cc1eed3c6945d9e5698d14fd35dea520d) on 2026-10-14 by shape-hardening; sigma_E made a parameter.
"""[Package note] D117 window re-integrated for other sigma_E (0.0846...0.1145): next order sigma_E = 0.094 gives the P_joint peak 0.377 at 1.065, P_joint >= 0.1 for [1.015, 1.168], integrated measure 0.0421. Paper: PRD Sec. X (CMB-14..17).

C153_ensemble (D125 part B): D117 read window re-integrated for other sigma_E, read-only on the saved run01 tables
(S2_profiles.json, S4_noise.json); no covariance or KL recomputation.
The saved fossils were built at physical amplitude a = A * 0.1041 (SIG_TABLE).  For a new ensemble width sigma the
physical amplitude is drawn from Rayleigh(sigma) and mapped to the table coordinate A_tab = a / SIG_TABLE; the shortest
-side factor f uses the physical amplitude.  Edges: cube-pool minimum (D117 main), convexity bounds lo/hi, mid = mean.
    python -I C153_ensemble.py   ->  run01/ensemble/C153_ensemble.json
"""
import json, math, sys, hashlib
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package: chain root (cmb/)
import _cmb_paths as WP  # noqa: E402
SRC = WP.wdir('ai/cmb_ensemble/run01')
OUT = WP.wdir('r3/shape_hardening/run01/ensemble')
T = 4.5; CIRC = 0.985; SIG_TABLE = 0.1041; SHALLOW = 60.0
SIGMAS = [0.1041, 0.0846, 0.094, 0.1034, 0.0937, 0.1145]
pr = json.loads((SRC / 'results/S2_profiles.json').read_text(encoding='utf-8'))
vals = {}
for r in pr.values():
    add = lambda k, d: vals.setdefault(k, []).append((r['A'], r['lab'], r['u'], d))
    for key in ('TE80', 'TE30'):
        add(key + '_grid', min(c[1] for c in r[key]['curve']))
    if r['TE80']['D_fixed']:
        add('TE80_fixed', r['TE80']['D_fixed'])


def edge_b(pts):
    pts = sorted(pts); us = [p[0] for p in pts]; l = [math.log(p[1]) for p in pts]; ab = [p[1] >= T for p in pts]
    lt = math.log(T)
    if len(pts) < 2:
        return None
    if all(ab):
        sl = (l[-1] - l[-2]) / (us[-1] - us[-2])
        c = us[-1] + (lt - l[-1]) / sl if sl < 0 else us[-1]
        return dict(lo=c, hi=us[-1] + (l[-1] - lt) / SHALLOW, chord=c, flag='extrap_up')
    if not any(ab):
        sl = (l[1] - l[0]) / (us[1] - us[0])
        c = us[0] + (lt - l[0]) / sl
        return dict(lo=c, hi=us[0], chord=c, flag='extrap_down')
    i = next((k for k in range(1, len(pts)) if ab[k - 1] and not ab[k]), None)
    if i is None:
        return None
    ua, ub, la, lb = us[i - 1], us[i], l[i - 1], l[i]
    hi = ua + (la - lt) * (ub - ua) / (la - lb); lo = hi
    if i >= 2:
        sp = (l[i - 1] - l[i - 2]) / (us[i - 1] - us[i - 2])
        if sp < 0:
            lo = min(hi, ua + (lt - la) / sp)
    return dict(lo=lo, hi=hi, chord=hi, flag='bracket')


def span_eval(tab, A, xi):
    v = []
    for lab in ('obl', 'tri', 'pro'):
        a = np.array(tab[lab][0]); sp = np.array(tab[lab][1]); o = np.argsort(a); a, sp = a[o], sp[o]
        x = np.interp(A, a, sp); sl = (sp[1] - sp[0]) / (a[1] - a[0])
        v.append(np.where(A < a[0], sp[0] + sl * (A - a[0]), x))
    sm, s0, sp_ = v
    return s0 + 0.5 * (sp_ - sm) * xi + (0.5 * (sp_ + sm) - s0) * xi ** 2


L0 = np.round(np.arange(0.98, 1.4001, 0.0025), 5)
pick = [1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.25, 1.3]
idx = [int(np.argmin(abs(L0 - p))) for p in pick]


def wins(P):
    o = {}
    for p in (0.5, 0.3, 0.2, 0.1, 0.05):
        k = np.flatnonzero(P >= p)
        o[str(p)] = None if not len(k) else [float(L0[k[0]]), float(L0[k[-1]]), bool(np.all(np.diff(k) == 1))]
    return o


def ensemble(sig):
    rng = np.random.default_rng(20261013); n = 2_000_000
    be = rng.normal(size=(n, 2)) * sig
    mag = np.hypot(be[:, 0], be[:, 1]); thb = np.arctan2(be[:, 1], be[:, 0])
    f = np.exp(np.min(math.sqrt(2 / 3) * mag[:, None] * np.cos(thb[:, None] - np.deg2rad([30, 150, 270])), axis=1))
    Am = mag / SIG_TABLE; xi = np.sin(3 * thb); slo = CIRC / f
    Pnc = np.searchsorted(np.sort(slo), L0, side='right') / n
    return Am, xi, slo, Pnc, n


out = {"code": Path(__file__).name, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "sig_table": SIG_TABLE, "L0_pick": pick, "results": {}}
for sig in SIGMAS:
    Am, xi, slo, Pnc, n = ensemble(sig)
    res = {"P_no_circles": [round(float(Pnc[i]), 4) for i in idx],
           "frac_A_tab_gt2(const extrapolation)": float(np.mean(Am > 2)), "frac_A_tab_lt0.25(linear extrapolation)": float(np.mean(Am < 0.25)),
           "variants": {}}
    for vn in ('TE80_grid', 'TE30_grid', 'TE80_fixed'):
        sh = {}
        for A, lab, u, d in vals[vn]:
            sh.setdefault((round(A, 6), lab), []).append((u, d))
        E = {k: edge_b(v) for k, v in sh.items()}
        vr = {}
        for mode in ('mid', 'lo', 'hi'):
            tab = {}
            for lab in ('obl', 'tri', 'pro'):
                ks = sorted([k for k in E if k[1] == lab and E[k]], key=lambda k: k[0])
                uu = [(E[k]['lo'] + E[k]['hi']) / 2 if mode == 'mid' else E[k][mode] for k in ks]
                tab[lab] = ([k[0] for k in ks], [math.log(u / CIRC) for u in uu])
            sp = span_eval(tab, Am, xi); ok = sp > 0; shi = slo * np.exp(sp)
            Pj = (np.searchsorted(np.sort(slo[ok]), L0, side='right') - np.searchsorted(np.sort(shi[ok]), L0, side='left')) / n
            Pc = np.where(Pnc > 0, Pj / np.maximum(Pnc, 1e-300), 0.0)
            Arep = math.sqrt(2) * sig / SIG_TABLE
            sp_rep = float(span_eval(tab, np.array([Arep]), np.array([0.0]))[0])
            frep = math.exp(min(Arep * SIG_TABLE * (math.cos(0.0) * np.array([1., -1., 0.]) / math.sqrt(2)
                                                    + math.sin(0.0) * np.array([1., 1., -2.]) / math.sqrt(6))))
            vr[mode] = dict(Pj_max=float(Pj.max()), L0_Pj_max=float(L0[int(Pj.argmax())]), integrated_measure=float(np.mean(np.maximum(sp, 0))),
                            Pj=[round(float(Pj[i]), 4) for i in idx], Pc=[round(float(Pc[i]), 4) for i in idx],
                            win_joint=wins(Pj), win_cond=wins(Pc),
                            rep_point_window=[CIRC / frep, CIRC / frep * math.exp(sp_rep)] if sp_rep > 0 else None,
                            rep_point_span=sp_rep)
        res["variants"][vn] = vr
    out["results"]["%.4f" % sig] = res
    m = res["variants"]["TE80_grid"]
    print("sigma=%.4f  TE80prof mid: Pjmax=%.4f@%.4f [lo %.4f, hi %.4f]  measure=%.5f [%.5f,%.5f]  P>=0.1 %s  P>=0.3 %s  Pc>=0.5 %s  rep %s  fracA>2=%.3f" % (
        sig, m['mid']['Pj_max'], m['mid']['L0_Pj_max'], m['lo']['Pj_max'], m['hi']['Pj_max'], m['mid']['integrated_measure'],
        m['lo']['integrated_measure'], m['hi']['integrated_measure'], m['mid']['win_joint']['0.1'], m['mid']['win_joint']['0.3'],
        m['mid']['win_cond']['0.5'], [round(x, 4) for x in m['mid']['rep_point_window']] if m['mid']['rep_point_window'] else None,
        res["frac_A_tab_gt2(const extrapolation)"]))
    print("    Pj  " + " ".join("%.3f" % x for x in m['mid']['Pj']))
    print("    Pc  " + " ".join("%.3f" % x for x in m['mid']['Pc']))
    t3 = res["variants"]["TE30_grid"]["mid"]
    print("    TE30prof: Pjmax=%.4f@%.4f measure=%.5f P>=0.1 %s" % (t3['Pj_max'], t3['L0_Pj_max'], t3['integrated_measure'], t3['win_joint']['0.1']))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'C153_ensemble.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print('written', OUT / 'C153_ensemble.json')
