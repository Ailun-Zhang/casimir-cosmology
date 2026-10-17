# PROVENANCE: 04_calculations/code/20261013_argon_codex_intake/cmb_ensemble/postcheck_ens.py (copied 2026-10-15 by code-shape-cmb)
# postcheck_ens.py (C145/D117) -- WORKING COPY: read-only post-processing of run01 saved tables
# (results/S2_profiles.json, results/S4_noise.json); no covariance or KL recomputation.
# Run: python -I postcheck_ens.py  -> run01/results/S3_postcheck.json
"""[Package note] Read-only post-processing of the saved run01 tables (pool-min profile, convexity edge bounds) -> S3_postcheck.json: P_joint peak 0.344 at L0 = 1.075 d_LSS, P_joint >= 0.1 for [1.018, 1.183], integrated readable measure 0.0422. Paper: PRD Sec. X (CMB-02..05), PRD figure of P(read|L0).

Read-only post-processing of saved run01 tables (no covariance / KL recomputation).
Replaces the 3-point parabolic profile refinement (can undershoot for strongly asymmetric volume
profiles) by the minimum over the actual cube pool (Brent: continuous minimum within 0.04-0.55% below it),
and gives per-shape edge bounds from convexity of ln D(u): chord = upper bound, extension of the
previous (steeper) segment = lower bound.  Writes run01/results/S3_postcheck.json."""
import json, math
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import _cmb_paths as WP  # noqa: E401,E402 (package)
R = WP.wdir('ai/cmb_ensemble/run01')
T = 4.5; CIRC = 0.985; SIG = 0.1041; SHALLOW = 60.0
pr = json.loads((R / 'results/S2_profiles.json').read_text(encoding='utf-8'))
noise = json.loads((R / 'results/S4_noise.json').read_text(encoding='utf-8'))
TH = {'obl': -math.pi / 6, 'tri': 0.0, 'pro': math.pi / 6}
XD = np.array([1., -1., 0.]) / math.sqrt(2); XD2 = np.array([1., 1., -2.]) / math.sqrt(6)
fs = lambda A, th: math.exp(min(A * SIG * (math.cos(th) * XD + math.sin(th) * XD2)))
flags = []; vals = {}
for r in pr.values():
    cur = r['TE80']['curve']; ys = [c[1] for c in cur]; i = int(np.argmin(ys)); dg = ys[i]
    if r['TE80']['D_prof'] < 0.97 * dg or r['TE80']['D_prof'] < r['TE40']['D_prof']:
        flags.append(dict(A=r['A'], lab=r['lab'], s=r['s'], u=round(r['u'], 4), D_parab=round(r['TE80']['D_prof'], 4),
                          D_grid=round(dg, 4), s_grid=cur[i][0], D40_prof=round(r['TE40']['D_prof'], 4),
                          nbrs=[[round(c[0], 4), round(c[1], 3)] for c in cur[max(0, i - 2):i + 3]]))
    add = lambda k, d: vals.setdefault(k, []).append((r['A'], r['lab'], r['u'], d))
    for key in ('TE80', 'TE40', 'TE30', 'T80'):
        add(key + '_grid', min(c[1] for c in r[key]['curve']))
    add('TE80_parab', r['TE80']['D_prof'])
    if r['TE80']['D_fixed']:
        add('TE80_fixed', r['TE80']['D_fixed'])
    rows = noise.get(r['name'])
    if rows:
        add('noise_TE80_grid', min(x['planck143_TE80'] for x in rows))
        add('noise_TE30_grid', min(x['planck143_TE30'] for x in rows))


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


rng = np.random.default_rng(20261013); n = 2_000_000
be = rng.normal(size=(n, 2)) * SIG
mag = np.hypot(be[:, 0], be[:, 1]); thb = np.arctan2(be[:, 1], be[:, 0])
f = np.exp(np.min(math.sqrt(2 / 3) * mag[:, None] * np.cos(thb[:, None] - np.deg2rad([30, 150, 270])), axis=1))
Am = mag / SIG; xi = np.sin(3 * thb); slo = CIRC / f
L0 = np.round(np.arange(0.98, 1.4001, 0.0025), 5)
Pnc = np.searchsorted(np.sort(slo), L0, side='right') / n
pick = [0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.25, 1.3]
idx = [int(np.argmin(abs(L0 - p))) for p in pick]


def curves(tab):
    sp = span_eval(tab, Am, xi); ok = sp > 0; shi = slo * np.exp(sp)
    Pj = (np.searchsorted(np.sort(slo[ok]), L0, side='right') - np.searchsorted(np.sort(shi[ok]), L0, side='left')) / n
    Pc = np.where(Pnc > 0, Pj / np.maximum(Pnc, 1e-300), 0.0)
    return Pj, Pc, float(np.mean(np.maximum(sp, 0)))


def wins(P):
    o = {}
    for p in (0.5, 0.3, 0.2, 0.1, 0.05):
        k = np.flatnonzero(P >= p)
        o[str(p)] = None if not len(k) else [float(L0[k[0]]), float(L0[k[-1]]), bool(np.all(np.diff(k) == 1))]
    return o


flags = sorted(flags, key=lambda f: (f['lab'], f['A'], f['s'], f['u']))   # R18: deterministic order (the run tables are filled in completion order)
out = dict(note=__doc__, flags=flags, L0_pick=pick, P_no_circles=[round(float(Pnc[i]), 4) for i in idx], variants={})
print('FLAGGED (parab undershoot):', json.dumps(flags))
print('L0  ', ' '.join('%6.3f' % p for p in pick)); print('Pnc ', ' '.join('%6.3f' % Pnc[i] for i in idx))
for vn, pts in vals.items():
    sh = {}
    for A, lab, u, d in pts:
        sh.setdefault((round(A, 6), lab), []).append((u, d))
    E = {k: edge_b(v) for k, v in sh.items()}
    res = {}
    for mode in ('lo', 'mid', 'hi', 'chord'):
        tab = {}
        for lab in ('obl', 'tri', 'pro'):
            ks = sorted([k for k in E if k[1] == lab and E[k]], key=lambda k: k[0])
            uu = [(E[k]['lo'] + E[k]['hi']) / 2 if mode == 'mid' else E[k][mode] for k in ks]
            tab[lab] = ([k[0] for k in ks], [math.log(u / CIRC) for u in uu])
        Pj, Pc, ms = curves(tab)
        rep = E.get((round(math.sqrt(2), 6), 'tri'))
        repw = None
        if rep:
            ff = fs(math.sqrt(2), 0.0); ur = (rep['lo'] + rep['hi']) / 2 if mode == 'mid' else rep[mode]
            repw = [CIRC / ff, ur / ff]
        res[mode] = dict(Pj_max=float(Pj.max()), L0_Pj_max=float(L0[int(Pj.argmax())]), mean_span=ms, rep_window=repw,
                         rep_span=math.log(repw[1] / repw[0]) if repw else None, Pj=[round(float(Pj[i]), 4) for i in idx],
                         Pc=[round(float(Pc[i]), 4) for i in idx], win_joint=wins(Pj), win_cond=wins(Pc))
    out['variants'][vn] = dict(edges={'%s|%.3f' % (k[1], k[0]): {q: (round(x, 5) if isinstance(x, float) else x) for q, x in e.items()}
                                      for k, e in sorted(E.items(), key=lambda kv: (kv[0][1], kv[0][0])) if e}, modes=res)
    m = res['mid']
    print('== %s mid: Pjmax=%.4f@%.4f meanspan=%.5f rep=%s span=%.5f | lo Pjmax=%.4f | hi Pjmax=%.4f | chord Pjmax=%.4f' % (
        vn, m['Pj_max'], m['L0_Pj_max'], m['mean_span'], [round(x, 5) for x in m['rep_window']], m['rep_span'],
        res['lo']['Pj_max'], res['hi']['Pj_max'], res['chord']['Pj_max']))
    print('   edges u[lo,hi]:', ' '.join('%s%.2f[%.4f,%.4f]%s' % (k[1][0], k[0], e['lo'], e['hi'], e['flag'][0]) for k, e in sorted(E.items(), key=lambda kv: (kv[0][1], kv[0][0])) if e))
    for mode in ('mid', 'lo', 'hi'):
        print('   %-3s Pj ' % mode + ' '.join('%6.3f' % x for x in res[mode]['Pj']))
        print('   %-3s Pc ' % mode + ' '.join('%6.3f' % x for x in res[mode]['Pc']))
    for mode in ('mid', 'lo', 'hi'):
        print('   %-3s winJ ' % mode + '; '.join('%s:%s' % (p, ('[%.4f,%.4f]' % (w[0], w[1])) if w else '-') for p, w in res[mode]['win_joint'].items())
              + ' | winC ' + '; '.join('%s:%s' % (p, ('[%.4f,%.4f]' % (w[0], w[1])) if w else '-') for p, w in res[mode]['win_cond'].items()))
(R / 'results/S3_postcheck.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print('written', R / 'results/S3_postcheck.json')
