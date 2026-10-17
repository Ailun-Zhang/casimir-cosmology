# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/cleanup_t12/make_fossil_table_v2.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] fossil_table_v2.csv: per-shape D (fixed / pool-min profiled) for TE30/40/80 and T80, shortest side u (note: column xi = sin 3 theta, column s = L0/d_LSS). Paper: PRD Sec. X (read-edge mechanism figure).
"""
# make_fossil_table_v2.py -- T12 / C158 (W14), agent cleanup-t12, 2026-10-14
# Read-only rebuild of cmb_ensemble/run01/results/fossil_table.csv. The prof/sprof columns are replaced by the
# minimum over the 19 actual reference cubes s' in [0.985, 1.22] ("pool-min" = D117 main result = S3_postcheck
# *_grid variants), instead of the deprecated 3-point parabolic refinement. No covariance or KL recomputation.
# Old files are not modified. The ensemble helpers edge_b/span_eval/curves are verbatim copies of
# 04_calculations/code/20261013_argon_codex_intake/cmb_ensemble/postcheck_ens.py (used only as a consistency check).
# Run: python -I make_fossil_table_v2.py
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, csv, math, hashlib, datetime
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import sys  # package: work-tree path shim of the cmb chain (see cmb/_cmb_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cmb_paths as WP  # noqa: E402
SRC = WP.wdir('ai/cmb_ensemble/run01/results')
OUT = WP.wdir('r3/cleanup_t12/run01')
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ('TE30', 'TE40', 'TE80', 'T80')
T = 4.5; CIRC = 0.985; SIG = 0.1041; SHALLOW = 60.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


INP = {n: SRC / n for n in ('fossil_table.csv', 'S2_profiles.json', 'kl_table.json', 'S2_brent.json', 'S3_postcheck.json')}
pr = json.loads(INP['S2_profiles.json'].read_text(encoding='utf-8'))
kl = json.loads(INP['kl_table.json'].read_text(encoding='utf-8'))
br = json.loads(INP['S2_brent.json'].read_text(encoding='utf-8'))
pc = json.loads(INP['S3_postcheck.json'].read_text(encoding='utf-8'))
with open(INP['fossil_table.csv'], newline='', encoding='utf-8-sig') as fh:
    rd = csv.DictReader(fh)
    old_cols = list(rd.fieldnames)
    old = list(rd)

ck = dict(n_rows=len(old), n_S2=len(pr), n_kl=len(kl), geom_max_absdiff=0.0, fixed_max_reldiff_vs_S2=0.0,
          parab_max_reldiff_vs_S2=0.0, curve_vs_kl_max_reldiff=0.0, kl_vs_curve_set_mismatch=0, n_kl_cubes_below_circ=0,
          curve_npts=set(), prof_kind={}, n_fixed_empty=0, poolmin_S2_vs_kl_max_reldiff=0.0)
v2rows, diff, newval, oldval = [], [], {}, {}
for r in old:
    nm = r['name']; P = pr[nm]
    for c in ('A', 'theta', 'xi', 's', 'u', 's_low'):
        ck['geom_max_absdiff'] = max(ck['geom_max_absdiff'], abs(float(r[c]) - float(P[c])))
    assert r['lab'] == P['lab'], nm
    row = {c: r[c] for c in old_cols}
    klr = {float(s): v for s, v in kl[nm].items()}
    ck['n_kl_cubes_below_circ'] += sum(1 for s in klr if s < CIRC - 1e-9)
    newval[nm], oldval[nm] = {}, {}
    for k in KEYS:
        q = P[k]
        cur = [(float(a), float(b)) for a, b in q['curve']]
        ck['curve_npts'].add(len(cur)); ck['prof_kind'][str(q.get('prof_kind'))] = ck['prof_kind'].get(str(q.get('prof_kind')), 0) + 1
        for s_, d_ in cur:
            sk = min(klr, key=lambda x: abs(x - s_))
            assert abs(sk - s_) < 1e-9, (nm, k, s_)
            ck['curve_vs_kl_max_reldiff'] = max(ck['curve_vs_kl_max_reldiff'], abs(klr[sk][k][0] - d_) / abs(d_))
        poolk = {s_: v[k][0] for s_, v in klr.items() if s_ >= CIRC - 1e-9 and k in v}
        if sorted(round(s, 6) for s in poolk) != sorted(round(s, 6) for s, _ in cur):
            ck['kl_vs_curve_set_mismatch'] += 1
        i = int(np.argmin([d for _, d in cur])); smin, dmin = cur[i]
        skl = min(poolk, key=poolk.get)
        assert abs(skl - smin) < 1e-9, (nm, k)
        ck['poolmin_S2_vs_kl_max_reldiff'] = max(ck['poolmin_S2_vs_kl_max_reldiff'], abs(poolk[skl] - dmin) / dmin)
        fx = r[k + '_fixed']
        if fx.strip() == '' or not q.get('D_fixed'):
            ck['n_fixed_empty'] += 1
        else:
            ck['fixed_max_reldiff_vs_S2'] = max(ck['fixed_max_reldiff_vs_S2'], abs(float(fx) - q['D_fixed']) / q['D_fixed'])
        pa = float(r[k + '_prof']); spa = float(r[k + '_sprof'])
        ck['parab_max_reldiff_vs_S2'] = max(ck['parab_max_reldiff_vs_S2'], abs(pa - q['D_prof']) / abs(q['D_prof']))
        row[k + '_prof'] = '%.6g' % dmin; row[k + '_sprof'] = '%.5f' % smin
        newval[nm][k] = dmin; oldval[nm][k] = pa
        rel = (pa - dmin) / dmin
        diff.append(dict(name=nm, A=r['A'], lab=r['lab'], s=r['s'], u=r['u'], readout=k, D_fixed=fx,
                         D_prof_parab_old=r[k + '_prof'], s_prof_parab_old=r[k + '_sprof'],
                         D_poolmin_new=row[k + '_prof'], s_poolmin_new=row[k + '_sprof'],
                         rel_old_minus_new='%.6f' % rel, ds_old_minus_new='%.5f' % (spa - smin),
                         low_gt_1pct=int(rel < -0.01), low_ge_3pct=int(rel <= -0.03),
                         poolmin_at_boundary_0p985=int(abs(smin - CIRC) < 1e-9)))
    v2rows.append(row)
ck['curve_npts'] = sorted(ck['curve_npts'])

# Brent minima (S2_brent.json): identify the readout by matching the pool evaluations to the S2 curves
brent = {}
for b in br:
    nm = b['fossil']; P = pr[nm]
    pev = [(e['s_prime'], e['D']) for e in b['evals'] if e['source'] == 'pool']
    match = [k for k in KEYS if all(any(abs(s_ - cs) < 1e-9 and abs(d_ - cd) <= 1e-9 * abs(cd) for cs, cd in P[k]['curve'])
                                    for s_, d_ in pev)]
    k = match[0]; pm = min(c[1] for c in P[k]['curve'])
    brent[nm] = dict(readout=k, matched_readouts=match, s_star=b['s_star'], D_star=b['D_star'], poolmin=pm,
                     rel_below_poolmin=(pm - b['D_star']) / pm, ok=b.get('ok'), interior=b.get('interior'), in_csv=nm in newval)
bkeys = sorted({v['readout'] for v in brent.values()})
v2cols = old_cols + [c for k in bkeys for c in (k + '_brent', k + '_sbrent')] + ['prof_source']
for row in v2rows:
    for k in bkeys:
        b = brent.get(row['name'])
        row[k + '_brent'] = ('%.6g' % b['D_star']) if b and b['readout'] == k else ''
        row[k + '_sbrent'] = ('%.7f' % b['s_star']) if b and b['readout'] == k else ''
    row['prof_source'] = 'poolmin19'


def dpi(v):
    o = []
    if not v['TE30'] <= v['TE40'] * (1 + 1e-12): o.append('TE30>TE40')
    if not v['TE40'] <= v['TE80'] * (1 + 1e-12): o.append('TE40>TE80')
    if not v['T80'] <= v['TE80'] * (1 + 1e-12): o.append('T80>TE80')
    return o


dpi_old = {nm: dpi(v) for nm, v in oldval.items() if dpi(v)}
dpi_new = {nm: dpi(v) for nm, v in newval.items() if dpi(v)}
fixval = {r['name']: {k: float(r[k + '_fixed']) for k in KEYS} for r in old if all(r[k + '_fixed'].strip() for k in KEYS)}
dpi_fixed = {nm: dpi(v) for nm, v in fixval.items() if dpi(v)}
rels = [(float(d['rel_old_minus_new']), d) for d in diff]
per = {}
for k in KEYS:
    rk = [(x, d) for x, d in rels if d['readout'] == k]; x, d = min(rk, key=lambda t: t[0])
    per[k] = dict(min_rel=round(x, 4), at='(%s,%s,%s)' % (d['A'], d['lab'], d['s']),
                  n_low_gt_1pct=sum(1 for x, _ in rk if x < -0.01), n_low_ge_3pct=sum(1 for x, _ in rk if x <= -0.03),
                  n_cells_prof_changed=sum(1 for _, d in rk if d['D_prof_parab_old'] != d['D_poolmin_new']),
                  n_cells_sprof_changed=sum(1 for _, d in rk if d['s_prof_parab_old'] != d['s_poolmin_new']))
# S3_postcheck flag criterion (D_parab < 0.97 D_grid or D_parab < TE40_parab), compared with its flag list
pc_flags = sorted((round(f['A'], 4), f['lab'], round(f['s'], 4)) for f in pc['flags'])
my_flags = sorted((round(float(P['A']), 4), P['lab'], round(float(P['s']), 4)) for nm, P in pr.items() if nm in newval and
                  (oldval[nm]['TE80'] < 0.97 * newval[nm]['TE80'] or oldval[nm]['TE80'] < oldval[nm]['TE40']))
summary = dict(n_pairs=len(rels), n_pairs_low_gt_1pct=sum(1 for x, _ in rels if x < -0.01),
               n_pairs_old_above_new=sum(1 for x, _ in rels if x > 1e-9), per_readout=per,
               worst=dict(rel=round(min(rels, key=lambda t: t[0])[0], 4), row=min(rels, key=lambda t: t[0])[1]),
               dpi_violations_old_prof=dpi_old, dpi_violations_new_prof=dpi_new, dpi_violations_fixed=dpi_fixed,
               boundary_minima=[(d['A'], d['lab'], d['s'], d['readout'], d['D_poolmin_new']) for d in diff if d['poolmin_at_boundary_0p985']],
               max_abs_dsprof=max(abs(float(d['ds_old_minus_new'])) for d in diff),
               postcheck_flags_match=(pc_flags == my_flags), n_flags=len(my_flags), flags=my_flags)

with open(OUT / 'fossil_table_v2.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=v2cols, lineterminator='\n'); w.writeheader(); w.writerows(v2rows)
with open(OUT / 'fossil_table_v2_diff.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(diff[0].keys()), lineterminator='\n'); w.writeheader(); w.writerows(diff)


# ---- consistency check: reproduce S3_postcheck 'mid' mode from the written csv (verbatim helpers) ----
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


def mid_from_rows(rows, col):
    sh = {}
    for r in rows:
        if r[col].strip() == '':
            continue
        P = pr[r['name']]
        sh.setdefault((round(P['A'], 6), P['lab']), []).append((P['u'], float(r[col])))
    E = {k: edge_b(v) for k, v in sh.items()}
    tab = {}
    for lab in ('obl', 'tri', 'pro'):
        ks = sorted([k for k in E if k[1] == lab and E[k]], key=lambda k: k[0])
        tab[lab] = ([k[0] for k in ks], [math.log(((E[k]['lo'] + E[k]['hi']) / 2) / CIRC) for k in ks])
    Pj, Pc, ms = curves(tab)
    return dict(Pj_max=float(Pj.max()), L0_Pj_max=float(L0[int(Pj.argmax())]), mean_span=ms,
                Pj=[round(float(Pj[i]), 4) for i in idx], Pc=[round(float(Pc[i]), 4) for i in idx])


with open(OUT / 'fossil_table_v2.csv', newline='', encoding='utf-8') as fh:
    v2back = list(csv.DictReader(fh))
rep = {}
for src_rows, col, var in [(v2back, 'TE80_prof', 'TE80_grid'), (v2back, 'TE40_prof', 'TE40_grid'), (v2back, 'TE30_prof', 'TE30_grid'),
                           (v2back, 'T80_prof', 'T80_grid'), (v2back, 'TE80_fixed', 'TE80_fixed'), (old, 'TE80_prof', 'TE80_parab')]:
    m = mid_from_rows(src_rows, col); ref = pc['variants'][var]['modes']['mid']
    rep[('v2:' if src_rows is v2back else 'old:') + col + ' vs S3_postcheck.' + var] = dict(
        Pj_max=round(m['Pj_max'], 5), ref_Pj_max=round(ref['Pj_max'], 5), L0=m['L0_Pj_max'], ref_L0=ref['L0_Pj_max'],
        mean_span=round(m['mean_span'], 6), ref_mean_span=round(ref['mean_span'], 6),
        max_abs_dPj=max(abs(a - b) for a, b in zip(m['Pj'], ref['Pj'])), max_abs_dPc=max(abs(a - b) for a, b in zip(m['Pc'], ref['Pc'])))

outs = ['fossil_table_v2.csv', 'fossil_table_v2_diff.csv']
res = dict(task='T12/C158 (W14) fossil_table_v2', agent='cleanup-t12',
           generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
           inputs={n: dict(path=str(p), sha256=sha(p)) for n, p in INP.items()},
           outputs={n: dict(path=str(OUT / n), sha256=sha(OUT / n)) for n in outs},
           columns=dict(old=old_cols, v2=v2cols), integrity=ck, brent=brent, diff_summary=summary, reproduction_mid_mode=rep)
(OUT / 'fossil_table_v2_checks.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
print(json.dumps(dict(script_sha256=res['script_sha256'], outputs=res['outputs'], integrity=ck, brent=brent,
                      diff_summary={k: v for k, v in summary.items() if k != 'worst'}, worst=summary['worst'],
                      reproduction=rep), indent=1, ensure_ascii=False, default=str))
