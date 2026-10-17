# summarize_d114.py -- summary of own D114 runs -> results/D114_own.json (T19, unverified-check)
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'd114_runs')
SGRID = [1.0, 3.0, 5.0, 10.0, 20.0, 40.0]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def load(job):
    p = os.path.join(OUT, job + '.jsonl')
    if not os.path.exists(p):
        return None
    recs = {}
    with open(p, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            recs[r['id']] = r
    return [recs[k] for k in sorted(recs)]


def boot_ratio(a, b, rng, nb=2000):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    r = []
    for _ in range(nb):
        ii = rng.integers(0, n, n)
        r.append(np.sum(a[ii]) / np.sum(b[ii]))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def counts(fates):
    c = {}
    for f in fates:
        c[f] = c.get(f, 0) + 1
    return c


meta = json.load(open(os.path.join(OUT, 'meta.json'), encoding='utf-8'))
rng = np.random.default_rng(5)
S = {'meta_brief': {k: {kk: v[kk] for kk in v if kk in ('kind', 'bg', 'N', 'seed', 'kE', 'kT', 'delta', 'tx', 'pinch', 'vb')}
                    for k, v in meta.items()}}
for job in meta:
    recs = load(job)
    if not recs:
        continue
    mt = meta[job]
    js = {'n_done': len(recs)}
    if mt['kind'] == 'turn':
        for s in SGRID:
            key = '%g' % s
            f = [r['runs'][key]['fate'] for r in recs]
            k = sum(x == 'survive' for x in f)
            js['s=%g' % s] = {'survive': k, 'n': len(f), 'frac': k / len(f), 'wilson95': wilson(k, len(f)), 'fates': counts(f)}
        js['nonmonotone_draws'] = int(sum(bool(r['nonmono']) for r in recs))
        r1 = [r['runs']['1'] for r in recs]
        ok = [r for r in r1 if r['fate'] == 'survive']
        if ok:
            bn = np.array([r['beta_out'] for r in ok])
            bl = np.array([r['beta_lin'] for r in ok])
            vE_n, vE_l = (bn[:, :2] ** 2).sum(1), (bl[:, :2] ** 2).sum(1)
            vT_n, vT_l = (bn[:, 2:] ** 2).sum(1), (bl[:, 2:] ** 2).sum(1)
            js['s1_varratio_E'] = float(vE_n.sum() / vE_l.sum())
            js['s1_varratio_E_boot95'] = boot_ratio(vE_n, vE_l, rng)
            js['s1_varratio_T2'] = float(vT_n.sum() / vT_l.sum())
            js['s1_varratio_T2_boot95'] = boot_ratio(vT_n, vT_l, rng)
            js['s1_sigmaE_nl'] = float(math.sqrt(vE_n.mean() / 2))
            js['s1_sigmaE_lin_paired'] = float(math.sqrt(vE_l.mean() / 2))
            js['s1_rms_shift_rel_E'] = float(math.sqrt((((bn[:, :2] - bl[:, :2]) ** 2).sum(1)).mean() / vE_l.mean()))
            Lr = []
            for b in bn:
                B = np.zeros(3)
                # diagonal (aspect) part of log-shape: beta_D X_D + beta_D2 X_D2 (+T2 ignored for side ratio)
                d = b[0] * np.array([1, -1, 0]) / math.sqrt(2) + b[1] * np.array([1, 1, -2]) / math.sqrt(6)
                Lr.append(d.max() - d.min())
            js['s1_median_lnLmax_over_Lmin_diag'] = float(np.median(Lr))
            js['s1_gates_max_constraint'] = float(max(abs(r['gates_end']['constraint']) for r in ok))
        if all(r.get('s_c') is not None for r in recs):
            sc = np.array([1e9 if r['s_c'] == '>40' else float(r['s_c']) for r in recs])
            js['s_c_quantiles'] = {('q%02d' % int(q * 100)): (float(np.quantile(sc, q)) if np.quantile(sc, q) < 1e8 else '>40')
                                   for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)}
            js['s_c_definitions'] = 's_p = amplification at which a fraction p of draws still survives = quantile (1-p) of per-draw critical s_c (log-bisection, 6-7 steps, assumes survival monotone along each ray)'
            js['s99_surv'] = js['s_c_quantiles']['q01']
            js['s90_surv'] = js['s_c_quantiles']['q10']
            js['s50_surv'] = js['s_c_quantiles']['q50']
            js['frac_sc_gt40'] = float(np.mean(sc > 1e8))
    else:
        f = [r['fate'] for r in recs]
        k = sum(x == 'survive' for x in f)
        js['survive'] = k
        js['frac'] = k / len(f)
        js['wilson95'] = wilson(k, len(f))
        js['fates'] = counts(f)
        M = mt['M']
        dl = mt['delta']
        js['sigma_lin_E_pred'] = math.sqrt(2 * abs(M[0][0][0] * M[0][0][1]) * dl)
        bl = np.array([r['beta_lin'] for r in recs])
        js['sigma_lin_E_sample'] = float(math.sqrt((bl[:, :2] ** 2).mean()))
        ok = [r for r in recs if r['fate'] == 'survive']
        if ok:
            bn = np.array([r['beta_out'] for r in ok])
            js['sigmaE_survivors_nl'] = float(math.sqrt((bn[:, :2] ** 2).mean()))
            bls = np.array([r['beta_lin'] for r in ok])
            js['sigmaE_survivors_lin'] = float(math.sqrt((bls[:, :2] ** 2).mean()))
        tf = [r['t_end'] for r in recs if r['fate'] not in ('survive', 'no_turning')]
        if tf:
            tb = [r['t_bounce'] for r in recs if r['fate'] not in ('survive', 'no_turning') and r.get('t_bounce') is not None]
            js['fail_t_end_median'] = float(np.median(tf))
            js['fail_t_end_range'] = [float(min(tf)), float(max(tf))]
            js['H_down_before_pinch_count'] = int(sum(1 for r in recs if r['fate'] == 'pinch_after_bounce' and r.get('H_down_times')))
        js['bounce_count'] = int(sum(1 for r in recs if r.get('t_bounce') is not None))
    S[job] = js
# paired comparisons
def paired(jobA, jobB, key=None):
    a, b = load(jobA), load(jobB)
    if not a or not b:
        return None
    da = {r['id']: r for r in a}
    db = {r['id']: r for r in b}
    ids = sorted(set(da) & set(db))
    def fs(r):
        return (r['runs'][key]['fate'] if key else r['fate']) == 'survive'
    n11 = sum(fs(da[i]) and fs(db[i]) for i in ids)
    n10 = sum(fs(da[i]) and not fs(db[i]) for i in ids)
    n01 = sum((not fs(da[i])) and fs(db[i]) for i in ids)
    return {'n': len(ids), 'both_survive': n11, 'A_only': n10, 'B_only': n01,
            'fracA': (n11 + n10) / len(ids), 'fracB': (n11 + n01) / len(ids)}
S['paired_plat_vs_lead_turn'] = {('s=%g' % s): paired('turn_lead', 'turn_plat', '%g' % s) for s in SGRID}
S['paired_plat_vs_lead_r1'] = {('tx=%g' % tx): paired('r1_lead_%g' % tx, 'r1_plat_%g' % tx) for tx in (-0.5, -1.0, -1.093)}
S['paired_deep_vs_1e-3'] = {'r1_-1.093': paired('r1_lead_-1.093', 'r1_lead_-1.093_deep'),
                            'turn': {('s=%g' % s): paired('turn_lead', 'turn_lead_deep', '%g' % s) for s in SGRID}}
det = load('det')
if det:
    S['det_thresholds'] = [{k: r[k] for k in ('dir', 'which', 'threshold', 'fate_above', 'nonmono')} for r in det]
with open(os.path.join(HERE, '..', 'results', 'D114_own.json'), 'w', encoding='utf-8') as f:
    json.dump(S, f, indent=1, default=float)
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(S, indent=1, default=float)[:15000])
