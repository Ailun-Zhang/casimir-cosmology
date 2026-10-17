# analyze_v4.py -- aggregate nonlinear shards: survival rates, binomial/Wilson errors, t_*(q) with
# common-random-number bootstrap, robustness flips, gates; writes survival_table.json and BLIND_RESULTS.json.
# Written by agent shape-blind (R16 V4), 2026-10-15.
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, glob, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTB = os.path.dirname(HERE)
RES = os.path.join(ROOTB, 'results')
SH = os.path.join(RES, 'shards')
BRIEF_TX = [-1.0, -0.5, -0.38, -0.2, 0.0, 0.2, 0.5]


def sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def load():
    data = {}
    metas = {}
    for f in sorted(glob.glob(os.path.join(SH, '*.json'))):
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        m = d['meta']
        key = (m['tag'], m['dir'], round(m['tx'], 3))
        data.setdefault(key, {})
        for r in d['recs']:
            data[key][r['i']] = r
        metas[key] = m
    return data, metas


def lnratio(beta5):
    from numpy.linalg import eigvalsh
    X = np.zeros((5, 3, 3))
    X[0] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    X[1] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    for a, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        X[2 + a, i, j] = X[2 + a, j, i] = 1.0 / np.sqrt(2.0)
    b = eigvalsh(np.tensordot(beta5, X, 1))
    return b[-1] - b[0]


def tstar(txs, S, q):
    """first crossing of level q coming from t=0 toward negative t (linear interpolation)."""
    order = np.argsort(txs)[::-1]           # from largest (closest to 0 / positive) downward
    t = np.asarray(txs)[order]
    s = np.asarray(S)[order]
    for j in range(1, len(t)):
        if s[j - 1] >= q > s[j]:
            return float(t[j] + (q - s[j]) * (t[j - 1] - t[j]) / (s[j - 1] - s[j]))
    return float('nan')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data, metas = load()
    table = {}
    for key in sorted(data):
        tag, d, tx = key
        recs = data[key]
        idx = np.array(sorted(recs))
        surv = np.array([recs[i]['surv'] for i in idx])
        ent = {'tag': tag, 'dir': d, 'tx': tx, 'N': int(len(idx)), 'i_max': int(idx.max()) + 1}
        for lab, sel in (('N256', idx < 256), ('all', np.ones(len(idx), bool))):
            n = int(sel.sum())
            k = int(surv[sel].sum())
            if n == 0:
                continue
            p = k / n
            ent[lab] = {'n': n, 'k': k, 'S': p, 'se': float(np.sqrt(p * (1 - p) / n)), 'wilson95': wilson(k, n)}
        reasons = {}
        for i in idx:
            reasons[recs[i]['reason']] = reasons.get(recs[i]['reason'], 0) + 1
        ent['reasons'] = reasons
        ok = [recs[i] for i in idx if recs[i]['surv']]
        if ok:
            bo = np.array([r['beta_out'] for r in ok])
            bl = np.array([r['beta_lin'] for r in ok])
            ent['var_ratio_E_nonlin_over_lin'] = float(np.sum(bo[:, :2] ** 2) / np.sum(bl[:, :2] ** 2))
            ent['var_ratio_T2_nonlin_over_lin'] = float(np.sum(bo[:, 2:] ** 2) / np.sum(bl[:, 2:] ** 2))
            ent['rms_betaE_out_survivors'] = float(np.sqrt(np.mean(bo[:, :2] ** 2)))
            ent['median_lnLmax_Lmin_survivors'] = float(np.median([lnratio(b) for b in bo]))
            ent['median_lnLmax_Lmin_lin_survivors'] = float(np.median([lnratio(b) for b in bl]))
            ent['max_gate_cons'] = float(max(abs(r['cons']) for r in ok))
            ent['max_gate_det'] = float(max(abs(r['det_err']) for r in ok))
            ent['max_gate_tr'] = float(max(abs(r['tr_err']) for r in ok))
            ent['min_lam_survivors'] = float(min(r['lam_min'] for r in ok))
        bad = [recs[i] for i in idx if not recs[i]['surv']]
        if bad:
            ent['collapse_time_median_tau'] = float(np.median([r['t_last'] for r in bad]))
            ent['collapse_time_range_tau'] = [float(min(r['t_last'] for r in bad)), float(max(r['t_last'] for r in bad))]
            ent['collapse_nturn_values'] = sorted(set(int(r['nturn']) for r in bad))
        bl_all = np.array([recs[i]['beta_lin'] for i in idx])
        ent['rms_betaE_lin_all'] = float(np.sqrt(np.mean(bl_all[:, :2] ** 2)))
        ent['wall_mean_s'] = float(np.mean([recs[i]['wall'] for i in idx]))
        if key in metas:
            ent['meta'] = {k2: metas[key][k2] for k2 in ('xi', 'rtol', 'conv', 'eps_q', 'kE', 'kT', 'sE', 'sT', 'GE', 'dev_seed')}
        table['%s|%s|%+.3f' % key] = ent
    # ---- survival curves and t_* (tag A, fwd)
    fw = {k[2]: data[k] for k in data if k[0] == 'A' and k[1] == 'fwd'}
    txs = sorted(fw)
    curves = {}
    rng = np.random.default_rng(12345)
    for lab, nmax in (('N256', 256), ('N2048', 2048)):
        use = [t for t in txs if all(i in fw[t] for i in range(nmax))]
        if len(use) < 2:
            continue
        M = np.array([[fw[t][i]['surv'] for i in range(nmax)] for t in use], float)   # (ntx, nmax)
        S = M.mean(1)
        res = {'tx': use, 'S': S.tolist(), 'se': np.sqrt(S * (1 - S) / nmax).tolist()}
        for q in (0.90, 0.99):
            t0 = tstar(use, S, q)
            bs = []
            for _ in range(2000):
                jj = rng.integers(0, nmax, nmax)
                bs.append(tstar(use, M[:, jj].mean(1), q))
            bs = np.array(bs)
            bs = bs[np.isfinite(bs)]
            res['t_star_%d' % int(q * 100)] = {'value': t0, 'boot_median': float(np.median(bs)) if len(bs) else None,
                                                'boot68': np.percentile(bs, [16, 84]).tolist() if len(bs) else None,
                                                'boot95': np.percentile(bs, [2.5, 97.5]).tolist() if len(bs) else None,
                                                'boot_finite_frac': len(bs) / 2000}
        # monotonicity under CRN: draws surviving at a more negative tx but failing at a less negative one
        viol = 0
        neg = [j for j, t in enumerate(use) if t <= 0]
        for a_ in range(len(neg)):
            for b_ in range(a_ + 1, len(neg)):
                viol += int(np.sum((M[neg[a_]] == 1) & (M[neg[b_]] == 0)))   # use sorted ascending: neg[a_] more negative
        res['crn_monotonicity_violations'] = viol
        curves[lab] = res
    # ---- two-sided at brief points (i < 256)
    two = {}
    for t in BRIEF_TX:
        kf, kb = ('A', 'fwd', round(t, 3)), ('A', 'bwd', round(t, 3))
        if kf in data and kb in data:
            ii = [i for i in range(256) if i in data[kf] and i in data[kb]]
            k = sum(data[kf][i]['surv'] and data[kb][i]['surv'] for i in ii)
            two['%+.3f' % t] = {'n': len(ii), 'k_two_sided': int(k), 'S_two_sided': k / max(1, len(ii)),
                                'S_bwd': sum(data[kb][i]['surv'] for i in ii) / max(1, len(ii))}
    # ---- robustness flips
    rob = {}
    for tag in ('R12', 'V11'):
        for k in data:
            if k[0] == tag:
                ka = ('A', k[1], k[2])
                if ka in data:
                    ii = [i for i in data[k] if i in data[ka]]
                    flips = [i for i in ii if data[k][i]['surv'] != data[ka][i]['surv']]
                    bo1 = np.array([data[k][i]['beta_out'] for i in ii if data[k][i]['surv'] and data[ka][i]['surv']])
                    bo0 = np.array([data[ka][i]['beta_out'] for i in ii if data[k][i]['surv'] and data[ka][i]['surv']])
                    rel = float(np.sqrt(np.mean(np.sum((bo1 - bo0) ** 2, 1)) / np.mean(np.sum(bo0 ** 2, 1)))) if len(bo0) else None
                    rob['%s|%s|%+.3f' % k] = {'n': len(ii), 'flips': len(flips), 'flip_idx': flips[:20],
                                              'S_variant': float(np.mean([data[k][i]['surv'] for i in ii])),
                                              'S_A_same_draws': float(np.mean([data[ka][i]['surv'] for i in ii])),
                                              'rms_rel_shift_beta_out_common_survivors': rel}
    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'table': table, 'curves_fwd_tagA': curves,
           'two_sided': two, 'robustness': rob}
    with open(os.path.join(RES, 'survival_table.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    # ---- printout
    for key, e in table.items():
        s = e.get('all', {})
        print('%-22s N=%5d S=%.4f +- %.4f  W95=(%.3f,%.3f)  reasons=%s  varE=%s rmsE_out=%s' % (
            key, e['N'], s.get('S', np.nan), s.get('se', np.nan), s.get('wilson95', (np.nan,))[0], s.get('wilson95', (0, np.nan))[1],
            e['reasons'], ('%.4f' % e['var_ratio_E_nonlin_over_lin']) if 'var_ratio_E_nonlin_over_lin' in e else '-',
            ('%.4f' % e['rms_betaE_out_survivors']) if 'rms_betaE_out_survivors' in e else '-'))
    for lab, c in curves.items():
        print(lab, 'tx', c['tx'])
        print(lab, 'S ', ['%.4f' % x for x in c['S']])
        for q in ('t_star_90', 't_star_99'):
            print(lab, q, c[q])
        print(lab, 'CRN monotonicity violations', c['crn_monotonicity_violations'])
    print('two-sided', two)
    print('robustness', json.dumps(rob, indent=0))


if __name__ == '__main__':
    main()
