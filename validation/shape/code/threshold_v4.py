# threshold_v4.py -- sensitivity of survival to the collapse threshold lambda_min(G) (shape-blind, R16 V4), 2026-10-15.
# (1) stricter thresholds post hoc (survivors with min lambda below thr reclassified; exact, no re-run needed);
# (2) looser threshold 1e-5: re-run every draw that hit lambda_min = 1e-3 at selected t_x (same deviates, same widths).
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
sys.path.insert(0, HERE)
from analyze_v4 import load, tstar  # noqa: E402

LOOSE = 1e-5


def rerun(task):
    import nonlin_v4 as nl
    nl.LMIN = LOOSE
    tx, idx = task
    m = nl.Model(0.30)
    Z = np.load(nl.DEVFILE)
    gE, gT = nl.widths(m, tx)
    out = []
    for i in idx:
        beta, p = nl.draw(m, Z[i], gE['s_opt'], gT['s_opt'])
        r = m.run(tx, beta, p, 'fwd')
        out.append(dict(i=int(i), surv=r['surv'], reason=r['reason'], lam_min=r['lam_min'], t_last=r['t_last'], u_end=r['u_end']))
    return tx, out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data, _ = load()
    fw = {k[2]: data[k] for k in data if k[0] == 'A' and k[1] == 'fwd'}
    txs = sorted(t for t in fw if all(i in fw[t] for i in range(2048)))
    res = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'strict': {}, 'loose_rerun': {}}
    for thr in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        S = [float(np.mean([fw[t][i]['surv'] and fw[t][i]['lam_min'] > thr for i in range(2048)])) for t in txs]
        res['strict']['%.0e' % thr] = {'tx': txs, 'S': S, 't_star_90': tstar(txs, S, 0.90), 't_star_99': tstar(txs, S, 0.99)}
        print('thr %.0e' % thr, 't90 %.4f t99 %.4f' % (tstar(txs, S, 0.90), tstar(txs, S, 0.99)), ['%+.2f:%.4f' % (t, s) for t, s in zip(txs, S)])
    tasks = []
    for tx in (-0.36, -0.24, -0.20):
        bad = [i for i in range(2048) if not fw[tx][i]['surv']]
        for k in range(0, len(bad), 16):
            tasks.append((tx, bad[k:k + 16]))
    from multiprocessing import Pool
    with Pool(4) as pool:
        for tx, out in pool.imap_unordered(rerun, tasks):
            d = res['loose_rerun'].setdefault('%+.3f' % tx, [])
            d.extend(out)
    for k, v in res['loose_rerun'].items():
        rec = sum(r['surv'] for r in v)
        print('loose 1e-5 rerun tx', k, 'n_collapsed_at_1e-3', len(v), 'recovered', rec,
              'reasons', {r['reason']: sum(1 for x in v if x['reason'] == r['reason']) for r in v})
        res['loose_rerun'][k] = {'n': len(v), 'recovered': int(rec), 'recs': v}
    with open(os.path.join(RES, 'threshold_sensitivity.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
