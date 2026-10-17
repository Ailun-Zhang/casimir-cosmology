# run_d114.py -- production runner for the independent D114 check (T19, unverified-check, 2026-10-14)
# usage: python run_d114.py run [jobs...] | summarize
import os, sys, json, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d114_own as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'd114_runs')
SGRID = [1.0, 3.0, 5.0, 10.0, 20.0, 40.0]
TXS = [-0.25, -0.5, -1.0, -1.093, -2.0]
N = 2048
SEED_TURN = 2026101901
SEED_R1 = {-0.25: 2026101911, -0.5: 2026101912, -1.0: 2026101913, -1.093: 2026101914, -2.0: 2026101915}
_models = {}


def model(bg):
    m = _models.get(bg)
    if m is None:
        m = D.Model(bg)
        _models[bg] = m
    return m


def setup(bg):
    """kappas, delta, natural widths and transfers for background bg."""
    m = model(bg)
    lam = []
    h = 2e-3
    for a in (1, 3):
        zp = D.zeff(D.expm_sym(2 * h * D.XB[a]), False)
        zm = D.zeff(D.expm_sym(-2 * h * D.XB[a]), False)
        lam.append((zp + zm - 2 * m.Zf) / h ** 2)
    kE, kT = 3 * lam[0] / m.Z0, 3 * lam[1] / m.Z0
    d_plat = 1.565e-4 / (1 - D.R_ZM) ** 0.75
    vb_p = D.vpot(D.PHIB)
    delta = d_plat * (3.0 / vb_p) ** 0.75 if bg == 'lead' else d_plat
    ks = [kE, kE, kT, kT, kT]
    snat = [3 * delta / (2 * D.S2 * math.sqrt(abs(k) * m.vb / 3)) for k in ks]
    return {'kE': kE, 'kT': kT, 'ks': ks, 'delta': delta, 'snat': snat, 'vb': m.vb}


def transfers(bg, ks, t0):
    m = model(bg)
    cache = {}
    out = []
    for k in ks:
        if k not in cache:
            cache[k] = m.lin_transfer(k, t0).tolist()
        out.append(cache[k])
    return out


def run_one(bg, t0, beta, Pi, Ms, pinch=1e-3):
    m = model(bg)
    r = m.run(t0, np.asarray(beta), np.asarray(Pi), pinch=pinch, stop_on_down=(pinch >= 1e-3))
    r['beta_lin'] = [Ms[a][0][0] * beta[a] + Ms[a][0][1] * Pi[a] for a in range(5)]
    return r


def task_turn(args):
    bg, ids, Z, snat, delta, Ms, bisect, pinch = args
    res = []
    sb = np.sqrt(snat)
    for i, z in zip(ids, Z):
        b0 = np.asarray(z[:5]) * sb
        p0 = np.asarray(z[5:]) * delta / sb
        runs = {}
        for s in SGRID:
            r = run_one(bg, 0.0, (s * b0).tolist(), (s * p0).tolist(), Ms, pinch)
            runs['%g' % s] = r
        surv = [runs['%g' % s]['fate'] == 'survive' for s in SGRID]
        first_fail = next((j for j, v in enumerate(surv) if not v), None)
        nonmono = first_fail is not None and any(surv[first_fail:])
        sc = None
        if bisect:
            if first_fail is None:
                sc = '>40'
            elif first_fail == 0:
                lo, hi = 0.25, 1.0
                for _ in range(7):
                    mid = math.sqrt(lo * hi)
                    ok = run_one(bg, 0.0, (mid * b0).tolist(), (mid * p0).tolist(), Ms, pinch)['fate'] == 'survive'
                    lo, hi = (mid, hi) if ok else (lo, mid)
                sc = math.sqrt(lo * hi)
            else:
                lo, hi = SGRID[first_fail - 1], SGRID[first_fail]
                for _ in range(6):
                    mid = math.sqrt(lo * hi)
                    ok = run_one(bg, 0.0, (mid * b0).tolist(), (mid * p0).tolist(), Ms, pinch)['fate'] == 'survive'
                    lo, hi = (mid, hi) if ok else (lo, mid)
                sc = math.sqrt(lo * hi)
        res.append({'id': int(i), 'runs': runs, 'nonmono': nonmono, 's_c': sc})
    return res


def task_r1(args):
    bg, tx, ids, Z, sopt, delta, Ms, pinch = args
    res = []
    sb = np.sqrt(sopt)
    for i, z in zip(ids, Z):
        b0 = (np.asarray(z[:5]) * sb).tolist()
        p0 = (np.asarray(z[5:]) * delta / sb).tolist()
        r = run_one(bg, tx, b0, p0, Ms, pinch)
        r['id'] = int(i)
        res.append(r)
    return res


def task_det(args):
    bg, label, vec, which, Ms = args
    m = model(bg)
    grid = np.geomspace(0.01, 2.0, 40)

    def fate(a):
        beta = np.zeros(5); Pi = np.zeros(5)
        if which == 'beta':
            beta = a * np.asarray(vec)
        else:
            Pi = a * np.asarray(vec)
        return m.run(0.0, beta, Pi)['fate']
    fates = [fate(a) for a in grid]
    j = next((k for k, f in enumerate(fates) if f != 'survive'), None)
    thr = None
    if j is not None and j > 0:
        lo, hi = grid[j - 1], grid[j]
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            if fate(mid) == 'survive':
                lo = mid
            else:
                hi = mid
        thr = 0.5 * (lo + hi)
        fhi = fate(hi)
    else:
        fhi = None
    return [{'bg': bg, 'dir': label, 'which': which, 'threshold': thr, 'fate_above': fhi,
             'grid_fates': fates, 'nonmono': j is not None and any(f == 'survive' for f in fates[j:])}]


def jobs_spec():
    specs = {}
    for bg in ('lead', 'plat'):
        st = setup(bg)
        Ms0 = transfers(bg, st['ks'], 0.0)
        Zt = np.random.default_rng(SEED_TURN).standard_normal((N, 10))
        specs['turn_' + bg] = ('turn', bg, Zt, st, Ms0, bg == 'lead', 1e-3)
        txs = TXS if bg == 'lead' else [-0.5, -1.0, -1.093]
        for tx in txs:
            Ms = transfers(bg, st['ks'], tx)
            sopt = [abs(M[0][1] / M[0][0]) * st['delta'] for M in Ms]
            Zr = np.random.default_rng(SEED_R1[tx]).standard_normal((N, 10))
            specs['r1_%s_%g' % (bg, tx)] = ('r1', bg, tx, Zr, sopt, st['delta'], Ms, 1e-3)
    # pinch-threshold sensitivity (deep criterion 1e-6) on subsamples
    st = setup('lead')
    Ms = transfers('lead', st['ks'], -1.093)
    sopt = [abs(M[0][1] / M[0][0]) * st['delta'] for M in Ms]
    Zr = np.random.default_rng(SEED_R1[-1.093]).standard_normal((N, 10))[:96]
    specs['r1_lead_-1.093_deep'] = ('r1', 'lead', -1.093, Zr, sopt, st['delta'], Ms, 1e-6)
    Zt = np.random.default_rng(SEED_TURN).standard_normal((N, 10))[:96]
    specs['turn_lead_deep'] = ('turn', 'lead', Zt, st, transfers('lead', st['ks'], 0.0), False, 1e-6)
    return specs


def meta_of(specs):
    meta = {}
    for k, v in specs.items():
        if v[0] == 'turn':
            st = v[3]
            meta[k] = {'kind': 'turn', 'bg': v[1], 'N': len(v[2]), 'seed': SEED_TURN, 'kE': st['kE'], 'kT': st['kT'],
                       'delta': st['delta'], 'snat': st['snat'], 'vb': st['vb'], 'M': v[4], 'pinch': v[6]}
        else:
            meta[k] = {'kind': 'r1', 'bg': v[1], 'tx': v[2], 'N': len(v[3]), 'seed': SEED_R1[v[2]], 'sopt': v[4],
                       'delta': v[5], 'M': v[6], 'pinch': v[7]}
    return meta


def main_run(jobnames):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(OUT, exist_ok=True)
    specs = jobs_spec()
    with open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_of(specs), f, indent=1)
    if not jobnames:
        jobnames = ['det', 'turn_lead', 'r1_lead_-0.5', 'r1_lead_-1', 'r1_lead_-1.093', 'r1_lead_-0.25', 'r1_lead_-2',
                    'turn_plat', 'r1_plat_-0.5', 'r1_plat_-1', 'r1_plat_-1.093', 'r1_lead_-1.093_deep', 'turn_lead_deep']
    with ProcessPoolExecutor(max_workers=3) as ex:
        for jn in jobnames:
            path = os.path.join(OUT, jn + '.jsonl')
            done = set()
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        try:
                            done.add(json.loads(line)['id'])
                        except Exception:
                            pass
            tasks = []
            if jn == 'det':
                st = setup('lead')
                Ms0 = transfers('lead', st['ks'], 0.0)
                dirs = {'slab(+XD2)': [0, 1, 0, 0, 0], 'rod(-XD2)': [0, -1, 0, 0, 0], 'mid(XD)': [1, 0, 0, 0, 0]}
                for lab, vec in dirs.items():
                    for which in ('beta', 'Pi'):
                        if lab + '|' + which not in done:
                            tasks.append(('det', ('lead', lab, vec, which, Ms0)))
            else:
                sp = specs[jn]
                if sp[0] == 'turn':
                    _, bg, Z, st, Ms, bis, pinch = sp
                    ids = [i for i in range(len(Z)) if i not in done]
                    for c in range(0, len(ids), 16):
                        ii = ids[c:c + 16]
                        tasks.append(('turn', (bg, ii, Z[ii].tolist(), st['snat'], st['delta'], Ms, bis, pinch)))
                else:
                    _, bg, tx, Z, sopt, delta, Ms, pinch = sp
                    ids = [i for i in range(len(Z)) if i not in done]
                    for c in range(0, len(ids), 32):
                        ii = ids[c:c + 32]
                        tasks.append(('r1', (bg, tx, ii, Z[ii].tolist(), sopt, delta, Ms, pinch)))
            t0 = time.time()
            fn = {'turn': task_turn, 'r1': task_r1, 'det': task_det}
            futs = [ex.submit(fn[k], a) for k, a in tasks]
            with open(path, 'a', encoding='utf-8') as f:
                for fu in as_completed(futs):
                    for rec in fu.result():
                        if jn == 'det':
                            rec['id'] = rec['dir'] + '|' + rec['which']
                        f.write(json.dumps(rec, default=float) + '\n')
                    f.flush()
            with open(os.path.join(OUT, 'progress.log'), 'a', encoding='utf-8') as g:
                g.write('%s done %d tasks in %.1f s at %s\n' % (jn, len(tasks), time.time() - t0, time.strftime('%H:%M:%S')))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1] == 'run':
        main_run(sys.argv[2:])
