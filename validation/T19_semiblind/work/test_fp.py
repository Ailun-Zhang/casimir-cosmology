# test_fp.py -- consistency of Fincke-Pohst vs box enumeration in zeff, and timing of large-amplitude runs
import os, sys, json, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d114_own as D
sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(7)
res = {'fp_vs_box': []}
for amp in (0.3, 1.0, 2.0):
    B = np.tensordot(rng.normal(0, amp, 5), D.XB, 1)
    G = D.expm_sym(2 * B)
    Gi = np.linalg.inv(G)
    # forced box path
    old = D.lat_points
    D.lat_points = lambda A, ad, half: (D._hbox(np.floor(np.sqrt(D.QMAX * ad) + 0.5)) if half
                                        else D._ibox(np.floor(np.sqrt(D.QMAX * ad) + 1e-12)))
    try:
        zb, Db = D.zeff(G)
    except RuntimeError as e:
        zb, Db = None, None
    D.lat_points = old
    D.lat_points_fp = None
    # forced FP path
    zf = None
    old2 = D.lat_points
    D.lat_points = lambda A, ad, half: D.fp_enum(A, D.QMAX * (1 + 1e-9), 0.5 if half else 0.0)
    t = time.time(); zf, Df = D.zeff(G); tf = time.time() - t
    D.lat_points = old2
    t = time.time(); zh, Dh = D.zeff(G); th = time.time() - t
    res['fp_vs_box'].append({'amp': amp, 'mineig': float(np.linalg.eigvalsh(G)[0]), 'Z_box': zb, 'Z_fp': zf, 'Z_hybrid': zh,
                             'relZ': None if zb is None else abs(zf / zb - 1),
                             'relD': None if Db is None else float(np.abs(Df - Db).max() / np.abs(Db).max()),
                             't_fp_ms': tf * 1e3, 't_hybrid_ms': th * 1e3})
m = D.Model('lead')
d_plat = 1.565e-4 / (1 - D.R_ZM) ** 0.75
dl = d_plat * (3.0 / D.vpot(D.PHIB)) ** 0.75
kE, kT = 19.32374589, -9.07346116
sn = np.array([3 * dl / (2 * D.S2 * math.sqrt(abs(k))) for k in (kE, kE, kT, kT, kT)])
Z = np.random.default_rng(2026101901).standard_normal((6, 10))
res['timing'] = []
for i in range(6):
    for s in (10.0, 20.0, 40.0):
        b = s * Z[i, :5] * np.sqrt(sn)
        p = s * Z[i, 5:] * dl / np.sqrt(sn)
        r = m.run(0.0, b, p)
        res['timing'].append({'i': i, 's': s, 'fate': r['fate'], 'cpu_s': r.get('cpu_s'), 'nfev': r.get('nfev'), 't_end': r.get('t_end')})
print(json.dumps(res, indent=1, default=float))
