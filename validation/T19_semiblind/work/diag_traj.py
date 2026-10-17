# diag_traj.py -- diagnose large-amplitude failures: constraint, zeta, S, min eig along a trajectory
import os, sys, json, math
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d114_own as D
sys.stdout.reconfigure(encoding='utf-8')
m = D.Model('lead')
d_plat = 1.565e-4 / (1 - D.R_ZM) ** 0.75
dl = d_plat * (3.0 / D.vpot(D.PHIB)) ** 0.75
kE, kT = 19.32374589, -9.07346116
sn = np.array([3 * dl / (2 * D.S2 * math.sqrt(abs(k))) for k in (kE, kE, kT, kT, kT)])
Z = np.random.default_rng(2026101901).standard_normal((6, 10))
out = {}
for i, s in ((0, 10.0), (3, 10.0), (0, 5.0)):
    b = s * Z[i, :5] * np.sqrt(sn)
    p = s * Z[i, 5:] * dl / np.sqrt(sn)
    y0 = m.initial(0.0, b, p)
    m.nfev = 0
    ev = lambda t, y: np.linalg.eigvalsh(D.unpack(y[2:8]))[0] - 1e-6
    ev.terminal = True
    sol = solve_ivp(m.rhs, (0, 12), y0, method='DOP853', rtol=1e-10, atol=1e-12, events=[ev], dense_output=True)
    rows = []
    tt = np.unique(np.concatenate([np.linspace(0, sol.t[-1], 25), [sol.t[-1]]]))
    for t in tt:
        y = sol.sol(t)
        G = D.unpack(y[2:8]); Y = D.unpack(y[8:14])
        A = np.linalg.solve(G, Y)
        S = float(np.trace(A @ A)) * math.exp(-6 * y[0])
        zt = m.zeta(G)
        g = m.gates(y)
        rows.append({'t': round(float(t), 4), 'H': float(y[1]), 'lr': float(y[0]), 'mineig': float(np.linalg.eigvalsh(G)[0]),
                     'zeta_e4': zt * math.exp(-4 * y[0]), 'S': S, 'constraint': g['constraint'],
                     'beta': [0.5 * float(np.trace(D.XB[a] @ D.logm_spd(G / np.linalg.det(G) ** (1 / 3)))) for a in range(5)]})
    out['i%d_s%g' % (i, s)] = {'status': int(sol.status), 't_end': float(sol.t[-1]), 'nfev': m.nfev, 'rows': rows}
print(json.dumps(out, indent=0, default=float)[:12000])
