# verify_v4.py -- independent cross-checks for shape-blind (R16 V4), 2026-10-15.
# (1) G(|t_x|) from a second formulation: physical-velocity variables (beta, dbeta/dtau), LSODA, rtol 1e-12.
# (2) Nonlinear code in the small-amplitude limit vs linear rows (amplitude scan: deviation must scale ~ amplitude).
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'results')
sys.path.insert(0, HERE)
from linear_v4 import minmax_var  # noqa: E402
from nonlin_v4 import Model, widths, draw, deviates  # noqa: E402


def rows_velocity_form(kappa, tx, tend):
    """beta row of the (beta,p) transfer from tx to tend, computed in (beta, beta_dot) with LSODA."""
    l3 = np.cosh(2 * tx) ** 1.5

    def rhs(t, y):
        return [y[1], -3 * np.tanh(2 * t) * y[1] + kappa * y[0] / np.cosh(2 * t) ** 2]
    r = []
    for y0 in ([1.0, 0.0], [0.0, 1.0 / l3]):
        s = solve_ivp(rhs, (tx, tend), y0, method='LSODA', rtol=1e-12, atol=1e-14)
        r.append(s.y[0, -1])
    return np.array(r)


def Gvel(kappa, t):
    A0 = rows_velocity_form(kappa, 0.0, 20.0)
    floor = abs(A0[0] * A0[1])
    A = rows_velocity_form(kappa, -t, 20.0)
    B = rows_velocity_form(kappa, -t, -20.0)
    return minmax_var(A, B)[1] / floor, floor


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    lin = json.load(open(os.path.join(RES, 'linear_results.json'), encoding='utf-8'))
    chk = {}
    for name in ('LO_xi0.30', 'NLO_xi0.28_kE18.0'):
        rec = lin['items_1_3'][name]
        kE = rec['kappa_E']
        c = {}
        for tkey, g in rec['G'].items():
            if not isinstance(g, dict):
                continue
            gv, fl = Gvel(kE, float(tkey))
            c[tkey] = {'G_DOP853_beta_p': g['G'], 'G_LSODA_velocity': gv, 'rel': gv / g['G'] - 1}
        c['floor_LSODA_velocity'] = fl
        c['floor_rel'] = fl / rec['floor_dimless_closed'] - 1
        chk[name] = c
        print(name, json.dumps(c))
    out['G_second_formulation'] = chk
    # amplitude scan of the nonlinear code
    m = Model(0.30)
    Z = deviates()
    scan = {}
    for tx in (0.0, -0.38, -1.0):
        gE, gT = widths(m, tx)
        for d in ('fwd', 'bwd'):
            row = gE['A_row'] if d == 'fwd' else gE['B_row']
            rowT = gT['A_row'] if d == 'fwd' else gT['B_row']
            res = []
            for sc in (1e-4, 1e-5, 1e-6):
                beta, p = draw(m, Z[1], gE['s_opt'], gT['s_opt'], scale=sc)
                r = m.run(tx, beta, p, d, rtol=1e-12, atol=1e-18)
                lin5 = np.array([(row if a < 2 else rowT)[0] * beta[a] + (row if a < 2 else rowT)[1] * p[a] for a in range(5)])
                res.append([sc, float(np.max(np.abs(np.array(r['beta_out']) - lin5)) / np.max(np.abs(lin5))), r['surv']])
            scan['%+.2f_%s' % (tx, d)] = res
            print('tx %+.2f %s' % (tx, d), ['scale %.0e dev %.2e' % (a, b) for a, b, _ in res])
    out['nonlinear_small_amplitude_scan'] = scan
    with open(os.path.join(RES, 'verify_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
