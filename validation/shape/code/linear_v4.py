# linear_v4.py -- linear shape modes on the v11 leading cosh background: transfer matrices, floor,
# real-state-moment function G(|t_x|), optimal widths, next-order point, emptiness-gate numbers.
# Written from scratch by agent shape-blind (R16 V4), 2026-10-15.
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, hashlib
import numpy as np
from scipy.integrate import solve_ivp
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTB = os.path.dirname(HERE)
RES = os.path.join(ROOTB, 'results')
sys.path.insert(0, HERE)
from ewald_v4 import zeff, shape_basis, expm_sym, direct_sum, PI  # noqa: E402

TAU_INF = 20.0
H_OVER_MP = 6.116e-6        # v11 Sec. "The turning point" (A_s calibration, instant reheating)


# ---------------------------------------------------------------- linear transfer
def transfer(kappa, tau0, tau1, rtol=1e-13, atol=1e-15):
    """(beta, p) transfer, p = l^3 dbeta/dtau, beta' = p sech^{3/2}(2tau), p' = kappa sech^{1/2}(2tau) beta."""
    def rhs(t, y):
        c = 1.0 / np.cosh(2.0 * t)
        a = c * np.sqrt(c)
        b = kappa * np.sqrt(c)
        return np.array([a * y[1], b * y[0], a * y[3], b * y[2]])
    if tau0 == tau1:
        return np.eye(2)
    sol = solve_ivp(rhs, (tau0, tau1), [1.0, 0.0, 0.0, 1.0], method='DOP853', rtol=rtol, atol=atol)
    y = sol.y[:, -1]
    return np.array([[y[0], y[2]], [y[1], y[3]]])


def closed_form(kappa):
    mp.mp.dps = 30
    mu = mp.sqrt(4 * mp.mpf(kappa) - 1) / 4
    D = 3 * mp.sqrt(2) / 4
    gg = mp.gamma(mp.mpf(5) / 4 + 1j * mu)
    T12 = mp.gamma(mp.mpf(7) / 4) * mp.gamma(mp.mpf(3) / 4) / (D * abs(gg) ** 2)
    T11 = mp.sqrt(2) * mp.cosh(mp.pi * mu)
    return float(T11), float(T12), float(T12 / 2)


def minmax_var(A, Bv):
    """min_s max(A11^2 s + A12^2/(4s), B11^2 s + B12^2/(4s)) (units eps_q = 1)."""
    a1, b1 = A[0] ** 2, A[1] ** 2 / 4.0
    a2, b2 = Bv[0] ** 2, Bv[1] ** 2 / 4.0
    cands = [np.sqrt(b / a) for a, b in ((a1, b1), (a2, b2)) if b > 0 and a > 0]
    if (a1 - a2) * (b2 - b1) > 0:
        cands.append(np.sqrt((b2 - b1) / (a1 - a2)))
    f = lambda s: max(a1 * s + b1 / s, a2 * s + b2 / s)
    s = min(cands, key=f)
    return s, f(s), a1 * s + b1 / s, a2 * s + b2 / s


def floor_of(kappa):
    M0 = transfer(kappa, 0.0, TAU_INF)
    return abs(M0[0, 0] * M0[0, 1]), M0


def Gfun(kappa, tx, floor=None):
    if floor is None:
        floor, _ = floor_of(kappa)
    A = transfer(kappa, tx, TAU_INF)[0]
    Bv = transfer(kappa, tx, -TAU_INF)[0]
    s, vmax, vp, vm = minmax_var(A, Bv)
    return dict(G=vmax / floor, s_opt=s, Vplus_over_floor=vp / floor, Vminus_over_floor=vm / floor,
                A_row=A.tolist(), B_row=Bv.tolist(), larger_branch='future' if vp >= vm else 'past',
                at_crossing=bool(abs(vp - vm) < 1e-9 * vmax))


def t_freeze(kappa):
    return float(np.arcsinh(np.sqrt(kappa)) / 2.0)


# ---------------------------------------------------------------- lattice quantities
def hessian_along(X, h=1e-3):
    f = lambda b: zeff(expm_sym(2.0 * b * X), grad=False)[0]
    return (-f(2 * h) + 16 * f(h) - 30 * f(0.0) + 16 * f(-h) - f(-2 * h)) / (12 * h * h)


def eps_q(Z1, xi, h_over_mp=H_OVER_MP):
    """eps_q = 1/(M_P^2 L_b^3 h) on the leading background, L_b^4 = (1-xi) N_C gamma/(3 M_P^2 h^2)."""
    ncg = (1.0 - xi) * Z1 / PI ** 2
    mpl3h = (1.0 / h_over_mp) ** 0.5 * (ncg / 3.0) ** 0.75
    return 1.0 / mpl3h, mpl3h


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'agent': 'shape-blind'}
    X = shape_basis()
    # ---- Ewald checks
    I = np.eye(3)
    z0 = zeff(I, B=1.0, F=0.0, grad=False)[0]
    zA = zeff(I, B=0.0, F=-1.0, grad=False)[0]
    dA = direct_sum((0.5, 0.5, 0.5), 60)
    Z1 = zeff(I, grad=False)[0]
    ew = {'Z2_cube_periodic': z0, 'Z2_cube_AAA': zA, 'direct_alternating_N60': dA,
          'v11_ref': [16.5323159598, -3.8631638072], 'Zeff_cube': Z1,
          'NC': Z1 / z0, 'gamma': z0 / PI ** 2, 'NC_gamma': Z1 / PI ** 2,
          'AAA_weyl_contribution': -zA / z0}
    # Hessian along E and T2 directions
    lam = [hessian_along(X[a]) for a in range(5)]
    ew['lambda'] = lam
    ew['trace_hess_over_Z'] = sum(lam) / Z1
    ew['v11_lambda_ref'] = {'E': 1902.24, 'T2': -893.20}
    # analytic projected gradient vs finite differences at a generic det-1 shape
    rng = np.random.default_rng(7)
    b = rng.normal(0, 0.25, 5)
    G = expm_sym(2 * np.tensordot(b, X, 1))
    Z, dZ = zeff(G)
    w, O = np.linalg.eigh(G)
    Gh = (O * np.sqrt(w)) @ O.T
    errs = []
    for a in range(5):
        hh = 1e-4
        f = lambda e: zeff(Gh @ expm_sym(e * X[a]) @ Gh, grad=False)[0]
        fd = (-f(2 * hh) + 8 * f(hh) - 8 * f(-hh) + f(-2 * hh)) / (12 * hh)
        an = float(np.trace(Gh @ dZ @ Gh @ X[a]))
        errs.append([fd, an])
    ew['gradient_check_generic_shape'] = errs
    ew['gradient_max_rel_err'] = max(abs(x - y) / max(1.0, abs(x)) for x, y in errs)
    # non-cubic check of the Laplace eigen-identity Tr Hess = 8/3 Z (geodesic second derivatives)
    lam_g = []
    for a in range(5):
        hh = 1e-3
        f = lambda e: zeff(Gh @ expm_sym(2 * e * X[a]) @ Gh, grad=False)[0]
        lam_g.append((-f(2 * hh) + 16 * f(hh) - 30 * f(0.0) + 16 * f(-hh) - f(-2 * hh)) / (12 * hh * hh))
    ew['trace_hess_over_Z_generic'] = sum(lam_g) / Z
    out['ewald'] = ew
    lamE, lamT = lam[0], lam[2]
    kap = {}
    for xi in (0.0, 0.30, 0.2999, 0.28, 0.258):
        Cb = (1 - xi) * Z1
        kap[str(xi)] = {'kappa_E': 3 * lamE / Cb, 'kappa_T2': 3 * lamT / Cb, 'sum': 3 * sum(lam) / Cb,
                        'eps_q': eps_q(Z1, xi)[0], 'MP2Lb3h': eps_q(Z1, xi)[1],
                        'delta_v11norm': np.sqrt(2) / 3 * eps_q(Z1, xi)[0]}
    out['kappa'] = kap
    # ---- item 1: floor and G(|t_x|) at xi = 0.30 (leading order), plus item 3 (kappa_E = 18.0, xi ~ 0.28)
    cases = {'LO_xi0.30': (kap['0.3']['kappa_E'], 0.30), 'NLO_xi0.28_kE18.0': (18.0, 0.28),
             'LO_xi0.2999': (kap['0.2999']['kappa_E'], 0.2999), 'free_xi0': (kap['0.0']['kappa_E'], 0.0)}
    txs = [0.1, 0.38, 0.5, 1.0]
    item = {}
    for name, (kE, xi) in cases.items():
        fl, M0 = floor_of(kE)
        T = transfer(kE, -TAU_INF, TAU_INF)
        T11c, T12c, flc = closed_form(kE)
        e_q, m3 = eps_q(Z1, xi)
        rec = {'kappa_E': kE, 'xi': xi, 'floor_dimless_ode': fl, 'floor_dimless_closed': flc,
               'T11_ode': T[0, 0], 'T12_ode': T[0, 1], 'T11_closed_gain': T11c, 'T12_closed_Pnorm': T12c,
               'T12_v11_Pi_norm': T12c * 3 * np.sqrt(2) / 4, 'M_0_to_inf': M0.tolist(),
               'det_T': float(np.linalg.det(T)), 'MP2Lb3h_leading': m3, 'eps_q': e_q,
               'sigma_floor2': flc * e_q, 'sigma_floor': np.sqrt(flc * e_q),
               'sigma_floor_alt': {'MP2Lb3h=2279.28(D109 setup)': np.sqrt(flc / 2279.28),
                                   'MP2Lb3h from v11 delta=2.04e-4 (plateau)': np.sqrt(flc * 3 * 2.04e-4 / np.sqrt(2))}}
        tf = t_freeze(kE)
        rec['t_freeze'] = tf
        G = {}
        for t in txs + [tf]:
            g = Gfun(kE, -t, fl)
            g2 = Gfun(kE, +t, fl)
            G['%.6f' % t] = dict(g, G_plus_tx=g2['G'])
        G['inf_limit_2gain'] = 2 * T11c
        G['tau=6'] = Gfun(kE, -6.0, fl)['G']
        rec['G'] = G
        grid = np.round(np.arange(0.0, 2.0001, 0.05), 4)
        rec['G_grid'] = [[float(t), Gfun(kE, -float(t), fl)['G']] for t in grid]
        item[name] = rec
    out['items_1_3'] = item
    # ---- optimal widths for the nonlinear runs (xi = 0.30)
    kE, kT = kap['0.3']['kappa_E'], kap['0.3']['kappa_T2']
    flE, _ = floor_of(kE)
    flT, _ = floor_of(kT)
    W = {}
    for tx in (-1.0, -0.75, -0.6, -0.5, -0.45, -0.38, -0.3, -0.2, -0.1, 0.0, 0.2, 0.5):
        gE, gT = Gfun(kE, tx, flE), Gfun(kT, tx, flT)
        W['%+.3f' % tx] = {'sE': gE['s_opt'], 'sT': gT['s_opt'], 'GE': gE['G'], 'GT': gT['G'],
                           'rmsE_future_lin': float(np.sqrt(gE['Vplus_over_floor'] * flE * kap['0.3']['eps_q'])),
                           'rmsE_past_lin': float(np.sqrt(gE['Vminus_over_floor'] * flE * kap['0.3']['eps_q']))}
    out['widths_xi0.30'] = W
    # ---- item 4: emptiness gate and preparation window
    gate = {}
    for xi in (0.30, 0.28, 0.258, 0.0):
        bud = (1 - xi) * Z1 / (2 * PI ** 3)
        gate[str(xi)] = {'NCeff': (1 - xi) * Z1 / z0, 'budget_NCeff_gamma_over_2pi': bud,
                         'max_lowest_boson_quanta(|n|=1)': int(np.ceil(bud) - 1),
                         'fermion_budget_in_lowest_AAA_quanta': bud / (np.sqrt(3) / 2),
                         'max_lowest_AAA_fermion_quanta(|n-th|=sqrt3/2)': int(np.ceil(bud / (np.sqrt(3) / 2)) - 1),
                         'single_quantum_max_|n|^2_allowed': int(np.floor(bud ** 2 - 1e-12)),
                         'coefficient_gamma_over_2pi': z0 / PI ** 2 / (2 * PI)}
    out['item4_gate'] = gate
    out['item4_window'] = {str(x): float(np.log(2 * PI / x)) for x in (0.0044, 0.0048, 0.00483, 0.0044158)}
    out['runtime_s'] = time.time() - t0
    out['code_sha256'] = {'linear_v4.py': sha(os.path.abspath(__file__)),
                          'ewald_v4.py': sha(os.path.join(HERE, 'ewald_v4.py'))}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'linear_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    # short printout
    print('Z2 cube', z0, zA, 'direct', dA, 'Zeff', Z1, 'NC', Z1 / z0)
    print('lambda', lam, 'TrH/Z', ew['trace_hess_over_Z'], 'generic', ew['trace_hess_over_Z_generic'],
          'grad err', ew['gradient_max_rel_err'])
    for k, v in kap.items():
        print('xi', k, 'kE %.5f kT %.5f sum %.5f eps_q %.6e MP2Lb3h %.3f' % (v['kappa_E'], v['kappa_T2'], v['sum'], v['eps_q'], v['MP2Lb3h']))
    for name, rec in item.items():
        print(name, 'kE %.4f floor ode %.6f closed %.6f T11 %.4f/%.4f T12 %.5f/%.5f sigma_floor %.5f tf %.4f' % (
            rec['kappa_E'], rec['floor_dimless_ode'], rec['floor_dimless_closed'], rec['T11_ode'], rec['T11_closed_gain'],
            rec['T12_ode'], rec['T12_closed_Pnorm'], rec['sigma_floor'], rec['t_freeze']))
        for t, g in rec['G'].items():
            print('   |tx|', t, g if isinstance(g, float) else 'G %.5f (+tx %.5f) s %.5f V+ %.4f V- %.4f %s cross %s' % (
                g['G'], g['G_plus_tx'], g['s_opt'], g['Vplus_over_floor'], g['Vminus_over_floor'], g['larger_branch'], g['at_crossing']))
    for k, v in W.items():
        print('tx', k, 'sE %.5f sT %.5f GE %.4f GT %.4f rmsE+ %.4f rmsE- %.4f' % (v['sE'], v['sT'], v['GE'], v['GT'], v['rmsE_future_lin'], v['rmsE_past_lin']))
    print(json.dumps(gate, indent=0))
    print(out['item4_window'])
    print('runtime', out['runtime_s'])


if __name__ == '__main__':
    main()
