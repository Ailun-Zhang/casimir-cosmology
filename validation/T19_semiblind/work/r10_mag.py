# r10_mag.py -- magnitudes for the R10 leftovers (T19, unverified-check, 2026-10-14)
# (1) DA-5: quantum spread about the photon-pair branch centre (v11 App. I minisuperspace, own Ewald sums)
# (2) METHODOLOGY-2: sensitivity of D114 survival and sigma_E to the zero-mode fraction xi (higher orders lower xi)
import os, sys, json, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d114_own as D

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
XP = np.diag([2.0, -1.0, -1.0]) / math.sqrt(6.0)
CFREE = D.zeff(np.eye(3), False) / math.pi ** 2


def branch_setup():
    bg = np.linspace(-1.0, 3.0, 4001)
    Cf = np.array([D.zeff(D.expm_sym(2 * b * XP), False) for b in bg]) / math.pi ** 2
    return bg, Cf


def branch(xi, bg, Cf, delta_vac_lead):
    Cpair = Cf - 4 * math.pi * np.exp(-2 * bg / math.sqrt(6.0))   # (G^-1)_11 = exp(-4 beta/sqrt6)
    C = Cpair - xi * CFREE
    sp = CubicSpline(bg, C)
    dsp = sp.derivative()
    Cb = float(sp(0.0))

    def rhs(t, y):
        lrho, u, b, v = y
        e4 = math.exp(-4 * lrho)
        return [u, 2 * sp(b) / Cb * e4 - v * v / 2, v, -3 * u * v + 3 * dsp(b) / Cb * e4]

    def binf(b0, v0):
        r04 = sp(b0) / (Cb * (1 + v0 * v0 / 6))
        y0 = [0.25 * math.log(r04), 0.0, b0, v0]
        s = solve_ivp(rhs, (0, 14), y0, method='DOP853', rtol=1e-11, atol=1e-13)
        return float(s.y[2, -1]), float(math.exp(s.y[0, -1]))
    b_c, _ = binf(0.0, 0.0)
    h = 1e-5
    db0 = (binf(h, 0)[0] - binf(-h, 0)[0]) / (2 * h)
    dv0 = (binf(0, h)[0] - binf(0, -h)[0]) / (2 * h)
    # natural width at the turn: local curvature of C along the axis, delta scaled as C_b^{-3/4}
    Cvac = (1 - xi) * CFREE
    delta = delta_vac_lead * (Cvac / Cb) ** 0.75 * ((1 - 0.299897) * CFREE / Cvac) ** 0.75 if False else None
    # delta for this branch: delta = sqrt2/(3 M_P^2 h L_b^3), L_b^4 = C_b/(3 M_P^2 h^2) -> delta propto C_b^{-3/4}
    delta = delta_vac_lead * (((1 - 0.299897) * CFREE) / Cb) ** 0.75
    kap = 3 * float(dsp(0.0, 1)) / Cb
    snat = 3 * delta / (2 * D.S2 * math.sqrt(abs(kap)))
    rho0 = (sp(0.0) / Cb) ** 0.25
    var_v0 = (3 / (2 * D.S2)) ** 2 * (delta ** 2 / snat) / rho0 ** 6
    sig_lin = math.sqrt(db0 ** 2 * snat + dv0 ** 2 * var_v0)
    # nonlinear MC about the centre
    rng = np.random.default_rng(2026101941)
    z = rng.standard_normal((400, 2))
    bi = np.array([binf(z[i, 0] * math.sqrt(snat), z[i, 1] * math.sqrt(var_v0))[0] for i in range(len(z))])
    ratio = np.exp(math.sqrt(1.5) * bi)
    return {'xi': xi, 'C_b': Cb, 'beta_inf_centre': b_c, 'side_ratio_centre': math.exp(math.sqrt(1.5) * b_c),
            'dbinf_dbeta0': db0, 'dbinf_dv0': dv0, 'kappa_axis_at_cube': kap, 'delta_branch': delta, 's_nat': snat,
            'sigma_beta0': math.sqrt(snat), 'sigma_binf_lin': sig_lin, 'amplification_rms': sig_lin / math.sqrt(snat),
            'MC_N': len(z), 'MC_beta_inf_mean': float(bi.mean()), 'MC_beta_inf_sd': float(bi.std()),
            'MC_ratio_5_50_95': [float(np.percentile(ratio, q)) for q in (5, 50, 95)]}


def surv_task(args):
    xi, kind, tx, s_list, Z, delta, ks, snat_or_sopt, Ms = args
    m = D.Model('lead', r=xi)
    out = []
    sb = np.sqrt(snat_or_sopt)
    for z in Z:
        b0 = np.asarray(z[:5]) * sb
        p0 = np.asarray(z[5:]) * delta / sb
        if kind == 'turn':
            out.append({('%g' % s): m.run(0.0, s * b0, s * p0)['fate'] for s in s_list})
        else:
            out.append({'r1': m.run(tx, b0, p0)['fate']})
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    res = {}
    d_plat = 1.565e-4 / (1 - D.R_ZM) ** 0.75
    d_lead = d_plat * (3.0 / D.vpot(D.PHIB)) ** 0.75
    t0 = time.time()
    bg, Cf = branch_setup()
    res['DA5_photon_branch'] = [branch(xi, bg, Cf, d_lead) for xi in (0.0, 0.246, 0.2999)]
    res['DA5_cpu_s'] = time.time() - t0
    print(json.dumps(res, indent=1, default=float), flush=True)
    # METHODOLOGY-2 magnitudes: xi = 0.28 (first next order), 0.2484 (kappa_E = 18.0 equivalent)
    from concurrent.futures import ProcessPoolExecutor
    N = 512
    Zt = np.random.default_rng(2026101901).standard_normal((2048, 10))[:N]
    Zf = np.random.default_rng(2026101914).standard_normal((2048, 10))[:N]
    Zh = np.random.default_rng(2026101912).standard_normal((2048, 10))[:N]
    jobs = {}
    for xi in (0.28, 0.2484):
        m = D.Model('lead', r=xi)
        lam = []
        h = 2e-3
        for a in (1, 3):
            zp = D.zeff(D.expm_sym(2 * h * D.XB[a]), False)
            zm = D.zeff(D.expm_sym(-2 * h * D.XB[a]), False)
            lam.append((zp + zm - 2 * m.Zf) / h ** 2)
        kE, kT = 3 * lam[0] / m.Z0, 3 * lam[1] / m.Z0
        delta = d_plat * (1 - D.R_ZM) ** 0.75 / (1 - xi) ** 0.75 * (3.0 / D.vpot(D.PHIB)) ** 0.75
        ks = [kE, kE, kT, kT, kT]
        snat = [3 * delta / (2 * D.S2 * math.sqrt(abs(k))) for k in ks]
        M0 = m.lin_transfer(kE, 0.0)
        sig = math.sqrt(M0[0, 0] ** 2 * snat[0] + M0[0, 1] ** 2 * delta ** 2 / snat[0])
        res['M2_xi%.4f' % xi] = {'kappa_E': kE, 'kappa_T': kT, 'delta_lead': delta, 'sigma_E_lin_lead': sig}
        jobs[(xi, 'turn', 0.0)] = (snat, Zt, delta, ks)
        for tx, ZZ in ((-1.093, Zf), (-0.5, Zh)):
            Ms = [m.lin_transfer(k, tx) for k in ks]
            sopt = [abs(M[0, 1] / M[0, 0]) * delta for M in Ms]
            jobs[(xi, 'r1', tx)] = (sopt, ZZ, delta, ks)
    with ProcessPoolExecutor(max_workers=1) as ex:
        futs = {}
        for (xi, kind, tx), (w, ZZ, delta, ks) in jobs.items():
            for c in range(0, N, 64):
                futs[ex.submit(surv_task, (xi, kind, tx, [1.0, 3.0, 5.0, 10.0], ZZ[c:c + 64].tolist(), delta, ks, w, None))] = (xi, kind, tx)
        acc = {}
        for fu, key in futs.items():
            acc.setdefault(key, []).extend(fu.result())
    for (xi, kind, tx), lst in acc.items():
        if kind == 'turn':
            res['M2_xi%.4f' % xi]['turn'] = {s: sum(r[s] == 'survive' for r in lst) / len(lst) for s in ('1', '3', '5', '10')}
        else:
            res['M2_xi%.4f' % xi]['r1_tx%g' % tx] = sum(r['r1'] == 'survive' for r in lst) / len(lst)
        res['M2_xi%.4f' % xi]['N'] = len(lst)
    with open(os.path.join(RES, 'R10_magnitudes_own.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
