# sm-blind: diagnose the matching-scale (mu0) dependence of y_t and lambda (uses stored mr records; no new mr calls)
import os, sys, math, json
os.environ['OMP_NUM_THREADS'] = '1'
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
import numpy as np
import smcore as S
from scipy.integrate import solve_ivp
RES = os.path.join(os.path.dirname(WORK), 'results')
recs = json.load(open(os.path.join(RES, 'prod_mr_records.json')))
R = S.RGE(3, 'full', True)
Mt = 172.60
QCD = ('x01', 'x02', 'x03', 'x04'); EW = ('x10', 'x11', 'x20')
out = {}


def yt_qcd_run(yt0, mu0, mu1, asMZ=0.1180):
    """pure-QCD (nf=6) running of y_t (i.e. of mbar_t) with alpha_s^(6) decoupled at M_t, 4-loop gamma_m."""
    gm = S.gamma_m(6)[:4]
    def f(t, y):
        a = S.as6(math.exp(t / 2), asMZ, Mt, 91.1879, 4, 3) / math.pi
        return [-y[0] * sum(g * a ** (i + 1) for i, g in enumerate(gm))]
    s = solve_ivp(f, (2 * math.log(mu0), 2 * math.log(mu1)), [yt0], method='RK45', rtol=1e-10, atol=1e-14)
    return s.y[0, -1]


for k in ('mu0_0.5', 'mu0_0.707', 'central', 'mu0_1.414', 'mu0_2'):
    rec = recs[k]; mu0 = rec['mu']
    row = dict(mu0=mu0, as6_mu0=rec['as_mu'], alpha_GF_inv=1 / rec['alpha_GF'])
    for tag, orders in (('all', S.ALL_ORDERS), ('qcd_only', QCD), ('ew_only', EW), ('tree', ())):
        c0 = S.mr_couplings(rec, 'GF', orders, yb=rec['yb_own'])
        r = R.run(c0, mu0, [Mt])[Mt] if abs(mu0 - Mt) > 1e-9 else c0
        row[tag] = dict(yt_mu0=c0['yt'], yt_Mt=r['yt'], lam_mu0=c0['lam'], lam_Mt=r['lam'], g2_Mt=r['g2'], gY_Mt=r['gY'], g3_Mt=r['g3'])
    # QCD-only Yukawa with pure-QCD running back to Mt (isolates the alpha_s/QCD-coefficient convention)
    c0 = S.mr_couplings(rec, 'GF', QCD, yb=rec['yb_own'])
    row['qcd_only_pureQCDrun_yt_Mt'] = yt_qcd_run(c0['yt'], mu0, Mt) if abs(mu0 - Mt) > 1e-9 else c0['yt']
    row['t_coeffs'] = {c: rec['t_' + c] for c in ('y01', 'y02', 'y03', 'y04', 'y10', 'y11', 'y20')}
    row['H_coeffs'] = {c: rec['H_' + c] for c in ('y10', 'y11', 'y20')}
    out[k] = row
    print('%-10s mu0=%7.2f as6=%.5f | yt(Mt): all %.6f qcd-only %.6f (pureQCD run %.6f) ew-only %.6f | lam(Mt): all %.6f ew-only %.6f tree %.6f | g2 %.6f gY %.6f g3 %.6f' % (
        k, mu0, rec['as_mu'], row['all']['yt_Mt'], row['qcd_only']['yt_Mt'], row['qcd_only_pureQCDrun_yt_Mt'], row['ew_only']['yt_Mt'],
        row['all']['lam_Mt'], row['ew_only']['lam_Mt'], row['tree']['lam_Mt'], row['all']['g2_Mt'], row['all']['gY_Mt'], row['all']['g3_Mt']))
    print('            t:', {c: round(v, 4) for c, v in row['t_coeffs'].items()}, ' H:', {c: round(v, 3) for c, v in row['H_coeffs'].items()})
json.dump(out, open(os.path.join(RES, 'diag_mu0.json'), 'w'), indent=1)
