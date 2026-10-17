# sm-blind R16 V1 post-processing: (i) FJ-tadpole "ratio-of-running-masses" variant (tadpole-convention test),
# (ii) asymmetric mu0 errors, (iii) standard vs conservative error budget, (iv) summary print. Updates BLIND_RESULTS.json.
import os, sys, math, json, hashlib, datetime
os.environ['OMP_NUM_THREADS'] = '1'
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
sys.stdout.reconfigure(encoding='utf-8')
import smcore as S
RES = os.path.join(os.path.dirname(WORK), 'results')
B = json.load(open(os.path.join(RES, 'BLIND_RESULTS.json')))
runs = json.load(open(os.path.join(RES, 'prod_runs.json')))
recs = json.load(open(os.path.join(RES, 'prod_mr_records.json')))
diag = json.load(open(os.path.join(RES, 'diag_mu0.json')))
rc = recs['central']; C = runs['central']
PI = math.pi
aQ = rc['as_mu'] / (4 * PI); aE = rc['alpha_GF'] / (4 * PI)
# (i) convention check at one loop: y = x - dr (bosons, squared masses) ; y_t = x_t - dr/2 (fermion, linear)
chk = dict(H=(rc['H_y10'], rc['H_x10'] - rc['dr10']), W=(rc['W_y10'], rc['W_x10'] - rc['dr10']), Z=(rc['Z_y10'], rc['Z_x10'] - rc['dr10']),
           t=(rc['t_y10'], rc['t_x10'] - rc['dr10'] / 2))
def ser(p, qcd=False):
    s = 1 + aE * rc[p + '_x10'] + aE * aQ * rc[p + '_x11'] + aE ** 2 * rc[p + '_x20']
    if qcd:
        s += aQ * rc[p + '_x01'] + aQ ** 2 * rc[p + '_x02'] + aQ ** 3 * rc[p + '_x03'] + aQ ** 4 * rc[p + '_x04']
    return s
dR = 1 + aE * rc['dr10'] + aE * aQ * rc['dr11'] + aE ** 2 * rc['dr20']
Gf = rc['Gf']
alt = dict(lam=Gf / math.sqrt(2) * rc['MH'] ** 2 * ser('H') / dR, yt=2 ** 0.75 * math.sqrt(Gf) * rc['Mt'] * ser('t', True) / math.sqrt(dR),
           g2=math.sqrt(2 ** 2.5 * Gf * rc['MW'] ** 2 * ser('W') / dR))
alt['gY'] = math.sqrt(2 ** 2.5 * Gf * (rc['MZ'] ** 2 * ser('Z') - rc['MW'] ** 2 * ser('W')) / dR)
c_alt = dict(alt, g3=math.sqrt(4 * PI * rc['as_mu']), yb=C['at_mu0']['yb'], ytau=C['at_mu0']['ytau']); c_alt['g1'] = math.sqrt(5 / 3) * c_alt['gY']
r_alt = S.RGE(3, 'full', True).run(c_alt, rc['mu'], [B['conventions']['mu_b_GeV']])[B['conventions']['mu_b_GeV']]
tad = dict(one_loop_convention_check={k: dict(y10=v[0], x10_minus_dr=v[1]) for k, v in chk.items()},
           tadpole_sizes=dict(t_x10=rc['t_x10'], t_y10=rc['t_y10'], dr10=rc['dr10'], H_x10=rc['H_x10'], H_y10=rc['H_y10']),
           ratio_variant_at_Mt=alt, ratio_variant_at_mub=dict(lam=r_alt['lam'], yt=r_alt['yt']),
           delta_vs_central_at_mub=dict(lam=r_alt['lam'] - C['at_mub']['lam'], yt=r_alt['yt'] - C['at_mub']['yt']),
           note='couplings from FJ-scheme running masses (tadpoles included) divided by v(mu)^2 without re-expansion; differs from the '
                'y-expansion only at O(alpha^3)-type products of large tadpole terms; reported, not adopted')
# (ii) asymmetric mu0 errors
mu0s = ['mu0_0.5', 'mu0_0.707', 'central', 'mu0_1.414', 'mu0_2']
asym = {}
for q in ('lam', 'yt', 'g2', 'gY', 'g3'):
    v = [runs[k]['at_mub'][q] for k in mu0s]; c = C['at_mub'][q]
    asym[q] = dict(plus=max(v) - c, minus=min(v) - c, values=dict(zip(mu0s, v)))
asym['lam_Mt_backrun'] = {k: diag[k]['all']['lam_Mt'] for k in mu0s}
asym['yt_Mt_backrun'] = {k: diag[k]['all']['yt_Mt'] for k in mu0s}
asym['yt_Mt_QCDonly_pureQCDbackrun'] = {k: diag[k]['qcd_only_pureQCDrun_yt_Mt'] for k in mu0s}
# (iii) budgets
dl = abs(B['derivatives_at_mub']['dlam_dMt'])
E = B['errors']['lam']
ren = 0.11 * dl
std = dict(a_mu0_maxdev=E['a_mu0_maxdev'], b_rge_4l5l=abs(E['b_rge_4l5l_total(3l-only minus central)']), b_as_order=abs(E['b_as_5l_run_4l_dec']),
           b_pole_renormalon_0p11GeV=ren, c_alpha=abs(E['c_alpha_solve_vs_GF']), c_MWMZ_conv=abs(E['c_MWMZ_PDG_as_pole']),
           c_impl_SMDR_vs_mr=abs(E['c_SMDR_minus_mr_MWpred(implementation)']), c_input_set=abs(E['c_input_set(mr MW_pred - mr MW_meas)']))
std['quadrature'] = math.sqrt(sum(v ** 2 for v in std.values()))
cons = dict(std); cons.pop('quadrature'); cons['b_pole_full_4loop_term'] = abs(E['b_pole_series_4loop_term(no_x04)'])
cons['a_mu0_maxdev'] = E['a_mu0_maxdev']; cons['quadrature'] = math.sqrt(sum(v ** 2 for v in cons.values()))
Y = B['errors']['yt']
std_yt = dict(a_mu0_maxdev=Y['a_mu0_maxdev'], b_rge_4l5l=abs(Y['b_rge_4l5l_total(3l-only minus central)']), b_as_order=abs(Y['b_as_5l_run_4l_dec']),
              b_pole_renormalon_0p11GeV=0.11 * abs(B['derivatives_at_mub']['dyt_dMt']), c_alpha=abs(Y['c_alpha_solve_vs_GF']),
              c_MWMZ_conv=abs(Y['c_MWMZ_PDG_as_pole']), c_impl_SMDR_vs_mr=abs(Y['c_SMDR_minus_mr_MWpred(implementation)']))
std_yt['quadrature'] = math.sqrt(sum(v ** 2 for v in std_yt.values()))
B['tadpole_convention_test'] = tad
B['mu0_asymmetric'] = asym
B['error_budget_lambda_mub_standard'] = std
B['error_budget_lambda_mub_standard_Mt_equiv_GeV'] = {k: v / dl for k, v in std.items()}
B['error_budget_lambda_mub_conservative'] = cons
B['error_budget_lambda_mub_conservative_Mt_equiv_GeV'] = {k: v / dl for k, v in cons.items()}
B['error_budget_yt_mub_standard'] = std_yt
B['notes'] = [
    'mr must be run single-threaded: with OMP_NUM_THREADS=4 its OpenMP sections give race-corrupted TSIL integrals (W_y20 3472 vs 4728).',
    'SMDR_Fit_Inputs output at Q_target != M_t is obtained by RG running from SMDR-internal scales, so it provides no matching-scale variation.',
    'QCD-only part of y_t shows +-(3-4)e-4 relative mu0-dependence after 4-loop pure-QCD back-running: consistent with the O(alpha_s^5) '
    'renormalon-dominated truncation of the pole-mass series (~60-70 MeV in M_t), not an implementation error (RG structure of mr y02..y04 checked).',
    'Buttazzo et al. g3(M_t)=1.1666 is ~+0.0018 above standard 4-loop running + 3-loop decoupling (own 1.1648; SMDR consistent with own).']
B['adopted_error_lambda_mub'] = std
B['adopted_error_lambda_Mt_equiv_GeV'] = B['error_budget_lambda_mub_standard_Mt_equiv_GeV']
B['timestamp_utc_post'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
B['code_sha256']['work/post.py'] = hashlib.sha256(open(os.path.join(WORK, 'post.py'), 'rb').read()).hexdigest()
B['code_sha256']['work/diag_mu0.py'] = hashlib.sha256(open(os.path.join(WORK, 'diag_mu0.py'), 'rb').read()).hexdigest()
json.dump(B, open(os.path.join(RES, 'BLIND_RESULTS.json'), 'w'), indent=1, default=float)
# (iv) summary
f = lambda d: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
print('one-loop convention check:', {k: (round(v[0], 4), round(v[1], 4)) for k, v in chk.items()})
print('tadpole sizes:', f(tad['tadpole_sizes']), ' ratio variant d(lam,yt)@mub:', f(tad['delta_vs_central_at_mub']))
print('mu0 asym lam:', f({k: asym['lam'][k] for k in ('plus', 'minus')}), ' yt:', f({k: asym['yt'][k] for k in ('plus', 'minus')}))
print('lam(mub) by mu0:', f(asym['lam']['values']))
print('std budget:', f(std), '\n  Mt-equiv:', f(B['error_budget_lambda_mub_standard_Mt_equiv_GeV']))
print('cons quadrature %.6f (%.3f GeV)' % (cons['quadrature'], cons['quadrature'] / dl), ' yt std:', f(std_yt))
print('central @mub:', f(B['central_at_mub']))
print('central @Mt :', f(B['central_at_Mt']))
print('derivs:', f(B['derivatives_at_mub']))
print('nodes:', [(n['u'], '%.4e' % n['mu'], round(n['lam'], 6), round(n['yt'], 5)) for n in B['nodes_central']])
print('SMDR:', f(B['smdr_central']['at_Mt']), ' MW_pred', B['smdr_central']['MW_PDG_pred'], ' own-RG lam', B['smdr_central']['smdr_own_RG_mub']['lam'])
print('as_5l4d g3(Mt) %.6f vs central %.6f; MW/MZ PDG-as-pole g2 %.6f gY %.6f' % (runs['as_5l4d']['at_mu0']['g3'], C['at_mu0']['g3'],
      runs['MWMZ_PDG_as_pole']['at_mu0']['g2'], runs['MWMZ_PDG_as_pole']['at_mu0']['gY']))
print('MPbar:', f(B['central_at_MPbar']), ' MPl:', f(B['central_at_MPl']))
print('grid lam(mub) as=0.1180:', [(g['Mt'], round(g['lam_mub_as0.1180'], 6)) for g in B['grid']])
print('mtbar grid:', [(g['Mt'], round(g['mtbar_mtbar'], 4)) for g in B['grid']])
