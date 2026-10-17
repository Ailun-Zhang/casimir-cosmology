# sm-blind diagnosis phase (after unblinding): assemble results/DIAGNOSIS.json, write report section 8, update STATUS.
import os, sys, math, json, re, hashlib, datetime
os.environ['OMP_NUM_THREADS'] = '1'
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
sys.stdout.reconfigure(encoding='utf-8')
import smcore as S
MY = os.path.dirname(WORK); RES = os.path.join(MY, 'results')
ROOT = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology'
PFITS = ROOT + r'\04_calculations\results\20261014_claude_v12_r3\sm_mapping\run02\fits'
J = lambda f: json.load(open(os.path.join(RES, f), encoding='utf-8'))
B, runs, recs = J('BLIND_RESULTS.json'), J('prod_runs.json'), J('prod_mr_records.json')
DS, DM = J('diag_smdr_scale.json'), J('diag_mr_decomp.json')
Sl = B['derivatives_at_mub']['dlam_dMt']; aS = abs(Sl)
mub = B['conventions']['mu_b_GeV']; Mt = 172.60
R = S.RGE(3, 'full', True)

# --- (A) mr FJ "x-route": couplings = FJ running masses (tadpoles incl.) / FJ v^2, not re-expanded; scale scan
def xroute(rec):
    aQ = rec['as_mu'] / (4 * math.pi); aE = rec['alpha_GF'] / (4 * math.pi)
    ser = lambda p, q=False: 1 + aE * rec[p + '_x10'] + aE * aQ * rec[p + '_x11'] + aE ** 2 * rec[p + '_x20'] + (
        aQ * rec[p + '_x01'] + aQ ** 2 * rec[p + '_x02'] + aQ ** 3 * rec[p + '_x03'] + aQ ** 4 * rec[p + '_x04'] if q else 0)
    dR = 1 + aE * rec['dr10'] + aE * aQ * rec['dr11'] + aE ** 2 * rec['dr20']; G = rec['Gf']
    c = dict(lam=G / math.sqrt(2) * rec['MH'] ** 2 * ser('H') / dR, yt=2 ** 0.75 * math.sqrt(G) * rec['Mt'] * ser('t', True) / math.sqrt(dR),
             g2=math.sqrt(2 ** 2.5 * G * rec['MW'] ** 2 * ser('W') / dR), g3=math.sqrt(4 * math.pi * rec['as_mu']), yb=rec['yb_own'], ytau=0.009995)
    c['gY'] = math.sqrt(2 ** 2.5 * G * (rec['MZ'] ** 2 * ser('Z') - rec['MW'] ** 2 * ser('W')) / dR); c['g1'] = math.sqrt(5 / 3) * c['gY']
    return c
xr = {}
lx0 = R.run(xroute(recs['central']), Mt, [mub])[mub]['lam']
for k in ('mu0_0.5', 'mu0_0.707', 'mu0_1.414', 'mu0_2'):
    rec = recs[k]; l = R.run(xroute(rec), rec['mu'], [mub])[mub]['lam']
    xr[k] = dict(dlam_b=l - lx0, dmc=(l - lx0) / aS)
xr['central_minus_yroute_dlam_b'] = lx0 - B['central_at_mub']['lam']

# --- (B) producer's own L2.5 single-scale fits at Mt=172.60 (read, not recomputed)
def parse(fn):
    d = {}
    for ln in open(fn):
        q = ln.split()
        if q and q[0] == 'RG_mub':
            d['RG_mub'] = {q[i]: float(q[i + 1]) for i in range(2, len(q) - 1, 2)}
        elif q:
            d[q[0]] = float(q[1])
    return d
p1 = parse(os.path.join(PFITS, 'fit_xi1_Mt172.60.txt'))
prod_piece = {}
for tag in ('t', 'h', 'Z', 'G', 'd', 'xi'):
    for x in ('0.5', '2'):
        fn = os.path.join(PFITS, 'fit_%s%s_Mt172.60.txt' % (tag, x))
        if os.path.exists(fn):
            d = parse(fn); prod_piece['%s%s' % (tag, x)] = (d['RG_mub']['lambda'] - p1['RG_mub']['lambda']) / aS
own_L2 = {k: v['dmc_GeV'] for k, v in DS['deltas'].items()}
# independent (uncorrelated) envelope of the SMDR evaluation scales (L2.5, producer fits)
pieces = ['t', 'h', 'Z', 'G', 'd']
env_hi = sum(max(prod_piece[p + '0.5'], prod_piece[p + '2'], 0) for p in pieces)
env_lo = sum(min(prod_piece[p + '0.5'], prod_piece[p + '2'], 0) for p in pieces)
qd = math.sqrt(sum(((prod_piece[p + '0.5'] - prod_piece[p + '2']) / 2) ** 2 for p in pieces))
# m-bar_t version: remove the pure-QCD y_t tower (own mr numbers, same O(alpha_s^5) object) from the M_t piece
qt = {x: DM['mu0']['mu0_' + x]['dm_c_GeV']['yt_QCD_tower'] for x in ('0.5', '2')}
mt_nonqcd = {'0.5': prod_piece['t0.5'] - qt['0.5'], '2': prod_piece['t2'] - qt['2']}
pieces_mb = {'t': (mt_nonqcd['0.5'], mt_nonqcd['2'])}
pieces_mb.update({p: (prod_piece[p + '0.5'], prod_piece[p + '2']) for p in ('h', 'Z', 'G', 'd')})
envmb_hi = sum(max(a, b, 0) for a, b in pieces_mb.values()); envmb_lo = sum(min(a, b, 0) for a, b in pieces_mb.values())

# --- (C) recommended budgets (GeV in m_c)
spread = abs(B['errors']['lam']['c_SMDR_minus_mr_MWpred(implementation)']) / aS          # mr vs SMDR at mu0=M_t
lead3 = abs(DS['L2_minus_L25_xi1']['lam_mub']) / aS                                     # size of SMDR leading 3-loop pieces
rge, asord, wz = 0.005, abs(B['errors']['lam']['b_as_5l_run_4l_dec']) / aS, abs(B['errors']['lam']['c_MWMZ_PDG_as_pole']) / aS
a_mb = max(envmb_hi, -envmb_lo); a_pole = max(env_hi, -env_lo)
mbar_tot = math.sqrt(a_mb ** 2 + spread ** 2 + rge ** 2 + asord ** 2 + wz ** 2)
pole_series = 0.075   # 5-loop estimate (producer, geometric); same O(alpha_s^5) object as the QCD-tower scale variation and Beneke's ambiguity
pole_tot = math.sqrt(mbar_tot ** 2 + pole_series ** 2)
pole_tot_alt = math.sqrt(a_pole ** 2 + spread ** 2 + rge ** 2 + asord ** 2 + wz ** 2)   # QCD tower inside the scale envelope instead
PROD = dict(source='main-session message 2026-10-15T00:47Z + D120 (run01/run02)', lam_mub=-0.009774, yt_mub=0.42783, g2_mub=0.52776, gY_mub=0.44277,
            g3_mub=0.53565, dlam_dMt=-0.00656, scale_run02=dict(xi0p5=+0.033, xi2=-0.019), run01_envelope=[-0.088, 0.102],
            total_pole=0.086, total_mbar=0.047, pole_series_5loop=0.075, FJ_vs_SMDR=0.022, BKPV_mu=[-0.36, 0.17], mtbar_at_mc_0p9=162.802)
C, SM = B['central_at_mub'], B['smdr_central']
cmp = {k: dict(blind_mr=C[k2], blind_smdr_own_rge=SM['at_mub'][k2], production=PROD[k3], diff_mr=C[k2] - PROD[k3],
               diff_smdr=SM['at_mub'][k2] - PROD[k3]) for k, k2, k3 in
       (('lambda', 'lam', 'lam_mub'), ('yt', 'yt', 'yt_mub'), ('g2', 'g2', 'g2_mub'), ('gY', 'gY', 'gY_mub'), ('g3', 'g3', 'g3_mub'))}
mu = DM['mu0']
D = dict(task='R16 V1 diagnosis (after unblinding)', agent='sm-blind', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
         sign_convention='dm_c = dlam_b/|S| (producer convention; + = critical m_t up); the blind report table used M_t-equivalent = -dm_c',
         comparison_central=cmp, comparison_dlam_dMt=dict(blind=Sl, production=PROD['dlam_dMt']),
         reproduction_of_producer_run02=dict(own_compiled_copy_minus_producer_xi1=DS['own_minus_producer_xi1'],
                                             dlam_b_xi0p5=dict(own=DS['deltas']['L25_xi0.5']['dlam_mub'], producer=DS.get('producer_dlam_mub_xi0.5')),
                                             dlam_b_xi2=dict(own=DS['deltas']['L25_xi2']['dlam_mub'], producer=DS.get('producer_dlam_mub_xi2'))),
         item1_tadpole=dict(mr_y_route_is_tadpole_free_identically=B['tadpole_convention_test']['one_loop_convention_check'],
                            mr_has_no_tadpole_free_vev_option=True, mr_FJ_x_route_scale_scan=xr,
                            BKPV_mu_reproduced_by_blind_mr=dict(blind=[mu['mu0_2']['dm_c_GeV']['full'], mu['mu0_0.5']['dm_c_GeV']['full']], BKPV=[-0.36, 0.17])),
         item2_scales=dict(mr_decomposition_dmc=dict({k: mu[k]['dm_c_GeV'] for k in mu}),
                           smdr_L25_single_scale_producer_fits_172p6=prod_piece, smdr_L2_own_fits=own_L2,
                           smdr_L25_independent_envelope=[env_lo, env_hi], smdr_L25_quadrature_halfranges=qd,
                           smdr_L25_minus_L2_at_xi1_dmc=DS['L2_minus_L25_xi1']['lam_mub'] / aS * -1),
         item3_qcd=dict(rg_consistency_mr_y0i=B['rg_check_mr_qcd_coeffs']['max_rel_dev_by_order'],
                        qcd_tower_dmc=dict(xi0p5=qt['0.5'], xi2=qt['2']), producer_qcd_tower=dict(xi0p5=0.064, xi2=-0.057),
                        smdr_Mt_piece=dict(xi0p5=prod_piece['t0.5'], xi2=prod_piece['t2'])),
         item4_renormalon=dict(Beneke2016='arXiv:1605.03609: abstract "uncertainty is around 110 MeV" (incl. b,c mass effects); "five-loop correction and beyond ... around 300 MeV"; '
                               'Eq.(4.7) delta(5+)m_P = 0.272 +0.016-0.041(N) +-0.001(c4) +-0.011(alpha_s) +-0.066(ambiguity) GeV; p.13 "ambiguity of about 70 MeV" (massless light quarks)',
                               blind_assumption_GeV=0.11, verdict='0.11 = abstract value incl. b,c; 0.066-0.071 for massless light quarks; 0.27 is a one-sided remainder (all-orders pole mass higher), not a symmetric error'),
         known_missing_3loop_in_mr=dict(lambda3_QCD_Buttazzo_fn5_dmc=DM['lambda3_QCD_Buttazzo_fn5']['dm_c_GeV']),
         recommended=dict(mbar=dict(matching_scale_envelope=[envmb_lo, envmb_hi], scheme_spread_mr_vs_smdr=spread, rge=rge, alpha_s_order=asord,
                                    WZ_pole_convention=wz, total=mbar_tot, leading3loop_size_info=lead3),
                          pole_4loop_truncated=dict(mbar_total=mbar_tot, pole_series_O_as5=pole_series, total=pole_tot,
                                                    alt_total_with_QCD_tower_in_envelope=pole_tot_alt),
                          pole_all_orders_offset_GeV=+0.272, renormalon_ambiguity_GeV=[0.07, 0.11]))
json.dump(D, open(os.path.join(RES, 'DIAGNOSIS.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(dict(xr=xr, prod_piece=prod_piece, env=[env_lo, env_hi], qd=qd, envmb=[envmb_lo, envmb_hi], mt_nonqcd=mt_nonqcd,
                      spread=spread, lead3=lead3, mbar_tot=mbar_tot, pole_tot=pole_tot, pole_tot_alt=pole_tot_alt,
                      cmp={k: (round(v['diff_mr'], 6), round(v['diff_smdr'], 6)) for k, v in cmp.items()}), indent=0))
