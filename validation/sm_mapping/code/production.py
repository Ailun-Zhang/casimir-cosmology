# sm-blind R16 V1 production: PDG-2026 inputs -> MSbar couplings at mu0 (mr two-loop matching [A], SMDR fit [B])
# -> own SM RGE (3-loop BPV + known 4-loop + 5-loop QCD) -> mu_b = Mbar_P/722 and hazard nodes.
# Writes results/prod_*.json checkpoints and results/BLIND_RESULTS.json.
import os, sys, math, json, time, hashlib, datetime, subprocess
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import smcore as S

MY = os.path.dirname(WORK); RES = os.path.join(MY, 'results')
P = S.PDG26
GF = P['GF']; MZas = P['MZ_PDG']
MWp, MZp = S.bw_to_pole(P['MW_PDG'], P['GW']), S.bw_to_pole(P['MZ_PDG'], P['GZ'])
mub, US, NODES = S.mu_b_and_nodes()
MPl = 1.2209e19; MPbar = 2.435e18
T0 = time.time()
log = lambda *a: print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

CENT = dict(Mt=172.60, asMZ=0.1180, Mh=125.13, f0=1.0, MW=MWp, MZ=MZp, as_cfg=(4, 3))
RGE_C = S.RGE(3, 'full', True)


def point(cfg):
    Mt, mu0 = cfg['Mt'], cfg['f0'] * cfg['Mt']
    a6 = S.as6(mu0, cfg['asMZ'], Mt, MZas, cfg['as_cfg'][0], cfg['as_cfg'][1], mu_dec=mu0)
    yb = 2 ** 0.75 * math.sqrt(GF) * S.mq_run(P['mbmb'], P['mbmb'], mu0, cfg['asMZ'], 5, 4, MZas)
    return dict(Mb=4.78, MW=cfg['MW'], MZ=cfg['MZ'], MH=cfg['Mh'], Mt=Mt, Gf=GF, as_mu=a6, mu=mu0), yb


def run_all(c0, mu0, rge=RGE_C):
    mus = [mub, MPbar, MPl] + NODES + ([172.60] if abs(mu0 - 172.60) > 1e-9 else [])
    r = rge.run(c0, mu0, sorted(set(mus)))
    keep = ('lam', 'yt', 'g1', 'gY', 'g2', 'g3', 'yb', 'ytau')
    return dict(mu0=mu0, at_mu0={k: c0[k] for k in keep if k in c0},
                at_Mt172p6=({k: r[172.60][k] for k in keep} if 172.60 in r else {k: c0[k] for k in keep if k in c0}),
                at_mub={k: r[mub][k] for k in keep}, at_MPbar={k: r[MPbar][k] for k in keep}, at_MPl={k: r[MPl][k] for k in keep},
                nodes=[dict(u=u, mu=m, **{k: r[m][k] for k in ('lam', 'yt', 'g1', 'gY', 'g2', 'g3')}) for u, m in zip(US, NODES)])


# ---------------- 1. mr points ----------------
cfgs = {}
cfgs['central'] = dict(CENT)
for Mt in [round(171.8 + 0.2 * i, 2) for i in range(11)]:
    for a in (0.1171, 0.1180, 0.1189):
        cfgs['grid_Mt%.2f_as%.4f' % (Mt, a)] = dict(CENT, Mt=Mt, asMZ=a)
for s in (-1, 1):
    cfgs['Mh%+d' % s] = dict(CENT, Mh=125.13 + 0.11 * s)
    cfgs['MW%+d' % s] = dict(CENT, MW=S.bw_to_pole(P['MW_PDG'] + 0.0077 * s, P['GW']))
for f0, nm in ((0.5, 'mu0_0.5'), (2 ** -0.5, 'mu0_0.707'), (2 ** 0.5, 'mu0_1.414'), (2.0, 'mu0_2')):
    cfgs[nm] = dict(CENT, f0=f0)
cfgs['as_5l4d'] = dict(CENT, as_cfg=(5, 4))
cfgs['MWMZ_PDG_as_pole'] = dict(CENT, MW=P['MW_PDG'], MZ=P['MZ_PDG'])
ckpt = os.path.join(RES, 'prod_mr_records.json')
recs = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
todo = [k for k in cfgs if k not in recs]
if todo:
    pts = [point(cfgs[k]) for k in todo]
    log('mr: evaluating %d points with 4 single-threaded processes' % len(todo))
    out = S.mr_eval_par([p for p, _ in pts], 4)
    for k, (p, yb), r in zip(todo, pts, out):
        r['yb_own'] = yb
        recs[k] = r
    json.dump(recs, open(ckpt, 'w'), indent=0)
log('mr done')

res = {}
for k, cfg in cfgs.items():
    rec = recs[k]
    c0 = S.mr_couplings(rec, 'GF', S.ALL_ORDERS, yb=rec['yb_own'])
    res[k] = run_all(c0, rec['mu'])
    res[k]['as6_mu0'] = rec['as_mu']; res[k]['alpha_GF_inv'] = 1 / rec['alpha_GF']; res[k]['alpha_solve_inv'] = 1 / rec['alpha_solve']
# scheme / truncation recombinations at central point (no new mr call)
rc = recs['central']
var = {}
var['alpha_solve'] = S.mr_couplings(rc, 'SV', S.ALL_ORDERS, yb=rc['yb_own'])
var['no_x04'] = S.mr_couplings(rc, 'GF', ('x01', 'x02', 'x03', 'x10', 'x11', 'x20'), yb=rc['yb_own'])
var['no_x03_x04'] = S.mr_couplings(rc, 'GF', ('x01', 'x02', 'x10', 'x11', 'x20'), yb=rc['yb_own'])
var['no_x20'] = S.mr_couplings(rc, 'GF', ('x01', 'x02', 'x03', 'x04', 'x10', 'x11'), yb=rc['yb_own'])
var['no_x11_x20'] = S.mr_couplings(rc, 'GF', ('x01', 'x02', 'x03', 'x04', 'x10'), yb=rc['yb_own'])
for k, c0 in var.items():
    res['var_' + k] = run_all(c0, rc['mu'])
c_cent = S.mr_couplings(rc, 'GF', S.ALL_ORDERS, yb=rc['yb_own'])
for nm, R in (('rge_2loop', S.RGE(2, 'none', False)), ('rge_3loop', S.RGE(3, 'none', False)),
              ('rge_3loop_4qcd', S.RGE(3, 'qcd', False)), ('rge_3loop_4full', S.RGE(3, 'full', False))):
    res[nm] = run_all(c_cent, rc['mu'], R)
log('mr-based RG runs done')

# ---------------- 2. SMDR fits (matching B) ----------------
GF_smdr = GF * (1 - 0.3 * (0.1056583755 / P['MW_PDG']) ** 2)  # SMDR convention (smdr_pdg_2025.h comment)
def smdr_line(Q, Mt, asMZ, Mh):
    return 'fit %.10g %.10g %.10g %.10g %.10g %.17g %.12g %.6g %.6g 0.0929 0.00470 0.00216 %.8g 0.1056583755 0.00051099895 0.02783' % (
        Q, Mt, Mh, P['MZ_PDG'], asMZ, 1 / 137.035999177, GF_smdr, P['mbmb'], P['mcmc'], P['Mtau'])
sm_jobs = {'smdr_central': (172.60, 172.60, 0.1180), 'smdr_Mt-0.2': (172.40, 172.40, 0.1180), 'smdr_Mt+0.2': (172.80, 172.80, 0.1180),
           'smdr_Q0.5': (86.30, 172.60, 0.1180), 'smdr_Q2': (345.20, 172.60, 0.1180)}
ck2 = os.path.join(RES, 'prod_smdr_records.json')
smr = json.load(open(ck2)) if os.path.exists(ck2) else {}
todo = [k for k in sm_jobs if k not in smr]
if todo:
    log('SMDR: %d fits in parallel' % len(todo))
    procs = {}
    for k in todo:
        Q, Mt, a = sm_jobs[k]
        cmds = [smdr_line(Q, Mt, a, 125.13), 'run %.10g 5' % mub]
        pr = subprocess.Popen([S.SMDR_EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=S._env(1), cwd=S.BUILD)
        procs[k] = (pr, '\n'.join(cmds) + '\n')
        if len(procs) >= 4:
            for kk, (p_, inp) in list(procs.items()):
                so, se = p_.communicate(inp); smr[kk] = [json.loads(l) for l in so.splitlines() if l.startswith('{')]
            procs = {}
    for kk, (p_, inp) in procs.items():
        so, se = p_.communicate(inp); smr[kk] = [json.loads(l) for l in so.splitlines() if l.startswith('{')]
    json.dump(smr, open(ck2, 'w'), indent=0)
log('SMDR done')
for k, v in smr.items():
    f, rr = v[0], v[1]
    Q = f['Q_in']
    if abs(f['Q'] - Q) > 1e-9 or abs(f['g3'] - f['g3_in']) > 1e-12 or abs(f['yt'] - f['yt_in']) > 1e-12:
        log('WARNING SMDR running != _in values for', k, f['Q'], Q, f['g3'], f['g3_in'])
    c0 = dict(lam=f['lambda_in'], yt=f['yt_in'], g2=f['g'], gY=f['gp'], g3=f['g3_in'], yb=f['yb'], ytau=f['ytau'])
    # NB after 'fit' the running values equal the *_in values at Q_in (checked: g3 == g3_in)
    c0['g1'] = math.sqrt(5 / 3) * c0['gY']
    res[k] = run_all(c0, Q)
    res[k]['smdr_own_RG_mub'] = dict(lam=rr['lambda'], yt=rr['yt'], g2=rr['g'], gY=rr['gp'], g3=rr['g3'], Q=rr['Q'])
    res[k]['smdr_MW_PDG_pred'] = f['MW_PDG']; res[k]['smdr_MW_pole'] = f['MW_pole']
# mr with SMDR-predicted MW (isolates input-set effect)
MW_pred = smr['smdr_central'][0]['MW_PDG']
k = 'mr_MW_SMDRpred'
if k not in recs:
    p, yb = point(dict(CENT, MW=S.bw_to_pole(MW_pred, P['GW'])))
    r = S.mr_eval([p])[0]; r['yb_own'] = yb; recs[k] = r
    json.dump(recs, open(ckpt, 'w'), indent=0)
res[k] = run_all(S.mr_couplings(recs[k], 'GF', S.ALL_ORDERS, yb=recs[k]['yb_own']), recs[k]['mu'])
log('all RG runs done')
json.dump(res, open(os.path.join(RES, 'prod_runs.json'), 'w'), indent=0)

# ---------------- 3. derived quantities ----------------
C = res['central']
lamb, ytb = C['at_mub']['lam'], C['at_mub']['yt']
Mts = [round(171.8 + 0.2 * i, 2) for i in range(11)]
def series(q, a, key):
    return np.array([res['grid_Mt%.2f_as%.4f' % (m, a)]['at_mub'][key] for m in Mts])
der = {}
for key in ('lam', 'yt', 'g2', 'gY', 'g3'):
    y = series(None, 0.1180, key)
    A = np.polyfit(np.array(Mts) - 172.60, y, 3)
    der['d%s_dMt' % key] = A[-2]
    der['d2%s_dMt2' % key] = 2 * A[-3]
    der['d%s_das' % key] = (res['grid_Mt172.60_as0.1189']['at_mub'][key] - res['grid_Mt172.60_as0.1171']['at_mub'][key]) / 0.0018
    der['d%s_dMh' % key] = (res['Mh+1']['at_mub'][key] - res['Mh-1']['at_mub'][key]) / 0.22
    der['d%s_dMW' % key] = (res['MW+1']['at_mub'][key] - res['MW-1']['at_mub'][key]) / (2 * 0.0077)
dl = der['dlam_dMt']
grid_tab = [dict(Mt=m, **{('lam_mub_as%.4f' % a): res['grid_Mt%.2f_as%.4f' % (m, a)]['at_mub']['lam'] for a in (0.1171, 0.1180, 0.1189)},
                 **{('yt_mub_as%.4f' % a): res['grid_Mt%.2f_as%.4f' % (m, a)]['at_mub']['yt'] for a in (0.1171, 0.1180, 0.1189)},
                 lam_Mt=res['grid_Mt%.2f_as0.1180' % m]['at_mu0']['lam'], yt_Mt=res['grid_Mt%.2f_as0.1180' % m]['at_mu0']['yt'],
                 mtbar_mtbar=S.mt_msbar(m, 0.1180, MZas, 4)) for m in Mts]
# mbar_t table extras
mtb = dict(central_4l=S.mt_msbar(172.60, 0.1180, MZas, 4), central_3l=S.mt_msbar(172.60, 0.1180, MZas, 3),
           central_2l=S.mt_msbar(172.60, 0.1180, MZas, 2), central_1l=S.mt_msbar(172.60, 0.1180, MZas, 1),
           c4_plus=S.mt_msbar(172.60, 0.1180, MZas, 4, c4=8.74 * math.pi ** 4), c4_minus=S.mt_msbar(172.60, 0.1180, MZas, 4, c4=8.24 * math.pi ** 4),
           as_0p1171=S.mt_msbar(172.60, 0.1171, MZas, 4), as_0p1189=S.mt_msbar(172.60, 0.1189, MZas, 4),
           as_5l4d=S.mt_msbar(172.60, 0.1180, MZas, 4, 5, 4))

def dd(k, q='lam', where='at_mub'):
    return res[k][where][q] - C[where][q]
mu0s = ['mu0_0.5', 'mu0_0.707', 'central', 'mu0_1.414', 'mu0_2']
err = {}
for q in ('lam', 'yt'):
    e = {}
    vals = [res[k]['at_mub'][q] for k in mu0s]
    e['a_mu0_scan'] = {k: res[k]['at_mub'][q] for k in mu0s}
    e['a_mu0_halfrange'] = (max(vals) - min(vals)) / 2
    e['a_mu0_maxdev'] = max(abs(v - C['at_mub'][q]) for v in vals)
    e['b_rge_4l5l_total(3l-only minus central)'] = dd('rge_3loop', q)
    e['b_rge_5loop_only'] = dd('rge_3loop_4full', q)
    e['b_rge_4l_nonQCD(full vs qcd-only)'] = res['rge_3loop_4full']['at_mub'][q] - res['rge_3loop_4qcd']['at_mub'][q]
    e['b_rge_2loop_vs_central'] = dd('rge_2loop', q)
    e['b_as_5l_run_4l_dec'] = dd('as_5l4d', q)
    e['b_pole_series_4loop_term(no_x04)'] = dd('var_no_x04', q)
    e['b_pole_series_3l4l(no_x03_x04)'] = dd('var_no_x03_x04', q)
    e['b_EW2loop_size(no_x20)'] = dd('var_no_x20', q)
    e['b_mixed_EW2loop_size(no_x11_x20)'] = dd('var_no_x11_x20', q)
    e['c_alpha_solve_vs_GF'] = dd('var_alpha_solve', q)
    e['c_MWMZ_PDG_as_pole'] = dd('MWMZ_PDG_as_pole', q)
    e['c_SMDR_minus_mr'] = res['smdr_central']['at_mub'][q] - C['at_mub'][q]
    e['c_SMDR_minus_mr_MWpred(implementation)'] = res['smdr_central']['at_mub'][q] - res['mr_MW_SMDRpred']['at_mub'][q]
    e['c_input_set(mr MW_pred - mr MW_meas)'] = res['mr_MW_SMDRpred']['at_mub'][q] - C['at_mub'][q]
    e['c_SMDR_scale_Q0.5'] = res['smdr_Q0.5']['at_mub'][q] - res['smdr_central']['at_mub'][q]
    e['c_SMDR_scale_Q2'] = res['smdr_Q2']['at_mub'][q] - res['smdr_central']['at_mub'][q]
    err[q] = e
# combined (adopted) theory error on lambda(mu_b): quadrature of a, b(RGE 4/5l total, as, pole 4l term), c(alpha, MW/MZ conv, A-B impl)
L = err['lam']
adopted = dict(a=L['a_mu0_maxdev'], b_rge=abs(L['b_rge_4l5l_total(3l-only minus central)']), b_as=abs(L['b_as_5l_run_4l_dec']),
               b_pole=abs(L['b_pole_series_4loop_term(no_x04)']), c_alpha=abs(L['c_alpha_solve_vs_GF']),
               c_MWMZ=abs(L['c_MWMZ_PDG_as_pole']), c_impl=abs(L['c_SMDR_minus_mr_MWpred(implementation)']))
adopted['quadrature'] = math.sqrt(sum(v ** 2 for v in adopted.values()))
adopted_Mt = {k: v / abs(dl) for k, v in adopted.items()}
renormalon = dict(assumed_Mt_ambiguity_GeV=0.11, dlam_mub=0.11 * abs(dl), status='conditional: literature-range value, not re-derived here')

# ---------------- 4. hashes and BLIND_RESULTS ----------------
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
code = {os.path.relpath(p, MY): sha(p) for p in [os.path.join(WORK, f) for f in
        ('smcore.py', 'production.py', 'rge_bpv3_gen.py', 'rge_convert_check.py', 'test_core.py', 'build_all.py',
         'drivers/mr_driver.cpp', 'drivers/smdr_driver.c', 'build/mr_driver.exe', 'build/smdr_driver.exe')]}
srcs = {}
for d in os.listdir(os.path.join(MY, 'inputs')):
    sj = os.path.join(MY, 'inputs', d, 'SOURCES.json')
    if os.path.exists(sj):
        for r in json.load(open(sj)):
            if 'sha256' in r:
                srcs[d + '/' + r['file']] = dict(sha256=r['sha256'], url=r.get('url'))
val = {}
for f in ('validation_T1_T2_B1.json', 'validation_B1_B2_B3.json'):
    p = os.path.join(RES, f)
    if os.path.exists(p):
        val[f] = json.load(open(p))
B = dict(
    task='R16 V1 sm_mapping_blind', agent='sm-blind', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    conventions=dict(lambda_norm='V = lambda(|H|^2 - v^2/2)^2 (m_h^2 = 2 lambda v^2)', gY='hypercharge g\' (non-GUT); g1 = sqrt(5/3) gY',
                     scheme='MSbar, full SM (6 flavours) at mu >= mu0', mu_b='Mbar_P/722 with Mbar_P = 2.435e18 GeV', mu_b_GeV=mub,
                     nodes='mu(u) = mu_b/sqrt(cosh 2u)'),
    inputs=dict(P, MW_pole=MWp, MZ_pole=MZp, pole_convention='Mbar = M_PDG/sqrt(1+Gamma^2/M^2) (s_P = Mbar^2 - i Mbar Gbar)', Mb_pole_in_loops=4.78,
                ytau_mu0=0.009995, note_GF='PDG 2026 phys-constants table: 1.1663785(6)e-5'),
    method=dict(matching_A='mr (KPV) GitHub master, full 2-loop EW+mixed, QCD to 4 loops (y_t), alpha(mu) from G_F (AlphaGF); run single-threaded',
                matching_B='SMDR 1.3 SMDR_Fit_Inputs (G_F, alpha, M_Z input set; M_W predicted)',
                alpha_s='4-loop running (nf=5) from M_Z, 3-loop decoupling at mu0 with on-shell M_t, 4-loop (nf=6) running to mu0',
                rge='3-loop full SM (BPV, all of y_t,y_b,y_tau) + 4-loop (g3 full SM, g2,gY [1912.07624], y_t O(g3^8), lambda O(yt^4 g3^6)) + 5-loop pure QCD (g3, y_t)'),
    central_at_mub=C['at_mub'], central_at_Mt=C['at_mu0'], central_at_MPbar=C['at_MPbar'], central_at_MPl=C['at_MPl'],
    derivatives_at_mub=der, nodes_central=C['nodes'], grid=grid_tab, mtbar=mtb,
    errors=err, adopted_error_lambda_mub=adopted, adopted_error_lambda_Mt_equiv_GeV=adopted_Mt, renormalon=renormalon,
    smdr_central=dict(at_Mt=res['smdr_central']['at_mu0'], at_mub=res['smdr_central']['at_mub'],
                      smdr_own_RG_mub=res['smdr_central']['smdr_own_RG_mub'], MW_PDG_pred=res['smdr_central']['smdr_MW_PDG_pred']),
    benchmarks_file=list(val.keys()), code_sha256=code, inputs_sha256=srcs)
if 'validation_B1_B2_B3.json' in val:
    v = val['validation_B1_B2_B3.json']
    B['benchmarks'] = dict(Buttazzo_weak=dict(own=v['B1']['mr_NNLO_plus_3lQCD'], published=v['B1']['published_Mt_scale']),
                           Buttazzo_Planck_from_their_Mt_values=dict(own=v['B1']['Planck_from_published_Mt_values_4lqcd'], published=v['B1']['published_Planck']),
                           BKPV_Mtcri=dict(own=v['B2']['Mt_cri'], own_log10mu=v['B2']['log10_mu_cri'], published=v['B2']['published']),
                           SMDR_ref2025=dict(own_fit=v['B3']['smdr_fit'], reference=v['B3']['reference']))
json.dump(B, open(os.path.join(RES, 'BLIND_RESULTS.json'), 'w'), indent=1, default=float)
log('lambda(mu_b)=%.6f yt=%.6f g2=%.6f gY=%.6f g3=%.6f | dlam/dMt=%.6f /GeV dlam/das=%.4f' % (
    lamb, ytb, C['at_mub']['g2'], C['at_mub']['gY'], C['at_mub']['g3'], dl, der['dlam_das']))
log('at Mt: lam=%.6f yt=%.6f g2=%.6f gY=%.6f g3=%.6f' % tuple(C['at_mu0'][k] for k in ('lam', 'yt', 'g2', 'gY', 'g3')))
log('SMDR at Mt:', {k: round(v, 6) for k, v in res['smdr_central']['at_mu0'].items()}, ' at mub lam=%.6f yt=%.6f; SMDR own RG lam=%.6f' % (
    res['smdr_central']['at_mub']['lam'], res['smdr_central']['at_mub']['yt'], res['smdr_central']['smdr_own_RG_mub']['lam']))
for q in ('lam', 'yt'):
    log(q, json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in err[q].items() if not isinstance(v, dict)}))
log('adopted', json.dumps({k: round(v, 6) for k, v in adopted.items()}), ' Mt-equiv', json.dumps({k: round(v, 3) for k, v in adopted_Mt.items()}))
log('mtbar', json.dumps({k: round(v, 4) for k, v in mtb.items()}))
