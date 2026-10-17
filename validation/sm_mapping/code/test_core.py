# sm-blind R16 V1: validation + literature benchmarks
#  T1 QCD constants vs SMDR numbers; T2 beta functions vs SMDR_Betas (C) at random points (1..5 loops)
#  B1 Buttazzo et al. 2013 (weak-scale NNLO values + Planck-scale couplings)
#  B2 BKPV 2015 critical top mass (lambda = beta_lambda = 0)
#  B3 SMDR ReferenceModel_2025 (SMDR build check) + mr at equivalent inputs
import os, sys, math, json, time
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import smcore as S
which = sys.argv[1:] or ['T1', 'T2', 'B1', 'B2', 'B3']
RES = os.path.join(os.path.dirname(WORK), 'results')
out = {}
t00 = time.time()

if 'T1' in which:
    b6, g6 = S.bQCD(6), S.gamma_m(6)
    out['T1'] = dict(b3_nf6=b6[3], b4_nf6=b6[4], SMDR_betag35=-271.428, gamma3_nf6=g6[3], SMDR_gamma3=-2308.18 / 512,
                     gamma4_nf6=g6[4], SMDR_gamma4=-19988.8 / 2048, b4_nf5=S.bQCD(5)[4], c_pole=S.C_POLE)
    print('T1', json.dumps(out['T1']))

if 'T2' in which:
    rng = np.random.default_rng(20261014)
    rows = []
    for ip in range(3):
        c = dict(g3=rng.uniform(0.5, 1.2), g2=rng.uniform(0.5, 0.65), gY=rng.uniform(0.35, 0.48), yt=rng.uniform(0.4, 1.0),
                 yb=rng.uniform(0.005, 0.02), ytau=rng.uniform(0.005, 0.012), lam=rng.uniform(-0.02, 0.13))
        x = S.g_to_a(c)
        cmds = ['beta %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d' % (c['g3'], c['g2'], c['gY'], c['yt'], c['yb'], c['ytau'], c['lam'], L)
                for L in (1, 2, 3, 4, 5)]
        sm, err = S.smdr_cmds(cmds)
        for L, r in zip((1, 2, 3, 4, 5), sm):
            R = S.RGE(nloop=min(L, 3), four='full' if L >= 4 else 'none', five=(L >= 5))
            d = R(0.0, x)
            mine = dict(b_g3=d[2] / (S.KAP * c['g3']), b_g=d[1] / (S.KAP * c['g2']), b_gp=d[0] / (5 / 3 * S.KAP * c['gY']),
                        b_yt=d[3] / (S.KAP * c['yt']), b_yb=d[4] / (S.KAP * c['yb']), b_ytau=d[5] / (S.KAP * c['ytau']),
                        b_k=2 * d[6] / S.KAP)
            rel = {k: abs(mine[k] - r[k]) / max(abs(r[k]), 1e-30) for k in mine}
            rows.append(dict(point=ip, loop=L, maxrel=max(rel.values()), rel=rel))
    out['T2'] = rows
    for r in rows:
        print('T2 point %d loop %d max rel diff %.3e  %s' % (r['point'], r['loop'], r['maxrel'],
              ' '.join('%s=%.1e' % (k, v) for k, v in r['rel'].items())))

if 'B1' in which:
    Mt, Mh, MW, MZ, asMZ = 173.34, 125.15, 80.384, 91.1876, 0.1184
    Gf = 1 / (math.sqrt(2) * 246.21971 ** 2)
    a6 = S.as6(Mt, asMZ, Mt, MZ, 4, 3)
    mbMt = S.mq_run(4.2, 4.2, Mt, asMZ, 5, 4, MZ)
    yb = 2 ** 0.75 * math.sqrt(Gf) * mbMt
    rec = S.mr_eval([dict(Mb=4.78, MW=MW, MZ=MZ, MH=Mh, Mt=Mt, Gf=Gf, as_mu=a6, mu=Mt)])[0]
    c3 = S.mr_couplings(rec, 'GF', ('x01', 'x02', 'x03', 'x10', 'x11', 'x20'), yb=yb, ytau=0.0100)
    c4 = S.mr_couplings(rec, 'GF', S.ALL_ORDERS, yb=yb, ytau=0.0100)
    c2 = S.mr_couplings(rec, 'GF', ('x01', 'x02', 'x10', 'x11', 'x20'), yb=yb, ytau=0.0100)
    cSV = S.mr_couplings(rec, 'SV', ('x01', 'x02', 'x03', 'x10', 'x11', 'x20'), yb=yb, ytau=0.0100)
    cNLO = S.mr_couplings(rec, 'GF', ('x01', 'x10'), yb=yb, ytau=0.0100)
    cLO = S.mr_couplings(rec, 'GF', (), yb=yb, ytau=0.0100)
    print('B1 LO  mr:', {k: round(cLO[k], 5) for k in ('lam', 'yt', 'g2', 'gY')}, ' pub LO  {lam:0.12917, yt:0.99561, g2:0.65294, gY:0.34972}')
    print('B1 NLO mr:', {k: round(cNLO[k], 5) for k in ('lam', 'yt', 'g2', 'gY')}, ' pub NLO {lam:0.12774, yt:0.95113, g2:0.64754, gY:0.35940}')
    print('B1 alpha_GF(Mt)^-1 = %.4f, alpha_solve^-1 = %.4f' % (4 * math.pi / (4 * math.pi * rec['alpha_GF']), 1 / rec['alpha_solve']))
    # same with complex-pole-converted MW, MZ (Mbar convention)
    rec2 = S.mr_eval([dict(Mb=4.78, MW=S.bw_to_pole(MW, 2.085), MZ=S.bw_to_pole(MZ, 2.4952), MH=Mh, Mt=Mt, Gf=Gf, as_mu=a6, mu=Mt)])[0]
    cP = S.mr_couplings(rec2, 'GF', ('x01', 'x02', 'x03', 'x10', 'x11', 'x20'), yb=yb, ytau=0.0100)
    print('B1 mr with MW,MZ -> pole (Mbar):', {k: round(cP[k], 5) for k in ('lam', 'yt', 'g2', 'gY')})
    pub = dict(lam=0.12604, yt=0.93690, g2=0.64779, gY=0.35830, g3=1.1666)
    b1 = dict(inputs=dict(Mt=Mt, Mh=Mh, MW=MW, MZ=MZ, asMZ=asMZ, Gf=Gf), published_Mt_scale=pub,
              published_NNLO_table=dict(lam=0.12604, yt=0.94018, g2=0.64779, gY=0.35830),
              mr_NNLO_plus_3lQCD=c3, mr_NNLO_2lQCD=c2, mr_all_incl_4lQCD=c4, mr_alphaSolve=cSV,
              P2MS_check=dict(GF_lam=rec['GF_lam'], GF_yt=rec['GF_yt'], GF_g2=rec['GF_g2'], GF_g1=rec['GF_g1']))
    # Planck scale: published Eq. (sys:Pl) at M_Pl = 1.22e19 GeV (non-reduced)
    pubPl = dict(g1=0.6154, g2=0.5055, g3=0.4873, yt=0.3825, lam=-0.0143)
    MPl = 1.2209e19
    for tag, c0 in [('from_published_Mt_values', dict(pub, yb=yb, ytau=0.0100)), ('from_mr_matching', c3)]:
        c0 = dict(c0)
        for four in ('none', 'qcd', 'full'):
            r = S.RGE(3, four, four == 'full').run(c0, Mt, [MPl])[MPl]
            b1['Planck_%s_4l%s' % (tag, four)] = {k: r[k] for k in ('g1', 'g2', 'g3', 'yt', 'lam')}
    b1['published_Planck'] = pubPl
    out['B1'] = b1
    print('B1 mr (NNLO+3lQCD):', {k: round(c3[k], 5) for k in ('lam', 'yt', 'g2', 'gY', 'g3')}, ' published', pub)
    print('B1 mr 2lQCD yt=%.5f (pub NNLO 0.94018); incl 4lQCD yt=%.5f; alphaSolve lam=%.5f yt=%.5f' % (c2['yt'], c4['yt'], cSV['lam'], cSV['yt']))
    print('B1 P2MS check', b1['P2MS_check'], ' my assembly all orders:', c4['lam'], c4['yt'], c4['g2'], c4['g1'])
    for k, v in b1.items():
        if k.startswith('Planck_'):
            print('B1', k, {kk: round(vv, 5) for kk, vv in v.items()}, ' pub', pubPl)

if 'B2' in which:
    MH, asMZ, MW, MZ, Gf = 125.7, 0.1185, 80.385, 91.1876, 1.1663787e-5
    R = S.RGE(3, 'qcd', False)
    def lmin(Mt):
        a6 = S.as6(Mt, asMZ, Mt, MZ, 4, 3)
        yb = 2 ** 0.75 * math.sqrt(Gf) * S.mq_run(4.18, 4.18, Mt, asMZ, 5, 4, MZ)
        rec = S.mr_eval([dict(Mb=4.78, MW=MW, MZ=MZ, MH=MH, Mt=Mt, Gf=Gf, as_mu=a6, mu=Mt)])[0]
        c = S.mr_couplings(rec, 'GF', S.ALL_ORDERS, yb=yb, ytau=0.0100)
        return R.lam_min(c, Mt, 15.5, 20.0, 91)
    xs = [171.2, 171.7]; fs = [lmin(x)[1] for x in xs]
    for it in range(8):
        x2 = xs[-1] - fs[-1] * (xs[-1] - xs[-2]) / (fs[-1] - fs[-2])
        xs.append(x2); fs.append(lmin(x2)[1])
        if abs(fs[-1]) < 1e-8:
            break
    lm, lv = lmin(xs[-1])
    out['B2'] = dict(inputs=dict(MH=MH, asMZ=asMZ, MW=MW, MZ=MZ, Gf=Gf, matching='mr all orders at mu=Mt',
                                 rge='3-loop + 4-loop pure QCD'), Mt_cri=xs[-1], log10_mu_cri=lm, lam_min=lv,
                     published=dict(Mt_cri=171.44, log10_mu_cri=17.752, source='BKPV PRL115(2015)201802 Table I'))
    print('B2 Mt_cri = %.3f GeV, log10 mu_cri = %.3f (published 171.44, 17.752)' % (xs[-1], lm))

if 'B3' in which:
    ref = dict(Q=172.4, g3=1.1635013016923942, g=0.6475889528084995, gp=0.3585955591554844, lam=0.1262957254536527,
               yt=0.9309669896297980, MW_PDG=80.34900999721387, MZ_pole=91.16195975258292)
    cmd = 'fit 172.4 172.4 125.2 91.1876 0.1180 %.17g 0.000011663788 4.183 1.273 0.0935 0.00470 0.00216 1.77693 0.1056583755 0.00051099895 0.027830' % (1 / 137.035999178)
    t0 = time.time(); sm, err = S.smdr_cmds([cmd]); dt = time.time() - t0
    f = sm[0] if sm else {}
    b3 = dict(reference=ref, smdr_fit=f, seconds=dt, stderr_tail=err[-500:])
    if f:
        b3['diff'] = {k: f[kk] - ref[k] for k, kk in [('g3', 'g3_in'), ('lam', 'lambda_in'), ('yt', 'yt_in')]}
        # mr at the equivalent on-shell inputs (MW from SMDR prediction, PDG-convention -> pole Mbar)
        MWp = S.bw_to_pole(f['MW_PDG'], 2.085); MZp = S.bw_to_pole(91.1876, 2.4952)
        Gf = 1.1663788e-5
        a6 = S.as6(172.4, 0.1180, 172.4, 91.1876, 4, 3)
        yb = 2 ** 0.75 * math.sqrt(Gf) * S.mq_run(4.183, 4.183, 172.4, 0.1180, 5, 4, 91.1876)
        recs = S.mr_eval([dict(Mb=4.78, MW=MWp, MZ=MZp, MH=125.2, Mt=172.4, Gf=Gf, as_mu=a6, mu=172.4),
                          dict(Mb=4.78, MW=f['MW_PDG'], MZ=91.1876, MH=125.2, Mt=172.4, Gf=Gf, as_mu=a6, mu=172.4)])
        b3['mr_pole_inputs'] = S.mr_couplings(recs[0], 'GF', S.ALL_ORDERS, yb=yb)
        b3['mr_PDG_inputs'] = S.mr_couplings(recs[1], 'GF', S.ALL_ORDERS, yb=yb)
        b3['as6_own'] = a6
    out['B3'] = b3
    print('B3 SMDR fit (%.1fs):' % dt, {k: f.get(k) for k in ('g3_in', 'yt_in', 'lambda_in', 'g', 'gp', 'MW_PDG')})
    print('B3 reference      :', ref)
    if f:
        for k in ('mr_pole_inputs', 'mr_PDG_inputs'):
            print('B3', k, {kk: round(b3[k][kk], 6) for kk in ('lam', 'yt', 'g2', 'gY', 'g3')})

tag = '_'.join(which)
json.dump(out, open(os.path.join(RES, 'validation_%s.json' % tag), 'w'), indent=1, default=float)
print('done in %.1fs' % (time.time() - t00))
