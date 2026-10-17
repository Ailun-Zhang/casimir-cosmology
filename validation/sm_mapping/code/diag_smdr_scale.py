# sm-blind diagnosis (after unblinding, 2026-10-15): compiled-SMDR scale variation with and without the leading
# 3-loop pieces (M_h 2.5 -> 2, M_Z/M_W 2.5 -> 2, G_F 3 -> 2), using a PROVENANCE copy of the producer's c148_fit.c
# built against the sm-blind SMDR build. <=4 parallel fits. -> results/diag_smdr_scale.json
import os, sys, re, json, time, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
R = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology'
MY = R + r'\07_reviews\R16_v12_round3\sm_mapping_blind'
SRC_PROD = R + r'\04_calculations\code\20261014_claude_v12_r3\sm_mapping\c148_fit.c'
PROD_FITS = R + r'\04_calculations\results\20261014_claude_v12_r3\sm_mapping\run02\fits'
DRV = MY + r'\work\drivers'; B = MY + r'\work\build'; OUT = MY + r'\results\diag_smdr'
os.makedirs(OUT, exist_ok=True)
SMDRSRC = MY + r'\inputs\20261014_smdr_extracted\SMDR-1.3\src'
TSIL = MY + r'\inputs\20261014_tsil_extracted\TSIL-1.46'
T3 = [os.path.join(MY, 'inputs', '20261014_3vil_extracted', d) for d in os.listdir(MY + r'\inputs\20261014_3vil_extracted') if d.lower().startswith('3vil')][0]
MINGW = r'D:\Python\envs\cbuild\Library\mingw-w64\bin'
ENV = dict(os.environ, PATH=MINGW + ';' + os.environ['PATH'], OMP_NUM_THREADS='1')
# 1. provenance copy with loop orders as optional arguments
src = open(SRC_PROD, encoding='utf-8').read()
rep = [('SMDR_Eval_Mh_pole (Qh, 2.5,', 'SMDR_Eval_Mh_pole (Qh, C148_LH,'),
       ('SMDR_Eval_MZ_pole (QZ, 2.5,', 'SMDR_Eval_MZ_pole (QZ, C148_LZ,'),
       ('SMDR_Eval_MW_pole (QZ, 2.5,', 'SMDR_Eval_MW_pole (QZ, C148_LZ,'),
       ('SMDR_GFermi = SMDR_Eval_GFermi (QG, 3);', 'SMDR_GFermi = SMDR_Eval_GFermi (QG, C148_LG);'),
       ('#include <time.h>', '#include <time.h>\nstatic float C148_LH = 2.5f, C148_LZ = 2.5f, C148_LG = 3.0f;'),
       ('SMDR_REAL tol = strtold(argv[11], 0);', 'SMDR_REAL tol = strtold(argv[11], 0);\n  if (argc > 15) { C148_LH = (float) atof(argv[13]); C148_LZ = (float) atof(argv[14]); C148_LG = (float) atof(argv[15]); }')]
for a, b in rep:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
cp = os.path.join(DRV, 'c148_fit_copy.c')
open(cp, 'w', encoding='utf-8').write('/* PROVENANCE: copied from %s on 2026-10-15 by sm-blind; loop orders of M_h, M_Z/M_W, G_F made optional args 13-15 */\n' % SRC_PROD + src)
exe = os.path.join(B, 'c148_fit_copy.exe')
p = subprocess.run([os.path.join(MINGW, 'gcc.exe'), '-O2', '-DTSIL_SIZE_LONG', '-D__USE_MINGW_ANSI_STDIO=1', '-fcommon', '-I' + SMDRSRC,
                    '-I' + TSIL, '-I' + T3, cp, '-o', exe, '-L' + B, '-lsmdr', '-l3vil', '-ltsil146', '-lm', '-Wl,--stack,67108864'],
                   env=ENV, capture_output=True, text=True)
print('compile rc', p.returncode, p.stderr[-1500:])
assert p.returncode == 0
# 2. fits (Mt=172.6, Mh=125.13, alpha_s=0.1180; other inputs = SMDR 1.3 PDG defaults, as the producer)
L25 = ('2.5', '2.5', '3'); L2 = ('2', '2', '2')
cfg = {'L25_xi1': ((1,) * 5, L25), 'L25_xi0.5': ((0.5,) * 5, L25), 'L25_xi2': ((2,) * 5, L25),
       'L2_xi1': ((1,) * 5, L2), 'L2_xi0.5': ((0.5,) * 5, L2), 'L2_xi2': ((2,) * 5, L2),
       'L2_t0.5': ((0.5, 1, 1, 1, 1), L2), 'L2_t2': ((2, 1, 1, 1, 1), L2), 'L2_h0.5': ((1, 0.5, 1, 1, 1), L2), 'L2_h2': ((1, 2, 1, 1, 1), L2),
       'L2_G0.5': ((1, 1, 1, 0.5, 1), L2), 'L2_G2': ((1, 1, 1, 2, 1), L2)}


def parse(fn):
    d = {}
    for ln in open(fn):
        q = ln.split()
        if not q:
            continue
        if q[0] == 'RG_mub':
            d['RG_mub'] = {q[i]: float(q[i + 1]) for i in range(2, len(q) - 1, 2)}
        else:
            d[q[0]] = float(q[1])
    return d


def one(k):
    x, lo = cfg[k]
    fn = os.path.join(OUT, 'fit_%s.txt' % k)
    if not os.path.exists(fn):
        args = [exe, 'fit', '172.6000', '125.1300', '0.11800'] + ['%g' % v for v in x] + ['172.6000', '1e-10', fn] + list(lo)
        t0 = time.time(); r = subprocess.run(args, env=ENV, capture_output=True, text=True, timeout=3000, cwd=B)
        open(fn + '.log', 'w').write(r.stdout + r.stderr + '\nrc=%d t=%.1f\n' % (r.returncode, time.time() - t0))
    return k, parse(fn) if os.path.exists(fn) else None


t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    res = dict(ex.map(one, list(cfg)))
print('fits done %.0fs' % (time.time() - t0), [k for k, v in res.items() if v is None])
S = -0.006592  # sm-blind d lambda(mu_b)/dM_t
out = dict(note='Delta m_c := Delta lambda_b/|S| (producer sign convention), S=%g/GeV' % S, fits=res, deltas={})
for fam in ('L25', 'L2'):
    c = res[fam + '_xi1']
    for k in res:
        if k.startswith(fam + '_') and k != fam + '_xi1' and res[k]:
            dlb = res[k]['RG_mub']['lambda'] - c['RG_mub']['lambda']
            out['deltas'][k] = dict(dlam_mub=dlb, dmc_GeV=dlb / abs(S), dlam_Mt=res[k]['lambda'] - c['lambda'], dyt_Mt=res[k]['yt'] - c['yt'],
                                    dg3_Mt=res[k]['g3'] - c['g3'], dg_Mt=res[k]['g'] - c['g'], dgp_Mt=res[k]['gp'] - c['gp'])
out['L2_minus_L25_xi1'] = {q: res['L2_xi1'][q] - res['L25_xi1'][q] for q in ('lambda', 'yt', 'g3', 'g', 'gp')}
out['L2_minus_L25_xi1']['lam_mub'] = res['L2_xi1']['RG_mub']['lambda'] - res['L25_xi1']['RG_mub']['lambda']
# cross-check with the producer's own fits at the same point
pf = os.path.join(PROD_FITS, 'fit_xi1_Mt172.60.txt')
if os.path.exists(pf):
    pd = parse(pf)
    out['producer_xi1_Mt172.60'] = pd
    out['own_minus_producer_xi1'] = {q: res['L25_xi1'][q] - pd[q] for q in ('lambda', 'yt', 'g3', 'g', 'gp')}
    for kk in ('xi0.5', 'xi2'):
        f2 = os.path.join(PROD_FITS, 'fit_%s_Mt172.60.txt' % kk)
        if os.path.exists(f2):
            d2 = parse(f2); out['producer_dlam_mub_' + kk] = d2['RG_mub']['lambda'] - pd['RG_mub']['lambda']
out['code_sha256'] = {'work/drivers/c148_fit_copy.c': hashlib.sha256(open(cp, 'rb').read()).hexdigest(),
                      'work/diag_smdr_scale.py': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}
json.dump(out, open(MY + r'\results\diag_smdr_scale.json', 'w'), indent=1)
for k, v in out['deltas'].items():
    print('%-10s dlam_mub %+.3e  dm_c %+.4f GeV | dlam(Mt) %+.2e dyt(Mt) %+.2e dg3 %+.1e' % (k, v['dlam_mub'], v['dmc_GeV'], v['dlam_Mt'], v['dyt_Mt'], v['dg3_Mt']))
print('L2-L25 at xi=1:', out['L2_minus_L25_xi1'])
print('own-producer xi1:', out.get('own_minus_producer_xi1'), ' producer dlam_mub xi0.5/xi2:', out.get('producer_dlam_mub_xi0.5'), out.get('producer_dlam_mub_xi2'))
