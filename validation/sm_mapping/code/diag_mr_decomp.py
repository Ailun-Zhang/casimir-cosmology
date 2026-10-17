# sm-blind diagnosis: decompose the mr (G_F-scheme, 2-loop EW + 4-loop QCD) matching-scale dependence of lambda(mu_b)
# by coupling (hybrid initial conditions at M_t), split y_t into pure-QCD tower and the rest, test the alpha_s
# decoupling-scale treatment; no new mr calls (stored coefficients). Also: download + read Beneke et al. 2016.
import os, sys, math, json, hashlib, datetime, urllib.request, re
os.environ['OMP_NUM_THREADS'] = '1'
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
sys.stdout.reconfigure(encoding='utf-8')
import smcore as S
MY = os.path.dirname(WORK); RES = os.path.join(MY, 'results')
recs = json.load(open(os.path.join(RES, 'prod_mr_records.json')))
diag = json.load(open(os.path.join(RES, 'diag_mu0.json')))
R = S.RGE(3, 'full', True)
Mt = 172.60; mub = S.mu_b_and_nodes()[0]; Sl = -0.006592
c_cent = S.mr_couplings(recs['central'], 'GF', S.ALL_ORDERS, yb=recs['central']['yb_own'])
lam_c = R.run(c_cent, Mt, [mub])[mub]['lam']
def lamb(c0, mu0=Mt):
    return R.run(c0, mu0, [mub])[mub]['lam']
out = dict(note='dm_c := dlam_b/|S| (producer sign convention: + means critical m_t up), S=%g' % Sl, mu0={})
for k in ('mu0_0.5', 'mu0_0.707', 'mu0_1.414', 'mu0_2'):
    rec = recs[k]; mu0 = rec['mu']
    c0 = S.mr_couplings(rec, 'GF', S.ALL_ORDERS, yb=rec['yb_own'])
    back = R.run(c0, mu0, [Mt])[Mt]
    full = lamb(c0, mu0) - lam_c
    row = dict(full=full)
    for q, keys in (('yt', ('yt',)), ('lam', ('lam',)), ('g3', ('g3',)), ('g2_gY', ('g2', 'gY'))):
        h = dict(c_cent); h.update({kk: back[kk] for kk in keys}); h['g1'] = math.sqrt(5 / 3) * h['gY']
        row[q] = lamb(h) - lam_c
    dq = diag[k]['qcd_only_pureQCDrun_yt_Mt'] / diag['central']['qcd_only_pureQCDrun_yt_Mt'] - 1
    h = dict(c_cent); h['yt'] = c_cent['yt'] * (1 + dq)
    row['yt_QCD_tower'] = lamb(h) - lam_c
    row['yt_nonQCD'] = row['yt'] - row['yt_QCD_tower']
    row['sum_linear'] = row['yt'] + row['lam'] + row['g3'] + row['g2_gY']
    # alpha_s decoupled at M_t then 6-flavour running to mu0 (instead of decoupling at mu0); alpha(mu0) recomputed
    a6 = S.as6(mu0, 0.1180, Mt, 91.1879, 4, 3, mu_dec=Mt)
    aF = math.sqrt(2) * rec['Gf'] * rec['MW'] ** 2 / math.pi * (1 - rec['MW'] ** 2 / rec['MZ'] ** 2)
    k4 = aF / (4 * math.pi)
    alpha = aF * (1 + k4 * rec['aGF_a10'] + k4 * a6 / (4 * math.pi) * rec['aGF_a11'] + k4 ** 2 * rec['aGF_a20'])
    r2 = dict(rec, as_mu=a6, alpha_GF=alpha)
    row['alpha_s_dec_at_Mt_variant'] = lamb(S.mr_couplings(r2, 'GF', S.ALL_ORDERS, yb=rec['yb_own']), mu0) - lam_c
    row['alpha_GF_check(own recompute - mr)'] = aF * (1 + k4 * rec['aGF_a10'] + k4 * rec['as_mu'] / (4 * math.pi) * rec['aGF_a11'] + k4 ** 2 * rec['aGF_a20']) - rec['alpha_GF']
    row['dm_c_GeV'] = {kk: v / abs(Sl) for kk, v in row.items() if isinstance(v, float) and not kk.startswith('alpha_GF')}
    row['couplings_back_at_Mt_minus_central'] = {kk: back[kk] - c_cent[kk] for kk in ('lam', 'yt', 'g2', 'gY', 'g3')}
    out['mu0'][k] = row
# known 3-loop O(alpha_t alpha_s^2) lambda term (Buttazzo et al. footnote 5, gaugeless, mu=M_t) absent from mr
g3 = c_cent['g3']; Gm = recs['central']['Gf']
l3 = -(8 / 135) * g3 ** 4 * Gm ** 2 * Mt ** 4 / (4 * math.pi) ** 6 * (176 * math.pi ** 4 + 240 * math.pi ** 2 * (3 + 4 * math.log(2) ** 2 + 6 * math.log(2))
     + 15 * (607 - 64 * math.log(2) ** 4 - 1536 * S.LI4 + 576 * S.Z3))
h = dict(c_cent); h['lam'] += l3
out['lambda3_QCD_Buttazzo_fn5'] = dict(value_at_Mt=l3, dlam_b=lamb(h) - lam_c, dm_c_GeV=(lamb(h) - lam_c) / abs(Sl))
# Beneke, Marquard, Nason, Steinhauser 2016 (arXiv:1605.03609): download into a new empty dir and extract passages
d = os.path.join(MY, 'inputs', '20261015_beneke2016'); os.makedirs(d, exist_ok=True)
pdf = os.path.join(d, 'Beneke2016_1605.03609.pdf')
if not os.path.exists(pdf):
    req = urllib.request.Request('https://arxiv.org/pdf/1605.03609', headers={'User-Agent': 'Mozilla/5.0 academic-research-download'})
    data = urllib.request.urlopen(req, timeout=240).read(); open(pdf, 'wb').write(data)
    json.dump([dict(file=os.path.basename(pdf), url='https://arxiv.org/pdf/1605.03609', date_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest())], open(os.path.join(d, 'SOURCES.json'), 'w'), indent=1)
from pypdf import PdfReader
pages = [pg.extract_text() or '' for pg in PdfReader(pdf).pages]
open(os.path.join(WORK, 'txt', 'Beneke2016_1605.03609.txt'), 'w', encoding='utf-8').write('\n'.join('=== PAGE %d ===\n%s' % (i + 1, t) for i, t in enumerate(pages)))
hits = []
for i, t in enumerate(pages):
    tt = re.sub(r'\s+', ' ', t)
    for m in re.finditer(r'[^.]{0,260}(\d+\s?MeV|ambiguit|five-loop|5-loop|\(4\.7\)|\(4\.8\)|0\.27\d|minimal term)[^.]{0,200}\.', tt):
        s = m.group(0).strip()
        if s not in [h_['text'] for h_ in hits]:
            hits.append(dict(page=i + 1, text=s))
out['Beneke2016_passages'] = hits[:40]
json.dump(out, open(os.path.join(RES, 'diag_mr_decomp.json'), 'w'), indent=1)
for k, r in out['mu0'].items():
    print(k, 'dm_c [GeV]:', {kk: round(v, 4) for kk, v in r['dm_c_GeV'].items()})
    print('    back@Mt:', {kk: '%+.2e' % v for kk, v in r['couplings_back_at_Mt_minus_central'].items()}, ' alphaGF recompute check %.1e' % r['alpha_GF_check(own recompute - mr)'])
print('lambda3_QCD:', out['lambda3_QCD_Buttazzo_fn5'])
for h_ in hits[:22]:
    print('p%d: %s' % (h_['page'], h_['text'][:420]))
