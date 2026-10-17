# sm-blind: RG-consistency check of mr's pure-QCD top-mass coefficients y01..y04 at mu != M_t.
# Prediction: mbar(mu)/M = [1 + sum_i a0^i y0i(0)] * exp(-int_0^L gamma(a(L')) dL'), a0 = a(L=0), L = ln(mu^2/M^2),
# re-expanded in a = alpha_s^(6)(mu)/(4 pi), nf = 6.  Compared with mr's y0i(L) at L = 2 ln f0, f0 in {1/2,1/sqrt2,sqrt2,2}.
import os, sys, json, math
import sympy as sp
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
import smcore as S
RES = os.path.join(os.path.dirname(WORK), 'results')
recs = json.load(open(os.path.join(RES, 'prod_mr_records.json')))
N = 4
a0, L, x, a = sp.symbols('a0 L x a')
b = [sp.nsimplify(v) if i < 3 else sp.Float(v, 30) for i, v in enumerate(S.bQCD(6)[:N])]
g = [sp.Float(4 ** (i + 1) * v, 30) for i, v in enumerate(S.gamma_m(6)[:N])]  # a-normalisation
# a(L) by Picard iteration, truncated at a0^(N+1)
aL = a0
for _ in range(N + 1):
    aL = sp.expand(a0 - sp.integrate(sum(b[i] * aL ** (i + 2) for i in range(N)).subs(L, x), (x, 0, L)))
    aL = sp.series(aL, a0, 0, N + 2).removeO()
lnr = -sp.integrate(sp.expand(sp.series(sum(g[i] * aL ** (i + 1) for i in range(N)), a0, 0, N + 1).removeO()).subs(L, x), (x, 0, L))
c0 = recs['central']
y0 = [c0['t_y0%d' % i] for i in range(1, N + 1)]
m0 = 1 + sum(sp.Float(y0[i], 30) * a0 ** (i + 1) for i in range(N))
mL = sp.expand(sp.series(m0 * sp.exp(lnr), a0, 0, N + 1).removeO())
# invert a = aL(a0) -> a0(a)
inv = a
for _ in range(N + 1):
    inv = sp.expand(sp.series(a - (aL.subs(a0, inv) - inv), a, 0, N + 2).removeO())
mLa = sp.expand(sp.series(mL.subs(a0, inv), a, 0, N + 1).removeO())
pred = [sp.lambdify(L, mLa.coeff(a, i + 1)) for i in range(N)]
rows = []
for k, f0 in (('mu0_0.5', 0.5), ('mu0_0.707', 2 ** -0.5), ('mu0_1.414', 2 ** 0.5), ('mu0_2', 2.0)):
    Lv = 2 * math.log(f0)
    for i in range(N):
        mr = recs[k]['t_y0%d' % (i + 1)]; pv = float(pred[i](Lv))
        rows.append(dict(point=k, L=Lv, order=i + 1, mr=mr, rg_pred=pv, rel=(mr - pv) / abs(pv)))
mx = {o: max(abs(r['rel']) for r in rows if r['order'] == o) for o in range(1, N + 1)}
out = dict(description='RG consistency of mr pure-QCD y0i(L) with nf=6 4-loop QCD running (gamma_m, beta)', max_rel_dev_by_order=mx, rows=rows)
json.dump(out, open(os.path.join(RES, 'rg_check_y0i.json'), 'w'), indent=1)
Bp = os.path.join(RES, 'BLIND_RESULTS.json'); B = json.load(open(Bp)); B['rg_check_mr_qcd_coeffs'] = dict(max_rel_dev_by_order=mx, file='results/rg_check_y0i.json')
json.dump(B, open(Bp, 'w'), indent=1)
for r in rows:
    print('%-10s L=%+.4f y0%d  mr=%14.4f  RG=%14.4f  rel=%+.2e' % (r['point'], r['L'], r['order'], r['mr'], r['rg_pred'], r['rel']))
print('max rel dev by order:', mx)
