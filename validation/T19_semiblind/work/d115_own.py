# d115_own.py -- independent check of the D115 shape-side closed form (T19, unverified-check, 2026-10-14)
# F(kappa) = floor variance |T12^can|/2 in units M_P^2 L_b^3 h = 1 on the leading cosh background.
# Own derivation: with z=(1+tanh 2t)/2 the mode eq. is hypergeometric (c=1/4, a+b=-1/2, ab=kappa/4).
# The solution z^{3/4} 2F1(a+3/4,b+3/4;7/4;z) has beta_-=0, Pi_-=1 (Pi=(2sqrt2/3)L^3 beta_dot), so
# T12 = 2F1(a+3/4,b+3/4;7/4;1) = Gamma(7/4)Gamma(3/4)/(Gamma(1-a)Gamma(1-b)) = Gamma(7/4)Gamma(3/4)/|Gamma(5/4+i nu/4)|^2,
# nu = sqrt(4 kappa - 1);  T11 = sqrt2 cosh(pi nu/4).  T12^can = T12/p0, p0 = 3/(2 sqrt2) -> F = (sqrt2/3) T12.
import os, sys, json, math
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import mpmath as mp
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d114_own as D

mp.mp.dps = 30


def T12(k):
    nu = mp.sqrt(4 * mp.mpf(k) - 1)
    return mp.gamma(mp.mpf(7) / 4) * mp.gamma(mp.mpf(3) / 4) / abs(mp.gamma(mp.mpf(5) / 4 + 1j * nu / 4)) ** 2


def T12_hyp(k):
    k = mp.mpf(k)
    s = mp.sqrt(1 - 4 * k + 0j)
    a = (-1 + s) / 4
    b = (-1 - s) / 4
    return mp.re(mp.hyp2f1(a + mp.mpf(3) / 4, b + mp.mpf(3) / 4, mp.mpf(7) / 4, 1))


def F(k):
    return mp.sqrt(2) / 3 * T12(k)


res = {}
for k in ('19.32', '19.3237', '19.32374589', '13.528612469', '18.0', '6.0'):
    res['F(%s)' % k] = float(F(k))
    res['T12(%s)' % k] = float(T12(k))
    res['T12_hyp2f1(%s)' % k] = float(T12_hyp(k))
# independent numeric ODE transfer (own d114 module, leading background)
m = D.Model('lead')
for k in (19.32, 19.3237):
    M = m.lin_transfer(k, 0.0)
    res['T12_ode(%g)' % k] = float(2 * M[0, 0] * M[0, 1])
    res['F_ode(%g)' % k] = float(math.sqrt(2) / 3 * 2 * M[0, 0] * M[0, 1])
# local log-slope d ln F / d kappa at 19.32
h = mp.mpf('1e-6')
res['dlnF_dkappa(19.32)'] = float((mp.log(F(mp.mpf('19.32') + h)) - mp.log(F(mp.mpf('19.32') - h))) / (2 * h))
# scaling exponents (symbolic)
As, r, C, Fk, MP, h_, c0 = sp.symbols('A_s r C_eff F M_P h c0', positive=True)
Lb = (C / (3 * MP ** 2 * h_ ** 2)) ** sp.Rational(1, 4)
sigma2 = Fk / (MP ** 2 * Lb ** 3 * h_)
sigma2 = sp.simplify(sigma2.subs(h_, c0 * MP * sp.sqrt(As * r)))   # h/M_P = c0 (A_s r)^(1/2)
sig = sp.sqrt(sigma2)
ex = {str(v): str(sp.simplify(sp.diff(sp.log(sig), v) * v)) for v in (As, r, C, Fk)}
res['sigma_E_log_exponents'] = ex
# consistency with the leading-background fossil: sigma_floor^2 = F/(M_P^2 h L_b^3), M_P^2 h L_b^3 = sqrt2/(3 delta)
d_lead = 1.565e-4 / (1 - D.R_ZM) ** 0.75 * (3.0 / D.vpot(D.PHIB)) ** 0.75
res['delta_lead_xi'] = d_lead
res['sigma_floor_from_F(19.3237)'] = math.sqrt(float(F('19.32374589')) * 3 * d_lead / math.sqrt(2))
# H_*/h for Starobinsky plateau at N_*=55 (slow roll, eps(phi_end)=1)
a = math.sqrt(2 / 3)
xe = math.log((2 * a + math.sqrt(2)) / math.sqrt(2))
def Nfold(x):
    return 0.75 * (math.exp(x) - math.exp(xe)) - (x - xe) * 0.75
lo, hi = 2.0, 8.0
for _ in range(80):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if Nfold(mid) < 55 else (lo, mid)
res['Hstar_over_h_N55_slowroll'] = 1 - math.exp(-lo)
res['phi_star_N55'] = lo / a
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(res, indent=1))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'D115_own.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=1)
