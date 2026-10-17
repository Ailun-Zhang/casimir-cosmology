# rep-A checks, part 2: fractions of the conformal duration, misc.
import sys, math
sys.stdout.reconfigure(encoding='utf-8')
from scipy.integrate import quad
from scipy.optimize import brentq
f = lambda u: (math.sqrt(2) * math.exp(-u) / math.sqrt(1 + math.exp(-4 * u)))
tot = quad(f, 0, math.inf)[0]
print('eta_inf*hLb =', tot)
for frac in [0.5, 0.9, 0.99]:
    tau = brentq(lambda t: quad(f, 0, t)[0] / tot - frac, 0, 20)
    print('fraction %.2f reached at h t = %.3f' % (frac, tau))
# freezing onset fraction
for t in [1.093, 0.35]:
    print('fraction of eta_inf by ht=%.3f: %.3f' % (t, quad(f, 0, t)[0] / tot))
# epsilon relation
gC = 1.6750738; NC = 25.515
print('sqrt(3/gamma_C) =', math.sqrt(3 / gC), '; eps from hLb=0.00483:', math.sqrt(3 / gC) * 0.00483**2)
# Var H of the Gaussian family check (sympy)
import sympy as sp
a, b, m, k, x = sp.symbols('a b m k x', positive=True)
psi = sp.exp(-(a - sp.I * b) * x**2)
norm = sp.integrate(sp.Abs(psi)**2, (x, -sp.oo, sp.oo))
H = lambda f: -sp.diff(f, x, 2) / (2 * m) - k * x**2 / 2 * f
EH = sp.simplify(sp.integrate(sp.conjugate(psi) * H(psi), (x, -sp.oo, sp.oo)) / norm)
EH2 = sp.simplify(sp.integrate(sp.conjugate(H(psi)) * H(psi), (x, -sp.oo, sp.oo)) / norm)
var = sp.simplify(sp.expand(EH2 - EH**2))
target = (16*a**4 + 32*a**2*b**2 + 8*a**2*k*m + 16*b**4 - 8*b**2*k*m + k**2*m**2) / (32*a**2*m**2)
print('Var H - PRD formula =', sp.simplify(var - target))
