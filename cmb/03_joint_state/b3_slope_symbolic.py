# PROVENANCE: 04_calculations/code/20261013_argon_codex_intake/joint_state/b3_slope_symbolic.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] SymPy check of the B3 slope term of the inflaton width (stdout; archived as b3_slope_symbolic.txt).

C147/D119: symbolic check of the B3 lemma with a plateau slope.
H = p^2/2m - k x^2/2 + F x (k = m|V''|, F = m V'), Gaussian psi = exp(-(a-ib)x^2 + i p0 x), a>0.
H is Weyl-ordered and quadratic, so <H^2> = <H^2>_Wigner + k/(4m) exactly (Moyal: H*H = H^2 - (1/4)(H_xx H_pp - H_xp^2)).
Checks: (1) p0=0 reproduces D113's inverted-oscillator variance + F^2/(4a);
(2) dVar/db = 0 -> b=0 or b^2 = km/4 - a^2; (3) optimal p0 and the reduced variance;
(4) b=0 stationarity cubic in a; (5) slope-free limit gives a = sqrt(km)/2 (Codex/D113 width).
"""
import sympy as sp

a, m, k = sp.symbols('a m k', positive=True)
b, p0, F = sp.symbols('b p0 F', real=True)
sx = 1/(4*a); c = b/(2*a); sp_ = a + b**2/a             # <x^2>, <xp>_sym, Var p
# Gaussian Wigner moments with mean (0, p0): use characteristic-function expansion via Isserlis
X, Pc = sp.symbols('X Pc')                                 # centred variables
def E(poly):
    poly = sp.Poly(sp.expand(poly), X, Pc); tot = 0
    for (i, j), coef in poly.terms():
        # E[X^i Pc^j] for bivariate normal, via derivative of MGF
        t1, t2 = sp.symbols('t1 t2')
        mgf = sp.exp(sx*t1**2/2 + c*t1*t2 + sp_*t2**2/2)
        mom = sp.diff(mgf, t1, i, t2, j).subs({t1: 0, t2: 0})
        tot += coef*mom
    return sp.simplify(tot)
Hcl = (Pc + p0)**2/(2*m) - k*X**2/2 + F*X
VarH = sp.simplify(E(Hcl**2) - E(Hcl)**2 + k/(4*m))
print('VarH =', sp.factor(sp.simplify(VarH)))
D113 = (16*a**4 + 32*a**2*b**2 + 8*a**2*k*m + 16*b**4 - 8*b**2*k*m + k**2*m**2)/(32*a**2*m**2)
chk1 = sp.simplify(VarH.subs(p0, 0) - D113 - F**2/(4*a))
print('(1) VarH(p0=0) - [D113 + F^2/(4a)] =', chk1)
dVb = sp.factor(sp.diff(VarH.subs(p0, 0), b))
print('(2) dVar/db at p0=0 =', dVb)
p0s = sp.solve(sp.diff(VarH, p0), p0)
print('(3) optimal p0 =', p0s)
Vred = sp.simplify(VarH.subs(p0, p0s[0]))
print('    reduced Var - D113 =', sp.simplify(Vred - D113), '  (expect F^2 a/(4(a^2+b^2)))')
dVa = sp.factor(sp.diff(VarH.subs({p0: 0, b: 0}), a))
print('(4) dVar/da at b=p0=0 =', dVa)
print('(5) slope-free stationary a:', sp.solve(sp.diff(D113.subs(b, 0), a), a))
print('    d2Var/db2 at b=0,p0=0:', sp.factor(sp.diff(VarH.subs(p0, 0), b, 2).subs(b, 0)))
