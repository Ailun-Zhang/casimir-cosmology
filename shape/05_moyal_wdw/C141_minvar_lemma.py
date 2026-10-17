# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/time_symmetry_origin/C141_minvar_lemma.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] B3 width lemma (D113 Sec. 3, SymPy): among centred Gaussians of the inverted oscillator the energy variance is minimal, Var H = k/2m, for the real member of width 1/(2m|omega|) (the natural width). Prints to stdout (archived as C141_minvar_lemma_output.txt). Paper: PRD Sec. VIII B (sec:s-equivalent), App. app:theorem.
"""
# Check: energy variance of pure centred Gaussians for H = p^2/(2m) -/+ k x^2/2 (hbar=1)
import sympy as sp

x = sp.symbols('x', real=True)
m, k, a = sp.symbols('m k a', positive=True)   # a = 1/(4X) real part of exponent coefficient
b = sp.symbols('b', real=True)                  # imaginary part -> x-p correlation
psi = (2*a/sp.pi)**sp.Rational(1,4) * sp.exp(-(a - sp.I*b)*x**2)
psic = sp.conjugate(psi)

def expect(op_psi):
    return sp.simplify(sp.integrate(sp.simplify(psic*op_psi), (x, -sp.oo, sp.oo)))

def p(f):  return -sp.I*sp.diff(f, x)

for sign, name in [(-1, 'inverted'), (+1, 'normal')]:
    H = lambda f: p(p(f))/(2*m) + sign*k*x**2*f/2
    E1 = expect(H(psi))
    E2 = expect(H(H(psi)))
    var = sp.simplify(sp.expand(E2 - E1**2))
    X = sp.simplify(expect(x**2*psi))
    print(name, ' <x^2> =', X, ' Var(H) =', sp.factor(var))
    if sign == -1:
        # minimise over a at b=0 (real Gaussian)
        v0 = sp.simplify(var.subs(b, 0))
        sol = sp.solve(sp.diff(v0, a), a)
        print('  real-Gaussian stationary a:', sol, ' -> X =', [sp.simplify(1/(4*s)) for s in sol])
        print('  natural width 1/(2 m |w|) with |w|=sqrt(k/m):', sp.simplify(1/(2*m*sp.sqrt(k/m))))
        print('  Var at optimum:', [sp.simplify(v0.subs(a, s)) for s in sol])
        # flat direction: minimise over a for fixed b, show min Var independent of b
        crit = sp.solve(sp.diff(var, a), a)
        print('  stationary a(b):', crit)
        for s in crit:
            print('   Var_min(b) =', sp.simplify(var.subs(a, s)))
