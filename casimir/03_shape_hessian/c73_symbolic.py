# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_symbolic.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] SymPy checks S1-S7 of D55: shape basis identities, shape-Laplacian eigenvalue 4p(3+2p)/3 (p = -2 gives the trace identity Tr_sh Hess E_C = (8/3) E_C), Gibbs/Legendre identities, Delta U = (8/3)[U - Q - (3/2) N], gain ODE -> hypergeometric, zero-mode canonical map. Paper: PRD Eqs. eq:s-Hesstrace, eq:s-nogo; App. app:shapelaplacian.

C73 symbolic checks (SymPy).  Output: results/.../C73/C73_symbolic.json
 S1  basis: tr(X_A X_B)=delta, sum X_A^2 = (5/3) I, sum (k^T X_A k)^2 = (2/3)|k|^4
 S2  Delta_sh (k^T G^-1 k)^p and (n^T G n)^p at G=I: eigenvalue 4p(3+2p)/3  (p symbolic)
     (any SPD unimodular G0 reduces to G=I by the isometry G -> S G S^T, k -> S^{-1}k)
 S3  G_ij d^2/dc_i dc_j sqrt((k+c)^T G^-1 (k+c)) |_{c=0} = 2/r, r = sqrt(k^T G^-1 k)   (generic SPD G)
 S4  Gibbs: F_xx = <E''> - Var(E')/T ; F_TT = -Var(E)/T^3 ; F_xT = Cov(E',E)/T^2 (3 generic levels)
     Legendre: d^2/dx^2 U(S fixed) = F_xx - F_xT^2/F_TT  (generic F(T,x), implicit T(x))
 S5  combined identity Delta U = (8/3)[U - Q - (3/2)N] (+ flux term), sum kappa = 3 Delta U / U
 S6  gain ODE -> hypergeometric equation under z = (1+tanh tau)/2
 S7  zero-mode canonical map: c = S x, G = S S^T: S^T G^-1 S = I (kinetic & quartic shape-free),
     <c c^T> = S <x x^T> S^T = v G for an O(3)-singlet ground state.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from c73_common import RES

out = {}
# ---------------------------------------------------------------- S1
s2, s6 = sp.sqrt(2), sp.sqrt(6)
X = [sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / s2,
     sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / s6,
     sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / s2,
     sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / s2,
     sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / s2]
gram = sp.Matrix(5, 5, lambda i, j: sp.simplify((X[i] * X[j]).trace()))
k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
k = sp.Matrix([k1, k2, k3])
sumX2 = sp.simplify(sum((Xa * Xa for Xa in X), sp.zeros(3, 3)))
q = (k.T * k)[0]
sum4 = sp.expand(sum(((k.T * Xa * k)[0]) ** 2 for Xa in X))
out["S1"] = {"gram_is_identity": gram == sp.eye(5), "sum_X2 == 5/3 I": sumX2 == sp.Rational(5, 3) * sp.eye(3),
             "sum (kXk)^2 - (2/3)|k|^4": str(sp.simplify(sum4 - sp.Rational(2, 3) * q ** 2))}
# ---------------------------------------------------------------- S2
p, t = sp.symbols('p t')
lap_inv, lap_dir = 0, 0
for Xa in X:
    # exp(-2tX) to second order suffices for d^2/dt^2 at t=0
    Em = sp.eye(3) - 2 * t * Xa + 2 * t ** 2 * Xa * Xa
    Ep = sp.eye(3) + 2 * t * Xa + 2 * t ** 2 * Xa * Xa
    lap_inv += sp.diff(((k.T * Em * k)[0]) ** p, t, 2).subs(t, 0)
    lap_dir += sp.diff(((k.T * Ep * k)[0]) ** p, t, 2).subs(t, 0)
mu = 4 * p * (3 + 2 * p) / 3
r_inv = sp.simplify(sp.powsimp(sp.expand(lap_inv / q ** p), force=True))
r_dir = sp.simplify(sp.powsimp(sp.expand(lap_dir / q ** p), force=True))
out["S2"] = {"Delta q^p / q^p (q=k^T G^-1 k)": str(sp.factor(r_inv)),
             "Delta q^p / q^p (q=n^T G n)": str(sp.factor(r_dir)),
             "minus 4p(3+2p)/3 (inverse)": str(sp.simplify(r_inv - mu)),
             "minus 4p(3+2p)/3 (direct)": str(sp.simplify(r_dir - mu)),
             "values": {str(pv): str(mu.subs(p, pv)) for pv in (sp.Rational(1, 2), -sp.Rational(1, 2), -1, -2, -sp.Rational(3, 2), 1)}}
# exact single-mode identity Delta omega = (8/3) omega  (p = 1/2)
out["S2"]["Delta omega / omega"] = str(sp.simplify(r_inv.subs(p, sp.Rational(1, 2))))
# ---------------------------------------------------------------- S3
h11, h12, h13, h22, h23, h33 = sp.symbols('h11 h12 h13 h22 h23 h33', real=True)
H = sp.Matrix([[h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])   # H = G^{-1}
c1, c2, c3 = sp.symbols('c1 c2 c3', real=True)
cv = sp.Matrix([c1, c2, c3])
f = sp.sqrt(((k + cv).T * H * (k + cv))[0])
Hess = sp.Matrix(3, 3, lambda i, j: sp.diff(f, [c1, c2, c3][i], [c1, c2, c3][j])).subs({c1: 0, c2: 0, c3: 0})
G = H.inv()
contr = sum(G[i, j] * Hess[i, j] for i in range(3) for j in range(3))
r = sp.sqrt((k.T * H * k)[0])
out["S3"] = {"G_ij Hess_ij - 2/r": str(sp.simplify(sp.together(contr - 2 / r)))}
# ---------------------------------------------------------------- S4
x, T = sp.symbols('x T', positive=True)
Es = [sp.Function(f'E{i}')(x) for i in range(3)]
gs = sp.symbols('g0:3', positive=True)
Zp = sum(g * sp.exp(-E / T) for g, E in zip(gs, Es))
F = -T * sp.log(Zp)
pr = [g * sp.exp(-E / T) / Zp for g, E in zip(gs, Es)]
avg = lambda vals: sum(pi * v for pi, v in zip(pr, vals))
Ep = [sp.diff(E, x) for E in Es]
Epp = [sp.diff(E, x, 2) for E in Es]
U = avg(Es)
var_f = avg([e ** 2 for e in Ep]) - avg(Ep) ** 2
varE = avg([E ** 2 for E in Es]) - U ** 2
cov = avg([a * b for a, b in zip(Ep, Es)]) - avg(Ep) * U
chk1 = sp.simplify(sp.diff(F, x, 2) - (avg(Epp) - var_f / T))
chk2 = sp.simplify(sp.diff(F, T, 2) + varE / T ** 3)
chk3 = sp.simplify(sp.diff(F, x, T) - cov / T ** 2)
# Legendre with a generic F(T,x) (algebraic restatement; the genuine test is the implicit-T numerics in c73_thermo)
Fg = sp.Function('F')
a_, b_ = sp.symbols('a b')
Fab = Fg(a_, b_)
Faa, Fab_, Fbb = sp.diff(Fab, a_, 2), sp.diff(Fab, a_, b_), sp.diff(Fab, b_, 2)
# S(T,x) = -F_T ; constraint S(T(x),x) = S0 => T' = -S_x/S_T = -F_Tx/F_TT
Tp = -Fab_ / Faa
# U(S0,x) = F(T(x),x) + T(x) S0 ; dU/dx = F_x (since F_T = -S0) ; d2U/dx2 = F_xx + F_xT T'
d2U = Fbb + Fab_ * Tp
target = Fbb - Fab_ ** 2 / Faa
chk4 = sp.simplify(d2U - target)
out["S4"] = {"F_xx - [<E''> - Var(E')/T]": str(chk1), "F_TT + Var(E)/T^3": str(chk2),
             "F_xT - Cov(E',E)/T^2": str(chk3), "Legendre d2U|_S - (F_xx - F_xT^2/F_TT)": str(chk4)}
# ---------------------------------------------------------------- S5
Us, Qs, Ns, Ph = sp.symbols('U Q N Phi')
Ff = Us - Qs - Ns - Ph
lapU = sp.Rational(8, 3) * Ff + 0 * Qs - sp.Rational(4, 3) * Ns + sp.Rational(20, 3) * Ph
out["S5"] = {"Delta U - (8/3)[U - Q - 3N/2] (Phi=0)": str(sp.simplify((lapU - sp.Rational(8, 3) * (Us - Qs - sp.Rational(3, 2) * Ns)).subs(Ph, 0))),
             "Delta U - (8/3)[U - Q - 3N/2 + 3Phi/2]": str(sp.simplify(lapU - sp.Rational(8, 3) * (Us - Qs - sp.Rational(3, 2) * Ns + sp.Rational(3, 2) * Ph))),
             "3 Delta U/U - 8[1 + (Q + 3N/2)/(-U)] (Phi=0)": str(sp.simplify((3 * lapU / Us - 8 * (1 + (Qs + sp.Rational(3, 2) * Ns) / (-Us))).subs(Ph, 0)))}
rr = sp.Symbol('r', positive=True)
Cf = sp.Symbol('C', positive=True)
# Q = r C_free, N = 0, U = -(1-r) C_free  => sum kappa = 8/(1-r)
out["S5"]["sum kappa with Q=rC, N=0"] = str(sp.simplify((8 * (1 + (rr * Cf) / ((1 - rr) * Cf)))))
# ---------------------------------------------------------------- S6
tau, kap = sp.symbols('tau kappa')
z = sp.Symbol('z')
y = sp.Function('y')
zz = (1 + sp.tanh(tau)) / 2
# beta(tau) = y(z(tau)); chain rule by hand with Y0=y, Y1=y', Y2=y'' ; compare with 4z(1-z) x hypergeometric operator
Y0, Y1, Y2 = sp.symbols('Y0 Y1 Y2')
dz = sp.diff(zz, tau)
d2z = sp.diff(zz, tau, 2)
lhs = Y2 * dz ** 2 + Y1 * d2z + sp.Rational(3, 2) * sp.tanh(tau) * Y1 * dz - kap / 4 / sp.cosh(tau) ** 2 * Y0
hyp = z * (1 - z) * Y2 + (sp.Rational(1, 4) - z / 2) * Y1 - kap / 4 * Y0
ratio = sp.simplify((lhs - 4 * zz * (1 - zz) * hyp.subs(z, zz)).rewrite(sp.exp))
out["S6"] = {"chain-rule ODE minus 4z(1-z)*hypergeometric operator": str(ratio),
             "a+b": "-1/2", "ab": "kappa/4", "c": "1/4"}
# ---------------------------------------------------------------- S7
Ssym = sp.Matrix(3, 3, sp.symbols('s0:9', real=True))
Gs = Ssym * Ssym.T
chk7 = sp.simplify(Ssym.T * Gs.inv() * Ssym - sp.eye(3))
out["S7"] = {"S^T G^-1 S - I": str(chk7), "G^-1 S - S^-T": str(sp.simplify(Gs.inv() * Ssym - Ssym.T.inv()))}
(RES / "C73_symbolic.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
