(* R15 Wolfram receipt: independent re-derivation of the D110 matching ingredients (r15-matching, 2026-10-13) *)
t0 = AbsoluteTime[];
Print["== 1. Coulomb-gauge Goldstone/longitudinal mode =="];
lag = 1/2 k^2 A0^2 + 1/2 (pd - mV A0)^2 - 1/2 (k^2 + bG) p^2;
solA = Solve[D[lag, A0] == 0, A0][[1]];
leff = Simplify[lag /. solA];
zk = Simplify[2 Coefficient[Expand[leff], pd^2]];
Print["A0 = ", A0 /. solA, " ;  Z_k = ", zk];
w2 = Simplify[(k^2 + bG)/zk];
Print["omega_C^2 - (k^2+a)(k^2+b)/k^2 = ", Simplify[(w2 /. mV -> Sqrt[aa]) - (k^2 + aa) (k^2 + bG)/k^2, aa > 0]];
serE = Normal[Series[Sqrt[(1 + aa e^2) (1 + bG e^2)]/e, {e, 0, 5}]];
Print["omega_C large-k series (e = 1/k): ", Collect[Expand[serE], e, Factor]];
cK3 = Coefficient[Expand[serE], e, 3];
c2C = Function[{x, y}, Evaluate[-8 cK3 /. {aa -> x, bG -> y}]];
Print["-8 * k^-3 coefficient = ", Factor[c2C[aa, bG]]];

Print["== 2. MS-bar constants in d = 3 - 2 eps spatial dimensions =="];
ival[d_] := (1/2) (2 Pi^(d/2)/Gamma[d/2]) (2 Pi)^(-d) m2^((d + 1)/2) Gamma[d/2] Gamma[-1/2 - d/2]/(2 Gamma[-1/2]);
expr = (mub^2 Exp[EulerGamma]/(4 Pi))^eps ival[3 - 2 eps];
pole = Simplify[SeriesCoefficient[expr, {eps, 0, -1}], m2 > 0];
fin = SeriesCoefficient[expr, {eps, 0, 0}];
Print["scalar pole = ", pole];
Print["scalar finite - m2^2(Log[m2/mub^2]-3/2)/(64Pi^2) = ",
  FullSimplify[PowerExpand[fin - m2^2 (Log[m2] - 2 Log[mub] - 3/2)/(64 Pi^2)], m2 > 0 && mub > 0]];
finT = SeriesCoefficient[(2 - 2 eps) expr, {eps, 0, 0}];
Print["(2-2eps) transverse finite - 2 m2^2(Log[m2/mub^2]-1/2)/(64Pi^2) = ",
  FullSimplify[PowerExpand[finT - 2 m2^2 (Log[m2] - 2 Log[mub] - 1/2)/(64 Pi^2)], m2 > 0 && mub > 0]];
Print["log coefficient of continuum (d/dLog[m2] of finite at fixed m2^2) = ",
  FullSimplify[PowerExpand[Coefficient[Expand[PowerExpand[fin]], Log[m2]]]]];

Print["== 3. One-loop RG consistency: gamma_phi for MC (Coulomb) and Landau counting =="];
mudE[c2_] := -c2/(32 Pi^2);
xW = g2^2/4; xZ = (g2^2 + gY^2)/4; xb = lam;
muV1C = 2 (2 mudE[xW^2] + mudE[c2C[xW, xb]]) + (2 mudE[xZ^2] + mudE[c2C[xZ, xb]]) + mudE[(3 lam)^2] - 12 mudE[(yt^2/2)^2];
muV1L = 2 (3 mudE[xW^2]) + 3 mudE[xZ^2] + mudE[(3 lam)^2] + 3 mudE[lam^2] - 12 mudE[(yt^2/2)^2];
beta1 = (24 lam^2 - 6 yt^4 + 3/8 (2 g2^4 + (g2^2 + gY^2)^2) + lam (12 yt^2 - 9 g2^2 - 3 gY^2))/(16 Pi^2);
gC = Simplify[gam /. Solve[muV1C + beta1/4 - gam lam == 0, gam][[1]]];
gL = Simplify[gam /. Solve[muV1L + beta1/4 - gam lam == 0, gam][[1]]];
Print["gamma_phi^C = ", gC, " ;  minus Feynman (3yt^2-3/2g2^2-1/2gY^2)/16Pi^2: ",
  Simplify[gC - (3 yt^2 - 3/2 g2^2 - 1/2 gY^2)/(16 Pi^2)]];
Print["gamma_phi^L = ", gL, " ;  minus Landau (3yt^2-9/4g2^2-3/4gY^2)/16Pi^2: ",
  Simplify[gL - (3 yt^2 - 9/4 g2^2 - 3/4 gY^2)/(16 Pi^2)]];
smc = {g2 -> 0.5277191152984825, gY -> 0.4423966476448222, yt -> 0.4283588578824067};
Print["numeric at mu_b: gamma_C = ", gC /. smc, " ; gamma_L = ", gL /. smc];

Print["== 4. Numeric: T_B/2, c2 (via Bessel D_P(4)), Delta(a,b) =="];
theta[t_?NumericQ] := EllipticTheta[3, 0, Exp[-Pi t]];
z12 = NIntegrate[(theta[t]^3 - 1) (t^(-1/2) + 1), {t, 1, Infinity}, WorkingPrecision -> 30, PrecisionGoal -> 20] - 3;
z2 = Pi^2 (NIntegrate[(theta[t]^3 - 1) (t + t^(-3/2)), {t, 1, Infinity}, WorkingPrecision -> 30, PrecisionGoal -> 20] + 3/2);
tb2 = N[z12/(8 Pi), 20];
Print["Z(1/2) = ", z12, " ; T_B/2 = ", tb2, " ; Z(2) = ", z2];
qmax = 2500;
reps = N[Select[Table[{q, SquaresR[3, q]}, {q, 1, qmax}], #[[2]] > 0 &], 30];
neff = (3 (Total[reps[[All, 2]]] + 1)/(4 Pi))^(1/3);
qs = reps[[All, 1]]; ns = reps[[All, 2]]; ks = 2 Pi Sqrt[qs];
gfun[x_, k_] := Module[{s = Sqrt[1 + x]}, k x^3 (s + 3)/(8 (1 + s)^3)];
rcw[a_, b_] := 1/2 (Total[ns (-(a + b) a b/(4 ks^5) - (a b)^2/(8 ks^7) + gfun[(a + b)/ks^2 + a b/ks^4, ks])] +
    (a + b) (a - b)^2/16 (2 Pi)^-5 2 Pi/neff^2);
eth[x_] := -(x^2/(4 Pi^2)) Total[ns BesselK[2, x Sqrt[qs]]/qs];
eth0 = -z2/(2 Pi^2);
dp[m2_] := eth[Sqrt[m2]] - eth0 - Sqrt[m2]/2;
vcw[m2_, c_] := m2^2 (Log[Abs[m2]] - c)/(64 Pi^2);
estd[m2_] := dp[m2] + vcw[m2, 3/2];
k3 = (estd[4] - rcw[4, 0] - tb2 4)/16;
k3b = (estd[9] - rcw[9, 0] - tb2 9)/81;
Print["K3 from m^2=4: ", N[k3, 16], " ; from m^2=9: ", N[k3b, 16], " ; c2 = K3 + 3/(128 Pi^2) = ", N[k3 + 3/2/(64 Pi^2), 16]];
ecw[a_, b_] := rcw[a, b] + tb2 (a + b) + k3 (a - b)^2;
dlt[a_, b_] := ecw[a, b] - estd[a];
Print["Delta(2.9637,-0.25) = ", N[dlt[29637/10000, -1/4], 16]];
g2n = 0.5277191152984825`30; gYn = 0.4423966476448222`30;
Do[
  Module[{aw = g2n^2 r^2/4, az = (g2n^2 + gYn^2) r^2/4, bb = -1/100 r^2},
   Print["r = ", r, " : sum_V Delta = dV_MC - dV_MU_CW = ", N[2 dlt[aw, bb] + dlt[az, bb], 16],
    " ; r*C(r) = sum_V dDelta/db|0 = ", N[2 (dlt[aw, 10^-8] - dlt[aw, -10^-8])/(2 10^-8) + (dlt[az, 10^-8] - dlt[az, -10^-8])/(2 10^-8), 12]]],
  {r, {3/2, 3, 5}}];
Print["time (s) = ", AbsoluteTime[] - t0];
