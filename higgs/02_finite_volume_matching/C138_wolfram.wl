(* PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_matching/C138_wolfram.wl (copied 2026-10-15 by code-higgs); optional Wolfram cross-check of gamma(lam = -0.010) *)
(* C138 / D110 official-Wolfram checks (higgs-matching).  Run: wolframscript -file C138_wolfram.wl *)
(* [v12 package; code-finalize 2026-10-15] input/output live in the higgs work tree (env DNEC_HIGGS_WORK, default
   <package>/output/higgs/quick), mirror path r2/higgs_matching/run01; the input table is written there by C138_checks.py
   (python higgs/02_finite_volume_matching/run.py --quick).  Research-tree absolute paths removed. *)
dncWork = Environment["DNEC_HIGGS_WORK"];
If[! StringQ[dncWork], dncWork = FileNameJoin[{ParentDirectory[DirectoryName[$InputFileName], 2], "output", "higgs", "quick"}]];
dncDir = FileNameJoin[{dncWork, "r2", "higgs_matching", "run01"}];
res = <||>;
out[key_, val_] := (res[key] = ToString[val, InputForm]; Print[key, " = ", ToString[val, InputForm]]);

(* 1. MS-bar per-mode zero-point energy in d = 3 - 2 eps spatial dimensions *)
Ifun[a_, b_, d_, m2_] := m2^(d/2 + a - b) Gamma[d/2 + a] Gamma[b - a - d/2]/((4 Pi)^(d/2) Gamma[d/2] Gamma[b]);
msb = (mu^2 Exp[EulerGamma]/(4 Pi))^eps;
zp = (1/2) Ifun[0, -1/2, 3 - 2 eps, m2] msb;
ser = Normal[Series[zp, {eps, 0, 0}]];
pole = Coefficient[Expand[ser], eps, -1];
fin = Simplify[PowerExpand[ser - pole/eps], m2 > 0 && mu > 0];
out["scalar_pole", Simplify[pole]];
out["scalar_MSbar_minus_CW(3/2)", Simplify[PowerExpand[fin - m2^2/(64 Pi^2) (Log[m2/mu^2] - 3/2)]]];
(* transverse polarisations: (d-1) = 2 - 2 eps copies -> constant 1/2 each; vector total 3 x 5/6 *)
serT = Normal[Series[(2 - 2 eps) zp, {eps, 0, 0}]];
finT = PowerExpand[serT - Coefficient[Expand[serT], eps, -1]/eps];
out["transverse_MSbar_minus_2CW(1/2)", Simplify[finT - 2 m2^2/(64 Pi^2) (Log[m2/mu^2] - 1/2)]];
out["vector_2T(1/2)+1L(3/2)_minus_3CW(5/6)",
  Simplify[2 (Log[x] - 1/2) + (Log[x] - 3/2) - 3 (Log[x] - 5/6)]];
(* Coulomb-mode first-order integral  I = Int d^dk/(2pi)^d sqrt(k^2+m^2)/k^2  (MS-bar) *)
serI = Normal[Series[Ifun[-1, -1/2, 3 - 2 eps, m2] msb, {eps, 0, 0}]];
finI = PowerExpand[serI - Coefficient[Expand[serI], eps, -1]/eps];
out["coulomb_I_MSbar_minus_m2(3-log)/(8pi^2)", Simplify[finI - m2/(8 Pi^2) (3 - Log[m2/mu^2])]];

(* 2. log cancellation: the continuum carries +m^4 log m^2/(64 pi^2); the k != 0 sum is analytic => D has -1/(64pi^2) *)
cont = Integrate[k^2 Sqrt[k^2 + m^2], {k, 0, Lam}, Assumptions -> Lam > 0 && m > 0]/(4 Pi^2);
sc = Normal[Series[cont, {Lam, Infinity, 1}]];
out["continuum_coeff_of_m^4*Log[m^2]", Simplify[Coefficient[Expand[PowerExpand[sc]], Log[m]]/(2 m^4)]];
out["expected", 1/(64 Pi^2)];

(* 3. Coulomb-gauge Goldstone/longitudinal dispersion from Gauss's law *)
Lk = (1/2) k^2 A0^2 + (1/2) (pd - mV A0)^2 - (1/2) (k^2 + mG2) p^2;
a0 = A0 /. First[Solve[D[Lk, A0] == 0, A0]];
Z = Simplify[Coefficient[Expand[Lk /. A0 -> a0], pd, 2] 2];
out["kinetic_Z", Z];
out["omega2_minus_expected", Simplify[(k^2 + mG2)/Z - (k^2 + mG2) (k^2 + mV^2)/k^2]];

(* 4. large-k expansion of w_C and the subtraction set (b; a; 2a) *)
wC = Sqrt[(k^2 + a) (k^2 + b)]/k;
out["wC_series", Simplify[Normal[Series[wC, {k, Infinity, 5}]]]];
w[m_] := Sqrt[k^2 + m];
summand = (wC - w[a]) - (w[b] + (2 b/a) w[a] - (b/a) w[2 a] - (1 + b/a) k);
out["subtracted_summand_series", Simplify[Normal[Series[summand, {k, Infinity, 5}]]]];
out["expected_k^-5_coeff", (5 a^2 b - a b^2)/16];

(* 5. one resonance width: MC + CW, lam = -0.010, v11 couplings; potential table exported by C138_checks.py *)
dat = Import[FileNameJoin[{dncDir, "wolfram_MC_U_v11_lam-0.010.json"}], "RawJSON"];
rr = N[dat["r"]]; UU = N[dat["U"]]; s = N[dat["s"]]; al = N[dat["alpha"]];
Uf = Interpolation[Transpose[{rr, UU}], InterpolationOrder -> 3, Method -> "Spline"];
r0 = 0.01; rout = 15.9;
F[e_?NumericQ] := Module[{sol, u, uu, du, U0, U1, U2, kk, k1, k2, y},
   sol = NDSolve[{u''[x] == 2 (Uf[x] - e) u[x], u[r0] == r0^s (1 + al/s r0),
       u'[r0] == s r0^(s - 1) + al (s + 1)/s r0^s}, u, {x, r0, rout},
      MaxSteps -> 10^7, PrecisionGoal -> 12, AccuracyGoal -> 30, WorkingPrecision -> MachinePrecision];
   uu = (u[rout] /. First[sol]); du = (u'[rout] /. First[sol]);
   U0 = Uf[rout]; U1 = Uf'[rout]; U2 = Uf''[rout];
   kk = Sqrt[2 (e - U0)]; k1 = -U1/kk; k2 = -U2/kk - U1^2/kk^3;
   y = I kk - k1/(2 kk) + (I/(2 kk)) (3 k1^2/(4 kk^2) - k2/(2 kk));
   du - y uu];
t0 = AbsoluteTime[];
root = FindRoot[F[e] == 0, {e, 4.39 - 0.0006 I, 4.391 - 0.0005 I}, AccuracyGoal -> 10, PrecisionGoal -> 10, MaxIterations -> 60];
eR = e /. root;
out["MC_v11_lam-0.010_E", eR];
out["MC_v11_lam-0.010_Q", Re[eR]];
out["MC_v11_lam-0.010_gamma", -2 Im[eR]];
out["seconds", AbsoluteTime[] - t0];
Export[FileNameJoin[{dncDir, "wolfram_results.json"}], res, "RawJSON"];
