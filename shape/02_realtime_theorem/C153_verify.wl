(* PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/shape_hardening/C153_verify.wl (copied 2026-10-15 by code-shape-cmb) *)
(* Optional Wolfram cross-check (30 digits) of K(kappa), the gain, m11..m22 and G(inf) = 2 calG. *)
(* C153_verify.wl (shape-hardening, D125): official-Wolfram cross-check of the closed-form floor, the gain and the
   transfer matrix of beta'' + 3 tanh(2t) beta' - kappa sech^2(2t) beta = 0, C60 normalisation Pi = cosh^{3/2}(2t) beta'/C32. *)
c32 = 3/2^(3/2);
kcl[k_] := Module[{mu = Sqrt[4 k - 1]/4, d = 3 Sqrt[2]/4}, Gamma[7/4] Gamma[3/4]/(2 d Abs[Gamma[5/4 + I mu]]^2)];
gain[k_] := Sqrt[2] Cosh[Pi Sqrt[4 k - 1]/4];
tr[k_, t0_, t1_] := Module[{sol, b, p},
  sol = NDSolve[{b'[t] == c32 p[t]/Cosh[2 t]^(3/2), p'[t] == k b[t]/(c32 Sqrt[Cosh[2 t]]), b[t0] == #[[1]], p[t0] == #[[2]]},
     {b, p}, {t, Min[t0, t1], Max[t0, t1]}, WorkingPrecision -> 30, PrecisionGoal -> 20, AccuracyGoal -> 25, MaxSteps -> 10^6] & /@ {{1, 0}, {0, 1}};
  Transpose[({b[t1], p[t1]} /. #[[1]]) & /@ sol]];
res = Table[
  Module[{M0 = tr[kk, 0, 16], Mm = tr[kk, -16, 16], Mh = tr[kk, -1/2, 16]},
   {kk, N[kcl[kk], 16], N[2 Abs[M0[[1, 1]] M0[[1, 2]]] 2^(3/2)/6, 16], N[gain[kk], 16], N[M0[[1, 1]], 16],
    N[Abs[Mm[[1, 1]] Mm[[1, 2]]]/Abs[M0[[1, 1]] M0[[1, 2]]], 16], N[2 gain[kk], 16],
    N[Abs[Mh[[1, 1]] Mh[[1, 2]]]/Abs[M0[[1, 1]] M0[[1, 2]]], 16], N[M0[[2, 1]], 16], N[M0[[2, 2]], 16]}],
  {kk, {19323700/1000000, 1932/100, 18}}];
Print["rows: kappa, K_closed, K_numeric, calG_closed, m11, G_inf_numeric, 2calG, G(0.5), m21, m22"];
Print[ToString[res, InputForm]];
Print[ToString[N[{Sqrt[kcl[19323700/1000000]/(722508/1000)^2/(437243/100000000)], Sqrt[kcl[18]/((722508/1000)^2 437243/100000000 ((1 - 28/100)/(1 - 299897/1000000))^(3/4))]}, 10], InputForm]];
