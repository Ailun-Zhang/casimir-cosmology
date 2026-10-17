(* PROVENANCE: 04_calculations/code/20261013_claude_v12/state_selection/C137_verify.wl (copied 2026-10-15 by code-shape-cmb) *)
(* Optional Wolfram cross-check of C137 (freezing identity, transfer, G(t_x)); run with wolframscript -file. *)
(* C137 independent Wolfram check (Claude v12, state_selection). Run with wolframscript -file. *)
res = <||>;
(* 1. freezing ratio omega/|H| on v11 background, omega^2 = |k| h^2 Sech[2 h t]^2, H = h Tanh[2 h t] *)
res["freeze_identity"] = FullSimplify[(Abs[k] h^2 Sech[2 h t]^2)/(h Tanh[2 h t])^2 == Abs[k]/Sinh[2 h t]^2, Assumptions -> {h > 0, t != 0, Element[k, Reals]}];
res["freeze_times"] = Table[{kk, N[ArcSinh[Sqrt[kk]]/2, 12]}, {kk, {19.32, 9.07, 13.53, 6.35}}];
(* 2. asymptotic exponents in N = ln L for rho = A/L^2 with stiffness k_w: beta'' + 2 beta' + 3 kw beta = 0 *)
res["winding_exponents"] = Solve[r^2 + 2 r + 3 kw == 0, r];
(* general m_d: d ln H/dN = (m_d-3)/2 -> beta'' + (3+m_d)/2 beta' + c beta = 0, oscillatory iff c > ((3+m_d)/4)^2 *)
res["osc_condition"] = Reduce[((3 + md)/2)^2 - 4 c < 0, c, Reals];
(* 3. turn criterion for V + traceless Casimir: rho - pbar at H=0 *)
res["turn_criterion"] = Simplify[(2 V + 2/3 rhoC) /. rhoC -> -V - S/2];
(* 4. transfer of the v11 shape ODE in units L_b = 1, time 1/H_eff; m(t) = Cosh[2t]^(3/2) *)
transfer[kap_, t0_, t1_] := Module[{b1, b2, s1, s2, m},
  m[t_] := Cosh[2 t]^(3/2);
  s1 = NDSolveValue[{b''[t] + 3 Tanh[2 t] b'[t] - kap Sech[2 t]^2 b[t] == 0, b[t0] == 1, b'[t0] == 0}, {b[t1], b'[t1]}, {t, Min[t0, t1], Max[t0, t1]}, WorkingPrecision -> 30, PrecisionGoal -> 15, AccuracyGoal -> 20, MaxSteps -> 10^6];
  s2 = NDSolveValue[{b''[t] + 3 Tanh[2 t] b'[t] - kap Sech[2 t]^2 b[t] == 0, b[t0] == 0, b'[t0] == 1/m[t0]}, {b[t1], b'[t1]}, {t, Min[t0, t1], Max[t0, t1]}, WorkingPrecision -> 30, PrecisionGoal -> 15, AccuracyGoal -> 20, MaxSteps -> 10^6];
  {{s1[[1]], s2[[1]]}, {m[t1] s1[[2]], m[t1] s2[[2]]}}];
M0 = transfer[1932/100, 0, 6];
floorE = Abs[M0[[1, 1]] M0[[1, 2]]];
res["m11_E"] = N[M0[[1, 1]], 12];
natE = Module[{w = Sqrt[1932/100], s}, s = 1/(2 w); M0[[1, 1]]^2 s + M0[[1, 2]]^2/(4 s)];
res["natural_over_floor_E"] = N[natE/floorE, 12];
res["opt_uncorrelated_over_floor_E"] = Table[{tx, N[Abs[Times @@ (transfer[1932/100, tx, 6][[1]])]/floorE, 10]}, {tx, {-1/10, -1/2, -1, -8}}];
(* 5. best case: adiabatic Gaussian (omega = |H|) at L = e^{Npre} L_b, propagate to the turn; R_P and shear ratio *)
bestcase[Npre_] := Module[{tx, mx, wx, cov, M, covT, wb = Sqrt[1932/100]},
  tx = -ArcCosh[Exp[2 Npre]]/2; mx = Cosh[2 tx]^(3/2); wx = Abs[Tanh[2 tx]];
  cov = DiagonalMatrix[{1/(2 mx wx), mx wx/2}];
  M = transfer[1932/100, tx, 0];
  covT = M.cov.Transpose[M];
  {Npre, N[covT[[2, 2]]/(wb/2), 10]}];
res["bestcase_RP_E"] = Table[bestcase[n], {n, {1/2, 2}}];
Scan[Print[#, " -> ", ToString[res[#], InputForm]] &, Keys[res]];
