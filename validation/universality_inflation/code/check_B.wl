(* check_B.wl -- ui-blind (V5): independent Wolfram evaluation of the MMR/Markkanen RGI potential barrier and B_HM *)
dir = DirectoryName[$InputFileName];
tab = Import[FileNameJoin[{dir, "wolfram_table.csv"}], "CSV"];
tgt = Import[FileNameJoin[{dir, "wolfram_target.json"}], "RawJSON"];
sv = tab[[All, 1]];
mk[i_] := Interpolation[Transpose[{sv, tab[[All, i]]}], InterpolationOrder -> 3];
G1f = mk[2]; G2f = mk[3]; Ytf = mk[4]; Lf = mk[5]; Kf = mk[6]; alf = mk[7];
R = tgt["R"]; c = tgt["c"]; H2 = R/12;
nn = {2, 6, -2, 1, 3, -1, -12, -12, -12, -12, -12, -12, -4, -4, -4, 1, 2, 1, -2, -1, 1, 3, -1, 8, 24, -8, -2, -2, -2, -1, -8};
dd = Join[{3/2, 5/6, 3/2, 3/2, 5/6, 3/2}, Table[3/2, 14], {3/2, 5/6, 3/2, 3/2, 5/6, 3/2, 3/2, 3/2, 3/2, 3/2, 3/2}];
np = Join[{-34/15, -34/5, 4/15, -17/15, -17/5, 2/15}, Table[38/5, 6], Table[38/15, 3], {-2/15, -4/15, -2/15, 4/15, 2/15},
   {-17/15, -17/5, 2/15, -136/15, -136/5, 16/15, 19/15, 19/15, 19/15, 2/15, 16/15}];
m2[h2_, s_] := Module[{g2 = G2f[s], gy2 = 3/5 G1f[s], yt = Ytf[s], l = Lf[s], xi = 1/6 + c Kf[s], mW, mZ, a},
   mW = g2 h2/4; mZ = (g2 + gy2) h2/4; a = 12 (xi - 1/6) H2;
   {mW + H2, mW + H2, mW - 2 H2, mZ + H2, mZ + H2, mZ - 2 H2, yt h2/2 + H2, H2, H2, H2, H2, H2, H2, H2, H2,
    3 l h2 + a, l h2 + mW + a, l h2 + mZ + a, mW - 2 H2, mZ - 2 H2, H2, H2, -2 H2, H2, H2, -2 H2, H2, H2, H2, -2 H2, -2 H2}];
F[s_?NumericQ, h2_?NumericQ] := Module[{M = m2[h2, s]/(H2 + h2), lg}, lg = Log[Abs[M]] + Log[H2 + h2] - s; Total[nn M^2 (lg - dd) + np (H2/(H2 + h2))^2 lg]];
sstar[h2_?NumericQ] := s /. FindRoot[F[s, h2], {s, Log[h2 + H2] - 1, Log[h2 + H2] + 1}, Method -> "Secant", AccuracyGoal -> 12, PrecisionGoal -> 12, MaxIterations -> 200];
V[h_?NumericQ] := Module[{ss = sstar[h^2]}, ((1/6 + c Kf[ss])/2 R h^2 + Lf[ss]/4 h^4 + alf[ss] R^2/144)/H2^2];
V0 = V[0];
hb0 = tgt["hbar_python"];
mx = FindMaximum[V[h], {h, 0.9 hb0, 1.1 hb0}, Method -> "PrincipalAxis", AccuracyGoal -> 10, PrecisionGoal -> 10];
dV = mx[[1]] - V0;
B = 384 Pi^2 dV H2^2/R^2;
res = <|"B_wolfram" -> N[B, 12], "B_python" -> tgt["B_python"], "hbar_wolfram" -> N[h /. mx[[2]], 12],
   "hbar_python" -> hb0, "V0_wolfram" -> N[V0 H2^2, 12], "V0_python" -> tgt["V0_python"],
   "mustar_bar_wolfram" -> N[Exp[sstar[(h /. mx[[2]])^2]/2], 12], "mustar_python" -> tgt["mustar_python"]|>;
Print[ToString[res, InputForm]];
