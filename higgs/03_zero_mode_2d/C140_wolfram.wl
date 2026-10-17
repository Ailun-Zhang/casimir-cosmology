(* PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/higgs_beyond_bo/C140_wolfram.wl (copied 2026-10-15 by code-higgs); optional Wolfram check of the decomposition identities *)
(* C140_wolfram.wl -- DNEC v12 round 2, agent higgs-beyond-bo (D112), Claude 2026-10-13.
   Official Wolfram checks of the analytic core of the 2D model:
   id1  D108 conditional-Gaussian energy with c = k beta, b = bt + k beta^2  ==  C0 + (3/4)(bt + Om^2/bt)
   id2  D108 Born-Huang (3/16) tr[(K^-1 K')^2]  ==  (3/16)[2(a'/a)^2 + (k'/k)^2 + (bt'/bt)^2] + 3 k beta'^2/(8 bt)
   id3  1D oscillator Born-Huang  (1/2)<(d_r psi)^2> == (1/16)(w'/w)^2
   id4  w-derivative of the shifted conditional Gaussian: (1/2) int (d_w phi)^2 == k beta^2/4   (per direction)
   id5  r-derivative of the shifted conditional Gaussian: (1/2) int (d_r phi)^2 == (k/4) beta'^2 w^2 + (1/16)(k'/k)^2
   id6  angular term: (2/r^2)<L1^2 + L2^2> part of id1 split into 3(k-a)^2/(a k r^2) + (3/4)(4 a beta^2/r^2)/bt
   num  3D isotropic oscillator s-wave ground state of -1/2 Lap + 1/2 W^2 s^2 (W = 0.3): 3/2 W via NDEigenvalues (radial)
*)
(* [v12 package; code-finalize 2026-10-15] output directory = higgs work tree (env DNEC_HIGGS_WORK, default
   <package>/output/higgs/quick), mirror path r2/higgs_beyond_bo/run01.  Research-tree absolute path removed. *)
dncWork = Environment["DNEC_HIGGS_WORK"];
If[! StringQ[dncWork], dncWork = FileNameJoin[{ParentDirectory[DirectoryName[$InputFileName], 2], "output", "higgs", "quick"}]];
res = FileNameJoin[{dncWork, "r2", "higgs_beyond_bo", "run01"}];
If[! DirectoryQ[res], CreateDirectory[res, CreateIntermediateDirectories -> True]];
eD108[a_, k_, c_, b_, r_, g_, gy_] := Module[{dd = k b - c^2},
   3/4 (2 a + k + b) + 3/16 r^2 (2 g^2/a + (g^2 b - 2 g gy c + gy^2 k)/dd) + 3/4 g^2/a^2 + 3/2 g^2 b/(a dd) +
    3 (b (k - a)^2 - 2 c^2 (k - a) + k c^2)/(a r^2 dd)];
angD108[a_, k_, c_, b_, r_] := Module[{dd = k b - c^2}, 3 (b (k - a)^2 - 2 c^2 (k - a) + k c^2)/(a r^2 dd)];
om2 = r^2/4 (gy - g be)^2 + 2 g^2 be^2/a + 4 a be^2/r^2;
c0 = 3/4 (2 a + k) + 3/4 k be^2 + 3/16 r^2 (2 g^2/a + g^2/k) + 3/4 g^2/a^2 + 3/2 g^2/(a k) + 3 (k - a)^2/(a k r^2);
id1 = Simplify[eD108[a, k, k be, bt + k be^2, r, g, gy] - (c0 + 3/4 (bt + om2/bt))];
id6 = Simplify[angD108[a, k, k be, bt + k be^2, r] - (3 (k - a)^2/(a k r^2) + 3/4 (4 a be^2/r^2)/bt)];
Kf = {{aa[r], 0, 0, 0}, {0, aa[r], 0, 0}, {0, 0, kk[r], kk[r] bb[r]}, {0, 0, kk[r] bb[r], tt[r] + kk[r] bb[r]^2}};
brFull = 3/16 Tr[MatrixPower[Inverse[Kf] . D[Kf, r], 2]];
brSplit = 3/16 (2 (aa'[r]/aa[r])^2 + (kk'[r]/kk[r])^2 + (tt'[r]/tt[r])^2) + 3 kk[r] bb'[r]^2/(8 tt[r]);
id2 = Simplify[brFull - brSplit];
psi = (w[r]/Pi)^(1/4) Exp[-w[r] y^2/2];
id3 = Simplify[Integrate[1/2 D[psi, r]^2, {y, -Infinity, Infinity}, Assumptions -> w[r] > 0] - 1/16 (w'[r]/w[r])^2,
   Assumptions -> w[r] > 0];
phi = (k/Pi)^(1/4) Exp[-k (x + be ww)^2/2];
id4 = Simplify[Integrate[1/2 D[phi, ww]^2, {x, -Infinity, Infinity}, Assumptions -> k > 0] - k be^2/4,
   Assumptions -> k > 0];
phir = (kk[r]/Pi)^(1/4) Exp[-kk[r] (x + bb[r] ww)^2/2];
id5 = Simplify[Integrate[1/2 D[phir, r]^2, {x, -Infinity, Infinity}, Assumptions -> kk[r] > 0] -
    (kk[r]/4 bb'[r]^2 ww^2 + 1/16 (kk'[r]/kk[r])^2), Assumptions -> kk[r] > 0];
(* numeric: radial 3D s-wave oscillator, v = s chi, -1/2 v'' + 1/2 W^2 s^2 v, v(0) = v(L) = 0 *)
W0 = 3/10;
numEv = NDEigenvalues[{-1/2 v''[s] + 1/2 W0^2 s^2 v[s], DirichletCondition[v[s] == 0, True]}, v[s], {s, 0, 14}, 1,
   Method -> {"SpatialDiscretization" -> {"FiniteElement", {"MeshOptions" -> {"MaxCellMeasure" -> 0.005}}}}];
out = <|"id1_E_decomposition" -> ToString[id1, InputForm], "id2_BornHuang_split" -> ToString[id2, InputForm],
   "id3_oscillator_BH" -> ToString[id3, InputForm], "id4_BH_w" -> ToString[id4, InputForm],
   "id5_BH_r_mean" -> ToString[id5, InputForm], "id6_angular_split" -> ToString[id6, InputForm],
   "num_3D_oscillator_E0" -> N[First[numEv], 12], "num_expected_1.5W" -> N[3/2 W0, 12],
   "version" -> $Version, "date" -> DateString[]|>;
Export[FileNameJoin[{res, "wolfram_C140.json"}], out, "JSON"];
Print[out];
