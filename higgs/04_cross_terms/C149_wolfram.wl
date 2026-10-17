(* PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/holonomy_cross/C149_wolfram.wl (copied 2026-10-15 by code-higgs); optional Wolfram check of r = 0 curvatures *)
(* C149 Wolfram checks (holonomy-cross, D121).  Independent of the Python code: shells via SquaresR[3,q]. *)
qmax = 3600;
qs = Range[1, qmax];
r3 = SquaresR[3, #] & /@ qs;
sg = (-1)^qs;
(* curvature per real dof, C(m) = (1/3) m^2 sum'_w K2(m|w|) [x (-1)^{|w|^2} AAA]; periodic: - 2 Pi^2/m (n = 0 slow mode) *)
cA[m_] := (m^2/3) Total[sg r3 BesselK[2, m Sqrt[N[qs]]]];
cP[m_] := (m^2/3) Total[r3 BesselK[2, m Sqrt[N[qs]]]] - 2 Pi^2/m;
res = <||>;
res["C_A(1)"] = N[cA[1.], 12];
res["C_P(1)"] = N[cP[1.], 12];
res["C_A(2)"] = N[cA[2.], 12];
res["C_P(2)"] = N[cP[2.], 12];
(* top mass at r = 3 (v11 y_t = 0.4585): m_t = y_t r/Sqrt[2]; SU(3)-top shift DeltaU_3 = -(3.66/(4 Pi^2)) (C_A(m_t) - C_A(0)) *)
mt3 = 0.4585*3/Sqrt[2.];
(* C_A(0) = (2/3) Z(1; AAA): Z(1;AAA) from the small-m limit of the Bessel sum is slow; use the Madelung-type Ewald value *)
zA1 = -2.519356152089445;
res["m_t(r=3)"] = mt3;
res["C_A(m_t(r=3))"] = N[cA[mt3], 12];
res["DeltaU3(r=3)"] = N[-(3.66/(4 Pi^2)) (cA[mt3] - (2/3) zA1), 10];
(* v11 curvatures from Epstein values (closed forms) *)
zP1 = -8.91363291758515;
res["kappa_SU3"] = N[(zP1 - 2 zA1)/Pi^2, 10];
res["kappa_SU2"] = N[(5 zP1 - 12 zA1)/(6 Pi^2), 10];
res["kappa_Y"] = N[(zP1 - 20 zA1)/(6 Pi^2), 10];
(* direction-averaged massless AAA energy per real dof, Ebar_A(t) = -(1/2Pi^2) sum'_w (-1)^q j0(2 Pi |w| t)/|w|^4 *)
eb[t_] := -(1/(2 Pi^2)) Total[sg r3 Sinc[2 Pi Sqrt[N[qs]] t]/N[qs]^2];
res["Ebar_A(0)"] = N[eb[0.], 12];
res["Ebar_A(0.1)-Ebar_A(0)"] = N[eb[0.1] - eb[0.], 12];
res["Ebar_A(0.3)-Ebar_A(0)"] = N[eb[0.3] - eb[0.], 12];
(* quadratic check: Ebar_A(t) - Ebar_A(0) ~ (1/2) C_A(0) t^2 *)
res["half_CA0_t2(0.1)"] = N[(1/2) (2/3) zA1 0.01, 12];
Export[FileNameJoin[{DirectoryName[$InputFileName], "C149_wolfram_out.json"}], res, "JSON"];
Print[ToString[res, InputForm]];
