# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/mk_wolfram.py (copied 2026-10-15 by code-higgs)
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Optional Wolfram cross-check of the SM mapping (C148_wolfram.wl; env WOLFRAMSCRIPT).

C148: Wolfram (Mathematica 15) cross-checks, run via wolframscript; code + output stored in wolfram_receipts.json.
 W1: exact symbolic z_m(mu) recursion (exact beta, gamma for nf=6) -> P4 log coefficients; c_n from z_n by Series.
 W2: alpha_s^(6)(172.4) from alpha_s^(5)(MZ)=0.1180 (5-loop NDSolve, 4-loop decoupling at Mt) vs Python.
 W3: NDSolve of the full SM system (expressions auto-converted from SMDR betas.c) from the central matching at
     (172.6, 0.1180, 125.13) to mu_b; compare lambda, yt, g2, gY, g3 with the Python provider."""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
import re
import sys
import json
import math
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import r3util as U  # noqa: E402
import sm_rge as RG  # noqa: E402
import sm_provider as SP  # noqa: E402
import qcd as Q  # noqa: E402

WS = os.environ.get("WOLFRAMSCRIPT", "wolframscript")
WL = os.path.join(HERE, "C148_wolfram.wl")


def smdr_exprs():
    src = open(RG.BETAS_C, encoding="utf-8", errors="replace").read()
    txt = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    body = txt[txt.index("int SMDR_Betas"):]
    want = set(RG.AUX) | {n for v in RG.ORD.values() for n in v if n}
    ex = {}
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;=]+);", body):
        if m.group(1) in want and m.group(1) not in ex:
            ex[m.group(1)] = " ".join(m.group(2).split())
    return ex


def wl_number(e):
    # C literals like 4166./3. or 38.25 -> Mathematica-safe (insert 0 after trailing dot)
    return re.sub(r"(\d)\.(?!\d)", r"\1.0", e)


def build_wl(y0, mu0, mub):
    ex = smdr_exprs()
    loc = ["g2", "g4", "g6", "g8", "g32", "g34", "g36", "g38", "gp2", "gp4", "gp6", "gp8", "yt2", "yt4", "yt6", "yt8",
           "yb2", "yb4", "yb6", "yb8", "ytau2", "ytau4", "ytau6", "ytau8", "k2", "k3", "k4", "yc", "ys", "yu", "yd",
           "ymu", "ye"] + RG.AUX + [n for v in RG.ORD.values() for n in v if n]
    defs = ["g2=g*g", "g4=g2*g2", "g6=g4*g2", "g8=g4*g4", "g32=g3*g3", "g34=g32*g32", "g36=g34*g32", "g38=g34*g34",
            "gp2=gp*gp", "gp4=gp2*gp2", "gp6=gp4*gp2", "gp8=gp4*gp4", "yt2=yt*yt", "yt4=yt2*yt2", "yt6=yt4*yt2",
            "yt8=yt4*yt4", "yb2=yb*yb", "yb4=yb2*yb2", "yb6=yb4*yb2", "yb8=yb4*yb4", "ytau2=ytau*ytau",
            "ytau4=ytau2*ytau2", "ytau6=ytau4*ytau2", "ytau8=ytau4*ytau4", "k2=k*k", "k3=k2*k", "k4=k2*k2",
            "yc=0", "ys=0", "yu=0", "yd=0", "ymu=0", "ye=0"]
    for a in RG.AUX:
        defs.append("%s=%s" % (a, wl_number(ex[a])))
    tot = []
    for n in range(1, 6):
        for nm in RG.ORD[n]:
            if nm:
                defs.append("%s=%s" % (nm, wl_number(ex[nm])))
        tot.append("K^%d*{%s}" % (n, ",".join(nm if nm else "0" for nm in RG.ORD[n])))
    W = []
    W.append('SetOptions[$Output, PageWidth -> Infinity]; Zeta3 = N[Zeta[3], 20]; K = 1/(16 Pi^2);')
    # W1 symbolic z_m
    W.append(r'''nf = 6; z3v = Zeta[3]; z4v = Zeta[4]; z5v = Zeta[5];
bas = {11 - 2 nf/3, 102 - 38 nf/3, 2857/2 - 5033 nf/18 + 325 nf^2/54, (149753/6 + 3564 z3v) - (1078361/162 + 6508 z3v/27) nf + (50065/162 + 6472 z3v/81) nf^2 + 1093 nf^3/729};
bp = Table[bas[[n + 1]]/4^(n + 1), {n, 0, 3}];
gm = {1, (202/3 - 20 nf/9)/16, (1249 + (-2216/27 - 160 z3v/3) nf - 140 nf^2/81)/64, (4603055/162 + 135680 z3v/27 - 8800 z5v + (-91723/27 - 34192 z3v/9 + 880 z4v + 18400 z5v/9) nf + (5242/243 + 800 z3v/9 - 160 z4v/3) nf^2 + (-332/243 + 64 z3v/27) nf^3)/256};
Clear[l, a, zz]; P[0] = 1;
Do[P[n] = zz[n] + Integrate[Sum[k bp[[n - k]] P[k], {k, 1, n - 1}] - Sum[gm[[n - j]] P[j], {j, 0, n - 1}] /. l -> s, {s, 0, l}], {n, 1, 4}];
p4 = CoefficientList[Expand[P[4] /. {zz[1] -> -4/3, zz[2] -> z2n, zz[3] -> z3n}], l];
X = 1 + Sum[x[i] a^i, {i, 1, 4}];
eqs = Series[Sum[a^k (P[k] /. l -> 2 Log[X]), {k, 0, 4}] - X, {a, 0, 4}];
cl = CoefficientList[Normal[eqs], a];
sol = {}; Do[sol = Join[sol, Solve[(cl[[n + 1]] /. sol) == 0, x[n]][[1]]], {n, 1, 4}];
cser = CoefficientList[Normal[Series[1/(X /. sol), {a, 0, 4}]], a];
csub = {zz[1] -> -4/3, zz[2] -> z2n, zz[3] -> z3n, zz[4] -> -859.96};
z2s = z2n /. Solve[(cser[[3]] /. csub) == 13.443433 - 1.041367*5, z2n][[1]];
z3s = z3n /. Solve[(cser[[4]] /. csub /. z2n -> z2s) == 190.595 - 26.655*5 + 0.6527*25, z3n][[1]];
c4v = cser[[5]] /. csub /. {z2n -> z2s, z3n -> z3s};
p4n = N[p4 /. {z2n -> z2s, z3n -> z3s}, 12];
Print["W1 ", ToString[{N[z2s, 10], N[z3s, 10], N[c4v, 10], Rest[p4n]}, InputForm]];''')
    # W2 alpha_s
    W.append(r'''b5[nf_] := {11 - 2 nf/3, 102 - 38 nf/3, 2857/2 - 5033 nf/18 + 325 nf^2/54, (149753/6 + 3564 Zeta[3]) - (1078361/162 + 6508 Zeta[3]/27) nf + (50065/162 + 6472 Zeta[3]/81) nf^2 + 1093 nf^3/729, 8157455/16 + 621885/2 Zeta[3] - 88209/2 Zeta[4] - 288090 Zeta[5] + nf (-336460813/1944 - 4811164/81 Zeta[3] + 33935/6 Zeta[4] + 1358995/27 Zeta[5]) + nf^2 (25960913/1944 + 698531/81 Zeta[3] - 10526/9 Zeta[4] - 381760/81 Zeta[5]) + nf^3 (-630559/5832 - 48722/243 Zeta[3] + 1618/27 Zeta[4] + 460/9 Zeta[5]) + nf^4 (1205/2916 - 152/81 Zeta[3])};
runA[a0_, mu0_, mu1_, nf_] := Module[{bb = N[b5[nf]/4^Range[5], 30], s}, s = NDSolveValue[{A'[t] == -2 Sum[bb[[n + 1]] A[t]^(n + 2), {n, 0, 4}], A[Log[mu0]] == a0}, A, {t, Log[mu0], Log[mu1]}, WorkingPrecision -> 25, PrecisionGoal -> 18, AccuracyGoal -> 22, MaxSteps -> 10^6]; s[Log[mu1]]];
zP = Table[P[n] /. {zz[1] -> -4/3, zz[2] -> z2s, zz[3] -> z3s, zz[4] -> -859.96}, {n, 0, 4}];
zmv[av_, lv_] := Sum[av^k (zP[[k + 1]] /. l -> lv), {k, 0, 4}];
Zas[aS_, L_] := Module[{TF = 1/2, CF = 4/3, nq = 5}, 1 + 4/3 TF aS L + 16/9 TF^2 aS^2 L^2 - TF aS CF aS (4 L + 13/3) + TF aS^2 (20 L + 32/3) + aS^3 (8 L^3/27 + (53/9 - 16 nq/9) L^2 + (955/9 - 67 nq/9) L + 62.211628 - 2633 nq/486) + aS^4 (16 L^4/81 + (3766/81 + 508 nq/81 - 64 nq^2/81) L^3 + (4354/27 - 2966 nq/81 - 77 nq^2/81) L^2 + (2157.863053 - 335.316171 nq - 6865 nq^2/729) L + 1323.608830 - 258.542470 nq - 5.62646 nq^2)];
Mt0 = 172.4`25; a5 = runA[0.1180`25/Pi, 91.1876`25, Mt0, 5];
a6 = FixedPoint[Function[av, a5/Zas[av/4, Log[(Mt0 zmv[av, 0])^2/Mt0^2]]], a5, 200, SameTest -> (Abs[#1 - #2] < 10^-22 &)];
Print["W2 ", ToString[N[Pi a6, 15], InputForm]];''')
    # W3 full SM NDSolve
    W.append("bet[g3_, g_, gp_, yt_, yb_, ytau_, k_] := Module[{%s}, %s; Total[{%s}]];" % (
        ",".join(loc), "; ".join(defs), ",".join(tot)))
    ys = ["G3", "G", "GP", "YT", "YB", "YTAU", "LAM", "V"]
    rhs = ("b = bet[G3[t], G[t], GP[t], YT[t], YB[t], YTAU[t], LAM[t]]; ")
    W.append(r'''eqs3 = Join[Table[{G3, G, GP, YT, YB, YTAU, LAM}[[i]]'[t] == bet[G3[t], G[t], GP[t], YT[t], YB[t], YTAU[t], LAM[t]][[i]], {i, 1, 7}], {V'[t] == -V[t] bet[G3[t], G[t], GP[t], YT[t], YB[t], YTAU[t], LAM[t]][[8]]}];
ics = {G3[%s] == %s, G[%s] == %s, GP[%s] == %s, YT[%s] == %s, YB[%s] == %s, YTAU[%s] == %s, LAM[%s] == %s, V[%s] == %s};
sol3 = NDSolve[Join[eqs3, ics], {G3, G, GP, YT, YB, YTAU, LAM, V}, {t, %s, %s}, PrecisionGoal -> 12, AccuracyGoal -> 14, MaxSteps -> 10^6, Method -> "ExplicitRungeKutta"];
Print["W3 ", ToString[N[{LAM[%s], YT[%s], G[%s], GP[%s], G3[%s]} /. sol3[[1]], 14], InputForm]];''' % tuple(
        sum([[repr(math.log(mu0)), repr(float(v))] for v in y0], []) + [repr(math.log(mu0)), repr(math.log(mub))]
        + [repr(math.log(mub))] * 5))
    with open(WL, "w", encoding="utf-8") as f:
        f.write("\n".join(W) + "\n")
    return "\n".join(W)


def main():
    Mt = 172.6
    sol, y0, mu0 = SP._solution(Mt, 0.1180, 125.13, SP.MW_PDG, "central")
    code = build_wl(list(y0), mu0, SP.MU_B)
    t0 = __import__("time").time()
    p = subprocess.run([WS, "-file", WL], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=1800)
    py = SP.coup_point(Mt, 0.1180, 125.13, SP.MU_B)
    zc = Q.zconst()
    rec = dict(code_file=WL, code_sha256=U.sha256(WL), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
               time_s=__import__("time").time() - t0,
               python=dict(W1=[zc["z"][2], zc["z"][3], zc["c"][4]], W2=Q.as6_at(172.4, 0.1180),
                           W3=[py["lam"], py["yt"], py["g2"], py["gY"], py["g3"]]))
    U.wjson("wolfram_receipts.json", rec)
    print(p.stdout[-3000:])
    print("STDERR:", p.stderr[-1500:])
    print("python:", json.dumps(rec["python"]))


if __name__ == "__main__":
    main()
