# export_for_wolfram.py -- writes coupling table + target point for the independent Wolfram check of B_HM
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import smhiggs as S

d = json.load(open(os.path.join(os.path.dirname(HERE), "results", "part1", "v11_172.6_base.json"), encoding="utf-8"))
xi = d["xi_EW"]; c = xi - 1 / 6
R = d["info"]["R_peak"]
c0 = S.buttazzo_matching(172.6, 125.13, 0.1180)
run = S.SMRun(172.6, c0, nloop=3)
ve = S.Veff(run, c)
B, hb, mus = ve.B_HM(R)
# table on s in [ln((1e11)^2), ln((1e15)^2)]
sg = np.linspace(2 * np.log(1e11), 2 * np.log(1e15), 801)
rows = []
for s in sg:
    G1, G2, G3, Yt, L = run.coup(s)
    rows.append([s, G1, G2, Yt, L, run.K(s, 1), run.alpha(s, c, 1)])
np.savetxt(os.path.join(HERE, "wolfram_table.csv"), np.array(rows), delimiter=",", fmt="%.17g")
tgt = dict(R=R, c=c, xi_EW=xi, B_python=B, hbar_python=hb, mustar_python=mus, V0_python=ve.V(0.0, R),
           Vbar_python=ve.V(hb, R))
json.dump(tgt, open(os.path.join(HERE, "wolfram_target.json"), "w"), indent=1)
print(json.dumps(tgt, indent=1))
