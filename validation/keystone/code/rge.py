# keystone-blind (T3): two-loop SM RGE (Buttazzo et al. 2013, arXiv:1307.3536v4, App. B, y_b=y_tau=0),
# thresholds at mu=M_t from Eqs. (55),(57)-(60); 3-loop calibration Delta_lambda from Eq. (61e).
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

K1 = 1.0 / (16 * np.pi**2); K2 = K1**2
MPL_RED = 2.435e18
MU_B = MPL_RED / 722.0
MPL = 1.2209e19
HLB = 0.00437243

def beta(tau, y):
    """d/d ln mu^2 of (g1^2, g2^2, g3^2, yt^2, lam); g1 GUT-normalised."""
    G1, G2, G3, Yt, lam = y
    dG1 = G1**2 * (K1 * 41 / 10 + K2 * (44 * G3 / 5 + 27 * G2 / 10 + 199 * G1 / 50 - 17 * Yt / 10))
    dG2 = G2**2 * (K1 * (-19 / 6) + K2 * (12 * G3 + 35 * G2 / 6 + 9 * G1 / 10 - 3 * Yt / 2))
    dG3 = G3**2 * (K1 * (-7) + K2 * (-26 * G3 + 9 * G2 / 2 + 11 * G1 / 10 - 2 * Yt))
    dYt = Yt * (K1 * (9 * Yt / 2 - 8 * G3 - 9 * G2 / 4 - 17 * G1 / 20)
                + K2 * (Yt * (-12 * Yt - 12 * lam + 36 * G3 + 225 * G2 / 16 + 393 * G1 / 80)
                        + 6 * lam**2 - 108 * G3**2 - 23 * G2**2 / 4 + 1187 * G1**2 / 600
                        + 9 * G3 * G2 + 19 / 15 * G3 * G1 - 9 / 20 * G2 * G1))
    b1 = (lam * (12 * lam + 6 * Yt - 9 * G2 / 2 - 9 * G1 / 10) - 3 * Yt**2
          + 9 * G2**2 / 16 + 27 * G1**2 / 400 + 9 * G2 * G1 / 40)
    b2 = (lam**2 * (-156 * lam - 72 * Yt + 54 * G2 + 54 * G1 / 5)
          + lam * Yt * (-3 * Yt / 2 + 40 * G3 + 45 * G2 / 4 + 17 * G1 / 4)
          + lam * (-73 * G2**2 / 16 + 1887 * G1**2 / 400 + 117 * G2 * G1 / 40)
          + Yt**2 * (15 * Yt - 16 * G3 - 4 * G1 / 5)
          + Yt * (-9 * G2**2 / 8 - 171 * G1**2 / 200 + 63 * G2 * G1 / 20)
          + 305 * G2**3 / 32 - 3411 * G1**3 / 4000 - 289 * G2**2 * G1 / 160 - 1677 * G2 * G1**2 / 800)
    dlam = K1 * b1 + K2 * b2
    return [dG1, dG2, dG3, dYt, dlam]

def thresholds(Mt, als, Mh, MW):
    lam = 0.12604 + 0.00206 * (Mh - 125.15) - 0.00004 * (Mt - 173.34)
    yt = 0.93690 + 0.00556 * (Mt - 173.34) - 0.00042 * (als - 0.1184) / 0.0007
    g2 = 0.64779 + 0.00004 * (Mt - 173.34) + 0.00011 * (MW - 80.384) / 0.014
    gY = 0.35830 + 0.00011 * (Mt - 173.34) - 0.00020 * (MW - 80.384) / 0.014
    g3 = 1.1666 + 0.00314 * (als - 0.1184) / 0.0007 - 0.00046 * (Mt - 173.34)
    return dict(lam=lam, yt=yt, g2=g2, gY=gY, g3=g3)

def lam61e(Mt, als, Mh):
    return -0.0143 - 0.0066 * (Mt - 173.34) + 0.0018 * (als - 0.1184) / 0.0007 + 0.0029 * (Mh - 125.15)

def run(Mt, als, Mh, MW, mu_out, dense=False):
    c = thresholds(Mt, als, Mh, MW)
    y0 = [5 / 3 * c["gY"]**2, c["g2"]**2, c["g3"]**2, c["yt"]**2, c["lam"]]
    t0 = 2 * np.log(Mt); t1 = 2 * np.log(mu_out)
    sol = solve_ivp(beta, (t0, t1), y0, method="DOP853", rtol=1e-12, atol=1e-15, dense_output=dense)
    return sol

def couplings_from_y(y):
    G1, G2, G3, Yt, lam = y
    return dict(gY=float(np.sqrt(3 / 5 * G1)), g2=float(np.sqrt(G2)), g3=float(np.sqrt(G3)), yt=float(np.sqrt(Yt)), lam=float(lam))

CAL = [(173.34, 0.1184, 125.15), (172.60, 0.1180, 125.13), (172.10, 0.1180, 125.13), (173.34, 0.1191, 125.15), (172.60, 0.1171, 125.13)]

def delta_lambda():
    rows = []
    for Mt, als, Mh in CAL:
        s = run(Mt, als, Mh, 80.384, MPL)
        l2 = s.y[4, -1]
        rows.append(dict(Mt=Mt, als=als, Mh=Mh, lam2loop_MPl=float(l2), lam61e=lam61e(Mt, als, Mh), diff=lam61e(Mt, als, Mh) - float(l2)))
    return float(np.mean([r["diff"] for r in rows])), rows

U_NODES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

def mu_of_u(u):
    return MU_B / np.sqrt(np.cosh(2 * u))

def couplings_at_nodes(Mt, als, Mh=125.13, MW=80.3692, dl=0.0, nodes=U_NODES):
    s = run(Mt, als, Mh, MW, MU_B, dense=True)
    out = []
    for u in nodes:
        y = s.sol(2 * np.log(mu_of_u(u)))
        c = couplings_from_y(y); c["lam"] += dl; c["u"] = u; c["mu"] = float(mu_of_u(u))
        out.append(c)
    return out

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    dl, rows = delta_lambda()
    for r in rows: print(r)
    print("Delta_lambda =", dl)
    res = dict(Delta_lambda=dl, calibration=rows, mu_b=MU_B, MPl=MPL)
    # reference point
    ref = couplings_at_nodes(172.6, 0.1180, dl=dl)
    print("ref (172.6,0.1180) at mu_b:", ref[0])
    res["ref_172.6_0.1180"] = ref
    grid = {}
    for als in [0.1180, 0.1171, 0.1189]:
        for Mt in np.round(np.arange(171.0, 174.61, 0.1), 2):
            grid[f"{als:.4f}_{Mt:.2f}"] = couplings_at_nodes(float(Mt), als, dl=dl)
    res["grid"] = grid
    # 1-loop-only sanity: lambda at mu_b for 172.6 without Delta
    s = run(172.6, 0.1180, 125.13, 80.3692, MU_B)
    res["lam2loop_mub_nodelta_172.6"] = float(s.y[4, -1])
    print("lam(mu_b) 2-loop no Delta (172.6):", s.y[4, -1], " with Delta:", s.y[4, -1] + dl)
    for Mt in [172.2, 172.6, 173.0]:
        c = grid[f"0.1180_{Mt:.2f}"][0]
        print(Mt, "mu_b:", {k: round(v, 6) for k, v in c.items()})
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "couplings.json"), "w"), indent=1)
