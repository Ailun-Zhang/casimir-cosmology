# -*- coding: utf-8 -*-
"""C170 S1: static boxes with the run05 final model, lambda varied freely at fixed (g2, gY, y_t, g3).

Boxes (couplings of (m_t, alpha_s, M_h) = (172.60, 0.1180, 125.13) at mu = 1/L, sm_provider 'central'):
  A mu = mu_b (node u = 0), B mu_b/10, C mu_b/100, D 10 mu_b, E mu_b/1000 (coarser grid).
lambda grid in x = |lambda|^(-1/2); every box also contains lambda_SM(mu) for m_t = 172.2, 172.6, 173.0.
Two continuation chains per box start at lambda_SM(mu; 172.6) (box A: anchored on the archived run05 root):
'up' (x increasing, towards lambda -> 0-) and 'down' (x decreasing, towards more negative lambda; stops after two
consecutive points without a narrow resonance).  Chains resume from their JSON files in results/.../C170/S1/chains.
Usage: python -B c170_s1.py run [nw] | conv [nw]
"""
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c170_core as CC  # noqa: E402
import numpy as np  # noqa: E402

BOXF = {"A": 1.0, "B": 0.1, "C": 0.01, "D": 10.0, "E": 0.001}
MTS = (172.2, 172.6, 173.0)
ALS = 0.1180
XFULL = sorted(set([5.0, 5.5, 5.75, 6.0, 6.5, 7.0] + [round(7.25 + 0.25 * i, 2) for i in range(20)]
                   + [12.5, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]))
XCOARSE = [5.5, 5.75, 6.0, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]   # 5.75, 6.0 added (cease refinement)
XEXTRA = {"E": [14.0, 16.0, 18.0, 19.0], "D": [13.5, 14.5, 16.0, 17.0, 18.0]}
CHAINS = os.path.join(CC.OUT, "S1", "chains")


def boxes():
    mub = CC.mu_node(0.0)
    out = {}
    for b, f in BOXF.items():
        mu = mub * f
        c = CC.couplings_at(172.6, ALS, mu)
        lsm = {"%.1f" % mt: CC.couplings_at(mt, ALS, mu)["lam"] for mt in MTS}
        out[b] = dict(mu_GeV=mu, mu_over_mu_b=f, couplings=c, lam_SM=lsm)
    return out


def grid(b, B):
    xs = list(XFULL if b in ("A", "B", "C") else XCOARSE + XEXTRA[b])
    lsm = [v for v in B["lam_SM"].values() if v < 0]
    lams = sorted(set([round(CC.lam_of_x(x), 12) for x in xs] + [round(v, 15) for v in lsm]))
    return lams


def specs():
    B = boxes()
    arch = CC.jload(os.path.join(CC.WORK, "archive", "r3", "holonomy_cross", "run05", "shards", "X_main_u0.0000.json"))
    a0 = [x for x in arch["cols"] if round(x["mt"], 2) == 172.6][0]
    out = []
    for b, bx in B.items():
        c = bx["couplings"]
        base = dict(model="final", g2=c["g2"], gY=c["gY"], yt=c["yt"], g3=c["g3"])
        lams = grid(b, bx)
        l0 = bx["lam_SM"]["172.6"]
        up = [l0] + sorted([l for l in lams if l > l0 + 1e-15])                       # towards 0-
        dn = [l0] + sorted([l for l in lams if l < l0 - 1e-15], reverse=True)        # more negative
        anchor = None
        if b == "A":
            assert abs(a0["lam"] - l0) < 1e-15 and abs(a0["g2"] - c["g2"]) < 1e-15
            anchor = dict(eps=a0["Q"], gamma=a0["gamma"])
        for tag, ls in (("up", up), ("down", dn)):
            out.append(dict(name="S1_%s_%s" % (b, tag), base=base, lams=ls, anchor=anchor, stop_after_fail=2,
                            outdir=CHAINS, box=b))
    return out, B


def run(nw=10):
    t0 = time.time()
    sp, B = specs()
    CC.wjson(os.path.join(CC.OUT, "S1", "boxes.json"), B)
    CC.status_update({"stage": "S1 running (%d chains)" % len(sp),
                      "running": ["S1 chains: " + ", ".join(s["name"] for s in sp)],
                      "resume": "cd code/.../C170/higgs/c170; python -B c170_s1.py run 10   (chains resume from "
                                "results/.../C170/S1/chains/*.json)"})
    with ProcessPoolExecutor(max_workers=max(1, min(nw, 14, len(sp)))) as ex:
        res = list(ex.map(CC.run_chain, sp))
    summ = {r["name"]: dict(n=len(r["points"]), ok=sum(1 for p in r["points"] if p.get("status") == "ok"),
                            last=[(p["pt"]["lam"], p.get("status")) for p in r["points"][-2:]]) for r in res}
    CC.status_update({"stage": "S1 chains done", "running": [],
                      "S1": {"state": "chains done", "chains": summ, "time_s": time.time() - t0}})
    print(json.dumps(CC.clean(summ), indent=1))


def conv(nw=6):
    """numerical convergence: BASE configuration (N, Smax, h) = (32, 14, 0.002) and Rout = 14 at selected box-A
    points, seeded with the FAST root."""
    t0 = time.time()
    fa = CC.jload(os.path.join(CHAINS, "S1_A_up.json"))["points"] + CC.jload(os.path.join(CHAINS, "S1_A_down.json"))["points"]
    sel = []
    for xt in (6.0, 7.5, 9.0, 10.5, 12.0, 14.0, 16.0, 18.0):
        cand = [p for p in fa if p.get("status") == "ok" and abs(CC.x_of_lam(p["pt"]["lam"]) - xt) < 1e-6]
        if cand:
            sel.append(cand[0])
    jobs = []
    for p in sel:
        for name, cfg in (("BASE", dict(CC.BASE)), ("Rout14", dict(CC.FAST, Rout=14.0)),
                          ("N32_S16", dict(CC.FAST, N=32, Smax=16.0))):
            jobs.append((p, name, cfg))
    with ProcessPoolExecutor(max_workers=max(1, min(nw, 14, len(jobs)))) as ex:
        outs = list(ex.map(_conv_job, jobs))
    CC.wjson(os.path.join(CC.OUT, "S1", "convergence_boxA.json"), dict(rows=outs, time_s=time.time() - t0))
    print(json.dumps(CC.clean(outs), indent=1))


def _conv_job(a):
    p, name, cfg = a
    r = CC.solve_point(dict(p["pt"]), center=p["eps"], gim=p["gamma"], cfg=cfg, window=(0.004, 0.0004), with1d=False)
    return dict(lam=p["pt"]["lam"], x=CC.x_of_lam(p["pt"]["lam"]), cfg=name, gamma_fast=p["gamma"], eps_fast=p["eps"],
                gamma=r.get("gamma"), eps=r.get("eps"), flags=r.get("flags"),
                rel_gamma=(r["gamma"] / p["gamma"] - 1) if r.get("gamma") else None,
                d_eps=(r["eps"] - p["eps"]) if r.get("eps") else None, time_s=r.get("time_s"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    md = sys.argv[1] if len(sys.argv) > 1 else "run"
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if md == "run":
        run(nw)
    elif md == "conv":
        conv(nw)
