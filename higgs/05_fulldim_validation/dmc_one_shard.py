# PROVENANCE: new verification script written 2026-10-15 by code-higgs (v12 package); calls the production function
# T14a_dmc.run_shard on one production shard spec (deterministic SHA-256 seed -> reproducible bit for bit).
"""Recompute one production DMC shard of the walled full-dimensional validation (D123):
case m010_rw5.12 (lam = -0.010, wall r_w = 5.12), tau = 0.04, shard k = 0 (W = 10000 walkers, T_eq 10, T_prod 40).
Usage: python dmc_one_shard.py [name] [lam] [rw] [tau] [k]
Output: <work>/r3/fulldim_dmc/run01/shards/<name>_tau<tau>_k<k>.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import T14a_dmc as D  # noqa: E402


def main():
    a = sys.argv[1:]
    name = a[0] if len(a) > 0 else "m010_rw5.12"
    lam = float(a[1]) if len(a) > 1 else -0.010
    rw = float(a[2]) if len(a) > 2 else 5.12
    tau = float(a[3]) if len(a) > 3 else 0.04
    k = int(a[4]) if len(a) > 4 else 0
    os.makedirs(D.T.SHARDS, exist_ok=True)
    o = D.run_shard(dict(name=name, lam=lam, rw=rw, tau=tau, k=k))
    print(json.dumps({x: o[x] for x in ("E_mixed", "se_mixed", "E_growth", "acceptance", "seed", "wall_s")}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
