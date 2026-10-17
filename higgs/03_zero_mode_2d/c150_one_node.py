# PROVENANCE: new verification script written 2026-10-15 by code-higgs (v12 package); it calls the production
# function C150_twod.job2d on one production node spec (unchanged code path).
"""Recompute ONE production node job of the direct 2D hazard (D122/T14b): all m_t columns of the spec
'main' (alpha_s = 0.1180, m_t = 172.0..173.8) or 'als' at hazard node u, with the coupling source and run directory
of the environment (default: NNLO central mapping, run02 = the baseline of the PRL centre).

Usage: python c150_one_node.py [main|als] [u]      -> <work>/r3/twod_direct_sm/<run>/shards2d/2Dex_<which>_u<u>.json
The shard is recomputed only if absent (job2d caches by file name); run.py deletes it first.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZM = os.path.join(os.path.dirname(HERE), "zmlib")
sys.path.insert(1, ZM)
os.environ.setdefault("C150_COUPLINGS", "module:" + os.path.join(ZM, "c150_nnlo_central.py"))
os.environ.setdefault("C150_RUN", "run02")
import C150_twod as W  # noqa: E402


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    u = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    cal, prm = W.Couplings(W.KIND).cal, W.load_prm()
    specs = [s for s in W._specs2d(which, cal, prm) if abs(float(s["u"]) - u) < 1e-9]
    if len(specs) != 1:
        raise SystemExit("no unique spec for %s u=%g" % (which, u))
    o = W.job2d(specs[0])
    print(json.dumps(W.clean([{k: x.get(k) for k in ("mt", "als", "Q2", "gamma2", "flags")} for x in o["cols"]]), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
