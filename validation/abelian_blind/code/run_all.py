"""One-command reproduction of the blind C171 calculation.

    D:\\Python\\envs\\py312-main\\python.exe run_all.py [--fresh]

Stages (each writes JSON next to this folder, in ..):
  1. run_methodB.py  -> methodB_results.json  (HO basis + complex rotation, ~1.5 min)
  2. run_methodA.py  -> methodA_results.json  (ECS finite-difference grid + flux, ~6 min on 3 processes)
  3. slope_scan.py   -> slope_results.json    (optional item: Gamma(lambda), local slope, ~2 min)
  4. collect.py      -> blind_results.json    (summary table, scaling check)
  5. make_tables.py  -> convergence_tables.md, METHOD.md (= METHOD_head.md + tables)
--fresh deletes existing JSON results first (otherwise method A resumes and skips finished runs).
"""
import os, sys
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")   # conda MKL threading layer crashes in SuperLU
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if __name__ == "__main__":
    if "--fresh" in sys.argv:
        for f in ["methodA_results.json", "methodB_results.json", "slope_results.json", "blind_results.json"]:
            p = os.path.join(HERE, "..", f)
            if os.path.exists(p):
                os.remove(p)
    import run_methodB, run_methodA, slope_scan, collect, make_tables
    run_methodB.main()
    run_methodA.main()
    slope_scan.main()
    collect.main()
    make_tables.main()
