"""Copy the inputs of the new PRL Fig. 1 from the C170/C171 result folders into ./data_c170 (with SHA-256 in
SOURCES_c170.json), so that fig_prl_fig1_mechanism.py reads only files in this folder, as the published figure
scripts do.  Usage: python -I -B prepare_data_c170.py
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"  # [supplement] results of this supplement
DST = HERE / "data_c170"
FILES = {
    "C170/S1/C170_S1_summary.json": "S1_summary.json",
    "C170/S1/C170_S1_lambda_c_N.csv": "S1_lambda_c_N.csv",
    "C170/S1/C170_S1_lambdaSM_Nstar.csv": "S1_lambdaSM_Nstar.csv",
    "C170/S1/C170_lambdaSM_running.csv": "lambdaSM_running.csv",
    "C170/S1/C170_S1_gamma_lambda.csv": "S1_gamma_lambda.csv",
    "C170/S2/C170_S2_potential_lam-0.012.csv": "S2_potential_lam-0.012.csv",
    "C170/S2/C170_S2_lambda_c_kappa.csv": "S2_lambda_c_kappa.csv",
    "C170/S2/C170_S2_summary.json": "S2_summary.json",
    "C170/S2/C170_S2_curves.json": "S2_curves.json",
    "C170/S3/C170_S3_coupling_scan.csv": "S3_coupling_scan.csv",
    "C170/S3/C170_S3_summary.json": "S3_summary.json",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    DST.mkdir(exist_ok=True)
    src_map = {}
    for rel, name in FILES.items():
        src = RES / rel
        if not src.exists():
            print("missing (skipped):", rel)
            continue
        shutil.copy2(src, DST / name)
        src_map[name] = {"source": f"results/{rel}", "sha256": sha(src)}
        print("copied", rel, "->", name)
    (DST / "SOURCES_c170.json").write_text(json.dumps(src_map, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
