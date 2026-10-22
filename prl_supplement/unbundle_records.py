"""Restore the point-by-point records of this Supplemental Material and verify their SHA-256.

The resumable chains of the static-box calculation (results/C170/S1/chains, S2/chains, S3/chains) and the per-task
records and run logs of the abelian Higgs calculation (results/C171/raw, results/C171/*_log*.txt) are stored as two
JSON-Lines bundles, results/C170/C170_point_records.jsonl and results/C171/C171_point_records.jsonl.  Each line is
{"path": <path relative to the bundle's folder>, "bytes": <size>, "sha256": <of the file>, "text": <file content>}.
Usage: python -B unbundle_records.py [--check]      (--check: verify only, write nothing)
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLES = (os.path.join("results", "C170", "C170_point_records.jsonl"),
           os.path.join("results", "C171", "C171_point_records.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the checksums only")
    args = ap.parse_args()
    n = bad = 0
    for rel in BUNDLES:
        bundle = os.path.join(HERE, rel)
        base = os.path.dirname(bundle)
        with open(bundle, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                data = rec["text"].encode("utf-8")
                ok = len(data) == rec["bytes"] and hashlib.sha256(data).hexdigest() == rec["sha256"]
                n += 1
                if not ok:
                    bad += 1
                    print("checksum mismatch:", rec["path"])
                    continue
                if not args.check:
                    dst = os.path.join(base, *rec["path"].split("/"))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, "wb") as out:
                        out.write(data)
    print(f"{n} records, {bad} checksum mismatches" + ("" if args.check else ", restored under results/"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
