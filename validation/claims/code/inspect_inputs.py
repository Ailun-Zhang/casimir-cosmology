# claims-blind: inspect structure of keystone-blind result files (read-only)
import sys, os, json, glob
sys.stdout.reconfigure(encoding="utf-8")
KB = r"D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\keystone_blind"
C = json.load(open(os.path.join(KB, "results", "couplings.json")))
print("couplings keys:", list(C.keys())[:10])
for k, v in C.items():
    if isinstance(v, list):
        print(k, "list", len(v), v[0] if v else None, v[-1] if v else None)
    elif isinstance(v, dict):
        ks = list(v.keys()); print(k, "dict", len(ks), ks[:5], ks[-5:])
        x = v[ks[0]]; print("   sample", ks[0], type(x).__name__, (x[0] if isinstance(x, list) else x))
    else:
        print(k, v)
H = json.load(open(os.path.join(KB, "results", "hazard_summary.json")))
print("hazard_summary keys:", list(H.keys()))
for k, v in H.items():
    if isinstance(v, dict):
        print(" ", k, list(v.keys())[:12])
    elif isinstance(v, list):
        print(" ", k, "list", len(v), v[:2])
crit = H.get("critical", {})
for k in list(crit.keys())[:8]:
    print("critical", k, json.dumps(crit[k])[:600])
for k in ["P1", "P1_grid", "table", "rows"]:
    if k in H: print(k, json.dumps(H[k])[:1500])
print("v2 cache:", sorted(os.path.basename(f) for f in glob.glob(os.path.join(KB, "v2", "cache", "*")))[:40])
sh = sorted(glob.glob(os.path.join(KB, "v2", "hazard_shards_all", "*.json")))
print("v2 all shards:", len(sh), [os.path.basename(f) for f in sh[:3]], [os.path.basename(f) for f in sh[-3:]])
if sh:
    x = json.load(open(sh[0])); print("shard keys", list(x.keys())); print({k: x[k] for k in ["als", "Mt", "inode", "u", "mu", "lam", "yt"] if k in x}); print("2D", x.get("2D", {}).get("Q"), x.get("2D", {}).get("gamma"), "V1D", x.get("V1D", {}).get("gamma"))
B = json.load(open(os.path.join(KB, "results", "BLIND_RESULTS.json")))
print("BLIND_RESULTS keys:", list(B.keys()))
for k in ["dlam", "Delta_lambda", "couplings_mub", "lam_mub", "yt_mub", "inputs"]:
    if k in B: print(k, json.dumps(B[k])[:800])
