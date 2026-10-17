# cache_cmp.py -- compare own torus covariances / KL with the producer's snapshot cache (run only AFTER own numbers saved)
import os, sys, json, math, glob
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
SNAP = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\04_calculations\results\20261013_argon_codex_intake\snapshot\P5_CMB\cache'
tag = sys.argv[1] if len(sys.argv) > 1 else 's1.13000_l40_x160_t0.25'
own_files = sorted(glob.glob(os.path.join(RES, 'kl_cov_own_*.npz')))


def kl(P, Q):
    L = np.linalg.cholesky(Q)
    X = np.linalg.solve(L, np.linalg.solve(L, P).T).T
    lam = np.linalg.eigvalsh(0.5 * (X + X.T))
    return 0.5 * float(np.sum(lam - 1 - np.log(lam))), lam


out = {'tag': tag}
pf = np.load(os.path.join(SNAP, 'fossil_%s.npz' % tag), allow_pickle=False)
pc = np.load(os.path.join(SNAP, 'cube_%s.npz' % tag), allow_pickle=False)
out['producer_keys'] = {k: [list(pf[k].shape), str(pf[k].dtype)] for k in pf.files}
print(json.dumps(out['producer_keys'], indent=0))
# find square block matrices
blocks_f = [k for k in pf.files if pf[k].ndim == 2 and pf[k].shape[0] == pf[k].shape[1] and pf[k].shape[0] > 10]
res_prod = {}
lam_prod = {}
tot = 0.0
for k in sorted(blocks_f):
    if k not in pc.files:
        continue
    d, lam = kl(pf[k].astype(float), pc[k].astype(float))
    res_prod[k] = {'n': pf[k].shape[0], 'D': d}
    lam_prod[k] = np.sort(lam)
    tot += d
out['producer_blockKL'] = res_prod
out['producer_D_allindices'] = tot
# label arrays (heuristic): 1D int arrays
labels = {k: pf[k] for k in pf.files if pf[k].ndim == 1}
out['producer_1d_arrays'] = {k: [int(v.shape[0]), str(v.dtype), (v[:6].tolist() if v.dtype.kind in 'iuf' else None)] for k, v in labels.items()}
# own
own = {}
for fn in own_files:
    if tag.split('_t')[0].replace('s', 's') in fn and ('_l40_' in fn):
        z = np.load(fn)
        idx = [json.loads(s) for s in z['idx']]
        tot_o = 0.0
        lam_o = []
        for c in range(8):
            d, lam = kl(z['fossil_%d' % c], z['cube_%d' % c])
            tot_o += d
            lam_o.append(np.sort(lam))
        # match block spectra by size
        sizes_o = sorted([len(l) for l in lam_o])
        sizes_p = sorted([len(v) for v in lam_prod.values()])
        mx = None
        if sizes_o == sizes_p:
            used = set()
            diffs = []
            for lo in lam_o:
                best = None
                for kp, lp in lam_prod.items():
                    if kp in used or len(lp) != len(lo):
                        continue
                    e = float(np.max(np.abs(np.log(lo) - np.log(lp))))
                    if best is None or e < best[0]:
                        best = (e, kp)
                used.add(best[1])
                diffs.append(best[0])
            mx = max(diffs)
        own[os.path.basename(fn)] = {'D_allindices': tot_o, 'rel_diff_vs_producer': tot_o / tot - 1,
                                     'block_sizes_match': sizes_o == sizes_p, 'max_abs_dlog_geneig': mx}
out['own'] = own
with open(os.path.join(RES, 'C145_cache_comparison.json'), 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'producer_keys'}, indent=1, default=float))
