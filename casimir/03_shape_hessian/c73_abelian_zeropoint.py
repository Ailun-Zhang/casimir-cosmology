# PROVENANCE: 04_calculations/code/20261011_claude_followup/C73/c73_abelian_zeropoint.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Abelian holonomy pinned by charged-fermion loops: Delta_sh E_zp / E_zp at the cube vs the first-matching eigenvalue -4/3 (D55 Lemma B). Paper: App. app:shapelaplacian. Pool size = env DNEC_WORKERS.

C73: does the first-matching eigenvalue -4/3 apply to an ABELIAN holonomy pinned by charged-fermion
loops?  For a U(1) holonomy c with kinetic (L/2g^2) c^T G^-1 c' and fermion-induced curvature
K_ij(G) = d^2/dc_i dc_j Z_G(2; AAA + c/2pi)|_{c=0} (times a positive constant), the leading zero-mode
energy is the harmonic zero-point energy  E_zp = (g/2L) tr sqrt(G^{1/2} K(G) G^{1/2})  (not a
first-order insertion).  We compute Delta_sh E_zp / E_zp at the cube by nested finite differences
(c-Hessian by Richardson FD, then geodesic second differences), and compare with
  -4/3 (first-matching lemma)  and  -2/3 - sum_A tr(W1_A^2)/(12 w^2)  (derived in D55; W = wI at cube).
Output: results/.../C73/C73_abelian_zeropoint.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, itertools, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
from c73_common import ewald_Z, GeodesicMP, symfun_mp, RES, richardson

DPS = 40
HC = [mp.mpf("1e-7"), mp.mpf("5e-8")]


def worker(t):
    mp.mp.dps = DPS
    G = mp.matrix([[mp.mpf(x) for x in row] for row in t["G"]])
    return t["key"], mp.nstr(ewald_Z(2, G, [mp.mpf(x) for x in t["a"]]), DPS - 2)


def offsets():
    offs = [(0, 0, 0)]
    for i in range(3):
        for sg in (1, -1):
            v = [0, 0, 0]; v[i] = sg; offs.append(tuple(v))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for si, sj in itertools.product((1, -1), repeat=2):
            v = [0, 0, 0]; v[i] = si; v[j] = sj; offs.append(tuple(v))
    return offs


def main():
    t0 = time.time()
    mp.mp.dps = 60
    geo = GeodesicMP(mp.eye(3))
    eps_list = [mp.mpf("1e-4"), mp.mpf("5e-5")]
    pts = {"c": mp.eye(3)}
    for ie, e in enumerate(eps_list):
        for A in range(5):
            for sg in (1, -1):
                pts[(ie, A, sg)] = geo.point(A, sg * e)
    tasks = []
    for pk, G in pts.items():
        Gs = [[mp.nstr(G[i, j], 55) for j in range(3)] for i in range(3)]
        for ih, h in enumerate(HC):
            for off in offsets():
                a = [mp.nstr(mp.mpf(1) / 2 + off[i] * h / (2 * mp.pi), 55) for i in range(3)]
                tasks.append(dict(G=Gs, a=a, key=(str(pk), ih, off)))
    with Pool(int(os.environ.get("DNEC_WORKERS", "14"))) as pool:  # package: --workers
        R = dict(pool.map(worker, tasks, chunksize=4))
    mp.mp.dps = DPS

    def Kmat(pk):
        Ks = []
        for ih, h in enumerate(HC):
            z = lambda off: mp.mpf(R[(str(pk), ih, off)])
            K = mp.zeros(3, 3)
            for i in range(3):
                e = [0, 0, 0]; e[i] = 1
                m = [0, 0, 0]; m[i] = -1
                K[i, i] = (z(tuple(e)) - 2 * z((0, 0, 0)) + z(tuple(m))) / h ** 2
            for i, j in ((0, 1), (0, 2), (1, 2)):
                def zz(si, sj):
                    v = [0, 0, 0]; v[i] = si; v[j] = sj
                    return z(tuple(v))
                K[i, j] = K[j, i] = (zz(1, 1) - zz(1, -1) - zz(-1, 1) + zz(-1, -1)) / (4 * h ** 2)
            Ks.append(K)
        return (4 * Ks[1] - Ks[0]) / 3   # Richardson in h^2

    def W_of(pk):
        G = pts[pk]
        Sg = symfun_mp(G, mp.sqrt)
        W = Sg * Kmat(pk) * Sg
        return (W + W.T) / 2

    def ezp(W):
        E, Q = mp.eigsy(W)
        return mp.fsum(mp.sqrt(x) for x in E)

    W0 = W_of("c")
    f0 = ezp(W0)
    w = W0[0, 0]
    D = []
    for ie, e in enumerate(eps_list):
        D.append(mp.fsum((ezp(W_of((ie, A, 1))) - 2 * f0 + ezp(W_of((ie, A, -1)))) / e ** 2 for A in range(5)))
    lap = richardson(D)[-1][0]
    # trace of W: should have eigenvalue -4/3 (it is proportional to -Z_G(1;AAA))
    trW = lambda pk: sum(W_of(pk)[i, i] for i in range(3))
    Dt = []
    for ie, e in enumerate(eps_list):
        Dt.append(mp.fsum((trW((ie, A, 1)) - 2 * trW("c") + trW((ie, A, -1))) / e ** 2 for A in range(5)))
    lap_tr = richardson(Dt)[-1][0]
    # first-order W1_A by central differences (Richardson) and the predicted ratio
    sumtr = mp.mpf(0)
    W1n = []
    for A in range(5):
        d1 = [(W_of((ie, A, 1)) - W_of((ie, A, -1))) / (2 * e) for ie, e in enumerate(eps_list)]
        W1 = (4 * d1[1] - d1[0]) / 3
        W1n.append(mp.nstr(mp.sqrt(sum(W1[i, j] ** 2 for i in range(3) for j in range(3))), 12))
        sumtr += sum(W1[i, j] * W1[j, i] for i in range(3) for j in range(3))
    pred = -mp.mpf(2) / 3 - sumtr / (12 * w ** 2)
    out = {"description": __doc__,
           "W(cube) diagonal w": mp.nstr(w, 20), "offdiag max |W_ij|": mp.nstr(max(abs(W0[0, 1]), abs(W0[0, 2]), abs(W0[1, 2])), 3),
           "-Z_I(1;AAA)/3 (expected w)": "0.8397853840298151",
           "Delta trW / trW (expected -4/3)": mp.nstr(lap_tr / (3 * w), 15),
           "Delta E_zp / E_zp (numeric)": mp.nstr(lap / f0, 15),
           "predicted -2/3 - sum tr(W1^2)/(12 w^2)": mp.nstr(pred, 15),
           "|W1_A| Frobenius": W1n, "wall_s": time.time() - t0}
    (RES / "C73_abelian_zeropoint.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
