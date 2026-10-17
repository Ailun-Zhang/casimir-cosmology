# kl_own.py -- independent torus CMB covariance + Gaussian KL (T19, unverified-check, 2026-10-14)
# Model definition: v11 App. "Shape information in the microwave background" + D117 sec.1-2 + snapshot
# SYMMETRY_AND_CONVENTIONS.md (description only).  No producer code read.  Own CAMB 2.0.5 transfer functions.
# usage: python kl_own.py transfer | build <s> <lmax> <xmax> <taper> <tapform> <chi_mode> [A theta_deg]
import os, sys, json, math, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pylibs'))
import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline

RES = os.path.join(HERE, '..', 'results')
TRF = os.path.join(RES, 'kl_camb_transfer_own.npz')
CHI_PRODUCER = 13871.186029881328     # input convention read from C59_transfer.json metadata
SIGMA_E = 0.1041


def transfer():
    import camb
    t0 = time.time()
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.4, ombh2=0.02237, omch2=0.12, mnu=0.06, omk=0.0, tau=0.0544)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars.set_for_lmax(700, lens_potential_accuracy=0)
    pars.set_accuracy(AccuracyBoost=3, lSampleBoost=50, lAccuracyBoost=3)
    pars.Accuracy.IntkAccuracyBoost = 3
    pars.WantTensors = False
    pars.DoLensing = False
    data = camb.get_transfer_functions(pars)
    tr = data.get_cmb_transfer_data('scalar')
    L = np.array(tr.L)
    q = np.array(tr.q)
    Dl = np.array(tr.delta_p_l_k)
    der = data.get_derived_params()
    zstar = der['zstar']
    chi = float(data.tau0 - data.conformal_time(zstar))
    data.power_spectra_from_transfer()
    cl = data.get_unlensed_scalar_cls(raw_cl=True, CMB_unit=None)
    np.savez(TRF, L=L, q=q, D=Dl, cl=cl, chi=chi, zstar=zstar, tau0=data.tau0)
    info = {'camb_version': camb.__version__, 'nL': len(L), 'Lmax': int(L.max()), 'nq': len(q), 'qmin': q.min(),
            'qmax': q.max(), 'xmax_grid': q.max() * chi, 'chi_star': chi, 'zstar': zstar,
            'chi_rel_diff_vs_producer': chi / CHI_PRODUCER - 1, 'cpu_s': time.time() - t0, 'shape_D': list(Dl.shape)}
    print(json.dumps(info, indent=1, default=float))
    return info


class Transfer:
    def __init__(self, lmax, chi_mode='own'):
        z = np.load(TRF)
        L, q, Dl, cl = z['L'], z['q'], z['D'], z['cl']
        self.chi = float(z['chi']) if chi_mode == 'own' else CHI_PRODUCER
        chi_tr = float(z['chi'])
        self.lmax = lmax
        self.lnq = np.log(q)
        self.q = q
        self.As, self.ns, self.k0 = 2.1e-9, 0.9649, 0.05
        Pq = self.As * (q / self.k0) ** (self.ns - 1)
        self.Pq = Pq
        self.sp = {}
        self.tab = {}
        self.norm = {}
        chk = {}
        for X, src in (('T', 0), ('E', 1)):
            for l in range(2, lmax + 1):
                i = int(np.where(L == l)[0][0])
                d = Dl[src, i, :].copy()
                nz = np.nonzero(d)[0]
                j0 = nz[0]
                if j0 > 0:   # low-k continuation with spherical Bessel shape, matched at first nonzero q
                    jb = spherical_jn(l, q[:j0 + 1] * chi_tr)
                    d[:j0] = d[j0] * jb[:j0] / jb[j0]
                self.tab[(X, l)] = d
                cint = 4 * np.pi * np.trapezoid(Pq * d * d, self.lnq)
                ccamb = cl[l, 0] if X == 'T' else cl[l, 1]
                self.norm[(X, l)] = math.sqrt(ccamb / cint)
                chk[(X, l)] = (j0, cint, ccamb)
        for l in range(2, lmax + 1):
            for X in 'TE':
                self.tab[(X, l)] = self.tab[(X, l)] * self.norm[(X, l)]
                self.sp[(X, l)] = CubicSpline(q, self.tab[(X, l)])
            cte = 4 * np.pi * np.trapezoid(Pq * self.tab[('T', l)] * self.tab[('E', l)], self.lnq)
            chk[('TE', l)] = (0, cte, cl[l, 3])
        self.chk = chk

    def P(self, k):
        return 2 * np.pi ** 2 * self.As * (k / self.k0) ** (self.ns - 1) / k ** 3

    def cont(self, W):
        """continuum complement 4 pi int dlnk calP Delta^X Delta^Y (1-W(k chi))."""
        w1 = 1.0 - W(self.q * self.chi)
        out = {}
        for l in range(2, self.lmax + 1):
            tT, tE = self.tab[('T', l)], self.tab[('E', l)]
            f = 4 * np.pi * self.Pq * w1
            out[l] = (np.trapezoid(f * tT * tT, self.lnq), np.trapezoid(f * tT * tE, self.lnq),
                      np.trapezoid(f * tE * tE, self.lnq))
        return out


def window(xmax, taper, form):
    x1 = (1.0 - taper) * xmax

    def W(x):
        x = np.asarray(x, dtype=float)
        t = np.clip((x - x1) / (xmax - x1), 0.0, 1.0)
        if form == 'cos2':
            w = np.cos(0.5 * np.pi * t) ** 2
        else:   # smoothstep
            w = 1 - 3 * t * t + 2 * t ** 3
        return np.where(x >= xmax, 0.0, w)
    return W


def real_harmonics(ct, phi, lmax):
    """orthonormal real harmonics, no Condon-Shortley; returns dict (l,m)->array."""
    st = np.sqrt(np.maximum(0.0, 1 - ct * ct))
    out = {}
    pmm = np.full_like(ct, 1.0 / math.sqrt(4 * math.pi))
    Pbar = {}
    for m in range(0, lmax + 1):
        if m > 0:
            pmm = pmm * st * math.sqrt((2 * m + 1) / (2.0 * m))
        Pbar[(m, m)] = pmm
        if m + 1 <= lmax:
            Pbar[(m + 1, m)] = math.sqrt(2 * m + 3) * ct * pmm
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4.0 * l * l - 1) / (l * l - m * m))
            b = math.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1) ** 2 - 1))
            Pbar[(l, m)] = a * (ct * Pbar[(l - 1, m)] - b * Pbar[(l - 2, m)])
    for m in range(0, lmax + 1):
        if m == 0:
            for l in range(max(2, m), lmax + 1):
                out[(l, 0)] = Pbar[(l, 0)]
        else:
            c, s = np.cos(m * phi), np.sin(m * phi)
            for l in range(max(2, m), lmax + 1):
                out[(l, m)] = math.sqrt(2) * Pbar[(l, m)] * c
                out[(l, -m)] = math.sqrt(2) * Pbar[(l, m)] * s
    return out


def character(l, m):
    if m >= 0:
        b = (m % 2, 0, (l + m) % 2)
    else:
        a = -m
        b = ((a + 1) % 2, 1, (l + a) % 2)
    return 4 * b[0] + 2 * b[1] + b[2]


def build(s, lmax, xmax, taper, form, chi_mode, A=math.sqrt(2), theta=0.0, cube=False, tr=None):
    tr = tr or Transfer(lmax, chi_mode)
    chi = tr.chi
    L0 = s * 2 * chi
    bvec = np.zeros(3) if cube else A * SIGMA_E * math.sqrt(2 / 3) * np.cos(
        np.radians(theta) - np.radians([30.0, 150.0, 270.0]))
    Mdiag = np.exp(-bvec)
    W = window(xmax, taper, form)
    # index sets
    idx = [[] for _ in range(8)]
    for X in 'TE':
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                idx[character(l, m)].append((X, l, m))
    C = [np.zeros((len(ix), len(ix))) for ix in idx]
    # octant enumeration
    kfac = 2 * math.pi / L0
    nmax = [int(math.floor(xmax / (chi * kfac * Mdiag[i]))) + 1 for i in range(3)]
    n0 = np.arange(0, nmax[0] + 1)
    n1 = np.arange(0, nmax[1] + 1)
    n2 = np.arange(0, nmax[2] + 1)
    G = np.stack(np.meshgrid(n0, n1, n2, indexing='ij'), -1).reshape(-1, 3).astype(float)
    kv = kfac * G * Mdiag
    kk = np.sqrt((kv * kv).sum(1))
    sel = (kk > 0) & (kk * chi < xmax)
    G, kv, kk = G[sel], kv[sel], kk[sel]
    mult = 2.0 ** (G > 0).sum(1)
    V = L0 ** 3
    wt = (4 * math.pi) ** 2 / V * mult * tr.P(kk) * W(kk * chi)
    ct = kv[:, 2] / kk
    phi = np.arctan2(kv[:, 1], kv[:, 0])
    nmodes_full = float(mult.sum())
    CH = 6000
    for c0 in range(0, len(kk), CH):
        sl = slice(c0, c0 + CH)
        Y = real_harmonics(ct[sl], phi[sl], lmax)
        Dx = {(X, l): tr.sp[(X, l)](kk[sl]) for X in 'TE' for l in range(2, lmax + 1)}
        w = wt[sl]
        for c in range(8):
            S = np.empty((len(idx[c]), len(w)))
            for r, (X, l, m) in enumerate(idx[c]):
                sg = -1.0 if (l // 2) % 2 else 1.0
                S[r] = sg * Dx[(X, l)] * Y[(l, m)]
            C[c] += (S * w) @ S.T
    cont = tr.cont(W)
    for c in range(8):
        pos = {key: r for r, key in enumerate(idx[c])}
        for (X, l, m), r in pos.items():
            tt, te, ee = cont[l]
            if X == 'T':
                C[c][r, r] += tt
                C[c][r, pos[('E', l, m)]] += te
            else:
                C[c][r, r] += ee
                C[c][r, pos[('T', l, m)]] += te
    return {'C': C, 'idx': idx, 'nmodes_full': nmodes_full, 'n_octant': len(kk), 'L0': L0, 'chi': chi, 'bvec': bvec.tolist()}


def kl_blocks(P, Q, idx, lsel, Xsel):
    tot = 0.0
    lam_all = []
    for c in range(8):
        ii = [r for r, (X, l, m) in enumerate(idx[c]) if l <= lsel and X in Xsel]
        if not ii:
            continue
        Pc = P[c][np.ix_(ii, ii)]
        Qc = Q[c][np.ix_(ii, ii)]
        Lq = np.linalg.cholesky(Qc)
        Xm = np.linalg.solve(Lq, np.linalg.solve(Lq, Pc).T).T
        lam = np.linalg.eigvalsh(0.5 * (Xm + Xm.T))
        lam_all.append(lam)
        tot += 0.5 * np.sum(lam - 1 - np.log(lam))
    return tot, np.concatenate(lam_all)


def main_build(argv):
    s, lmax, xmax, taper, form, chi_mode = float(argv[0]), int(argv[1]), float(argv[2]), float(argv[3]), argv[4], argv[5]
    A = float(argv[6]) if len(argv) > 6 else math.sqrt(2)
    th = float(argv[7]) if len(argv) > 7 else 0.0
    t0 = time.time()
    tr = Transfer(lmax, chi_mode)
    fo = build(s, lmax, xmax, taper, form, chi_mode, A, th, False, tr)
    cu = build(s, lmax, xmax, taper, form, chi_mode, A, th, True, tr)
    out = {'s': s, 'lmax': lmax, 'xmax': xmax, 'taper': taper, 'form': form, 'chi_mode': chi_mode, 'chi': fo['chi'],
           'A': A, 'theta': th, 'L0': fo['L0'], 'bvec': fo['bvec'], 'n_octant': fo['n_octant'],
           'nmodes_full': fo['nmodes_full'], 'cpu_s': None}
    for lsel in sorted({30, 40, lmax}):
        if lsel > lmax:
            continue
        for Xs, lab in (('TE', 'TE'), ('T', 'T'), ('E', 'E')):
            d, lam = kl_blocks(fo['C'], cu['C'], fo['idx'], lsel, Xs)
            out['D_%s%d' % (lab, lsel)] = d
            out['lam_minmax_%s%d' % (lab, lsel)] = [float(lam.min()), float(lam.max())]
    # transfer self-checks
    rel = [abs(v[1] / v[2] - 1) for k, v in tr.chk.items() if k[0] == 'TE' and v[2] != 0]
    out['TE_cl_reldiff_max'] = max(rel)
    out['lowk_cont_first_nonzero_index'] = {('%s%d' % k): int(v[0]) for k, v in tr.chk.items() if k[0] != 'TE' and v[0] > 0}
    out['cpu_s'] = time.time() - t0
    tag = 's%.5f_l%d_x%g_t%g_%s_%s_A%.4f_th%g' % (s, lmax, xmax, taper, form, chi_mode, A, th)
    np.savez(os.path.join(RES, 'kl_cov_own_' + tag + '.npz'),
             **{'fossil_%d' % c: fo['C'][c] for c in range(8)}, **{'cube_%d' % c: cu['C'][c] for c in range(8)},
             idx=np.array([json.dumps(ix) for ix in fo['idx']]))
    print(json.dumps(out, indent=1, default=float))
    path = os.path.join(RES, 'C145_KL_own.json')
    allr = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            allr = json.load(f)
    allr[tag] = out
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(allr, f, indent=1, default=float)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1] == 'transfer':
        transfer()
    elif sys.argv[1] == 'build':
        main_build(sys.argv[2:])
