# PROVENANCE: 04_calculations/code/20261013_argon_codex_intake/cmb_ensemble/ens_core.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] C145/D117 core: E-plane fossils beta = A sigma_E (cos th X_d + sin th X_d2), octant-block covariance builds (l <= 80, x_max = 240), fast KL from cached packed inverses, noise/beam variants. The l <= 80 transfer table of the frozen Codex snapshot is shipped as data/C59_transfer.npz (run.py --full sets env DNEC_CMB_TRANSFER to it; reduced runs use inputs/C59_transfer_l12.npz); the snapshot P5_CMB cache (3.3 GB, only for the S1 regression) is not shipped: files placed in <work>/ai/snapshot are used read-only. Paper: PRD Sec. VIII B and App. F.

cmb_ensemble core library (C145 / D117) -- WORKING COPY, not frozen evidence.

Provenance (frozen snapshot, read-only; nothing there is executed or written):
  SNAPSHOT_INPUT/P5_CMB/C59_torus.py      sha256 44E17F0CF4344D78045DAC511C111A0D355B4AA6664DB3BBA631BC6188AC7437
      -> window, Transfer, real_ylm, add_diag_tl, half_lattice, TorusCov, kl_gauss, GENERATORS, kmatrix
  SNAPSHOT_INPUT/P5_CMB/high_cov.py       sha256 858F996C3C5CF02781EB5D836208DAAF29AB6EDCAB40A19F707F1218F5C95669
      -> chars, layouts, octant_modes, octant build loop (build_cov), kl (kl_eig)
  SNAPSHOT_INPUT/P5_CMB/compare_volume.py sha256 E27805AE36AB399C037B925DFA0EE396D4EFEFFB0B691E99AB9945A84A80EE23
      -> reference-cube comparison, generalised to a dense pool + Brent profile (run_ens.py)
  SNAPSHOT_INPUT/P5_CMB/window_delta.py   sha256 2C2674F98D107489ED1C9C99CEA67094A8D61A4E802D8471C3B058ABC0A083B5
      -> exact taper update (taper_update)
  input data SNAPSHOT_INPUT/P5_CMB/C59_transfer.npz sha256 8D0EA21FA49CA6DF088780FB5C2A95C009B7A4AE17E928BCE91B246CC814DF8F
Changes: explicit SNAPSHOT_INPUT (read) / NEW_RUN_OUTPUT (write, guarded); BLAS/OMP/MKL = 1 thread per
process; fossil generalised from beta = sqrt2 sigma_E X_d to any E-plane shape
beta = A sigma_E (cos th X_d + sin th X_d2) (diagonal M = exp(-beta), so the 8 reflection-character
blocks stay exact); fast trace/log-det KL from cached packed inverses (validated against kl_eig);
Cholesky KL with optional isotropic white noise + Gaussian beam (diagonal in l,m: block preserving).
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import json, time, itertools, math, gc, traceback
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigvalsh, cho_factor, cho_solve, expm, svdvals
from scipy.linalg.blas import dsyrk
from scipy.linalg.lapack import dpotrf, dpotri
from scipy.integrate import simpson

import sys  # package: work-tree path shim of the cmb chain (see cmb/_cmb_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cmb_paths as WP  # noqa: E402
ROOT = None  # package: the research tree is not shipped
SNAPSHOT_INPUT = WP.wdir('ai/snapshot')  # package: frozen Codex snapshot (not shipped); read-only if present
SNAP_P5 = SNAPSHOT_INPUT / 'P5_CMB'
SNAP_CACHE = SNAP_P5 / 'cache'
TRANSFER = Path(os.environ.get('DNEC_CMB_TRANSFER') or (SNAP_P5 / 'C59_transfer.npz'))  # package: env override
_DEFAULT_OUT = WP.wdir('ai/cmb_ensemble/run01')
NEW_RUN_OUTPUT = Path(os.environ.get('CMB_ENS_OUT', str(_DEFAULT_OUT)))
CACHE = NEW_RUN_OUTPUT / 'cache'
SIGMA_E = 0.1041
CIRCLE = 0.985
THRESH = 4.5
T_CMB_UK = 2.7255e6
ARCMIN = np.pi / 10800.0
S2, S6 = np.sqrt(2.0), np.sqrt(6.0)


# ----------------------------------------------------------------------------- io helpers
def out_path(p):
    p = Path(p).resolve()
    base = NEW_RUN_OUTPUT.resolve()
    snap = SNAPSHOT_INPUT.resolve()
    if p == snap or snap in p.parents:
        raise RuntimeError('refusing to write into the frozen snapshot: %s' % p)
    if p != base and base not in p.parents:
        raise RuntimeError('refusing to write outside NEW_RUN_OUTPUT: %s' % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def save_json(path, obj):
    p = out_path(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=_jd) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding='utf-8'))


# ----------------------------------------------------------------------------- C59_torus copies
def _e(i, j):
    m = np.zeros((3, 3)); m[i, j] = 1.0
    return m


GENERATORS = {
    "Xd": np.diag([1.0, -1.0, 0.0]) / S2,
    "Xd2": np.diag([1.0, 1.0, -2.0]) / S6,
    "X12": (_e(0, 1) + _e(1, 0)) / S2,
    "X13": (_e(0, 2) + _e(2, 0)) / S2,
    "X23": (_e(1, 2) + _e(2, 1)) / S2,
}
SHAPE = ["Xd", "Xd2", "X12", "X13", "X23"]


def kmatrix(beta5):
    X = sum(float(b) * GENERATORS[g] for b, g in zip(beta5, SHAPE))
    return expm(-X)


def window(x, x1, x2):
    """C-infinity taper: 1 for x<=x1, 0 for x>=x2."""
    x = np.asarray(x, float)
    t = np.clip((x - x1) / (x2 - x1), 0.0, 1.0)
    with np.errstate(divide="ignore", over="ignore"):
        a = np.where(t > 0, np.exp(-1.0 / np.where(t > 0, t, 1.0)), 0.0)
        b = np.where(t < 1, np.exp(-1.0 / np.where(t < 1, 1.0 - t, 1.0)), 0.0)
    return b / (a + b)


class Transfer:
    """Delta^T_l, Delta^E_l (l=2..lmax) on the uniform x = k chi* grid (verbatim logic)."""

    def __init__(self, lmax=30, path=TRANSFER):
        a = np.load(path)
        self.path = str(path)
        self.lmax = lmax
        nl = lmax - 1
        self.nl = nl
        self.chi = float(a["chi"]); self.step = float(a["xstep"]); self.xend = float(a["xend"])
        self.x = a["x"]
        self.DT = a["DT"][:nl]; self.DE = a["DE"][:nl]
        self.tab = np.ascontiguousarray(np.concatenate([self.DT, self.DE], axis=0).T)
        self.tail400 = {k: a["tail_" + k][:nl].copy() for k in ("TT", "EE", "TE")}
        self.cl = {k: a["cl_" + k][:nl].copy() for k in ("TT", "EE", "TE")}
        self.camb = {k: a["camb_" + k][:nl].copy() for k in ("TT", "EE", "TE")}
        self.As, self.ns, self.pivot = float(a["As"]), float(a["ns"]), float(a["pivot"])

    def calP(self, k):
        return self.As * (np.asarray(k) / self.pivot) ** (self.ns - 1.0)

    def eval(self, x):
        u = np.asarray(x) / self.step
        i = u.astype(np.int64)
        t = u - i
        if i.min() < 1 or i.max() > len(self.x) - 3:
            raise ValueError("x outside the tabulated transfer grid")
        w0 = -t * (t - 1) * (t - 2) / 6.0
        w1 = (t + 1) * (t - 1) * (t - 2) / 2.0
        w2 = -(t + 1) * t * (t - 2) / 2.0
        w3 = (t + 1) * t * (t - 1) / 6.0
        tab = self.tab
        return (tab[i - 1] * w0[:, None] + tab[i] * w1[:, None]
                + tab[i + 1] * w2[:, None] + tab[i + 2] * w3[:, None])

    def tail(self, x1, x2):
        if x2 > self.xend + 1e-9:
            raise ValueError("taper end beyond tabulated grid")
        j0 = max(int(np.floor(x1 / self.step)) - 1, 1)
        xs = self.x[j0:]
        om = 1.0 - window(xs, x1, x2)
        wt = 4 * np.pi * self.calP(xs / self.chi) * om
        lnx = np.log(xs)
        T, E = self.DT[:, j0:], self.DE[:, j0:]
        out = {"TT": simpson(wt * T * T, x=lnx, axis=1), "EE": simpson(wt * E * E, x=lnx, axis=1),
               "TE": simpson(wt * T * E, x=lnx, axis=1)}
        for k in out:
            out[k] = out[k] + self.tail400[k]
        return out


_TR = {}


def get_tr(lmax):
    if lmax not in _TR:
        _TR[lmax] = Transfer(lmax, path=TRANSFER)
    return _TR[lmax]


def real_ylm(u, lmax):
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    n = len(z)
    out = np.empty((n, (lmax + 1) ** 2))
    cm = np.ones(n); sm = np.zeros(n)
    cmm = 1.0 / np.sqrt(4 * np.pi)

    def put(l, m, q):
        c = l * l + l
        if m == 0:
            out[:, c] = q
        else:
            out[:, c + m] = S2 * q * cm
            out[:, c - m] = S2 * q * sm

    for m in range(lmax + 1):
        if m > 0:
            cm, sm = cm * x - sm * y, cm * y + sm * x
            cmm *= np.sqrt((2 * m + 1) / (2.0 * m))
        qa = np.full(n, cmm)
        put(m, m, qa)
        if m < lmax:
            qb = z * (np.sqrt(2 * m + 3) * cmm)
            put(m + 1, m, qb)
            for l in range(m + 2, lmax + 1):
                a = np.sqrt((4.0 * l * l - 1) / (l * l - m * m))
                b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1) ** 2 - 1))
                qc = a * (z * qb - b * qa)
                qa, qb = qb, qc
                put(l, m, qb)
    return out


def half_lattice(R):
    r2 = R * R
    rows = []
    for n1 in range(0, int(np.floor(R)) + 1):
        rem = r2 - n1 * n1
        m2 = int(np.floor(np.sqrt(rem)))
        n2 = np.arange(-m2, m2 + 1)
        m3 = np.floor(np.sqrt(np.maximum(rem - n2 * n2, 0.0))).astype(int)
        cnt = 2 * m3 + 1
        n2r = np.repeat(n2, cnt)
        off = np.repeat(np.cumsum(cnt) - cnt, cnt)
        n3r = np.arange(cnt.sum()) - off - np.repeat(m3, cnt)
        blk = np.column_stack([np.full(len(n2r), n1), n2r, n3r]).astype(np.int32)
        if n1 == 0:
            blk = blk[(blk[:, 1] > 0) | ((blk[:, 1] == 0) & (blk[:, 2] > 0))]
        rows.append(blk)
    return np.concatenate(rows)


def add_diag_tl(c, lab, spec):
    d = np.where(lab[:, 0] == 0, spec["TT"][lab[:, 1] - 2], spec["EE"][lab[:, 1] - 2])
    c[np.diag_indices(len(lab))] += d
    iT = np.flatnonzero(lab[:, 0] == 0)
    if len(iT) and len(iT) * 2 == len(lab):
        iE = iT + len(iT)
        te = spec["TE"][lab[iT, 1] - 2]
        c[iT, iE] += te
        c[iE, iT] += te


class TorusCov:
    """Parity-blocked (full, no reflection compression) T+E covariance builder, verbatim C59 logic."""

    def __init__(self, tr, lmax=30):
        self.tr, self.lmax = tr, lmax
        nl = tr.nl
        self.blocks = []
        for p in (0, 1):
            lab = np.array([(X, l, m) for X in (0, 1) for l in range(2, lmax + 1) if l % 2 == p
                            for m in range(-l, l + 1)], int)
            ycol = lab[:, 1] ** 2 + lab[:, 1] + lab[:, 2]
            ls = np.array([l for l in range(2, lmax + 1) if l % 2 == p])
            self.blocks.append(dict(labels=lab, ycol=ycol, half=len(lab) // 2,
                                    ycolT=ycol[: len(lab) // 2], fT=ls - 2, fE=ls - 2 + nl,
                                    counts=2 * ls + 1))
        ph = (-1.0) ** (np.arange(2, tr.lmax + 1) // 2)
        self.phase2 = np.concatenate([ph, ph])

    def _amatrix(self, Y, F, blk):
        h = blk["half"]
        Yp = Y[:, blk["ycolT"]]
        A = np.empty((Y.shape[0], 2 * h))
        np.multiply(Yp, np.repeat(F[:, blk["fT"]], blk["counts"], axis=1), out=A[:, :h])
        np.multiply(Yp, np.repeat(F[:, blk["fE"]], blk["counts"], axis=1), out=A[:, h:])
        return A

    def mode_list(self, L0, xmax, mats):
        smin = min(float(svdvals(M).min()) for M in mats)
        R = xmax / smin * L0 / (2 * np.pi * self.tr.chi) * (1 + 1e-9)
        return half_lattice(R), R

    def compute(self, L0, M, xmax, nlist, taper=0.25, batch=4096, chunk=400000):
        tr = self.tr; chi = tr.chi
        V = L0 ** 3
        pref = 2.0 * (4 * np.pi) ** 2 / V
        x1 = (1 - taper) * xmax
        C = [np.zeros((len(b["labels"]),) * 2, order="F") for b in self.blocks]
        kfac = 2 * np.pi / L0
        Mt = np.asarray(M, float).T
        nused = 0
        for s in range(0, len(nlist), chunk):
            nf = nlist[s:s + chunk].astype(float)
            kv = kfac * (nf @ Mt)
            k = np.sqrt(np.einsum("ij,ij->i", kv, kv))
            xw = k * chi
            keep = xw < xmax
            kv, k, xw = kv[keep], k[keep], xw[keep]
            nused += len(k)
            for b0 in range(0, len(k), batch):
                kb = k[b0:b0 + batch]; vb = kv[b0:b0 + batch]
                xb = kb * chi
                w = pref * 2 * np.pi ** 2 * tr.calP(kb) / kb ** 3 * window(xw[b0:b0 + batch], x1, xmax)
                Y = real_ylm(vb / kb[:, None], self.lmax)
                F = tr.eval(xb) * (np.sqrt(w)[:, None] * self.phase2[None, :])
                for p, blk in enumerate(self.blocks):
                    A = self._amatrix(Y, F, blk)
                    C[p] = dsyrk(1.0, A.T, beta=1.0, c=C[p], trans=0, lower=0, overwrite_c=1)
        tail = tr.tail(x1, xmax)
        out = []
        for p, blk in enumerate(self.blocks):
            c = np.triu(C[p]); c = c + np.triu(c, 1).T
            add_diag_tl(c, blk["labels"], tail)
            out.append(c)
        return out, dict(modes_half_used=int(nused), modes_full_used=int(2 * nused), x_taper_start=x1)

    def select(self, p, lmax_sel, fields):
        lab = self.blocks[p]["labels"]
        fl = {"T": [0], "E": [1], "TE": [0, 1]}[fields]
        return np.flatnonzero((lab[:, 1] <= lmax_sel) & np.isin(lab[:, 0], fl))


def kl_gauss(Cp, Cq):
    L = cholesky(Cq, lower=True, check_finite=False)
    m = solve_triangular(L, solve_triangular(L, Cp, lower=True, check_finite=False).T,
                         lower=True, check_finite=False)
    lam = eigvalsh(0.5 * (m + m.T), check_finite=False)
    if lam.min() <= 0:
        raise ValueError("non positive definite")
    return float(0.5 * np.sum(lam - 1.0 - np.log(lam))), lam


# ----------------------------------------------------------------------------- high_cov copies
def chars(l, m):
    a = abs(m)
    return ((a + (m < 0)) % 2, int(m < 0), (l + a) % 2)


def layouts(lmax):
    all_lab = np.array([(x, l, m) for x in (0, 1) for l in range(2, lmax + 1) for m in range(-l, l + 1)], int)
    blocks = []
    for c in itertools.product((0, 1), repeat=3):
        lab = all_lab[np.array([chars(l, m) == c for x, l, m in all_lab])]
        n = len(lab) // 2
        blocks.append(dict(char=c, labels=lab, ycol=lab[:n, 1] ** 2 + lab[:n, 1] + lab[:n, 2],
                           ls=lab[:n, 1], n=n))
    return blocks


_LAY = {}


def layouts_c(lmax):
    if lmax not in _LAY:
        _LAY[lmax] = layouts(lmax)
    return _LAY[lmax]


def octant_modes(s, diag, xmax):
    rad = xmax * s / np.pi
    nmax = int(np.floor(rad / diag[0]))
    rows = []
    for a in range(nmax + 1):
        rem = rad ** 2 - (a * diag[0]) ** 2
        b = np.arange(int(np.floor(np.sqrt(max(rem, 0)) / diag[1])) + 1)
        remz = np.maximum(rem - (b * diag[1]) ** 2, 0)
        top = np.floor(np.sqrt(remz) / diag[2]).astype(int)
        count = top + 1
        br = np.repeat(b, count)
        offsets = np.repeat(np.cumsum(count) - count, count)
        zr = np.arange(count.sum()) - offsets
        rows.append(np.column_stack((np.full(len(br), a), br, zr)).astype(np.int32))
    modes = np.concatenate(rows)
    modes = modes[np.any(modes != 0, axis=1)]
    return modes


def kl_eig(P, Q, lmax, lsel, fields='TE', check=True):
    """Verbatim high_cov.kl: generalized-eigenvalue KL D(N(0,P)||N(0,Q)) summed over 8 characters."""
    blks = layouts_c(lmax)
    vals = []; val2 = 0.; lo = np.inf; hi = 0.
    for p, q, b in zip(P, Q, blks):
        lab = b['labels']
        fl = (0, 1) if fields == 'TE' else ((0,) if fields == 'T' else (1,))
        ix = np.flatnonzero((lab[:, 1] <= lsel) & np.isin(lab[:, 0], fl))
        d = 1 / np.sqrt(np.diag(q)[ix])
        p = p[np.ix_(ix, ix)] * d[:, None] * d[None, :]
        q = q[np.ix_(ix, ix)] * d[:, None] * d[None, :]
        L = cholesky(q, lower=True, check_finite=False)
        W = solve_triangular(L, p, lower=True, check_finite=False)
        W = solve_triangular(L, W.T, lower=True, check_finite=False)
        eig = eigvalsh((W + W.T) * .5, check_finite=False)
        if eig.min() <= 0:
            raise ValueError('nonpositive generalized eigenvalue')
        vals.append(float(.5 * np.sum(eig - 1 - np.log(eig))))
        lo = min(lo, float(eig.min())); hi = max(hi, float(eig.max()))
        if check:
            cf = cho_factor(q, lower=True, check_finite=False)
            trace = np.trace(cho_solve(cf, p, check_finite=False))
            lp = cholesky(p, lower=True, check_finite=False)
            val2 += .5 * (trace - len(p) + 2 * np.log(np.diag(L)).sum() - 2 * np.log(np.diag(lp)).sum())
    return dict(D=sum(vals), D_by_character=vals, lam_min=lo, lam_max=hi,
                D_logdet=float(val2) if check else None)


# ----------------------------------------------------------------------------- shapes and names
XD_DIAG = np.array([1.0, -1.0, 0.0]) / S2
XD2_DIAG = np.array([1.0, 1.0, -2.0]) / S6


def beta_diag(A, theta, sigma=SIGMA_E):
    """beta = A sigma_E (cos th X_d + sin th X_d2) (diagonal); physical sides L0 exp(beta_ii)."""
    return A * sigma * (np.cos(theta) * XD_DIAG + np.sin(theta) * XD2_DIAG)


def short_factor(A, theta):
    return float(np.exp(np.min(beta_diag(A, theta))))


def s_low(A, theta):
    return CIRCLE / short_factor(A, theta)


def is_rep(A, theta):
    return abs(A - S2) < 1e-9 and abs(theta) < 1e-12


def cov_name(kind, s, lmax, xmax, taper, A=0.0, theta=0.0):
    if kind == 'cube':
        return 'cube_s%.5f_l%d_x%g_t%g' % (s, lmax, xmax, taper)
    return 'shape_A%.5f_th%+.5f_s%.5f_l%d_x%g_t%g' % (A, theta, s, lmax, xmax, taper)


def find_cov(kind, s, lmax, xmax, taper, A=0.0, theta=0.0):
    """Snapshot cache (read-only) first, then the new cache. Returns (path, exists)."""
    cands = []
    if kind == 'cube':
        cands.append(SNAP_CACHE / (cov_name('cube', s, lmax, xmax, taper) + '.npz'))
    elif is_rep(A, theta):
        cands.append(SNAP_CACHE / ('fossil_s%.5f_l%d_x%g_t%g.npz' % (s, lmax, xmax, taper)))
    own = CACHE / (cov_name(kind, s, lmax, xmax, taper, A, theta) + '.npz')
    cands.append(own)
    for c in cands:
        if c.exists():
            return c, True
    return own, False


def build_cov(kind, s, lmax, xmax, taper, A=0.0, theta=0.0, batch=2048, cube_scale=None):
    """high_cov.build generalised: M = exp(-beta) diagonal; cube if kind=='cube'."""
    tr = get_tr(lmax)
    if len(tr.DT) != lmax - 1:
        raise ValueError('transfer table does not cover requested lmax')
    bd = np.zeros(3) if kind == 'cube' else beta_diag(A, theta)
    diag = np.exp(-bd)
    L0 = s * 2 * tr.chi
    M = np.diag(diag)
    t0 = time.perf_counter()
    modes = octant_modes(s, diag, xmax)
    blks = layouts_c(lmax)
    cc = [np.zeros((len(b['labels']),) * 2, order='F') for b in blks]
    nused = 0; nfull = 0
    phase = (-1.) ** (np.arange(2, lmax + 1) // 2)
    pref = (4 * np.pi) ** 2 / L0 ** 3
    x1 = (1 - taper) * xmax
    for b0 in range(0, len(modes), batch):
        ns = modes[b0:b0 + batch]
        kv = 2 * np.pi / L0 * ns * diag
        k = np.sqrt(np.sum(kv * kv, axis=1))
        xx = k * tr.chi
        keep = xx < xmax
        ns, kv, k, xx = ns[keep], kv[keep], k[keep], xx[keep]
        if len(k) == 0:
            continue
        mult = 2. ** np.sum(ns != 0, axis=1)
        nused += len(k); nfull += int(np.sum(mult))
        weights = pref * mult * 2 * np.pi ** 2 * tr.calP(k) / k ** 3 * window(xx, x1, xmax)
        yy = real_ylm(kv / k[:, None], lmax)
        ff = tr.eval(xx)
        ff[:, :lmax - 1] *= np.sqrt(weights)[:, None] * phase
        ff[:, lmax - 1:] *= np.sqrt(weights)[:, None] * phase
        for ib, blk in enumerate(blks):
            n = blk['n']; ls = blk['ls'] - 2
            aa = np.empty((len(k), 2 * n))
            yp = yy[:, blk['ycol']]
            np.multiply(yp, ff[:, ls], out=aa[:, :n])
            np.multiply(yp, ff[:, ls + lmax - 1], out=aa[:, n:])
            cc[ib] = dsyrk(1., aa.T, beta=1., c=cc[ib], trans=0, lower=0, overwrite_c=1)
    tail = tr.tail(x1, xmax)
    for i, blk in enumerate(blks):
        cc[i] = np.triu(cc[i]) + np.triu(cc[i], 1).T
        add_diag_tl(cc[i], blk['labels'], tail)
    meta = dict(name=cov_name(kind, s, lmax, xmax, taper, A, theta), s=s, kind=kind, A=float(A),
                theta=float(theta), lmax=lmax, xmax=xmax, taper=taper, sigma_E=SIGMA_E,
                beta_diag=bd.tolist(), M=M.tolist(), L0_Mpc=L0, chi_Mpc=tr.chi, transfer=str(tr.path),
                dims=[len(b['labels']) for b in blks], chars=[b['char'] for b in blks],
                octant_modes_used=nused, full_modes_used=nfull, seconds=time.perf_counter() - t0,
                pid=os.getpid(), continuum_complement=True, comparison_volume='L0^3',
                builder='ens_core.build_cov (working copy of snapshot high_cov.build)')
    return cc, meta


def save_cov(path, cc, meta):
    p = out_path(path)
    tmp = p.with_name(p.stem + '.tmp.npz')
    np.savez(tmp, meta=json.dumps(meta, default=_jd), **{'C%d' % i: c for i, c in enumerate(cc)})
    os.replace(tmp, p)


def load_cov(path):
    with np.load(path) as f:
        meta = json.loads(str(f['meta']))
        return [np.array(f['C%d' % i]) for i in range(8)], meta


# ----------------------------------------------------------------------------- fast / Cholesky KL
def sel_key(sel):
    return '%s%d' % (sel[0], sel[1])


def sel_index(labels, lsel, fields):
    fl = (0, 1) if fields == 'TE' else ((0,) if fields == 'T' else (1,))
    return np.flatnonzero((labels[:, 1] <= lsel) & np.isin(labels[:, 0], fl))


def uscale(labels, tr):
    """Universal whitening 1/sqrt(C_l^XX covering space): same for every cube/fossil (KL invariant)."""
    d = np.where(labels[:, 0] == 0, tr.cl['TT'][labels[:, 1] - 2], tr.cl['EE'][labels[:, 1] - 2])
    return 1.0 / np.sqrt(d)


_IU = {}


def _triu(n):
    if n not in _IU:
        _IU[n] = np.triu_indices(n)
    return _IU[n]


def cube_inverse(cc, lmax, sels):
    tr = get_tr(lmax); blks = layouts_c(lmax); out = {}
    for sel in sels:
        packs, diags = [], []; ld = 0.0; n = 0
        for c, b in zip(cc, blks):
            ix = sel_index(b['labels'], sel[1], sel[0])
            d = uscale(b['labels'][ix], tr)
            q = c[np.ix_(ix, ix)] * d[:, None] * d[None, :]
            u, info = dpotrf(q, lower=0, clean=1, overwrite_a=1)
            if info != 0:
                raise ValueError('cube block not positive definite (info=%d)' % info)
            ld += 2.0 * float(np.sum(np.log(np.diag(u))))
            inv, info = dpotri(u, lower=0, overwrite_c=1)
            if info != 0:
                raise ValueError('dpotri failed (info=%d)' % info)
            packs.append(np.ascontiguousarray(inv[_triu(len(ix))]))
            diags.append(np.ascontiguousarray(np.diag(inv)))
            n += len(ix)
        out[sel_key(sel)] = dict(packs=packs, diags=diags, logdet=ld, n=n)
    return out


def save_inv(path, inv):
    arrs = {}; meta = {}
    for k, v in inv.items():
        meta[k] = dict(logdet=v['logdet'], n=v['n'], nb=len(v['packs']))
        for i, (p, d) in enumerate(zip(v['packs'], v['diags'])):
            arrs['%s_p%d' % (k, i)] = p; arrs['%s_d%d' % (k, i)] = d
    p = out_path(path)
    tmp = p.with_name(p.stem + '.tmp.npz')
    np.savez(tmp, meta=json.dumps(meta), **arrs)
    os.replace(tmp, p)


def load_inv(path, keys=None):
    with np.load(path) as f:
        meta = json.loads(str(f['meta'])); out = {}
        for k, m in meta.items():
            if keys is not None and k not in keys:
                continue
            out[k] = dict(logdet=m['logdet'], n=m['n'],
                          packs=[np.array(f['%s_p%d' % (k, i)]) for i in range(m['nb'])],
                          diags=[np.array(f['%s_d%d' % (k, i)]) for i in range(m['nb'])])
    return out


def fossil_packs(cc, lmax, sels):
    tr = get_tr(lmax); blks = layouts_c(lmax); out = {}
    for sel in sels:
        pk, dg = [], []
        for c, b in zip(cc, blks):
            ix = sel_index(b['labels'], sel[1], sel[0])
            d = uscale(b['labels'][ix], tr)
            p = c[np.ix_(ix, ix)] * d[:, None] * d[None, :]
            pk.append(np.ascontiguousarray(p[_triu(len(ix))])); dg.append(np.diag(p).copy())
        out[sel_key(sel)] = dict(packs=pk, diags=dg)
    return out


def fossil_logdets(cc, lmax, sels):
    tr = get_tr(lmax); blks = layouts_c(lmax); out = {}
    for sel in sels:
        ld = 0.0; n = 0
        for c, b in zip(cc, blks):
            ix = sel_index(b['labels'], sel[1], sel[0])
            d = uscale(b['labels'][ix], tr)
            p = c[np.ix_(ix, ix)] * d[:, None] * d[None, :]
            L = cholesky(p, lower=True, check_finite=False)
            ld += 2.0 * float(np.sum(np.log(np.diag(L)))); n += len(ix)
        out[sel_key(sel)] = dict(logdet=ld, n=n)
    return out


def kl_fast(fp, ld_p, ci):
    """D = 1/2 [tr(Q^-1 P) - n + ln det Q - ln det P]; trace from packed upper triangles.
    Returns (D, t/n) where t/n is the free-amplitude optimum a* (diagnostic)."""
    t = 0.0
    for pP, dP, pQ, dQ in zip(fp['packs'], fp['diags'], ci['packs'], ci['diags']):
        t += 2.0 * float(np.dot(pQ, pP)) - float(np.dot(dQ, dP))
    return 0.5 * (t - ci['n'] + ci['logdet'] - ld_p), t / ci['n']


def kl_chol(P, Q, lmax, lsel, fields='TE', noise=None):
    """Cholesky KL: tr(Q^-1 P) = ||Lq^-1 Lp||_F^2; optional additive isotropic noise on both."""
    tr = get_tr(lmax); blks = layouts_c(lmax); D = 0.0
    for p, q, b in zip(P, Q, blks):
        ix = sel_index(b['labels'], lsel, fields); lab = b['labels'][ix]
        d = uscale(lab, tr)
        ps = p[np.ix_(ix, ix)]; qs = q[np.ix_(ix, ix)]
        if noise is not None:
            nv = np.where(lab[:, 0] == 0, noise['TT'][lab[:, 1] - 2], noise['EE'][lab[:, 1] - 2])
            di = np.diag_indices(len(ix)); ps[di] += nv; qs[di] += nv
        ps *= d[:, None] * d[None, :]; qs *= d[:, None] * d[None, :]
        lq = cholesky(qs, lower=True, check_finite=False)
        lp = cholesky(ps, lower=True, check_finite=False)
        w = solve_triangular(lq, lp, lower=True, check_finite=False)
        D += 0.5 * (float(np.sum(w * w)) - len(ix) + 2 * float(np.sum(np.log(np.diag(lq))))
                    - 2 * float(np.sum(np.log(np.diag(lp)))))
    return float(D)


def noise_spec(lmax, dT, dP, fwhm):
    """White noise (uK-arcmin) deconvolved by a Gaussian beam (FWHM arcmin), in (dT/T)^2 units."""
    l = np.arange(2, lmax + 1)
    sig = fwhm * ARCMIN / np.sqrt(8 * np.log(2))
    binv2 = np.exp(l * (l + 1) * sig ** 2)
    return {'TT': (dT * ARCMIN / T_CMB_UK) ** 2 * binv2, 'EE': (dP * ARCMIN / T_CMB_UK) ** 2 * binv2}


def taper_update(base_path, new_path, lmax, xmax, old=0.25, new=0.125):
    """window_delta.update generalised: C_new = C_old + lattice[W_new-W_old] + tail_new - tail_old."""
    cc, meta = load_cov(base_path)
    tr = get_tr(lmax)
    diag = np.array(meta['M']).diagonal(); L0 = meta['L0_Mpc']; s = meta['s']
    modes = octant_modes(s, diag, xmax)
    x = np.linalg.norm(modes * diag, axis=1) * np.pi / s
    modes = modes[x > (1 - old) * xmax]
    phase = (-1.) ** (np.arange(2, lmax + 1) // 2)
    blks = layouts_c(lmax); pref = (4 * np.pi) ** 2 / L0 ** 3
    inc = [np.zeros_like(c, order='F') for c in cc]
    for start in range(0, len(modes), 2048):
        ns = modes[start:start + 2048]
        kv = 2 * np.pi / L0 * ns * diag; k = np.linalg.norm(kv, axis=1); xx = k * tr.chi
        dw = window(xx, (1 - new) * xmax, xmax) - window(xx, (1 - old) * xmax, xmax)
        if dw.min() < -1e-13:
            raise ValueError('window difference must be nonnegative')
        keep = (xx < xmax) & (dw > 0)
        ns, kv, k, xx, dw = ns[keep], kv[keep], k[keep], xx[keep], dw[keep]
        if not len(xx):
            continue
        mult = 2. ** np.sum(ns != 0, axis=1)
        wt = pref * mult * 2 * np.pi ** 2 * tr.calP(k) / k ** 3 * dw
        y = real_ylm(kv / k[:, None], lmax); f = tr.eval(xx)
        f[:, :lmax - 1] *= np.sqrt(wt)[:, None] * phase
        f[:, lmax - 1:] *= np.sqrt(wt)[:, None] * phase
        for i, b in enumerate(blks):
            n = b['n']; ls = b['ls'] - 2
            Am = np.empty((len(k), 2 * n)); yp = y[:, b['ycol']]
            np.multiply(yp, f[:, ls], out=Am[:, :n]); np.multiply(yp, f[:, ls + lmax - 1], out=Am[:, n:])
            inc[i] = dsyrk(1., Am.T, beta=1., c=inc[i], trans=0, lower=0, overwrite_c=1)
    tn = tr.tail((1 - new) * xmax, xmax); to = tr.tail((1 - old) * xmax, xmax)
    td = {f_: tn[f_] - to[f_] for f_ in tn}
    for i, b in enumerate(blks):
        cc[i] += np.triu(inc[i]) + np.triu(inc[i], 1).T
        add_diag_tl(cc[i], b['labels'], td)
    meta = dict(meta, taper=new, exact_window_update_base=str(base_path))
    save_cov(new_path, cc, meta)
    return cc, meta


# ----------------------------------------------------------------------------- ensemble edge interpolation
def span_eval(var, A, xi):
    """var: {'thetas': [th_obl, 0, th_pro], 'A': {th_label: [...]}, 'span': {th_label: [...]}}.
    span = ln(u_max/0.985), piecewise linear in A per canonical theta (linear extrapolation below the
    smallest A, constant u_max above the largest A), quadratic in xi = sin(3 theta) through xi=-1,0,1."""
    A = np.asarray(A, float); xi = np.asarray(xi, float)
    vals = []
    for lab in ('obl', 'tri', 'pro'):
        a = np.asarray(var['A'][lab], float); sp = np.asarray(var['span'][lab], float)
        o = np.argsort(a); a = a[o]; sp = sp[o]
        v = np.interp(A, a, sp)
        if len(a) >= 2:
            lo = A < a[0]
            slope = (sp[1] - sp[0]) / (a[1] - a[0])
            v = np.where(lo, sp[0] + slope * (A - a[0]), v)
        vals.append(v)
    sm, s0, sp_ = vals
    return s0 + 0.5 * (sp_ - sm) * xi + (0.5 * (sp_ + sm) - s0) * xi ** 2


# ----------------------------------------------------------------------------- worker tasks
def init_worker():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def blas_info():
    try:
        from threadpoolctl import threadpool_info
        return [dict(api=d.get('internal_api'), n=d.get('num_threads')) for d in threadpool_info()]
    except Exception as e:  # pragma: no cover
        return repr(e)


def _t_cov(sp):
    kind = sp['kind']; s = sp['s']; lmax = sp['lmax']; xmax = sp['xmax']; taper = sp['taper']
    A = sp.get('A', 0.0); th = sp.get('theta', 0.0)
    path, exists = find_cov(kind, s, lmax, xmax, taper, A, th)
    out = dict(name=path.stem, path=str(path), kind=kind, s=s, A=A, theta=th, lmax=lmax, xmax=xmax,
               taper=taper, built=False, build_seconds=None, from_snapshot=SNAP_CACHE in path.parents)
    cc = None; meta = None
    if not exists:
        t = time.perf_counter()
        cc, meta = build_cov(kind, s, lmax, xmax, taper, A, th)
        save_cov(path, cc, meta)
        out.update(built=True, build_seconds=time.perf_counter() - t,
                   octant_modes=meta['octant_modes_used'], full_modes=meta['full_modes_used'])
    prep = sp.get('prep')
    if prep == 'inv':
        ip = CACHE / ('inv_' + path.stem + '.npz')
        if not ip.exists():
            if cc is None:
                cc, meta = load_cov(path)
            save_inv(ip, cube_inverse(cc, lmax, [tuple(x) for x in sp['sels']]))
        out['inv'] = str(ip)
    elif prep == 'ld':
        lp = CACHE / ('ld_' + path.stem + '.json')
        if not lp.exists():
            if cc is None:
                cc, meta = load_cov(path)
            save_json(lp, fossil_logdets(cc, lmax, [tuple(x) for x in sp['sels']]))
        out['ld'] = str(lp)
    if sp.get('blas'):
        out['blas'] = blas_info()
    return out


def _t_fresh(sp):
    lmax, xmax, taper, s = sp['lmax'], sp['xmax'], sp['taper'], sp['s']
    t = time.perf_counter()
    cc, meta = build_cov('shape', s, lmax, xmax, taper, S2, 0.0)
    bsec = time.perf_counter() - t
    fp = CACHE / ('fresh_' + cov_name('shape', s, lmax, xmax, taper, S2, 0.0) + '.npz')
    save_cov(fp, cc, meta)
    out = dict(build_seconds=bsec, octant_modes=meta['octant_modes_used'], path=str(fp), blas=blas_info())
    snapf = SNAP_CACHE / ('fossil_s%.5f_l%d_x%g_t%g.npz' % (s, lmax, xmax, taper))
    snapc = SNAP_CACHE / ('cube_s%.5f_l%d_x%g_t%g.npz' % (s, lmax, xmax, taper))
    if snapf.exists() and snapc.exists():
        P0, m0 = load_cov(snapf)
        out['max_whitened_diff_vs_snapshot'] = max(
            float(np.max(np.abs(a - b) / np.sqrt(np.outer(np.diag(b), np.diag(b))))) for a, b in zip(cc, P0))
        out['octant_modes_snapshot'] = m0.get('octant_modes_used')
        Q, _ = load_cov(snapc)
        k1 = kl_eig(cc, Q, lmax, lmax, 'TE'); k0 = kl_eig(P0, Q, lmax, lmax, 'TE', check=False)
        out.update(D_fresh=k1['D'], D_fresh_logdet=k1['D_logdet'], D_snapshot_cache=k0['D'],
                   D_P5_table=sp.get('ref'), abs_diff_fresh_vs_cache=abs(k1['D'] - k0['D']))
    return out


def _t_regress(sp):
    lmax = 80
    fpath = SNAP_CACHE / 'fossil_s1.13000_l80_x320_t0.25.npz'
    cpath = SNAP_CACHE / 'cube_s1.13000_l80_x320_t0.25.npz'
    P, _ = load_cov(fpath); Q, _ = load_cov(cpath)
    ref = {'TE30': 5.226312900009626, 'TE40': 7.12526495300258, 'TE60': 9.767124436871306,
           'TE80': 11.349524263076837, 'T80': 6.18594990443896}
    sels = [('TE', 30), ('TE', 40), ('TE', 60), ('TE', 80), ('T', 80)]
    res = {}
    ci = cube_inverse(Q, lmax, sels); fpk = fossil_packs(P, lmax, sels); fld = fossil_logdets(P, lmax, sels)
    for sel in sels:
        key = sel_key(sel)
        k = kl_eig(P, Q, lmax, sel[1], sel[0])
        dfast, a = kl_fast(fpk[key], fld[key]['logdet'], ci[key])
        dch = kl_chol(P, Q, lmax, sel[1], sel[0])
        res[key] = dict(D_eig=k['D'], D_logdet=k['D_logdet'], ref=ref[key], abs_err=abs(k['D'] - ref[key]),
                        lam_min=k['lam_min'], lam_max=k['lam_max'], D_fast=dfast, a_star=a, D_chol=dch,
                        fast_minus_eig=dfast - k['D'], chol_minus_eig=dch - k['D'])
    return dict(regression=res, fossil=str(fpath), cube=str(cpath), recomputed_covariance=False)


def _t_par(sp):
    lmax, xmax, taper, s = sp['lmax'], sp['xmax'], sp['taper'], sp['s']
    p = CACHE / ('par_%s_s%.5f_l%d_x%g_t%g.npz' % (sp['label'], s, lmax, xmax, taper))
    if not p.exists():
        tr = get_tr(lmax); tc = TorusCov(tr, lmax)
        L0 = s * 2 * tr.chi
        M = kmatrix(sp['beta5'])
        nlist, R = tc.mode_list(L0, xmax, [M])
        C, info = tc.compute(L0, M, xmax, nlist, taper=taper)
        meta = dict(label=sp['label'], beta5=sp['beta5'], M=M.tolist(), s=s, lmax=lmax, xmax=xmax,
                    taper=taper, R=R, **info)
        pp = out_path(p); tmp = pp.with_name(pp.stem + '.tmp.npz')
        np.savez(tmp, meta=json.dumps(meta, default=_jd), C0=C[0], C1=C[1]); os.replace(tmp, pp)
    return dict(path=str(p), label=sp['label'])


def _load_par(path):
    with np.load(path) as f:
        return [np.array(f['C0']), np.array(f['C1'])], json.loads(str(f['meta']))


def _t_oct(sp):
    """octant build at verification lmax; 'fresh' forces a new file even if the snapshot has one."""
    lmax, xmax, taper, s = sp['lmax'], sp['xmax'], sp['taper'], sp['s']
    A, th = sp.get('A', 0.0), sp.get('theta', 0.0)
    if sp['kind'] == 'cube':
        path, ex = find_cov('cube', s, lmax, xmax, taper)
    else:
        path = CACHE / (('fresh_' if sp.get('fresh') else '') + cov_name('shape', s, lmax, xmax, taper, A, th) + '.npz')
        ex = path.exists()
    if not ex:
        cc, meta = build_cov(sp['kind'], s, lmax, xmax, taper, A, th)
        save_cov(path, cc, meta)
    return dict(path=str(path), label=sp['label'])


def kl_parity(Cp, Cq, tc, lsel, fields='TE'):
    D = 0.0
    for p in (0, 1):
        ix = tc.select(p, lsel, fields); lab = tc.blocks[p]['labels'][ix]
        d = uscale(lab, tc.tr)
        a = Cp[p][np.ix_(ix, ix)] * d[:, None] * d[None, :]
        b = Cq[p][np.ix_(ix, ix)] * d[:, None] * d[None, :]
        D += kl_gauss(a, b)[0]
    return D


def cross_char_max(Cpar, tc):
    out = 0.0
    for p in (0, 1):
        lab = tc.blocks[p]['labels']
        cid = np.array([4 * a + 2 * b + c for a, b, c in (chars(l, m) for _, l, m in lab)])
        dg = np.sqrt(np.diag(Cpar[p]))
        sc = np.abs(Cpar[p]) / np.outer(dg, dg)
        out = max(out, float(sc[cid[:, None] != cid[None, :]].max()))
    return out


def par_vs_oct(Cpar, Coct, tc, lmax):
    """Reorder parity blocks into the 8 octant character blocks and compare (whitened max diff)."""
    err = 0.0
    for b, co in zip(layouts_c(lmax), Coct):
        lab = b['labels']; p = int(lab[0, 1] % 2)
        plab = tc.blocks[p]['labels']
        lookup = {tuple(v): i for i, v in enumerate(plab)}
        ix = np.array([lookup[tuple(v)] for v in lab])
        sub = Cpar[p][np.ix_(ix, ix)]
        dg = np.sqrt(np.diag(co))
        err = max(err, float(np.max(np.abs(sub - co) / np.outer(dg, dg))))
    return err


def _t_l40an(sp):
    v = sp['ver']; lmax, xmax, taper, s = v['lmax'], v['xmax'], v['taper'], v['s']
    paths = sp['paths']; out = {}
    tr = get_tr(lmax); tc = TorusCov(tr, lmax)
    Coc, _ = load_cov(paths['oct_cube'])
    # (i) fresh generalised-builder rep vs snapshot fossil
    Crep, _ = load_cov(paths['oct_rep'])
    out['rep_D_TE%d' % lmax] = kl_eig(Crep, Coc, lmax, lmax)['D']
    snapf = SNAP_CACHE / ('fossil_s%.5f_l%d_x%g_t%g.npz' % (s, lmax, xmax, taper))
    if snapf.exists():
        Cs, _ = load_cov(snapf)
        out['rep_fresh_vs_snapshot_max_whitened'] = max(
            float(np.max(np.abs(a - b) / np.sqrt(np.outer(np.diag(b), np.diag(b))))) for a, b in zip(Crep, Cs))
        out['rep_D30'] = kl_eig(Crep, Coc, lmax, 30)['D']; out['rep_D30_ref'] = 5.26139590158844
        out['rep_D40_ref'] = 7.225715703569919
        del Cs
    # (ii) generic theta: exactness of the 8-block reduction
    Cgo, _ = load_cov(paths['oct_gen'])
    Cgp, _ = _load_par(paths['par_gen']); Ccp, _ = _load_par(paths['par_cube'])
    out['generic'] = dict(A=sp['gen'][0], theta=sp['gen'][1],
                          cross_character_max_whitened_parity_fossil=cross_char_max(Cgp, tc),
                          cross_character_max_whitened_parity_cube=cross_char_max(Ccp, tc),
                          parity_vs_octant_fossil_max_whitened=par_vs_oct(Cgp, Cgo, tc, lmax),
                          parity_vs_octant_cube_max_whitened=par_vs_oct(Ccp, Coc, tc, lmax))
    dfull = kl_parity(Cgp, Ccp, tc, lmax); dblk = kl_eig(Cgo, Coc, lmax, lmax)
    out['generic'].update(D_full_parity=dfull, D_8blocks=dblk['D'], abs_diff=abs(dfull - dblk['D']),
                          D_8blocks_logdet=dblk['D_logdet'])
    # (iii) D3 equivalence (A, pi/3) ~ (A, 0)
    Cd3, _ = load_cov(paths['oct_d3'])
    dd3 = kl_eig(Cd3, Coc, lmax, lmax)['D']
    out['D3'] = dict(D_A_sqrt2_theta_pi3=dd3, D_rep=out['rep_D_TE%d' % lmax], abs_diff=abs(dd3 - out['rep_D_TE%d' % lmax]))
    # (iv) T2 magnitude at the representative point and at the cube point
    t2 = {}
    for lab in ('rep_d1', 'rep_d2', 'eps_Xd', 'eps_X12'):
        Cx, mx = _load_par(paths['par_' + lab])
        t2[lab] = dict(beta5=mx['beta5'], D=kl_parity(Cx, Ccp, tc, lmax),
                       cross_character_max_whitened=cross_char_max(Cx, tc))
        del Cx
    t2['D_rep_reference'] = out['rep_D_TE%d' % lmax]
    t2['dD_rep_d1'] = t2['rep_d1']['D'] - t2['D_rep_reference']
    t2['dD_rep_d2'] = t2['rep_d2']['D'] - t2['D_rep_reference']
    t2['quadratic_ratio_d2_over_d1'] = t2['dD_rep_d2'] / t2['dD_rep_d1'] if t2['dD_rep_d1'] != 0 else None
    t2['fisher_ratio_T2_over_E_at_cube'] = t2['eps_X12']['D'] / t2['eps_Xd']['D']
    out['T2'] = t2
    return out


def _t_brent(sp):
    from scipy.optimize import minimize_scalar
    lmax, xmax, taper = sp['lmax'], sp['xmax'], sp['taper']
    P, _ = load_cov(sp['fossil_path'])
    memo = {round(float(s_), 7): (float(D_), 'pool') for s_, D_ in sp['pool']}
    evals = []

    def f(x):
        x = float(x); key = round(x, 7)
        if key in memo:
            v, src = memo[key]
        else:
            path = CACHE / ('cubeB_s%.7f_l%d_x%g_t%g.npz' % (x, lmax, xmax, taper))
            if path.exists():
                Q, _ = load_cov(path)
            else:
                Q, m = build_cov('cube', x, lmax, xmax, taper); save_cov(path, Q, m)
            v = kl_chol(P, Q, lmax, lmax, 'TE'); src = 'build'
            del Q; gc.collect()
            memo[key] = (v, src)
        evals.append(dict(s_prime=x, D=v, source=src))
        return v

    a, b, c = sp['bracket']
    res = minimize_scalar(f, bracket=(a, b, c), method='brent',
                          options=dict(xtol=sp.get('xtol', 4e-4), maxiter=sp.get('maxiter', 8)))
    uniq = sorted({round(e['s_prime'], 7): e['D'] for e in evals}.items())
    return dict(fossil=sp['fossil_name'], bracket=[a, b, c], s_star=float(res.x), D_star=float(res.fun),
                n_unique_evaluations=len(uniq), n_new_builds=sum(1 for e in evals if e['source'] == 'build'),
                unique=uniq, evals=evals, success=bool(getattr(res, 'success', True)),
                interior=bool(a < float(res.x) < c))


def _t_taper(sp):
    base, ex = find_cov(sp['kind'], sp['s'], sp['lmax'], sp['xmax'], 0.25, sp.get('A', 0.0), sp.get('theta', 0.0))
    if not ex:
        raise FileNotFoundError(str(base))
    newp, ex2 = find_cov(sp['kind'], sp['s'], sp['lmax'], sp['xmax'], 0.125, sp.get('A', 0.0), sp.get('theta', 0.0))
    if not ex2:
        taper_update(base, newp, sp['lmax'], sp['xmax'])
    return dict(path=str(newp), base=str(base))


def _t_conv_an(sp):
    lmax = sp['lmax']; res = {}
    for setting, d in sp['settings'].items():
        P, _ = load_cov(d['fossil']); Qf, _ = load_cov(d['cube_fixed']); Qp, _ = load_cov(d['cube_prof'])
        res[setting] = dict(D_fixed_TE80=kl_chol(P, Qf, lmax, lmax), D_prof_TE80=kl_chol(P, Qp, lmax, lmax),
                            D_fixed_TE30=kl_chol(P, Qf, lmax, 30), D_prof_TE30=kl_chol(P, Qp, lmax, 30))
        del P, Qf, Qp; gc.collect()
    base = res.get('x240_t0.25')
    if base:
        for k, r in res.items():
            r['rel_vs_x240_t0.25'] = {q: r[q] / base[q] - 1.0 for q in base if not q.startswith('rel')}
    return dict(target=sp['target'], s_prime_profile=sp['s_prime'], results=res)


def _t_noise(sp):
    lmax = sp['lmax']; P, _ = load_cov(sp['fossil_path'])
    levels = sp['levels']; Ns = {lv['name']: noise_spec(lmax, lv['dT'], lv['dP'], lv['fwhm']) for lv in levels}
    rows = []
    for s_, cpath in sp['cubes']:
        Q, _ = load_cov(cpath)
        row = dict(s_prime=s_)
        for nm, N in Ns.items():
            for sel in sp['sels']:
                row['%s_%s' % (nm, sel_key(sel))] = kl_chol(P, Q, lmax, sel[1], sel[0], noise=N)
        rows.append(row); del Q; gc.collect()
    return dict(fossil=sp['fossil_name'], rows=rows)


_TASKS = dict(cov=_t_cov, fresh=_t_fresh, regress=_t_regress, par=_t_par, oct=_t_oct, l40an=_t_l40an,
              brent=_t_brent, taper=_t_taper, conv_an=_t_conv_an, noise=_t_noise)


def run_task(sp):
    t0 = time.perf_counter()
    try:
        r = _TASKS[sp['task']](sp)
        r.update(ok=True, task=sp['task'], tid=sp.get('tid'), seconds=time.perf_counter() - t0, pid=os.getpid())
        return r
    except Exception as e:
        return dict(ok=False, task=sp.get('task'), tid=sp.get('tid'), spec={k: v for k, v in sp.items() if k != 'pool'},
                    error=repr(e), tb=traceback.format_exc(), seconds=time.perf_counter() - t0, pid=os.getpid())
