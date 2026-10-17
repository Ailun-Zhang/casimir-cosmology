# PROVENANCE: 04_calculations/code/20261009_claude_R5/obs_fisher/C59_torus.py (copied 2026-10-15 by code-shape-cmb)
r"""[Package note] E1 three-torus T+E harmonic covariance, Gaussian KL and Fisher utilities (D52): lattice k_n = (2 pi/L0) M n, fixed mode list, smooth taper + isotropic continuum. Transfer table: the CAMB 2.0.4 table r5/obs_fisher/C59_transfer.npz (l <= 40, Planck 2018), shipped in expected/ and used unless a fresh one was computed in the work tree by C59_transfer.py. Paper: PRD Sec. X (s10_sky.tex), a19_cmb.tex (covariance formula).

C59_torus: E1 three-torus scalar T+E harmonic covariance, Gaussian KL and Fisher utilities.

Independent implementation (C35/C48 were read for conventions only; nothing is imported from them).

Geometry
    Lattice basis B = L0 R G^{1/2},  G = exp(2 sum_A beta_A X_A) (det G = 1, V = L0^3 fixed),
    R = exp(sum_a theta_a J_a).  Reciprocal vectors k_n = 2 pi B^{-T} n = (2 pi/L0) M n with
    M = exp(-sum beta_A X_A) for a pure shape and M = exp(theta J) for a pure rotation.
    X_d=diag(1,-1,0)/sqrt2, X_d2=diag(1,1,-2)/sqrt6 (cubic irrep E); X_ij=(E_ij+E_ji)/sqrt2 (T2);
    J_ij=(E_ij-E_ji)/sqrt2 (T1, rotations).  Tr X_A X_B = delta_AB.

Covariance (real orthonormal harmonics Y_lm, X,Y in {T,E})
    C^{XY}_{lm,l'm'} = (4pi)^2/V i^{l-l'} sum_n P_zeta(k_n) W(k_n chi*) Delta^X_l Delta^Y_l' Y_lm Y_l'm'
                       + delta_ll' delta_mm' 4pi int dlnk calP Delta^X_l Delta^Y_l (1 - W),
    P_zeta = 2 pi^2 calP/k^3.  n and -n are paired (factor 2 on a half space); because
    Y_lm(-k) = (-1)^l Y_lm(k) the opposite-parity blocks vanish identically and
    i^{l-l'} = (-1)^{(l-l')/2} = (-1)^{floor(l/2)} (-1)^{floor(l'/2)} on the same-parity blocks,
    so C_p = A_p^T A_p with A_{n,(X,l,m)} = sqrt(2 (4pi)^2 P W / V) (-1)^{floor(l/2)} Delta^X_l Y_lm.
    W is a C-infinity taper (1 for x <= (1-f) xmax, 0 for x >= xmax) evaluated at the DEFORMED
    |k_n|; the mode list is FIXED (selected once at beta=0 with a margin covering every deformation
    used in the task), so C(beta) is a smooth function of beta.  The complement (1-W) is added as the
    isotropic continuum, which is exactly shape independent at fixed volume.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
from pathlib import Path
import numpy as np
from scipy.linalg import expm, cholesky, solve_triangular, eigvalsh, svdvals
from scipy.linalg.blas import dsyrk
from scipy.integrate import simpson

HERE = Path(__file__).resolve().parent
import sys  # package
sys.path.insert(0, str(HERE.parent))  # package: chain root (cmb/)
import _cmb_paths as WP  # noqa: E402
OUT = WP.wdir("r5/obs_fisher")
TRANSFER = OUT / "C59_transfer.npz"
if not TRANSFER.exists():  # package: the shipped CAMB 2.0.4 table unless a fresh one exists in the work tree
    TRANSFER = WP.find_archived("r5/obs_fisher/C59_transfer.npz") or TRANSFER
T_CMB_UK = 2.7255e6
ARCMIN = np.pi / 10800.0

_S2, _S6 = np.sqrt(2.0), np.sqrt(6.0)


def _e(i, j):
    m = np.zeros((3, 3)); m[i, j] = 1.0
    return m


GENERATORS = {
    "Xd": np.diag([1.0, -1.0, 0.0]) / _S2,
    "Xd2": np.diag([1.0, 1.0, -2.0]) / _S6,
    "X12": (_e(0, 1) + _e(1, 0)) / _S2,
    "X13": (_e(0, 2) + _e(2, 0)) / _S2,
    "X23": (_e(1, 2) + _e(2, 1)) / _S2,
    "J12": (_e(0, 1) - _e(1, 0)) / _S2,
    "J13": (_e(0, 2) - _e(2, 0)) / _S2,
    "J23": (_e(1, 2) - _e(2, 1)) / _S2,
}
SHAPE = ["Xd", "Xd2", "X12", "X13", "X23"]
ROTATIONS = ["J12", "J13", "J23"]


def kmatrix(gen=None, amp=0.0, beta5=None):
    """M with k_n = (2 pi/L0) M n.  gen: name in GENERATORS; beta5: full 5-vector of shape amplitudes."""
    if beta5 is not None:
        X = sum(float(b) * GENERATORS[g] for b, g in zip(beta5, SHAPE))
        return expm(-X)
    if gen is None or amp == 0.0:
        return np.eye(3)
    if gen.startswith("X"):
        return expm(-amp * GENERATORS[gen])
    return expm(amp * GENERATORS[gen])


def window(x, x1, x2):
    """C-infinity taper: 1 for x<=x1, 0 for x>=x2."""
    x = np.asarray(x, float)
    t = np.clip((x - x1) / (x2 - x1), 0.0, 1.0)
    with np.errstate(divide="ignore", over="ignore"):
        a = np.where(t > 0, np.exp(-1.0 / np.where(t > 0, t, 1.0)), 0.0)
        b = np.where(t < 1, np.exp(-1.0 / np.where(t < 1, 1.0 - t, 1.0)), 0.0)
    return b / (a + b)


class Transfer:
    """Delta^T_l, Delta^E_l (l=2..lmax) on the uniform x = k chi* grid of C59_transfer.npz."""

    def __init__(self, lmax=30, path=TRANSFER):
        a = np.load(path)
        self.path = str(path)
        self.lmax = lmax
        nl = lmax - 1
        self.nl = nl
        self.chi = float(a["chi"]); self.step = float(a["xstep"]); self.xend = float(a["xend"])
        self.x = a["x"]
        self.DT = a["DT"][:nl]; self.DE = a["DE"][:nl]
        self.tab = np.ascontiguousarray(np.concatenate([self.DT, self.DE], axis=0).T)  # (nx, 2nl)
        self.tail400 = {k: a["tail_" + k][:nl].copy() for k in ("TT", "EE", "TE")}
        self.cl = {k: a["cl_" + k][:nl].copy() for k in ("TT", "EE", "TE")}
        self.camb = {k: a["camb_" + k][:nl].copy() for k in ("TT", "EE", "TE")}
        self.As, self.ns, self.pivot = float(a["As"]), float(a["ns"]), float(a["pivot"])

    def calP(self, k):
        return self.As * (np.asarray(k) / self.pivot) ** (self.ns - 1.0)

    def eval(self, x):
        """4-point Lagrange interpolation -> (N, 2 nl) = [Delta^T_2..lmax, Delta^E_2..lmax]."""
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
        """4 pi int dlnx calP Delta^X Delta^Y (1 - W(x)) for the taper W(x1, x2), x2 <= xend."""
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


def real_ylm(u, lmax):
    """Orthonormal real spherical harmonics for unit vectors u (N,3); column l*l+l+m, l=0..lmax.
    Y_l0 = N P_l, Y_lm = sqrt2 N_lm P_l^m cos(m phi) (m>0), sqrt2 N_l|m| P_l^|m| sin(|m| phi) (m<0),
    P_l^m without Condon-Shortley phase (identical to C35's real basis)."""
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
            out[:, c + m] = _S2 * q * cm
            out[:, c - m] = _S2 * q * sm

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
    """Integer n != 0, |n| <= R, in the half space n1>0 or (n1=0,n2>0) or (n1=n2=0,n3>0)."""
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


class TorusCov:
    """Parity-blocked T+E covariance builder for l = 2..lmax."""

    def __init__(self, tr: Transfer, lmax=30):
        if lmax > tr.lmax:
            raise ValueError("Transfer lmax too small")
        self.tr, self.lmax = tr, lmax
        nl = tr.nl
        self.blocks = []
        for p in (0, 1):
            lab = np.array([(X, l, m) for X in (0, 1) for l in range(2, lmax + 1) if l % 2 == p
                            for m in range(-l, l + 1)], int)
            ycol = lab[:, 1] ** 2 + lab[:, 1] + lab[:, 2]
            fcol = lab[:, 0] * nl + lab[:, 1] - 2
            ls = np.array([l for l in range(2, lmax + 1) if l % 2 == p])
            self.blocks.append(dict(labels=lab, ycol=ycol, fcol=fcol, half=len(lab) // 2,
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
        """Fixed half-space list: every n that can satisfy |k_n(M)| chi* < xmax for any M in mats."""
        smin = min(float(svdvals(M).min()) for M in mats)
        R = xmax / smin * L0 / (2 * np.pi * self.tr.chi) * (1 + 1e-9)
        return half_lattice(R), R

    def compute(self, L0, M, xmax, nlist, taper=0.25, wmode="deformed", batch=4096, chunk=400000):
        """wmode='deformed': taper evaluated at |k_n(beta)| (default, smooth, isotropic complement);
        wmode='undeformed': taper evaluated at |k_n(0)| (literal fixed list; taper->0 = sharp list)."""
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
            if wmode == "deformed":
                xw = k * chi
            else:
                xw = kfac * chi * np.sqrt(np.einsum("ij,ij->i", nf, nf))
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

    def cover(self):
        """Covering-space (isotropic) blocks diag_lm [[C_l^TT, C_l^TE],[C_l^TE, C_l^EE]]."""
        cl = dict(self.tr.cl)
        out = []
        for blk in self.blocks:
            c = np.zeros((len(blk["labels"]),) * 2)
            add_diag_tl(c, blk["labels"], cl)
            out.append(c)
        return out

    def select(self, p, lmax_sel, fields):
        lab = self.blocks[p]["labels"]
        fl = {"T": [0], "E": [1], "TE": [0, 1]}[fields]
        return np.flatnonzero((lab[:, 1] <= lmax_sel) & np.isin(lab[:, 0], fl))

    def noise_diag(self, p, NlEE):
        """Diagonal noise vector for block p: N_l^EE on E entries (array over l=2..), 0 on T."""
        lab = self.blocks[p]["labels"]
        return np.where(lab[:, 0] == 1, NlEE[lab[:, 1] - 2], 0.0)

    def scale(self, p):
        """1/sqrt(C_l^XX) whitening vector (congruence; leaves KL and Fisher invariant)."""
        lab = self.blocks[p]["labels"]
        d = np.where(lab[:, 0] == 0, self.tr.cl["TT"][lab[:, 1] - 2], self.tr.cl["EE"][lab[:, 1] - 2])
        return 1.0 / np.sqrt(d)


def add_diag_tl(c, lab, spec):
    d = np.where(lab[:, 0] == 0, spec["TT"][lab[:, 1] - 2], spec["EE"][lab[:, 1] - 2])
    c[np.diag_indices(len(lab))] += d
    iT = np.flatnonzero(lab[:, 0] == 0)
    if len(iT) and len(iT) * 2 == len(lab):
        iE = iT + len(iT)
        te = spec["TE"][lab[iT, 1] - 2]
        c[iT, iE] += te
        c[iE, iT] += te


def noise_EE(muK_arcmin, nl):
    """White polarization noise N_l^EE (dimensionless, Delta T/T_CMB units) for l=2..nl+1."""
    return np.full(nl, (muK_arcmin * ARCMIN / T_CMB_UK) ** 2)


def _whiten(L, a):
    return solve_triangular(L, solve_triangular(L, a, lower=True, check_finite=False).T,
                            lower=True, check_finite=False)


def kl_gauss(Cp, Cq):
    """D( N(0,Cp) || N(0,Cq) ) in nats (eigenvalues of Cq^{-1/2} Cp Cq^{-1/2})."""
    L = cholesky(Cq, lower=True, check_finite=False)
    m = _whiten(L, Cp)
    lam = eigvalsh(0.5 * (m + m.T), check_finite=False)
    if lam.min() <= 0:
        raise ValueError("non positive definite")
    return float(0.5 * np.sum(lam - 1.0 - np.log(lam))), lam


def fisher(C, dlist):
    """F_ab = 1/2 Tr[C^-1 dC_a C^-1 dC_b]."""
    L = cholesky(C, lower=True, check_finite=False)
    T = [_whiten(L, d) for d in dlist]
    n = len(T)
    F = np.empty((n, n))
    for a in range(n):
        for b in range(a, n):
            F[a, b] = F[b, a] = 0.5 * float(np.sum(T[a] * T[b]))
    return F
