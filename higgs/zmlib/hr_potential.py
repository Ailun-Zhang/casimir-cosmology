# PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/realtime_2d/hr_potential.py (copied 2026-10-15 by code-higgs)
# PROVENANCE: copied from 04_calculations/code/20261013_claude_v12_r2/higgs_final/hr_potential.py on 2026-10-14 by realtime-2d (C157/D129); code unchanged unless a later '# R2D:' comment says otherwise.
# PROVENANCE: copied UNCHANGED (this line added) from 04_calculations/code/20261013_claude_v12_r2/higgs_realtime/hr_potential.py on 2026-10-13 by agent higgs-final (C144/D116).
# -*- coding: utf-8 -*-
"""[v12 package, 2026-10-15] Potential tables for the real-time zero-mode evolution (D111/D129): U(r, u) along the turn with running lam, pluggable sources.  Paper: PRD App. H3. Units: box length L = 1 (energies in 1/L); couplings MS-bar at the stated scale.

C139 / D111 (higgs-realtime): PLUGGABLE zero-mode potential, conformal frame, curvature term NOT included.

Table file (npz, one per m_t), keys:
  r[nr]   uniform radial grid (conformal variable, units 1/L), r[0] = dr
  u[nu]   nodes u = h t
  U[nu,nr] U_nc(r;u)  (gauge zero modes + finite-volume matching + lambda r^4/4 + 3/(8r^2) + Born-Huang), 1/L units
  lam[nu] lambda multiplying r^4/4 (MS-bar at mu = 1/L(u))
  logmu[nm], lam_run[nm]  running lambda(mu) (GeV), only used by the optional RG(phi) variant of the classical stage
  meta    json string
Anything else (e.g. the corrected MS-bar CW / Coulomb-gauge matching of higgs-matching) can be dropped in by writing a
file with the same keys; the TDSE / classical pipeline only talks to PotentialTable.

Build of the MU table (R14/xi_scan recipe: gauge profile with the m_t = 172.6 couplings at mu = mu_b/sqrt(cosh 2u),
lambda, y_t per m_t, 2-loop + 3-loop calibration offset):   python -B hr_potential.py build <outdir>
R14_res.py / R14_fv.py / R14_rge.py are copies (header added, code unchanged) of
04_calculations/code/20261013_claude_v12/higgs_survival/xi_scan/.
"""
import os as _os  # [v12-package] path bootstrap (added 2026-10-15 by code-higgs): zmlib + package root
import sys as _sys  # [v12-package]
_ZM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "zmlib")  # [v12-package]
if _ZM not in _sys.path:  # [v12-package]
    _sys.path.insert(1, _ZM)  # [v12-package]
import _zpaths as _zp  # noqa: E402  [v12-package] package paths (higgs/zmlib/_zpaths.py)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HLB = 0.00437243
MU_B = 2.435e18 / 722.0
H_GEV = HLB * MU_B
MTS = (172.0, 172.2, 172.6)
AS, MH = 0.1180, 125.13
UNODES = np.round(np.arange(0.0, 7.0001, 0.25), 6)
DR_TAB = 0.0025
R_TAB = 17.5


def _node(args):
    i, u, g2, gY, lam_list, yt_list = args
    import R14_res as R
    gp = R.gauss_profile(g2, gY)
    nr = int(round(R_TAB / DR_TAB))
    r = np.arange(1, nr + 1) * DR_TAB
    s, alpha, beta = R.spar(gp)
    rows = [R.U_radial(gp, "MU", lam, g2, gY, yt, r, 0.0) for lam, yt in zip(lam_list, yt_list)]
    return i, np.array(rows), dict(s=float(s), alpha=float(alpha), beta=float(beta), maxdec=float(gp["maxdec"]),
                                   min_hess_eig=float(gp["min_hess_eig"]))


def build(outdir):
    import R14_rge
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    cal, _ = R14_rge.calibration()
    mus = R14_rge.MU_B / np.sqrt(np.cosh(2 * UNODES))
    Yc = R14_rge.solve(172.6, AS, MH).sol(np.log(mus))
    sols = {mt: R14_rge.solve(mt, AS, MH) for mt in MTS}
    Y = {mt: sols[mt].sol(np.log(mus)) for mt in MTS}
    jobs = [(i, float(u), float(Yc[1, i]), float(Yc[0, i]), [float(Y[mt][4, i] + cal) for mt in MTS],
             [float(Y[mt][3, i]) for mt in MTS]) for i, u in enumerate(UNODES)]
    nr = int(round(R_TAB / DR_TAB))
    U = np.zeros((len(MTS), len(UNODES), nr))
    info = [None] * len(UNODES)
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, rows, inf in ex.map(_node, jobs):
            U[:, i, :] = rows
            info[i] = inf
    r = np.arange(1, nr + 1) * DR_TAB
    logmu = np.linspace(np.log(1e10), np.log(3e18), 600)
    files = {}
    for k, mt in enumerate(MTS):
        lam = np.array([Y[mt][4, i] + cal for i in range(len(UNODES))])
        meta = dict(model="MU (D108/R14), xi_scan coupling recipe", m_t=mt, alpha_s=AS, M_h=MH, cal=cal, hLb=HLB,
                    mu_b=MU_B, h_GeV=H_GEV, profile_couplings="m_t=172.6 g2,gY at each node", node_info=info,
                    built=time.strftime("%Y-%m-%d %H:%M:%S"), build_s=time.time() - t0)
        fn = os.path.join(outdir, "U_table_MU_mt%.1f.npz" % mt)
        np.savez(fn, r=r, u=UNODES, U=U[k], lam=lam, g2=Yc[1], gY=Yc[0], yt=Y[mt][3], mu=mus, logmu=logmu,
                 lam_run=sols[mt].sol(logmu)[4] + cal, meta=json.dumps(meta))
        files["%.1f" % mt] = fn
    print(json.dumps(dict(files=files, time_s=time.time() - t0, lam_u0={"%.1f" % mt: float(Y[mt][4, 0] + cal)
                                                                         for mt in MTS}), indent=1))


class PotentialTable:
    """U_nc(r;u): cubic spline in u of r^2 U (smooth at r -> 0), linear in r on the fine table grid;
    beyond r_T: U(r_T) + lam_T (r^4 - r_T^4)/4 + c_T (r - r_T), c_T from derivative matching (as xi_scan)."""

    def __init__(self, path, u_clamp=None):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.r = np.array(d["r"])
        self.u = np.array(d["u"])
        self.dr = float(self.r[1] - self.r[0])
        self.rT = float(self.r[-1])
        self.r2 = self.r ** 2
        self._spl = CubicSpline(self.u, np.array(d["U"]) * self.r2[None, :], axis=0)
        self.lam_nodes = np.array(d["lam"])
        self._lam = CubicSpline(self.u, self.lam_nodes)
        self.logmu = np.array(d["logmu"])
        self.lam_run = np.array(d["lam_run"])
        self.beta_run = np.gradient(self.lam_run, self.logmu)
        self.meta = json.loads(str(d["meta"]))
        self.umax = float(self.u[-1]) if u_clamp is None else float(u_clamp)
        self._cache_u = None

    def _uc(self, u):
        return min(max(float(u), float(self.u[0])), self.umax)

    def lam(self, u):
        return float(self._lam(self._uc(u)))

    def _vec(self, u):
        uc = self._uc(u)
        if self._cache_u != uc:
            v = self._spl(uc) / self.r2
            dv = np.gradient(v, self.dr, edge_order=2)
            self._cache = (v, dv)
            self._cache_u = uc
        return self._cache

    def U_grid(self, u):
        """U_nc on the table grid itself."""
        return self._vec(u)[0].copy()

    def U_on(self, rq, u):
        v, dv = self._vec(u)
        rq = np.asarray(rq, float)
        out = np.empty_like(rq)
        inn = rq <= self.rT
        r2u = np.interp(rq[inn], self.r, v * self.r2)
        out[inn] = r2u / np.maximum(rq[inn], 1e-12) ** 2
        ext = ~inn
        if ext.any():
            lamT, UT, FT = self.lam(u), v[-1], dv[-1]
            x = rq[ext]
            out[ext] = UT + lamT * (x ** 4 - self.rT ** 4) / 4 + (FT - lamT * self.rT ** 3) * (x - self.rT)
        return out

    def dU_on(self, rq, u):
        v, dv = self._vec(u)
        rq = np.asarray(rq, float)
        out = np.empty_like(rq)
        inn = rq <= self.rT
        out[inn] = np.interp(rq[inn], self.r, dv)
        ext = ~inn
        if ext.any():
            lamT, FT = self.lam(u), dv[-1]
            out[ext] = lamT * rq[ext] ** 3 + (FT - lamT * self.rT ** 3)
        return out

    def lam_of_mu(self, mu_gev):
        lm = np.log(np.asarray(mu_gev, float))
        return np.interp(lm, self.logmu, self.lam_run), np.interp(lm, self.logmu, self.beta_run)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
