# PROVENANCE: 04_calculations/code/20261009_claude_R5/C56_preparation_time_gate.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Latest natural preparation time of non-conformal light fields (D48 Sec. 3): occupation of massless minimally coupled modes declared T e-folds before the turn, budget per degree of freedom; T_max = ln(2 pi/hL_b). Paper: PRD Sec. VI C (sec:s-preparation, Eq. eq:s-Tmax, Fig. fig:s-gate b). Runs: NC100 (v11 figure) and SM_AAA (hL_b = 0.00483, N_C = 25.515), ~20-25 min each on 11 processes.

C56: latest natural preparation time for non-conformal light fields (round R5, D48).

Massless minimally coupled field (each graviton polarization; the inflaton on the plateau;
a minimally coupled Higgs component) on the leading Casimir-de Sitter bounce
a = a_b sqrt(cosh 2ht), for which a''/a = 2 h^2 a^2 exactly (conformal time).
Canonical v = a chi obeys  v_tt + H v_t + (k^2/a^2 - 2h^2) v = 0  (cosmic time).

A Gaussian state is declared T e-folds before the turning point, a(t_i) = a_b e^T, and
evolved to the turning point t = 0, where every torus mode is deep inside the Hubble
radius (k/(a_b h) = 2 pi |n|/(h L_b) >= 924 for N_C = 100). The occupation at t = 0 is
taken relative to the adiabatic vacuum there (omega^2 = k^2 - 2h^2 a_b^2; H = 0).

Hadamard requirement: modes inside the Hubble radius at t_i (k > 1.5 a_i h) are put in the
second-order adiabatic vacuum ('adiabatic2'). Modes outside (k < a_i h) have no adiabatic
vacuum; two natural choices are compared:
  'ground'    : instantaneous ground state of the physical field chi
                (v = 1/sqrt(2k), dv/deta = (aH - i k) v),
  'conformal' : instantaneous conformal vacuum (v = 1/sqrt(2k), dv/deta = -i k v).
Modes with a_i < k < 1.5 a_i use 'conformal'.
Budget: E_exc L_b/(2 pi) = sum_modes |n| N_k must stay below N_C gamma/(2 pi) (D48).
Units h = 1, a_b = 1.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import json, time, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp

import sys  # package: work-tree path shim of the shape chain (see shape/_shape_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shape_paths as WP  # noqa: E402
OUT = WP.wdir('r5/C56_preparation_gate')
GAMMA = 16.532315959761668 / np.pi ** 2


def occupation(args):
    kk, T, state = args          # kk = k/(a_b h)
    ti = -0.5 * np.arccosh(np.exp(2.0 * T))
    a_i = np.sqrt(np.cosh(2 * ti)); H_i = np.tanh(2 * ti)
    if state == 'ground':
        v0 = 1.0 / np.sqrt(2 * kk); dveta = (a_i * H_i - 1j * kk) * v0
    elif state == 'conformal':
        v0 = 1.0 / np.sqrt(2 * kk); dveta = -1j * kk * v0
    elif state == 'adiabatic2':
        w2 = kk ** 2 - 2.0 * a_i ** 2
        w = np.sqrt(w2)
        dw = -4.0 * a_i ** 2 * (a_i * H_i) / (2 * w)     # d(w^2)/deta = -4 a^2 (a'/a), a'/a = a H
        v0 = 1.0 / np.sqrt(2 * w); dveta = (-1j * w - dw / (2 * w)) * v0
    else:
        raise ValueError(state)
    dvt = dveta / a_i

    def rhs(t, y):
        a2 = np.cosh(2 * t); H = np.tanh(2 * t)
        vr, vi, ur, ui = y
        w2 = kk * kk / a2 - 2.0
        return [ur, ui, -H * ur - w2 * vr, -H * ui - w2 * vi]

    period_min = 2 * np.pi / kk           # smallest period occurs at a = 1
    sol = solve_ivp(rhs, (ti, 0.0), [float(np.real(v0)), 0.0, dvt.real, dvt.imag],
                    method='DOP853', rtol=1e-10, atol=1e-13, max_step=min(0.05, period_min / 16))
    vr, vi, ur, ui = sol.y[:, -1]
    v = vr + 1j * vi; dv = ur + 1j * ui          # a_b = 1 -> d/deta = d/dt at t = 0
    w = np.sqrt(kk * kk - 2.0)
    wr = (v * np.conj(dv) - np.conj(v) * dv)     # Wronskian, should be i
    return float((abs(dv) ** 2 + w * w * abs(v) ** 2) / (2 * w) - 0.5), float(abs(wr - 1j))


def r3_counts(mmax):
    R = int(np.sqrt(mmax)) + 1
    r = np.arange(-R, R + 1)
    m = (r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2).ravel()
    return np.bincount(m[m <= mmax], minlength=mmax + 1)


def lattice_sum(nn, occ, mmin, mmax, cnt):
    mm = np.nonzero(cnt)[0]
    mm = mm[(mm >= mmin) & (mm <= mmax) & (mm > 0)]
    if mm.size == 0:
        return 0.0
    rn = np.sqrt(mm)
    occp = np.clip(occ, 1e-300, None)
    logN = np.interp(np.log(rn), np.log(nn), np.log(occp))
    return float(np.sum(cnt[mm] * rn * np.exp(logN)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hLb', type=float, default=0.006796212399230908)   # N_C=100 instant (D38)
    ap.add_argument('--NC', type=float, default=100.0)
    ap.add_argument('--label', type=str, default='NC100')
    ap.add_argument('--procs', type=int, default=int(os.environ.get('DNEC_WORKERS', '11')))  # package: --workers
    ap.add_argument('--Ts', type=str, default='2,4,5,6,6.5,7,7.5,8,9,10')
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    k1 = 2 * np.pi / args.hLb
    Ts = [float(x) for x in args.Ts.split(',')]
    budget = args.NC * GAMMA / (2 * np.pi)
    res = {'hLb': args.hLb, 'NC': args.NC, 'k1_over_ab_h': k1, 'budget_sum_n_quanta': budget, 'rows': []}
    for T in Ts:
        ti = -0.5 * np.arccosh(np.exp(2.0 * T)); a_i = np.sqrt(np.cosh(2 * ti))
        n_hor = a_i / k1                         # |n| at the Hubble radius at t_i
        nmax = max(12.0, 3.0 * n_hor)
        nn = np.unique(np.r_[np.arange(1, 13, dtype=float), np.geomspace(1.0, nmax, 48)])
        sup = nn[k1 * nn < a_i]                  # superhorizon at t_i
        mid = nn[(k1 * nn >= a_i) & (k1 * nn <= 1.5 * a_i)]
        sub = nn[k1 * nn > 1.5 * a_i]
        jobs = [(k1 * x, T, 'ground') for x in sup] + [(k1 * x, T, 'conformal') for x in sup] \
            + [(k1 * x, T, 'conformal') for x in mid] + [(k1 * x, T, 'adiabatic2') for x in sub]
        with Pool(args.procs) as pool:
            out = pool.map(occupation, jobs)
        occ = np.array([o[0] for o in out]); werr = max(o[1] for o in out)
        ns = len(sup); nm = len(mid)
        occ_g, occ_c, occ_m, occ_a = occ[:ns], occ[ns:2 * ns], occ[2 * ns:2 * ns + nm], occ[2 * ns + nm:]
        mmax = int(nmax ** 2) + 1
        cnt = r3_counts(mmax)
        m_hor = (a_i / k1) ** 2
        m_mid = (1.5 * a_i / k1) ** 2
        s_sup_g = lattice_sum(sup, occ_g, 1, m_hor, cnt) if ns > 1 else (float(cnt[1] * occ_g[0]) if ns == 1 else 0.0)
        s_sup_c = lattice_sum(sup, occ_c, 1, m_hor, cnt) if ns > 1 else (float(cnt[1] * occ_c[0]) if ns == 1 else 0.0)
        s_mid = lattice_sum(np.r_[sup[-1:], mid, sub[:1]] if nm else sub[:2], np.r_[occ_c[-1:], occ_m, occ_a[:1]] if nm else occ_a[:2],
                            int(np.floor(m_hor)) + 1, m_mid, cnt) if (nm or len(sub) > 1) else 0.0
        s_sub = lattice_sum(sub, occ_a, int(np.floor(m_mid)) + 1, mmax, cnt) if len(sub) > 1 else 0.0
        row = {'T': T, 'a_i_over_ab': float(a_i), 'n_at_horizon': float(n_hor), 'n_max': float(nmax),
               'sum_sup_ground': s_sup_g, 'sum_sup_conformal': s_sup_c, 'sum_mid': s_mid, 'sum_sub_adiabatic2': s_sub,
               'total_ground': s_sup_g + s_mid + s_sub, 'total_conformal': s_sup_c + s_mid + s_sub,
               'ratio_total_ground_to_budget_per_dof': (s_sup_g + s_mid + s_sub) / budget,
               'ratio_total_conformal_to_budget_per_dof': (s_sup_c + s_mid + s_sub) / budget,
               'max_wronskian_err': werr,
               'samples': {'sup_n': sup.tolist(), 'N_ground': occ_g.tolist(), 'N_conformal': occ_c.tolist(),
                           'mid_n': mid.tolist(), 'N_mid': occ_m.tolist(), 'sub_n': sub.tolist(), 'N_adiabatic2': occ_a.tolist()}}
        res['rows'].append(row)
        print(f"T={T:5.1f} n_hor={n_hor:8.3f} sup(ground)={s_sup_g:.3e} sup(conf)={s_sup_c:.3e} mid={s_mid:.3e} "
              f"sub={s_sub:.3e} | /budget per dof: ground {row['ratio_total_ground_to_budget_per_dof']:.3e} "
              f"conf {row['ratio_total_conformal_to_budget_per_dof']:.3e}  werr={werr:.1e}", flush=True)
    res['runtime_s'] = time.time() - t0
    (OUT / f'C56_{args.label}.json').write_text(json.dumps(res, indent=1))
    print('runtime', res['runtime_s'])


if __name__ == '__main__':
    main()
