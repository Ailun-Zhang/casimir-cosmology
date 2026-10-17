# kl_sens.py -- sigma_E +/-10% sensitivity of the ensemble reading window (T19, unverified-check, 2026-10-14)
# Own model: near-universal reading edge in shortest-cycle units (v11 App. fisher: upper edges 1.011-1.030 in
# shortest-cycle units; no-circle bound u >= 0.985).  read <=> 0.985 <= u = L0 f(b) <= u_edge.
# Ensemble: b ~ N(0, sigma_E^2 I_2) in the E plane, beta_ii = |b| sqrt(2/3) cos(theta - phi_i), phi = 30,150,270 deg.
import os, sys, json, math
import numpy as np
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
SIG0 = 0.1041
rng = np.random.default_rng(2026101951)
Nmc = 2_000_000
b = rng.standard_normal((Nmc, 2))
A = np.hypot(b[:, 0], b[:, 1])
th = np.arctan2(b[:, 1], b[:, 0])
phi = np.radians([30.0, 150.0, 270.0])
cmin = np.min(np.cos(th[:, None] - phi[None, :]), axis=1)   # in [-1, -1/2]
L0 = np.round(np.arange(0.95, 1.351, 0.0005), 5)
out = {'model': 'constant upper edge in shortest-cycle units; Nmc=%d' % Nmc, 'sigma_E0': SIG0, 'cases': {}}
for fac in (0.9, 1.0, 1.1):
    lnf = A * SIG0 * fac * math.sqrt(2 / 3) * cmin   # ln f <= 0
    f = np.exp(lnf)
    fs = np.sort(f)
    for ue in (1.011, 1.020, 1.027, 1.030):
        lo = np.searchsorted(fs, 0.985 / L0, side='left')
        hi = np.searchsorted(fs, ue / L0, side='right')
        P = (hi - lo) / Nmc
        Pnc = 1 - lo / Nmc            # P(u >= 0.985)
        ok = np.where(P >= 0.1)[0]
        ip = int(np.argmax(P))
        out['cases']['fac%.1f_ue%.3f' % (fac, ue)] = {
            'P_ge_0.1_range': [float(L0[ok[0]]), float(L0[ok[-1]])] if len(ok) else None,
            'peak_P': float(P[ip]), 'peak_L0': float(L0[ip]),
            'P_cond_at_peak': float(P[ip] / Pnc[ip]) if Pnc[ip] > 0 else None,
            'median_circle_bound_L0': float(np.median(0.985 / f))}
base = {k[7:]: v for k, v in out['cases'].items() if k.startswith('fac1.0')}
sens = {}
for ue in ('ue1.011', 'ue1.020', 'ue1.027', 'ue1.030'):
    b0 = out['cases']['fac1.0_' + ue]
    d = {}
    for fac in ('0.9', '1.1'):
        c = out['cases']['fac%s_%s' % (fac, ue)]
        d['fac' + fac] = {'dL_lo': c['P_ge_0.1_range'][0] - b0['P_ge_0.1_range'][0] if c['P_ge_0.1_range'] else None,
                          'dL_hi': c['P_ge_0.1_range'][1] - b0['P_ge_0.1_range'][1] if c['P_ge_0.1_range'] else None,
                          'peak_rel': c['peak_P'] / b0['peak_P'] - 1, 'dpeak_L0': c['peak_L0'] - b0['peak_L0']}
    sens[ue] = d
out['sensitivity'] = sens
with open(os.path.join(RES, 'C145_sigmaE_sensitivity_own.json'), 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=1)
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(out, indent=1))
