# PROVENANCE: 04_calculations/code/20261013_argon_codex_intake/joint_state/joint_state.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] C147/D119: constraint-projected real Gaussian (Wigner) inflaton zero-mode states at the time-symmetric turning point, classical nonlinear propagation, WKB of the Casimir forbidden region: the mean phi_b is not selected (window needs Delta phi_b = 6.2e-4), sigma_N >= 0.935, p(N) ~ 1/N, readability <~ 1.5%. Inputs: four small files of the frozen Codex snapshot (inputs/snapshot/) and the archived C79 table. Paper: PRD Sec. X (joint state), a20_conditional.tex (conditional evidence).

C147/D119 joint-state (DNEC, 2026-10-13): size distribution from the inflaton zero mode
at the Casimir turning point.

Working copy of Codex's constrained Gaussian-Wigner inflaton packet (snapshot
code/inflaton_packet.py: trajectory() verbatim; packet() takes an explicit sigma_phi), extended by
  T1  reproduction of sigma_N, sigma_lnL0 for Codex's prepared width rule
  T2  time-symmetric bounce family phi_b -> (L_b, N_post, ln L0); A_s fixes h
  T3  B3 width (time-symmetric Gaussian with minimum instantaneous energy variance) for the
      inflaton zero mode INCLUDING the plateau slope; size floor Var N >= |dN/dphi dN/dP| for
      real (Theta-invariant) states
  T4  measures (Liouville / branch-KG / Schroedinger slices / Godet 1/V) and the inner-boundary
      WKB weight exp(+-2 S_E) of the T^3 Casimir forbidden region; cutoff dependence
  T5  conditional size kernel + prior families for C145 (P(read|L0) integration)
NOT a solution of the joint WDW equation. Snapshot files are only read. Pool <= 2 workers.
"""
from pathlib import Path
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import json, time, math, hashlib
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr, gamma as Gam
from numpy.polynomial.hermite import hermgauss
from concurrent.futures import ProcessPoolExecutor

import sys  # package: work-tree path shim of the cmb chain (see cmb/_cmb_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cmb_paths as WP  # noqa: E402
SNAP = WP.step_dir('03_joint_state', 'inputs', 'snapshot')  # package: the four snapshot files used here
OUT = WP.wdir('ai/joint_state/run01')
C79 = WP.wfile('fu/C79/C79_results.json')
H = 6.1159966044e-6      # h/M_P, A_s-calibrated (v11 Sec. IV D)   [input]
D = 29.92                # N_C^eff*gamma (xi=0.30): rho_C=-D/L^4, M_P=1 [input]
A = math.sqrt(2/3)
# y = ln(L0/d_LSS) = N_post - Y_ANCHOR ; v11: N_post=68.25(+0.09 for xi=.30) at L0=0.95 d_LSS [input]
Y_ANCHOR = 68.25 + 0.09 - math.log(0.95)
C_SE = math.sqrt(12)*Gam(0.25)**2/(6*math.sqrt(2*math.pi))   # S_E = C_SE D^{3/4} V^{-1/4}


def U(phi): return 3*(1-np.exp(-A*phi))**2
def Up(phi):
    e = np.exp(-A*phi); return 6*A*e*(1-e)
def Upp(phi):
    e = np.exp(-A*phi); return 4*e*(2*e-1)
def dU(phi0, x):
    e0 = np.exp(-A*phi0); e1 = np.exp(-A*(phi0+x)); return 3*(e0-e1)*(2-e0-e1)
def Lb_of(phib): return float((D/(H*H*U(phib)))**.25)
def S_E(phib): return C_SE*D**.75*(H*H*U(phib))**-.25


# ---------------- verbatim from snapshot code/inflaton_packet.py ----------------
def trajectory(phi, momentum, refphi, rtol=2e-10):
    ur=U(refphi)
    lr=(D/(H*H*ur))**.25
    # Turning constraint: U(phi) w^3 - U(refphi) w + P^2/(2 h^2 Lref^6)=0,
    # w=(L/Lref)^2. The largest positive root is the expanding bounce.
    pp=momentum/(H*lr**3)
    roots=np.roots([U(phi),0,-ur,.5*pp**2])
    positive=[r.real for r in roots if abs(r.imag)<1e-9 and r.real>0]
    if not positive: raise ValueError('No positive turning volume')
    w=max(positive); n0=.5*math.log(w); v0=pp/w**1.5
    if v0*v0>=2*U(phi): raise ValueError('Turning point is not a bounce')
    def rhs(t,y):
        n,hhat,p,v=y
        c=ur*np.exp(-4*n)
        return [hhat, -.5*v*v+2*c/3, v, -3*hhat*v-Up(p)]
    def end(t,y):
        n,hh,p,v=y
        return .5*v*v-2*ur*np.exp(-4*n)/3-hh*hh
    end.terminal=True; end.direction=1
    sol=solve_ivp(rhs,[0,500],[n0,0,phi,v0],method='DOP853',rtol=rtol,
                  atol=rtol/100,events=end,dense_output=True,max_step=1.0)
    if not sol.success or not sol.t_events[0].size:
        raise RuntimeError('Inflation end absent: '+sol.message)
    tend=float(sol.t_events[0][0]); final=sol.sol(tend)
    y=sol.sol(np.linspace(0,tend,400))
    constraint=3*y[1]**2-.5*y[3]**2-U(y[2])+ur*np.exp(-4*y[0])
    rhoend=3*final[1]**2
    logsize=float(final[0]+.25*np.log(rhoend))
    return {'Npost':float(final[0]-n0),'logsize_offset':logsize,
            'log_turn_volume_ratio':n0,'initial_v':float(v0),'rhoend_hat':float(rhoend),
            'constraint_max_abs':float(np.max(np.abs(constraint))),
            'initial_constraint_abs':float(abs(.5*v0*v0+U(phi)-ur/w**2)),
            't_end_in_h_units':tend,'evaluations':sol.nfev}
# -------------------------------------------------------------------------------


def packet(refphi, sigma, order=9, rtol=2e-10, label='', keep_nodes=False):
    """Codex packet with explicit sigma_phi (sigma_P = 1/(2 sigma_phi), real Gaussian)."""
    lr = Lb_of(refphi); sigmap = 1/(2*sigma)
    pts, wts = hermgauss(order); wts = wts/math.sqrt(math.pi)
    center = trajectory(refphi, 0, refphi, rtol)
    rows = []; wgt = []
    for i, x in enumerate(pts):
        for j, z in enumerate(pts):
            ph = refphi+math.sqrt(2)*sigma*x; p = math.sqrt(2)*sigmap*z
            r = trajectory(ph, p, refphi, rtol)
            rows.append([ph, p, r['Npost'], r['logsize_offset']-center['logsize_offset'],
                         r['log_turn_volume_ratio'], r['constraint_max_abs']])
            wgt.append(wts[i]*wts[j])
    data = np.array(rows); ws = np.array(wgt)
    avg = ws@data; dev = data-avg; cov = dev.T@(ws[:, None]*dev)
    sN = math.sqrt(cov[2, 2]); sL = math.sqrt(cov[3, 3])
    out = {'label': label, 'refphi': refphi, 'order': order, 'rtol': rtol, 'Lb': lr,
           'sigma_phi': sigma, 'sigma_P': sigmap, 'N_center': center['Npost'],
           'mean_N': float(avg[2]), 'mean_dlnL0': float(avg[3]),
           'sigma_N': sN, 'sigma_lnL0': sL,
           'skew_N': float(ws@dev[:, 2]**3)/sN**3, 'exkurt_N': float(ws@dev[:, 2]**4)/sN**4-3,
           'skew_lnL0': float(ws@dev[:, 3]**3)/sL**3, 'corr_N_lnL0': float(cov[2, 3]/(sN*sL)),
           'mean_log_turn_volume_ratio': float(avg[4]), 'sd_log_turn_volume_ratio': math.sqrt(cov[4, 4]),
           'constraint_max_abs': float(data[:, 5].max())}
    if keep_nodes:
        out['nodes_dN_dlnL0_w'] = [[float(a-center['Npost']), float(b), float(c)]
                                   for a, b, c in zip(data[:, 2], data[:, 3], ws)]
    return out


def _job(args):
    t0 = time.perf_counter(); r = packet(*args); r['seconds'] = time.perf_counter()-t0; return r


def response(phib, rtol=2e-10, step=1e-4, dp=1.0):
    """Linear response of N_post and ln L0 to (phi_b, P_b) at a time-symmetric bounce."""
    c = trajectory(phib, 0, phib, rtol)
    a = trajectory(phib+step, 0, phib, rtol); b = trajectory(phib-step, 0, phib, rtol)
    pp = trajectory(phib, dp, phib, rtol); pm = trajectory(phib, -dp, phib, rtol)
    r = {'N': c['Npost'],
         'dN_dphi': (a['Npost']-b['Npost'])/(2*step), 'dN_dP': (pp['Npost']-pm['Npost'])/(2*dp),
         'dlnL0_dphi': (a['logsize_offset']-b['logsize_offset'])/(2*step),
         'dlnL0_dP': (pp['logsize_offset']-pm['logsize_offset'])/(2*dp),
         'd2N_dP2': (pp['Npost']-2*c['Npost']+pm['Npost'])/dp**2,
         'd2lnL0_dP2': (pp['logsize_offset']-2*c['logsize_offset']+pm['logsize_offset'])/dp**2}
    r['floor_sigma_N'] = math.sqrt(abs(r['dN_dphi']*r['dN_dP']))
    r['floor_sigma_lnL0'] = math.sqrt(abs(r['dlnL0_dphi']*r['dlnL0_dP']))
    r['sigma_phi_opt_N'] = math.sqrt(abs(r['dN_dP']/(2*r['dN_dphi'])))
    return r


def varH(sigma, phib, order=100, quadratic_only=False):
    """Var of H_phi = P^2/2m + m V(phi), m=L_b^3, in the real centred Gaussian of width sigma."""
    m = Lb_of(phib)**3
    t, w = hermgauss(order); w = w/math.sqrt(math.pi); x = math.sqrt(2)*sigma*t
    if quadratic_only:      # slope-free (D113 B3 / Codex) limit: V = V_b + V''x^2/2
        dV = .5*H*H*Upp(phib)*x*x; Vpp = H*H*Upp(phib)*np.ones_like(x)
    else:
        dV = H*H*dU(phib, x); Vpp = H*H*Upp(phib+x)
    mV = float(w@dV); varV = float(w@(dV-mV)**2)
    cov = float(w@(dV*x*x))/(4*sigma**4)-.5*float(w@Vpp)-mV/(4*sigma**2)   # Cov_sym(P^2, V)
    return 1/(32*m*m*sigma**4)+m*m*varV+cov


def b3(phib):
    m = Lb_of(phib)**3
    s_codex = 1/math.sqrt(2*m*math.sqrt(abs(H*H*Upp(phib))))
    s_airy = (4*m*m*H*H*Up(phib))**(-1/3)
    f = lambda s: math.log(varH(math.exp(s), phib))
    s_b3 = math.exp(minimize_scalar(f, bounds=(math.log(2e-4), math.log(.3)), method='bounded',
                                    options={'xatol': 1e-10}).x)
    g = lambda s: math.log(varH(math.exp(s), phib, quadratic_only=True))
    s_q = math.exp(minimize_scalar(g, bounds=(math.log(2e-4), math.log(.3)), method='bounded',
                                   options={'xatol': 1e-10}).x)
    F = m*H*H*Up(phib); k = m*abs(H*H*Upp(phib))
    return {'phib': phib, 'm_Lb3': m, 'sigma_B3': s_b3, 'sigma_airy_analytic': s_airy,
            'sigma_codex_rule': s_codex, 'sigma_B3_slopefree_numeric': s_q,
            'VarH_at_B3': varH(s_b3, phib), 'VarH_at_codex': varH(s_codex, phib),
            'VarH_slopefree_min': varH(s_q, phib, quadratic_only=True), 'k_over_2m': k/(2*m),
            'slope_dominance_2mg2_over_k32': 2*m*(H*H*Up(phib))**2/abs(H*H*Upp(phib))**1.5,
            'a_star_over_a_IO=(sC/sB3)^2': (s_codex/s_b3)**2,
            'b0_is_min(a*>sqrt(km)/2)': bool((s_codex/s_b3)**2 > 1),
            'E_C=D/Lb': D/Lb_of(phib), 'F=mV_prime': F, 'k=m|V2|': k}


def N_sr(phi):
    Ae = math.log(1+2/math.sqrt(3))     # epsilon_V = 1
    return .75*(np.exp(A*phi)-A*phi)-.75*(math.exp(Ae)-Ae)


def main():
    t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    res = {'scope': 'C147/D119. Constraint-projected real Gaussian (Wigner) inflaton zero-mode states at a '
                    'time-symmetric Casimir turning point, classical nonlinear propagation; WKB analysis of the '
                    'T^3 Casimir forbidden region. Not the joint WDW wavefunction or its physical inner product.',
           'inputs': {'h': H, 'D=N_C^eff*gamma': D, 'Y_ANCHOR': Y_ANCHOR, 'C_SE': C_SE}}
    snapfiles = ['code/inflaton_packet.py', 'results/P2_inflaton/summary.json',
                 'results/P4_size_prior.json', 'results/P4_plateau_prior.json']
    res['snapshot_sha256_readonly'] = {f: hashlib.sha256((SNAP/f).read_bytes()).hexdigest() for f in snapfiles}
    res['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    codex = json.loads((SNAP/'results/P2_inflaton/summary.json').read_text(encoding='utf-8'))

    # ---------------- T2: time-symmetric family, A_s, window in N ----------------
    ref = brentq(lambda p: trajectory(p, 0, p)['Npost']-68.25, 5.3, 5.9, xtol=1e-11)
    N_lo = Y_ANCHOR+math.log(1.093); N_hi = Y_ANCHOR+math.log(1.133); N_win = .5*(N_lo+N_hi)
    phiwin = brentq(lambda p: trajectory(p, 0, p)['Npost']-N_win, 5.4, 5.8, xtol=1e-11)
    fam = []
    for pb in np.r_[np.linspace(4.9, 6.3, 15), ref, phiwin]:
        c = trajectory(pb, 0, pb); a = trajectory(pb+1e-4, 0, pb); b = trajectory(pb-1e-4, 0, pb)
        Lb = Lb_of(pb); bb = b3(pb)
        fam.append({'phib': float(pb), 'Lb_MP': Lb, 'hLb': H*Lb, 'U_b(v_b)': float(U(pb)),
                    'N_post': c['Npost'], 'lnL0_rel': math.log(Lb)+c['logsize_offset'],
                    'dN_dphi': (a['Npost']-b['Npost'])/2e-4, 'S_E': S_E(pb),
                    'dS_E_dphi': -.25*S_E(pb)*float(Up(pb)/U(pb)),
                    'sigma_B3': bb['sigma_B3'], 'VarH_B3': bb['VarH_at_B3'],
                    'VarH_B3_over_EC2': bb['VarH_at_B3']/bb['E_C=D/Lb']**2,
                    'constraint_max_abs': c['constraint_max_abs']})
    lref = [r for r in fam if abs(r['phib']-ref) < 1e-12][0]['lnL0_rel']
    for r in fam:
        r['dlnL0_vs_ref'] = r['lnL0_rel']-lref
        r['alpha_HH=-2dS_E/dN'] = -2*r['dS_E_dphi']/r['dN_dphi']
        r['alpha_HH_analytic=.75*S_inf/N^2'] = .75*C_SE*D**.75*(3*H*H)**-.25/r['N_post']**2
    # A_s from slow roll at N_* = 55.722 (shows h is fixed by A_s, independent of phi_b)
    phis = brentq(lambda p: N_sr(p)-55.722, 3, 8)
    eps = .5*(Up(phis)/U(phis))**2
    As = H*H*U(phis)/(24*math.pi**2*eps)
    # S_E check by quadrature at ref
    Vb = H*H*U(ref); Lb = Lb_of(ref)
    SE_quad = quad(lambda L: math.sqrt(max(12*(D-Vb*L**4), 0)), 0, Lb, limit=200)[0]
    res['T2'] = {'phib_ref(N=68.25)': ref, 'Lb_ref': Lb, 'phib_window': phiwin,
                 'N_window_typical': [N_lo, N_hi], 'window_width_efold': N_hi-N_lo,
                 'family': fam, 'A_s_slowroll_at_Nstar55.722': As, 'phi_star': phis,
                 'S_E_ref_closed': S_E(ref), 'S_E_ref_quad': SE_quad,
                 'volume_Airy_length_MP': (48*D/Lb)**(-1/3),
                 'volume_Airy_dlnLb': (48*D/Lb)**(-1/3)/Lb,
                 'S3_HH_exponent_24pi2_over_V_for_comparison': 24*math.pi**2/Vb}
    print('T2 done', round(time.perf_counter()-t0, 1), 's; ref', ref, 'phiwin', phiwin, flush=True)

    # ---------------- T3: B3 widths, response, floor ----------------
    B3 = {k: b3(p) for k, p in [('ref', ref), ('window', phiwin), ('smallN', ref-.20), ('largeN', ref+.32)]}
    R = {k: response(p) for k, p in [('ref', ref), ('window', phiwin), ('smallN', ref-.20), ('largeN', ref+.32)]}
    sC = B3['ref']['sigma_codex_rule']
    jobs = [(ref, sC, 9, 2e-10, 'T1_codex_ref_o9', False),
            (ref, B3['ref']['sigma_B3'], 9, 2e-10, 'T3_B3_ref_o9', False),
            (ref, B3['ref']['sigma_B3'], 13, 2e-10, 'T3_B3_ref_o13', False),
            (ref, R['ref']['sigma_phi_opt_N'], 9, 2e-10, 'T3_flooropt_ref_o9', False),
            (phiwin, B3['window']['sigma_B3'], 9, 2e-10, 'T5_B3_window_o9', True),
            (phiwin, R['window']['sigma_phi_opt_N'], 9, 2e-10, 'T5_flooropt_window_o9', True),
            (phiwin, B3['window']['sigma_codex_rule'], 9, 2e-10, 'T5_codex_window_o9', True),
            (ref-.20, B3['smallN']['sigma_B3'], 9, 2e-10, 'T3_B3_smallN_o9', False),
            (ref+.32, B3['largeN']['sigma_B3'], 9, 2e-10, 'T3_B3_largeN_o9', False)]
    for wf in (.2, .25, .3, .35, .45, .6):
        jobs.append((ref, wf*sC, 9, 2e-10, 'scan_wf%.2f' % wf, False))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("DNEC_WORKERS", "2"))) as ex:  # package: --workers
        P = {r['label']: r for r in ex.map(_job, jobs)}
    print('packets done', round(time.perf_counter()-t0, 1), 's', flush=True)
    c9 = codex['reference_9']
    res['T1'] = {'mine': {k: P['T1_codex_ref_o9'][k] for k in ('refphi', 'sigma_phi', 'N_center', 'mean_N',
                                                                'sigma_N', 'sigma_lnL0', 'constraint_max_abs')},
                 'codex_snapshot': {'refphi': c9['refphi'], 'sigma_phi': c9['sigma_phi'],
                                    'N_center': c9['center']['Npost'], 'mean_N': c9['means'][2],
                                    'sigma_N': c9['sigma_N'], 'sigma_lnL0': c9['sigma_logL0']},
                 'abs_diff_sigma_N': abs(P['T1_codex_ref_o9']['sigma_N']-c9['sigma_N']),
                 'abs_diff_sigma_lnL0': abs(P['T1_codex_ref_o9']['sigma_lnL0']-c9['sigma_logL0'])}
    res['T3'] = {'B3': B3, 'response': R,
                 'packets': {k: v for k, v in P.items() if not k.startswith('scan')},
                 'width_scan_ref': [{'widthfactor_vs_codex': v['sigma_phi']/sC, 'sigma_phi': v['sigma_phi'],
                                     'sigma_N': v['sigma_N'], 'sigma_lnL0': v['sigma_lnL0']}
                                    for k, v in sorted(P.items()) if k.startswith('scan')]}

    # ---------------- T4: measures, inner-boundary weights, cutoffs ----------------
    phitab = np.linspace(5.0, 22.0, 400001)
    off = R['ref']['N']-float(N_sr(ref))
    Ntab = N_sr(phitab)+off
    phi_of_N = lambda N: np.interp(N, Ntab, phitab)
    dNdphi_lo = [r for r in fam if abs(r['phib']-phiwin) < 1e-12][0]['dN_dphi']
    phi_lo = phiwin-(N_win-N_lo)/dNdphi_lo
    Delta = N_hi-N_lo
    weights = {'liouville_flat_dphi(=branch KG, |A|=1)': lambda p: 0.0,
               'schrodinger_turning_slice(V^-1/12)': lambda p: -math.log(U(p))/12,
               'schrodinger_late_slice(V^-1/2)': lambda p: -math.log(U(p))/2,
               'godet_leading_1_over_V(transplant)': lambda p: -math.log(U(p)),
               'inner_HH_like exp(+2S_E)': lambda p: 2*(S_E(p)-S_E(phi_lo)),
               'inner_tunneling_like exp(-2S_E)': lambda p: -2*(S_E(p)-S_E(phi_lo))}
    def logint(g, a, b):
        xs = np.linspace(a, b, 20001); gs = np.array([g(x) for x in xs]) if b-a < 1 else None
        if gs is None:
            xs = np.r_[np.linspace(a, a+1, 20001), np.linspace(a+1, b, 20001)[1:]]
            gs = np.array([g(x) for x in xs])
        trap = getattr(np, 'trapezoid', None) or np.trapz
        gm = gs.max(); return gm+math.log(trap(np.exp(gs-gm), xs))
    meas = {}
    for name, g in weights.items():
        rows = []
        for Nmax in (100., 250., 1000., 1e6):
            pw = phi_of_N(Nmax)
            lw = logint(g, phi_lo, phi_lo+Delta/dNdphi_lo); lt = logint(g, phi_lo, pw)
            rows.append({'Nmax': Nmax, 'phi_max': float(pw), 'log10_P_read_given_no_circles': (lw-lt)/math.log(10),
                         'P_read_given_no_circles': math.exp(lw-lt)})
        meas[name] = rows
    aHH = [r for r in fam if abs(r['phib']-phiwin) < 1e-12][0]['alpha_HH=-2dS_E/dN']
    alpha_eff = aHH+1/N_win          # include the 1/(dN/dphi) ~ 1/N Liouville Jacobian
    res['T4'] = {'typical_fossil_window': {'N_lo': N_lo, 'Delta': Delta, 'phi_lo': phi_lo,
                                           'dphi_window': Delta/dNdphi_lo},
                 'measures_x_cutoffs': meas, 'alpha_HH_at_window': aHH, 'alpha_eff_with_jacobian': alpha_eff,
                 'D_needed_for_alpha20': D*(20/alpha_eff)**(4/3)}

    # shape-marginalised (Codex/C79 edge framework, re-implemented; C79 read-only)
    try:
        old = json.loads(C79.read_text(encoding='utf-8'))
        rows = sorted([r for r in old['realizations'] if r['label'].startswith('Xd_q')], key=lambda r: r['amp'])
        amps = np.array([r['amp'] for r in rows]); edges = np.array([r['l_up_TE'] for r in rows])
        rng = np.random.default_rng(20261013); n = 400_000
        be = rng.normal(size=(n, 2))*.1041
        mag = np.hypot(be[:, 0], be[:, 1]); th = np.arctan2(be[:, 1], be[:, 0])
        f = np.exp(np.min(math.sqrt(2/3)*mag[:, None]*np.cos(th[:, None]-np.deg2rad([30, 150, 270])), axis=1))
        lower = np.log(.985/f)
        models = {'A_fixed_typical': np.full(n, 1.0213), 'B_C79_clamped': np.interp(mag, amps, edges),
                  'C_zero_fossil_edge': np.interp(mag, np.r_[0, amps], np.r_[.985, edges])}
        sm = {}
        for mname, edge in models.items():
            span = np.maximum(np.log(edge/.985), 0)
            d = {'mean_span': float(span.mean())}
            for a_ in (alpha_eff*.9, alpha_eff, alpha_eff*1.1):
                w = f**a_; d['exp_alpha=%.3f' % a_] = float((w*(-np.expm1(-a_*span))).mean()/w.mean())
            for kname, lab in (('B3', 'T5_B3_window_o9'), ('floor', 'T5_flooropt_window_o9'),
                               ('codex', 'T5_codex_window_o9')):
                sg = P[lab]['sigma_lnL0']; best = None; tab = []
                for mu in np.linspace(-1.0, 1.0, 81):
                    num = (ndtr((lower+span-mu)/sg)-ndtr((lower-mu)/sg)).mean()
                    den = ndtr((mu-lower)/sg).mean()
                    tab.append((float(mu), float(num), float(den)))
                    if best is None or num > best[1]: best = (float(mu), float(num), float(den))
                d['gauss_%s_sigma' % kname] = sg
                d['gauss_%s_max_P_read_joint' % kname] = {'mu_y': best[0], 'P_read': best[1], 'P_no_circles': best[2],
                                                          'P_read_given_no_circles': best[1]/best[2]}
                d['gauss_%s_at_mu' % kname] = {('%.3f' % m_): {'P_read': a_, 'P_no_circles': b_,
                                                               'P_read_given_no_circles': a_/b_}
                                               for m_, a_, b_ in tab if any(abs(m_-q) < 1e-9 for q in (0.0, 0.1, 0.125, 0.25, 0.5))}
            # Liouville flat with cutoff, per-shape lower edge (exact phi(N) table)
            Nl = Y_ANCHOR+lower; Nu = Nl+span
            for Nmax in (100., 250., 1e6):
                d['flat_dphi_Nmax=%g' % Nmax] = float((phi_of_N(Nu)-phi_of_N(Nl)).mean()/(phi_of_N(Nmax)-phi_of_N(Nl)).mean())
            sm[mname] = d
        res['T4']['shape_marginalised'] = sm
    except Exception as exc:
        res['T4']['shape_marginalised'] = {'error': repr(exc)}

    # ---------------- T5: interface for C145 ----------------
    def kern(lab):
        p = P[lab]; yc = p['N_center']-Y_ANCHOR
        return {'label': lab, 'status': 'state-output given (phi_b, width rule)', 'phi_b': p['refphi'],
                'y_center': yc, 'sigma_phi': p['sigma_phi'], 'sigma_P': p['sigma_P'],
                'mean_y_shift': p['mean_dlnL0'], 'sigma_y': p['sigma_lnL0'], 'skew_y': p['skew_lnL0'],
                'nodes_[dN,dy,w]': p['nodes_dN_dlnL0_w']}
    iface = {'y': 'ln(L0/d_LSS)', 'anchor': 'y = N_post - %.4f (xi=0.30, instant reheating; v11 Sec. IV D) [input]' % Y_ANCHOR,
             'kernel_B3 (recommended state-output)': kern('T5_B3_window_o9'),
             'kernel_floor_optimal': kern('T5_flooropt_window_o9'),
             'kernel_codex_rule': kern('T5_codex_window_o9'),
             'center_prior_options (state-input)': {
                 'prepared_delta': 'p(mu)=delta(mu-mu0); mu0 not selected by any principle found here',
                 'liouville_flat_dphi': 'p(N) dN = dphi_b (|A|=1); ~1/(sqrt(2/3) N); needs N_max; table in T4',
                 'inner_HH_like': 'p(y) ~ exp(-alpha y), alpha=%.4f per e-fold at the window (T^3 Casimir WKB, phi-independent inner datum, decaying branch)' % alpha_eff,
                 'inner_tunneling_like': 'p(y) ~ exp(+alpha y) up to N_max; P(read|no circles) ~ 0'},
             'formula': 'P(read|no circles) = Int dmu p(mu) Int dy K(y-mu) E_beta[1(y_c<y<y_u)] / Int dmu p(mu) Int dy K(y-mu) E_beta[1(y>y_c)], y_c=ln(.985/f(beta))',
             'note': 'For exponential or ~1/N center priors the kernel width drops out of P(read|no circles); it matters only for a prepared (delta) center.'}
    res['T5_C145_interface'] = iface
    (OUT/'C145_interface.json').write_text(json.dumps(iface, indent=1), encoding='utf-8')
    res['elapsed_s'] = time.perf_counter()-t0
    (OUT/'joint_state_results.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')

    # ---------------- console summary ----------------
    S = lambda k: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in P[k].items()
                   if kk in ('sigma_phi', 'sigma_P', 'N_center', 'mean_N', 'sigma_N', 'sigma_lnL0', 'skew_N', 'constraint_max_abs')}
    print(json.dumps({'T1': res['T1'], 'B3_ref': B3['ref'], 'B3_window': B3['window'],
                      'resp_ref': R['ref'], 'resp_window': R['window'],
                      'pk': {k: S(k) for k in P}, 'T2_misc': {k: v for k, v in res['T2'].items() if k != 'family'},
                      'family': [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                                  if k in ('phib', 'N_post', 'Lb_MP', 'dlnL0_vs_ref', 'dN_dphi', 'sigma_B3', 'VarH_B3_over_EC2',
                                           'S_E', 'alpha_HH=-2dS_E/dN')} for r in fam],
                      'T4': res['T4'], 'elapsed_s': res['elapsed_s']}, indent=1, default=float))


if __name__ == '__main__':
    main()
