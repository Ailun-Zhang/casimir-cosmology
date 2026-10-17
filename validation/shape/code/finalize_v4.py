# finalize_v4.py -- assemble results/BLIND_RESULTS.json (shape-blind, R16 V4), 2026-10-15.
import os, sys, json, time, hashlib, glob

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.dirname(HERE)
RES = os.path.join(BR, 'results')
ROOT = os.path.abspath(os.path.join(BR, '..', '..', '..'))


def sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    lin = json.load(open(os.path.join(RES, 'linear_results.json'), encoding='utf-8'))
    ver = json.load(open(os.path.join(RES, 'verify_results.json'), encoding='utf-8'))
    sv = json.load(open(os.path.join(RES, 'survival_table.json'), encoding='utf-8'))
    it = lin['items_1_3']

    def gpack(rec):
        g = rec['G']
        keys = [k for k in g if isinstance(g[k], dict)]
        return {('t_freeze=%s' % k if abs(float(k) - rec['t_freeze']) < 1e-5 else k): g[k]['G'] for k in keys} | {
            'inf_limit(=2*gain)': g['inf_limit_2gain'], 'tau=6': g['tau=6']}

    item1 = {'kappa_E': it['LO_xi0.30']['kappa_E'], 'xi': 0.30,
             'floor_dimless(units MP^2 Lb^3 h=1)': it['LO_xi0.30']['floor_dimless_closed'],
             'floor_dimless_ode': it['LO_xi0.30']['floor_dimless_ode'],
             'T11=gain': it['LO_xi0.30']['T11_closed_gain'], 'T12_P_norm': it['LO_xi0.30']['T12_closed_Pnorm'],
             'T12_v11_Pi_norm': it['LO_xi0.30']['T12_v11_Pi_norm'],
             'MP2Lb3h_leading(h/MP=6.116e-6)': it['LO_xi0.30']['MP2Lb3h_leading'],
             'sigma_floor2': it['LO_xi0.30']['sigma_floor2'], 'sigma_floor': it['LO_xi0.30']['sigma_floor'],
             'sigma_floor_alt_conventions': it['LO_xi0.30']['sigma_floor_alt'],
             't_freeze_h': it['LO_xi0.30']['t_freeze'], 'G_of_abs_tx': gpack(it['LO_xi0.30']),
             'G_grid_0_to_2': it['LO_xi0.30']['G_grid'],
             'xi0.2999_variant': {'kappa_E': it['LO_xi0.2999']['kappa_E'], 'floor': it['LO_xi0.2999']['floor_dimless_closed'],
                                  'sigma_floor': it['LO_xi0.2999']['sigma_floor'], 'G': gpack(it['LO_xi0.2999'])},
             'free_xi0_variant': {'kappa_E': it['free_xi0']['kappa_E'], 'floor': it['free_xi0']['floor_dimless_closed'],
                                  'sigma_floor': it['free_xi0']['sigma_floor'], 'G': gpack(it['free_xi0'])}}
    item3 = {'kappa_E': 18.0, 'xi': 0.28, 'floor_dimless': it['NLO_xi0.28_kE18.0']['floor_dimless_closed'],
             'MP2Lb3h_leading': it['NLO_xi0.28_kE18.0']['MP2Lb3h_leading'],
             'sigma_floor2': it['NLO_xi0.28_kE18.0']['sigma_floor2'], 'sigma_floor': it['NLO_xi0.28_kE18.0']['sigma_floor'],
             'sigma_floor_ratio_to_LO': it['NLO_xi0.28_kE18.0']['sigma_floor'] / it['LO_xi0.30']['sigma_floor'],
             't_freeze_h': it['NLO_xi0.28_kE18.0']['t_freeze'], 'G_of_abs_tx': gpack(it['NLO_xi0.28_kE18.0'])}
    surv = {}
    for k, e in sv['table'].items():
        surv[k] = {kk: e[kk] for kk in ('N', 'N256', 'all', 'reasons', 'var_ratio_E_nonlin_over_lin', 'rms_betaE_out_survivors',
                                       'rms_betaE_lin_all', 'median_lnLmax_Lmin_survivors', 'max_gate_cons', 'max_gate_det', 'max_gate_tr') if kk in e}
    # G and linear fossil rms at the survival crossings
    sys.path.insert(0, HERE)
    from linear_v4 import Gfun, floor_of
    kE = lin['kappa']['0.3']['kappa_E']
    flE, _ = floor_of(kE)
    at_cross = {}
    for lab, c in sv['curves_fwd_tagA'].items():
        for q in ('t_star_90', 't_star_99'):
            t = c.get(q, {}).get('value')
            if t is not None and t == t:
                g = Gfun(kE, t, flE)['G']
                at_cross['%s_%s' % (lab, q)] = {'t_x': t, 'G': g, 'linear_rms_E_fossil_bound': (g * it['LO_xi0.30']['sigma_floor2']) ** 0.5}
    import math
    for v in at_cross.values():
        v['efolds_before_turn'] = 0.5 * math.log(math.cosh(2 * v['t_x']))
    thr = {}
    tp = os.path.join(RES, 'threshold_sensitivity.json')
    if os.path.exists(tp):
        ts = json.load(open(tp, encoding='utf-8'))
        thr = {'strict_post_hoc': {k: {'t_star_90': v['t_star_90'], 't_star_99': v['t_star_99']} for k, v in ts['strict'].items()},
               'loose_1e-5_rerun': {k: {'n_collapsed_at_1e-3': v['n'], 'recovered': v['recovered']} for k, v in ts['loose_rerun'].items()}}
    efold_freeze = 0.5 * math.log(math.cosh(2 * it['LO_xi0.30']['t_freeze']))
    codes = sorted(glob.glob(os.path.join(HERE, '*.py')) + glob.glob(os.path.join(HERE, '*.cmd')))
    out = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'agent': 'shape-blind (R16 V4)',
        'blind_statement': 'No file under 04_calculations/*/20261014_claude_v12_r3/shape_hardening/ or 00_HANDOFF/ (except COMMON_RULES_R3_CN.md) was opened. Inputs: v11 main_v11.tex, D109 sec.1-2 (lines 12-27). All code written here.',
        'code_sha256': {os.path.basename(c): sha(c) for c in codes},
        'input_sha256': {'main_v11.tex': sha(os.path.join(ROOT, '06_paper_draft', 'versions', 'v11', 'main_v11.tex')),
                         'deviates_seed20261015.npy': sha(os.path.join(RES, 'deviates_seed20261015.npy'))},
        'conventions': {'time': 'tau = h t, leading cosh background L=L_b sqrt(cosh 2ht), V=3 M_P^2 h^2',
                        'momentum': 'p = l^3 dbeta/dtau (P/(M_P^2 L_b^3 h)), [beta,p]=i eps_q, eps_q=1/(M_P^2 L_b^3 h)',
                        'floor': '|M11 M12| of M(0->inf) = |T12|/2 (eps_q=1); sigma_floor^2 = eps_q*floor',
                        'G': 'min over states real at t_x of max(Var beta_-,Var beta_+)/floor (the larger branch is the future one for t_x<0)',
                        'optimal_width_state': 'per mode argmin of the larger-branch variance; E: kappa_E, T2: kappa_T2; pure Gaussian, Wigner-sampled',
                        'survival': 'forward from t_x to ht=12 with lambda_min(G)>1e-3 throughout and H(12)>0.9h; bwd: to ht=-12, H<-0.9h',
                        'nonlinear_model': 'xi=0.30, kappa_E=%.5f, kappa_T2=%.5f, eps_q=%.6e (MP^2Lb^3h=%.2f), exact canonical momentum map, l from constraint at u=tanh(2tau_x), DOP853 rtol 1e-10' % (
                            lin['kappa']['0.3']['kappa_E'], lin['kappa']['0.3']['kappa_T2'], lin['kappa']['0.3']['eps_q'], lin['kappa']['0.3']['MP2Lb3h'])},
        'item1_real_state_moment_theorem': item1,
        'item2_nonlinear_survival': {'table': surv, 'curves_fwd': sv['curves_fwd_tagA'], 'two_sided_N256': sv['two_sided'],
                                     'robustness': sv['robustness'], 'G_at_crossings': at_cross,
                                     'collapse_threshold_sensitivity': thr, 'efolds_before_turn_at_t_freeze': efold_freeze},
        'item3_next_order': item3,
        'item4_emptiness_gate': {'budgets': lin['item4_gate'], 'preparation_window_ln(2pi/hLb)': lin['item4_window']},
        'checks': {'ewald': {k: lin['ewald'][k] for k in ('Z2_cube_periodic', 'Z2_cube_AAA', 'direct_alternating_N60', 'Zeff_cube', 'NC',
                                                           'lambda', 'trace_hess_over_Z', 'trace_hess_over_Z_generic', 'gradient_max_rel_err')},
                   'G_second_formulation': ver['G_second_formulation'],
                   'nonlinear_small_amplitude_scan': ver['nonlinear_small_amplitude_scan'],
                   'wolfram': 'results/wolfram_receipts.json'},
        'comparison_with_production': None,
    }
    with open(os.path.join(RES, 'BLIND_RESULTS.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('written', os.path.join(RES, 'BLIND_RESULTS.json'))
    print(json.dumps(out['code_sha256'], indent=0))


if __name__ == '__main__':
    main()
