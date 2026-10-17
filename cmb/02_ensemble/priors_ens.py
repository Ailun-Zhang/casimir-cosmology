# PROVENANCE: 04_calculations/code/20261013_argon_codex_intake/cmb_ensemble/priors_ens.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Conditional size-prior experiments on the l <= 80 read edges (Godet-type exponential, plateau 1/V, log-size packets; near-window packet 1.7%). Reads S3_analysis.json and the archived C79 table.

priors_ens.py (C145/D117) -- WORKING COPY of the frozen snapshot scripts
  SNAPSHOT_INPUT/code/size_prior.py    sha256 585D0125005058548DF840B72286AAC2852683CE8C49FBACF98BF369DB492542
  SNAPSHOT_INPUT/code/plateau_prior.py sha256 5E318BB37DB1C1E2C7FF095B81A0CB4F6E434107D8487D14744446E254776466
Same sampling (seed 20261013, 2e6 Gaussian E-plane shapes, sigma_E=0.1041), same prior families and formulas.
Change: the readable shortest-side edge u_max(A, theta) is taken from the real ell<=80 E-plane ensemble of
run01 (S3_analysis.json span tables; profiled / fixed volume, TT-only, Planck-like noise variants), with the
original C79 ell30 amplitude-only edges kept side by side for comparison.
These remain CONDITIONAL prior experiments (Godet-type exponential in ln L0, plateau 1/V transplant with
N cutoffs, normal log-size packets incl. sigma_lnL=1.9091), not a derived no-boundary measure and not a
sky likelihood.  Reads run01/results/S3_analysis.json and C79_results.json (read only); writes run01/priors.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ens_core as ec


def main():
    RES = ec.NEW_RUN_OUTPUT / 'results'; OUT = ec.NEW_RUN_OUTPUT / 'priors'
    an = ec.load_json(RES / 'S3_analysis.json')
    src = ec.WP.wfile('fu/C79/C79_results.json')  # package: archived C79 table (staged by run.py)
    c79 = json.loads(src.read_text(encoding='utf-8'))
    rows = sorted([r for r in c79['realizations'] if r['label'].startswith('Xd_q')], key=lambda r: r['amp'])
    amps = np.array([r['amp'] for r in rows]); edges = np.array([r['l_up_TE'] for r in rows])
    n = 2_000_000
    be = np.random.default_rng(20261013).normal(size=(n, 2)) * .1041
    mag = np.hypot(be[:, 0], be[:, 1]); theta = np.arctan2(be[:, 1], be[:, 0])
    f = np.exp(np.min(math.sqrt(2 / 3) * mag[:, None] * np.cos(theta[:, None] - np.deg2rad([30, 150, 270])), axis=1))
    circle = .985; lower = np.log(circle / f)
    A = mag / ec.SIGMA_E; xi = np.sin(3 * theta)
    spans = {'A_fixed_typical': np.full(n, max(math.log(1.0213 / circle), 0.0)),
             'B_C79_clamped': np.maximum(np.log(np.interp(mag, amps, edges) / circle), 0),
             'C_zero_fossil_edge': np.maximum(np.log(np.interp(mag, np.r_[0, amps], np.r_[circle, edges]) / circle), 0)}
    for vn in ('TE80_prof', 'TE80_fixed', 'TE30_prof', 'T80_prof', 'TE80ampfree_prof',
               'noise_planck143_TE80_prof', 'noise_planck143_TE80_fixed'):
        c = an['curves'].get(vn)
        if c:
            spans['ENS_' + vn] = np.maximum(ec.span_eval(c['span_table'], A, xi), 0)

    def ratio_stats(num, den):
        value = float(num.mean() / den.mean()); infl = (num - value * den) / den.mean()
        return {'probability': value, 'mc_standard_error': float(infl.std(ddof=1) / math.sqrt(n))}

    size = {'scope': 'Normalized independent size-and-Gaussian-E-shape benchmark with the real ell<=80 ensemble '
                     'edges (ENS_*) and the original C79/C84 ell30 edges (A/B/C) for comparison. Conditional '
                     'priors; ideal full-sky readability threshold D>=4.5, not a sky likelihood or detection probability.',
            'source_edges': str(RES / 'S3_analysis.json'), 'source_C79': str(src), 'samples': n, 'seed': 20261013,
            'sigma_E': .1041, 'circle': circle, 'Godet_effective_alpha': 2 * (3 / (4 * 68.25 ** 2)),
            'Godet_observational_bound_alpha': .004, 'models': {}}
    for label, span in spans.items():
        def expo(alpha, span=span):
            w = np.exp(alpha * np.log(f)); num = w * (-np.expm1(-alpha * span))
            return ratio_stats(num, w)
        expcases = [{'alpha_per_efold': a, **expo(a), 'upper_bound_from_max_edge': float(-np.expm1(-a * span.max()))}
                    for a in (size['Godet_effective_alpha'], .004, .01, .1, 1, 10, 20, 30, 50, 100)]
        p300 = expo(300)['probability']
        thr = {str(p): (brentq(lambda a: expo(a)['probability'] - p, 1e-4, 300) if p300 >= p else None) for p in (.5, .6, .9)}
        gauss = []
        for sig in (.95, 1.9091, 3.8):
            for lc in (1.0, 1.13, 2.0, 10.0):
                mu = math.log(lc); den = ndtr((mu - lower) / sig)
                num = ndtr((lower + span - mu) / sig) - ndtr((lower - mu) / sig)
                gauss.append({'logsize_sigma': sig, 'geometric_center_L0_over_dLSS': lc,
                              'no_circle_prior_mass': float(den.mean()), **ratio_stats(num, den)})
        size['models'][label] = {'mean_log_readable_span': float(span.mean()), 'max_log_readable_span': float(span.max()),
                                 'exponential': expcases, 'alpha_needed_for_probability_within_0_to_300': thr,
                                 'at_alpha_300': expo(300), 'normal_logsize_prior': gauss}
    ec.save_json(OUT / 'P4_size_prior_ens.json', size)
    lowerN = 68.25 - math.log(1.13) + np.log(circle / f)

    def mass(lo, hi):
        d = hi - lo
        return np.log1p(d / (lo - .75)) + .75 * d / ((lo - .75) * (hi - .75))
    plat = {'prior': 'p(phi)dphi proportional to dphi/V(phi), leading plateau-to-N map (transplant, conditional)',
            'pN_unormalized': 'N/(N-3/4)^2', 'infinite_cutoff_normalizable': False, 'N_reference_at_s1p13': 68.25,
            'samples': n, 'models': {}}
    for label, span in spans.items():
        num = mass(lowerN, lowerN + span); cases = []
        for cut in (100., 250., 1000., 1_000_000.):
            den = mass(lowerN, cut); v = float(num.mean() / den.mean())
            cases.append({'Nmax': cut, 'probability': v,
                          'mc_standard_error': float(((num - v * den) / den.mean()).std(ddof=1) / math.sqrt(n))})
        plat['models'][label] = cases
    ec.save_json(OUT / 'P4_plateau_prior_ens.json', plat)
    summ = {k: dict(mean_span=v['mean_log_readable_span'], P_Godet=v['exponential'][0]['probability'],
                    P_alpha_0p1=v['exponential'][3]['probability'], alpha_for_half=v['alpha_needed_for_probability_within_0_to_300']['0.5'],
                    P_packet_sig1p909_c1p13=v['normal_logsize_prior'][5]['probability'],
                    P_packet_sig0p95_c1p13=v['normal_logsize_prior'][1]['probability'],
                    P_plateau_N1000=plat['models'][k][2]['probability']) for k, v in size['models'].items()}
    ec.save_json(OUT / 'priors_summary.json', summ)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
