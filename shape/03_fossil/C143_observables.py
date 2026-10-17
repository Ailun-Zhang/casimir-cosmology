# PROVENANCE: 04_calculations/code/20261013_claude_v12_r2/observables/C143_observables.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] Observable relations: K(19.32) = 24.5501, dln sigma/dln kappa = 1.366, sigma_E ~ (A_s r)^(1/8) C_eff^(-3/8) K^(1/2), r-measurement precision 3.1%/1.6% (LiteBIRD/CMB-S4). Output path = argv[1]. Paper: PRD Sec. VIII D (eq:s-scaling).

C143 (Claude R2, 2026-10-13): observable relations for the shape fossil and the Higgs criterion.
Inputs: v11 (xi=0.30) benchmark numbers; D109 floor closed form (R14); D108/Stage-3 slopes.
Output JSON: results/20261013_claude_v12_r2/observables/run01/C143_observables.json
"""
import json, sys, hashlib, pathlib
import mpmath as mp

mp.mp.dps = 30
D = 3*mp.sqrt(2)/4

def F(kappa):
    """floor variance in units M_P^2 L_b^3 h = 1: Gamma(7/4)Gamma(3/4)/(2 D |Gamma(5/4+i mu)|^2)."""
    mu = mp.sqrt(4*kappa - 1)/4
    return mp.gamma(mp.mpf(7)/4)*mp.gamma(mp.mpf(3)/4)/(2*D*abs(mp.gamma(mp.mpf(5)/4 + 1j*mu))**2)

out = {}
# 1. closed form check against D109/R14: F(19.32)=24.5501, sigma_floor=sqrt(24.5501/2279.28)=0.1038
kE = mp.mpf('19.32'); MLh = mp.mpf(722)**2*mp.mpf('0.00437243')
out['F_kappaE'] = float(F(kE)); out['M2L3h'] = float(MLh)
out['sigma_floor'] = float(mp.sqrt(F(kE)/MLh))
# free-field kappa_E (xi=0): kappa_A/(1-xi) => kappa_E(0)=19.32*0.70
out['F_kappaE_free'] = float(F(kE*mp.mpf('0.70')))
# d ln sigma / d ln kappa at kappa_E (fixed M2L3h)
h = mp.mpf('1e-6')
out['dlnsigma_dlnkappa'] = float(0.5*(mp.log(F(kE*(1+h))) - mp.log(F(kE*(1-h))))/(2*h))

# 2. scaling with (A_s r) and C_eff at fixed kappa: sigma_E^2 = F/(M_P^2 L_b^3 h), M_P^2 L_b^3 h ∝ (M_P/h)^{1/2} C_eff^{3/4}
#    h/M_P ∝ (A_s r)^{1/2}  =>  sigma_E ∝ (A_s r)^{1/8} C_eff^{-3/8} F^{1/2}
r0 = 0.004
rows = []
for r in [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 1e-2]:
    rows.append({'r': r, 'sigma_ratio': (r/r0)**0.125, 'sigma_E': float(out['sigma_floor'])*(r/r0)**0.125})
out['sigma_vs_r'] = rows
out['frac_sigma_per_frac_r'] = 0.125
out['LiteBIRD_sigma_r_1e-3_at_r_4e-3_frac_sigma'] = 0.125*1e-3/4e-3
out['CMBS4_sigma_r_5e-4_at_r_4e-3_frac_sigma'] = 0.125*5e-4/4e-3

# 3. Higgs criterion vs inflation scale. hazard ∫γ dη ∝ γ/(h L_b); h L_b ∝ (h/M_P)^{1/2} ∝ r^{1/4}; mu_b=1/L_b ∝ r^{1/4}
dlng_dlam = -790.0      # d ln gamma / d lambda near lambda=-0.010 (Stage 3, MU); A-I may update
dlam_dmt = -0.00656     # per GeV (D108/R14)
beta_lam = -2.97e-4     # d lambda / d ln mu at mu_b (2-loop)
# per e-fold of 1/(h L_b) (longer duration): gamma_c -> gamma_c/e  => d lambda_c = +1/|dlng_dlam| (less negative) => lower m_c
dmc_per_efold_duration = -(1.0/abs(dlng_dlam))/abs(dlam_dmt)
dmc_from_duration_per_lnr = -0.25*dmc_per_efold_duration           # duration ∝ r^{-1/4}
dmc_from_mub_per_lnr = 0.25*beta_lam/abs(dlam_dmt)                 # lambda(mu_b) shifts by beta*dln mu_b; more negative -> lower m_c
out['dmc_dlnr_GeV'] = dmc_from_duration_per_lnr + dmc_from_mub_per_lnr
out['dmc_dlnr_parts'] = {'duration': dmc_from_duration_per_lnr, 'mu_b_running': dmc_from_mub_per_lnr}
out['dmc_per_efold_duration_GeV'] = dmc_per_efold_duration
for lo, hi in [(1e-3, 1e-2), (1e-4, 1e-2)]:
    out[f'dmc_range_r_{lo}_{hi}'] = [out['dmc_dlnr_GeV']*float(mp.log(lo/r0)), out['dmc_dlnr_GeV']*float(mp.log(hi/r0))]
# C_eff dependence (e.g. nu_R adds 0.70 to N_C=25.5*...; inflaton +0.5): h L_b ∝ C_eff^{1/4}, mu_b ∝ C_eff^{-1/4}
out['dmc_dlnCeff_GeV'] = -0.25*dmc_per_efold_duration*(-1) * (-1) + (-0.25)*beta_lam/abs(dlam_dmt)
out['notes'] = ('sigma_E ∝ (A_s r)^{1/8} C_eff^{-3/8} F(kappa_E)^{1/2}; m_c shifts by dmc_dlnr per e-fold of r at fixed SM content; '
                'dlng_dlam from Stage-3 MU, to be updated with the A-I main model.')

code = pathlib.Path(__file__).read_bytes()
out['code_sha256'] = hashlib.sha256(code).hexdigest()
dst = pathlib.Path(sys.argv[1])
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({k: out[k] for k in ['F_kappaE', 'sigma_floor', 'F_kappaE_free', 'dlnsigma_dlnkappa', 'dmc_dlnr_GeV',
                                      'dmc_dlnr_parts', 'dmc_per_efold_duration_GeV', 'dmc_range_r_0.001_0.01',
                                      'dmc_range_r_0.0001_0.01', 'dmc_dlnCeff_GeV']}, indent=1))
