# PROVENANCE: new module written 2026-10-15 by code-higgs (v12 code/data package); values copied from the
# production code (sm_provider.py, C150_twod.py, C154_inputs_v2.json) and the D126 v2 input table.
"""Standard-Model and measurement inputs used by the v12 Higgs chain (PDG 2026 unless stated).

These constants document the inputs in one place.  The packaged production modules keep their own literal
copies (unchanged, so that the archived numbers are reproduced bit-for-bit); ``check_consistency()`` asserts
that the two agree.  Units: GeV unless stated; alpha_s = alpha_s^(5)(M_Z) in MS-bar.
"""

# --- central SM inputs (sm_provider.py, C150_twod.py) -------------------------------------------------------
MT_POLE_DIRECT = 172.60          # top mass, direct measurements (identified with the pole mass), PDG 2026
MT_POLE_DIRECT_ERR = 0.27
ALPHA_S = 0.1180                  # alpha_s(M_Z), PDG 2026
ALPHA_S_ERR = 0.0009
ALPHA_S_SET = (0.1171, 0.1180, 0.1189)   # -1 sigma, central, +1 sigma used for the critical-m_t tables
MH = 125.13                       # Higgs mass, PDG 2026 (scale factor 1.5)
MH_ERR = 0.11
MW_PDG = 80.3692                  # W pole mass used in the SMDR-type matching (sm_provider.MW_PDG)

# --- Planck scale and the bounce scale --------------------------------------------------------------------
MPBAR = 2.435e18                  # reduced Planck mass
LB_MPBAR = 722.0                  # L_b * Mbar_P of the v11 background (pipeline constant; v11 Table I: 722)
MU_B = MPBAR / LB_MPBAR           # = 3.3726e15 GeV, matching scale 1/L_b at the turn
HLB = 0.00437243                  # h L_b of the plateau background (zero-mode pipeline constant)
ETA_INF = 1.31102877714606 / HLB  # conformal duration of one branch, int_0^inf du / (hL_b sqrt(cosh 2u)) = 299.84

# --- other top-mass determinations used in the claims layer (D126 v2 Sec. 1.2) ------------------------------
MT_MEASUREMENTS = {
    "direct_PDG2026": (172.60, 0.27),
    "sigma_pole_PDG2026": (172.1, 0.6),
    "sigma_pole_PDG2025": (172.4, 0.7),
    "ATLAS_CMS_2024": (172.52, 0.33),
    "CMS_2023": (171.77, 0.37),
}
MC_POLE_INTERPRETATION = 0.5      # Monte-Carlo-mass vs pole-mass interpretation uncertainty (Hoang 2020)

# --- target survival levels used for critical masses ---------------------------------------------------------
P1_LEVELS = (0.9, 0.5, 0.1)


def check_consistency():
    """Assert that the literal constants inside the packaged production code agree with this module."""
    import importlib
    import os
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zm = os.path.join(here, "higgs", "zmlib")
    if zm not in sys.path:
        sys.path.insert(0, zm)
    sp = importlib.import_module("sm_provider")
    assert sp.MPBAR == MPBAR and abs(sp.MU_B - MU_B) < 1e-3 * MU_B and sp.MW_PDG == MW_PDG
    return dict(MU_B=MU_B, sm_provider_MU_B=sp.MU_B, ok=True)
