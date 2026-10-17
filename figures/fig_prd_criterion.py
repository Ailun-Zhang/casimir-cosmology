"""PRD version of PRL Fig. 2 (two-column width, panels side by side); same code and data as fig_prl_fig2_criterion.py.
The (m_t, xi_H) plane of the PRD is prl_fig3_xi (single column, written to both paper folders)."""
import sys

sys.dont_write_bytecode = True
import fig_prl_fig2_criterion as F  # noqa: E402

if __name__ == "__main__":
    F.make("prd")
