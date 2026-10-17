"""PRD version of PRL Fig. 1 (two-column width, panels side by side); same code and data as fig_prl_fig1_threshold.py."""
import sys

sys.dont_write_bytecode = True
import fig_prl_fig1_threshold as F  # noqa: E402

if __name__ == "__main__":
    F.make("prd")
