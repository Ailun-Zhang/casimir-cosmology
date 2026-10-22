"""Shared figure style for the v12 papers (PRD long paper and PRL letter).

Every figure script in this folder imports this module, so that both papers use one palette, one font and one set of
sizes. The palette is carried over from the v0.7-v1.1 figures (06_paper_draft/figures/prd_style.py). The colour-vision
checker that was ported there (palette report only, not used to build the figures; upstream licence not
determined) was removed from the release (R18).

Sizes (APS/REVTeX): single column 3.375 in, double column 7.0 in. Text: STIX serif and STIX math, matching the
Times-like REVTeX body; ticks 8 pt, axis labels 8.5 pt, legends 7.5 pt, panel labels 8.5 pt bold, nothing below
FS_MIN = 7 pt (checked by `check_fontsizes`). Output: vector PDF (fonts embedded as TrueType) plus a 300-dpi PNG preview.

Colour roles (fixed across figures; variants of one entity are carried by line style or weight, never by a new hue):
  shape figures   aspect (E) modes ORANGE, angle (T2) modes BLUE, third series AQUA;
  Higgs figures   Standard-Model running / measured inputs INK and GREY_FILL bands; the finite-volume threshold and
                  the survival region AQUA; ordered families (alpha_s values, survival levels P1) the blue RAMP;
                  signed contributions (stabilizing vs destabilizing) DIV_NEG (blue) and DIV_POS (red);
                  "excluded / unstable" HATCH_RED, used sparingly.
Data: figure scripts read only files under ./data (copied from the archived production results; ./data/SOURCES.json
maps each file to its original path and SHA-256), so the folder is self-contained.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"

# Paper folders exist only inside the research tree (ROOT/06_paper_draft/versions/v12/...). In the released package
# the figures go to ./out, or to the folders listed in the environment variable DNEC_FIG_OUT (separated by ';').
_ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE
_PAPERS = _ROOT / "06_paper_draft" / "versions" / "v12"
PAPER_DIRS = {"prd": _PAPERS / "PRD" / "figures", "prl": _PAPERS / "PRL" / "figures"}

SINGLE = 3.375
DOUBLE = 7.0

# Validated categorical colours (all pairs pass the protan/deutan separation test on white).
ORANGE = "#eb6834"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue, light -> dark
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
HAIR = "#c3c2b7"
SURFACE = "#ffffff"
GREY_FILL = "#e1e0d9"
GREY_FILL2 = "#f0efec"
AQUA_FILL = "#c9eedf"
HATCH_RED = "#e34948"

FS_MIN = 7.0


def use() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    })


def panel_label(ax, text: str, x: float = 0.02, y: float = 0.97) -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", color=INK)


def load_json(rel: str):
    return json.loads((DATA / rel).read_text(encoding="utf-8"))


def load_csv(rel: str) -> list[dict]:
    with open(DATA / rel, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_fontsizes(fig, fs_min: float = FS_MIN) -> list[str]:
    """Return the visible texts set below fs_min (tick exponents rendered by mathtext are reported as well)."""
    small = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < fs_min - 1e-6:
            small.append(f"{t.get_text()[:30]!r} {t.get_fontsize():.1f} pt")
    return small


def _out_dirs(papers) -> list[Path]:
    dirs = [OUT]
    env = os.environ.get("DNEC_FIG_OUT")
    if env:
        dirs += [Path(p) for p in env.split(";") if p.strip()]
    for key in papers:
        d = PAPER_DIRS[key]
        if d.parent.exists():
            dirs.append(d)
    return dirs


def save(fig, name: str, width: float | None = None, papers=("prd",)) -> list[Path]:
    """Write name.pdf (vector) and name.png (300 dpi preview) to ./out and to the paper folders in `papers`.

    If `width` (inches) is given, the figure width is iterated so that the tight bounding box equals the column
    width while the lettering keeps its design size. Raises if any text is below FS_MIN.
    """
    small = check_fontsizes(fig)
    if small:
        raise ValueError(f"{name}: text below {FS_MIN} pt: {small}")
    if width is not None:
        for _ in range(8):
            fig.canvas.draw()
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            w, h = fig.get_size_inches()
            if abs(bb.width - width) < 0.005:
                break
            fig.set_size_inches(w + (width - bb.width), h)
    written = []
    for d in _out_dirs(papers):
        d.mkdir(parents=True, exist_ok=True)
        pdf = d / f"{name}.pdf"
        # deterministic PDF (R18): no creation date, so a rebuild is byte-identical
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
        fig.savefig(d / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
        written.append(pdf)
    plt.close(fig)
    return written
