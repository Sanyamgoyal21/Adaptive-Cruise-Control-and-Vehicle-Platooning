"""Shared matplotlib style: one palette, recessive axes, consistent colours.

Colour roles
* CAT   - categorical (identity), fixed order, never cycled. The four follower
          strategies always use the same slot in every figure.
* RAMP  - single-hue sequential blue for vehicle position in the platoon
          (light = near the leader, dark = tail of the platoon).
* LEADER / text colours are neutral inks.
"""
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402
import numpy as np                                    # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
CRITICAL = "#c62828"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
RAMP = _BLUE
LEADER = TEXT
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])

MODE_COLORS = {"CRUISE": CAT[0], "FOLLOW": CAT[1], "CACC": CAT[2],
               "ACC-FALLBACK": CAT[3], "AEB": CRITICAL, "HUMAN": MUTED, "SCRIPTED": MUTED}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": TEXT,
    "lines.solid_capstyle": "round", "font.family": "DejaVu Sans",
})


def mode_strip(ax, t, mode_idx, legend_y=-0.9):
    """Draw the controller mode over time as a coloured band with a legend."""
    from platoon_sim.controllers import MODES
    present = []
    change = np.flatnonzero(np.diff(mode_idx)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(mode_idx) - 1]
    for s, e in zip(starts, ends):
        name = MODES[mode_idx[s]]
        ax.axvspan(t[s], t[e], color=MODE_COLORS[name], lw=0)
        if name not in present:
            present.append(name)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlim(t[0], t[-1])
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("mode", rotation=0, ha="right", va="center")
    ax.legend(handles=[Patch(color=MODE_COLORS[m], label=m) for m in present],
              loc="upper center", bbox_to_anchor=(0.5, legend_y),
              ncol=len(present))
