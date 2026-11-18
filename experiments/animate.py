"""Animated top-down view of three platoons in stop-and-go traffic.

    python experiments/animate.py          -> results/platoon_animation.gif

Each row is one strategy; cars are drawn relative to the leader, coloured by
speed (darker = slower). Watch the ACC h=0.6 s string compress and crash
while CACC h=0.6 s moves as one tight block.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")   # file output only, no window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import style                      # noqa: E402
from platoon_sim.scenarios import run_platoon       # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

plt = style.plt
ROWS = [("ACC  h=1.6 s", "ACC", {"h": 1.6}),
        ("ACC  h=0.6 s", "ACC", {"h": 0.6}),
        ("CACC h=0.6 s", "CACC", {"h": 0.6})]
N, T, FPS, STEP = 8, 80.0, 15, 0.2


def main():
    runs = [run_platoon(N, k, kw, profile="stop_and_go", T=T)[0] for _, k, kw in ROWS]
    frames = np.arange(0, T, STEP)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(11, 4.6), sharex=True)
    norm = plt.Normalize(0, 16)
    cars = []
    for ax, (lab, _, _), r in zip(axes, ROWS, runs):
        ax.set_xlim(-260, 8)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([])
        ax.grid(False)
        ax.axhline(-1.4, color=style.MUTED, lw=1)
        ax.axhline(1.4, color=style.MUTED, lw=1)
        ax.set_title(lab, loc="left")
        rects = []
        for i in range(N + 1):
            rc = Rectangle((0, -0.6), 4.5, 1.2, color=style.TEXT if i == 0 else "#3987e5")
            ax.add_patch(rc)
            rects.append(rc)
        txt = ax.text(0.99, 0.92, "", transform=ax.transAxes, ha="right", va="top",
                      color=style.TEXT2, fontsize=9)
        cars.append((rects, txt))
    axes[-1].set_xlabel("position relative to the leader [m]")
    clock = fig.text(0.06, 0.97, "", color=style.TEXT, fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=style.SPEED_CMAP), ax=axes,
                 shrink=0.8, pad=0.01).set_label("speed [m/s]")

    def update(t):
        k = int(round(t / runs[0].dt))
        for (rects, txt), r in zip(cars, runs):
            for i, rc in enumerate(rects):
                rc.set_x(r.x[i, k] - r.x[0, k] - 4.5)
                if i:
                    rc.set_color(style.SPEED_CMAP(norm(r.v[i, k])))
            crashed = [c for c in r.collisions if c[0] <= t]
            txt.set_text(f"min gap {np.nanmin(r.gap[1:, k]):5.1f} m"
                         + (f"   COLLISIONS: {len(crashed)}" if crashed else ""))
            txt.set_color(style.CRITICAL if crashed else style.TEXT2)
        clock.set_text(f"Stop-and-go traffic   t = {t:4.1f} s   leader speed {runs[0].v[0, k]:4.1f} m/s")
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = os.path.join(ROOT, "results", "platoon_animation.gif")
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=80)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
