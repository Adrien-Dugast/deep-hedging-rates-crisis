"""Compares the long runs (smallest lr, 2000 steps) with the best lr of the 1000-step grid, on both test sets.
For each test set: the full figure (log scale) and a zoom that skips the first steps (linear scale).

Usage: python plot_long.py 1_crisis/p_01_long 1_crisis/p_01   (long folder, grid folder; figures go to the long folder)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot import load
from train import best_lr, draw_references, steps

ZOOM_START = 500    # first step shown in the zoom: the early jumps of the loss hide the end of the curves


def draw(long, grid, references, test_set, ax, start=0):
    """Solid: long run. Dashed, same color: best lr of the grid. Only the steps >= start."""
    for (name, curves), color in zip(long.items(), ["C0", "C1", "C2"]):
        for runs, lr, ls, n in [(curves, best_lr(curves), "-", 2000), (grid[name], best_lr(grid[name]), "--", 1000)]:
            x, y = steps(runs[lr][test_set]), np.array(runs[lr][test_set])
            keep = x >= start
            ax.plot(x[keep], y[keep], color=color, ls=ls, label=f"{name} (lr = {lr:g}, {n} steps)")
    draw_references(ax, references[test_set])


if __name__ == "__main__":
    folder, grid_folder = Path(sys.argv[1]), Path(sys.argv[2])
    long, references = load(folder)
    grid, _ = load(grid_folder)
    grid = {name: {lr: c for lr, c in curves.items() if lr != 1} for name, curves in grid.items()}
    for test_set, suffix in [("all", ""), ("crisis", "_crisis_only")]:
        title = "Smallest lr, 2000 steps vs best lr, 1000 steps" + (", crisis test set" if test_set == "crisis" else "")
        for start, scale, zoom in [(0, "log", ""), (ZOOM_START, "linear", "_zoom")]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            draw(long, grid, references, test_set, ax, start)
            ax.set(xlabel="step", ylabel="test CVaR 5%", yscale=scale, title=title)
            ax.legend(fontsize=8)
            fig.savefig(folder / f"long_vs_grid{suffix}{zoom}.png", dpi=150, bbox_inches="tight")
