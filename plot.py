"""Redraws the figures of a folder from its results.json (no training), plus a zoom on the end of the Spectral runs.

Usage: python plot.py 0_no_crisis
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train import draw_references, plot, steps


def load(folder):
    with open(folder / "results.json") as f:
        data = json.load(f)
    curves = {name: {float(lr): c for lr, c in runs.items()} for name, runs in data["curves"].items()}
    if "optimum" in data:   # 0_no_crisis: one test set and one reference, saved before the crisis was added
        curves = {name: {lr: {"all": c} for lr, c in runs.items()} for name, runs in curves.items()}
        return curves, {"all": {"theoretical optimum": data["optimum"]}}
    return curves, data["references"]


def zoom(curves, references, folder, name, lrs, start=800, end=1000):
    """Test loss of some learning rates of one optimizer, between two steps only (linear scale)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for lr in lrs:
        x, y = steps(curves[name][lr]["all"]), np.array(curves[name][lr]["all"])
        keep = (x >= start) & (x <= end)
        if not keep.any():  # run stopped before the zoom window
            continue
        ax.plot(x[keep], y[keep], label=f"lr = {lr:g} (mean {y[keep].mean():.4f}, std {y[keep].std():.4f})")
    draw_references(ax, references["all"])
    ax.set(xlabel="step", ylabel="test CVaR 5%", title=f"{name}: steps {start} to {end}")
    ax.legend()
    fig.savefig(folder / f"zoom_{name}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    curves, references = load(folder)
    plot(curves, references, folder)
    zoom(curves, references, folder, "Spectral", [0.001, 0.01])
