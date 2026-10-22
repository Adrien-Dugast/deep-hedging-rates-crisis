""" OPTIMISATION PART

Trains the hedging network with GD, Adam and Spectral GD over a grid of learning rates, then plots.
Usage: python train.py 1_crisis/p_05 0.05   (folder, crisis probability; writes results.json and the figures there)
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from simulation import B, N_STEPS, SIGMA_S, S_BAR, bond_prices, cvar, dv01_hedge, oracle_hedge, pnl, simulate

P_CRISIS = 0.05     # default, overwritten by the command-line argument
N_ITER = 1000
BATCH_SIZE = 1024
N_TEST = 20_000
N_TEST_CRISIS = 1000    # second test set, crisis paths only
EVAL_EVERY = 10
PATIENCE = 300      # early stopping: stop when the test loss has not improved for 300 steps
LEARNING_RATES = {"GD": [1e-3, 1e-2, 1e-1],
                  "Adam": [1e-4, 1e-3, 1e-2, 1e-1],
                  "Spectral": [1e-3, 1e-2, 1e-1]}


def inputs(r, s, vol):
    """Network inputs at the decision dates t = 0, ..., N-1: (T-t)/T, rescaled rate, spread and log spread vol."""
    n_paths = r.shape[0]
    tau = (1 - torch.arange(N_STEPS) / N_STEPS).expand(n_paths, N_STEPS)
    return torch.stack([tau, (r[:, :-1] - B) / 0.01, (s[:, :-1] - S_BAR) / 0.001, torch.log(vol / SIGMA_S)], dim=-1)


def make_net():
    torch.manual_seed(0)    # same initialization for every run
    return torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(),
                               torch.nn.Linear(32, 32), torch.nn.ReLU(),
                               torch.nn.Linear(32, 1))


def loss_on(net, paths):
    """CVaR of the P&L when h_t = net(inputs at t). The net is applied to the N dates at once."""
    r, s, vol, _ = paths
    p_fr, p_de = bond_prices(r, s)
    h = net(inputs(r, s, vol)).squeeze(-1)
    return cvar(pnl(p_fr, p_de, h))


class Spectral(torch.optim.Optimizer):
    """Spectral GD: W <- W - lr U V^T for a weight matrix with gradient G = U S V^T (exact SVD),
    and b <- b - lr g / |g| for a bias vector (the same rule for a vector)."""

    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if p.ndim == 2:
                    U, _, Vh = torch.linalg.svd(g, full_matrices=False)
                    p -= group["lr"] * U @ Vh
                else:
                    p -= group["lr"] * g / (g.norm() + 1e-12)


def make_optimizer(name, params, lr):
    if name == "GD":
        return torch.optim.SGD(params, lr=lr)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    return Spectral(params, lr=lr)


def train(name, lr, test, test_crisis):
    """Up to 1000 steps on fresh batches (the same batches for every run).
    Returns the test CVaR curves: on the main test set ("all", used for early stopping) and on the crisis test set."""
    net = make_net()
    optimizer = make_optimizer(name, net.parameters(), lr)
    rng = np.random.default_rng(1)
    curve = {"all": [], "crisis": []}
    for it in range(N_ITER + 1):
        if it % EVAL_EVERY == 0:
            with torch.no_grad():
                curve["all"].append(loss_on(net, test).item())
                curve["crisis"].append(loss_on(net, test_crisis).item())
            losses = curve["all"]
            best_it = EVAL_EVERY * int(np.nanargmin(losses)) if np.isfinite(losses).any() else 0
            if not np.isfinite(losses[-1]) or it - best_it >= PATIENCE:
                break
        if it == N_ITER:
            break
        loss = loss_on(net, simulate(BATCH_SIZE, rng, P_CRISIS))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return curve


def best_lr(curves):
    """Model which performed best among the same with different training parameters (on the main test set)."""
    best = {lr: np.nanmin(c["all"]) if np.isfinite(c["all"]).any() else np.inf for lr, c in curves.items()}
    return min(best, key=best.get)


def steps(curve):
    return EVAL_EVERY * np.arange(len(curve))


def draw_curve(ax, curve, label):
    """Test curve until the training stopped; the legend says when it stopped early."""
    if len(curve) < N_ITER // EVAL_EVERY + 1:
        label += f", stopped at {steps(curve)[-1]}"
    ax.plot(steps(curve), curve, label=label)


def draw_references(ax, references):
    for (label, value), ls in zip(references.items(), ["--", ":"]):
        ax.axhline(value, color="gray", ls=ls, label=label)


def plot(results, references, folder):
    """references[test_set] = {label: CVaR of a reference hedge}, test_set in "all", "crisis".
    The lr = 1 runs are not drawn (they always diverge or stay far above the others)."""
    results = {name: {lr: c for lr, c in curves.items() if lr != 1} for name, curves in results.items()}
    for test_set, suffix in [("all", ""), ("crisis", "_crisis_only")]:
        if test_set not in references:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, curves in results.items():
            lr = best_lr(curves)
            draw_curve(ax, curves[lr][test_set], f"{name} (lr = {lr:g})")
        draw_references(ax, references[test_set])
        title = "Best learning rate of each optimizer" + (", crisis test set" if test_set == "crisis" else "")
        ax.set(xlabel="step", ylabel="test CVaR 5%", yscale="log", title=title)
        ax.legend()
        fig.savefig(folder / f"best_lr{suffix}.png", dpi=150, bbox_inches="tight")

    for name, curves in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for lr, curve in curves.items():
            draw_curve(ax, curve["all"], f"lr = {lr:g}")
        draw_references(ax, references["all"])
        ax.set(xlabel="step", ylabel="test CVaR 5%", yscale="log", title=f"{name}: learning rates")
        ax.legend()
        fig.savefig(folder / f"grid_{name}.png", dpi=150, bbox_inches="tight")


def reference_cvars(paths):
    r, s, _, start = paths
    p_fr, p_de = bond_prices(r, s)
    return {"duration hedge (ignores the crisis)": cvar(pnl(p_fr, p_de, dv01_hedge(p_fr, p_de))).item(),
            "oracle (knows the regime, min variance)": cvar(pnl(p_fr, p_de, oracle_hedge(p_fr, p_de, start))).item()}


def main():
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    global P_CRISIS
    P_CRISIS = float(sys.argv[2])
    test = simulate(N_TEST, np.random.default_rng(12345), P_CRISIS)
    test_crisis = simulate(N_TEST_CRISIS, np.random.default_rng(54321), 1.0)
    references = {"all": reference_cvars(test), "crisis": reference_cvars(test_crisis)}
    print("Reference CVaRs:", references)
    r, s, _, start = test
    p_fr, p_de = bond_prices(r, s)
    worst = (-pnl(p_fr, p_de, dv01_hedge(p_fr, p_de))).topk(int(0.05 * N_TEST)).indices
    print(f"Share of crisis paths in the 5% worst (duration hedge): {(start[worst] < N_STEPS).float().mean():.0%}")

    results = {}
    for name, lrs in LEARNING_RATES.items():
        results[name] = {}
        for lr in lrs:
            t0 = time.perf_counter()
            curve = results[name][lr] = train(name, lr, test, test_crisis)
            print(f"{name} lr={lr:g}: best CVaR {np.nanmin(curve['all']):.4f}, stopped at step "
                  f"{steps(curve['all'])[-1]} ({time.perf_counter() - t0:.0f} s)")

    with open(folder / "results.json", "w") as f:
        json.dump({"references": references, "curves": results}, f)
    plot(results, references, folder)


if __name__ == "__main__":
    main()
