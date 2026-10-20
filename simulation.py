""" FINANCE PART

In this notebook we simulate paths of the german bond rate and spread.
They both follow mean reverting OU processes (https://fr.wikipedia.org/wiki/Processus_d%27Ornstein-Uhlenbeck)
In the calm regime they are independent. A share p_crisis of the paths enters a crisis on a random day:
the spread jumps, then has a higher mean and volatility, and is negatively correlated with the German rate.
We also compute P&L (profit and losses) and CVAR of this P&L
P&L : how much money we gained at the end (can be negative)
CVAR (of the P&L) : in the worst 5% of the scenarios, what is the average loss

=> CVAR is the loss function : we want to control worst case scenario
"""

import numpy as np
import torch

N_STEPS = 63        # 3 months of business days
DT = 1 / 252
MATURITY = 10.0     # maturity of both zero-coupon bonds at t = 0, in years
VOL_WINDOW = 10     # days used to estimate the spread volatility (simulated as calm history before t = 0)

R0, B, A, SIGMA_R = 0.025, 0.025, 0.5, 0.008                 # German rate: start, mean, speed, vol
S0, S_BAR, KAPPA, SIGMA_S = 0.0065, 0.0065, 2.0, 0.0015      # spread, calm: start, mean, speed, vol
S_BAR_CRISIS, KAPPA_CRISIS, SIGMA_S_CRISIS = 0.0085, 1.0, 0.006   # spread, crisis: mean, speed, vol
RHO_CRISIS = -0.5                   # correlation of the shocks on r and s in crisis (0 in calm)
JUMP_MIN, JUMP_MAX = 0.002, 0.004   # spread jump on the first crisis day, uniform


def ou_step(x, mean, speed, vol, z):
    """Exact step of dx = speed (mean - x) dt + vol dW, with z ~ N(0, 1).
    This equation comes from the closed form of the OU DSE"""
    decay = np.exp(-speed * DT)
    return mean + (x - mean) * decay + vol * np.sqrt((1 - decay**2) / (2 * speed)) * z


def simulate(n_paths, rng, p_crisis):
    """Paths over the N days, after VOL_WINDOW days of calm history. Returns 4 tensors:
     - r, s  : (n_paths, N+1) German rate and spread at the dates t = 0, ..., N
     - vol   : (n_paths, N)   spread volatility estimated at t = 0, ..., N-1 from the last VOL_WINDOW daily moves
     - start : (n_paths,)     day of the crisis start (the step start -> start+1 has the jump); N if no crisis
    """
    W = VOL_WINDOW
    start = np.where(rng.random(n_paths) < p_crisis, rng.integers(0, N_STEPS, n_paths), N_STEPS)
    jump = rng.uniform(JUMP_MIN, JUMP_MAX, n_paths)
    r = np.empty((n_paths, W + N_STEPS + 1))
    s = np.empty((n_paths, W + N_STEPS + 1))
    r[:, 0], s[:, 0] = R0, S0
    for k in range(W + N_STEPS):            # step k of the full timeline is the day t = k - W
        crisis = k - W >= start
        rho = np.where(crisis, RHO_CRISIS, 0.0)
        z_r, z_perp = rng.standard_normal((2, n_paths))
        z_s = rho * z_r + np.sqrt(1 - rho**2) * z_perp
        r[:, k + 1] = ou_step(r[:, k], B, A, SIGMA_R, z_r)
        s[:, k + 1] = ou_step(s[:, k], np.where(crisis, S_BAR_CRISIS, S_BAR), np.where(crisis, KAPPA_CRISIS, KAPPA),
                              np.where(crisis, SIGMA_S_CRISIS, SIGMA_S), z_s)
        s[:, k + 1] += np.where(k - W == start, jump, 0.0)

    ds = np.diff(s, axis=1)                 # daily spread moves, W of them before t = 0
    vol = np.stack([np.sqrt((ds[:, t:t + W] ** 2).mean(axis=1) / DT) for t in range(N_STEPS)], axis=1)
    as_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    return as_tensor(r[:, W:]), as_tensor(s[:, W:]), as_tensor(vol), torch.tensor(start)


def bond_prices(r, s):
    """Prices (nominal 100) of the OAT and of the Bund at the dates t = 0, ..., N."""
    time_left = MATURITY - torch.arange(N_STEPS + 1) * DT
    return 100 * torch.exp(-(r + s) * time_left), 100 * torch.exp(-r * time_left)


def pnl(p_fr, p_de, h):
    """P&L = (P^FR_N - P^FR_0) - sum_t h_t (P^DE_{t+1} - P^DE_t). h: (n_paths, N) -> (n_paths,)."""
    return p_fr[:, -1] - p_fr[:, 0] - (h * torch.diff(p_de, dim=1)).sum(dim=1)


def dv01_hedge(p_fr, p_de):
    """Duration hedge h_t = P^FR_t / P^DE_t: cancels the German rate risk, optimal when rho = 0."""
    return p_fr[:, :-1] / p_de[:, :-1]


def oracle_hedge(p_fr, p_de, start):
    """Minimum-variance hedge knowing the regime: (1 + beta) * DV01 hedge, beta = rho sigma_s / sigma_r.
    beta = 0 in calm, -0.375 in crisis. It minimizes the variance, not the CVaR."""
    crisis = torch.arange(N_STEPS)[None, :] >= start[:, None]
    beta = torch.where(crisis, RHO_CRISIS * SIGMA_S_CRISIS / SIGMA_R, 0.0)
    return (1 + beta) * dv01_hedge(p_fr, p_de)


def cvar(pnl_values, level=0.05):
    """CVaR at 5%: mean loss (-P&L) over the 5% worst paths."""
    k = max(1, int(level * len(pnl_values)))
    return (-pnl_values).topk(k).values.mean()
