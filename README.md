# Crisis-robust rate hedging

Hedging a long 10-year French government bond (OAT) by selling German Bunds, with a small neural network trained on the CVaR of the P&L. We compare three optimizers: gradient descent, Adam and spectral gradient descent.

## Model

- German 10y rate: Vasicek. France spread: Ornstein-Uhlenbeck. Zero-coupon bonds, maturity 10 years, horizon 63 business days (3 months), one hedge per day.
- Crisis (a share `P_CRISIS` of the paths, on a random day): the spread jumps by 20 to 40 bp, then has a higher mean (0.85%) and volatility (0.6%/yr), and its shocks are correlated at -0.5 with the German rate.
- Network: MLP 4 → 32 → 32 → 1 (ReLU). Inputs at day t: (T - t)/T, rescaled German rate, rescaled spread, log of the spread volatility estimated over the last 10 days. Output: h_t, the number of Bunds sold.
- Loss: CVaR 5% of the final P&L (mean loss over the 5% worst paths of a batch).

## Training

- 1000 steps, batches of 1024 freshly simulated paths, early stopping after 300 steps without improvement.
- Optimizers: GD (`torch.optim.SGD`), Adam (`torch.optim.Adam`), Spectral (W ← W − lr·UVᵀ with G = UΣVᵀ the exact SVD of the gradient; g/‖g‖ for the biases).
- Grid of 4 learning rates per optimizer. Same initialization, same batches and same test paths for every run.
- Test sets: 20,000 paths (with crises in proportion `P_CRISIS`) and 1000 crisis-only paths.

## Benchmarks

- Duration (DV01) hedge: h_t = P^FR_t / P^DE_t, cancels the German rate risk. Optimal in the calm regime.
- Oracle: minimum-variance hedge that knows the regime, (1 + ρσ_s/σ_r) × duration hedge. It minimizes the variance, not the CVaR, so the network can beat it.

## Results

- `0_no_crisis/` (no crisis, 3 inputs without the volatility estimate): the three optimizers reach the duration hedge (CVaR ≈ 0.77 per 100 of nominal) in about 100 steps.
- `1_crisis/` (`P_CRISIS = 5%`): the three optimizers reach a CVaR ≈ 1.97, below the duration hedge (2.17) and the oracle (2.03). No clear difference between optimizers (one seed only).

## Usage

```
python train.py 1_crisis    # trains the grid, writes results.json and the figures in the folder
python plot.py 1_crisis     # redraws the figures from results.json
```

Requires Python, PyTorch, NumPy and Matplotlib.
