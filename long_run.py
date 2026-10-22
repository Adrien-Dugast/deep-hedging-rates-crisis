"""Longer training (2000 steps) with the smallest learning rate of each optimizer, which gave the smoothest curves.

Usage: python long_run.py 1_crisis/p_01_long 0.01   (folder, crisis probability), same outputs as train.py
"""

import train

train.N_ITER = 2000
train.LEARNING_RATES = {"GD": [1e-3], "Adam": [1e-4], "Spectral": [1e-3]}

if __name__ == "__main__":
    train.main()
