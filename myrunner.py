"""Bare-bones runner for the SARSA(lambda) sweep and learning-curve plot."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv
from myagent import SarsaLambdaAgent


LAMBDAS = (0.0, 0.3, 0.6, 0.9, 1.0)
SEEDS = (11, 22, 33, 44, 55)
EPISODES = 5000
WINDOW = 100


all_returns = {}
for lam in LAMBDAS:
    seed_returns = []

    for seed in SEEDS:
        env = gym.make("cs272/Staircase-v0")
        env.reset(seed=seed)
        agent = SarsaLambdaAgent(
            env,
            lam=lam,
            total_epi=EPISODES,
            seed=seed,
        )
        seed_returns.append(agent.learn())
        env.close()

    all_returns[lam] = np.asarray(seed_returns)

# Smooth each seed's returns, then plot the mean and spread across seeds.
episodes = np.arange(WINDOW, EPISODES + 1)
weights = np.ones(WINDOW) / WINDOW

for lam, returns in all_returns.items():
    smoothed = np.asarray([
        np.convolve(run, weights, mode="valid")
        for run in returns
    ])
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)

    plt.plot(episodes, mean, label=f"lambda={lam:g}")
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

plt.xlabel("Episode")
plt.ylabel(f"Mean return ({WINDOW}-episode moving average)")
plt.title("SARSA(lambda) sweep")
plt.legend()
plt.tight_layout()
plt.savefig("lambda_sweep.png", dpi=200)
plt.show()
