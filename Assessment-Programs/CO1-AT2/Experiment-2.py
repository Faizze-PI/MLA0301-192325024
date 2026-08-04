"""
Experiment 2: ε-greedy Multi-Armed Bandit (500-Trial Budget)
------------------------------------------------------------------------
Aim: Implement ε-greedy MAB to maximize rewards under a limited budget
of 500 trials. Compare ε = 0.1, 0.3, and 0.5 on exploration vs
exploitation and constraint satisfaction.

Bandits: 10 arms, rewards ~ N(μ_i, 1) with μ_i sampled from N(0,1)
Budget:  500 trials per ε value
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

N_ARMS = 10
N_TRIALS = 500
TRUE_MEANS = np.random.randn(N_ARMS)


def run_epsilon_greedy(epsilon, n_trials=N_TRIALS):
    Q = np.zeros(N_ARMS)
    N = np.zeros(N_ARMS)
    rewards = np.zeros(n_trials)
    cumulative = np.zeros(n_trials)
    cum_reward = 0

    for t in range(n_trials):
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ARMS)
        else:
            action = int(np.argmax(Q))

        reward = np.random.randn() + TRUE_MEANS[action]
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        cum_reward += reward
        rewards[t] = reward
        cumulative[t] = cum_reward

    return rewards, cumulative, Q


def plot_results():
    epsilons = [0.1, 0.3, 0.5]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for eps, color in zip(epsilons, colors):
        _, cumulative, _ = run_epsilon_greedy(eps)
        ax.plot(cumulative, label=f"ε = {eps}", color=color, linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward under 500-Trial Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    window = 20
    for eps, color in zip(epsilons, colors):
        rewards, _, _ = run_epsilon_greedy(eps)
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), avg, label=f"ε = {eps}", color=color, linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Average Reward (window={window})")
    ax.set_title("Exploration vs Exploitation Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment2_mab_epsilon.png", dpi=150)
    print("Plot saved as 'experiment2_mab_epsilon.png'")


if __name__ == "__main__":
    print("Experiment 2: epsilon-greedy Multi-Armed Bandit (500-Trial Budget)\n")
    print(f"Arms: {N_ARMS} | Trials: {N_TRIALS} | True means: {np.round(TRUE_MEANS, 2)}\n")

    for eps in [0.1, 0.3, 0.5]:
        rewards, cumulative, Q = run_epsilon_greedy(eps)
        print(f"  epsilon={eps}: Final cumulative reward = {cumulative[-1]:.1f} | "
              f"Estimated Q* = {np.round(Q, 2)}")

    print()
    plot_results()
    print("\nExperiment 2 completed successfully!")
