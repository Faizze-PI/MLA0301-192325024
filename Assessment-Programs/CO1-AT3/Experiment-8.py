"""
Experiment 8: Random vs ε-greedy MAB Comparison
------------------------------------------------------------------------
Aim: Analyse experimental comparison between random action selection and
ε-greedy action selection in a Multi-Armed Bandit problem under a fixed
time constraint. Record cumulative rewards and explain which strategy
provides better performance.

Bandits: 10 arms, rewards ~ N(μ_i, 1)
Budget:  500 trials
Compare: Pure random (ε=1.0) vs ε-greedy (ε=0.1)
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


def run_bandit(epsilon, n_trials=N_TRIALS):
    Q = np.zeros(N_ARMS)
    N = np.zeros(N_ARMS)
    cumulative = np.zeros(n_trials)
    arm_counts = np.zeros(N_ARMS)
    cum = 0

    for t in range(n_trials):
        if np.random.rand() < epsilon:
            action = np.random.randint(N_ARMS)
        else:
            action = int(np.argmax(Q))

        reward = np.random.randn() + TRUE_MEANS[action]
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        arm_counts[action] += 1
        cum += reward
        cumulative[t] = cum

    return cumulative, arm_counts, Q


def plot_results():
    cum_random, counts_random, q_random = run_bandit(1.0)
    cum_greedy, counts_greedy, q_greedy = run_bandit(0.1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: cumulative reward
    ax = axes[0]
    ax.plot(cum_random, label="Random (ε=1.0)", color="#F44336", linewidth=1.5)
    ax.plot(cum_greedy, label="ε-greedy (ε=0.1)", color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: arm selection distribution
    x = np.arange(N_ARMS)
    width = 0.35
    ax = axes[1]
    ax.bar(x - width / 2, counts_random, width, label="Random", color="#F44336", alpha=0.7)
    ax.bar(x + width / 2, counts_greedy, width, label="ε-greedy", color="#2196F3", alpha=0.7)
    best_arm = int(np.argmax(TRUE_MEANS))
    ax.axvline(x=best_arm, color="green", linestyle="--", linewidth=2, label=f"Best arm ({best_arm})")
    ax.set_xlabel("Arm")
    ax.set_ylabel("Times Selected")
    ax.set_title("Arm Selection Distribution")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: moving average reward
    ax = axes[2]
    window = 20
    for cum, label, color in [(cum_random, "Random", "#F44336"), (cum_greedy, "ε-greedy", "#2196F3")]:
        diffs = np.diff(cum, prepend=0)
        ma = np.convolve(diffs, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(ma)), ma, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"Average Reward (window={window})")
    ax.set_title("Exploitation Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment8_random_vs_greedy.png", dpi=150)
    print("Plot saved as 'experiment8_random_vs_greedy.png'")

    return cum_random[-1], cum_greedy[-1]


if __name__ == "__main__":
    print("Experiment 8: Random vs ε-greedy MAB Comparison\n")
    print(f"Arms: {N_ARMS} | Trials: {N_TRIALS} (fixed time constraint)")
    print(f"True means: {np.round(TRUE_MEANS, 2)}")
    print(f"Best arm: {np.argmax(TRUE_MEANS)} (μ={TRUE_MEANS[np.argmax(TRUE_MEANS)]:.2f})\n")

    final_random, final_greedy = plot_results()

    print(f"\nFinal cumulative rewards:")
    print(f"  Random  (ε=1.0): {final_random:.1f}")
    print(f"  ε-greedy (ε=0.1): {final_greedy:.1f}")
    print(f"  Improvement: {final_greedy - final_random:.1f}")
    print(f"\nConclusion: ε-greedy {'outperforms' if final_greedy > final_random else 'is outperformed by'} random selection")
    print("because it exploits learned Q-values while still exploring occasionally.\n")
    print("Experiment 8 completed successfully!")
