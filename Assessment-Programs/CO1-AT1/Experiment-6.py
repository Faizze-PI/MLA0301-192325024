"""
Experiment 6: Exploration vs. Exploitation Strategies
--------------------------------------------------------
Aim: To implement epsilon-Greedy, Greedy, and Random action-selection
strategies and compare their performance in balancing exploration and
exploitation during Reinforcement Learning.

Testbed: classic k-armed bandit (10 arms, stationary Gaussian rewards).
"""

import numpy as np
import matplotlib.pyplot as plt

N_ARMS = 10
N_STEPS = 1000
N_RUNS = 200  # average over many runs for smooth curves

rng_master = np.random.default_rng(42)
true_action_values = rng_master.normal(0, 1, N_ARMS)


def get_reward(action, rng):
    return rng.normal(true_action_values[action], 1)


def run_strategy(strategy, epsilon=0.1, n_steps=N_STEPS, rng=None):
    rng = rng or np.random.default_rng()
    Q = np.zeros(N_ARMS)
    N = np.zeros(N_ARMS)
    rewards = np.zeros(n_steps)

    for t in range(n_steps):
        if strategy == "random":
            action = rng.integers(N_ARMS)
        elif strategy == "greedy":
            action = int(np.argmax(Q))
        elif strategy == "epsilon_greedy":
            if rng.random() < epsilon:
                action = rng.integers(N_ARMS)
            else:
                action = int(np.argmax(Q))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        reward = get_reward(action, rng)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards[t] = reward

    return rewards


def average_over_runs(strategy, epsilon=0.1, n_runs=N_RUNS):
    all_rewards = np.zeros((n_runs, N_STEPS))
    for run in range(n_runs):
        rng = np.random.default_rng(1000 + run)
        all_rewards[run] = run_strategy(strategy, epsilon, rng=rng)
    return all_rewards.mean(axis=0)


if __name__ == "__main__":
    print("Running Random, Greedy, and Epsilon-Greedy (eps=0.1) over "
          f"{N_RUNS} runs of {N_STEPS} steps each...\n")

    random_rewards = average_over_runs("random")
    greedy_rewards = average_over_runs("greedy")
    eps_greedy_rewards = average_over_runs("epsilon_greedy", epsilon=0.1)

    print(f"Random          -> average reward: {random_rewards.mean():.4f}")
    print(f"Greedy          -> average reward: {greedy_rewards.mean():.4f}")
    print(f"Epsilon-Greedy  -> average reward: {eps_greedy_rewards.mean():.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(random_rewards, label="Random")
    plt.plot(greedy_rewards, label="Greedy")
    plt.plot(eps_greedy_rewards, label="Epsilon-Greedy (eps=0.1)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Exploration vs Exploitation Strategies (10-armed bandit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment6_exploration_vs_exploitation.png", dpi=150)
    print("\nPlot saved as 'experiment6_exploration_vs_exploitation.png'")
    plt.show()
