"""
Experiment 7: Multi-Armed Bandit Problem
--------------------------------------------
Aim: To implement the Multi-Armed Bandit problem using epsilon-Greedy and
Upper Confidence Bound (UCB) algorithms and evaluate cumulative rewards
obtained by different action-selection methods.
"""

import numpy as np
import matplotlib.pyplot as plt

N_ARMS = 10
N_STEPS = 1000
N_RUNS = 200

rng_master = np.random.default_rng(0)
true_action_values = rng_master.normal(0, 1, N_ARMS)


def get_reward(action, rng):
    return rng.normal(true_action_values[action], 1)


def epsilon_greedy_bandit(epsilon=0.1, n_steps=N_STEPS, rng=None):
    rng = rng or np.random.default_rng()
    Q = np.zeros(N_ARMS)
    N = np.zeros(N_ARMS)
    rewards = np.zeros(n_steps)
    for t in range(n_steps):
        if rng.random() < epsilon:
            action = rng.integers(N_ARMS)
        else:
            action = int(np.argmax(Q))
        reward = get_reward(action, rng)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards[t] = reward
    return rewards


def ucb_bandit(c=2, n_steps=N_STEPS, rng=None):
    rng = rng or np.random.default_rng()
    Q = np.zeros(N_ARMS)
    N = np.zeros(N_ARMS)
    rewards = np.zeros(n_steps)
    for t in range(1, n_steps + 1):
        if np.any(N == 0):
            action = int(np.argmin(N))  # try every arm at least once first
        else:
            ucb_values = Q + c * np.sqrt(np.log(t) / N)
            action = int(np.argmax(ucb_values))
        reward = get_reward(action, rng)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards[t - 1] = reward
    return rewards


def average_over_runs(func, n_runs=N_RUNS, **kwargs):
    all_rewards = np.zeros((n_runs, N_STEPS))
    for run in range(n_runs):
        rng = np.random.default_rng(2000 + run)
        all_rewards[run] = func(n_steps=N_STEPS, rng=rng, **kwargs)
    return all_rewards.mean(axis=0)


if __name__ == "__main__":
    print("Running Epsilon-Greedy and UCB bandit algorithms "
          f"over {N_RUNS} runs of {N_STEPS} steps each...\n")

    eps_rewards = average_over_runs(epsilon_greedy_bandit, epsilon=0.1)
    ucb_rewards = average_over_runs(ucb_bandit, c=2)

    eps_cumulative = np.cumsum(eps_rewards)
    ucb_cumulative = np.cumsum(ucb_rewards)

    print(f"Epsilon-Greedy -> total cumulative reward: {eps_cumulative[-1]:.2f}")
    print(f"UCB            -> total cumulative reward: {ucb_cumulative[-1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(eps_rewards, label="Epsilon-Greedy")
    axes[0].plot(ucb_rewards, label="UCB")
    axes[0].set_title("Average Reward per Step")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps_cumulative, label="Epsilon-Greedy")
    axes[1].plot(ucb_cumulative, label="UCB")
    axes[1].set_title("Cumulative Reward")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment7_multi_armed_bandit.png", dpi=150)
    print("\nPlot saved as 'experiment7_multi_armed_bandit.png'")
    plt.show()
