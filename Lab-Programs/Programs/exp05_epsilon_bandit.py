"""
Exp05 - Epsilon-Greedy Multi-Armed Bandit for Online Ad Recommendation
========================================================================
K=5 ads with hidden CTRs [0.05, 0.12, 0.03, 0.20, 0.08].
Compare epsilon = [0.01, 0.1, 0.3] over 1000 rounds.
Plots : cumulative reward comparison + % optimal arm chosen over time.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- bandit setup -------------------------------------------------
K = 5
TRUE_CTRS = np.array([0.05, 0.12, 0.03, 0.20, 0.08])
OPTIMAL_ARM = int(np.argmax(TRUE_CTRS))     # arm 3 (CTR=0.20)
N_ROUNDS = 1000
N_RUNS   = 200                              # independent runs for averaging

EPSILON_LIST = [0.01, 0.1, 0.3]

# -- single run ---------------------------------------------------
def run_bandit(epsilon, n_rounds=N_ROUNDS):
    """Simulate one run of epsilon-greedy bandit. Return per-round rewards & chosen arms."""
    Q = np.zeros(K)
    N = np.zeros(K)
    rewards = np.zeros(n_rounds)
    chosen  = np.zeros(n_rounds, dtype=int)

    for t in range(n_rounds):
        # epsilon-greedy
        if np.random.rand() < epsilon:
            arm = np.random.randint(K)
        else:
            arm = int(np.argmax(Q))

        # stochastic reward (Bernoulli with CTR)
        reward = float(np.random.rand() < TRUE_CTRS[arm])
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]    # incremental average

        rewards[t] = reward
        chosen[t]  = arm

    return rewards, chosen

# -- run all experiments ------------------------------------------
def run_experiments():
    results = {}
    for eps in EPSILON_LIST:
        cum_rewards  = np.zeros((N_RUNS, N_ROUNDS))
        opt_choices  = np.zeros((N_RUNS, N_RUNS))  # placeholder (reused below)

        all_chosen   = np.zeros((N_RUNS, N_ROUNDS), dtype=int)
        all_rewards  = np.zeros((N_RUNS, N_ROUNDS))

        for run in range(N_RUNS):
            rew, ch = run_bandit(eps)
            all_rewards[run] = rew
            all_chosen[run]  = ch

        # cumulative reward (sum over rounds, averaged over runs)
        cum_avg = np.cumsum(all_rewards, axis=1).mean(axis=0)
        # % optimal arm chosen (averaged over runs)
        opt_pct = (all_chosen == OPTIMAL_ARM).mean(axis=0) * 100.0

        results[eps] = {
            "cum_avg": cum_avg,
            "opt_pct": opt_pct,
        }

        print(f"  eps={eps:.2f}  final avg cumulative reward={cum_avg[-1]:.1f}  "
              f"optimal arm %={opt_pct[-1]:.1f}%")

    return results

# -- plotting -----------------------------------------------------
COLORS = {0.01: "#e74c3c", 0.1: "#3498db", 0.3: "#2ecc71"}

def plot_cumulative_reward(results, filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\exp05_cumulative_reward.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for eps, data in results.items():
        ax.plot(data["cum_avg"], label=f"epsilon = {eps:.2f}", color=COLORS[eps], linewidth=1.8)
    ax.axhline(y=TRUE_CTRS[OPTIMAL_ARM] * N_ROUNDS, color="gray", linestyle="--",
               alpha=0.6, label=f"Optimal arm CTR x T = {TRUE_CTRS[OPTIMAL_ARM]*N_ROUNDS:.0f}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward (averaged)")
    ax.set_title("Epsilon-Greedy Bandit: Cumulative Reward Comparison")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_optimal_arm_pct(results, filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\exp05_optimal_arm_pct.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for eps, data in results.items():
        ax.plot(data["opt_pct"], label=f"epsilon = {eps:.2f}", color=COLORS[eps], linewidth=1.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("% Chose Optimal Arm")
    ax.set_title("Epsilon-Greedy Bandit: Optimal Arm Selection Over Time")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Exp 05 : Epsilon-Greedy Multi-Armed Bandit (Online Ads)")
    print("=" * 60)

    print(f"\n  K = {K} ads")
    print(f"  True CTRs = {TRUE_CTRS.tolist()}")
    print(f"  Optimal arm = {OPTIMAL_ARM}  (CTR = {TRUE_CTRS[OPTIMAL_ARM]:.2f})")
    print(f"  N_rounds = {N_ROUNDS}, N_runs = {N_RUNS}")

    print("\n[1] Running experiments ...")
    results = run_experiments()

    print("\n[2] Saving plots ...")
    plot_cumulative_reward(results, "exp05_cumulative_reward.png")
    plot_optimal_arm_pct(results, "exp05_optimal_arm_pct.png")

    print("\n[Done] Experiment 05 complete.")

if __name__ == "__main__":
    main()

