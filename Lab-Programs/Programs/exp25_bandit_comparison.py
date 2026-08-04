"""
Exp 25 – Epsilon-Greedy vs UCB vs Thompson Sampling for Ad CTR
================================================================
K = 5 ads with fixed, unknown click-through probabilities.
1 000 rounds of arm pulls.  Three algorithms run independently on the same bandit.
Metrics: cumulative reward and cumulative regret.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

K = 5
ROUNDS = 1000
TRUE_CTRS = np.array([0.02, 0.05, 0.10, 0.18, 0.25])
BEST_CTR = TRUE_CTRS.max()
SEED = 42


# -- Epsilon-Greedy ----------------------------------------------------
def epsilon_greedy(K, rounds, cts, eps=0.1, seed=0):
    rng = np.random.RandomState(seed)
    counts = np.zeros(K)
    values = np.zeros(K)
    rewards, regrets = [], []
    cumulative = 0
    for t in range(1, rounds + 1):
        if rng.rand() < eps:
            a = rng.randint(K)
        else:
            a = int(np.argmax(values))
        r = float(rng.rand() < cts[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        cumulative += r
        regrets.append(t * BEST_CTR - cumulative)
        rewards.append(cumulative)
    return np.array(rewards), np.array(regrets)


# -- UCB ---------------------------------------------------------------
def ucb(K, rounds, cts, seed=0):
    rng = np.random.RandomState(seed)
    counts = np.zeros(K)
    values = np.zeros(K)
    rewards, regrets = [], []
    cumulative = 0
    for t in range(1, rounds + 1):
        if t <= K:
            a = t - 1  # pull each arm once first
        else:
            ucb_vals = values + np.sqrt(2 * np.log(t) / counts)
            a = int(np.argmax(ucb_vals))
        r = float(rng.rand() < cts[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        cumulative += r
        regrets.append(t * BEST_CTR - cumulative)
        rewards.append(cumulative)
    return np.array(rewards), np.array(regrets)


# -- Thompson Sampling -------------------------------------------------
def thompson_sampling(K, rounds, cts, seed=0):
    rng = np.random.RandomState(seed)
    alpha = np.ones(K)
    beta = np.ones(K)
    rewards, regrets = [], []
    cumulative = 0
    for t in range(1, rounds + 1):
        samples = rng.beta(alpha, beta)
        a = int(np.argmax(samples))
        r = float(rng.rand() < cts[a])
        if r:
            alpha[a] += 1
        else:
            beta[a] += 1
        cumulative += r
        regrets.append(t * BEST_CTR - cumulative)
        rewards.append(cumulative)
    return np.array(rewards), np.array(regrets)


def plot_results(rewards_eg, rewards_ucb, rewards_ts,
                 regrets_eg, regrets_ucb, regrets_ts):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(1, ROUNDS + 1)
    ax1.plot(x, rewards_eg, label="Epsilon-Greedy", alpha=0.8)
    ax1.plot(x, rewards_ucb, label="UCB", alpha=0.8)
    ax1.plot(x, rewards_ts, label="Thompson Sampling", alpha=0.8)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Ad CTR – Cumulative Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, regrets_eg, label="Epsilon-Greedy", alpha=0.8)
    ax2.plot(x, regrets_ucb, label="UCB", alpha=0.8)
    ax2.plot(x, regrets_ts, label="Thompson Sampling", alpha=0.8)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title("Ad CTR – Cumulative Regret (lower is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "exp25_bandit_comparison_results.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved -> {path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Exp 25 – Bandit Comparison: Eps-Greedy / UCB / Thompson")
    print("=" * 60)
    print(f"  K={K}  rounds={ROUNDS}  true CTRs={TRUE_CTRS}  best={BEST_CTR}")

    r_eg, reg_eg = epsilon_greedy(K, ROUNDS, TRUE_CTRS, eps=0.1, seed=SEED)
    r_ucb, reg_ucb = ucb(K, ROUNDS, TRUE_CTRS, seed=SEED)
    r_ts, reg_ts = thompson_sampling(K, ROUNDS, TRUE_CTRS, seed=SEED)

    print(f"\n  -- Final Cumulative Reward --")
    print(f"  Epsilon-Greedy  : {r_eg[-1]:.0f}")
    print(f"  UCB             : {r_ucb[-1]:.0f}")
    print(f"  Thompson        : {r_ts[-1]:.0f}")
    print(f"\n  -- Final Cumulative Regret --")
    print(f"  Epsilon-Greedy  : {reg_eg[-1]:.0f}")
    print(f"  UCB             : {reg_ucb[-1]:.0f}")
    print(f"  Thompson        : {reg_ts[-1]:.0f}")

    plot_results(r_eg, r_ucb, r_ts, reg_eg, reg_ucb, reg_ts)
    print("\nDone.")

