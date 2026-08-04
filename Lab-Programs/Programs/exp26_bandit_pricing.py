"""
Exp 26 – Epsilon-Greedy vs UCB vs Thompson Sampling for Dynamic Pricing
=========================================================================
K = 5 price points, each with a hidden (sigmoid) conversion probability.
Revenue = price x conversion.  Agent picks a price each round.
True optimal price is known analytically (computed from hidden probs).

Algorithms learn to maximise cumulative revenue.
Outputs: cumulative-revenue comparison plot + summary table.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

K = 5
ROUNDS = 2000
PRICES = np.array([9.99, 14.99, 19.99, 24.99, 29.99])
# Hidden conversion probs (sigmoid-like decreasing in price)
CONVERSION_PROBS = np.array([0.40, 0.30, 0.22, 0.15, 0.08])
TRUE_REVENUES = PRICES * CONVERSION_PROBS
OPTIMAL_IDX = int(np.argmax(TRUE_REVENUES))
OPTIMAL_PRICE = PRICES[OPTIMAL_IDX]
SEED = 42


# -- Epsilon-Greedy ----------------------------------------------------
def epsilon_greedy(prices, conv, rounds, eps=0.1, seed=0):
    rng = np.random.RandomState(seed)
    K_ = len(prices)
    counts = np.zeros(K_)
    values = np.zeros(K_)
    rewards, regrets = [], []
    cumulative = 0
    for t in range(1, rounds + 1):
        if rng.rand() < eps:
            a = rng.randint(K_)
        else:
            a = int(np.argmax(values))
        sold = float(rng.rand() < conv[a])
        r = prices[a] * sold
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        cumulative += r
        regrets.append(t * TRUE_REVENUES.max() - cumulative)
        rewards.append(cumulative)
    return np.array(rewards), np.array(regrets), counts.copy()


# -- UCB ---------------------------------------------------------------
def ucb(prices, conv, rounds, seed=0):
    rng = np.random.RandomState(seed)
    K_ = len(prices)
    counts = np.zeros(K_)
    values = np.zeros(K_)
    rewards, regrets = [], []
    cumulative = 0
    for t in range(1, rounds + 1):
        if t <= K_:
            a = t - 1
        else:
            ucb_vals = values + np.sqrt(2 * np.log(t) / counts)
            a = int(np.argmax(ucb_vals))
        sold = float(rng.rand() < conv[a])
        r = prices[a] * sold
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        cumulative += r
        regrets.append(t * TRUE_REVENUES.max() - cumulative)
        rewards.append(cumulative)
    return np.array(rewards), np.array(regrets), counts.copy()


# -- Thompson Sampling (Beta-Bernoulli with revenue weighting) --------
def thompson_sampling(prices, conv, rounds, seed=0):
    rng = np.random.RandomState(seed)
    K_ = len(prices)
    alpha = np.ones(K_)
    beta_ = np.ones(K_)
    rewards, regrets = [], []
    cumulative = 0
    for t in range(1, rounds + 1):
        samples = rng.beta(alpha, beta_)
        # weight by price to pick highest expected revenue
        a = int(np.argmax(samples * prices))
        sold = float(rng.rand() < conv[a])
        r = prices[a] * sold
        if sold:
            alpha[a] += 1
        else:
            beta_[a] += 1
        cumulative += r
        regrets.append(t * TRUE_REVENUES.max() - cumulative)
        rewards.append(cumulative)
    return np.array(rewards), np.array(regrets), alpha - 1  # pseudo-counts


def plot_results(r_eg, r_ucb, r_ts, reg_eg, reg_ucb, reg_ts):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(1, ROUNDS + 1)

    ax1.plot(x, r_eg, label="Epsilon-Greedy", alpha=0.8)
    ax1.plot(x, r_ucb, label="UCB", alpha=0.8)
    ax1.plot(x, r_ts, label="Thompson Sampling", alpha=0.8)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Revenue ($)")
    ax1.set_title("Dynamic Pricing – Cumulative Revenue")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, reg_eg, label="Epsilon-Greedy", alpha=0.8)
    ax2.plot(x, reg_ucb, label="UCB", alpha=0.8)
    ax2.plot(x, reg_ts, label="Thompson Sampling", alpha=0.8)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Regret ($)")
    ax2.set_title("Dynamic Pricing – Regret (lower is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "exp26_pricing_results.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved -> {path}")
    plt.close()


def print_table(counts_eg, counts_ucb, counts_ts):
    print(f"\n  {'Price':>8}  {'True Rev':>8}  {'EpsG':>6}  {'UCB':>6}  {'Thomp':>6}")
    print("  " + "-" * 42)
    for i in range(K):
        marker = " *" if i == OPTIMAL_IDX else ""
        print(f"  ${PRICES[i]:>7.2f}  ${TRUE_REVENUES[i]:>7.2f}  "
              f"{int(counts_eg[i]):>5}  {int(counts_ucb[i]):>5}  {int(counts_ts[i]):>5}{marker}")
    print(f"\n  Optimal price = ${OPTIMAL_PRICE:.2f}  (conversion={CONVERSION_PROBS[OPTIMAL_IDX]:.0%})")
    for name, c in [("Eps-Greedy", counts_eg), ("UCB", counts_ucb), ("Thompson", counts_ts)]:
        chosen = PRICES[np.argmax(c)]
        print(f"  {name:>12} most-chosen price = ${chosen:.2f}"
              f"  {'optimal' if chosen == OPTIMAL_PRICE else 'suboptimal'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Exp 26 – Dynamic Pricing Bandit Comparison")
    print("=" * 60)
    print(f"  K={K}  rounds={ROUNDS}  prices={PRICES}")
    print(f"  conversion probs={CONVERSION_PROBS}")
    print(f"  true revenues={TRUE_REVENUES.round(2)}")

    r_eg, reg_eg, c_eg = epsilon_greedy(PRICES, CONVERSION_PROBS, ROUNDS, eps=0.1, seed=SEED)
    r_ucb, reg_ucb, c_ucb = ucb(PRICES, CONVERSION_PROBS, ROUNDS, seed=SEED)
    r_ts, reg_ts, c_ts = thompson_sampling(PRICES, CONVERSION_PROBS, ROUNDS, seed=SEED)

    print(f"\n  Final cumulative revenue:")
    print(f"    Epsilon-Greedy : ${r_eg[-1]:.2f}")
    print(f"    UCB            : ${r_ucb[-1]:.2f}")
    print(f"    Thompson       : ${r_ts[-1]:.2f}")

    print_table(c_eg, c_ucb, c_ts)
    plot_results(r_eg, r_ucb, r_ts, reg_eg, reg_ucb, reg_ts)
    print("\nDone.")

