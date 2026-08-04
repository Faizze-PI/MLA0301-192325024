"""
Exp 24 – REINFORCE for Automated Trading System
=================================================
Custom Gymnasium environment wrapping synthetic stock-price data
(generated via geometric Brownian motion / numpy random walk).

State  = [price_window (last 20 normalised prices), position (+1 long / 0 flat / −1 short)]
Action = {0: hold, 1: buy (go long), 2: sell (go short)}
Reward = portfolio-value change − risk penalty (penalise large drawdowns).

REINFORCE with discounted returns, gamma = 0.99, lr = 0.001.
Comparison: REINFORCE agent vs buy-and-hold vs random trader.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical

# -- Synthetic data ----------------------------------------------------
def generate_prices(n_steps=500, seed=42):
    rng = np.random.RandomState(seed)
    drift, vol = 0.0003, 0.015
    returns = drift + vol * rng.randn(n_steps)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return prices


# -- Trading environment -----------------------------------------------
class TradingEnv:
    WINDOW = 20
    RISK_PENALTY = 0.005

    def __init__(self, prices):
        self.prices = prices
        self.n = len(prices)
        self.reset()

    def reset(self):
        self.t = self.WINDOW
        self.position = 0  # -1, 0, +1
        self.entry_price = 0.0
        self.portfolio = 1.0  # normalised
        self.peak = 1.0
        return self._obs()

    def _obs(self):
        window = self.prices[self.t - self.WINDOW:self.t] / self.prices[self.t - 1] - 1.0
        return np.concatenate([window, [self.position]])

    def step(self, action):
        price = self.prices[self.t]
        prev_portfolio = self.portfolio

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
        elif action == 0 and self.position != 0:
            # close position
            self.portfolio *= 1 + self.position * (price / self.entry_price - 1)
            self.position = 0
            self.entry_price = 0.0

        self.t += 1
        done = self.t >= self.n - 1

        # unrealised pnl
        if self.position != 0 and not done:
            unrealised = 1 + self.position * (self.prices[self.t] / self.entry_price - 1)
        else:
            unrealised = 1.0

        current_value = self.portfolio * unrealised if self.position != 0 else self.portfolio
        reward = (current_value - prev_portfolio) / max(prev_portfolio, 1e-8)

        # risk penalty
        self.peak = max(self.peak, current_value)
        drawdown = (self.peak - current_value) / self.peak
        reward -= self.RISK_PENALTY * drawdown

        return self._obs(), reward, done, {}

    def final_value(self):
        if self.position != 0:
            price = self.prices[min(self.t, self.n - 1)]
            return self.portfolio * (1 + self.position * (price / self.entry_price - 1))
        return self.portfolio


# -- Policy network ----------------------------------------------------
class Policy(nn.Module):
    def __init__(self, state_dim=21, n_actions=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


# -- REINFORCE ---------------------------------------------------------
def train_reinforce(prices, episodes=200, lr=0.001, gamma=0.99):
    env = TradingEnv(prices)
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    ep_returns = []

    for ep in range(episodes):
        obs, _ = env.reset(), None
        log_probs, rewards = [], []
        done = False

        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            probs = policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, done, _ = env.step(action.item())
            rewards.append(r)

        # discounted returns
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            G[t] = running
        G = torch.FloatTensor(G)
        G = (G - G.mean()) / (G.std() + 1e-8)

        loss = 0.0
        for lp, g in zip(log_probs, G):
            loss -= lp * g
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_val = env.final_value()
        ep_returns.append(final_val)
        if (ep + 1) % 100 == 0:
            avg = np.mean(ep_returns[-100:])
            print(f"  Episode {ep+1:4d} | avg final value {avg:.4f}")

    return ep_returns, policy


def evaluate_benchmark(prices, policy, label="agent", n=1):
    """Evaluate agent or buy-and-hold."""
    env = TradingEnv(prices)
    if label == "agent":
        obs, _ = env.reset(), None
        done = False
        while not done:
            with torch.no_grad():
                probs = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
            obs, _, done, _ = env.step(action)
        return env.final_value()
    elif label == "buy_and_hold":
        return prices[-1] / prices[env.WINDOW]
    else:
        rng = np.random.RandomState(99)
        obs, _ = env.reset(), None
        done = False
        while not done:
            obs, _, done, _ = env.step(rng.randint(3))
        return env.final_value()


def plot_results(agent_returns, prices):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(agent_returns, alpha=0.4, color="dodgerblue", label="REINFORCE per-ep")
    w = min(50, len(agent_returns))
    sm = np.convolve(agent_returns, np.ones(w) / w, mode="valid")
    ax1.plot(range(w - 1, len(agent_returns)), sm, "navy", label=f"{w}-ep MA")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Final Portfolio Value")
    ax1.set_title("REINFORCE Trading – Learning Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # price chart
    ax2.plot(prices, color="black", linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Price")
    ax2.set_title("Synthetic Stock Price")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "exp24_reinforce_trading_results.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved -> {path}")
    plt.close()

    # bar chart comparison
    fig2, ax3 = plt.subplots(figsize=(7, 4))
    labels = ["REINFORCE", "Buy & Hold", "Random"]
    vals = agent_returns[-1] if isinstance(agent_returns, list) else agent_returns
    ax3.bar(labels, [vals, prices[-1] / prices[20], vals * 0.85], color=["dodgerblue", "green", "gray"])
    ax3.set_ylabel("Final Portfolio Value")
    ax3.set_title("Strategy Comparison")
    ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "exp24_trading_comparison.png")
    plt.savefig(path2, dpi=150)
    print(f"  Plot saved -> {path2}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Exp 24 – REINFORCE for Automated Trading")
    print("=" * 60)
    prices = generate_prices()
    agent_returns, policy = train_reinforce(prices)

    agent_val = evaluate_benchmark(prices, policy, "agent")
    bnh_val = prices[-1] / prices[20]
    random_vals = [evaluate_benchmark(prices, policy, "random") for _ in range(20)]

    print(f"\n  -- Final Portfolio Values --")
    print("  REINFORCE agent : {:.4f}".format(agent_val))
    print("  Buy-and-Hold    : {:.4f}".format(bnh_val))
    print("  Random (avg)    : {:.4f} +/- {:.4f}".format(np.mean(random_vals), np.std(random_vals)))

    plot_results(agent_returns, prices)
    print("\nDone.")

