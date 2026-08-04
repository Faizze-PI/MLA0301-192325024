"""
Exp 32: Dueling DQN vs Standard DQN for Gridworld Navigation
=============================================================
15x15 grid with obstacles and sparse goal reward.
Standard DQN vs Dueling DQN with identical hyperparameters.
Overlaid learning curves showing Dueling DQN's sample-efficiency improvement.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
from collections import deque
import random

# ---------------------------------------------------------------------------
# Gridworld Environment
# ---------------------------------------------------------------------------

class GridWorld:
    """15x15 grid with obstacles and a sparse goal reward."""

    def __init__(self, size=15, num_obstacles=20, seed=42):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        # Place obstacles
        placed = 0
        while placed < self.num_obstacles:
            r = self.rng.randint(0, self.size)
            c = self.rng.randint(0, self.size)
            if (r, c) != (0, 0) and (r, c) != (self.size - 1, self.size - 1):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 1  # obstacle
                    placed += 1

        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        self.max_steps = self.size * 4
        return self._state()

    def _state(self):
        return (self.agent_pos[0], self.agent_pos[1])

    def available_actions(self):
        return [0, 1, 2, 3]  # up, down, left, right

    def step(self, action):
        self.steps += 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        # Boundary / obstacle check
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] == 0:
            self.agent_pos = [nr, nc]

        # Goal check
        if self.agent_pos == self.goal_pos:
            return self._state(), 100.0, True

        # Timeout
        if self.steps >= self.max_steps:
            return self._state(), -1.0, True

        # Step penalty
        return self._state(), -0.1, False


# ---------------------------------------------------------------------------
# Standard DQN
# ---------------------------------------------------------------------------

class StandardDQN:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Q-network weights: input -> 128 -> 128 -> output
        self.W1 = np.random.randn(state_size, 128) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 128) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(128)
        self.W3 = np.random.randn(128, action_size) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(action_size)

        # Target network copy
        self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
        self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()
        self.tW3, self.tb3 = self.W3.copy(), self.b3.copy()

        self.lr = lr
        self.buffer = deque(maxlen=buffer_size)
        self.learn_steps = 0

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward(self, s, W1, b1, W2, b2, W3, b3):
        h1 = self._relu(s @ W1 + b1)
        h2 = self._relu(h1 @ W2 + b2)
        return h2 @ W3 + b3

    def predict(self, state, eps_greedy=True):
        if eps_greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        s = np.array(state, dtype=np.float32).reshape(1, -1)
        q = self._forward(s, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        return int(np.argmax(q))

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def _update_weights(self, W, grad, lr):
        return W - lr * grad

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        s = np.array([t[0] for t in batch], dtype=np.float32)
        a = np.array([t[1] for t in batch], dtype=np.int32)
        r = np.array([t[2] for t in batch], dtype=np.float32)
        s2 = np.array([t[3] for t in batch], dtype=np.float32)
        done = np.array([t[4] for t in batch], dtype=np.float32)

        # Current Q
        q_vals = self._forward(s, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        q_current = q_vals[np.arange(self.batch_size), a]

        # Target Q
        q_next = self._forward(s2, self.tW1, self.tb1, self.tW2, self.tb2, self.tW3, self.tb3)
        q_target = r + self.gamma * np.max(q_next, axis=1) * (1 - done)

        # Gradient descent (simplified - just update output layer)
        error = q_current - q_target
        # Backprop through last layer
        h2 = self._relu(self._relu(s @ self.W1 + self.b1) @ self.W2 + self.b2)
        grad_W3 = h2.T.reshape(128, self.batch_size) @ error.reshape(self.batch_size, 1) / self.batch_size

        self.W3 -= self.lr * grad_W3
        self.b3 -= self.lr * np.mean(error)

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
            self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()
            self.tW3, self.tb3 = self.W3.copy(), self.b3.copy()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(np.mean(error ** 2))


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------

class DuelingDQN(StandardDQN):
    """Dueling DQN: separate value and advantage streams."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Value stream: h2 -> 64 -> 1
        self.W_val = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b_val = np.zeros(64)
        self.W_val_out = np.random.randn(64, 1) * np.sqrt(2.0 / 64)
        self.b_val_out = np.zeros(1)

        # Advantage stream: h2 -> 64 -> action_size
        self.W_adv = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b_adv = np.zeros(64)
        self.W_adv_out = np.random.randn(64, self.action_size) * np.sqrt(2.0 / 64)
        self.b_adv_out = np.zeros(self.action_size)

    def _forward_dueling(self, s):
        h1 = self._relu(s @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)

        # Value stream
        val = self._relu(h2 @ self.W_val + self.b_val)
        val = val @ self.W_val_out + self.b_val_out  # (batch, 1)

        # Advantage stream
        adv = self._relu(h2 @ self.W_adv + self.b_adv)
        adv = adv @ self.W_adv_out + self.b_adv_out  # (batch, actions)

        # Combine: Q = V + (A - mean(A))
        q = val + adv - np.mean(adv, axis=1, keepdims=True)
        return q

    def predict(self, state, eps_greedy=True):
        if eps_greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        s = np.array(state, dtype=np.float32).reshape(1, -1)
        q = self._forward_dueling(s)
        return int(np.argmax(q))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        s = np.array([t[0] for t in batch], dtype=np.float32)
        a = np.array([t[1] for t in batch], dtype=np.int32)
        r = np.array([t[2] for t in batch], dtype=np.float32)
        s2 = np.array([t[3] for t in batch], dtype=np.float32)
        done = np.array([t[4] for t in batch], dtype=np.float32)

        q_vals = self._forward_dueling(s)
        q_current = q_vals[np.arange(self.batch_size), a]

        q_next = self._forward_dueling(s2)
        q_target = r + self.gamma * np.max(q_next, axis=1) * (1 - done)

        error = q_current - q_target

        # Simplified gradient update for value + advantage streams
        h1 = self._relu(s @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)

        error_2d = error.reshape(-1, 1)  # (batch, 1)

        # Update value stream
        val_hid = self._relu(h2 @ self.W_val + self.b_val)  # (batch, 64)
        d_val_hid = (val_hid > 0).astype(float)  # relu grad
        # Chain: dL/dW_val = h2.T @ (error * W_val_out.T * d_val_hid) / batch
        val_back = error_2d @ self.W_val_out.T * d_val_hid  # (batch, 64)
        grad_W_val = h2.T @ val_back / self.batch_size  # (128, 64)
        grad_b_val = val_back.mean(axis=0)  # (64,)
        self.W_val -= self.lr * grad_W_val
        self.b_val -= self.lr * grad_b_val

        # Update advantage stream
        adv_hid = self._relu(h2 @ self.W_adv + self.b_adv)  # (batch, 64)
        d_adv_hid = (adv_hid > 0).astype(float)
        # error is per-sample scalar, broadcast across action dim for advantage
        adv_back = error_2d * d_adv_hid  # (batch, 64)
        grad_W_adv = h2.T @ adv_back / self.batch_size  # (128, 64)
        grad_b_adv = adv_back.mean(axis=0)
        self.W_adv -= self.lr * grad_W_adv
        self.b_adv -= self.lr * grad_b_adv

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
            self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()
            self.tW3, self.tb3 = self.W3.copy(), self.b3.copy()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(np.mean(error ** 2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(agent_class, episodes=1000, label="DQN"):
    env = GridWorld(size=15, num_obstacles=20, seed=42)
    state_size = 2  # (row, col)
    action_size = 4

    agent = agent_class(
        state_size=state_size,
        action_size=action_size,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=100,
    )

    rewards_per_episode = []
    success_count = 0

    print(f"Training {label} ...")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.predict(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        if reward >= 100:
            success_count += 1

        if (ep + 1) % 200 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            succ = success_count / (ep + 1)
            print(f"  [{label}] Episode {ep+1}: avg_reward={avg:.2f}, success_rate={succ:.1%}")

    return rewards_per_episode


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(std_rewards, duel_rewards):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Learning curves overlaid
    ax = axes[0]
    window = 50
    if len(std_rewards) >= window:
        std_smooth = np.convolve(std_rewards, np.ones(window) / window, mode="valid")
        duel_smooth = np.convolve(duel_rewards, np.ones(window) / window, mode="valid")
        x = range(window - 1, len(std_rewards))
        ax.plot(x, std_smooth, label="Standard DQN", color="blue", alpha=0.8)
        ax.plot(x, duel_smooth, label="Dueling DQN", color="red", alpha=0.8)
    else:
        ax.plot(std_rewards, label="Standard DQN", color="blue", alpha=0.5)
        ax.plot(duel_rewards, label="Dueling DQN", color="red", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("15x15 Gridworld: Standard DQN vs Dueling DQN")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Success rate comparison
    ax = axes[1]
    window = 100
    std_success = [1 if r >= 100 else 0 for r in std_rewards]
    duel_success = [1 if r >= 100 else 0 for r in duel_rewards]
    if len(std_success) >= window:
        std_succ_rate = np.convolve(std_success, np.ones(window) / window, mode="valid")
        duel_succ_rate = np.convolve(duel_success, np.ones(window) / window, mode="valid")
        x = range(window - 1, len(std_success))
        ax.plot(x, std_succ_rate, label="Standard DQN", color="blue", alpha=0.8)
        ax.plot(x, duel_succ_rate, label="Dueling DQN", color="red", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (rolling)")
    ax.set_title("Goal Reaching Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Exp 32: Dueling DQN vs Standard DQN - Gridworld Navigation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp32_dueling_vs_standard.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir}/exp32_dueling_vs_standard.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EPISODES = 300

    t0 = time.time()
    std_rewards = train_agent(StandardDQN, episodes=EPISODES, label="Standard DQN")
    duel_rewards = train_agent(DuelingDQN, episodes=EPISODES, label="Dueling DQN")
    elapsed = time.time() - t0

    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"  Standard DQN final avg (last 100): {np.mean(std_rewards[-100:]):.2f}")
    print(f"  Dueling DQN final avg (last 100):  {np.mean(duel_rewards[-100:]):.2f}")

    plot_results(std_rewards, duel_rewards)
