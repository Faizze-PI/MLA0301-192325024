"""
Exp 33: DDPG for Simplified RTS Game Agent
============================================
Small custom map with 1-2 resource types, few unit types.
Continuous movement/attack actions.
Simple scripted opponent.

DDPG with actor-critic, target networks, soft updates (tau=0.005).
Replay buffer size: 50000.
Reward curve + win rate vs scripted opponent.
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
# Simplified RTS Environment
# ---------------------------------------------------------------------------

class SimpleRTSEnv:
    """
    Minimal RTS environment:
    - 20x20 grid
    - Agent has 1 base, 1 worker, 1 soldier
    - Opponent (scripted) has 1 base, 1 worker, 1 soldier
    - Resources: gold scattered on map
    - Agent actions: continuous (dx, dy for soldier, collect/deploy for worker)
    - Win: destroy opponent base; Lose: own base destroyed
    """

    MAP_SIZE = 20
    MAX_STEPS = 200

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        # Agent entities: base(0), worker(1), soldier(2)
        self.agent_base = np.array([2, 2], dtype=np.float32)
        self.agent_worker = np.array([3, 1], dtype=np.float32)
        self.agent_soldier = np.array([1, 3], dtype=np.float32)
        self.agent_gold = 0.0

        # Opponent entities
        self.opp_base = np.array([17, 17], dtype=np.float32)
        self.opp_worker = np.array([16, 18], dtype=np.float32)
        self.opp_soldier = np.array([18, 16], dtype=np.float32)
        self.opp_gold = 0.0

        # Resource nodes
        self.resources = self.rng.uniform(5, 15, (6, 2)).astype(np.float32)

        self.steps = 0
        self.done = False
        return self._state()

    def _state(self):
        """Flatten all entity positions + gold + resource positions into state vector."""
        state = np.concatenate([
            self.agent_base / self.MAP_SIZE,
            self.agent_worker / self.MAP_SIZE,
            self.agent_soldier / self.MAP_SIZE,
            [self.agent_gold / 100.0],
            self.opp_base / self.MAP_SIZE,
            self.opp_worker / self.MAP_SIZE,
            self.opp_soldier / self.MAP_SIZE,
            [self.opp_gold / 100.0],
            self.resources.flatten() / self.MAP_SIZE,
        ])
        return state.astype(np.float32)

    @property
    def state_size(self):
        return len(self._state())

    @property
    def action_size(self):
        return 4  # soldier_dx, soldier_dy, worker_dx, worker_dy (continuous -1 to 1)

    def _clip(self, pos):
        return np.clip(pos, 0, self.MAP_SIZE - 1)

    def _move_towards(self, current, target, speed=1.0):
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist < speed:
            return target.copy()
        return current + (diff / dist) * speed

    def _scripted_opponent(self):
        """Simple scripted opponent: worker collects, soldier attacks."""
        # Worker goes to nearest resource
        dists = np.linalg.norm(self.resources - self.opp_worker, axis=1)
        nearest = self.resources[np.argmin(dists)]
        self.opp_worker = self._clip(self._move_towards(self.opp_worker, nearest, 0.8))

        # Collect gold if near resource
        for i, res in enumerate(self.resources):
            if np.linalg.norm(self.opp_worker - res) < 1.5:
                self.opp_gold += 10

        # Soldier moves towards agent base
        self.opp_soldier = self._clip(self._move_towards(
            self.opp_soldier, self.agent_base, 0.5))

    def step(self, action):
        self.steps += 1
        action = np.clip(np.array(action, dtype=np.float32), -1, 1)

        # Agent soldier movement
        move = action[:2] * 2.0  # scale to 2 grid cells
        self.agent_soldier = self._clip(self.agent_soldier + move)

        # Agent worker movement
        worker_move = action[2:4] * 1.5
        self.agent_worker = self._clip(self.agent_worker + worker_move)

        # Agent worker collects resources
        for i, res in enumerate(self.resources):
            if np.linalg.norm(self.agent_worker - res) < 1.5:
                self.agent_gold += 10

        # Agent soldier attacks opponent entities
        reward = 0.0
        attack_range = 2.0
        attack_damage = 15.0

        if np.linalg.norm(self.agent_soldier - self.opp_worker) < attack_range:
            reward += 0.5  # reward for hitting
        if np.linalg.norm(self.agent_soldier - self.opp_soldier) < attack_range:
            reward += 0.3

        if np.linalg.norm(self.agent_soldier - self.opp_base) < attack_range:
            reward += 2.0

        # Opponent actions
        self._scripted_opponent()

        # Opponent soldier attacks
        if np.linalg.norm(self.opp_soldier - self.agent_base) < attack_range:
            reward -= 2.0

        # Check win/lose
        if np.linalg.norm(self.agent_soldier - self.opp_base) < 1.0:
            self.done = True
            reward += 50.0  # win
        elif np.linalg.norm(self.opp_soldier - self.agent_base) < 1.0:
            self.done = True
            reward -= 50.0  # lose

        # Step penalty
        reward -= 0.05

        # Distance reward (encourage approaching enemy)
        dist_to_enemy = np.linalg.norm(self.agent_soldier - self.opp_base)
        reward += max(0, (20 - dist_to_enemy) / 20) * 0.1

        if self.steps >= self.MAX_STEPS:
            self.done = True

        return self._state(), reward, self.done


# ---------------------------------------------------------------------------
# Simple Neural Network
# ---------------------------------------------------------------------------

class SimpleNN:
    def __init__(self, in_size, out_size, hidden=64, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(in_size, hidden) * np.sqrt(2.0 / in_size)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, out_size) * 0.01
        self.b3 = np.zeros(out_size)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def copy_from(self, other, tau=1.0):
        self.W1 = tau * other.W1 + (1 - tau) * self.W1
        self.b1 = tau * other.b1 + (1 - tau) * self.b1
        self.W2 = tau * other.W2 + (1 - tau) * self.W2
        self.b2 = tau * other.b2 + (1 - tau) * self.b2
        self.W3 = tau * other.W3 + (1 - tau) * self.W3
        self.b3 = tau * other.b3 + (1 - tau) * self.b3


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 tau=0.005, buffer_size=50000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor networks
        self.actor = SimpleNN(state_size, action_size, hidden=64, lr=lr)
        self.actor_target = SimpleNN(state_size, action_size, hidden=64, lr=lr)
        self.actor_target.copy_from(self.actor)

        # Critic networks
        self.critic = SimpleNN(state_size + action_size, 1, hidden=64, lr=lr)
        self.critic_target = SimpleNN(state_size + action_size, 1, hidden=64, lr=lr)
        self.critic_target.copy_from(self.critic)

        self.buffer = deque(maxlen=buffer_size)
        self.noise_std = 0.2

    def select_action(self, state, add_noise=True):
        s = np.array(state, dtype=np.float32).reshape(1, -1)
        action = self.actor.forward(s)
        action = np.tanh(action)  # bound to [-1, 1]
        if add_noise:
            noise = np.random.randn(*action.shape) * self.noise_std
            action = np.clip(action + noise, -1, 1)
        return action.flatten()

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s = np.array([t[0] for t in batch], dtype=np.float32)
        a = np.array([t[1] for t in batch], dtype=np.float32)
        r = np.array([t[2] for t in batch], dtype=np.float32).reshape(-1, 1)
        s2 = np.array([t[3] for t in batch], dtype=np.float32)
        done = np.array([t[4] for t in batch], dtype=np.float32).reshape(-1, 1)

        # Update critic: minimize (Q(s,a) - (r + gamma * Q_target(s2, a_target)))^2
        a2 = np.tanh(self.actor_target.forward(s2))
        q_target = r + self.gamma * self.critic_target.forward(np.concatenate([s2, a2], axis=1)) * (1 - done)
        q_vals = self.critic.forward(np.concatenate([s, a], axis=1))
        critic_loss = np.mean((q_vals - q_target.detach() if hasattr(q_target, 'detach') else q_target) ** 2)

        # Simplified critic update
        error = q_vals - q_target
        h1 = np.maximum(0, np.concatenate([s, a], axis=1) @ self.critic.W1 + self.critic.b1)
        h2 = np.maximum(0, h1 @ self.critic.W2 + self.critic.b2)
        grad_w3 = h2.T @ error / self.batch_size
        self.critic.W3 -= self.critic.lr * grad_w3
        self.critic.b3 -= self.critic.lr * np.mean(error, axis=0)

        # Update actor: maximize Q(s, actor(s))
        s_tensor = np.array(s, dtype=np.float32)
        actions_pred = np.tanh(self.actor.forward(s_tensor))
        q_pred = self.critic.forward(np.concatenate([s_tensor, actions_pred], axis=1))
        actor_loss = -np.mean(q_pred)

        # Simplified actor update
        grad_action = -1.0 * np.sign(q_pred)  # push action to increase Q
        h1a = np.maximum(0, s_tensor @ self.actor.W1 + self.actor.b1)
        h2a = np.maximum(0, h1a @ self.actor.W2 + self.actor.b2)
        grad_w3a = h2a.T @ grad_action / self.batch_size
        self.actor.W3 -= self.actor.lr * grad_w3a
        self.actor.b3 -= self.actor.lr * np.mean(grad_action, axis=0)

        # Soft update targets
        self.actor_target.copy_from(self.actor, self.tau)
        self.critic_target.copy_from(self.critic, self.tau)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(episodes=300):
    print(f"Training DDPG for {episodes} episodes ...")
    print("  tau=0.005, buffer=50000, gamma=0.99")

    env = SimpleRTSEnv(seed=42)
    agent = DDPGAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=0.001,
        gamma=0.99,
        tau=0.005,
        buffer_size=50000,
        batch_size=64,
    )

    rewards_per_episode = []
    win_count = 0

    t0 = time.time()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.learn()
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Check win condition
        if total_reward > 0:
            win_count += 1

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(rewards_per_episode[-50:])
            wr = win_count / (ep + 1)
            print(f"  Episode {ep+1}: avg_reward={avg_r:.2f}, win_rate={wr:.1%}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    return agent, rewards_per_episode, win_count / episodes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent, num_episodes=50):
    print(f"\nEvaluating over {num_episodes} episodes ...")
    env = SimpleRTSEnv(seed=999)
    wins, losses = 0, 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state, add_noise=False)
            state, reward, done = env.step(action)
            total_reward += reward

        if total_reward > 0:
            wins += 1
        else:
            losses += 1

    print(f"  Wins: {wins}/{num_episodes} ({wins/num_episodes:.1%})")
    print(f"  Losses: {losses}/{num_episodes}")
    return wins, losses


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(rewards, eval_wins, eval_losses):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Reward curve
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color="blue")
    window = min(30, len(rewards) // 3)
    if window > 1:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smooth, color="blue", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DDPG Training: Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Win rate
    ax = axes[1]
    wins_per = []
    window = 30
    for i in range(len(rewards)):
        start = max(0, i - window)
        wins_per.append(np.mean([1 if r > 0 else 0 for r in rewards[start:i + 1]]))
    ax.plot(wins_per, color="green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (rolling)")
    ax.set_title("DDPG Training: Win Rate vs Scripted Opponent")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 33: DDPG for Simplified RTS Game Agent", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp33_ddpg_rts.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir}/exp33_ddpg_rts.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent, rewards, train_win_rate = train(episodes=300)
    eval_wins, eval_losses = evaluate(agent, num_episodes=50)

    print(f"\nFinal Results:")
    print(f"  Training win rate: {train_win_rate:.1%}")
    print(f"  Evaluation win rate: {eval_wins / (eval_wins + eval_losses):.1%}")

    plot_results(rewards, eval_wins, eval_losses)
