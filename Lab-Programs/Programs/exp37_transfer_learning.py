"""
Experiment 37: Transfer Learning in Reinforcement Learning
==========================================================
Demonstrate how an agent can transfer learned knowledge from a source task
to a related but different target task, accelerating learning.

Task Setup:
  - Source Task: 8x8 grid navigation with obstacles, goal at (7,7)
  - Target Task: 10x10 grid navigation with different obstacle layout, goal at (9,9)
  - Both share similar dynamics but different layouts

Approach:
  1. Train agent on source task until convergence (Q-learning)
  2. Transfer Q-table to target task (with/without adaptation)
  3. Compare:
     - Random init (no transfer) vs Transfer (Q-table copy)
     - Transfer with fine-tuning vs cold start
  4. Measure sample efficiency improvement

Algorithm: Q-learning with tabular representation
  alpha=0.1, gamma=0.95, epsilon decay 1.0->0.05

Deliverable:
  - Learning curve comparison: random init vs transfer vs transfer+finetune
  - Bar chart: episodes to reach threshold performance
  - Heatmap: source vs target Q-values visualization
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy


# ---------------------------------------------------------------------------
# Grid Environment
# ---------------------------------------------------------------------------

class GridNav:
    """Simple grid navigation with obstacles."""

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # U, D, L, R

    def __init__(self, size=8, obstacles=None, goal=(7, 7), seed=42):
        self.size = size
        self.goal = goal
        self.rng = np.random.RandomState(seed)
        if obstacles is None:
            n_obs = size * 2
            obstacles = set()
            while len(obstacles) < n_obs:
                o = (self.rng.randint(0, size), self.rng.randint(0, size))
                if o != (0, 0) and o != goal:
                    obstacles.add(o)
        self.obstacles = obstacles
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            self.state = (nr, nc)
        reward = 10.0 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self.state, reward, done


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QAgent:
    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.Q = {}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.get_q(state)))

    def update(self, state, action, reward, next_state, done):
        q = self.get_q(state)
        q_next = self.get_q(next_state)
        target = reward + (0 if done else self.gamma * np.max(q_next))
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def copy(self):
        new_agent = QAgent(self.n_actions, self.alpha, self.gamma,
                           self.epsilon, self.epsilon_min, self.epsilon_decay)
        new_agent.Q = copy.deepcopy(self.Q)
        return new_agent


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(agent, env, n_episodes=500, max_steps=100):
    rewards_per_ep = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        rewards_per_ep.append(total_reward)
    return rewards_per_ep


def evaluate_agent(agent, env, n_episodes=50, max_steps=100):
    rewards = []
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        total = 0
        for _ in range(max_steps):
            action = agent.act(state)
            state, r, done = env.env.step(action) if hasattr(env, 'env') else env.step(action)
            total += r
            if done:
                break
        rewards.append(total)
    agent.epsilon = old_eps
    return np.mean(rewards)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(rewards_random, rewards_transfer, rewards_finetune,
                 episodes_to_threshold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    window = 20

    # 1. Learning curves
    fig, ax = plt.subplots(figsize=(12, 6))
    ep_range = range(1, len(rewards_random) + 1)

    for rewards, label, color in [
        (rewards_random, 'Random Init', 'red'),
        (rewards_transfer, 'Transfer (Q-table copy)', 'blue'),
        (rewards_finetune, 'Transfer + Fine-tune', 'green'),
    ]:
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), smoothed,
                    color=color, linewidth=2, label=label)
            ax.plot(ep_range, rewards, alpha=0.15, color=color)

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Total Reward', fontsize=13)
    ax.set_title('Transfer Learning in RL: Source->Target Task\n'
                 '(Q-Learning, Grid Navigation)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp37_transfer_learning_curves.png'), dpi=150)
    plt.close(fig)
    print("[Saved] exp37_transfer_learning_curves.png")

    # 2. Episodes to threshold bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Random Init', 'Transfer', 'Transfer + Fine-tune']
    colors = ['red', 'blue', 'green']
    bars = ax.bar(methods, episodes_to_threshold, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, episodes_to_threshold):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episodes to Threshold', fontsize=13)
    ax.set_title('Sample Efficiency: Episodes to Reach 80% of Max Reward', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp37_episodes_to_threshold.png'), dpi=150)
    plt.close(fig)
    print("[Saved] exp37_episodes_to_threshold.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    save_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    print("=" * 60)
    print("Experiment 37: Transfer Learning in Reinforcement Learning")
    print("=" * 60)

    # Source task: 8x8 grid
    source_env = GridNav(size=8, goal=(7, 7), seed=42)
    # Target task: 10x10 grid with different obstacles
    target_env = GridNav(size=10, goal=(9, 9), seed=123)

    print(f"\nSource task: {source_env.size}x{source_env.size} grid, goal={source_env.goal}")
    print(f"Target task: {target_env.size}x{target_env.size} grid, goal={target_env.goal}")
    print(f"Source obstacles: {len(source_env.obstacles)}, Target obstacles: {len(target_env.obstacles)}")

    # Step 1: Train on source task
    print("\n[1] Training on source task...")
    source_agent = QAgent(epsilon_decay=0.995)
    rewards_source = train_agent(source_agent, source_env, n_episodes=500)
    print(f"    Source final avg reward (last 50): {np.mean(rewards_source[-50:]):.2f}")

    # Step 2: Random init (no transfer)
    print("\n[2] Training on target task (random init)...")
    random_agent = QAgent(epsilon_decay=0.995)
    rewards_random = train_agent(random_agent, target_env, n_episodes=500)
    print(f"    Random init final avg reward: {np.mean(rewards_random[-50:]):.2f}")

    # Step 3: Transfer Q-table directly
    print("\n[3] Training on target task (transfer Q-table)...")
    transfer_agent = source_agent.copy()
    transfer_agent.epsilon = 0.5  # start with some exploration for new layout
    rewards_transfer = train_agent(transfer_agent, target_env, n_episodes=500)
    print(f"    Transfer final avg reward: {np.mean(rewards_transfer[-50:]):.2f}")

    # Step 4: Transfer + fine-tune (lower learning rate)
    print("\n[4] Training on target task (transfer + fine-tune)...")
    finetune_agent = source_agent.copy()
    finetune_agent.alpha = 0.05  # lower LR for fine-tuning
    finetune_agent.epsilon = 0.3
    rewards_finetune = train_agent(finetune_agent, target_env, n_episodes=500)
    print(f"    Fine-tune final avg reward: {np.mean(rewards_finetune[-50:]):.2f}")

    # Compute episodes to threshold (80% of max possible reward)
    threshold = 0.8 * 10.0  # 80% of max reward (goal=10)
    def episodes_to_thresh(rewards, thresh):
        for i, r in enumerate(rewards):
            if r >= thresh:
                return i + 1
        return len(rewards)

    ep_random = episodes_to_thresh(rewards_random, threshold)
    ep_transfer = episodes_to_thresh(rewards_transfer, threshold)
    ep_finetune = episodes_to_thresh(rewards_finetune, threshold)

    # Results
    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'Episodes to Threshold':>22} {'Speedup':>10}")
    print(f"{'='*60}")
    print(f"{'Random Init':<25} {ep_random:>22} {'1.0x':>10}")
    print(f"{'Transfer':<25} {ep_transfer:>22} {ep_random/ep_transfer:>9.1f}x")
    print(f"{'Transfer + Fine-tune':<25} {ep_finetune:>22} {ep_random/ep_finetune:>9.1f}x")
    print(f"{'='*60}")

    # Plots
    print("\n[5] Generating plots...")
    plot_results(rewards_random, rewards_transfer, rewards_finetune,
                 [ep_random, ep_transfer, ep_finetune], save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

