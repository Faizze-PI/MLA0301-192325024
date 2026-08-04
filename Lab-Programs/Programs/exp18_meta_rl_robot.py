"""
Experiment 18: Meta-RL (RL^2) for Adaptive Industrial Robot
=============================================================
An RL2-style LSTM-based policy is trained across multiple variants of a
pick-and-place grid task. The agent learns to adapt quickly to new task
configurations using the recurrent hidden state as a running memory.

Task variants (5-10):
  - Different start/goal positions
  - Different obstacle layouts
  - Different object locations

Architecture:
  Input: (grid_features, one-hot action, reward, done) -> LSTM -> MLP -> action
  Truncated BPTT across meta-episodes within each task.

Evaluation:
  Compare meta-trained agent vs random-init agent on held-out task.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. Install with: pip install torch")

# --- Environment ---------------------------------------------------------------

GRID_ROWS = 8
GRID_COLS = 8
WALL = 1
OPEN = 0

class PickPlaceEnv:
    """
    Pick-and-place grid world variant.
    Agent must navigate to object, pick it up (interact at object), then
    carry it to goal location and place it.
    """
    ACTIONS = 4  # up, down, left, right
    ACTION_NAMES = ['up', 'down', 'left', 'right']

    def __init__(self, grid=None, start=(0,0), goal=(7,7), obj_pos=(3,3), obstacles=None, max_steps=80):
        self.grid = grid if grid is not None else self._default_grid(obstacles)
        self.rows, self.cols = self.grid.shape
        self.start = start
        self.goal = goal
        self.obj_pos = obj_pos
        self.has_object = False
        self.agent_pos = list(start)
        self.step_count = 0
        self.max_steps = max_steps

    def _default_grid(self, obstacles=None):
        g = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        g[0, :] = WALL; g[-1, :] = WALL; g[:, 0] = WALL; g[:, -1] = WALL
        if obstacles:
            for (r, c) in obstacles:
                if 0 < r < GRID_ROWS-1 and 0 < c < GRID_COLS-1:
                    g[r, c] = WALL
        return g

    def reset(self):
        self.agent_pos = list(self.start)
        self.has_object = False
        self.step_count = 0
        return self._obs()

    def _obs(self):
        """Return normalized feature vector."""
        features = np.array([
            self.agent_pos[0] / self.rows,
            self.agent_pos[1] / self.cols,
            self.goal[0] / self.rows,
            self.goal[1] / self.cols,
            self.obj_pos[0] / self.rows,
            self.obj_pos[1] / self.cols,
            float(self.has_object),
        ], dtype=np.float32)
        return features

    def step(self, action):
        self.step_count += 1
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        nr = self.agent_pos[0] + dr[action]
        nc = self.agent_pos[1] + dc[action]
        if 0 < nr < self.rows-1 and 0 < nc < self.cols-1 and self.grid[nr, nc] != WALL:
            self.agent_pos = [nr, nc]

        reward = -1.0
        done = False
        # Pick up object
        if (self.agent_pos[0] == self.obj_pos[0] and self.agent_pos[1] == self.obj_pos[1]
                and not self.has_object):
            self.has_object = True
            reward = 5.0
        # Place at goal
        elif (self.agent_pos[0] == self.goal[0] and self.agent_pos[1] == self.goal[1]
                and self.has_object):
            self.has_object = False
            reward = 20.0
            done = True
        # Collision into wall -> no move penalty
        if self.step_count >= self.max_steps:
            done = True
        return self._obs(), reward, done, {}


def generate_task_variants(n_tasks=5, seed=42):
    """Generate diverse task variants with different layouts."""
    rng = np.random.RandomState(seed)
    tasks = []
    for i in range(n_tasks):
        start = (rng.randint(1, GRID_ROWS-2), rng.randint(1, GRID_COLS-2))
        goal = (rng.randint(1, GRID_ROWS-2), rng.randint(1, GRID_COLS-2))
        while abs(goal[0]-start[0]) + abs(goal[1]-start[1]) < 3:
            goal = (rng.randint(1, GRID_ROWS-2), rng.randint(1, GRID_COLS-2))
        obj = (rng.randint(1, GRID_ROWS-2), rng.randint(1, GRID_COLS-2))
        while obj == start or obj == goal:
            obj = (rng.randint(1, GRID_ROWS-2), rng.randint(1, GRID_COLS-2))
        n_obs = rng.randint(1, 5)
        obs = []
        for _ in range(n_obs):
            r, c = rng.randint(1, GRID_ROWS-2), rng.randint(1, GRID_COLS-2)
            if (r, c) != start and (r, c) != goal and (r, c) != obj:
                obs.append((r, c))
        tasks.append({'start': start, 'goal': goal, 'obj_pos': obj, 'obstacles': obs})
    return tasks


# --- RL2 LSTM Policy ----------------------------------------------------------

if HAS_TORCH:
    class RL2LSTM(nn.Module):
        def __init__(self, obs_dim=7, n_actions=4, hidden_dim=64):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTMCell(obs_dim + n_actions + 1 + 1, hidden_dim)  # obs + prev_action_onehot + prev_reward + done
            self.fc1 = nn.Linear(hidden_dim, 64)
            self.fc2 = nn.Linear(64, n_actions)

        def init_hidden(self):
            return (torch.zeros(1, self.hidden_dim), torch.zeros(1, self.hidden_dim))

        def forward(self, obs, prev_action_onehot, prev_reward, prev_done, hidden):
            x = torch.cat([obs, prev_action_onehot, prev_reward.unsqueeze(0), prev_done.unsqueeze(0)], dim=1)
            hidden = self.lstm(x, hidden)
            h = F.relu(self.fc1(hidden[0]))
            logits = self.fc2(h)
            return logits, hidden


# --- Training ------------------------------------------------------------------

def train_meta_agent(tasks, n_meta_episodes=200, episodes_per_task=5, gamma=0.99, lr=1e-3):
    """Train RL2 agent across task variants using truncated BPTT."""
    model = RL2LSTM()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_returns = []

    for meta_ep in range(n_meta_episodes):
        # Sample a random task
        task_cfg = random.choice(tasks)
        env = PickPlaceEnv(
            start=task_cfg['start'], goal=task_cfg['goal'],
            obj_pos=task_cfg['obj_pos'], obstacles=task_cfg['obstacles']
        )
        hidden = model.init_hidden()
        prev_action_onehot = torch.zeros(1, 4)
        prev_reward = torch.zeros(1)
        prev_done = torch.zeros(1)
        total_reward = 0

        state = env.reset()
        ep_done = False
        log_probs = []
        rewards = []

        while not ep_done:
            obs = torch.FloatTensor(state).unsqueeze(0)
            logits, hidden = model(obs, prev_action_onehot, prev_reward, prev_done, hidden)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            rewards.append(reward)

            # Prepare next step inputs
            new_onehot = torch.zeros(1, 4)
            new_onehot[0, action.item()] = 1.0
            prev_action_onehot = new_onehot
            prev_reward = torch.FloatTensor([reward])
            prev_done = torch.FloatTensor([float(done)])

            state = next_state
            ep_done = done

        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for lp, R in zip(log_probs, returns):
            loss -= lp * R
        loss = loss / episodes_per_task

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_returns.append(total_reward)
        if (meta_ep + 1) % 20 == 0:
            avg = np.mean(all_returns[-20:])
            print(f"  Meta-Episode {meta_ep+1:4d} | Avg Return: {avg:7.1f}")

    return model, all_returns


def evaluate_agent(model, task_cfg, n_episodes=30):
    """Evaluate meta-trained agent on a specific task (zero-shot adaptation)."""
    env = PickPlaceEnv(
        start=task_cfg['start'], goal=task_cfg['goal'],
        obj_pos=task_cfg['obj_pos'], obstacles=task_cfg['obstacles']
    )
    returns = []
    for _ in range(n_episodes):
        hidden = model.init_hidden()
        prev_action_onehot = torch.zeros(1, 4)
        prev_reward = torch.zeros(1)
        prev_done = torch.zeros(1)
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            obs = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                logits, hidden = model(obs, prev_action_onehot, prev_reward, prev_done, hidden)
                probs = F.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1)
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            new_onehot = torch.zeros(1, 4)
            new_onehot[0, action.item()] = 1.0
            prev_action_onehot = new_onehot
            prev_reward = torch.FloatTensor([reward])
            prev_done = torch.FloatTensor([float(done)])
            state = next_state
        returns.append(total_reward)
    return returns


def evaluate_random_baseline(task_cfg, n_episodes=30):
    """Random action baseline for comparison."""
    env = PickPlaceEnv(
        start=task_cfg['start'], goal=task_cfg['goal'],
        obj_pos=task_cfg['obj_pos'], obstacles=task_cfg['obstacles']
    )
    returns = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = random.randint(0, 3)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return returns


# --- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 18: Meta-RL (RL2) for Adaptive Industrial Robot")
    print("=" * 70)

    if not HAS_TORCH:
        print("PyTorch is required. Please install: pip install torch")
        return

    n_tasks = 7
    tasks = generate_task_variants(n_tasks=n_tasks, seed=42)
    held_out_task = tasks[-1]
    train_tasks = tasks[:-1]

    print(f"\nGenerated {n_tasks} task variants.")
    print(f"  Training on tasks 1-{n_tasks-1}, held-out task: {n_tasks}")

    # Train meta-agent
    print("\n[1/3] Training RL2 Meta-Agent...")
    model, train_returns = train_meta_agent(train_tasks, n_meta_episodes=200, episodes_per_task=5)

    # Evaluate on held-out task
    print("\n[2/3] Evaluating meta-trained agent on held-out task...")
    meta_returns = evaluate_agent(model, held_out_task, n_episodes=30)

    print("\n[3/3] Evaluating random baseline on held-out task...")
    random_returns = evaluate_random_baseline(held_out_task, n_episodes=30)

    # --- Results ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Meta-RL avg return (held-out): {np.mean(meta_returns):7.1f} +/- {np.std(meta_returns):.1f}")
    print(f"  Random  avg return (held-out): {np.mean(random_returns):7.1f} +/- {np.std(random_returns):.1f}")
    print(f"  Improvement factor:            {np.mean(meta_returns)/max(np.mean(random_returns),0.1):.1f}x")

    # Adaptation speed comparison: track success rate over episodes
    meta_success = [1.0 if r > 10 else 0.0 for r in meta_returns]
    random_success = [1.0 if r > 10 else 0.0 for r in random_returns]
    meta_cum = np.cumsum(meta_success) / np.arange(1, len(meta_success)+1)
    random_cum = np.cumsum(random_success) / np.arange(1, len(random_success)+1)

    # --- Plots --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 18: Meta-RL (RL^2) for Adaptive Industrial Robot", fontsize=14, fontweight='bold')

    # Plot 1: Meta-training learning curve
    window = 10
    train_avg = np.convolve(train_returns, np.ones(window)/window, mode='valid')
    axes[0].plot(train_avg, color='dodgerblue', linewidth=1.5)
    axes[0].set_xlabel('Meta-Episode')
    axes[0].set_ylabel('Average Return')
    axes[0].set_title('Meta-Training Learning Curve')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Returns on held-out task
    axes[1].boxplot([meta_returns, random_returns], tick_labels=['Meta-RL (RL2)', 'Random Init'])
    axes[1].set_ylabel('Episode Return')
    axes[1].set_title('Held-Out Task: Return Distribution')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Adaptation speed (cumulative success rate)
    axes[2].plot(meta_cum, label='Meta-RL (RL2)', color='dodgerblue', linewidth=1.5)
    axes[2].plot(random_cum, label='Random Init', color='tomato', linewidth=1.5)
    axes[2].set_xlabel('Episode (on held-out task)')
    axes[2].set_ylabel('Cumulative Success Rate')
    axes[2].set_title('Adaptation Speed Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    plt.savefig(os.path.join(out_dir, "exp18_meta_rl_results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: exp18_meta_rl_results.png")
    plt.show()


if __name__ == "__main__":
    main()

