"""
Experiment 19: Multi-Agent RL for Multi-Robot Warehouse System
================================================================
Implements a warehouse grid world with 2-4 robot agents using a
PettingZoo-style parallel API (simulated). Agents share pickup/dropoff
locations and must coordinate to minimize task completion time.

Approaches:
  1. Independent Q-Learning (each agent learns separately)
  2. Shared reward vs individual reward structures

Evaluation:
  - Task completion time vs episode for 2 vs 4 agents
  - Emergent coordination analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict

# --- Environment ---------------------------------------------------------------

GRID_ROWS = 10
GRID_COLS = 12
WALL = 1
OPEN = 0

def make_warehouse_grid():
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    g[0, :] = WALL; g[-1, :] = WALL; g[:, 0] = WALL; g[:, -1] = WALL
    # Shelf aisles (horizontal walls with gaps)
    for r in [2, 5, 8]:
        for c in range(1, GRID_COLS-1):
            if c not in [3, 9]:
                g[r, c] = WALL
    return g


class WarehouseMAEnv:
    """
    Multi-Agent warehouse environment.
    Agents: robots that pick up items at pickup zone and deliver to dropoff zone.
    Shared pickup at (1,1), shared dropoff at (8,10).
    Each agent has its own inventory flag.
    """
    ACTIONS = 4  # up, down, left, right
    INTERACT = 4  # virtual: pickup/dropoff (encoded as special action in joint space)

    def __init__(self, n_agents=2, grid=None, max_steps=150):
        self.n_agents = n_agents
        self.grid = grid if grid is not None else make_warehouse_grid()
        self.rows, self.cols = self.grid.shape
        self.pickup = (1, 1)
        self.dropoff = (8, 10)
        self.max_steps = max_steps
        self.agent_starts = []
        self._assign_starts()
        self.step_count = 0
        self.delivered = 0
        self.total_deliveries_needed = n_agents * 3  # each agent delivers 3 items
        self.agent_carrying = [False] * n_agents

    def _assign_starts(self):
        starts = [(1, 5), (8, 5), (4, 5), (6, 5)]
        self.agent_starts = [starts[i % len(starts)] for i in range(self.n_agents)]

    def reset(self):
        self.agent_positions = [list(s) for s in self.agent_starts]
        self.agent_carrying = [False] * self.n_agents
        self.delivered = 0
        self.step_count = 0
        self.delivered_per_agent = [0] * self.n_agents
        return self._get_observations()

    def _get_observations(self):
        obs = []
        for i in range(self.n_agents):
            feat = np.array([
                self.agent_positions[i][0] / self.rows,
                self.agent_positions[i][1] / self.cols,
                self.pickup[0] / self.rows,
                self.pickup[1] / self.cols,
                self.dropoff[0] / self.rows,
                self.dropoff[1] / self.cols,
                float(self.agent_carrying[i]),
                self.delivered / max(self.total_deliveries_needed, 1),
            ], dtype=np.float32)
            obs.append(feat)
        return obs

    def step(self, actions):
        """
        actions: list of ints, one per agent.
        Each agent can move (0-3) or attempt interact (4).
        """
        self.step_count += 1
        rewards = []
        dr = [-1, 1, 0, 0, 0]
        dc = [0, 0, -1, 1, 0]

        for i in range(self.n_agents):
            action = actions[i]
            if action < 4:
                # Movement
                nr = self.agent_positions[i][0] + dr[action]
                nc = self.agent_positions[i][1] + dc[action]
                if (0 < nr < self.rows-1 and 0 < nc < self.cols-1
                        and self.grid[nr, nc] != WALL):
                    # Collision with other agents?
                    collision = False
                    for j in range(self.n_agents):
                        if j != i and self.agent_positions[j] == [nr, nc]:
                            collision = True
                            break
                    if not collision:
                        self.agent_positions[i] = [nr, nc]
                rewards.append(-1.0)  # step cost
            else:
                # Interact
                pos = tuple(self.agent_positions[i])
                if not self.agent_carrying[i] and pos == self.pickup:
                    self.agent_carrying[i] = True
                    rewards.append(5.0)
                elif self.agent_carrying[i] and pos == self.dropoff:
                    self.agent_carrying[i] = False
                    self.delivered += 1
                    self.delivered_per_agent[i] += 1
                    rewards.append(20.0)
                else:
                    rewards.append(-5.0)  # invalid interact

        # Shared reward bonus if all agents delivered in this episode
        done = self.delivered >= self.total_deliveries_needed or self.step_count >= self.max_steps
        if done:
            for i in range(self.n_agents):
                rewards[i] += 30.0 if self.delivered >= self.total_deliveries_needed else -10.0

        return self._get_observations(), rewards, done, [{}] * self.n_agents


# --- Independent Q-Learning ----------------------------------------------------

class IndependentQLearning:
    def __init__(self, agent_id, obs_dim, n_actions, lr=0.15, gamma=0.95, eps=1.0, eps_decay=0.993, eps_min=0.05):
        self.agent_id = agent_id
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_actions = n_actions

    def discretize(self, obs):
        """Discretize continuous obs into hashable tuple."""
        return tuple(np.round(obs * 10).astype(int))

    def select_action(self, obs):
        state = self.discretize(obs)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[state]))

    def update(self, obs, action, reward, next_obs, done):
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)
        target = reward + (0 if done else self.gamma * np.max(self.q[next_state]))
        self.q[state][action] += self.lr * (target - self.q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# --- Training ------------------------------------------------------------------

def train_marl(n_agents, n_episodes=200, reward_type='individual'):
    """
    reward_type: 'individual' or 'shared'
    """
    env = WarehouseMAEnv(n_agents=n_agents)
    agents = [IndependentQLearning(i, obs_dim=8, n_actions=5) for i in range(n_agents)]

    completion_times = []
    total_deliveries_log = []

    for ep in range(n_episodes):
        obs_list = env.reset()
        done = False
        ep_steps = 0
        ep_rewards = [0.0] * n_agents

        while not done:
            actions = [agents[i].select_action(obs_list[i]) for i in range(n_agents)]
            next_obs_list, rewards, done, _ = env.step(actions)
            ep_steps += 1

            for i in range(n_agents):
                if reward_type == 'shared':
                    # Use average reward
                    shared_r = np.mean(rewards)
                    agents[i].update(obs_list[i], actions[i], shared_r, next_obs_list[i], done)
                else:
                    agents[i].update(obs_list[i], actions[i], rewards[i], next_obs_list[i], done)
                ep_rewards[i] += rewards[i]
                agents[i].decay_epsilon()

            obs_list = next_obs_list

        completion_times.append(ep_steps)
        total_deliveries_log.append(env.delivered)

        if (ep + 1) % 50 == 0:
            avg_time = np.mean(completion_times[-50:])
            avg_del = np.mean(total_deliveries_log[-50:])
            print(f"  Agents={n_agents} | Ep {ep+1:4d} | Avg Steps: {avg_time:5.1f} | "
                  f"Avg Delivered: {avg_del:.1f} | Eps: {agents[0].epsilon:.3f}")

    return completion_times, total_deliveries_log


# --- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 19: Multi-Agent RL for Warehouse Robot System")
    print("=" * 70)

    n_episodes = 250

    # --- 2 Agents, Individual Reward ---
    print("\n[1/4] Training: 2 agents, individual reward...")
    ct_2_ind, del_2_ind = train_marl(2, n_episodes, 'individual')

    # --- 2 Agents, Shared Reward ---
    print("\n[2/4] Training: 2 agents, shared reward...")
    ct_2_shr, del_2_shr = train_marl(2, n_episodes, 'shared')

    # --- 4 Agents, Individual Reward ---
    print("\n[3/4] Training: 4 agents, individual reward...")
    ct_4_ind, del_4_ind = train_marl(4, n_episodes, 'individual')

    # --- 4 Agents, Shared Reward ---
    print("\n[4/4] Training: 4 agents, shared reward...")
    ct_4_shr, del_4_shr = train_marl(4, n_episodes, 'shared')

    # --- Results ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    window = 20
    for name, ct in [("2 agents, individual", ct_2_ind), ("2 agents, shared", ct_2_shr),
                      ("4 agents, individual", ct_4_ind), ("4 agents, shared", ct_4_shr)]:
        avg = np.mean(ct[-50:])
        print(f"  {name:30s} -> Final avg completion time: {avg:5.1f} steps")

    # --- Plots --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 19: Multi-Agent RL Warehouse System", fontsize=14, fontweight='bold')

    # Plot 1: Completion time comparison
    def smooth(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    w = window
    axes[0].plot(smooth(ct_2_ind, w), label='2 Agents (Individual)', color='dodgerblue', linewidth=1.5)
    axes[0].plot(smooth(ct_2_shr, w), label='2 Agents (Shared)', color='dodgerblue', linestyle='--', linewidth=1.5)
    axes[0].plot(smooth(ct_4_ind, w), label='4 Agents (Individual)', color='tomato', linewidth=1.5)
    axes[0].plot(smooth(ct_4_shr, w), label='4 Agents (Shared)', color='tomato', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Task Completion Time (steps)')
    axes[0].set_title('Completion Time vs Episode')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Deliveries completed
    axes[1].plot(smooth(del_2_ind, w), label='2 Agents (Individual)', color='dodgerblue', linewidth=1.5)
    axes[1].plot(smooth(del_2_shr, w), label='2 Agents (Shared)', color='dodgerblue', linestyle='--', linewidth=1.5)
    axes[1].plot(smooth(del_4_ind, w), label='4 Agents (Individual)', color='tomato', linewidth=1.5)
    axes[1].plot(smooth(del_4_shr, w), label='4 Agents (Shared)', color='tomato', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Deliveries per Episode')
    axes[1].set_title('Task Completion Count')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    plt.savefig(os.path.join(out_dir, "exp19_marl_results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: exp19_marl_results.png")
    plt.show()


if __name__ == "__main__":
    main()

