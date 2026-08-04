"""
Experiment 17: Hierarchical Reinforcement Learning (HRL) for Autonomous Household Robot
=========================================================================================
Uses MAXQ decomposition to solve a 3-room grid-world with sub-goals:
vacuum (clean dirty spots), dust (dispose dust piles), empty-trash (empty bins).

The hierarchy:
  Level 0 (flat): Standard Q-learning baseline
  Level 1 (HRL):  Option-based MAXQ with sub-policies:
       - Room navigation option
       - Vacuum task option
       - Dust removal option
       - Trash removal option

Environment layout (15x15 grid):
  Room 1 (top-left):    dust piles
  Room 2 (top-right):   trash bins
  Room 3 (bottom):      dirty floor patches
  Start position:       center hallway

gamma = 0.95
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict

# --- Environment ---------------------------------------------------------------

GRID_ROWS = 15
GRID_COLS = 15
WALL = 1
OPEN = 0

def make_grid():
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    # walls around perimeter
    g[0, :] = WALL
    g[-1, :] = WALL
    g[:, 0] = WALL
    g[:, -1] = WALL
    # walls to separate rooms (with doorways)
    g[1:7, 7] = WALL
    g[9:14, 7] = WALL
    g[7, 1:7] = WALL
    g[7, 9:14] = WALL
    return g

class HouseholdEnv:
    """
    Grid world with 3 rooms and task sites.
    Actions: 0=up, 1=down, 2=left, 3=right, 4=interact(vacuum/dust/trash)
    Sub-goals encoded as (row, col, task_type).
    """
    ACTIONS = 4
    TASK_VACUUM = 0
    TASK_DUST = 1
    TASK_TRASH = 2
    TASK_NAMES = ["vacuum", "dust", "empty-trash"]

    def __init__(self, grid=None, max_steps=200):
        self.grid = grid if grid is not None else make_grid()
        self.rows, self.cols = self.grid.shape
        self.agent_pos = [7, 7]  # center hallway
        self.task_sites = []
        self.completed = set()
        self.step_count = 0
        self.max_steps = max_steps
        self._place_tasks()
        self.state_space_size = self.rows * self.cols * (3**len(self.task_sites))
        self._assign_subgoals()

    def _place_tasks(self):
        """Place task sites in rooms."""
        self.task_sites = []
        # Room 1 (top-left): dust piles
        self.task_sites.append((2, 2, self.TASK_DUST))
        self.task_sites.append((4, 3, self.TASK_DUST))
        # Room 2 (top-right): trash bins
        self.task_sites.append((2, 12, self.TASK_TRASH))
        self.task_sites.append((4, 11, self.TASK_TRASH))
        # Room 3 (bottom): dirty floor
        self.task_sites.append((11, 3, self.TASK_VACUUM))
        self.task_sites.append((12, 10, self.TASK_VACUUM))
        self.task_sites.append((10, 6, self.TASK_VACUUM))

    def _assign_subgoals(self):
        """Assign sub-goals for HRL: each task site is a sub-goal."""
        self.subgoals = []
        for i, site in enumerate(self.task_sites):
            self.subgoals.append((site[0], site[1], site[2]))

    def reset(self):
        self.agent_pos = [7, 7]
        self.completed = set()
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return (self.agent_pos[0], self.agent_pos[1])

    def _get_task_mask(self):
        mask = []
        for i, s in enumerate(self.task_sites):
            mask.append(1 if i not in self.completed else 0)
        return tuple(mask)

    def _full_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self._get_task_mask())

    def step(self, action):
        self.step_count += 1
        r, c = self.agent_pos
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        nr, nc = r + dr[action], c + dc[action]
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] != WALL:
            self.agent_pos = [nr, nc]
        reward = -1  # step cost
        done = False
        # interaction
        if action == 4:
            for i, (tr, tc, ttype) in enumerate(self.task_sites):
                if i not in self.completed and self.agent_pos[0] == tr and self.agent_pos[1] == tc:
                    self.completed.add(i)
                    reward = 10
                    break
            else:
                reward = -5  # wasted action
        # goal check
        if len(self.completed) == len(self.task_sites):
            reward += 50
            done = True
        if self.step_count >= self.max_steps:
            done = True
        return self._get_state(), reward, done, {}

    def subgoal_reached(self, subgoal_idx):
        """Check if agent is at subgoal position."""
        sg = self.subgoals[subgoal_idx]
        return (self.agent_pos[0] == sg[0] and self.agent_pos[1] == sg[1]
                and subgoal_idx not in self.completed)


# --- Flat Q-Learning Baseline -------------------------------------------------

class FlatQLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95, eps=1.0, eps_decay=0.995, eps_min=0.01):
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_actions = n_actions

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward + (0 if done else self.gamma * np.max(self.q[next_state]))
        self.q[state][action] += self.lr * (target - self.q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# --- MAXQ Hierarchical Q-Learning ---------------------------------------------

class MAXQHRL:
    """
    Simplified MAXQ decomposition.
    Hierarchy:
      root -> [do_vacuum_task, do_dust_task, do_trash_task, navigate_room]
      each sub-policy has its own Q-table.
    Completion function V(c, s) estimates remaining cost from sub-task c.
    """
    def __init__(self, env, lr=0.1, gamma=0.95, eps=1.0, eps_decay=0.995, eps_min=0.01):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.n_actions = env.ACTIONS
        # One Q-table per sub-task + root
        # Sub-tasks: vacuum(0), dust(1), trash(2), navigate(3), root(4)
        self.task_q = {
            'root': defaultdict(lambda: np.zeros(self.n_actions)),
            'vacuum': defaultdict(lambda: np.zeros(self.n_actions)),
            'dust': defaultdict(lambda: np.zeros(self.n_actions)),
            'trash': defaultdict(lambda: np.zeros(self.n_actions)),
            'navigate': defaultdict(lambda: np.zeros(self.n_actions)),
        }
        # Completion function V_i(s): expected cost-to-go after completing sub-task i
        self.V = {
            'vacuum': defaultdict(float),
            'dust': defaultdict(float),
            'trash': defaultdict(float),
            'navigate': defaultdict(float),
        }
        # Sub-goals for each sub-task
        self.subtask_goals = {
            'vacuum': [i for i, s in enumerate(env.task_sites) if s[2] == env.TASK_VACUUM],
            'dust': [i for i, s in enumerate(env.task_sites) if s[2] == env.TASK_DUST],
            'trash': [i for i, s in enumerate(env.task_sites) if s[2] == env.TASK_TRASH],
        }

    def _get_relevant_task(self, state):
        """Determine which sub-task to execute based on remaining tasks."""
        remaining = []
        full = self.env._full_state()
        task_mask = full[2]
        for i, mask_val in enumerate(task_mask):
            if mask_val == 1:
                ttype = self.env.task_sites[i][2]
                if ttype == self.env.TASK_VACUUM:
                    remaining.append('vacuum')
                elif ttype == self.env.TASK_DUST:
                    remaining.append('dust')
                elif ttype == self.env.TASK_TRASH:
                    remaining.append('trash')
        if remaining:
            return remaining[0]
        return None

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        task = self._get_relevant_task(state)
        if task is None:
            return int(np.argmax(self.task_q['root'][state]))
        return int(np.argmax(self.task_q[task][state]))

    def update(self, state, action, reward, next_state, done):
        task = self._get_relevant_task(state)
        if task is None:
            q_table = self.task_q['root']
        else:
            q_table = self.task_q[task]

        target = reward
        if not done:
            # Check if current sub-task is complete
            next_task = self._get_relevant_task(next_state)
            if next_task != task:
                target += self.gamma * self.V.get(task, defaultdict(float))[next_state] if task in self.V else 0
                target += self.gamma * np.max(self.task_q['root'][next_state]) if next_task is None else self.gamma * np.max(self.task_q[next_task][next_state])
            else:
                target += self.gamma * np.max(q_table[next_state])

        q_table[state][action] += self.lr * (target - q_table[state][action])

        # Update completion function
        if task and task in self.V:
            self.V[task][state] = min(self.V[task][state], -reward) if self.V[task][state] != 0 else -reward

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# --- Training Loop ------------------------------------------------------------

def train_agent(agent, env, n_episodes, is_hrl=False):
    rewards_per_episode = []
    steps_per_episode = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            if is_hrl:
                action = agent.select_action(state)
            else:
                full_state = env._full_state()
                action = agent.select_action(full_state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            if is_hrl:
                agent.update(state, action, reward, next_state, done)
                state = next_state
            else:
                full_state_next = env._full_state()
                agent.update(full_state, action, reward, full_state_next, done)
            agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        if (ep + 1) % 50 == 0:
            avg_r = np.mean(rewards_per_episode[-50:])
            avg_s = np.mean(steps_per_episode[-50:])
            print(f"  Episode {ep+1:4d} | Avg Reward: {avg_r:7.1f} | Avg Steps: {avg_s:5.1f} | Eps: {agent.epsilon:.3f}")
    return rewards_per_episode, steps_per_episode


# --- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 17: Hierarchical RL (HRL) for Household Robot")
    print("=" * 70)

    n_episodes = 300
    env = HouseholdEnv()

    # --- Flat Q-Learning Baseline ---
    print("\n[1/2] Training Flat Q-Learning Baseline...")
    flat_agent = FlatQLearning(n_states=env.rows * env.cols, n_actions=env.ACTIONS)
    flat_rewards, flat_steps = train_agent(flat_agent, env, n_episodes, is_hrl=False)

    # --- MAXQ HRL ---
    print("\n[2/2] Training MAXQ Hierarchical RL Agent...")
    hrl_agent = MAXQHRL(env)
    hrl_rewards, hrl_steps = train_agent(hrl_agent, env, n_episodes, is_hrl=True)

    # --- Results ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    window = 20
    flat_avg = np.convolve(flat_rewards, np.ones(window)/window, mode='valid')
    hrl_avg = np.convolve(hrl_rewards, np.ones(window)/window, mode='valid')

    print(f"  Flat Q-Learning  -> Final avg reward (last 50 eps): {np.mean(flat_rewards[-50:]):.1f}")
    print(f"  MAXQ HRL         -> Final avg reward (last 50 eps): {np.mean(hrl_rewards[-50:]):.1f}")
    print(f"  Flat Q-Learning  -> Final avg steps  (last 50 eps): {np.mean(flat_steps[-50:]):.1f}")
    print(f"  MAXQ HRL         -> Final avg steps  (last 50 eps): {np.mean(hrl_steps[-50:]):.1f}")

    # --- Plots --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 17: Hierarchical RL (HRL) vs Flat Q-Learning", fontsize=14, fontweight='bold')

    axes[0].plot(flat_avg, label='Flat Q-Learning', color='tomato', linewidth=1.5)
    axes[0].plot(hrl_avg, label='MAXQ HRL', color='dodgerblue', linewidth=1.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Learning Curve: Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    flat_s_avg = np.convolve(flat_steps, np.ones(window)/window, mode='valid')
    hrl_s_avg = np.convolve(hrl_steps, np.ones(window)/window, mode='valid')
    axes[1].plot(flat_s_avg, label='Flat Q-Learning', color='tomato', linewidth=1.5)
    axes[1].plot(hrl_s_avg, label='MAXQ HRL', color='dodgerblue', linewidth=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Completion')
    axes[1].set_title('Learning Curve: Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    plt.savefig(os.path.join(out_dir, "exp17_hrl_results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: exp17_hrl_results.png")
    plt.show()


if __name__ == "__main__":
    main()

