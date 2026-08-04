"""
Experiment 20: POMDP for Search-and-Rescue Robot
==================================================
A disaster grid world where the robot has noisy/partial sensors.
The agent maintains a belief state updated via Bayesian filtering
and uses QMDP approximation for action selection.

Environment:
  12x12 grid with rubble, fire zones, survivors (targets).
  Sensors give noisy distance readings to nearest survivor/fire.
  sensor_noise levels: 0.1 to 0.2

Key outputs:
  - Belief-state heatmap evolution over time
  - Success rate vs sensor noise level
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict

# --- Environment ---------------------------------------------------------------

GRID_ROWS = 12
GRID_COLS = 12
WALL = 1
OPEN = 0
FIRE = 2
RUBBLE = 3

class RescuePOMDP:
    """
    Partially observable MDP for search-and-rescue.
    Observations: noisy distance (dx, dy) to nearest target and nearest fire.
    """
    ACTIONS = 4  # up, down, left, right
    N_TARGETS = 3
    N_FIRES = 2

    def __init__(self, grid=None, sensor_noise=0.1, max_steps=100):
        self.grid = grid if grid is not None else self._default_grid()
        self.rows, self.cols = self.grid.shape
        self.sensor_noise = sensor_noise
        self.max_steps = max_steps
        self.agent_pos = [1, 1]
        self.targets_found = set()
        self.step_count = 0
        self._place_entities()

    def _default_grid(self):
        g = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        g[0, :] = WALL; g[-1, :] = WALL; g[:, 0] = WALL; g[:, -1] = WALL
        # Rubble patches
        g[3, 4] = RUBBLE; g[5, 7] = RUBBLE; g[7, 2] = RUBBLE; g[9, 9] = RUBBLE
        return g

    def _place_entities(self):
        rng = np.random.RandomState(123)
        self.targets = []
        self.fires = []
        for _ in range(self.N_TARGETS):
            while True:
                r, c = rng.randint(1, self.rows-1), rng.randint(1, self.cols-1)
                if self.grid[r, c] == OPEN and [r, c] != [1, 1]:
                    self.targets.append([r, c])
                    break
        for _ in range(self.N_FIRES):
            while True:
                r, c = rng.randint(1, self.rows-1), rng.randint(1, self.cols-1)
                if self.grid[r, c] == OPEN and [r, c] != [1, 1]:
                    self.fires.append([r, c])
                    self.grid[r, c] = FIRE
                    break

    def reset(self):
        self.agent_pos = [1, 1]
        self.targets_found = set()
        self.step_count = 0
        return self._obs()

    def _obs(self):
        """Noisy observation: distance to nearest target and nearest fire."""
        obs = np.zeros(4, dtype=np.float32)
        # Nearest target
        min_d = float('inf')
        best_t = self.targets[0]
        for i, t in enumerate(self.targets):
            d = abs(self.agent_pos[0] - t[0]) + abs(self.agent_pos[1] - t[1])
            if d < min_d:
                min_d = d
                best_t = t
        obs[0] = (self.agent_pos[0] - best_t[0]) / self.rows + np.random.randn() * self.sensor_noise
        obs[1] = (self.agent_pos[1] - best_t[1]) / self.cols + np.random.randn() * self.sensor_noise
        # Nearest fire
        min_df = float('inf')
        best_f = self.fires[0]
        for f in self.fires:
            df = abs(self.agent_pos[0] - f[0]) + abs(self.agent_pos[1] - f[1])
            if df < min_df:
                min_df = df
                best_f = f
        obs[2] = (self.agent_pos[0] - best_f[0]) / self.rows + np.random.randn() * self.sensor_noise
        obs[3] = (self.agent_pos[1] - best_f[1]) / self.cols + np.random.randn() * self.sensor_noise
        return obs

    def step(self, action):
        self.step_count += 1
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        nr = self.agent_pos[0] + dr[action]
        nc = self.agent_pos[1] + dc[action]
        if (0 < nr < self.rows-1 and 0 < nc < self.cols-1
                and self.grid[nr, nc] not in (WALL, RUBBLE)):
            self.agent_pos = [nr, nc]

        reward = -1.0
        done = False

        # Check if on target
        for i, t in enumerate(self.targets):
            if i not in self.targets_found and self.agent_pos == t:
                self.targets_found.add(i)
                reward = 15.0

        # Check if on fire
        if self.grid[self.agent_pos[0], self.agent_pos[1]] == FIRE:
            reward = -20.0
            done = True

        if len(self.targets_found) == self.N_TARGETS:
            reward += 50.0
            done = True
        if self.step_count >= self.max_steps:
            done = True

        return self._obs(), reward, done, {}


# --- Belief State (Bayesian Filtering) ----------------------------------------

class BeliefState:
    """
    Maintain a belief distribution over the grid.
    Uses Bayesian filtering with the transition model and observations.
    """
    def __init__(self, rows, cols, sensor_noise):
        self.rows = rows
        self.cols = cols
        self.sensor_noise = sensor_noise
        self.belief = np.ones((rows, cols)) / (rows * cols)
        self.belief[0, :] = 0
        self.belief[-1, :] = 0
        self.belief[:, 0] = 0
        self.belief[:, -1] = 0
        self.belief = self.belief / self.belief.sum()

    def update(self, action, observation, grid):
        """Bayesian filter update: predict -> observe."""
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        # Predict step: shift belief according to action
        new_belief = np.zeros_like(self.belief)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.belief[r, c] > 0:
                    nr, nc = r + dr[action], c + dc[action]
                    if (0 < nr < self.rows-1 and 0 < nc < self.cols-1
                            and grid[nr, nc] not in (1, 3)):  # not wall/rubble
                        new_belief[nr, nc] += self.belief[r, c]
                    else:
                        new_belief[r, c] += self.belief[r, c]  # stay
        new_belief = new_belief / (new_belief.sum() + 1e-10)

        # Observation update (simplified: high observation likelihood near measured position)
        obs_x = observation[0] * self.rows
        obs_y = observation[1] * self.cols
        sigma = self.sensor_noise * self.rows
        for r in range(self.rows):
            for c in range(self.cols):
                dist = (r - obs_x)**2 + (c - obs_y)**2
                likelihood = np.exp(-dist / (2 * sigma**2 + 1e-10))
                new_belief[r, c] *= (likelihood + 0.01)

        new_belief[0, :] = 0
        new_belief[-1, :] = 0
        new_belief[:, 0] = 0
        new_belief[:, -1] = 0
        self.belief = new_belief / (new_belief.sum() + 1e-10)

    def get_best_action(self, grid, targets):
        """QMDP-style greedy action: move toward highest belief target area."""
        best_act = 0
        best_val = -float('inf')
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        for a in range(4):
            val = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.belief[r, c] > 0.01:
                        nr, nc = r + dr[a], c + dc[a]
                        if 0 < nr < self.rows-1 and 0 < nc < self.cols-1 and grid[nr, nc] not in (1, 3):
                            val += self.belief[nr, nc] * (1.0 / (1 + r + c))
            if val > best_val:
                best_val = val
                best_act = a
        return best_act

    def get_map(self):
        return self.belief.copy()


# --- QMDP Agent ----------------------------------------------------------------

class QMDPAgent:
    """Simple QMDP: maintain belief + Q-learning on believed states."""
    def __init__(self, rows, cols, n_actions=4, lr=0.1, gamma=0.95, eps=0.2):
        self.q = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        self.n_actions = n_actions

    def select_action(self, belief):
        state = self._discretize(belief)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[state]))

    def _discretize(self, belief):
        top_k = np.argsort(belief.ravel())[-5:]
        return tuple(sorted(top_k))

    def update(self, belief, action, reward, next_belief, done):
        s = self._discretize(belief)
        ns = self._discretize(next_belief)
        target = reward + (0 if done else self.gamma * np.max(self.q[ns]))
        self.q[s][action] += self.lr * (target - self.q[s][action])


# --- Training & Evaluation -----------------------------------------------------

def run_episode(env, agent, belief, training=True):
    obs = env.reset()
    belief_state = BeliefState(env.rows, env.cols, env.sensor_noise)
    total_reward = 0
    done = False
    maps = []

    while not done:
        if training:
            action = agent.select_action(belief_state.get_map())
        else:
            action = belief_state.get_best_action(env.grid, env.targets)

        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        belief_state.update(action, next_obs, env.grid)

        if training:
            agent.update(belief_state.get_map(), action, reward, belief_state.get_map(), done)

        maps.append(belief_state.get_map())

    return total_reward, maps, len(env.targets_found)


def main():
    print("=" * 70)
    print("Experiment 20: POMDP for Search-and-Rescue Robot")
    print("=" * 70)

    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    n_episodes = 150
    n_test = 30

    success_rates = []

    # Training at noise=0.1
    print("\nTraining QMDP agent (sensor_noise=0.1)...")
    env_train = RescuePOMDP(sensor_noise=0.1)
    agent = QMDPAgent(env_train.rows, env_train.cols)

    for ep in range(n_episodes):
        r, _, found = run_episode(env_train, agent, None, training=True)
        if (ep + 1) % 30 == 0:
            print(f"  Episode {ep+1:4d} | Reward: {r:6.1f} | Targets found: {found}")

    # Evaluate across noise levels
    print("\nEvaluating across sensor noise levels...")
    for noise in noise_levels:
        env_test = RescuePOMDP(sensor_noise=noise)
        successes = 0
        for t in range(n_test):
            r, maps, found = run_episode(env_test, agent, None, training=False)
            if found >= env_test.N_TARGETS:
                successes += 1
        rate = successes / n_test
        success_rates.append(rate)
        print(f"  Noise={noise:.2f} | Success rate: {rate:.2%}")

    # Belief heatmap evolution
    print("\nGenerating belief heatmap evolution...")
    env_viz = RescuePOMDP(sensor_noise=0.15)
    _, maps_viz, _ = run_episode(env_viz, agent, None, training=False)
    sample_steps = [0, len(maps_viz)//4, len(maps_viz)//2, len(maps_viz)-1]

    # --- Plots --------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Experiment 20: POMDP Search-and-Rescue Robot", fontsize=14, fontweight='bold')

    # Belief heatmaps (top row)
    for idx, step in enumerate(sample_steps[:3]):
        ax = axes[0, idx]
        im = ax.imshow(maps_viz[step], cmap='YlOrRd', interpolation='nearest')
        ax.set_title(f'Belief State at Step {step+1}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # More belief heatmaps (bottom row first 2)
    for idx, step in enumerate(sample_steps[3:]):
        ax = axes[1, idx]
        im = ax.imshow(maps_viz[step], cmap='YlOrRd', interpolation='nearest')
        ax.set_title(f'Belief State at Step {step+1}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Success rate plot
    ax = axes[1, 2]
    ax.plot(noise_levels, success_rates, 'o-', color='dodgerblue', linewidth=2, markersize=8)
    ax.set_xlabel('Sensor Noise Level')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs Sensor Noise')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    for i, (n, s) in enumerate(zip(noise_levels, success_rates)):
        ax.annotate(f'{s:.0%}', (n, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    plt.savefig(os.path.join(out_dir, "exp20_pomdp_rescue_results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: exp20_pomdp_rescue_results.png")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  Belief heatmaps show the robot's uncertainty distribution evolving over time.")
    print("  Success rate decreases as sensor noise increases.")
    for n, s in zip(noise_levels, success_rates):
        print(f"    Noise {n:.2f} -> Success Rate: {s:.0%}")

    plt.show()


if __name__ == "__main__":
    main()

