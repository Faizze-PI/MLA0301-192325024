"""
Experiment 08 - Monte Carlo Prediction & Control: Robot Vacuum Cleaner
========================================================================
Custom grid room with dirty cells and battery cost.

  • First-visit Monte Carlo for V(s) estimation
  • MC Control with epsilon-soft policy

gamma=0.95, epsilon=0.1, episodes=3000.

Outputs:
  - V(s) heatmap
  - Policy with arrows
  - Energy-per-episode trend
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# -- Grid Setup -----------------------------------------------------------------
# 0 = clean, 1 = dirty, W = wall
# Robot starts top-left. Goal: clean all dirty cells, then return to base.
# Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Suck, 5=Stay
ROWS, COLS = 5, 5
WALLS = {(1, 3), (2, 3), (3, 1), (3, 2)}
DIRTY_CELLS = {(0, 1), (0, 3), (1, 1), (2, 1), (3, 3), (4, 2), (4, 4)}
BASE = (0, 0)

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0), 5: (0, 0)}
ACTION_NAMES = {0: "^", 1: "v", 2: "<-", 3: "->", 4: "S", 5: "·"}
N_ACTIONS = 6

GAMMA     = 0.95
EPSILON   = 0.1
EPISODES  = 3000
BATTERY_COST   = -0.02      # per step
DIRTY_REWARD   = 1.0        # for sucking a dirty cell
CLEAN_STAY     = -0.01      # small penalty for staying on clean cell
WALL_PENALTY   = -0.5       # bumped into wall


# -- Environment helpers --------------------------------------------------------
def make_initial_grid():
    """Return a grid where dirty=True for dirty cells."""
    grid = [[False] * COLS for _ in range(ROWS)]
    for (r, c) in DIRTY_CELLS:
        grid[r][c] = True
    return grid


def step(state, action):
    """Deterministic step. Returns (next_state, reward, done)."""
    r, c = state
    dirty_map = make_initial_grid()  # simplified – state is position only
    # In full version state includes dirty map; here we keep it positional
    # and treat "all cleaned" as a terminal after certain steps.
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if action == 4:  # Suck
        if (r, c) in DIRTY_CELLS:
            return (r, c), DIRTY_REWARD, False
        return (r, c), CLEAN_STAY, False

    if action == 5:  # Stay
        return (r, c), CLEAN_STAY, False

    # Movement
    if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in WALLS:
        return (nr, nc), BATTERY_COST, False
    else:
        return (r, c), WALL_PENALTY, False


def generate_episode(policy_fn, max_steps=200):
    """Generate one episode using current policy."""
    state = BASE
    dirty_remaining = set(DIRTY_CELLS)
    episode = []

    for _ in range(max_steps):
        action = policy_fn(state)
        next_state, reward, _ = step(state, action)

        # Extra reward shaping for cleaning
        if action == 4 and state in dirty_remaining:
            dirty_remaining.remove(state)
            reward = DIRTY_REWARD

        # Episode ends when all cells are clean
        done = len(dirty_remaining) == 0

        episode.append((state, action, reward))
        state = next_state
        if done:
            break

    return episode


# -- Monte Carlo Prediction: First-visit V(s) ---------------------------------
def first_visit_mc_prediction(episodes=EPISODES):
    V = defaultdict(float)
    returns = defaultdict(list)

    for ep_num in range(1, episodes + 1):
        episode = generate_episode(lambda s: np.random.randint(N_ACTIONS))
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = GAMMA * G + r
            if s not in visited:
                visited.add(s)
                returns[s].append(G)
                V[s] = np.mean(returns[s])

        if ep_num % 500 == 0:
            avg_v = np.mean([V[s] for s in V]) if V else 0
            print(f"  MC Prediction ep {ep_num:5d}  |  avg V(s)={avg_v:.3f}")

    return dict(V)


# -- Monte Carlo Control: epsilon-soft -----------------------------------------
def mc_control(episodes=EPISODES):
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    returns = defaultdict(list)
    policy_history = []

    for ep_num in range(1, episodes + 1):
        # Epsilon-greedy policy from Q
        def policy_fn(s):
            if np.random.rand() < EPSILON:
                return np.random.randint(N_ACTIONS)
            return int(np.argmax(Q[s]))

        episode = generate_episode(policy_fn)

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = GAMMA * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns[(s, a)].append(G)
                Q[s][a] = np.mean(returns[(s, a)])

        if ep_num % 500 == 0:
            avg_q = np.mean([Q[s].max() for s in Q]) if Q else 0
            print(f"  MC Control  ep {ep_num:5d}  |  avg Q={avg_q:.3f}")

    # Derive greedy policy
    policy = {}
    for s in Q:
        policy[s] = int(np.argmax(Q[s]))
    return dict(Q), policy


# -- Energy per episode (from MC Control runs) ---------------------------------
def compute_energy_per_episode(Q, episodes=EPISODES):
    """Re-run episodes with greedy policy and record total energy cost."""
    def greedy_policy(s):
        return int(np.argmax(Q[s]))

    energies = []
    for _ in range(episodes):
        episode = generate_episode(greedy_policy)
        total = sum(r for _, _, r in episode)
        energies.append(total)
    return energies


# -- Plotting -------------------------------------------------------------------
def plot_v_heatmap(V):
    grid = np.zeros((ROWS, COLS))
    for (r, c), v in V.items():
        grid[r][c] = v

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="V(s)")
    for r in range(ROWS):
        for c in range(COLS):
            label = "W" if (r, c) in WALLS else f"{grid[r, c]:.2f}"
            ax.text(c, r, label, ha="center", va="center", fontsize=9,
                    color="white" if grid[r, c] < -0.2 else "black")
    ax.set_title("Monte Carlo V(s) – Robot Vacuum")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp08_mc_v_heatmap.png'), dpi=150)
    print("Plot saved: exp08_mc_v_heatmap.png")
    plt.close()


def plot_policy_arrows(policy):
    grid = np.full((ROWS, COLS), " ")
    for (r, c), a in policy.items():
        if (r, c) in WALLS:
            grid[r][c] = "W"
        else:
            grid[r][c] = ACTION_NAMES.get(a, "?")

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in range(ROWS):
        for c in range(COLS):
            color = "grey" if (r, c) in WALLS else "black"
            ax.text(c, r, grid[r][c], ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, linewidth=0.5)
    ax.set_title("MC Control Policy – Robot Vacuum\n"
                 "(^v<-->=move, S=Suck, ·=Stay)")
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp08_mc_policy_arrows.png'), dpi=150)
    print("Plot saved: exp08_mc_policy_arrows.png")
    plt.close()


def plot_energy(energies):
    window = 50
    smoothed = np.convolve(energies, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(energies) + 1), energies, alpha=0.25, label="Per-episode")
    plt.plot(range(window, len(energies) + 1), smoothed,
             color="red", linewidth=2, label=f"Smoothed ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (energy proxy)")
    plt.title("Energy per Episode – Robot Vacuum MC Control")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp08_mc_energy.png'), dpi=150)
    print("Plot saved: exp08_mc_energy.png")
    plt.close()


# -- Main -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Experiment 08: Monte Carlo – Robot Vacuum Cleaner ===\n")

    print("[1] First-Visit MC Prediction for V(s)...")
    V = first_visit_mc_prediction(episodes=EPISODES)
    print(f"    States with values: {len(V)}\n")

    print("[2] MC Control (epsilon-soft)...")
    Q, policy = mc_control(episodes=EPISODES)
    print(f"    States with Q-values: {len(Q)}\n")

    print("[3] Computing energy per episode...")
    energies = compute_energy_per_episode(Q, episodes=500)

    print("[4] Generating plots...")
    plot_v_heatmap(V)
    plot_policy_arrows(policy)
    plot_energy(energies)

    # Summary
    print(f"\n--- Summary ---")
    print(f"Dirty cells: {len(DIRTY_CELLS)}")
    print(f"Wall cells : {len(WALLS)}")
    print(f"Final avg energy (last 100): {np.mean(energies[-100:]):.3f}")
    print("Done.")

