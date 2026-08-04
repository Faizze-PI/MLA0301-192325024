"""
Experiment 3: MDP Robot with Energy Constraint
------------------------------------------------------------------------
Aim: Design an MDP for an autonomous robot reaching a destination while
minimizing energy consumption. Define states, actions, transition
probabilities, rewards, and energy constraints.

Grid:        5x5
States:      (row, col, battery_level)  → 5x5x5 = 125 states
Battery:     5 discrete levels (4,3,2,1,0 moves remaining)
Actions:     N, S, E, W
Transitions: 90% intended, 10% slip to adjacent perpendicular cell
Rewards:     +100 goal, -100 battery death, -1 per step
Constraint:  Battery = 5 moves; agent must reach goal before battery dies
Algorithm:   Value Iteration
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

ROWS, COLS = 5, 5
BATTERY_LEVELS = 5  # 4,3,2,1,0
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(1, 2), (2, 2), (3, 2)}

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
ACTION_NAMES = ["N", "S", "W", "E"]
N_ACTIONS = len(ACTIONS)

GAMMA = 0.9
N_ITERATIONS = 50


def get_transitions(state, action_idx):
    """Return list of (probability, next_state, reward)."""
    r, c, bat = state
    if bat <= 0 or (r, c) == GOAL:
        return [(1.0, state, 0.0)]

    dr, dc = ACTIONS[action_idx]
    intended = (min(max(r + dr, 0), ROWS - 1), min(max(c + dc, 0), COLS - 1), bat - 1)

    # Slip directions (perpendicular)
    if dr != 0:
        slips = [(0, 1), (0, -1)]
    else:
        slips = [(1, 0), (-1, 0)]

    transitions = []
    # 90% intended
    nr, nc = intended[0], intended[1]
    if (nr, nc) in OBSTACLES:
        transitions.append((0.9, (r, c, bat - 1), -100.0))
    elif (nr, nc) == GOAL:
        transitions.append((0.9, (nr, nc, bat - 1), 100.0))
    else:
        transitions.append((0.9, (nr, nc, bat - 1), -1.0))

    # 10% slip (5% each)
    for sdr, sdc in slips:
        sr, sc = min(max(r + sdr, 0), ROWS - 1), min(max(c + sdc, 0), COLS - 1)
        if (sr, sc) in OBSTACLES:
            transitions.append((0.05, (r, c, bat - 1), -100.0))
        elif (sr, sc) == GOAL:
            transitions.append((0.05, (sr, sc, bat - 1), 100.0))
        else:
            transitions.append((0.05, (sr, sc, bat - 1), -1.0))

    return transitions


def value_iteration():
    V = np.zeros((ROWS, COLS, BATTERY_LEVELS))
    policy = np.zeros((ROWS, COLS, BATTERY_LEVELS), dtype=int)
    errors = []

    for iteration in range(N_ITERATIONS):
        V_new = V.copy()
        max_error = 0
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in OBSTACLES:
                    continue
                for b in range(BATTERY_LEVELS):
                    if b == 0 or (r, c) == GOAL:
                        continue
                    q_values = np.zeros(N_ACTIONS)
                    for a in range(N_ACTIONS):
                        for prob, (nr, nc, nb), reward in get_transitions((r, c, b), a):
                            q_values[a] += prob * (reward + GAMMA * V[nr, nc, nb])
                    V_new[r, c, b] = np.max(q_values)
                    policy[r, c, b] = int(np.argmax(q_values))
                    max_error = max(max_error, abs(V_new[r, c, b] - V[r, c, b]))
        V = V_new
        errors.append(max_error)
        if max_error < 1e-6:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return V, policy, errors


def plot_results(V, policy, errors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: policy at full battery
    ax = axes[0]
    grid = np.zeros((ROWS, COLS))
    for r, c in OBSTACLES:
        grid[r, c] = -1
    grid[GOAL[0], GOAL[1]] = 1
    ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in OBSTACLES:
                ax.text(c, r, "X", ha="center", va="center", fontsize=12, fontweight="bold", color="red")
            elif (r, c) == GOAL:
                ax.text(c, r, "G", ha="center", va="center", fontsize=12, fontweight="bold", color="green")
            else:
                ax.text(c, r, ACTION_NAMES[policy[r, c, BATTERY_LEVELS - 1]],
                        ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("Optimal Policy (full battery)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Middle: V(s) heatmap at full battery
    ax = axes[1]
    V_display = V[:, :, BATTERY_LEVELS - 1].copy()
    for r, c in OBSTACLES:
        V_display[r, c] = np.nan
    im = ax.imshow(V_display, cmap="viridis")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in OBSTACLES:
                ax.text(c, r, f"{V_display[r, c]:.0f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title("V(s) at Full Battery")
    fig.colorbar(im, ax=ax)

    # Right: convergence
    ax = axes[2]
    ax.plot(range(1, len(errors) + 1), errors, "o-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |V_new - V_old|")
    ax.set_title("Value Iteration Convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment3_mdp_robot.png", dpi=150)
    print("Plot saved as 'experiment3_mdp_robot.png'")


if __name__ == "__main__":
    print("Experiment 3: MDP Robot with Energy Constraint\n")
    print(f"Grid: {ROWS}x{COLS} | Battery: {BATTERY_LEVELS} levels | Start: {START}, Goal: {GOAL}")
    print(f"Obstacles: {OBSTACLES}")
    print(f"Transitions: 90% intended, 10% slip | γ = {GAMMA}\n")
    print("Running Value Iteration...\n")

    V, policy, errors = value_iteration()
    print(f"\nConverged in {len(errors)} iterations (final error: {errors[-1]:.2e})")
    print(f"V(start, full battery) = {V[0, 0, BATTERY_LEVELS - 1]:.1f}")

    plot_results(V, policy, errors)
    print("\nExperiment 3 completed successfully!")
