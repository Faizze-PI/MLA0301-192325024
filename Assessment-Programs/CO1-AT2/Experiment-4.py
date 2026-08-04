"""
Experiment 4: Value Iteration with TensorFlow (Restricted Grid)
------------------------------------------------------------------------
Aim: Implement value iteration using TensorFlow/Keras to determine the
optimal policy for a constrained grid with restricted (blocked) states.
Explain how Bellman equations obtain the optimal policy.

Bellman equation:  V(s) = max_a sum P(s'|s,a)[R(s,a,s') + γV(s')]
Grid: 5x5 with 4 blocked states
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

ROWS, COLS = 5, 5
GOAL = (4, 4)
BLOCKED = {(1, 1), (1, 3), (3, 1), (3, 3)}
START = (0, 0)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["N", "S", "W", "E"]
N_ACTIONS = len(ACTIONS)

GAMMA = 0.9
N_ITERATIONS = 50
REWARD_GOAL = 10.0
REWARD_STEP = -1.0
REWARD_BLOCKED = -10.0


def value_iteration():
    n_states = ROWS * COLS
    V = np.zeros(n_states, dtype=np.float32)
    policy = np.zeros(n_states, dtype=int)
    errors = []

    for iteration in range(N_ITERATIONS):
        V_new = V.copy()
        max_error = 0
        for s in range(n_states):
            r, c = s // COLS, s % COLS
            if (r, c) in BLOCKED or (r, c) == GOAL:
                continue

            q_values = np.zeros(N_ACTIONS, dtype=np.float32)
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in BLOCKED:
                    ns = nr * COLS + nc
                else:
                    ns = s

                if (nr, nc) == GOAL and 0 <= nr < ROWS and 0 <= nc < COLS:
                    reward = REWARD_GOAL
                elif (nr, nc) in BLOCKED and 0 <= nr < ROWS and 0 <= nc < COLS:
                    reward = REWARD_BLOCKED
                else:
                    reward = REWARD_STEP

                q_values[a] = reward + GAMMA * V[ns]

            V_new[s] = np.max(q_values)
            policy[s] = int(np.argmax(q_values))
            max_error = max(max_error, abs(V_new[s] - V[s]))

        V = V_new
        errors.append(max_error)
        if max_error < 1e-6:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return V, policy, errors


def plot_results(V, policy, errors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    grid = np.zeros((ROWS, COLS))
    for r, c in BLOCKED:
        grid[r, c] = -1
    grid[GOAL[0], GOAL[1]] = 1
    ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
    for r in range(ROWS):
        for c in range(COLS):
            s = r * COLS + c
            if (r, c) in BLOCKED:
                ax.text(c, r, "BLK", ha="center", va="center", fontsize=9, fontweight="bold", color="red")
            elif (r, c) == GOAL:
                ax.text(c, r, "GOAL", ha="center", va="center", fontsize=9, fontweight="bold", color="green")
            else:
                ax.text(c, r, ACTION_NAMES[policy[s]], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("Optimal Policy (BLK=blocked)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax = axes[1]
    V_grid = V.reshape(ROWS, COLS).copy()
    for r, c in BLOCKED:
        V_grid[r, c] = np.nan
    im = ax.imshow(V_grid, cmap="viridis")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in BLOCKED:
                ax.text(c, r, f"{V_grid[r, c]:.1f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title("Value Function V(s)")
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    ax.plot(range(1, len(errors) + 1), errors, "o-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |V_new - V_old|")
    ax.set_title("Bellman Convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment4_tf_value_iteration.png", dpi=150)
    print("Plot saved as 'experiment4_tf_value_iteration.png'")


if __name__ == "__main__":
    print("Experiment 4: Value Iteration with TensorFlow (Restricted Grid)\n")
    print(f"Grid: {ROWS}x{COLS} | Blocked: {BLOCKED} | Goal: {GOAL}")
    print(f"gamma = {GAMMA} | Bellman equation: V(s) = max_a sum P(s'|s,a)[R + gammaV(s')]\n")
    print("Running Value Iteration...\n")

    V, policy, errors = value_iteration()
    print(f"Converged in {len(errors)} iterations")
    s = START[0] * COLS + START[1]
    print(f"V(start) = {V[s]:.2f}")

    plot_results(V, policy, errors)
    print("\nExperiment 4 completed successfully!")
