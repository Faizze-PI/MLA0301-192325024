"""
Exp 28 – Bellman Optimality Equation for Robot Grid Navigation
================================================================
10x10 grid, one goal cell (+10), several obstacles, 4-directional movement.
Every step costs −1.  Discount factor gamma = 0.9.

Bellman equation:  V(s) = max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + gamma V(s') ]

We solve via Value Iteration until convergence.
Outputs: V(s) heatmap + optimal path traced by greedy argmax.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROWS, COLS = 10, 10
GOAL = (9, 9)
GOAL_REWARD = 10
STEP_COST = -1
GAMMA = 0.9
THETA = 1e-6  # convergence threshold
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
ACTION_NAMES = ["R", "L", "D", "U"]

# Obstacles (blocked cells)
OBSTACLES = {(1, 2), (1, 3), (2, 3), (3, 3), (3, 4),
             (5, 5), (5, 6), (6, 6), (7, 2), (7, 3), (8, 3)}


def valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in OBSTACLES


def reward(s, a, s_next):
    if s_next == GOAL:
        return GOAL_REWARD
    return STEP_COST


def value_iteration():
    V = np.zeros((ROWS, COLS))
    policy = np.full((ROWS, COLS), " ", dtype=object)
    iteration = 0

    while True:
        delta = 0.0
        iteration += 1
        V_new = V.copy()
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) == GOAL:
                    V_new[r, c] = GOAL_REWARD
                    policy[r, c] = "G"
                    continue
                if (r, c) in OBSTACLES:
                    V_new[r, c] = 0
                    policy[r, c] = "#"
                    continue

                best_val = -np.inf
                best_a = " "
                for i, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = r + dr, c + dc
                    if valid(nr, nc):
                        s_next = (nr, nc)
                    else:
                        s_next = (r, c)  # stay in place
                    val = reward((r, c), i, s_next) + GAMMA * V[s_next]
                    if val > best_val:
                        best_val = val
                        best_a = ACTION_NAMES[i]
                V_new[r, c] = best_val
                policy[r, c] = best_a
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < THETA:
            break

    print(f"  Converged in {iteration} iterations (delta={delta:.2e})")
    return V, policy


def trace_optimal_path(V):
    path = [START]
    state = START
    visited = set()
    for _ in range(200):
        if state == GOAL:
            break
        if state in visited:
            print("  Warning: cycle detected")
            break
        visited.add(state)
        r, c = state
        best_val, best_next = -np.inf, state
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if valid(nr, nc):
                val = V[nr, nc]
                if val > best_val:
                    best_val = val
                    best_next = (nr, nc)
        path.append(best_next)
        state = best_next
    return path


def plot_results(V, policy, path):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap of V(s)
    display = V.copy()
    for r, c in OBSTACLES:
        display[r, c] = np.nan
    im = ax1.imshow(display, cmap="YlOrRd", interpolation="nearest")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in OBSTACLES:
                ax1.text(c, r, "#", ha="center", va="center", fontsize=10, color="gray")
            elif (r, c) == GOAL:
                ax1.text(c, r, f"{V[r,c]:.1f}", ha="center", va="center",
                         fontsize=8, fontweight="bold", color="white")
            else:
                ax1.text(c, r, f"{V[r,c]:.1f}", ha="center", va="center", fontsize=7)
    ax1.set_title("Value Function V(s)")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Policy + optimal path
    display2 = V.copy()
    for r, c in OBSTACLES:
        display2[r, c] = np.nan
    ax2.imshow(display2, cmap="Blues", interpolation="nearest")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in OBSTACLES:
                ax2.text(c, r, "#", ha="center", va="center", fontsize=10, color="gray")
            elif (r, c) == GOAL:
                ax2.text(c, r, "G", ha="center", va="center",
                         fontsize=12, fontweight="bold", color="red")
            else:
                ax2.text(c, r, policy[r, c], ha="center", va="center",
                         fontsize=12, fontweight="bold", color="navy")

    # trace path
    pr = [p[0] for p in path]
    pc = [p[1] for p in path]
    ax2.plot(pc, pr, "o-", color="lime", markersize=8, linewidth=2,
             markeredgecolor="darkgreen", label="Optimal path")
    ax2.plot(pc[0], pr[0], "s", color="yellow", markersize=12, markeredgecolor="black",
             label="Start")
    ax2.plot(pc[-1], pr[-1], "*", color="red", markersize=16, markeredgecolor="black",
             label="Goal")
    ax2.set_title("Optimal Policy & Path")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    path_out = os.path.join(out_dir, "exp28_bellman_navigation_results.png")
    plt.savefig(path_out, dpi=150)
    print(f"  Plot saved -> {path_out}")
    plt.close()


def print_grid(V, policy):
    print("\n  Value Function V(s):")
    print("  " + "".join(f"{c:>6}" for c in range(COLS)))
    for r in range(ROWS):
        row_str = f"{r:>2} "
        for c in range(COLS):
            if (r, c) in OBSTACLES:
                row_str += "  ### "
            elif (r, c) == GOAL:
                row_str += f" {V[r,c]:5.1f}"
            else:
                row_str += f" {V[r,c]:5.1f}"
        print(row_str)

    print("\n  Optimal Policy:")
    print("  " + "".join(f"{c:>4}" for c in range(COLS)))
    for r in range(ROWS):
        row_str = f"{r:>2} "
        for c in range(COLS):
            if (r, c) in OBSTACLES:
                row_str += "  # "
            else:
                row_str += f"  {policy[r,c]} "
        print(row_str)


START = (0, 0)

if __name__ == "__main__":
    print("=" * 60)
    print("Exp 28 – Bellman Optimality for Grid Navigation")
    print("=" * 60)
    print(f"  Grid: {ROWS}x{COLS}  Goal: {GOAL}  gamma={GAMMA}  step={STEP_COST}")
    print(f"  Obstacles: {len(OBSTACLES)} cells")

    V, policy = value_iteration()
    print_grid(V, policy)

    path = trace_optimal_path(V)
    print(f"\n  Optimal path ({len(path)} steps):")
    print("    " + " -> ".join(f"({r},{c})" for r, c in path))

    path_reward = GOAL_REWARD + STEP_COST * (len(path) - 1)
    print(f"  Path reward: {GOAL_REWARD} + ({STEP_COST}x{len(path)-1}) = {path_reward}")

    plot_results(V, policy, path)
    print("\nDone.")

