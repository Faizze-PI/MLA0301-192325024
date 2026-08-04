"""
Exp04 - Bellman Optimality for Delivery Robot (Minimum Travel Cost)
====================================================================
Grid with variable edge costs. Value Iteration to compute V*(s):
    V*(s) = min_a [ cost(s,a) + gamma * V*(s') ]
gamma = 0.95

Output : Convergence plot, optimal path on grid, total minimum cost.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- grid definition ----------------------------------------------
# 0 = free, 1 = road (cheap), 2 = rough (expensive), 3 = obstacle
CELL_TYPES = {0: "free", 1: "road", 2: "rough", 3: "wall"}

GRID = np.array([
    [0, 1, 0, 3, 0],
    [0, 1, 0, 3, 0],
    [0, 0, 0, 0, 0],
    [3, 3, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=int)

ROWS, COLS = GRID.shape
START = (0, 0)
GOAL  = (4, 4)

# -- edge costs ---------------------------------------------------
# Moving onto different cell types has different costs
EDGE_COST = {0: 1.0, 1: 0.5, 2: 3.0, 3: np.inf}  # obstacle = inf

# -- actions ------------------------------------------------------
ACTIONS     = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAME = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
N_ACTIONS   = 4

GAMMA   = 0.95
THETA   = 1e-6
MAX_ITER = 5000

def inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def get_cost(r, c):
    """Cost to step ONTO cell (r, c)."""
    if not inside(r, c):
        return np.inf
    return EDGE_COST[GRID[r, c]]

# -- Value Iteration (Bellman Optimality) -------------------------
def value_iteration():
    V = np.zeros((ROWS, COLS))
    policy = np.full((ROWS, COLS), -1, dtype=int)
    convergence = []

    for iteration in range(1, MAX_ITER + 1):
        delta = 0.0
        V_new = V.copy()

        for r in range(ROWS):
            for c in range(COLS):
                if GRID[r, c] == 3:
                    continue
                if (r, c) == GOAL:
                    V_new[r, c] = 0.0
                    continue

                values = []
                for a in range(N_ACTIONS):
                    dr, dc = ACTIONS[a]
                    nr, nc = r + dr, c + dc
                    cost = get_cost(nr, nc)
                    if cost == np.inf:
                        values.append(np.inf)
                    else:
                        values.append(cost + GAMMA * V[nr, nc])

                best = min(values)
                V_new[r, c] = best
                policy[r, c] = int(np.argmin(values))
                delta = max(delta, abs(V_new[r, c] - V[r, c]))

        V = V_new
        convergence.append(delta)

        if iteration % 500 == 0:
            print(f"  VI iter {iteration:4d}  delta={delta:.6f}")
        if delta < THETA:
            print(f"  Converged in {iteration} iterations  (delta={delta:.2e})")
            break

    return V, policy, convergence

# -- extract optimal path -----------------------------------------
def trace_path(policy):
    path = [START]
    s = START
    total_cost = 0.0
    for _ in range(100):
        if s == GOAL:
            break
        a = policy[s[0], s[1]]
        if a < 0:
            break
        dr, dc = ACTIONS[a]
        ns = (s[0] + dr, s[1] + dc)
        if not inside(*ns) or GRID[ns] == 3:
            break
        total_cost += get_cost(ns[0], ns[1])
        path.append(ns)
        s = ns
    return path, total_cost

# -- plotting -----------------------------------------------------
ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}

def plot_convergence(convergence, filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\exp04_convergence.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(convergence)+1), convergence, color="crimson", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max|V_new - V_old|")
    ax.set_title("Value Iteration Convergence (Bellman Optimality)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_grid_optimal_path(V, policy, path, filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\exp04_optimal_path.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_aspect("equal")

    # cell colours
    cell_color = {0: "#ecf0f1", 1: "#3498db", 2: "#e67e22", 3: "#2c3e50"}
    path_set = set(path)

    for r in range(ROWS):
        for c in range(COLS):
            color = cell_color[GRID[r, c]]
            if (r, c) in path_set:
                color = "#2ecc71"
            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=color, edgecolor="white", linewidth=2)
            ax.add_patch(rect)

            # value label
            if GRID[r, c] != 3:
                ax.text(c, r + 0.25, f"{V[r, c]:.1f}", ha="center", va="center",
                        fontsize=8, color="#7f8c8d")

            # arrow
            if GRID[r, c] != 3 and (r, c) != GOAL:
                a = policy[r, c]
                if a >= 0:
                    ax.text(c, r - 0.15, ARROW[a], ha="center", va="center",
                            fontsize=16, fontweight="bold", color="#2c3e50")

    # path overlay
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        ax.plot([c1, c2], [r1, r2], color="#e74c3c", linewidth=3, zorder=5)

    ax.text(START[1], START[0], "S", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round", fc="#3498db", ec="none"))
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round", fc="#e74c3c", ec="none"))

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Optimal Path on Delivery Grid (numbers = V*(s))")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Exp 04 : Bellman Optimality -- Delivery Robot (Min Cost)")
    print("=" * 60)

    print("\n  Grid (0=free, 1=road, 2=rough, 3=wall):")
    print(GRID)

    print("\n[1] Edge costs:", {k: v for k, v in EDGE_COST.items() if v < np.inf})

    print("\n[2] Running Value Iteration (Bellman Optimality) ...")
    V, policy, convergence = value_iteration()

    print("\n[3] Value Table V*(s):")
    print("    " + "".join(f"  c{c}  " for c in range(COLS)))
    for r in range(ROWS):
        row = f"  r{r}"
        for c in range(COLS):
            if GRID[r, c] == 3:
                row += "  --- "
            else:
                row += f" {V[r, c]:5.2f}"
        print(row)

    print("\n[4] Optimal Policy:")
    for r in range(ROWS):
        row = "  "
        for c in range(COLS):
            if GRID[r, c] == 3:
                row += "  #  "
            elif (r, c) == GOAL:
                row += "  G  "
            else:
                row += f"  {ARROW[policy[r, c]]}  "
        print(row)

    print("\n[5] Tracing optimal path ...")
    path, total_cost = trace_path(policy)
    for i, pos in enumerate(path):
        print(f"    Step {i}: ({pos[0]},{pos[1]})  V*={V[pos[0], pos[1]]:.3f}")
    print(f"\n    Total minimum cost = {total_cost:.2f}")

    print("\n[6] Saving plots ...")
    plot_convergence(convergence, "exp04_convergence.png")
    plot_grid_optimal_path(V, policy, path, "exp04_optimal_path.png")

    print("\n[Done] Experiment 04 complete.")

if __name__ == "__main__":
    main()

