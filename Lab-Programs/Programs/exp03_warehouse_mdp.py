"""
Exp03 - MDP for Autonomous Warehouse Robot
============================================
6x6 warehouse grid with shelves, pickup zone, dropoff zone.
Formal MDP tuple (S, A, P, R, gamma) with stochastic transitions
(0.1 slip probability). Value Iteration to find V* and pi*.

Output : Formal MDP tables, value table, worked optimal policy example.
"""

import numpy as np
import itertools

# -- grid constants -----------------------------------------------
ROWS, COLS = 6, 6
EMPTY = 0
SHELF = 1
PICKUP = 2
DROPOFF = 3

WAREHOUSE = np.array([
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [2, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 0, 3, 0, 0],
], dtype=int)

# -- MDP definition -----------------------------------------------
ACTIONS     = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAME = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
N_ACTIONS   = 4
GAMMA       = 0.95
SLIP_PROB   = 0.1        # probability of slipping to adjacent direction
THETA       = 1e-6
MAX_ITER    = 5000

# -- free cells ---------------------------------------------------
FREE_CELLS = [(r, c) for r in range(ROWS) for c in range(COLS)
              if WAREHOUSE[r, c] != SHELF]

def inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def free_neighbours(r, c):
    """Return list of valid neighbour cells (not shelves, inside grid)."""
    nbrs = []
    for a_idx in range(N_ACTIONS):
        dr, dc = ACTIONS[a_idx]
        nr, nc = r + dr, c + dc
        if inside(nr, nc) and WAREHOUSE[nr, nc] != SHELF:
            nbrs.append((nr, nc))
    return nbrs

# -- transition model (stochastic) -------------------------------
def build_transitions():
    """P[s][a] = list of (prob, next_state, reward)."""
    P = {}
    for (r, c) in FREE_CELLS:
        s = (r, c)
        P[s] = {}
        for a in range(N_ACTIONS):
            dr, dc = ACTIONS[a]
            intended = (r + dr, c + dc)
            if not inside(*intended) or WAREHOUSE[intended] == SHELF:
                intended = (r, c)          # stay in place

            # generate stochastic transitions
            trans = {}
            # intended direction
            p_intended = 1.0 - SLIP_PROB
            trans[intended] = trans.get(intended, 0) + p_intended

            # perpendicular directions
            perp = []
            if a in (0, 1):  # vertical -> slip left/right
                perp = [(r, c - 1), (r, c + 1)]
            else:            # horizontal -> slip up/down
                perp = [(r - 1, c), (r + 1, c)]

            slip_each = SLIP_PROB / 2.0
            for pr, pc in perp:
                if inside(pr, pc) and WAREHOUSE[pr, pc] != SHELF:
                    trans[(pr, pc)] = trans.get((pr, pc), 0) + slip_each
                else:
                    # stay in place if blocked
                    trans[(r, c)] = trans.get((r, c), 0) + slip_each

            # reward
            def reward(nr, nc):
                if (nr, nc) == (5, 3):   # dropoff
                    return 10.0
                return -0.1               # step cost

            trans_list = []
            for (nr, nc), prob in trans.items():
                if prob > 0:
                    trans_list.append((prob, (nr, nc), reward(nr, nc)))
            P[s][a] = trans_list
    return P

# -- Value Iteration ----------------------------------------------
def value_iteration(P, theta=THETA, gamma=GAMMA):
    V = {s: 0.0 for s in FREE_CELLS}
    policy = {}

    for iteration in range(1, MAX_ITER + 1):
        delta = 0.0
        for s in FREE_CELLS:
            v_old = V[s]
            values = []
            for a in range(N_ACTIONS):
                val = 0.0
                for prob, s_next, reward in P[s][a]:
                    val += prob * (reward + gamma * V[s_next])
                values.append(val)
            V[s] = max(values)
            policy[s] = int(np.argmax(values))
            delta = max(delta, abs(V[s] - v_old))

        if iteration % 500 == 0:
            print(f"  VI iter {iteration:4d}  delta={delta:.6f}")
        if delta < theta:
            print(f"  Converged in {iteration} iterations  (delta={delta:.2e})")
            break

    return V, policy

# -- printing helpers ---------------------------------------------
def print_mdp_tables(P):
    """Print formal MDP transition probabilities for sample states."""
    print("\n  Transition Probabilities P(s'|s,a) -- sample states:\n")
    sample = [(2, 0), (2, 1), (5, 2)]
    for s in sample:
        if s not in P:
            continue
        print(f"  State s = {s}  ({WAREHOUSE[s]})")
        for a in range(N_ACTIONS):
            print(f"    Action '{ACTION_NAME[a]}':")
            for prob, s_next, r in P[s][a]:
                print(f"      s'={s_next}  prob={prob:.2f}  R={r:+.1f}")
        print()

def print_value_grid(V):
    grid = np.full((ROWS, COLS), np.nan)
    for (r, c), v in V.items():
        grid[r, c] = v
    print("  Value Grid (V*)")
    print("  " + "-" * 50)
    header = "       " + "".join(f"  c={c:d} " for c in range(COLS))
    print(header)
    for r in range(ROWS):
        row_str = f"  r={r} |"
        for c in range(COLS):
            if np.isnan(grid[r, c]):
                row_str += "  --- "
            else:
                row_str += f" {grid[r, c]:+5.2f}"
        print(row_str)
    print()

def print_policy_arrows(policy):
    ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("  Optimal Policy (pi*)")
    print("  " + "-" * 40)
    for r in range(ROWS):
        row_str = "  "
        for c in range(COLS):
            if (r, c) not in policy:
                row_str += "  #  "
            elif (r, c) == (5, 3):
                row_str += "  G  "
            else:
                row_str += f"  {ARROW[policy[(r,c)]]}  "
        print(row_str)
    print()

def worked_example(policy):
    """Trace optimal path from pickup to dropoff."""
    ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}
    start = (2, 0)  # pickup zone
    print("  Worked Example: Optimal path from Pickup (2,0) to Dropoff (5,3)")
    print("  " + "-" * 55)
    path = [start]
    s = start
    for _ in range(50):
        if s not in policy or s == (5, 3):
            break
        a = policy[s]
        # deterministic walk (no slip for demonstration)
        dr, dc = ACTIONS[a]
        ns = (s[0] + dr, s[1] + dc)
        if not inside(*ns) or WAREHOUSE[ns] == SHELF:
            ns = s
        path.append(ns)
        s = ns

    for i, pos in enumerate(path):
        cell = WAREHOUSE[pos]
        tag = {0: "floor", 2: "PICKUP", 3: "DROPOFF"}.get(cell, "floor")
        arrow = ARROW.get(policy.get(pos, 0), "—")
        print(f"    Step {i:2d}: ({pos[0]},{pos[1]}) [{tag}]  action={arrow}")
    print()

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Exp 03 : MDP for Autonomous Warehouse Robot")
    print("=" * 60)

    print("\n  Warehouse grid (0=empty, 1=shelf, 2=pickup, 3=dropoff):")
    print(WAREHOUSE)

    print("\n[1] Building formal MDP (S, A, P, R, gamma) ...")
    P = build_transitions()
    n_states = len(FREE_CELLS)
    print(f"    |S| = {n_states}  (free cells)")
    print(f"    |A| = {N_ACTIONS}")
    print(f"    gamma = {GAMMA}")
    print(f"    slip probability = {SLIP_PROB}")

    print_mdp_tables(P)

    print("[2] Running Value Iteration ...")
    V, policy = value_iteration(P)

    print("\n[3] Value Table (V*):")
    print_value_grid(V)

    print("[4] Optimal Policy (pi*):")
    print_policy_arrows(policy)

    print("[5] Worked Example:")
    worked_example(policy)

    print("[Done] Experiment 03 complete.")

if __name__ == "__main__":
    main()

