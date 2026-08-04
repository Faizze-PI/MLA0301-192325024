"""
Exp01 - MDP for Simplified Chess Game (4x4 Mini-Chess)
========================================================
Build a 4x4 board mini-chess with King + 2 pawns.
Implement Value Iteration to solve for V(s).

State  = (king_pos, pawn1_pos, pawn2_pos, turn)
         turn: 0 = player (maximise), 1 = opponent (minimise)
Actions = legal King moves (one step in 8 directions)
Reward  : +10 capture opponent pawn, -10 lose own king, -0.1 per move

Output : Value table, optimal move trace from a sample start state.
"""

import numpy as np
import itertools
from collections import defaultdict

# -- board constants ----------------------------------------------
ROWS, COLS = 4, 4
KING       = "K"
PAWN       = "P"
EMPTY      = "."
WALL       = "#"

# 8-connected moves for the King
KING_MOVES = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

GAMMA   = 0.95
THETA   = 1e-6          # convergence threshold
MAX_ITER = 5000

# -- helper: inside board? ----------------------------------------
def inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def pos_to_rc(p):
    return divmod(p, COLS)

def rc_to_pos(r, c):
    return r * COLS + c

# -- state space --------------------------------------------------
# Each piece occupies one cell; all three are distinct
N_CELLS = ROWS * COLS          # 16
ALL_POS = list(range(N_CELLS))

def build_state_space():
    """Return list of all valid states and a mapping state->index."""
    states = []
    for k, p1, p2 in itertools.permutations(ALL_POS, 3):
        for turn in (0, 1):
            states.append((k, p1, p2, turn))
    return states

# -- actions: legal king moves from current king position ---------
def legal_actions(state):
    """Return list of (dr, dc) moves the king can make."""
    k, p1, p2, turn = state
    kr, kc = pos_to_rc(k)
    actions = []
    for dr, dc in KING_MOVES:
        nr, nc = kr + dr, kc + dc
        if inside(nr, nc):
            dest = rc_to_pos(nr, nc)
            # cannot stay on own pawn
            if dest == p1 or dest == p2:
                continue
            actions.append((dr, dc))
    return actions

# -- transition & reward ------------------------------------------
def apply_move(state, action):
    """Deterministic transition: move king, return new state, reward."""
    k, p1, p2, turn = state
    dr, dc = action
    kr, kc = pos_to_rc(k)
    nk = rc_to_pos(kr + dr, kc + dc)

    reward = -0.1  # step cost

    if turn == 0:  # player's turn (king)
        # check if king captures opponent pawn
        if nk == p2:
            reward = 10.0
            return (nk, p1, p1, 1), reward  # p2 removed, dummy pos
        if nk == p1:
            return state, -10.0             # walked into own pawn (illegal, penalise)
        return (nk, p1, p2, 1), reward
    else:
        # opponent pawn moves 1 step toward king (simplified)
        pr, pc = pos_to_rc(p1)
        kr2, kc2 = pos_to_rc(k)
        dr2 = np.sign(kr2 - pr)
        dc2 = np.sign(kc2 - pc)
        np_ = rc_to_pos(pr + dr2, pc + dc2)
        if np_ == k:
            return (k, p1, p2, 0), -10.0   # king captured
        return (k, np_, p2, 0), reward

# -- Value Iteration ----------------------------------------------
def value_iteration(states, state_idx, theta=THETA, gamma=GAMMA):
    V = {s: 0.0 for s in states}
    policy = {}

    for iteration in range(1, MAX_ITER + 1):
        delta = 0.0
        for s in states:
            if s not in state_idx:
                continue
            v_old = V[s]
            acts = legal_actions(s)
            if not acts:
                continue

            values = []
            for a in acts:
                s_next, r = apply_move(s, a)
                if s_next not in V:
                    s_next = s  # invalid transition stays
                values.append(r + gamma * V[s_next])

            V[s] = max(values)
            policy[s] = acts[np.argmax(values)]
            delta = max(delta, abs(V[s] - v_old))

        if iteration % 200 == 0:
            print(f"  VI iter {iteration:4d}  delta={delta:.6f}")
        if delta < theta:
            print(f"  Converged in {iteration} iterations  (delta={delta:.2e})")
            break

    return V, policy

# -- trace optimal path -------------------------------------------
def trace_optimal(start, policy, max_steps=30):
    path = [start]
    s = start
    for _ in range(max_steps):
        if s not in policy:
            break
        a = policy[s]
        s, _ = apply_move(s, a)
        path.append(s)
    return path

# -- pretty print -------------------------------------------------
def print_value_table(V, top_n=20):
    """Print top-N highest-value states."""
    ranked = sorted(V.items(), key=lambda x: -abs(x[1]))[:top_n]
    print(f"\n{'King':>5} {'P1':>5} {'P2':>5} {'Turn':>5}  =>  V(s)")
    print("-" * 40)
    for s, v in ranked:
        print(f"  {s[0]:2d}     {s[1]:2d}     {s[2]:2d}     {s[3]:2d}    => {v:+.3f}")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Exp 01 : MDP for Simplified 4x4 Chess  (Value Iteration)")
    print("=" * 60)

    # 1. Build state space
    print("\n[1] Building state space ...")
    states = build_state_space()
    state_idx = {s: i for i, s in enumerate(states)}
    print(f"    Total states : {len(states)}")

    # 2. Run Value Iteration
    print("\n[2] Running Value Iteration ...")
    V, policy = value_iteration(states, state_idx)

    # 3. Print value table
    print_value_table(V, top_n=15)

    # 4. Trace from sample start
    # King at (0,0)=0, pawn1 at (3,0)=12, pawn2 at (3,3)=15, player to move
    start = (0, 12, 15, 0)
    print(f"\n[4] Optimal move trace from start state {start} :")
    path = trace_optimal(start, policy)
    for i, s in enumerate(path):
        kr, kc = pos_to_rc(s[0])
        p1r, p1c = pos_to_rc(s[1])
        p2r, p2c = pos_to_rc(s[2])
        turn_str = "Player" if s[3] == 0 else "Opponent"
        print(f"    Step {i:2d}: K({kr},{kc})  P1({p1r},{p1c})  P2({p2r},{p2c})  turn={turn_str}  "
              f"V={V.get(s, 0):+.3f}")

    print("\n[Done] Experiment 01 complete.")

if __name__ == "__main__":
    main()

