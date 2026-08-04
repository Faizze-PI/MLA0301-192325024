"""
Experiment 4: Value Iteration with TensorFlow (Restricted Grid)
------------------------------------------------------------------------
Aim: Implement value iteration using TensorFlow/Keras to determine the
optimal policy for a constrained grid with restricted (blocked) states.
Explain how Bellman equations obtain the optimal policy.

Bellman equation:  V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
Grid: 5x5 with 4 blocked states
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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


def get_transition_matrix():
    """Build transition tensor: P[s_next, s, a] using tf."""
    n_states = ROWS * COLS
    P = tf.zeros((n_states, n_states, N_ACTIONS), dtype=tf.float32)
    # We'll work with numpy for building, then convert
    P_np = np.zeros((n_states, n_states, N_ACTIONS), dtype=np.float32)

    for r in range(ROWS):
        for c in range(COLS):
            s = r * COLS + c
            if (r, c) in BLOCKED or (r, c) == GOAL:
                P_np[s, s, :] = 1.0  # self-loop
                continue
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in BLOCKED:
                    ns = nr * COLS + nc
                else:
                    ns = s  # wall/block bounce
                P_np[ns, s, a] = 1.0

    return tf.constant(P_np)


def get_reward_vector():
    """Build reward vector R[s, a]."""
    n_states = ROWS * COLS
    R_np = np.full((n_states, N_ACTIONS), REWARD_STEP, dtype=np.float32)

    for r in range(ROWS):
        for c in range(COLS):
            s = r * COLS + c
            if (r, c) == GOAL:
                R_np[s, :] = REWARD_GOAL
            elif (r, c) in BLOCKED:
                R_np[s, :] = REWARD_BLOCKED

    return tf.constant(R_np)


@tf.function
def value_iteration_step(V, P, R, gamma):
    """One Bellman backup using TensorFlow ops (vectorized over all states)."""
    n_states = tf.shape(V)[0]
    # Q[s, a] = Σ_s' P[s', s, a] * (R[s, a] + γ * V[s'])
    # P shape: [n_states, n_states, N_ACTIONS]
    # V shape: [n_states]
    # R shape: [n_states, N_ACTIONS]
    future_value = tf.reduce_sum(P * V[:, tf.newaxis, tf.newaxis], axis=0)  # [n_states, N_ACTIONS]
    Q = R + gamma * tf.transpose(future_value)  # [n_states, N_ACTIONS] — need reshape
    # Actually: P[s', s, a] so sum over s' axis=0
    # Let's fix: Q[s, a] = Σ_s' P[s', s, a] * (R[s, a] + γ * V[s'])
    # future[s, a] = Σ_s' P[s', s, a] * V[s'] = sum over axis 0 of P * V
    # P[:, :, a] is [n_states, n_states] where P[next, curr, a]
    # So for each s: Q[s, a] = sum_s' P[s', s, a] * (R[s, a] + gamma * V[s'])
    # = R[s, a] + gamma * sum_s' P[s', s, a] * V[s']
    # sum_s' P[s', s, a] * V[s'] for all s,a at once:
    #   einsum: P[i,j,k] * V[i] -> sum over i -> result[j,k]
    expected_V = tf.einsum("ijk,i->jk", P, V)  # [n_states, N_ACTIONS]
    Q = R + gamma * expected_V
    V_new = tf.reduce_max(Q, axis=1)  # [n_states]
    return V_new, Q


def value_iteration():
    P = get_transition_matrix()
    R = get_reward_vector()
    n_states = ROWS * COLS
    V = tf.zeros((n_states,), dtype=tf.float32)
    errors = []

    for i in range(N_ITERATIONS):
        V_new, Q = value_iteration_step(V, P, R, GAMMA)
        error = tf.reduce_max(tf.abs(V_new - V))
        errors.append(float(error))
        V = V_new
        if error < 1e-6:
            print(f"  Converged at iteration {i + 1}")
            break

    policy = tf.argmax(Q, axis=1).numpy()
    V_np = V.numpy()
    return V_np, policy, errors


def plot_results(V, policy, errors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: policy
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

    # Middle: V(s) heatmap
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

    # Right: convergence
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
    print(f"γ = {GAMMA} | Bellman equation: V(s) = max_a Σ P(s'|s,a)[R + γV(s')]\n")
    print("Running TensorFlow-accelerated Value Iteration...\n")

    V, policy, errors = value_iteration()
    print(f"Converged in {len(errors)} iterations")
    print(f"V(start) = {V[0 * COLS + 0]:.2f}")

    plot_results(V, policy, errors)
    print("\nExperiment 4 completed successfully!")
