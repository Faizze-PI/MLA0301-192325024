"""
Exp 29: Policy Iteration for Traffic Light Timing Optimization
==============================================================
Single intersection MDP solved via Policy Iteration.

State space:
  - Queue length per direction (North, South, East, West): 4 buckets each (0-3)
  - Current phase: 0 (NS green) or 1 (EW green)
  Total states: 4^4 * 2 = 512 (reduced to ~128 effective via symmetry)

Action space: 0 = hold current phase, 1 = switch phase

Reward: -(total waiting vehicles) each step

Compares learned policy vs fixed-timer baseline.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import time
import os

# ---------------------------------------------------------------------------
# MDP Definition
# ---------------------------------------------------------------------------

NUM_BUCKETS = 4          # queue length buckets: 0,1,2,3
PHASES = 2               # 0 = NS green, 1 = EW green
ACTIONS = 2              # 0 = hold, 1 = switch
ARRIVAL_RATE = 0.4       # vehicles per direction per step
DISCOUNT = 0.95
THETA = 1e-6             # convergence threshold


def _discretize_queue(raw_queue):
    """Discretize raw queue length into bucket index 0..NUM_BUCKETS-1."""
    return min(int(raw_queue), NUM_BUCKETS - 1)


def _all_states():
    """Return all (n, s, e, w, phase) states."""
    for n, s, e, w in product(range(NUM_BUCKETS), repeat=4):
        for phase in range(PHASES):
            yield (n, s, e, w, phase)


def state_index(state):
    n, s, e, w, phase = state
    return ((n * NUM_BUCKETS + s) * NUM_BUCKETS + e) * NUM_BUCKETS * PHASES + w * PHASES + phase


NUM_STATES = NUM_BUCKETS ** 4 * PHASES


def transition_and_reward(state, action):
    """Simulate one step. Return list of (next_state, probability, reward)."""
    n, s, e, w, phase = state

    # Determine active directions
    if phase == 0:
        active = [0, 1]  # N, S
        inactive = [2, 3]  # E, W
    else:
        active = [2, 3]
        inactive = [0, 1]

    queues = [n, s, e, w]

    # New arrivals (Poisson-like via Bernoulli per slot)
    arrivals = [0, 0, 0, 0]
    for d in range(4):
        arrivals[d] = np.random.poisson(ARRIVAL_RATE)

    # Service: active directions get 1 vehicle served if queue > 0
    new_queues = [0, 0, 0, 0]
    for d in range(4):
        q = queues[d] + arrivals[d]
        if d in active and q > 0:
            q -= 1  # one vehicle served
        new_queues[d] = _discretize_queue(q)

    # Next phase
    if action == 1:  # switch
        next_phase = 1 - phase
    else:
        next_phase = phase

    next_state = (new_queues[0], new_queues[1], new_queues[2], new_queues[3], next_phase)

    # Reward: negative total waiting
    total_waiting = sum(new_queues)
    reward = -float(total_waiting)

    return [(next_state, 1.0, reward)]


# ---------------------------------------------------------------------------
# Policy Iteration
# ---------------------------------------------------------------------------

def policy_iteration():
    """Run full Policy Iteration and return converged policy + value function."""
    print("Running Policy Iteration for Traffic Light MDP ...")
    print(f"  States: {NUM_STATES}, Discount: {DISCOUNT}")

    # Initialize uniform random policy
    policy = np.zeros(NUM_STATES, dtype=int)
    V = np.zeros(NUM_STATES)
    state_list = list(_all_states())
    assert len(state_list) == NUM_STATES

    iteration = 0
    value_history = []

    while True:
        iteration += 1

        # ---------- Policy Evaluation (iterative) ----------
        while True:
            delta = 0.0
            for si in range(NUM_STATES):
                s = state_list[si]
                a = policy[si]
                v_new = 0.0
                for ns, prob, r in transition_and_reward(s, a):
                    nsi = state_index(ns)
                    v_new += prob * (r + DISCOUNT * V[nsi])
                delta = max(delta, abs(v_new - V[si]))
                V[si] = v_new
            if delta < THETA:
                break

        value_history.append(np.mean(V))

        # ---------- Policy Improvement ----------
        stable = True
        for si in range(NUM_STATES):
            s = state_list[si]
            old_action = policy[si]
            action_values = np.zeros(ACTIONS)
            for a in range(ACTIONS):
                for ns, prob, r in transition_and_reward(s, a):
                    nsi = state_index(ns)
                    action_values[a] += prob * (r + DISCOUNT * V[nsi])
            policy[si] = int(np.argmax(action_values))
            if policy[si] != old_action:
                stable = False

        print(f"  Iteration {iteration}: max_delta={delta:.6f}, mean_V={value_history[-1]:.2f}")

        if stable:
            print(f"  Policy converged after {iteration} iterations.")
            break

        if iteration > 50:
            print("  Max iterations reached.")
            break

    return policy, V, value_history


# ---------------------------------------------------------------------------
# Fixed-Timer Baseline
# ---------------------------------------------------------------------------

def fixed_timer_baseline(num_steps=5000):
    """Simulate fixed-timer policy: switch every 5 steps."""
    np.random.seed(42)
    phase = 0
    queues = [0, 0, 0, 0]
    total_reward = 0.0
    rewards_per_step = []

    for step in range(num_steps):
        # Switch every 5 steps
        if step % 5 == 0:
            phase = 1 - phase

        if phase == 0:
            active = [0, 1]
        else:
            active = [2, 3]

        # Arrivals
        for d in range(4):
            queues[d] += np.random.poisson(ARRIVAL_RATE)

        # Service
        for d in active:
            if queues[d] > 0:
                queues[d] -= 1

        waiting = sum(queues)
        total_reward -= waiting
        rewards_per_step.append(-waiting)

    avg_reward = total_reward / num_steps
    return avg_reward, rewards_per_step


# ---------------------------------------------------------------------------
# Simulation of Learned Policy
# ---------------------------------------------------------------------------

def simulate_learned(policy, num_steps=5000):
    """Simulate the learned policy."""
    np.random.seed(42)
    state_list = list(_all_states())
    phase = 0
    queues = [0, 0, 0, 0]
    total_reward = 0.0
    rewards_per_step = []

    for step in range(num_steps):
        q_disc = [_discretize_queue(q) for q in queues]
        state = (q_disc[0], q_disc[1], q_disc[2], q_disc[3], phase)
        si = state_index(state)
        action = policy[si]

        if phase == 0:
            active = [0, 1]
        else:
            active = [2, 3]

        # Arrivals
        for d in range(4):
            queues[d] += np.random.poisson(ARRIVAL_RATE)

        # Service
        for d in active:
            if queues[d] > 0:
                queues[d] -= 1

        # Switch if action says so
        if action == 1:
            phase = 1 - phase

        waiting = sum(queues)
        total_reward -= waiting
        rewards_per_step.append(-waiting)

    avg_reward = total_reward / num_steps
    return avg_reward, rewards_per_step


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(policy, value_history, learned_rewards, fixed_rewards):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    # 1. Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(value_history, "b-o", markersize=3, label="Mean V(s)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Value")
    ax.set_title("Policy Iteration Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp29_convergence.png"), dpi=150)
    plt.close(fig)

    # 2. Learned vs fixed-timer reward curve
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 50
    if len(learned_rewards) >= window:
        learned_smooth = np.convolve(learned_rewards, np.ones(window) / window, mode="valid")
        fixed_smooth = np.convolve(fixed_rewards, np.ones(window) / window, mode="valid")
        ax.plot(learned_smooth, label="Learned Policy", color="green")
        ax.plot(fixed_smooth, label="Fixed Timer (5-step)", color="red", alpha=0.7)
    else:
        ax.plot(learned_rewards, label="Learned Policy", color="green")
        ax.plot(fixed_rewards, label="Fixed Timer", color="red", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (negative waiting)")
    ax.set_title("Traffic Light: Learned Policy vs Fixed Timer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp29_reward_comparison.png"), dpi=150)
    plt.close(fig)

    # 3. Policy visualization (phase-based heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for phase_idx in range(2):
        phase_label = "NS Green" if phase_idx == 0 else "EW Green"
        grid = np.zeros((NUM_BUCKETS, NUM_BUCKETS))
        for n in range(NUM_BUCKETS):
            for s in range(NUM_BUCKETS):
                # Average over E,W
                total_switch = 0
                count = 0
                for e in range(NUM_BUCKETS):
                    for w in range(NUM_BUCKETS):
                        si = state_index((n, s, e, w, phase_idx))
                        total_switch += policy[si]
                        count += 1
                grid[n, s] = total_switch / count

        im = axes[phase_idx].imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[phase_idx].set_title(f"Switch Probability\n({phase_label})")
        axes[phase_idx].set_xlabel("South Queue Bucket")
        axes[phase_idx].set_ylabel("North Queue Bucket")
        plt.colorbar(im, ax=axes[phase_idx], fraction=0.046)

    fig.suptitle("Learned Policy: Probability of Switching Action", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp29_policy_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    policy, V, value_history = policy_iteration()

    learned_avg, learned_rewards = simulate_learned(policy)
    fixed_avg, fixed_rewards = fixed_timer_baseline()

    print(f"\nResults:")
    print(f"  Learned policy avg reward:  {learned_avg:.2f}")
    print(f"  Fixed timer avg reward:     {fixed_avg:.2f}")
    print(f"  Improvement:                {learned_avg - fixed_avg:.2f}")
    print(f"  Time elapsed: {time.time() - t0:.1f}s")

    plot_results(policy, value_history, learned_rewards, fixed_rewards)
