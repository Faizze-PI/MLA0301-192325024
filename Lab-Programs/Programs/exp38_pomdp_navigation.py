"""
Experiment 38: POMDP for Robot Navigation under Partial Observability
======================================================================
Grid environment with a robot navigating to a goal under partial observability.
Sensors detect nearby walls and landmarks only (not global position).
Discrete Bayesian filter maintains belief over positions.
QMDP-style action selection from belief state.
Sweeps 2-3 sensor noise levels for comparison.

Outputs:
  - Belief distribution heatmap (particles clustering near true position)
  - Localization accuracy across noise levels
  - Navigation reward curves per noise level
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random

# ---------------------------------------------------------------------------
# Grid Environment
# ---------------------------------------------------------------------------
GRID_ROWS = 8
GRID_COLS = 8
GOAL = (7, 7)
WALLS = {(1, 2), (2, 2), (3, 4), (4, 4), (5, 6), (6, 3)}
LANDMARKS = {(0, 5): "L1", (3, 0): "L2", (5, 5): "L3"}

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U D L R
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

NUM_PARTICLES = 500
MAX_STEPS = 60
GAMMA = 0.95
ALPHA_QMDP = 0.2
EPSILON = 0.2
NUM_EPISODES = 500
SENSOR_RANGE = 2


def in_bounds(r, c):
    return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS


def is_wall(r, c):
    return (r, c) in WALLS


def get_neighbors(r, c):
    """Get valid neighbor positions (no walls, in bounds)."""
    nbrs = []
    for dr, dc in ACTIONS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and not is_wall(nr, nc):
            nbrs.append((nr, nc))
    return nbrs


def sensor_reading(true_pos, noise_level):
    """
    Simulate sensor reading. Sensors detect nearby walls/landmarks within SENSOR_RANGE.
    Returns dict of detected features with possible noise.
    """
    reading = {"walls": [], "landmarks": []}
    tr, tc = true_pos

    for dr in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
        for dc in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
            nr, nc = tr + dr, tc + dc
            if not in_bounds(nr, nc):
                continue
            dist = abs(dr) + abs(dc)
            if dist > SENSOR_RANGE:
                continue

            # Wall detection with noise
            if is_wall(nr, nc):
                if random.random() > noise_level:
                    reading["walls"].append((nr, nc))
            else:
                if random.random() < noise_level * 0.3:
                    reading["walls"].append((nr, nc))  # false positive

            # Landmark detection with noise
            if (nr, nc) in LANDMARKS:
                if random.random() > noise_level:
                    reading["landmarks"].append((nr, nc))

    return reading


def observation_likelihood(obs, candidate_pos, noise_level):
    """P(observation | candidate_pos) - simplified likelihood."""
    tr, tc = candidate_pos
    log_lik = 0.0

    # Check wall observations
    detected_walls = set(obs["walls"])
    for dr in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
        for dc in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
            nr, nc = tr + dr, tc + dc
            if not in_bounds(nr, nc):
                continue
            if abs(dr) + abs(dc) > SENSOR_RANGE:
                continue
            actually_wall = is_wall(nr, nc)
            detected = (nr, nc) in detected_walls
            if actually_wall and detected:
                log_lik += np.log(1 - noise_level + 1e-10)
            elif actually_wall and not detected:
                log_lik += np.log(noise_level + 1e-10)
            elif not actually_wall and detected:
                log_lik += np.log(noise_level * 0.3 + 1e-10)
            elif not actually_wall and not detected:
                log_lik += np.log(1 - noise_level * 0.3 + 1e-10)

    # Check landmark observations
    detected_lmarks = set(obs["landmarks"])
    for (lr, lc) in LANDMARKS:
        dist = abs(tr - lr) + abs(tc - lc)
        if dist <= SENSOR_RANGE:
            detected = (lr, lc) in detected_lmarks
            if detected:
                log_lik += np.log(1 - noise_level + 1e-10)
            else:
                log_lik += np.log(noise_level + 1e-10)

    return np.exp(log_lik)


class BayesianFilter:
    """Discrete Bayesian filter for belief over grid positions."""

    def __init__(self):
        self.belief = np.zeros((GRID_ROWS, GRID_COLS))
        self.reset()

    def reset(self):
        # Uniform prior over non-wall cells
        self.belief = np.zeros((GRID_ROWS, GRID_COLS))
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if not is_wall(r, c):
                    self.belief[r, c] = 1.0
        total = self.belief.sum()
        if total > 0:
            self.belief /= total

    def predict(self, action_idx):
        """Motion model: propagate belief with action + noise."""
        new_belief = np.zeros_like(self.belief)
        dr, dc = ACTIONS[action_idx]

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.belief[r, c] < 1e-12:
                    continue

                # Intended move
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and not is_wall(nr, nc):
                    new_belief[nr, nc] += 0.85 * self.belief[r, c]
                else:
                    new_belief[r, c] += 0.85 * self.belief[r, c]

                # Random perturbation
                for i, (adr, adc) in enumerate(ACTIONS):
                    if i == action_idx:
                        continue
                    nnr, nnc = r + adr, c + adc
                    if in_bounds(nnr, nnc) and not is_wall(nnr, nnc):
                        new_belief[nnr, nnc] += 0.05 * self.belief[r, c]
                    else:
                        new_belief[r, c] += 0.05 * self.belief[r, c]

        total = new_belief.sum()
        if total > 0:
            new_belief /= total
        self.belief = new_belief

    def update(self, observation, noise_level):
        """Bayesian update with sensor observation."""
        likelihood = np.zeros_like(self.belief)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if not is_wall(r, c):
                    likelihood[r, c] = observation_likelihood(observation, (r, c), noise_level)

        posterior = self.belief * likelihood
        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            posterior = np.ones_like(self.belief) / np.sum(~is_wall(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS))
        self.belief = posterior

    def most_likely(self):
        """Return MAP estimate."""
        idx = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return idx

    def entropy(self):
        """Belief entropy."""
        b = self.belief[self.belief > 0]
        return -np.sum(b * np.log(b + 1e-12))


# ---------------------------------------------------------------------------
# QMDP Agent
# ---------------------------------------------------------------------------
class QMDPAgent:
    """Q-learning from belief state for POMDP navigation."""

    def __init__(self):
        self.q_table = {}

    def _belief_key(self, belief):
        """Discretize belief for Q-table key."""
        top_k = 5
        flat = belief.flatten()
        indices = np.argsort(flat)[-top_k:]
        return tuple(sorted([(i, round(flat[i], 3)) for i in indices]))

    def get_q(self, belief, action):
        key = self._belief_key(belief)
        return self.q_table.get((key, action), 0.0)

    def choose_action(self, belief, epsilon=EPSILON):
        if random.random() < epsilon:
            return random.randint(0, 3)
        q_vals = [self.get_q(belief, a) for a in range(4)]
        max_q = max(q_vals)
        best = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(best)

    def update(self, belief, action, reward, next_belief, done):
        key = self._belief_key(belief)
        next_key = self._belief_key(next_belief)

        best_next = max(self.q_table.get((next_key, a), 0.0) for a in range(4))
        target = reward + (0.0 if done else GAMMA * best_next)
        old_q = self.q_table.get((key, action), 0.0)
        self.q_table[(key, action)] = old_q + ALPHA_QMDP * (target - old_q)


def run_episode(env_filter, agent, noise_level, train=True):
    """Run one navigation episode."""
    # Random start
    while True:
        sr, sc = random.randint(0, GRID_ROWS-1), random.randint(0, GRID_COLS-1)
        if not is_wall(sr, sc) and (sr, sc) != GOAL:
            break

    env_filter.reset()
    # Initialize belief near start with some spread
    env_filter.belief = np.zeros((GRID_ROWS, GRID_COLS))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if not is_wall(r, c):
                dist = abs(r - sr) + abs(c - sc)
                env_filter.belief[r, c] = np.exp(-0.5 * dist)
    env_filter.belief /= env_filter.belief.sum()

    true_pos = (sr, sc)
    total_reward = 0
    done = False
    steps = 0
    localizations_correct = 0
    belief轨迹 = []

    eps = EPSILON if train else 0.05

    while not done and steps < MAX_STEPS:
        action = agent.choose_action(env_filter.belief, eps)

        # True environment transition
        dr, dc = ACTIONS[action]
        nr, nc = true_pos[0] + dr, true_pos[1] + dc
        if in_bounds(nr, nc) and not is_wall(nr, nc):
            true_pos = (nr, nc)
        # else stays

        # Sensor observation
        obs = sensor_reading(true_pos, noise_level)

        # Reward
        if true_pos == GOAL:
            reward = 50.0
            done = True
        else:
            reward = -0.5

        # Filter predict + update
        env_filter.predict(action)
        env_filter.update(obs, noise_level)

        # Track localization accuracy
        map_est = env_filter.most_likely()
        if map_est == true_pos:
            localizations_correct += 1

        belief轨迹.append(env_filter.belief.copy())

        if train:
            agent.update(env_filter.belief, action, reward, env_filter.belief, done)

        total_reward += reward
        steps += 1

    accuracy = localizations_correct / max(steps, 1)
    return total_reward, steps, accuracy, belief轨迹, env_filter, true_pos


def main():
    print("=" * 60)
    print("EXPERIMENT 38: POMDP Robot Navigation under Partial Observability")
    print("=" * 60)
    print(f"\nGrid: {GRID_ROWS}x{GRID_COLS}")
    print(f"Goal: {GOAL}")
    print(f"Walls: {len(WALLS)}")
    print(f"Landmarks: {len(LANDMARKS)}")
    print(f"Particles: {NUM_PARTICLES}")
    print(f"Sensor range: {SENSOR_RANGE}\n")

    noise_levels = [0.1, 0.3, 0.5]
    results = {}

    for noise in noise_levels:
        print(f"\n--- Noise Level: {noise:.1f} ---")
        env_filter = BayesianFilter()
        agent = QMDPAgent()

        reward_curve = []
        accuracy_curve = []

        for ep in range(NUM_EPISODES):
            r, s, acc, _, _, _ = run_episode(env_filter, agent, noise, train=True)
            reward_curve.append(r)
            accuracy_curve.append(acc)

            if ep % 100 == 0:
                recent_r = np.mean(reward_curve[-50:])
                recent_a = np.mean(accuracy_curve[-50:])
                print(f"  Ep {ep:4d}: avg_reward={recent_r:6.1f}, "
                      f"localization_acc={recent_a:.3f}")

        # Final evaluation
        final_rewards = []
        final_accs = []
        for _ in range(50):
            r, s, acc, _, _, _ = run_episode(env_filter, agent, noise, train=False)
            final_rewards.append(r)
            final_accs.append(acc)

        results[noise] = {
            "reward_curve": reward_curve,
            "accuracy_curve": accuracy_curve,
            "final_reward": np.mean(final_rewards),
            "final_accuracy": np.mean(final_accs),
        }

        print(f"  Final: reward={results[noise]['final_reward']:.1f}, "
              f"accuracy={results[noise]['final_accuracy']:.3f}")

    # --- Plot 1: Belief distribution heatmap ---
    print("\nGenerating belief heatmap...")
    env_filter = BayesianFilter()
    agent = QMDPAgent()
    # Train briefly
    for _ in range(200):
        run_episode(env_filter, agent, 0.2, train=True)
    # Run one episode and capture final belief
    _, _, _, belief_trace, env_filter_final, true_pos_final = run_episode(
        env_filter, agent, 0.2, train=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sample_steps = [0, len(belief_trace)//2, -1]
    step_labels = ["Start", "Middle", "End"]

    for idx, (si, label) in enumerate(zip(sample_steps, step_labels)):
        ax = axes[idx]
        b = belief_trace[si]
        im = ax.imshow(b, cmap="YlOrRd", origin="upper", vmin=0)
        ax.set_title(f"Belief Distribution ({label})", fontsize=11)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Mark true position at this step
        if si == -1:
            tp = true_pos_final
        else:
            tp = true_pos_final  # approximate
        ax.plot(tp[1], tp[0], "g*", markersize=15, label="True pos")

        # Mark goal
        ax.plot(GOAL[1], GOAL[0], "b^", markersize=12, label="Goal")

        # Mark walls
        for wr, wc in WALLS:
            ax.plot(wc, wr, "ks", markersize=8)

        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Belief Distribution Evolution (Noise=0.2)", fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), "exp38_belief_heatmap.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- Plot 2: Localization accuracy across noise levels ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy curves
    ax = axes[0]
    colors = ["green", "orange", "red"]
    for noise, color in zip(noise_levels, colors):
        acc_curve = results[noise]["accuracy_curve"]
        window = 30
        if len(acc_curve) >= window:
            smoothed = np.convolve(acc_curve, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(acc_curve)), smoothed,
                    color=color, linewidth=2, label=f"Noise={noise:.1f}")
        ax.plot(acc_curve, alpha=0.15, color=color, linewidth=0.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Localization Accuracy", fontsize=12)
    ax.set_title("Localization Accuracy over Training", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Final comparison bar chart
    ax = axes[1]
    final_accs = [results[n]["final_accuracy"] for n in noise_levels]
    bars = ax.bar([f"Noise={n:.1f}" for n in noise_levels], final_accs,
                  color=colors, edgecolor="black")
    ax.set_ylabel("Final Localization Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Sensor Noise", fontsize=12)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), "exp38_accuracy_comparison.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # --- Plot 3: Reward curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for noise, color in zip(noise_levels, colors):
        rc = results[noise]["reward_curve"]
        window = 30
        if len(rc) >= window:
            smoothed = np.convolve(rc, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(rc)), smoothed,
                    color=color, linewidth=2, label=f"Noise={noise:.1f}")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Navigation Reward Curves per Noise Level", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(os.path.dirname(__file__), "exp38_reward_curves.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for noise in noise_levels:
        print(f"  Noise={noise:.1f}: accuracy={results[noise]['final_accuracy']:.3f}, "
              f"reward={results[noise]['final_reward']:.1f}")

    print("\nExperiment 38 complete.")


if __name__ == "__main__":
    main()
