"""
Experiment 40: RL for Personalized Education / Intelligent Tutoring
====================================================================
Custom simulation of a student learning multiple topics.
Simplified BKT (Bayak Tracking Knowledge) mastery probability per topic.
Action = which topic/difficulty to present next.
Reward = learning gain - frustration penalty.
5 topics, Q-learning over discretized mastery bins.
Compare RL-tutored mastery curve vs fixed-curriculum baseline.

Outputs:
  - Per-topic mastery curves (RL vs fixed curriculum)
  - Overall knowledge gain comparison
  - Student frustration over time
  - Topic coverage analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random

# ---------------------------------------------------------------------------
# Student / Tutoring Parameters
# ---------------------------------------------------------------------------
NUM_TOPICS = 5
TOPIC_NAMES = ["Algebra", "Calculus", "Statistics", "Logic", "Geometry"]
DIFFICULTY_LEVELS = 3  # Easy, Medium, Hard
MASTERY_BINS = 10      # Discretize mastery into 0..9
LEARNING_RATE = 0.15   # P(skill increase) per practice
FORGETTING_RATE = 0.02  # P(skill decay) per idle step
FRUSTRATION_THRESHOLD = 0.7  # High difficulty when mastery < this
REWARD_GAIN = 2.0
REWARD_MASTERY = 10.0
PENALTY_FRUSTRATION = -3.0
PENALTY_EASY_BOREDOM = -1.0

GAMMA = 0.95
ALPHA = 0.15
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
NUM_EPISODES = 500
STEPS_PER_EPISODE = 60

# Q-learning state: (mastery_bin_per_topic, chosen_topic, difficulty)
# Simplified: (avg_mastery_bin, chosen_topic) -> difficulty
STATE_MASTERY_BINS = MASTERY_BINS


class Student:
    """Simulated student with per-topic mastery."""

    def __init__(self):
        self.mastery = np.zeros(NUM_TOPICS)  # 0..1 continuous
        self.frustration = np.zeros(NUM_TOPICS)
        self.total_learning = 0.0

    def reset(self):
        # Random initial knowledge
        self.mastery = np.random.uniform(0.0, 0.3, NUM_TOPICS)
        self.frustration = np.zeros(NUM_TOPICS)
        self.total_learning = 0.0

    def practice(self, topic_idx, difficulty):
        """
        Student practices a topic at given difficulty.
        Returns (learning_gain, frustration_change).
        """
        m = self.mastery[topic_idx]
        # Effective learning depends on difficulty vs mastery
        diff_factor = (difficulty + 1) / DIFFICULTY_LEVELS  # 1/3, 2/3, 1

        if m < FRUSTRATION_THRESHOLD and difficulty >= 2:
            # Too hard -> frustration
            frust_change = 0.15
            learn_gain = 0.02  # minimal learning
        elif m > 0.8 and difficulty == 0:
            # Too easy -> boredom
            frust_change = 0.05
            learn_gain = 0.01
        else:
            # Optimal zone
            frust_change = -0.05  # frustration decreases
            learn_gain = LEARNING_RATE * (1 - m) * (0.5 + 0.5 * diff_factor)

        # Add noise
        learn_gain += np.random.normal(0, 0.02)
        learn_gain = max(0, min(0.3, learn_gain))

        self.mastery[topic_idx] = min(1.0, self.mastery[topic_idx] + learn_gain)
        self.frustration[topic_idx] = max(0, min(1.0,
                                                  self.frustration[topic_idx] + frust_change))
        self.total_learning += learn_gain

        return learn_gain, frust_change

    def forget(self):
        """Slight forgetting for topics not practiced."""
        self.mastery -= FORGETTING_RATE
        self.mastery = np.maximum(0, self.mastery)


class TutorEnv:
    """Tutoring environment with student simulation."""

    def __init__(self):
        self.student = Student()
        self.reset()

    def reset(self):
        self.student.reset()
        self.step_count = 0
        self.topic_practice_count = np.zeros(NUM_TOPICS)
        self.mastery_history = []
        self.frustration_history = []
        return self._state()

    def _state(self):
        """Discretized state for Q-learning."""
        mastery_bins = tuple(np.clip(
            (self.student.mastery * MASTERY_BINS).astype(int), 0, MASTERY_BINS - 1))
        avg_bin = int(np.mean(mastery_bins))
        # Frustration level
        avg_frust = int(np.mean(self.student.frustration) * 3)  # 0,1,2
        return (avg_bin, avg_frust)

    def _all_mastered(self):
        return all(m >= 0.9 for m in self.student.mastery)

    def step(self, topic_idx, difficulty):
        """
        Present topic at difficulty to student.
        Returns: (next_state, reward, done)
        """
        self.step_count += 1
        self.topic_practice_count[topic_idx] += 1

        # Student practices
        learn_gain, frust_change = self.student.practice(topic_idx, difficulty)

        # Compute reward
        reward = REWARD_GAIN * learn_gain

        # Mastery bonus
        if self.student.mastery[topic_idx] >= 0.95:
            reward += REWARD_MASTERY

        # Frustration penalty
        if frust_change > 0:
            reward += PENALTY_FRUSTRATION * frust_change

        # Boredom penalty (too easy)
        if difficulty == 0 and self.student.mastery[topic_idx] > 0.7:
            reward += PENALTY_EASY_BOREDOM

        # Record
        self.mastery_history.append(self.student.mastery.copy())
        self.frustration_history.append(np.mean(self.student.frustration))

        # Forgetting for non-practiced topics
        self.student.forget()

        done = self.step_count >= STEPS_PER_EPISODE or self._all_mastered()
        return self._state(), reward, done


# ---------------------------------------------------------------------------
# Q-Learning Tutor Agent
# ---------------------------------------------------------------------------
class QLearningTutor:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON_START

    def _state_actions(self, state, topic_idx):
        """Return Q for each difficulty level for given topic."""
        return [(self.q_table.get((state, topic_idx, d), 0.0), d)
                for d in range(DIFFICULTY_LEVELS)]

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            topic = random.randint(0, NUM_TOPICS - 1)
            difficulty = random.randint(0, DIFFICULTY_LEVELS - 1)
            return topic, difficulty

        # Try all topic-difficulty combos
        best_topic = 0
        best_diff = 0
        best_q = float("-inf")

        for t in range(NUM_TOPICS):
            for d in range(DIFFICULTY_LEVELS):
                q = self.q_table.get((state, t, d), 0.0)
                if q > best_q:
                    best_q = q
                    best_topic = t
                    best_diff = d

        return best_topic, best_diff

    def update(self, state, topic, difficulty, reward, next_state, done):
        key = (state, topic, difficulty)
        best_next = max(
            self.q_table.get((next_state, t, d), 0.0)
            for t in range(NUM_TOPICS) for d in range(DIFFICULTY_LEVELS)
        )
        target = reward + (0.0 if done else GAMMA * best_next)
        old_q = self.q_table.get(key, 0.0)
        self.q_table[key] = old_q + ALPHA * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ---------------------------------------------------------------------------
# Fixed Curriculum Baseline
# ---------------------------------------------------------------------------
def fixed_curriculum_step(env):
    """Cycle through topics in order, easy-to-hard progression."""
    step = env.step_count
    topic_idx = step % NUM_TOPICS
    # Progress difficulty every 10 steps
    difficulty = min(2, step // 10 % DIFFICULTY_LEVELS)
    return topic_idx, difficulty


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_rl_tutor():
    """Train Q-learning tutor."""
    agent = QLearningTutor()
    env = TutorEnv()
    reward_curve = []
    mastery_snapshots = []

    print("Training RL tutor...")
    print(f"{'Episode':>8} | {'Reward':>8} | {'AvgMast':>8} | {'Epsilon':>7}")
    print("-" * 45)

    for ep in range(NUM_EPISODES):
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:
            topic, difficulty = agent.choose_action(state)
            next_state, reward, done = env.step(topic, difficulty)
            agent.update(state, topic, difficulty, reward, next_state, done)
            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        reward_curve.append(ep_reward)

        avg_mastery = np.mean(env.student.mastery)
        mastery_snapshots.append(env.student.mastery.copy())

        if ep % 100 == 0 or ep == NUM_EPISODES - 1:
            recent = reward_curve[-50:]
            print(f"{ep:8d} | {np.mean(recent):8.1f} | {avg_mastery:8.3f} | "
                  f"{agent.epsilon:7.3f}")

    return agent, reward_curve, mastery_snapshots


def evaluate_fixed_curriculum(num_eval_episodes=50):
    """Evaluate fixed curriculum baseline."""
    all_mastery_curves = []
    all_rewards = []

    for _ in range(num_eval_episodes):
        env = TutorEnv()
        env.reset()
        mastery_curve = [env.student.mastery.copy()]
        total_reward = 0

        for step in range(STEPS_PER_EPISODE):
            topic, difficulty = fixed_curriculum_step(env)
            _, reward, done = env.step(topic, difficulty)
            total_reward += reward
            mastery_curve.append(env.student.mastery.copy())
            if done:
                break

        all_mastery_curves.append(mastery_curve)
        all_rewards.append(total_reward)

    return all_mastery_curves, all_rewards


def evaluate_rl_tutor(agent, num_eval_episodes=50):
    """Evaluate trained RL tutor."""
    all_mastery_curves = []
    all_rewards = []

    for _ in range(num_eval_episodes):
        env = TutorEnv()
        state = env.reset()
        mastery_curve = [env.student.mastery.copy()]
        total_reward = 0
        done = False

        while not done:
            topic, difficulty = agent.choose_action(state, epsilon=0.05)
            next_state, reward, done = env.step(topic, difficulty)
            total_reward += reward
            state = next_state
            mastery_curve.append(env.student.mastery.copy())

        all_mastery_curves.append(mastery_curve)
        all_rewards.append(total_reward)

    return all_mastery_curves, all_rewards


def plot_results(rl_mastery, rl_rewards, fixed_mastery, fixed_rewards,
                 reward_curve, rl_mastery_snapshots):
    """Generate all plots."""
    # --- Plot 1: Per-topic mastery curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_TOPICS))

    # RL tutor
    for t in range(NUM_TOPICS):
        # Average across episodes
        max_len = max(len(mc) for mc in rl_mastery)
        topic_mastery = np.zeros((len(rl_mastery), max_len))
        for i, mc in enumerate(rl_mastery):
            for j, m in enumerate(mc):
                topic_mastery[i, j] = m[t]
            topic_mastery[i, len(mc):] = m[t]  # pad with final value
        avg = np.mean(topic_mastery, axis=0)
        ax.plot(avg, color=colors[t], linewidth=2, label=f"{TOPIC_NAMES[t]} (RL)")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mastery", fontsize=12)
    ax.set_title("RL Tutor: Per-Topic Mastery", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Fixed curriculum
    ax = axes[1]
    for t in range(NUM_TOPICS):
        max_len = max(len(mc) for mc in fixed_mastery)
        topic_mastery = np.zeros((len(fixed_mastery), max_len))
        for i, mc in enumerate(fixed_mastery):
            for j, m in enumerate(mc):
                topic_mastery[i, j] = m[t]
            topic_mastery[i, len(mc):] = m[t]
        avg = np.mean(topic_mastery, axis=0)
        ax.plot(avg, color=colors[t], linewidth=2, linestyle="--",
                label=f"{TOPIC_NAMES[t]} (Fixed)")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mastery", fontsize=12)
    ax.set_title("Fixed Curriculum: Per-Topic Mastery", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), "exp40_mastery_curves.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- Plot 2: Overall knowledge gain comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    rl_final = [mc[-1] for mc in rl_mastery]  # final mastery per episode
    rl_avg_final = [np.mean(mc[-1]) for mc in rl_mastery]
    fixed_avg_final = [np.mean(mc[-1]) for mc in fixed_mastery]

    ax.hist(rl_avg_final, bins=20, alpha=0.6, color="green", label="RL Tutor", density=True)
    ax.hist(fixed_avg_final, bins=20, alpha=0.6, color="gray", label="Fixed Curriculum",
            density=True)
    ax.set_xlabel("Average Final Mastery", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Final Mastery", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Average reward comparison
    ax = axes[1]
    methods = ["Fixed Curriculum", "RL Tutor"]
    avg_rewards = [np.mean(fixed_rewards), np.mean(rl_rewards)]
    std_rewards = [np.std(fixed_rewards), np.std(rl_rewards)]
    colors_bar = ["gray", "green"]
    bars = ax.bar(methods, avg_rewards, yerr=std_rewards, color=colors_bar,
                  edgecolor="black", capsize=5)
    ax.set_ylabel("Average Episode Reward", fontsize=12)
    ax.set_title("Average Reward Comparison", fontsize=13)
    for bar, val in zip(bars, avg_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), "exp40_knowledge_gain.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # --- Plot 3: Frustration over time ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # RL tutor frustration
    rl_frust = []
    for ep_mastery in rl_mastery[:100]:
        frust = []
        for m in ep_mastery:
            frust.append(np.mean(np.maximum(0, 0.5 - m) * 2))
        rl_frust.append(frust)

    max_len = max(len(f) for f in rl_frust)
    frust_array = np.zeros((len(rl_frust), max_len))
    for i, f in enumerate(rl_frust):
        frust_array[i, :len(f)] = f
        frust_array[i, len(f):] = f[-1] if f else 0
    avg_frust_rl = np.mean(frust_array, axis=0)

    ax.plot(avg_frust_rl, color="green", linewidth=2, label="RL Tutor")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Frustration threshold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Frustration Level", fontsize=12)
    ax.set_title("Student Frustration Over Time", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out3 = os.path.join(os.path.dirname(__file__), "exp40_frustration.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")

    # --- Plot 4: Topic coverage heatmap ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # RL tutor topic selection frequency
    rl_topic_freq = np.zeros(NUM_TOPICS)
    for mc in rl_mastery:
        # Count transitions (simplified: episodes that reached mastery)
        pass

    # Use average final mastery as proxy
    rl_final_by_topic = np.zeros((len(rl_mastery), NUM_TOPICS))
    for i, mc in enumerate(rl_mastery):
        rl_final_by_topic[i] = mc[-1]

    avg_by_topic = np.mean(rl_final_by_topic, axis=0)
    fixed_final_by_topic = np.zeros((len(fixed_mastery), NUM_TOPICS))
    for i, mc in enumerate(fixed_mastery):
        fixed_final_by_topic[i] = mc[-1]
    fixed_avg_by_topic = np.mean(fixed_final_by_topic, axis=0)

    data = np.array([fixed_avg_by_topic, avg_by_topic])
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_TOPICS))
    ax.set_xticklabels(TOPIC_NAMES, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fixed", "RL Tutor"], fontsize=11)
    ax.set_title("Average Final Mastery by Topic", fontsize=13)

    # Annotate
    for i in range(2):
        for j in range(NUM_TOPICS):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if data[i, j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Mastery")
    plt.tight_layout()
    out4 = os.path.join(os.path.dirname(__file__), "exp40_topic_coverage.png")
    fig.savefig(out4, dpi=150)
    plt.close(fig)
    print(f"Saved: {out4}")

    # --- Plot 5: Learning curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 50
    if len(reward_curve) >= window:
        smoothed = np.convolve(reward_curve, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(reward_curve)), smoothed, color="darkgreen", linewidth=2,
                label="RL Tutor (smoothed)")
    ax.plot(reward_curve, alpha=0.15, color="green", linewidth=0.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("RL Tutor Training Learning Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out5 = os.path.join(os.path.dirname(__file__), "exp40_learning_curve.png")
    fig.savefig(out5, dpi=150)
    plt.close(fig)
    print(f"Saved: {out5}")


def main():
    print("=" * 60)
    print("EXPERIMENT 40: RL for Intelligent Tutoring System")
    print("=" * 60)
    print(f"\nTopics: {TOPIC_NAMES}")
    print(f"Difficulty levels: {DIFFICULTY_LEVELS}")
    print(f"Mastery bins: {MASTERY_BINS}")
    print(f"Q-learning: episodes={NUM_EPISODES}, steps/ep={STEPS_PER_EPISODE}")
    print(f"Gamma={GAMMA}, Alpha={ALPHA}\n")

    # Train RL tutor
    agent, reward_curve, mastery_snapshots = train_rl_tutor()

    # Evaluate RL tutor
    print("\nEvaluating RL tutor...")
    rl_mastery, rl_rewards = evaluate_rl_tutor(agent, num_eval_episodes=100)

    # Evaluate fixed curriculum
    print("Evaluating fixed curriculum baseline...")
    fixed_mastery, fixed_rewards = evaluate_fixed_curriculum(num_eval_episodes=100)

    # Print comparison
    print("\n--- Results ---")
    rl_avg = np.mean(rl_rewards)
    fixed_avg = np.mean(fixed_rewards)
    print(f"  RL Tutor:     avg_reward={rl_avg:.1f} +/- {np.std(rl_rewards):.1f}")
    print(f"  Fixed Curric: avg_reward={fixed_avg:.1f} +/- {np.std(fixed_rewards):.1f}")

    rl_final_mastery = np.mean([np.mean(mc[-1]) for mc in rl_mastery])
    fixed_final_mastery = np.mean([np.mean(mc[-1]) for mc in fixed_mastery])
    print(f"\n  RL Tutor final mastery:     {rl_final_mastery:.3f}")
    print(f"  Fixed Curric final mastery: {fixed_final_mastery:.3f}")

    improvement = (rl_final_mastery - fixed_final_mastery) / max(fixed_final_mastery, 0.01) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(rl_mastery, rl_rewards, fixed_mastery, fixed_rewards,
                 reward_curve, mastery_snapshots)

    # Per-topic breakdown
    print("\n--- Per-Topic Mastery Breakdown ---")
    print(f"  {'Topic':>12s} | {'RL Tutor':>10s} | {'Fixed':>10s} | {'Delta':>8s}")
    print("  " + "-" * 50)
    for t in range(NUM_TOPICS):
        rl_m = np.mean([mc[-1][t] for mc in rl_mastery])
        fix_m = np.mean([mc[-1][t] for mc in fixed_mastery])
        delta = rl_m - fix_m
        print(f"  {TOPIC_NAMES[t]:>12s} | {rl_m:10.3f} | {fix_m:10.3f} | {delta:+8.3f}")

    print("\nExperiment 40 complete.")


if __name__ == "__main__":
    main()
