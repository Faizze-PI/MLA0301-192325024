"""
Experiment 39: RL for Healthcare Management (Patient Scheduling)
=================================================================
Patients arrive via Poisson process with varying urgency and treatment times.
Limited resources: 4 doctors available.
State = waiting queue + resource availability.
Action = which patient to treat next.
Reward = -(weighted wait time) - urgent penalty (5x weight for urgent patients).
Q-learning agent vs FIFO and priority-only baselines.

Outputs:
  - Weighted average wait time comparison (bar chart)
  - Urgent patient penalty comparison
  - Learning curve for Q-learning agent
  - Queue length distribution over episodes
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random
from collections import deque

# ---------------------------------------------------------------------------
# Simulation Parameters
# ---------------------------------------------------------------------------
NUM_DOCTORS = 4
MAX_QUEUE = 10
SIMULATION_STEPS = 500
ARRIVAL_RATE = 0.3
URGENT_PROB = 0.25
MIN_TREATMENT = 2
MAX_TREATMENT = 8
URGENT_PENALTY_WEIGHT = 5.0

NUM_EPISODES = 3000
GAMMA = 0.95
ALPHA = 0.15
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997

MAX_QUEUE_BINS = 6


class Patient:
    _id_counter = 0

    def __init__(self, arrival_time, urgency, treatment_time):
        Patient._id_counter += 1
        self.id = Patient._id_counter
        self.arrival_time = arrival_time
        self.urgency = urgency
        self.treatment_time = treatment_time
        self.start_time = None
        self.wait_time = 0

    def __repr__(self):
        u = "URG" if self.urgency == 2 else "NRM"
        return f"P{self.id}({u},t={self.treatment_time})"


class HospitalEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        Patient._id_counter = 0
        self.time = 0
        self.queue = deque()
        self.doctors = [None] * NUM_DOCTORS
        self.total_patients = 0
        self.total_wait = 0
        self.urgent_wait_sum = 0
        self.urgent_count = 0
        self.total_weighted_wait = 0
        self.patients_treated = 0
        self.patients_lost = 0
        return self._state()

    def _state(self):
        q_len = min(len(self.queue), MAX_QUEUE_BINS - 1)
        doctors_busy = sum(1 for d in self.doctors if d is not None)
        urgent_in_queue = sum(1 for p in self.queue if p.urgency == 2)
        urgent_bin = min(urgent_in_queue, 3)
        return (q_len, doctors_busy, urgent_bin)

    def _arrival(self):
        if random.random() < ARRIVAL_RATE:
            urgency = 2 if random.random() < URGENT_PROB else 1
            treatment = random.randint(MIN_TREATMENT, MAX_TREATMENT)
            patient = Patient(self.time, urgency, treatment)
            if len(self.queue) < MAX_QUEUE:
                self.queue.append(patient)
                self.total_patients += 1
            else:
                self.patients_lost += 1

    def _treat(self, patient_idx):
        if patient_idx >= len(self.queue):
            return False
        free_doc = None
        for i, d in enumerate(self.doctors):
            if d is None:
                free_doc = i
                break
        if free_doc is None:
            return False
        patient = self.queue[patient_idx]
        wait = self.time - patient.arrival_time
        patient.wait_time = wait
        patient.start_time = self.time
        self.doctors[free_doc] = patient
        self.total_wait += wait
        self.total_weighted_wait += wait * patient.urgency
        if patient.urgency == 2:
            self.urgent_wait_sum += wait
            self.urgent_count += 1
        self.patients_treated += 1
        self.queue.remove(patient)
        return True

    def _advance_time(self):
        self.time += 1
        for i in range(NUM_DOCTORS):
            if self.doctors[i] is not None:
                self.doctors[i].treatment_time -= 1
                if self.doctors[i].treatment_time <= 0:
                    self.doctors[i] = None

    def step(self, action):
        self._arrival()
        reward = 0.0
        done = self.time >= SIMULATION_STEPS
        if 0 <= action < len(self.queue):
            success = self._treat(action)
            if success:
                reward += 2.0
            else:
                reward -= 1.0
        for p in self.queue:
            reward -= 0.1 * p.urgency
        self._advance_time()
        if done and self.total_patients > 0:
            avg_ww = self.total_weighted_wait / max(self.patients_treated, 1)
            reward -= avg_ww * 0.1
        return self._state(), reward, done


class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON_START

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, queue_len, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, max(queue_len - 1, 0))
        best_action = 0
        best_q = float("-inf")
        for a in range(max(queue_len, 1)):
            q = self.get_q(state, a)
            if q > best_q:
                best_q = q
                best_action = a
        return best_action

    def update(self, state, action, reward, next_state, done):
        best_next = max(self.get_q(next_state, a) for a in range(MAX_QUEUE))
        target = reward + (0.0 if done else GAMMA * best_next)
        old_q = self.get_q(state, action)
        self.q_table[(state, action)] = old_q + ALPHA * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def fifo_policy(queue, doctors):
    if len(queue) > 0 and any(d is None for d in doctors):
        return 0
    return -1


def priority_policy(queue, doctors):
    if not queue or not any(d is None for d in doctors):
        return -1
    best_idx = 0
    best_urgency = 0
    best_arrival = float("inf")
    for i, p in enumerate(queue):
        if p.urgency > best_urgency or (p.urgency == best_urgency and p.arrival_time < best_arrival):
            best_urgency = p.urgency
            best_arrival = p.arrival_time
            best_idx = i
    return best_idx


def train_q_learning():
    agent = QLearningAgent()
    env = HospitalEnv()
    reward_curve = []
    queue_lengths = []

    print("Training Q-learning agent for patient scheduling...")
    print(f"{'Episode':>8} | {'Reward':>8} | {'Patients':>8} | {'AvgWait':>8} | {'Epsilon':>7}")
    print("-" * 60)

    for ep in range(NUM_EPISODES):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            queue_len = len(env.queue)
            action = agent.choose_action(state, queue_len)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
        agent.decay_epsilon()
        reward_curve.append(ep_reward)
        avg_wait = env.total_wait / max(env.patients_treated, 1)
        queue_lengths.append(len(env.queue))
        if ep % 300 == 0 or ep == NUM_EPISODES - 1:
            recent = reward_curve[-50:]
            print(f"{ep:8d} | {np.mean(recent):8.1f} | {env.patients_treated:8d} | "
                  f"{avg_wait:8.1f} | {agent.epsilon:7.3f}")

    return agent, reward_curve, queue_lengths


def evaluate_policy(env, policy_fn, agent=None, use_agent=False):
    env.reset()
    wait_distribution = []
    for step in range(SIMULATION_STEPS):
        env._arrival()
        if use_agent:
            state = env._state()
            queue_len = len(env.queue)
            action = agent.choose_action(state, queue_len, epsilon=0.05)
        else:
            action = policy_fn(list(env.queue), env.doctors)
        if 0 <= action < len(env.queue):
            env._treat(action)
        for p in env.queue:
            wait_distribution.append(env.time - p.arrival_time)
        env._advance_time()
    avg_wait = env.total_wait / max(env.patients_treated, 1)
    avg_urgent = env.urgent_wait_sum / max(env.urgent_count, 1) if env.urgent_count > 0 else 0
    weighted_avg = env.total_weighted_wait / max(env.patients_treated, 1)
    return {
        "avg_wait": avg_wait,
        "urgent_wait": avg_urgent,
        "weighted_avg_wait": weighted_avg,
        "treated": env.patients_treated,
        "lost": env.patients_lost,
        "wait_dist": wait_distribution,
    }


def plot_results(results, reward_curve, queue_lengths):
    policies = ["FIFO", "Priority", "Q-Learning"]
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    weighted_waits = [results[p]["weighted_avg_wait"] for p in policies]
    bars = ax.bar(policies, weighted_waits, color=colors, edgecolor="black", width=0.5)
    ax.set_ylabel("Weighted Avg Wait Time", fontsize=12)
    ax.set_title("Weighted Average Wait Time Comparison", fontsize=13)
    for bar, val in zip(bars, weighted_waits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    urgent_waits = [results[p]["urgent_wait"] for p in policies]
    bars = ax.bar(policies, urgent_waits, color=colors, edgecolor="black", width=0.5)
    ax.set_ylabel("Avg Urgent Patient Wait", fontsize=12)
    ax.set_title("Urgent Patient Wait Time (5x penalty)", fontsize=13)
    for bar, val in zip(bars, urgent_waits):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), "exp39_wait_comparison.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    fig, ax = plt.subplots(figsize=(10, 5))
    window = 100
    if len(reward_curve) >= window:
        smoothed = np.convolve(reward_curve, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(reward_curve)), smoothed, color="darkgreen", linewidth=2,
                label="Smoothed")
    ax.plot(reward_curve, alpha=0.15, color="green", linewidth=0.5)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Q-Learning Patient Scheduling: Training Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), "exp39_learning_curve.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, color in zip(policies, colors):
        wd = results[policy]["wait_dist"]
        if wd:
            ax.hist(wd, bins=30, alpha=0.4, color=color, label=policy, density=True)
    ax.set_xlabel("Wait Time", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Wait Time Distribution by Policy", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = os.path.join(os.path.dirname(__file__), "exp39_wait_distribution.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")

    fig, ax = plt.subplots(figsize=(8, 5))
    treated = [results[p]["treated"] for p in policies]
    lost = [results[p]["lost"] for p in policies]
    x = np.arange(len(policies))
    ax.bar(x - 0.2, treated, 0.35, label="Treated", color="#2ecc71", edgecolor="black")
    ax.bar(x + 0.2, lost, 0.35, label="Lost", color="#e74c3c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Patients Treated vs Lost", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out4 = os.path.join(os.path.dirname(__file__), "exp39_patients_outcome.png")
    fig.savefig(out4, dpi=150)
    plt.close(fig)
    print(f"Saved: {out4}")


def main():
    print("=" * 60)
    print("EXPERIMENT 39: RL for Healthcare Patient Scheduling")
    print("=" * 60)
    print(f"\nSim: {SIMULATION_STEPS} steps, {NUM_DOCTORS} doctors")
    print(f"Arrival rate={ARRIVAL_RATE}, Urgent prob={URGENT_PROB}")
    print(f"Urgent penalty weight={URGENT_PENALTY_WEIGHT}")
    print(f"Q-learning: episodes={NUM_EPISODES}, gamma={GAMMA}\n")

    agent, reward_curve, queue_lengths = train_q_learning()

    env = HospitalEnv()
    results = {}
    print("\n--- Evaluating Policies ---")
    for policy_name, policy_fn, use_agent in [
        ("FIFO", fifo_policy, False),
        ("Priority", priority_policy, False),
        ("Q-Learning", None, True),
    ]:
        r = evaluate_policy(env, policy_fn, agent=agent, use_agent=use_agent)
        results[policy_name] = r
        print(f"  {policy_name:12s}: avg_wait={r['avg_wait']:.1f}, "
              f"urgent_wait={r['urgent_wait']:.1f}, "
              f"weighted={r['weighted_avg_wait']:.1f}, "
              f"treated={r['treated']}, lost={r['lost']}")

    plot_results(results, reward_curve, queue_lengths)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best = min(results, key=lambda p: results[p]["weighted_avg_wait"])
    print(f"  Best policy (weighted wait): {best} "
          f"({results[best]['weighted_avg_wait']:.1f})")
    best_urgent = min(results, key=lambda p: results[p]["urgent_wait"])
    print(f"  Best for urgent patients: {best_urgent} "
          f"({results[best_urgent]['urgent_wait']:.1f})")
    print("\nExperiment 39 complete.")


if __name__ == "__main__":
    main()
