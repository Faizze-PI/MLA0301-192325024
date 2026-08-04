"""
Experiment 10: Smart Energy Management System
------------------------------------------------------------------------
Aim: Implement an RL solution for a smart energy management system where
electricity consumption must remain below a specified threshold. Explain
how reward shaping, policy learning, and feasibility constraints improve
decision-making.

States:      (hour_of_day, consumption_level)  -> 24x5 = 120 states
Consumption: 5 levels (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
Actions:     Reduce, Maintain, Increase  -> 3 actions
Rewards:     +10 below threshold, -20 above threshold, -2 comfort cost
Constraint:  Consumption must stay below 80% threshold
Algorithm:   Q-learning with reward shaping
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

N_HOURS = 24
N_LEVELS = 5
N_ACTIONS = 3
ACTION_NAMES = ["Reduce", "Maintain", "Increase"]
THRESHOLD = 3

BASE_DEMAND = np.array([
    0.2, 0.15, 0.1, 0.1, 0.15, 0.25,
    0.4, 0.6, 0.7, 0.65, 0.6, 0.55,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.8,
    0.85, 0.75, 0.6, 0.45, 0.35, 0.25
])

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
N_EPISODES = 200
MAX_STEPS = 24


def get_level_idx(level):
    return min(max(int(level * 5), 0), 4)


def step(state, action):
    hour, level = state

    if action == 0:
        new_level = max(level - 1, 0)
    elif action == 2:
        new_level = min(level + 1, 4)
    else:
        new_level = level

    demand_push = 1 if BASE_DEMAND[hour] > 0.6 else 0
    new_level = min(new_level + demand_push, 4)

    if new_level < THRESHOLD:
        reward = 10
    elif new_level == THRESHOLD:
        reward = -5
    else:
        reward = -20

    if action == 0:
        reward -= 2

    potential = (THRESHOLD - new_level) * 2
    reward += potential * 0.1

    next_hour = (hour + 1) % N_HOURS
    done = (next_hour == 0)
    return (next_hour, new_level), reward, done


def train():
    Q = np.zeros((N_HOURS, N_LEVELS, N_ACTIONS))
    epsilon = EPSILON_START
    episode_rewards = []
    compliance_per_episode = []

    for ep in range(N_EPISODES):
        state = (0, get_level_idx(BASE_DEMAND[0]))
        total_reward = 0
        compliant_steps = 0

        for _ in range(MAX_STEPS):
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state[0], state[1]]))

            next_state, reward, done = step(state, action)
            best_next = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * best_next - Q[state[0], state[1], action]
            )
            state = next_state
            total_reward += reward
            if next_state[1] < THRESHOLD:
                compliant_steps += 1
            if done:
                break

        episode_rewards.append(total_reward)
        compliance_per_episode.append(compliant_steps / MAX_STEPS * 100)
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_c = np.mean(compliance_per_episode[-10:])
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward: {avg_r:.1f} | Compliance: {avg_c:.0f}%")

    return Q, episode_rewards, compliance_per_episode


def simulate_day(Q):
    levels = []
    hour = 0
    level = get_level_idx(BASE_DEMAND[0])
    actions_taken = []

    for _ in range(N_HOURS):
        action = int(np.argmax(Q[hour, level]))
        levels.append(level)
        actions_taken.append(ACTION_NAMES[action])
        next_state, _, _ = step((hour, level), action)
        hour, level = next_state

    return levels, actions_taken


def plot_results(Q, episode_rewards, compliance):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    window = 10
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), ma, linewidth=2, label=f"{window}-ep avg")
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if len(compliance) >= window:
        ma = np.convolve(compliance, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(compliance)), ma, linewidth=2, color="green")
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="100% compliance")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Compliance Rate (%)")
    ax.set_title("Threshold Compliance Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    levels, actions = simulate_day(Q)
    level_pct = [(l + 0.5) * 20 for l in levels]
    ax.plot(range(N_HOURS), level_pct, "o-", linewidth=2, color="blue")
    ax.axhline(y=THRESHOLD * 20, color="red", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD * 20}%)")
    ax.fill_between(range(N_HOURS), 0, THRESHOLD * 20, alpha=0.1, color="green", label="Safe zone")
    ax.fill_between(range(N_HOURS), THRESHOLD * 20, 100, alpha=0.1, color="red", label="Violation zone")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Consumption Level (%)")
    ax.set_title("Simulated Day with Learned Policy")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    policy_grid = np.argmax(Q, axis=2)
    im = ax.imshow(policy_grid.T, cmap="RdYlGn_r", aspect="auto")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Consumption Level")
    ax.set_title("Learned Policy (0=Reduce, 1=Maintain, 2=Increase)")
    ax.set_yticks(range(N_LEVELS))
    ax.set_yticklabels(["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2], label="Action")

    plt.tight_layout()
    plt.savefig("experiment10_energy_management.png", dpi=150)
    print("Plot saved as 'experiment10_energy_management.png'")


if __name__ == "__main__":
    print("Experiment 10: Smart Energy Management System\n")
    print(f"States: {N_HOURS} hours x {N_LEVELS} levels = {N_HOURS * N_LEVELS}")
    print(f"Actions: {ACTION_NAMES}")
    print(f"Threshold: {THRESHOLD * 20}% | Constraint: stay below threshold\n")
    print("Training Q-learning with reward shaping...\n")

    Q, rewards, compliance = train()
    print(f"\nFinal compliance rate: {np.mean(compliance[-20:]):.0f}%")

    plot_results(Q, rewards, compliance)
    print("\nExperiment 10 completed successfully!")
