"""
Experiment 9: Delivery Drone with No-Fly Zones and Battery Limits
------------------------------------------------------------------------
Aim: Design an RL framework for a delivery drone that maximizes successful
deliveries while avoiding no-fly zones and maintaining battery limits.
Explain MDP formulation, Bellman equation, and policy learning.

Grid:        4x4 with 1 no-fly zone, 2 delivery points
State:       (row, col, battery_level)  -> 4x4x3 = 48 states
Battery:     30 steps (3 levels: 30,15,0)
Actions:     N, S, E, W, Deliver (5 actions)
Rewards:     +50 delivery, -1 move, -100 no-fly, -200 battery death
Algorithm:   Q-learning
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

ROWS, COLS = 4, 4
BATTERY_LEVELS = 3
BAT_MAP = {2: 30, 1: 15, 0: 0}
START = (0, 0)
DELIVERIES = {(0, 1), (1, 0)}
NO_FLY = {(2, 2)}

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["N", "S", "W", "E", "Deliver"]
N_ACTIONS = 5

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.993
N_EPISODES = 500
MAX_STEPS = 30


def step(state, action_idx):
    r, c, bat_idx = state

    if action_idx == 4:  # Deliver
        if (r, c) in DELIVERIES and bat_idx > 0:
            return (r, c, max(bat_idx - 1, 0)), 50, False
        return (r, c, bat_idx), -5, False

    dr, dc = ACTIONS[action_idx]
    nr, nc = r + dr, c + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS:
        if (nr, nc) in NO_FLY:
            return (r, c, bat_idx), -100, False
        new_bat = max(bat_idx - 1, 0)
        if new_bat == 0:
            return (nr, nc, 0), -200, True
        return (nr, nc, new_bat), -1, False
    return (r, c, bat_idx), -1, False


def train():
    Q = np.zeros((ROWS, COLS, BATTERY_LEVELS, N_ACTIONS))
    epsilon = EPSILON_START
    episode_rewards = []
    deliveries_per_episode = []
    total_deliveries = 0

    for ep in range(N_EPISODES):
        state = (START[0], START[1], 2)  # full battery
        total_reward = 0
        deliveries = 0

        for _ in range(MAX_STEPS):
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state[0], state[1], state[2]]))

            next_state, reward, done = step(state, action)
            best_next = np.max(Q[next_state[0], next_state[1], next_state[2]])
            Q[state[0], state[1], state[2], action] += ALPHA * (
                reward + GAMMA * best_next - Q[state[0], state[1], state[2], action]
            )

            state = next_state
            total_reward += reward
            if reward == 50:
                deliveries += 1
                total_deliveries += 1
            if done:
                break

        episode_rewards.append(total_reward)
        deliveries_per_episode.append(deliveries)
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_d = np.mean(deliveries_per_episode[-10:])
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward: {avg_r:.1f} | Avg deliveries: {avg_d:.1f}")

    return Q, episode_rewards, deliveries_per_episode, total_deliveries


def plot_results(episode_rewards, deliveries, total):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
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

    ax = axes[1]
    if len(deliveries) >= window:
        ma = np.convolve(deliveries, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(deliveries)), ma, linewidth=2, color="green", label=f"{window}-ep avg")
    ax.plot(deliveries, alpha=0.3, label="Deliveries")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Deliveries Completed")
    ax.set_title("Deliveries vs No-Fly + Battery Constraints")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment9_delivery_drone.png", dpi=150)
    print("Plot saved as 'experiment9_delivery_drone.png'")


if __name__ == "__main__":
    print("Experiment 9: Delivery Drone with No-Fly Zones + Battery\n")
    print(f"Grid: {ROWS}x{COLS} | Battery: {BAT_MAP[2]} steps (3 levels)")
    print(f"Deliveries: {DELIVERIES} | No-fly zones: {NO_FLY}")
    print(f"Actions: {ACTION_NAMES}")
    print(f"Rewards: +50 delivery, -1 move, -100 no-fly, -200 battery death\n")
    print("Training Q-learning agent...\n")

    Q, rewards, deliveries, total = train()
    print(f"\nTotal deliveries across all episodes: {total}")

    plot_results(rewards, deliveries, total)
    print("\nExperiment 9 completed successfully!")
