"""
Experiment 5: Warehouse Robot with 30-Minute Battery Constraint
------------------------------------------------------------------------
Aim: Design an RL framework for a warehouse robot that completes maximum
deliveries within a 30-minute battery limit.

Grid:        5x5 warehouse with 3 delivery locations
State:       (row, col, battery_level)  → 5x5x5 = 125 states
Battery:     20 minutes (discretized to 5 levels)
Actions:     N, S, E, W, Deliver (5 actions)
Rewards:     +50 per delivery, -1 per move, -200 battery death
Constraint:  Battery depletes each step; must deliver before death
Algorithm:   Q-learning
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

ROWS, COLS = 5, 5
BATTERY_LEVELS = 5  # 20,15,10,5,0 minutes
BATTERY_MAP = {4: 20, 3: 15, 2: 10, 1: 5, 0: 0}
START = (0, 0)
DEPOT = (0, 0)
DELIVERY_LOCATIONS = {(1, 3), (3, 3), (4, 2)}
OBSTACLES = {(2, 1), (2, 2)}

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # N, S, W, E, Deliver
ACTION_NAMES = ["N", "S", "W", "E", "Deliver"]
N_ACTIONS = len(ACTIONS)

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
N_EPISODES = 300
MAX_STEPS = 20


def get_battery_level_idx(minutes):
    if minutes >= 20: return 4
    if minutes >= 15: return 3
    if minutes >= 10: return 2
    if minutes >= 5: return 1
    return 0


def step(state, action_idx):
    r, c, bat_idx = state
    minutes = BATTERY_MAP[bat_idx]

    if action_idx == 4:  # Deliver
        if (r, c) in DELIVERY_LOCATIONS and minutes >= 5:
            return (r, c, max(bat_idx - 1, 0)), 50, False
        return (r, c, bat_idx), -5, False  # invalid delivery

    dr, dc = ACTIONS[action_idx]
    nr, nc = r + dr, c + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in OBSTACLES:
        new_bat = max(bat_idx - 1, 0)
        if new_bat == 0:
            return (nr, nc, 0), -200, True  # battery death
        return (nr, nc, new_bat), -1, False
    return (r, c, bat_idx), -1, False  # wall/bump


def train():
    Q = np.zeros((ROWS, COLS, BATTERY_LEVELS, N_ACTIONS))
    epsilon = EPSILON_START
    episode_rewards = []
    deliveries_per_episode = []

    for ep in range(N_EPISODES):
        state = START
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
            if done:
                break

        episode_rewards.append(total_reward)
        deliveries_per_episode.append(deliveries)
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_d = np.mean(deliveries_per_episode[-10:])
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward: {avg_r:.1f} | Avg deliveries: {avg_d:.1f}")

    return Q, episode_rewards, deliveries_per_episode


def plot_results(episode_rewards, deliveries):
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
        ax.plot(range(window - 1, len(deliveries)), ma, linewidth=2, label=f"{window}-ep avg")
    ax.plot(deliveries, alpha=0.3, label="Deliveries per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Deliveries Completed")
    ax.set_title("Deliveries vs Battery Constraint")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment5_warehouse_robot.png", dpi=150)
    print("Plot saved as 'experiment5_warehouse_robot.png'")


if __name__ == "__main__":
    print("Experiment 5: Warehouse Robot with Battery Constraint\n")
    print(f"Grid: {ROWS}x{COLS} | Battery: {BATTERY_MAP[4]} min (5 levels)")
    print(f"Deliveries: {DELIVERY_LOCATIONS} | Obstacles: {OBSTACLES}")
    print(f"Deliver cost: 5 min battery | Move cost: 5 min battery\n")
    print("Training Q-learning agent...\n")

    Q, rewards, deliveries = train()
    plot_results(rewards, deliveries)
    print("\nExperiment 5 completed successfully!")
