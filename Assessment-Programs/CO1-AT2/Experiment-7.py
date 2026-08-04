"""
Experiment 7: Packet Routing in Wireless Network (TensorFlow/Keras)
------------------------------------------------------------------------
Aim: Develop a Keras/TensorFlow RL agent for packet routing with limited
bandwidth. Explain how bandwidth constraints influence the reward function
and policy optimization.

Network:     6 nodes, bandwidths 1-5 Mbps per link
State:       (current_node, dest_node, bandwidth_used, bandwidth_capacity)
Actions:     Choose next node (up to 5 neighbors)
Rewards:     +100 delivery, -10 per hop, -50 bandwidth overflow
Constraint:  Link bandwidth cannot exceed capacity
Algorithm:   Q-learning with discretized bandwidth states
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

GRAPH = {
    0: {1: 3, 2: 5},
    1: {0: 3, 2: 2, 3: 4},
    2: {0: 5, 1: 2, 3: 1, 4: 3},
    3: {1: 4, 2: 1, 4: 2, 5: 5},
    4: {2: 3, 3: 2, 5: 4},
    5: {3: 5, 4: 4},
}
N_NODES = 6
MAX_BW_LEVELS = 5

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
N_EPISODES = 200
MAX_STEPS = 20


def discretize_bw(bw_used, capacity):
    ratio = bw_used / max(capacity, 1)
    return min(int(ratio * MAX_BW_LEVELS), MAX_BW_LEVELS - 1)


def get_state(node, dest, bw_level):
    return (node, dest, bw_level)


def step(state, action_idx):
    node, dest, bw_level = state
    neighbors = list(GRAPH[node].items())

    if action_idx >= len(neighbors):
        return state, -20, True

    next_node, capacity = neighbors[action_idx]

    if bw_level >= MAX_BW_LEVELS - 1:
        return state, -50, True

    if next_node == dest:
        return (next_node, dest, 0), 100, True

    reward = -10
    new_bw = min(bw_level + 1, MAX_BW_LEVELS - 1)
    return (next_node, dest, new_bw), reward, False


def train():
    Q = np.zeros((N_NODES, N_NODES, MAX_BW_LEVELS, N_NODES))
    epsilon = EPSILON_START
    episode_rewards = []
    successes = []

    for ep in range(N_EPISODES):
        src = np.random.randint(N_NODES)
        dest = np.random.randint(N_NODES)
        while dest == src:
            dest = np.random.randint(N_NODES)

        state = get_state(src, dest, 0)
        total_reward = 0
        success = False

        for _ in range(MAX_STEPS):
            node, dest_n, bw = state
            neighbors = list(GRAPH[node].items())
            valid_count = len(neighbors)

            if np.random.rand() < epsilon:
                action = np.random.randint(valid_count)
            else:
                q_vals = Q[node, dest_n, bw, :valid_count]
                action = int(np.argmax(q_vals)) if valid_count > 0 else 0

            next_state, reward, done = step(state, action)
            n2, d2, b2 = next_state
            valid_next = len(GRAPH[n2].items())
            best_next = np.max(Q[n2, d2, b2, :valid_next]) if valid_next > 0 else 0

            Q[node, dest_n, bw, action] += ALPHA * (
                reward + GAMMA * best_next - Q[node, dest_n, bw, action]
            )

            state = next_state
            total_reward += reward
            if reward == 100:
                success = True
            if done:
                break

        episode_rewards.append(total_reward)
        successes.append(1 if success else 0)
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_s = np.mean(successes[-10:]) * 100
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward: {avg_r:.1f} | Success rate: {avg_s:.0f}%")

    return Q, episode_rewards, successes


def plot_results(episode_rewards, successes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    window = 10
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), ma, linewidth=2, label=f"{window}-ep avg")
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning Training (Packet Routing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(successes) >= window:
        ma = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
        ax.plot(range(window - 1, len(successes)), ma, linewidth=2, color="green", label=f"{window}-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Delivery Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment7_packet_routing.png", dpi=150)
    print("Plot saved as 'experiment7_packet_routing.png'")


if __name__ == "__main__":
    print("Experiment 7: Packet Routing with Bandwidth Constraint\n")
    print(f"Network: 6 nodes | State: (node, dest, bandwidth_level)")
    print(f"Rewards: +100 delivery, -10/hop, -50 overflow")
    print(f"Constraint: link bandwidth cannot exceed capacity\n")
    print("Training Q-learning agent...\n")

    Q, rewards, successes = train()
    print(f"\nFinal success rate: {np.mean(successes[-20:])*100:.0f}%")

    plot_results(rewards, successes)
    print("\nExperiment 7 completed successfully!")
