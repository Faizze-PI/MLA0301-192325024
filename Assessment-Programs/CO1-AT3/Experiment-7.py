"""
Experiment 7: Packet Routing in Wireless Network (TensorFlow/Keras)
------------------------------------------------------------------------
Aim: Develop a Keras/TensorFlow RL agent for packet routing with limited
bandwidth. Explain how bandwidth constraints influence the reward function
and policy optimization.

Network:     6 nodes, bandwidths 1-5 Mbps per link
State:       (current_node, dest_node, bandwidth_used)  → continuous
Actions:     Choose next node (up to 5 neighbors)
Rewards:     +100 delivery, -10 per hop, -50 bandwidth overflow
Constraint:  Link bandwidth cannot exceed capacity
Algorithm:   DQN with Keras (fast direct model calls)
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 6-node network topology
# Adjacency list with bandwidth capacity
GRAPH = {
    0: {1: 3, 2: 5},
    1: {0: 3, 2: 2, 3: 4},
    2: {0: 5, 1: 2, 3: 1, 4: 3},
    3: {1: 4, 2: 1, 4: 2, 5: 5},
    4: {2: 3, 3: 2, 5: 4},
    5: {3: 5, 4: 4},
}
N_NODES = 6
MAX_NEIGHBORS = 5  # pad action space

# State: [current_node, dest_node, bandwidth_used/current_capacity]
STATE_SIZE = 3
ACTION_SIZE = MAX_NEIGHBORS  # index into neighbor list (padded)

ALPHA = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
N_EPISODES = 200
MAX_STEPS = 20
BATCH_SIZE = 32
REPLAY_SIZE = 500

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(ACTION_SIZE, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA), loss="mse")
    return model


def get_state(node, dest, bw_used, capacity):
    return np.array([node / N_NODES, dest / N_NODES, bw_used / max(capacity, 1)], dtype=np.float32)


def get_valid_actions(node, neighbors):
    """Return (action_index, actual_neighbor) pairs."""
    valid = list(GRAPH[node].items())
    return valid


def step(state, action_idx, neighbors_list):
    """Execute action. neighbors_list = [(neighbor, capacity), ...]"""
    current_node = int(state[0] * N_NODES)
    dest = int(state[1] * N_NODES)

    if action_idx >= len(neighbors_list):
        return state, -20, True  # invalid action

    next_node, capacity = neighbors_list[action_idx]
    bw_used = state[2] * capacity

    if bw_used > capacity:
        return state, -50, True  # bandwidth overflow

    if next_node == dest:
        return get_state(next_node, dest, 0, 1), 100, True  # delivered

    reward = -10  # hop penalty
    new_bw = min(bw_used + 1, capacity)
    next_state = get_state(next_node, dest, new_bw, capacity)

    return next_state, reward, False


class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.memory = deque(maxlen=REPLAY_SIZE)
        self.epsilon = EPSILON_START

    def act(self, state, valid_count):
        if np.random.rand() < self.epsilon:
            return np.random.randint(valid_count)
        state_t = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q = self.model(state_t, training=False).numpy()[0]
        # mask invalid actions
        q[valid_count:] = -1e9
        return int(np.argmax(q))

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)

        target_q = self.model(states_t, training=False).numpy()
        next_q = self.model(next_states_t, training=False).numpy()

        max_next_q = np.max(next_q, axis=1)
        targets = rewards + GAMMA * max_next_q * (1.0 - dones)
        target_q[np.arange(BATCH_SIZE), actions] = targets

        self.model.train_on_batch(states_t, target_q)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


def train():
    agent = DQNAgent()
    episode_rewards = []
    successes = []

    for ep in range(N_EPISODES):
        src = np.random.randint(N_NODES)
        dest = np.random.randint(N_NODES)
        while dest == src:
            dest = np.random.randint(N_NODES)

        state = get_state(src, dest, 0, 1)
        total_reward = 0
        success = False

        for _ in range(MAX_STEPS):
            current_node = int(state[0] * N_NODES)
            neighbors = list(GRAPH[current_node].items())
            valid_count = len(neighbors)

            action = agent.act(state, valid_count)
            next_state, reward, done = step(state, action, neighbors)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

            if reward == 100:
                success = True
            if done:
                break

        episode_rewards.append(total_reward)
        successes.append(1 if success else 0)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_s = np.mean(successes[-10:]) * 100
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward: {avg_r:.1f} | Success rate: {avg_s:.0f}%")

    return agent, episode_rewards, successes


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
    ax.set_title("DQN Training Performance (Packet Routing)")
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
    print("Experiment 7: Packet Routing with Bandwidth (TensorFlow/Keras)\n")
    print(f"Network: 6 nodes | State: [current, dest, bandwidth_ratio]")
    print(f"Actions: choose next hop (padded to {ACTION_SIZE})")
    print(f"Rewards: +100 delivery, -10/hop, -50 overflow\n")
    print("Training DQN agent...\n")

    agent, rewards, successes = train()
    print(f"\nFinal success rate: {np.mean(successes[-20:])*100:.0f}%")

    plot_results(rewards, successes)
    print("\nExperiment 7 completed successfully!")
