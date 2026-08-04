"""
Experiment 10: DQN for Autonomous Drone Delivery with Battery Constraints
==========================================================================

This experiment implements a Deep Q-Network (DQN) agent to navigate a drone
through a 2D grid environment for package delivery, subject to battery constraints.

Environment:
  - Grid world with a starting position, delivery target, and charging station
  - Drone state: (x, y, battery_level)
  - Actions: Move Up, Move Down, Move Left, Move Right, Charge (5 actions)

Rewards:
  - +50 for reaching the delivery target with sufficient battery
  - -50 if battery depletes before delivery
  - -1 per time step (penalize delays)

Network Architecture:
  - Input: 3 (x, y, battery)
  - Dense(64, ReLU) -> Dense(64, ReLU) -> Dense(n_actions, Linear)

Replay Buffer: 5000 transitions, batch_size=64
Target Network: updated every 100 steps
Epsilon: 1.0 -> 0.05 over 300 episodes (linear decay)
Discount (gamma): 0.99

Outputs:
  - Training reward curve
  - Battery-remaining-at-delivery trend
  - Console output of key metrics

Author: Lab Course - MLA03
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------- Hyperparameters ---------------------------

GRID_SIZE = 8
MAX_BATTERY = 100
BATTERY_COST_PER_STEP = 3
CHARGE_AMOUNT = 25
N_EPISODES = 300
MAX_STEPS_PER_EPISODE = 200
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / N_EPISODES
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100
HIDDEN_SIZE = 64

# --------------------------- Environment -------------------------------

ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: "charge"}
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Charge"]
N_ACTIONS = 5


class DroneDeliveryEnv:
    """Custom grid-world environment for drone delivery with battery."""

    def __init__(self, grid_size=GRID_SIZE, max_battery=MAX_BATTERY):
        self.grid_size = grid_size
        self.max_battery = max_battery
        self.reset()

    def reset(self):
        self.drone_pos = np.array([0, 0])
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        self.charging_pos = np.array([0, self.grid_size - 1])
        self.battery = self.max_battery
        self.steps = 0
        self.delivered = False
        self.battery_depleted = False
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.drone_pos[0] / self.grid_size,
            self.drone_pos[1] / self.grid_size,
            self.battery / self.max_battery,
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -1.0  # step penalty

        if action == 4:  # Charge
            if np.array_equal(self.drone_pos, self.charging_pos):
                self.battery = min(self.max_battery, self.battery + CHARGE_AMOUNT)
            else:
                reward -= 2  # wasted action penalty
        else:
            dx, dy = ACTIONS[action]
            new_x = int(np.clip(self.drone_pos[0] + dx, 0, self.grid_size - 1))
            new_y = int(np.clip(self.drone_pos[1] + dy, 0, self.grid_size - 1))
            self.drone_pos = np.array([new_x, new_y])
            self.battery -= BATTERY_COST_PER_STEP

        done = False
        if self.battery <= 0:
            self.battery = 0
            self.battery_depleted = True
            reward += -50
            done = True
        elif np.array_equal(self.drone_pos, self.target_pos):
            self.delivered = True
            reward += 50
            done = True
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            done = True

        return self._get_state(), reward, done, {
            "battery": self.battery,
            "delivered": self.delivered,
            "battery_depleted": self.battery_depleted,
        }

# --------------------------- DQN Agent ---------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=3, n_actions=N_ACTIONS):
        self.n_actions = n_actions
        self.epsilon = EPSILON_START
        self.step_count = 0

        self.model = self._build_model(state_dim, n_actions)
        self.target_model = self._build_model(state_dim, n_actions)
        self.update_target()

        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss_fn = keras.losses.Huber()

    def _build_model(self, state_dim, n_actions):
        inputs = keras.Input(shape=(state_dim,))
        x = layers.Dense(HIDDEN_SIZE, activation="relu")(inputs)
        x = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
        outputs = layers.Dense(n_actions, activation=None)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.model(state[np.newaxis], training=False)
        return int(tf.argmax(q_values[0]))

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        next_q = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + (1.0 - dones) * GAMMA * max_next_q

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            indices = tf.stack(
                [tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1
            )
            q_selected = tf.gather_nd(q_values, indices)
            loss = self.loss_fn(targets, q_selected)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        loss = self._train_step(states, actions, rewards, next_states, dones)

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()

        return float(loss)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)

# --------------------------- Training Loop -----------------------------

def train_agent():
    env = DroneDeliveryEnv()
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []
    battery_at_delivery = []
    losses = []

    print("=" * 70)
    print("  Experiment 10: DQN for Autonomous Drone Delivery")
    print("=" * 70)
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} | Battery: {MAX_BATTERY} | "
          f"Episodes: {N_EPISODES}")
    print("=" * 70)

    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_battery_at_delivery = None

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, float(done))
            loss = agent.train(replay_buffer)

            state = next_state
            total_reward += reward

            if info["delivered"] and episode_battery_at_delivery is None:
                episode_battery_at_delivery = info["battery"]

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        battery_at_delivery.append(
            episode_battery_at_delivery if episode_battery_at_delivery is not None else 0
        )
        if loss is not None:
            losses.append(loss)

        if (episode + 1) % 50 == 0:
            recent = episode_rewards[-50:]
            avg_r = np.mean(recent)
            avg_bat = np.mean(battery_at_delivery[-50:])
            print(
                f"  Episode {episode+1:4d}/{N_EPISODES} | "
                f"Avg Reward: {avg_r:7.1f} | "
                f"Avg Battery@Delivery: {avg_bat:5.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\n  Training complete. Generating plots...\n")
    return episode_rewards, battery_at_delivery, losses

# --------------------------- Plotting ----------------------------------

def plot_results(rewards, battery_at_delivery, losses):
    episodes = np.arange(1, len(rewards) + 1)

    def moving_average(data, window=20):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Reward Curve ---
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.15, color="steelblue")
    ma_rewards = moving_average(rewards, 20)
    ax.plot(episodes[len(episodes) - len(ma_rewards) :], ma_rewards,
            color="steelblue", linewidth=2, label="Moving Avg (20 ep)")
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("DQN Drone Delivery - Training Reward Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Battery at Delivery ---
    ax = axes[1]
    valid_bat = [(i + 1, b) for i, b in enumerate(battery_at_delivery) if b > 0]
    if valid_bat:
        xs, ys = zip(*valid_bat)
        ax.plot(xs, ys, alpha=0.2, color="darkorange")
        ma_bat = moving_average(
            [b for _, b in valid_bat],
            min(20, len(valid_bat))
        )
        offset = len(xs) - len(ma_bat)
        ax.plot(xs[offset:], ma_bat, color="darkorange", linewidth=2,
                label="Moving Avg (20 ep)")
    ax.set_ylabel("Battery at Delivery", fontsize=12)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_title("Battery Remaining When Delivery Succeeded", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp10_dqn_drone_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved to: {out_path}")

# --------------------------- Main --------------------------------------

if __name__ == "__main__":
    rewards, battery, losses = train_agent()
    plot_results(rewards, battery, losses)

    # Final summary
    final_50 = rewards[-50:]
    print("=" * 70)
    print("  FINAL RESULTS (last 50 episodes)")
    print("=" * 70)
    print(f"    Avg Reward       : {np.mean(final_50):.2f}")
    print(f"    Std Reward       : {np.std(final_50):.2f}")
    print(f"    Max Reward       : {np.max(final_50):.2f}")
    valid_final = [b for b in battery[-50:] if b > 0]
    if valid_final:
        print(f"    Avg Battery@Dlv  : {np.mean(valid_final):.1f}")
    else:
        print(f"    Avg Battery@Dlv  : N/A (no successful deliveries)")
    print(f"    Epsilon Final    : {EPSILON_END:.3f}")
    print("=" * 70)

