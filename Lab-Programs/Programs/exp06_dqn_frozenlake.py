"""
Experiment 06 - DQN on FrozenLake-v1
======================================
Deep Q-Network using OpenAI Gymnasium + TensorFlow/Keras.

Architecture:
  Dense(64, relu) -> Dense(64, relu) -> Dense(n_actions, linear)

Experience replay buffer=2000, batch_size=32, gamma=0.95,
epsilon_decay=0.995, episodes=500.

Plots reward-per-episode learning curve.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import random

import tensorflow as tf
from tensorflow import keras

# -- Force CPU if GPU is available ----------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices([], "GPU")
    print("GPU detected but forced to use CPU.")

# -- Reproducibility ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -- Hyperparameters ------------------------------------------------------------
EPISODES       = 500
GAMMA          = 0.95
EPSILON_START  = 1.0
EPSILON_MIN    = 0.01
EPSILON_DECAY  = 0.995
REPLAY_BUFFER  = 2000
BATCH_SIZE     = 32
LR             = 0.001
HIDDEN1        = 64
HIDDEN2        = 64


# -- DQN Agent ------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = EPSILON_START
        self.memory      = deque(maxlen=REPLAY_BUFFER)
        self.model       = self._build_model()

    # ------------------------------------------------------------------ #
    def _build_model(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(HIDDEN1, activation="relu"),
            keras.layers.Dense(HIDDEN2, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                      loss="mse")
        return model

    # ------------------------------------------------------------------ #
    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return int(np.argmax(q_values[0]))

    # ------------------------------------------------------------------ #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------ #
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        states      = np.array([t[0] for t in batch])
        actions     = np.array([t[1] for t in batch])
        rewards     = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones       = np.array([t[4] for t in batch], dtype=np.float32)

        q_current = self.model.predict(states,      verbose=0)
        q_next    = self.model.predict(next_states,  verbose=0)

        targets = q_current.copy()
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(q_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

    # ------------------------------------------------------------------ #
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# -- Training loop --------------------------------------------------------------
def train_dqn():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)
    # One-hot encode the discrete observation (0-15) -> (16,)
    state_size  = int(env.observation_space.n)
    action_size = int(env.action_space.n)

    agent = DQNAgent(state_size, action_size)

    # Helper: one-hot encoding
    def one_hot(s: int) -> np.ndarray:
        v = np.zeros(state_size, dtype=np.float32)
        v[s] = 1.0
        return v

    reward_history = []

    print(f"{'Ep':>4} | {'Reward':>7} | {'Epsilon':>8} | {'Avg100':>8}")
    print("-" * 40)

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset(seed=SEED + ep)
        state = one_hot(state)
        total_reward = 0

        for _ in range(100):          # max steps per episode
            action = agent.act(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = one_hot(next_state_raw)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.replay()
        agent.decay_epsilon()
        reward_history.append(total_reward)

        if ep % 50 == 0 or ep == 1:
            avg = np.mean(reward_history[-100:])
            print(f"{ep:4d} | {total_reward:7.3f} | {agent.epsilon:8.4f} | {avg:8.4f}")

    env.close()
    return reward_history


# -- Plotting -------------------------------------------------------------------
def plot_rewards(rewards):
    episodes = range(1, len(rewards) + 1)

    # Smoothed curve (running average of 20)
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Per-episode reward")
    plt.plot(range(window, len(rewards) + 1), smoothed,
             color="red", linewidth=2, label=f"Running avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on FrozenLake-v1 – Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp06_dqn_frozenlake_rewards.png'), dpi=150)
    print("\nPlot saved: exp06_dqn_frozenlake_rewards.png")
    plt.close()


# -- Main -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Experiment 06: DQN on FrozenLake-v1 ===\n")
    rewards = train_dqn()
    plot_rewards(rewards)

    final_avg = np.mean(rewards[-100:])
    print(f"\nFinal 100-episode average reward: {final_avg:.4f}")
    print("Done.")

