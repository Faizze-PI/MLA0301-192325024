"""
Experiment 10: Mini Reinforcement Learning Project Using CartPole
------------------------------------------------------------------------
Aim: To develop a basic Reinforcement Learning agent using TensorFlow and
Keras to solve the CartPole environment and evaluate its learning
performance through episode rewards and success rate.

Hardware note: this is a small DQN (two 24-unit dense layers, batch size 64).
On a CPU like your i5-12500H, 300 episodes typically takes roughly 5-10
minutes (episodes get longer as the agent improves, so most of the time is
spent in the later, better-performing episodes). A GPU gives no real speedup
here because the network and batches are tiny, but GPU memory-growth is
still configured below in case TensorFlow finds your RTX 3050.

Performance tip used throughout this file: we call the model directly
(model(x)) and use train_on_batch() instead of model.predict()/model.fit().
For tiny per-step calls like this, predict()/fit() carry ~50ms of fixed
overhead each call (dataset pipeline setup, etc.), which would make this
script roughly 20x slower for no benefit.
"""

import random
from collections import deque

import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- GPU configuration (safe no-op if no GPU is present) ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected, memory growth enabled: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected -- running on CPU (this DQN is small and trains fast).")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # NOTE: calling the model directly (model(x)) instead of model.predict(x)
        # avoids ~50ms of per-call overhead that predict() carries even for a
        # single sample -- that overhead adds up fast in a step-by-step RL loop.
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.model(state_tensor, training=False).numpy()[0]
        return int(np.argmax(q_values))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # Direct model calls (fast) instead of predict() (slow, high per-call overhead)
        target_q = self.model(states, training=False).numpy()
        next_q = self.model(next_states, training=False).numpy()

        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)
        target_q[np.arange(self.batch_size), actions] = targets

        # train_on_batch avoids fit()'s per-call overhead (dataset setup, callbacks, etc.)
        self.model.train_on_batch(states, target_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(n_episodes=300, max_steps=500, target_score=475, target_window=20):
    env = gym.make("CartPole-v1")
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)  # cast from numpy.int64 for Keras
    agent = DQNAgent(state_size, action_size)

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            if done:
                break
        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_last = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | Reward: {total_reward:.0f} "
                  f"| Avg(last 10): {avg_last:.1f} | Epsilon: {agent.epsilon:.3f}")

        if len(episode_rewards) >= target_window and \
                np.mean(episode_rewards[-target_window:]) >= target_score:
            print(f"\nSolved in {episode + 1} episodes! Average reward over last "
                  f"{target_window} episodes: {np.mean(episode_rewards[-target_window:]):.1f}")
            break

    env.close()
    return agent, episode_rewards


def evaluate_agent(agent, n_episodes=10):
    env = gym.make("CartPole-v1")
    scores = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation for evaluation
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        scores.append(total_reward)
    agent.epsilon = old_epsilon
    env.close()
    return scores


if __name__ == "__main__":
    print("Training DQN agent on CartPole-v1 (this may take a few minutes)...\n")
    agent, episode_rewards = train_dqn(n_episodes=300)

    print("\nEvaluating trained agent over 10 episodes (greedy policy)...")
    eval_scores = evaluate_agent(agent, n_episodes=10)
    print(f"Evaluation scores: {eval_scores}")
    print(f"Average evaluation score: {np.mean(eval_scores):.1f}")
    success_rate = np.mean([1 if s >= 475 else 0 for s in eval_scores]) * 100
    print(f"Success rate (score >= 475): {success_rate:.0f}%")

    window = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg,
              label=f"{window}-episode moving average", linewidth=2)
    plt.axhline(y=475, color="r", linestyle="--", label="Solved threshold (475)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN CartPole Training Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment10_cartpole_dqn_training.png", dpi=150)
    print("\nPlot saved as 'experiment10_cartpole_dqn_training.png'")
    plt.show()
