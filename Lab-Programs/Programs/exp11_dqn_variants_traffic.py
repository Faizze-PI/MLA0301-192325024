"""
Experiment 11: DQN Variants for Smart Traffic Signal Control
=============================================================

Compares four Deep RL algorithms for adaptive traffic signal control:
  1. Vanilla DQN
  2. Double DQN
  3. Dueling DQN
  4. Prioritized Experience Replay (PER) + DQN

Environment:
  - 4-way intersection with 4 lanes (N, S, E, W)
  - State: queue lengths per direction (4 floats, normalised)
  - Action: select green-light phase (4 phases: NS-straight, EW-straight,
            NS-left, EW-left)
  - Reward: negative total waiting vehicles (minimise congestion)

All variants share identical hyperparameters for valid ablation:
  - Replay buffer: 10 000 | Batch: 64 | Gamma: 0.99
  - Epsilon: 1.0 -> 0.05 over 200 episodes | LR: 1e-3
  - Network: Dense(64,ReLU) -> Dense(64,ReLU) -> Dueling heads or standard
  - Target network updated every 200 steps

Outputs:
  - Overlaid reward curves (avg wait time vs episode) for all 4 variants
  - Final performance comparison table (console + saved)
  - Plot saved as PNG

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

N_EPISODES = 200
MAX_STEPS = 100
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / N_EPISODES
REPLAY_BUFFER_SIZE = 10_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 200
HIDDEN_SIZE = 64
N_DIRECTIONS = 4
N_PHASES = 4  # NS-straight, EW-straight, NS-left, EW-left
MAX_QUEUE = 20

# --------------------------- Environment -------------------------------

class TrafficIntersectionEnv:
    """
    4-way intersection with signal phases.

    State  : [q_N, q_S, q_E, q_W] normalised by MAX_QUEUE
    Action : 0..3 (phase index)
    Reward : -total_waiting_vehicles (lower is better)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.queues = np.random.randint(0, 8, size=N_DIRECTIONS).astype(np.float32)
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return self.queues / MAX_QUEUE

    def step(self, action):
        self.step_count += 1

        # Spawn new vehicles (random arrival)
        arrivals = np.random.poisson(2, size=N_DIRECTIONS).astype(np.float32)
        self.queues = np.clip(self.queues + arrivals, 0, MAX_QUEUE)

        # Serve vehicles according to phase
        served = self._serve(action)

        # Compute reward: negative total waiting
        reward = -float(np.sum(self.queues))

        # Normalise reward to roughly [-1, 0] range for stability
        reward /= MAX_QUEUE * N_DIRECTIONS

        done = self.step_count >= MAX_STEPS
        return self._get_state(), reward, done, {"queues": self.queues.copy()}

    def _serve(self, phase):
        """Serve vehicles according to the selected phase."""
        served = np.zeros(N_DIRECTIONS)
        capacity = 4  # vehicles per green cycle
        if phase == 0:      # NS straight
            served[0] = min(self.queues[0], capacity)
            served[1] = min(self.queues[1], capacity)
        elif phase == 1:    # EW straight
            served[2] = min(self.queues[2], capacity)
            served[3] = min(self.queues[3], capacity)
        elif phase == 2:    # NS left
            served[0] = min(self.queues[0], capacity // 2)
            served[1] = min(self.queues[1], capacity // 2)
        else:               # EW left
            served[2] = min(self.queues[2], capacity // 2)
            served[3] = min(self.queues[3], capacity // 2)

        self.queues = np.clip(self.queues - served, 0, MAX_QUEUE)
        return served

# --------------------------- Replay Buffers ----------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """PER with proportional prioritisation (alpha=0.6, beta=0.4)."""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, s, a, r, ns, d):
        max_prio = self.priorities[: len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, ns, d))
        else:
            self.buffer[self.pos] = (s, a, r, ns, d)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        self.frame += 1
        N = len(self.buffer)
        prios = self.priorities[:N] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(N, batch_size, p=probs, replace=False)

        beta = min(1.0, self.beta_start + self.frame / self.beta_frames * (1.0 - self.beta_start))
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)

# --------------------------- Network Builders --------------------------

def build_dqn(state_dim, n_actions):
    inputs = keras.Input(shape=(state_dim,))
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(inputs)
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
    outputs = layers.Dense(n_actions, activation=None)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="DQN")


def build_double_dqn(state_dim, n_actions):
    return build_dqn(state_dim, n_actions)  # same architecture, different update


def build_dueling_dqn(state_dim, n_actions):
    inputs = keras.Input(shape=(state_dim,))
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(inputs)
    x = layers.Dense(HIDDEN_SIZE, activation="relu")(x)

    # Value stream
    v = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
    v = layers.Dense(1)(v)

    # Advantage stream
    a = layers.Dense(HIDDEN_SIZE, activation="relu")(x)
    a = layers.Dense(n_actions)(a)

    # Combine: Q = V + (A - mean(A))
    q = v + a - keras.ops.mean(a, axis=1, keepdims=True)
    return keras.Model(inputs=inputs, outputs=q, name="DuelingDQN")

# --------------------------- Agents ------------------------------------

class DQNAgent:
    """Vanilla DQN agent."""

    def __init__(self, state_dim=N_DIRECTIONS, n_actions=N_PHASES, variant="dqn"):
        self.variant = variant
        self.epsilon = EPSILON_START
        self.step_count = 0

        if variant == "dueling":
            self.model = build_dueling_dqn(state_dim, n_actions)
            self.target_model = build_dueling_dqn(state_dim, n_actions)
        elif variant == "double":
            self.model = build_double_dqn(state_dim, n_actions)
            self.target_model = build_double_dqn(state_dim, n_actions)
        else:
            self.model = build_dqn(state_dim, n_actions)
            self.target_model = build_dqn(state_dim, n_actions)

        self.update_target()
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss_fn = keras.losses.Huber()
        self.use_per = False

    def enable_per(self):
        self.use_per = True

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, N_PHASES - 1)
        q = self.model(state[np.newaxis], training=False)
        return int(tf.argmax(q[0]))

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        if self.variant == "double":
            # Double DQN: use online net for action selection, target net for eval
            online_next = self.model(next_states, training=False)
            best_actions = tf.cast(tf.argmax(online_next, axis=1), tf.int32)
            target_next = self.target_model(next_states, training=False)
            indices = tf.stack(
                [tf.range(tf.shape(best_actions)[0], dtype=tf.int32), best_actions], axis=1
            )
            max_next_q = tf.gather_nd(target_next, indices)
        else:
            target_next = self.target_model(next_states, training=False)
            max_next_q = tf.reduce_max(target_next, axis=1)

        targets = rewards + (1.0 - dones) * GAMMA * max_next_q

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            idx = tf.stack(
                [tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1
            )
            q_sel = tf.gather_nd(q_values, idx)
            loss = self.loss_fn(targets, q_sel)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, targets - q_sel  # return td errors

    def train(self, replay_buffer, per_buffer=None):
        if self.use_per and per_buffer is not None:
            if len(per_buffer) < BATCH_SIZE:
                return None, None
            s, a, r, ns, d, indices, weights = per_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = s, a, r, ns, d
        else:
            if len(replay_buffer) < BATCH_SIZE:
                return None, None
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        loss, td_errors = self._train_step(states, actions, rewards, next_states, dones)

        if self.use_per and per_buffer is not None:
            idx_np = indices if isinstance(indices, np.ndarray) else indices.numpy()
            td_np = td_errors.numpy().flatten() if hasattr(td_errors, 'numpy') else np.array(td_errors).flatten()
            per_buffer.update_priorities(idx_np, td_np)

        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()

        return float(loss), None

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)

# --------------------------- Training ----------------------------------

def train_variant(variant_name, variant_key, use_per=False):
    env = TrafficIntersectionEnv()
    agent = DQNAgent(variant=variant_key)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    per_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE) if use_per else None
    if use_per:
        agent.enable_per()

    episode_rewards = []
    episode_wait_times = []

    for ep in range(N_EPISODES):
        state = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            if use_per:
                per_buffer.push(state, action, reward, next_state, float(done))
            else:
                replay_buffer.push(state, action, reward, next_state, float(done))

            agent.train(replay_buffer, per_buffer)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        # Convert normalised reward back to avg wait time
        avg_wait = -total_reward * MAX_QUEUE * N_DIRECTIONS / MAX_STEPS
        episode_wait_times.append(avg_wait)

        if (ep + 1) % 50 == 0:
            avg_50 = np.mean(episode_wait_times[-50:])
            print(
                f"  [{variant_name:15s}] Ep {ep+1:3d}/{N_EPISODES} | "
                f"Avg Wait: {avg_50:.2f} | Eps: {agent.epsilon:.3f}"
            )

    return episode_rewards, episode_wait_times

# --------------------------- Plotting ----------------------------------

def plot_comparison(all_results):
    """Plot overlaid reward curves for all 4 variants."""
    episodes = np.arange(1, N_EPISODES + 1)
    colors = {"DQN": "#1f77b4", "Double DQN": "#ff7f0e",
              "Dueling DQN": "#2ca02c", "PER+DQN": "#d62728"}

    fig, ax = plt.subplots(figsize=(14, 7))

    for name, (_, waits) in all_results.items():
        ma = np.convolve(waits, np.ones(10) / 10, mode="valid")
        ax.plot(episodes[len(episodes) - len(ma):], ma,
                label=name, color=colors[name], linewidth=2.2)

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Average Waiting Vehicles", fontsize=13)
    ax.set_title("DQN Variants for Smart Traffic Signal Control\n"
                 "(10-Episode Moving Average)", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp11_traffic_variants_comparison.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Plot saved to: {out_path}")
    return out_path


def print_results_table(all_results):
    print("\n" + "=" * 72)
    print("  FINAL PERFORMANCE COMPARISON (last 50 episodes)")
    print("=" * 72)
    print(f"  {'Variant':<18s} {'Avg Wait':>10s} {'Std Wait':>10s} "
          f"{'Best Avg':>10s} {'Final Eps':>10s}")
    print("-" * 72)

    for name, (_, waits) in all_results.items():
        recent = waits[-50:]
        # Compute best rolling 50-episode average
        if len(waits) >= 50:
            rolling = [np.mean(waits[i:i+50]) for i in range(len(waits) - 49)]
            best = min(rolling)
        else:
            best = min(recent)
        print(f"  {name:<18s} {np.mean(recent):10.2f} {np.std(recent):10.2f} "
              f"{best:10.2f} {EPSILON_END:10.3f}")

    print("=" * 72)

# --------------------------- Main --------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Experiment 11: DQN Variants for Traffic Signal Control")
    print("=" * 70)
    print(f"  Episodes: {N_EPISODES} | Steps/ep: {MAX_STEPS} | "
          f"Phases: {N_PHASES} | Buffer: {REPLAY_BUFFER_SIZE}")
    print("=" * 70)

    variants = [
        ("DQN",         "dqn",    False),
        ("Double DQN",  "double", False),
        ("Dueling DQN", "dueling", False),
        ("PER+DQN",     "dqn",    True),
    ]

    all_results = {}
    for name, key, use_per in variants:
        print(f"\n  >>> Training {name} ...")
        rewards, waits = train_variant(name, key, use_per=use_per)
        all_results[name] = (rewards, waits)

    plot_comparison(all_results)
    print_results_table(all_results)

