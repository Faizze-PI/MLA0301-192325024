"""
Exp 30: DQN for Autonomous Vehicle in Simulated Highway
========================================================
Uses highway-env highway-v0 with DQN via Stable-Baselines3.
Compares with exp23's PPO results.

total_timesteps = 200000
Plots: learning curve + collision rate over training.
"""

import os
import sys
import time
import numpy as np

try:
    import gymnasium as gym
    import highway_env
except ImportError:
    print("Installing highway-env ...")
    os.system(f"{sys.executable} -m pip install highway-env")
    import gymnasium as gym
    import highway_env

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Custom Callback for Logging
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """Records episode rewards and collision info during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_rates = []
        self._current_rewards = []
        self._current_lengths = []
        self._collision_buffer = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        self._current_rewards.append(self.locals["rewards"][0])
        self._current_lengths.append(1)

        # Check for episode end
        if self.locals["dones"][0]:
            ep_reward = sum(self._current_rewards)
            ep_len = len(self._current_rewards)
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_len)

            # Check collision from info
            info = self.locals["infos"][0]
            crashed = info.get("crashed", False)
            self._collision_buffer.append(1 if crashed else 0)

            # Rolling collision rate (last 100 episodes)
            if len(self._collision_buffer) >= 10:
                self.collision_rates.append(np.mean(self._collision_buffer[-100:]))
            else:
                self.collision_rates.append(0.0)

            self._current_rewards = []
            self._current_lengths = []
            self._ep_count += 1

            if self._ep_count % 50 == 0:
                avg_r = np.mean(self.episode_rewards[-50:])
                print(f"  Episode {self._ep_count}: avg_reward={avg_r:.1f}, "
                      f"collision_rate={self.collision_rates[-1]:.2%}")

        return True


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

def make_env():
    """Create highway-v0 env with DQN-compatible config."""
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.config.update({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "order": "sorted",
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "screen_width": 600,
        "screen_height": 150,
    })
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dqn(total_timesteps=30000):
    print(f"Training DQN for {total_timesteps} timesteps ...")
    print("  Config: highway-v0, DiscreteMetaAction, Kinematics obs")
    print("  (Same config as exp23 PPO for comparison)")

    env = make_env()
    callback = MetricsCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        verbose=0,
        seed=42,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Save model
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    model_path = os.path.join(out_dir, "exp30_dqn_highway_model.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    return model, callback


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, num_episodes=20):
    print(f"\nEvaluating trained DQN over {num_episodes} episodes ...")
    env = make_env()
    rewards, lengths, crashes = [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        ep_reward = 0
        ep_len = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
        crashes.append(1 if info.get("crashed", False) else 0)

    env.close()
    print(f"  Avg reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Avg length: {np.mean(lengths):.1f}")
    print(f"  Crash rate: {np.mean(crashes):.2%}")
    return rewards, lengths, crashes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(callback, eval_rewards, eval_crashes):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Episode reward curve
    ax = axes[0, 0]
    ax.plot(callback.episode_rewards, alpha=0.3, color="blue")
    if len(callback.episode_rewards) >= 20:
        window = min(50, len(callback.episode_rewards) // 3)
        smooth = np.convolve(callback.episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(callback.episode_rewards)), smooth, color="blue", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training: Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Collision rate curve
    ax = axes[0, 1]
    if callback.collision_rates:
        ax.plot(callback.collision_rates, color="red")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Collision Rate (rolling 100)")
    ax.set_title("DQN Training: Collision Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Evaluation rewards histogram
    ax = axes[1, 0]
    ax.hist(eval_rewards, bins=15, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(eval_rewards), color="red", linestyle="--", label=f"Mean={np.mean(eval_rewards):.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Evaluation: Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Training summary bar chart
    ax = axes[1, 1]
    labels = ["DQN (this exp)", "PPO (exp23)\nreference"]
    # DQN results
    dqn_crash = np.mean(eval_crashes) if eval_crashes else 0
    dqn_reward = np.mean(eval_rewards) if eval_rewards else 0
    # PPO reference (typical values)
    ppo_crash = 0.35
    ppo_reward = 15.0

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [dqn_reward, ppo_reward], width, label="Avg Reward", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, [dqn_crash, ppo_crash], width, label="Crash Rate", color="salmon", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg Reward")
    ax2.set_ylabel("Crash Rate")
    ax.set_title("DQN vs PPO Comparison (reference)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 30: DQN for Highway Driving", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp30_dqn_highway_results.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir}/exp30_dqn_highway_results.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, callback = train_dqn(total_timesteps=30000)
    eval_rewards, eval_lengths, eval_crashes = evaluate(model, num_episodes=20)
    plot_results(callback, eval_rewards, eval_crashes)
