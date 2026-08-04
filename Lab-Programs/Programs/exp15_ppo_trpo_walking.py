"""
=============================================================================
Experiment 15: PPO and TRPO for BipedalWalker-v3
=============================================================================
Uses Stable-Baselines3 (SB3) for PPO and sb3-contrib for TRPO to train
agents on the BipedalWalker-v3 continuous control task.

  - PPO: Clipped surrogate objective, GAE, parallel rollouts
  - TRPO: Trust region constraint on KL divergence

Total timesteps: 200000 (partial training demo — full convergence needs ~1M+)

Outputs:
  - Console: training progress, final eval rewards
  - Plots: reward curves PPO vs TRPO
  - Rendered GIFs: early (5000 steps) vs late (200000 steps) checkpoints
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Check and import SB3 / sb3-contrib
# -----------------------------------------------------------------------------
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[WARN] stable-baselines3 not found. Install: pip install stable-baselines3")

try:
    from sb3_contrib import TRPO
    TRPO_AVAILABLE = True
except ImportError:
    TRPO_AVAILABLE = False
    print("[WARN] sb3-contrib not found. Install: pip install sb3-contrib")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("[WARN] gym/gymnasium not found. Install: pip install gymnasium")


# -----------------------------------------------------------------------------
# Training with Callback for Logging
# -----------------------------------------------------------------------------
class RewardLogger:
    """Simple callback to log episode rewards during training."""
    
    def __init__(self):
        self.episode_rewards = []
        self.timesteps = []
        self._current_reward = 0
        self._current_length = 0
        self._ep_count = 0
    
    def __call__(self, locals_dict, globals_dict):
        # Called after each step
        reward = locals_dict.get('rewards', [0])
        done = locals_dict.get('dones', [False])
        
        if hasattr(reward, '__len__'):
            reward = reward[0] if len(reward) > 0 else 0
            done = done[0] if len(done) > 0 else False
        
        self._current_reward += reward
        self._current_length += 1
        
        if done:
            self.episode_rewards.append(self._current_reward)
            self.timesteps.append(globals_dict.get('num_timesteps', 0))
            self._current_reward = 0
            self._current_length = 0
            self._ep_count += 1
        
        return True  # continue training


def train_ppo(total_timesteps=10000, seed=42):
    """Train PPO on BipedalWalker-v3."""
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        return None, [], []
    
    print("\n" + "="*60)
    print("Training PPO on BipedalWalker-v3")
    print("="*60)
    
    env = gym.make("BipedalWalker-v3")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        device="cpu"
    )
    
    callback = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"\nPPO Final Eval: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return model, callback.episode_rewards, callback.timesteps


def train_trpo(total_timesteps=10000, seed=42):
    """Train TRPO on BipedalWalker-v3."""
    if not TRPO_AVAILABLE or not GYM_AVAILABLE:
        return None, [], []
    
    print("\n" + "="*60)
    print("Training TRPO on BipedalWalker-v3")
    print("="*60)
    
    env = gym.make("BipedalWalker-v3")
    model = TRPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        cg_max_steps=10,
        cg_damping=0.1,
        line_search_shrink_factor=0.8,
        verbose=1,
        seed=seed,
        device="cpu"
    )
    
    callback = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"\nTRPO Final Eval: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return model, callback.episode_rewards, callback.timesteps


# -----------------------------------------------------------------------------
# GIF Rendering
# -----------------------------------------------------------------------------
def render_gif(model, env_name="BipedalWalker-v3", filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\demo.gif", n_steps=200):
    """Render an episode and save as GIF (or series of frames)."""
    try:
        import imageio
    except ImportError:
        print("[INFO] imageio not available, skipping GIF. Install: pip install imageio")
        return False
    
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    
    for _ in range(n_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    
    if frames:
        imageio.mimsave(filename, frames, fps=30)
        print(f"[Saved] {filename}")
        return True
    return False


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(ppo_rewards, ppo_ts, trpo_rewards, trpo_ts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Reward curves overlaid
    fig, ax = plt.subplots(figsize=(10, 5))
    
    window = 20
    
    if ppo_rewards and len(ppo_rewards) >= window:
        s_ppo = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ppo_rewards)), s_ppo, 'b-', linewidth=2, label='PPO')
    
    if trpo_rewards and len(trpo_rewards) >= window:
        s_trpo = np.convolve(trpo_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(trpo_rewards)), s_trpo, 'r-', linewidth=2, label='TRPO')
    
    if ppo_rewards:
        ax.plot(ppo_rewards, alpha=0.15, color='blue')
    if trpo_rewards:
        ax.plot(trpo_rewards, alpha=0.15, color='red')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('PPO vs TRPO: BipedalWalker-v3 Reward Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp15_ppo_trpo_reward.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp15_ppo_trpo_reward.png")
    
    # 2. Final performance summary
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    means = []
    stds = []
    
    if ppo_rewards:
        labels.append('PPO')
        last_50 = ppo_rewards[-50:] if len(ppo_rewards) >= 50 else ppo_rewards
        means.append(np.mean(last_50))
        stds.append(np.std(last_50))
    
    if trpo_rewards:
        labels.append('TRPO')
        last_50 = trpo_rewards[-50:] if len(trpo_rewards) >= 50 else trpo_rewards
        means.append(np.mean(last_50))
        stds.append(np.std(last_50))
    
    if labels:
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                       color=['steelblue', 'indianred'][:len(labels)],
                       edgecolor='black', width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Mean Reward (last 50 episodes)', fontsize=12)
        ax.set_title('PPO vs TRPO: Final Performance', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp15_final_performance.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp15_final_performance.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    np.random.seed(42)
    save_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    
    total_timesteps = 200000
    
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n[ERROR] Required packages not available.")
        print("Install with: pip install stable-baselines3 sb3-contrib gymnasium")
        return
    
    # Train PPO
    ppo_model, ppo_rewards, ppo_ts = train_ppo(total_timesteps=total_timesteps, seed=42)
    
    # Train TRPO
    trpo_model, trpo_rewards, trpo_ts = train_trpo(total_timesteps=total_timesteps, seed=42)
    
    # Render GIFs (late checkpoint)
    if ppo_model is not None:
        print("\nRendering PPO GIF...")
        render_gif(ppo_model, filename=os.path.join(save_dir, "exp15_ppo_late.gif"))
    
    if trpo_model is not None:
        print("\nRendering TRPO GIF...")
        render_gif(trpo_model, filename=os.path.join(save_dir, "exp15_trpo_late.gif"))
    
    # Results summary
    print(f"\n{'='*60}")
    print("PPO vs TRPO Summary (BipedalWalker-v3)")
    print(f"{'='*60}")
    if ppo_rewards:
        print(f"PPO  - Last 50 ep avg reward: {np.mean(ppo_rewards[-50:]):.2f}")
        print(f"PPO  - Total episodes: {len(ppo_rewards)}")
    if trpo_rewards:
        print(f"TRPO - Last 50 ep avg reward: {np.mean(trpo_rewards[-50:]):.2f}")
        print(f"TRPO - Total episodes: {len(trpo_rewards)}")
    print(f"{'='*60}")
    
    plot_results(ppo_rewards, ppo_ts, trpo_rewards, trpo_ts, save_dir)
    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    main()

