"""
=============================================================================
Experiment 16: Compare REINFORCE, A2C, PPO for Autonomous Lane-Keeping
=============================================================================
Trains and compares three policy gradient methods on highway-env's
lane-keeping task:
  - REINFORCE (custom implementation)
  - A2C (Stable-Baselines3)
  - PPO (Stable-Baselines3)

All methods use identical total_timesteps=100000 for fair comparison.
Uses highway-env lane-keeping-v0 or highway-v0 as fallback.

Outputs:
  - Console: training progress, final accuracy table
  - Plots: 3-line overlaid learning curve, final accuracy bar chart
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Package availability checks
# -----------------------------------------------------------------------------
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# -----------------------------------------------------------------------------
# Create Highway Env (with fallbacks)
# -----------------------------------------------------------------------------
def make_env(env_id="lane-keeping-v0"):
    """Try to create highway-env with multiple fallback options."""
    if not GYM_AVAILABLE:
        return None
    
    # Try lane-keeping-v0 first
    try:
        import highway_env
        from gymnasium.wrappers import FlattenObservation
        env = gym.make(env_id)
        env = FlattenObservation(env)
        print(f"Using environment: {env_id}")
        return env
    except Exception:
        pass
    
    # Try highway-v0
    try:
        import highway_env
        from gymnasium.wrappers import FlattenObservation
        env = gym.make("highway-v0")
        env = FlattenObservation(env)
        print("Using environment: highway-v0")
        return env
    except Exception:
        pass
    
    # Try parking-v0 as last resort
    try:
        import highway_env
        from gymnasium.wrappers import FlattenObservation
        env = gym.make("parking-v0")
        env = FlattenObservation(env)
        print("Using environment: parking-v0")
        return env
    except Exception:
        pass
    
    print("[WARN] highway-env not available. Using CartPole-v1 as fallback.")
    try:
        env = gym.make("CartPole-v1")
        return env
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Custom REINFORCE
# -----------------------------------------------------------------------------
class SimplePolicyNet:
    """Simple 2-layer policy network for continuous or discrete actions."""
    
    def __init__(self, obs_dim, act_dim, lr=0.001, hidden=64, continuous=False):
        self.act_dim = act_dim
        self.lr = lr
        self.continuous = continuous
        self.hidden = hidden
        scale = 1.0 / np.sqrt(hidden)
        
        self.W1 = np.random.randn(hidden, obs_dim) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(act_dim, hidden) * scale
        self.b2 = np.zeros(act_dim)
        
        if continuous:
            self.log_std = np.zeros(act_dim)
        
        self._t = 0
        params = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, obs):
        h = self._relu(self.W1 @ obs + self.b1)
        logits = self.W2 @ h + self.b2
        
        if self.continuous:
            mean = logits
            std = np.exp(self.log_std)
            return mean, std
        
        logits -= logits.max()
        probs = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
        return probs
    
    def sample(self, obs):
        if self.continuous:
            mean, std = self.forward(obs)
            action = mean + std * np.random.randn(self.act_dim)
            log_prob = -0.5 * np.sum(((action - mean)/std)**2 + 2*self.log_std + np.log(2*np.pi))
            return action, log_prob
        else:
            probs = self.forward(obs)
            action = np.random.choice(self.act_dim, p=probs)
            log_prob = np.log(probs[action] + 1e-8)
            return action, log_prob
    
    def update(self, obs_list, actions_list, returns):
        self._t += 1
        obs = np.array(obs_list)
        actions = np.array(actions_list)
        returns = np.array(returns, dtype=np.float64)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)
        
        for i in range(len(obs)):
            obs_i = obs[i]
            G = returns[i]
            
            if self.continuous:
                mean, std = self.forward(obs_i)
                action = actions_i = actions[i]
                score = (action - mean) / (std**2)
                h = self._relu(self.W1 @ obs_i + self.b1)
                
                grad_W2 += np.outer(score, h) * G
                grad_b2 += score * G
                
                pre = self.W1 @ obs_i + self.b1
                d_relu = (pre > 0).astype(float)
                grad_h = self.W2.T @ score
                grad_pre = grad_h * d_relu
                grad_W1 += np.outer(grad_pre, obs_i) * G
                grad_b1 += grad_pre * G
            else:
                probs = self.forward(obs_i)
                one_hot = np.eye(self.act_dim)[actions[i]]
                score = (one_hot - probs)
                h = self._relu(self.W1 @ obs_i + self.b1)
                
                grad_W2 += np.outer(score, h) * G
                grad_b2 += score * G
                
                pre = self.W1 @ obs_i + self.b1
                d_relu = (pre > 0).astype(float)
                grad_h = self.W2.T @ score
                grad_pre = grad_h * d_relu
                grad_W1 += np.outer(grad_pre, obs_i) * G
                grad_b1 += grad_pre * G
        
        n = len(obs)
        params = [self.W1, self.b1, self.W2, self.b2]
        grads = [grad_W1/n, grad_b1/n, grad_W2/n, grad_b2/n]
        
        for i in range(len(params)):
            self._m[i] = 0.9 * self._m[i] + 0.1 * grads[i]
            self._v[i] = 0.999 * self._v[i] + 0.001 * grads[i]**2
            m_hat = self._m[i] / (1 - 0.9**self._t)
            v_hat = self._v[i] / (1 - 0.999**self._t)
            params[i] += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        self.W1, self.b1, self.W2, self.b2 = params


def compute_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float64)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def train_reinforce(env, total_timesteps=100000, gamma=0.99, lr=0.001):
    """Train REINFORCE and return reward history."""
    obs_space = env.observation_space
    act_space = env.action_space
    
    # Handle different space types
    if hasattr(obs_space, 'shape') and obs_space.shape is not None:
        obs_dim = int(np.prod(obs_space.shape))
    else:
        obs_dim = 4  # fallback
    
    if hasattr(act_space, 'n'):
        act_dim = act_space.n
        continuous = False
    elif hasattr(act_space, 'shape'):
        act_dim = int(np.prod(act_space.shape))
        continuous = True
    else:
        act_dim = 2
        continuous = False
    
    policy = SimplePolicyNet(obs_dim, act_dim, lr=lr, hidden=64, continuous=continuous)
    
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    
    print(f"\nTraining REINFORCE (obs_dim={obs_dim}, act_dim={act_dim})...")
    
    while total_steps < total_timesteps:
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        obs_list, act_list, rew_list = [], [], []
        
        while not done:
            # Handle OrderedDict observations from highway-env
            if isinstance(obs, dict):
                # Extract the main observation array from dict
                obs_array = obs.get('observation', obs.get('state', None))
                if obs_array is None:
                    # Try first value
                    for v in obs.values():
                        if isinstance(v, np.ndarray):
                            obs_array = v
                            break
                if obs_array is None:
                    obs_array = np.zeros(obs_dim)
                obs_flat = np.array(obs_array, dtype=np.float64).flatten()
            elif hasattr(obs, 'flatten'):
                obs_flat = obs.flatten()
            else:
                obs_flat = np.array(obs, dtype=np.float64)
            
            if obs_flat.shape[0] != obs_dim:
                obs_flat = np.zeros(obs_dim)
            
            action, log_prob = policy.sample(obs_flat)
            
            if continuous:
                action = np.clip(action, act_space.low, act_space.high)
                step_result = env.step(action)
            else:
                step_result = env.step(int(action) % act_dim)
            
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            obs_list.append(obs_flat)
            act_list.append(action)
            rew_list.append(reward)
            ep_reward += reward
            total_steps += 1
            obs = next_obs
        
        returns = compute_returns(rew_list, gamma)
        policy.update(obs_list, act_list, returns)
        episode_rewards.append(ep_reward)
        episode_lengths.append(len(rew_list))
        
        if len(episode_rewards) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  REINFORCE ep {len(episode_rewards):5d}: avg_reward={avg:.2f}")
    
    return episode_rewards


# -----------------------------------------------------------------------------
# SB3-based training (A2C and PPO)
# -----------------------------------------------------------------------------
def train_sb3(method, total_timesteps=100000, seed=42):
    """Train A2C or PPO using SB3."""
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        return []
    
    env = make_env()
    if env is None:
        return []
    
    algo_name = method.__name__
    print(f"\nTraining {algo_name}...")
    
    model = method(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        verbose=0,
        seed=seed,
        device="cpu"
    )
    
    # Custom callback for logging
    class RewardCallback:
        def __init__(self):
            self.episode_rewards = []
            self._current = 0
        
        def __call__(self, locals_dict, globals_dict):
            r = locals_dict.get('rewards', [0])
            d = locals_dict.get('dones', [False])
            if hasattr(r, '__len__'):
                r = r[0] if len(r) > 0 else 0
                d = d[0] if len(d) > 0 else False
            self._current += r
            if d:
                self.episode_rewards.append(self._current)
                self._current = 0
            return True
    
    cb = RewardCallback()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    
    mean_rew, std_rew = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"  {algo_name} final eval: {mean_rew:.2f} +/- {std_rew:.2f}")
    
    env.close()
    return cb.episode_rewards


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(rewards_r, rewards_a2c, rewards_ppo, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    window = 30
    
    # 1. Learning curves overlaid
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'REINFORCE': 'green', 'A2C': 'blue', 'PPO': 'red'}
    
    for name, rewards, color in [
        ('REINFORCE', rewards_r, colors['REINFORCE']),
        ('A2C', rewards_a2c, colors['A2C']),
        ('PPO', rewards_ppo, colors['PPO'])
    ]:
        if rewards and len(rewards) >= window:
            s = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), s, color=color, linewidth=2, label=name)
            ax.plot(rewards, alpha=0.1, color=color)
    
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Episode Reward', fontsize=13)
    ax.set_title('Policy Gradient Comparison: REINFORCE vs A2C vs PPO\n(Lane-Keeping Task)', 
                 fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp16_learning_curves.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp16_learning_curves.png")
    
    # 2. Final accuracy table as bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = []
    means = []
    stds = []
    
    for name, rewards in [('REINFORCE', rewards_r), ('A2C', rewards_a2c), ('PPO', rewards_ppo)]:
        if rewards:
            last = rewards[-min(50, len(rewards)):]
            labels.append(name)
            means.append(np.mean(last))
            stds.append(np.std(last))
    
    if labels:
        x = np.arange(len(labels))
        bar_colors = [colors[l] for l in labels]
        bars = ax.bar(x, means, yerr=stds, capsize=8, color=bar_colors,
                       edgecolor='black', width=0.5, alpha=0.85)
        
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                    f'{m:.1f}', ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Mean Reward (last 50 episodes)', fontsize=12)
        ax.set_title('Final Performance Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp16_final_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp16_final_comparison.png")
    
    # 3. Accuracy table (text)
    print(f"\n{'='*55}")
    print(f"{'Method':<15} {'Avg Reward (last 50)':<22} {'Episodes':<10}")
    print(f"{'='*55}")
    for name, rewards in [('REINFORCE', rewards_r), ('A2C', rewards_a2c), ('PPO', rewards_ppo)]:
        if rewards:
            last = rewards[-min(50, len(rewards)):]
            print(f"{name:<15} {np.mean(last):<22.2f} {len(rewards):<10}")
        else:
            print(f"{name:<15} {'N/A':<22} {'N/A':<10}")
    print(f"{'='*55}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    np.random.seed(42)
    save_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    
    total_timesteps = 100000
    
    # --- REINFORCE ---
    env_r = make_env()
    if env_r is not None:
        rewards_r = train_reinforce(env_r, total_timesteps=total_timesteps)
        env_r.close()
    else:
        rewards_r = []
    
    # --- A2C ---
    rewards_a2c = train_sb3(A2C, total_timesteps=total_timesteps, seed=42)
    
    # --- PPO ---
    rewards_ppo = train_sb3(PPO, total_timesteps=total_timesteps, seed=42)
    
    plot_results(rewards_r, rewards_a2c, rewards_ppo, save_dir)
    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    main()

