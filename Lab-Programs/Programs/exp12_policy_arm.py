"""
=============================================================================
Experiment 12: Policy-Based RL for Industrial Robotic Arm Pick-and-Place
=============================================================================
REINFORCE (Actor-Only) Policy Gradient on a simplified 2D 2-joint arm
that must reach a target position. Actions are continuous joint torques
sampled from a Gaussian policy.

Environment:
  - 2-link planar arm (shoulder + elbow joints)
  - State: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), tx, ty] (6-dim)
  - Action: [tau1, tau2] continuous torques ∈ [-1, 1]
  - Reward: -distance_to_target - 0.01*||action||²
  - Success: end-effector within 0.15 of target

Algorithm: REINFORCE with Gaussian policy (learned mean, learned log-std)
  gamma = 0.99, lr = 0.001, episodes = 500

Outputs:
  - Console: episode rewards, success rate
  - Plots saved: learning curve, success rate, arm trajectory animation
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
import sys

# -----------------------------------------------------------------------------
# Custom 2-Joint Planar Arm Environment
# -----------------------------------------------------------------------------
class SimpleArmEnv:
    """2-link planar arm reaching task."""
    
    LINK_LENGTHS = np.array([1.0, 0.8])  # l1, l2
    
    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.obs_dim = 6
        self.act_dim = 2
        self.action_space_low = np.array([-1.0, -1.0])
        self.action_space_high = np.array([1.0, 1.0])
        self.reset()
    
    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi, size=2)
        self.target = np.random.uniform(-1.5, 1.5, size=2)
        self.step_count = 0
        self.joint_history = [self.theta.copy()]
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([
            np.cos(self.theta), np.sin(self.theta), self.target
        ]).astype(np.float32)
    
    def _forward_kinematics(self):
        x1 = self.LINK_LENGTHS[0] * np.cos(self.theta[0])
        y1 = self.LINK_LENGTHS[0] * np.sin(self.theta[0])
        x2 = x1 + self.LINK_LENGTHS[1] * np.cos(self.theta[0] + self.theta[1])
        y2 = y1 + self.LINK_LENGTHS[1] * np.sin(self.theta[0] + self.theta[1])
        return np.array([x2, y2])  # end-effector
    
    def step(self, action):
        action = np.clip(action, self.action_space_low, self.action_space_high)
        self.theta = self.theta + action * 0.1  # small dt
        self.theta = np.mod(self.theta + np.pi, 2 * np.pi) - np.pi
        self.step_count += 1
        self.joint_history.append(self.theta.copy())
        
        ee = self._forward_kinematics()
        dist = np.linalg.norm(ee - self.target)
        reward = -dist - 0.01 * np.sum(action ** 2)
        
        done = dist < 0.15 or self.step_count >= self.max_steps
        success = dist < 0.15
        
        return self._get_obs(), reward, done, {"distance": dist, "success": success}
    
    def get_ee_trajectory(self, thetas):
        """Compute end-effector positions for a sequence of joint angles."""
        pts = [np.array([0.0, 0.0])]
        for th in thetas:
            x1 = self.LINK_LENGTHS[0] * np.cos(th[0])
            y1 = self.LINK_LENGTHS[0] * np.sin(th[0])
            x2 = x1 + self.LINK_LENGTHS[1] * np.cos(th[0] + th[1])
            y2 = y1 + self.LINK_LENGTHS[1] * np.sin(th[0] + th[1])
            pts.append(np.array([x1, y1]))
            pts.append(np.array([x2, y2]))
        return np.array(pts)


# -----------------------------------------------------------------------------
# Gaussian Policy Network (two-layer MLP)
# -----------------------------------------------------------------------------
class GaussianPolicy:
    """Simple 2-layer Gaussian policy with learnable mean and log-std."""
    
    def __init__(self, obs_dim, act_dim, lr=0.001, hidden=128):
        self.lr = lr
        self.act_dim = act_dim
        scale = 1.0 / np.sqrt(hidden)
        
        # Mean network
        self.W1 = np.random.randn(hidden, obs_dim) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(act_dim, hidden) * scale
        self.b2 = np.zeros(act_dim)
        
        # Learnable log standard deviation
        self.log_std = np.zeros(act_dim)
        
        # Adam optimizer state for mean params
        self._m = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2]]
        self._v = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2]]
        # Adam state for log_std
        self._m_std = np.zeros_like(self.log_std)
        self._v_std = np.zeros_like(self.log_std)
        self._t = 0
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, obs):
        """Return mean and std of action distribution."""
        h = self._relu(self.W1 @ obs + self.b1)
        mean = self.W2 @ h + self.b2
        std = np.exp(self.log_std)
        return mean, std
    
    def sample(self, obs):
        """Sample action, return (action, log_prob)."""
        mean, std = self.forward(obs)
        noise = np.random.randn(self.act_dim)
        action = mean + std * noise
        # Gaussian log probability
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
        return action, log_prob, mean
    
    def adam_update(self, params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        for i in range(len(params)):
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def update(self, log_probs, returns):
        """REINFORCE update with policy gradient theorem."""
        self._t += 1
        returns = np.array(returns, dtype=np.float64)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient: ∇J = -E[log pi(a|s) * R]
        # We accumulate gradients for each transition
        total_loss = 0.0
        
        # Re-run forward pass to get gradients (simplified: use finite diff approximation)
        # For the mean network gradient: d(log_prob)/d(mean) = (action - mean) / std²
        # We need: ∇_theta log pi(a|s) * G_t
        # Approximate: use (a - mu) / sigma² * ∇_theta mu  (score function trick)
        
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)
        grad_log_std = np.zeros_like(self.log_std)
        
        # We stored transitions - recompute to get exact gradients
        # For simplicity in numpy, we'll do a simpler gradient:
        # The gradient of log pi w.r.t. mean is (action - mean)/std²
        # Then chain through the network
        
        # Let's use a simpler approach: compute numerical gradients
        # via REINFORCE score function: ∇_theta J = Σ_t G_t * ∇_theta log pi(a_t|s_t)
        # ∇_mu log pi = (a - mu)/sigma², then ∇_theta mu = network Jacobian
        
        # Store per-transition data
        for t_idx in range(len(self.stored_obs)):
            obs = self.stored_obs[t_idx]
            action = self.stored_actions[t_idx]
            G = returns[t_idx]
            
            mean, std = self.forward(obs)
            
            # Gradient of log_prob w.r.t. mean
            score = (action - mean) / (std ** 2)  # (act_dim,)
            
            # Backprop through network to get grad w.r.t. W2, b2, W1, b1
            h = self._relu(self.W1 @ obs + self.b1)  # (hidden,)
            
            # ∂mean/∂W2 = h^T, ∂mean/∂b2 = 1
            grad_W2 += np.outer(score, h) * G
            grad_b2 += score * G
            
            # ∂mean/∂h = W2^T, ∂h/∂pre = relu'(pre)
            pre_act = self.W1 @ obs + self.b1
            d_relu = (pre_act > 0).astype(float)
            grad_h = self.W2.T @ score  # (hidden,)
            grad_pre = grad_h * d_relu  # (hidden,)
            
            grad_W1 += np.outer(grad_pre, obs) * G
            grad_b1 += grad_pre * G
            
            # Gradient of log_std: ∂log_prob/∂log_std = -1 + (a-mu)²/sigma²
            grad_log_std += (-1 + ((action - mean) / std) ** 2) * G
        
        # Average over batch
        n = len(self.stored_obs)
        grad_W1 /= n
        grad_b1 /= n
        grad_W2 /= n
        grad_b2 /= n
        grad_log_std /= n
        
        # Gradient ascent (add, not subtract)
        grads = [grad_W1, grad_b1, grad_W2, grad_b2]
        params = [self.W1, self.b1, self.W2, self.b2]
        self.adam_update(params, grads, self._m, self._v, self._t, lr=self.lr)
        self.W1, self.b1, self.W2, self.b2 = params
        
        # Update log_std
        self._m_std = 0.9 * self._m_std + 0.1 * grad_log_std
        self._v_std = 0.999 * self._v_std + 0.001 * grad_log_std ** 2
        m_hat = self._m_std / (1 - 0.9 ** self._t)
        v_hat = self._v_std / (1 - 0.999 ** self._t)
        self.log_std += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# -----------------------------------------------------------------------------
# REINFORCE Training Loop
# -----------------------------------------------------------------------------
def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t = Σ gamma^k * r_{t+k}."""
    returns = np.zeros_like(rewards, dtype=np.float64)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def train_reinforce(env, policy, episodes=500, gamma=0.99):
    """Train REINFORCE agent."""
    episode_rewards = []
    success_history = []
    
    print(f"Training REINFORCE for {episodes} episodes...")
    print(f"{'Ep':>5} | {'Reward':>8} | {'Steps':>5} | {'Success':>7}")
    print("-" * 40)
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        log_probs = []
        rewards = []
        
        policy.stored_obs = []
        policy.stored_actions = []
        
        while not done:
            action, log_prob, _ = policy.sample(obs)
            obs, reward, done, info = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            policy.stored_obs.append(obs.copy())
            policy.stored_actions.append(action.copy())
            total_reward += reward
        
        returns = compute_returns(rewards, gamma)
        policy.update(log_probs, returns)
        
        success = info.get("success", False)
        episode_rewards.append(total_reward)
        success_history.append(success)
        
        if (ep + 1) % 50 == 0 or ep == 0:
            recent_success = np.mean(success_history[-50:])
            print(f"{ep+1:5d} | {total_reward:8.2f} | {len(rewards):5d} | {recent_success:6.1%}")
    
    return episode_rewards, success_history


# -----------------------------------------------------------------------------
# Plotting & Visualization
# -----------------------------------------------------------------------------
def plot_results(episode_rewards, success_history, env, policy, save_dir):
    """Generate and save all plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Learning curve
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_rewards)), smoothed, 'b-', linewidth=2, label='Smoothed (w=20)')
    ax.plot(episode_rewards, alpha=0.3, color='lightblue', label='Raw')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('REINFORCE: Robotic Arm Reaching - Learning Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp12_learning_curve.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_dir}/exp12_learning_curve.png")
    
    # 2. Success rate
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 50
    if len(success_history) >= window:
        success_rate = np.convolve(
            [float(s) for s in success_history], np.ones(window)/window, mode='valid'
        )
        ax.plot(range(window-1, len(success_history)), success_rate * 100, 'g-', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('REINFORCE: Arm Reaching Success Rate (50-ep window)', fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp12_success_rate.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_dir}/exp12_success_rate.png")
    
    # 3. Arm trajectory visualization (last episode or a demo)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    demos = []
    for i in range(3):
        obs = env.reset()
        thetas = [env.theta.copy()]
        ee_traj = [env._forward_kinematics()]
        done = False
        while not done:
            mean_action = policy.forward(obs)[0]  # use mean, not sample
            obs, _, done, _ = env.step(mean_action)
            thetas.append(env.theta.copy())
            ee_traj.append(env._forward_kinematics())
        demos.append((env.target, np.array(ee_traj), thetas))
    
    for idx, (target, ee_traj, thetas) in enumerate(demos):
        ax = axes[idx]
        # Draw base
        ax.plot(0, 0, 'ks', markersize=10)
        
        # Draw arm links at selected timesteps
        n_frames = len(thetas)
        for t in range(0, n_frames, max(1, n_frames // 8)):
            th = thetas[t]
            l1, l2 = env.LINK_LENGTHS
            x1 = l1 * np.cos(th[0])
            y1 = l1 * np.sin(th[0])
            x2 = x1 + l2 * np.cos(th[0] + th[1])
            y2 = y1 + l2 * np.sin(th[0] + th[1])
            alpha = 0.2 + 0.8 * (t / n_frames)
            ax.plot([0, x1, x2], [0, y1, y2], 'o-', color='steelblue', 
                    alpha=alpha, linewidth=1.5, markersize=3)
        
        # End-effector trajectory
        ax.plot(ee_traj[:, 0], ee_traj[:, 1], 'r--', linewidth=1.5, label='EE path')
        ax.plot(ee_traj[-1, 0], ee_traj[-1, 1], 'r*', markersize=15, label='Final EE')
        ax.plot(target[0], target[1], 'g^', markersize=15, label='Target')
        
        # Draw success radius
        circle = plt.Circle(target, 0.15, fill=False, color='green', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        dist = np.linalg.norm(ee_traj[-1] - target)
        ax.set_title(f'Demo {idx+1} (dist={dist:.3f})', fontsize=12)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle('REINFORCE: Arm Trajectory Visualization', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp12_arm_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {save_dir}/exp12_arm_trajectory.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    np.random.seed(42)
    
    save_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    
    env = SimpleArmEnv(max_steps=200)
    policy = GaussianPolicy(obs_dim=env.obs_dim, act_dim=env.act_dim, lr=0.001, hidden=128)
    
    episodes = 500
    episode_rewards, success_history = train_reinforce(env, policy, episodes=episodes, gamma=0.99)
    
    # Final statistics
    final_50 = success_history[-50:]
    print(f"\n{'='*50}")
    print(f"Final 50-episode success rate: {np.mean(final_50):.1%}")
    print(f"Final 50-episode avg reward:   {np.mean(episode_rewards[-50:]):.2f}")
    print(f"{'='*50}")
    
    plot_results(episode_rewards, success_history, env, policy, save_dir)
    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    main()

