"""
=============================================================================
Experiment 13: REINFORCE Algorithm for Autonomous Parking
=============================================================================
REINFORCE with discounted returns + running-mean baseline (variance reduction)
applied to an autonomous parking task.

Environment:
  - Custom parking grid: 10x10 grid, car must navigate from start to a
    designated parking spot while avoiding obstacles.
  - State: [car_x, car_y, target_x, target_y, heading, obstacle_flag_8]
  - Action: {0: forward, 1: left, 2: right, 3: brake}
  - Reward: +100 for parking, -1 per step, -10 for collision

Algorithm:
  - REINFORCE with baseline (running mean of returns)
  - gamma = 0.99, lr = 0.001, episodes = 1000
  - Comparison: with baseline vs without baseline (variance reduction)

Outputs:
  - Console: episode rewards, success rate
  - Plots saved: learning curves (with/without baseline),
    variance comparison, parking trajectory
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from collections import deque

# -----------------------------------------------------------------------------
# Custom Parking Environment
# -----------------------------------------------------------------------------
class ParkingEnv:
    """Simple grid-based parking environment."""
    
    GRID_SIZE = 10
    ACTIONS = {0: 'forward', 1: 'left', 2: 'right', 3: 'brake'}
    
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.obs_dim = 13  # car_x, car_y, tx, ty, heading, 8 obstacle flags
        self.act_dim = 4
        self.reset()
    
    def reset(self):
        # Random start (not on target)
        self.car_pos = np.array([np.random.randint(0, self.GRID_SIZE), 
                                  np.random.randint(0, self.GRID_SIZE)])
        self.heading = np.random.choice([0, 1, 2, 3])  # N, E, S, W
        
        # Target parking spot
        self.target = np.array([np.random.randint(7, self.GRID_SIZE),
                                 np.random.randint(7, self.GRID_SIZE)])
        while np.array_equal(self.car_pos, self.target):
            self.target = np.array([np.random.randint(7, self.GRID_SIZE),
                                     np.random.randint(7, self.GRID_SIZE)])
        
        # Random obstacles (not on car or target)
        self.obstacles = set()
        n_obs = np.random.randint(5, 10)
        for _ in range(n_obs):
            ox, oy = np.random.randint(0, self.GRID_SIZE, size=2)
            if not np.array_equal([ox, oy], self.car_pos) and \
               not np.array_equal([ox, oy], self.target):
                self.obstacles.add((ox, oy))
        
        self.step_count = 0
        self.trajectory = [self.car_pos.copy()]
        return self._get_obs()
    
    def _get_obs(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = self.car_pos[0] / self.GRID_SIZE
        obs[1] = self.car_pos[1] / self.GRID_SIZE
        obs[2] = self.target[0] / self.GRID_SIZE
        obs[3] = self.target[1] / self.GRID_SIZE
        obs[4] = self.heading / 4.0
        # 8-direction obstacle flags
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for i, (dx, dy) in enumerate(directions):
            nx, ny = self.car_pos[0] + dx, self.car_pos[1] + dy
            if (nx, ny) in self.obstacles:
                obs[5 + i] = 1.0
        return obs
    
    def step(self, action):
        self.step_count += 1
        reward = -1.0
        done = False
        info = {"collision": False, "success": False}
        
        # Movement vectors: N, E, S, W
        move_map = {0: np.array([0, 1]), 1: np.array([1, 0]),
                     2: np.array([0, -1]), 3: np.array([-1, 0])}
        
        if action == 0:  # forward
            new_pos = self.car_pos + move_map[self.heading]
        elif action == 1:  # left
            self.heading = (self.heading - 1) % 4
            new_pos = self.car_pos
        elif action == 2:  # right
            self.heading = (self.heading + 1) % 4
            new_pos = self.car_pos
        else:  # brake
            new_pos = self.car_pos
        
        # Bounds check
        if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
            self.car_pos = new_pos
        
        # Collision check
        if tuple(self.car_pos) in self.obstacles:
            reward -= 10.0
            info["collision"] = True
            done = True
        
        self.trajectory.append(self.car_pos.copy())
        
        # Success check
        if np.array_equal(self.car_pos, self.target):
            reward += 100.0
            info["success"] = True
            done = True
        
        if self.step_count >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, info


# -----------------------------------------------------------------------------
# Softmax Policy
# -----------------------------------------------------------------------------
class SoftmaxPolicy:
    """Softmax policy for discrete actions."""
    
    def __init__(self, obs_dim, act_dim, lr=0.001, hidden=64):
        self.act_dim = act_dim
        self.lr = lr
        scale = 1.0 / np.sqrt(hidden)
        
        self.W1 = np.random.randn(hidden, obs_dim) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(act_dim, hidden) * scale
        self.b2 = np.zeros(act_dim)
        
        self._t = 0
        self._m = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2]]
        self._v = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2]]
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, obs):
        obs = np.atleast_2d(obs)  # ensure (N, obs_dim)
        h = self._relu(obs @ self.W1.T + self.b1)  # (N, hidden)
        logits = h @ self.W2.T + self.b2  # (N, act_dim)
        # Stable softmax
        logits -= logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-8)
        if probs.shape[0] == 1:
            return probs[0]  # return 1D for single obs
        return probs
    
    def sample(self, obs):
        probs = self.forward(obs)
        action = np.random.choice(self.act_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob
    
    def get_log_probs_batch(self, obs):
        probs = self.forward(obs)
        return np.log(probs + 1e-8)
    
    def update(self, obs_list, actions_list, advantages):
        """REINFORCE with advantage (returns - baseline)."""
        self._t += 1
        obs = np.array(obs_list)
        actions = np.array(actions_list)
        advantages = np.array(advantages, dtype=np.float64)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute gradient via log-prob trick
        log_probs_all = self.get_log_probs_batch(obs)  # (N, act_dim)
        
        # One-hot for taken actions
        one_hot = np.eye(self.act_dim)[actions]
        
        # Score function: (one_hot - probs) * advantage
        probs = self.forward(obs)
        diff = (one_hot - probs)  # (N, act_dim)
        
        # Backprop gradient w.r.t. W2, b2, W1, b1
        h_all = self._relu(obs @ self.W1.T + self.b1)  # (N, hidden)
        
        # Gradient of loss = -log_prob * advantage => dL/dlogits = (probs - one_hot) * adv
        # But we want gradient ascent: dJ/dtheta = Σ G_t * ∇log pi
        # So: grad = Σ advantage * (one_hot - probs) applied to logits gradient
        
        # ∂logits/∂W2 = h^T, ∂logits/∂b2 = 1
        grad_W2 = (diff * advantages[:, None]).T @ h_all / len(obs)
        grad_b2 = (diff * advantages[:, None]).mean(axis=0)
        
        # Chain through relu
        pre_act = obs @ self.W1.T + self.b1  # (N, hidden)
        d_relu = (pre_act > 0).astype(float)
        grad_h = diff * advantages[:, None] @ self.W2  # (N, hidden)
        grad_pre = grad_h * d_relu
        
        grad_W1 = grad_pre.T @ obs / len(obs)
        grad_b1 = grad_pre.mean(axis=0)
        
        # Adam update
        grads = [grad_W1, grad_b1, grad_W2, grad_b2]
        params = [self.W1, self.b1, self.W2, self.b2]
        for i in range(len(params)):
            self._m[i] = 0.9 * self._m[i] + 0.1 * grads[i]
            self._v[i] = 0.999 * self._v[i] + 0.001 * grads[i] ** 2
            m_hat = self._m[i] / (1 - 0.9 ** self._t)
            v_hat = self._v[i] / (1 - 0.999 ** self._t)
            params[i] += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        self.W1, self.b1, self.W2, self.b2 = params


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def compute_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float64)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def train(env, policy, episodes=1000, gamma=0.99, use_baseline=True):
    """Train REINFORCE with optional baseline."""
    episode_rewards = []
    success_history = []
    baseline = 0.0
    baseline_decay = 0.99
    reward_variances = []
    
    label = "with baseline" if use_baseline else "without baseline"
    print(f"\nTraining REINFORCE {label} for {episodes} episodes...")
    print(f"{'Ep':>5} | {'Reward':>8} | {'Steps':>5} | {'Success':>7}")
    print("-" * 40)
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        obs_list, actions_list, rewards_list = [], [], []
        
        while not done:
            action, log_prob = policy.sample(obs)
            obs_next, reward, done, info = env.step(action)
            obs_list.append(obs.copy())
            actions_list.append(action)
            rewards_list.append(reward)
            total_reward += reward
            obs = obs_next
        
        returns = compute_returns(rewards_list, gamma)
        
        if use_baseline:
            advantages = returns - baseline
            baseline = baseline_decay * baseline + (1 - baseline_decay) * returns.mean()
        else:
            advantages = returns
        
        policy.update(obs_list, actions_list, advantages)
        
        episode_rewards.append(total_reward)
        success_history.append(info.get("success", False))
        reward_variances.append(np.var(returns))
        
        if (ep + 1) % 100 == 0 or ep == 0:
            recent = success_history[-100:]
            print(f"{ep+1:5d} | {total_reward:8.2f} | {len(rewards_list):5d} | {np.mean(recent):6.1%}")
    
    return episode_rewards, success_history, reward_variances


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(rewards_no_base, rewards_with_base,
                 success_no, success_with,
                 var_no, var_with,
                 env, policy_no, policy_with, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    window = 50
    
    # 1. Learning curve comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(rewards_no_base) >= window:
        s_no = np.convolve(rewards_no_base, np.ones(window)/window, mode='valid')
        s_with = np.convolve(rewards_with_base, np.ones(window)/window, mode='valid')
        x = range(window-1, len(rewards_no_base))
        ax.plot(x, s_no, 'r-', linewidth=2, label='Without Baseline')
        ax.plot(x, s_with, 'b-', linewidth=2, label='With Baseline')
    ax.plot(rewards_no_base, alpha=0.15, color='red')
    ax.plot(rewards_with_base, alpha=0.15, color='blue')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('REINFORCE Parking: With vs Without Baseline', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp13_learning_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp13_learning_comparison.png")
    
    # 2. Success rate comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(success_no) >= window:
        sr_no = np.convolve([float(s) for s in success_no], np.ones(window)/window, mode='valid')
        sr_with = np.convolve([float(s) for s in success_with], np.ones(window)/window, mode='valid')
        x = range(window-1, len(success_no))
        ax.plot(x, sr_no * 100, 'r-', linewidth=2, label='Without Baseline')
        ax.plot(x, sr_with * 100, 'b-', linewidth=2, label='With Baseline')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Parking Success Rate: Baseline Impact', fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp13_success_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp13_success_comparison.png")
    
    # 3. Return variance comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(var_no) >= window:
        v_no = np.convolve(var_no, np.ones(window)/window, mode='valid')
        v_with = np.convolve(var_with, np.ones(window)/window, mode='valid')
        x = range(window-1, len(var_no))
        ax.plot(x, v_no, 'r-', linewidth=2, label='Without Baseline')
        ax.plot(x, v_with, 'b-', linewidth=2, label='With Baseline')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return Variance', fontsize=12)
    ax.set_title('Variance Reduction: Return Variance Over Training', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp13_variance_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"[Saved] exp13_variance_comparison.png")
    
    # 4. Parking trajectory visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (policy, label) in enumerate([(policy_no, "Without Baseline"), 
                                            (policy_with, "With Baseline")]):
        ax = axes[idx]
        obs = env.reset()
        traj = [env.car_pos.copy()]
        target = env.target.copy()
        obstacles = env.obstacles.copy()
        done = False
        
        while not done:
            mean_action = np.argmax(policy.forward(obs))
            obs, _, done, _ = env.step(mean_action)
            traj.append(env.car_pos.copy())
        
        traj = np.array(traj)
        
        # Draw grid
        for i in range(env.GRID_SIZE + 1):
            ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
            ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
        
        # Draw obstacles
        for ox, oy in obstacles:
            rect = mpatches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, 
                                       facecolor='gray', edgecolor='black')
            ax.add_patch(rect)
        
        # Draw target
        ax.plot(target[0], target[1], 'g^', markersize=15, label='Target')
        
        # Draw trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'b-o', markersize=3, linewidth=1.5, label='Path')
        ax.plot(traj[0, 0], traj[0, 1], 'bs', markersize=10, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
        
        success = np.array_equal(traj[-1], target)
        ax.set_title(f'{label} ({"Success" if success else "Failed"})', fontsize=12)
        ax.set_xlim(-0.5, env.GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, env.GRID_SIZE - 0.5)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(False)
    
    fig.suptitle('Parking Trajectory: REINFORCE with vs without Baseline', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp13_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] exp13_trajectory.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    np.random.seed(42)
    save_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    
    episodes = 1000
    env = ParkingEnv(max_steps=80)
    
    # Train WITHOUT baseline
    env1 = ParkingEnv(max_steps=80)
    policy_no = SoftmaxPolicy(env1.obs_dim, env1.act_dim, lr=0.001, hidden=64)
    rewards_no, success_no, var_no = train(env1, policy_no, episodes=episodes, use_baseline=False)
    
    # Train WITH baseline
    env2 = ParkingEnv(max_steps=80)
    policy_with = SoftmaxPolicy(env2.obs_dim, env2.act_dim, lr=0.001, hidden=64)
    rewards_with, success_with, var_with = train(env2, policy_with, episodes=episodes, use_baseline=True)
    
    # Final comparison
    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'No Baseline':>14} {'With Baseline':>14}")
    print(f"{'='*60}")
    print(f"{'Final 100-ep avg reward':<30} {np.mean(rewards_no[-100:]):>14.2f} {np.mean(rewards_with[-100:]):>14.2f}")
    print(f"{'Final 100-ep success rate':<30} {np.mean(success_no[-100:]):>13.1%} {np.mean(success_with[-100:]):>13.1%}")
    print(f"{'Final 100-ep return variance':<30} {np.mean(var_no[-100:]):>14.2f} {np.mean(var_with[-100:]):>14.2f}")
    print(f"{'='*60}")
    
    plot_results(rewards_no, rewards_with, success_no, success_with,
                 var_no, var_with, env1, policy_no, policy_with, save_dir)
    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    main()

