"""
Experiment 21: RL-Based Smart Energy Management (Safe RL)
==========================================================
Custom environment for smart grid energy management where the agent
decides how to draw power: from the grid, from a battery, or defer
non-critical load.

State: (time_of_day, battery_level, demand_forecast)
Action: 0=grid, 1=battery, 2=defer
Constrained reward: cost minimization + safety penalty for power shortfall.

Safety-weight lambda swept: [0.5, 1.0, 2.0]
Pareto plot: total cost vs constraint-violation rate across lambda.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import deque, namedtuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. Install with: pip install torch")

# --- Environment ---------------------------------------------------------------

class SmartEnergyEnv:
    """
    Smart energy management environment.
    24 time slots (hours), battery capacity 0-100.
    Demand varies sinusoidally with peak at noon.
    Grid price varies (cheap off-peak, expensive peak).
    Safety constraint: battery must never drop below critical threshold
    (simulating power shortfall for critical load).
    """
    ACTIONS = 3  # grid, battery, defer
    ACTION_NAMES = ['grid', 'battery', 'defer']
    BATTERY_CAPACITY = 100.0
    CRITICAL_BATTERY = 10.0  # safety constraint threshold
    DEFER_PENALTY = 5.0  # user discomfort from deferring

    def __init__(self, n_hours=24, seed=42):
        self.n_hours = n_hours
        self.rng = np.random.RandomState(seed)
        self.demand_profile = self._generate_demand()
        self.price_profile = self._generate_price()
        self.reset()

    def _generate_demand(self):
        """Sinusoidal demand with noise."""
        t = np.linspace(0, 2*np.pi, self.n_hours)
        demand = 50 + 30 * np.sin(t - np.pi/3) + self.rng.randn(self.n_hours) * 5
        return np.clip(demand, 10, 90)

    def _generate_price(self):
        """Time-of-use pricing: cheap off-peak, expensive peak."""
        t = np.linspace(0, 2*np.pi, self.n_hours)
        price = 0.08 + 0.04 * np.sin(t - np.pi/2) + self.rng.randn(self.n_hours) * 0.005
        return np.clip(price, 0.03, 0.15)

    def reset(self):
        self.hour = 0
        self.battery = 50.0
        self.total_cost = 0.0
        self.total_deferred = 0
        self.constraint_violations = 0
        self.step_count = 0
        self.demand_profile = self._generate_demand()
        return self._obs()

    def _obs(self):
        return np.array([
            self.hour / self.n_hours,
            self.battery / self.BATTERY_CAPACITY,
            self.demand_profile[self.hour] / 90.0,
        ], dtype=np.float32)

    def step(self, action):
        demand = self.demand_profile[self.hour]
        price = self.price_profile[self.hour]
        self.step_count += 1
        reward = 0.0
        violation = False

        if action == 0:  # Draw from grid
            cost = demand * price
            self.battery = min(self.BATTERY_CAPACITY, self.battery + demand * 0.1)
            self.total_cost += cost
            reward = -cost
        elif action == 1:  # Draw from battery
            if self.battery >= demand * 0.5:
                self.battery -= demand * 0.5
                cost = 0.0
                reward = 0.0
            else:
                # Battery insufficient -> safety violation
                cost = demand * price * 0.5
                self.battery = max(0, self.battery - demand * 0.5)
                self.total_cost += cost
                reward = -cost
                violation = True
        elif action == 2:  # Defer load
            self.total_deferred += 1
            cost = self.DEFER_PENALTY
            reward = -cost * 0.5  # lighter penalty than grid cost
            self.battery = min(self.BATTERY_CAPACITY, self.battery + 2)

        # Battery self-discharge each step
        self.battery = max(0, self.battery - 0.5)

        # Safety constraint check
        if self.battery < self.CRITICAL_BATTERY:
            self.constraint_violations += 1
            reward -= 20.0  # large safety penalty

        # Move to next hour
        self.hour = (self.hour + 1) % self.n_hours
        done = self.step_count >= self.n_hours * 3  # 3 days simulation
        return self._obs(), reward, done, {'violation': violation}


# --- DQN Agent -----------------------------------------------------------------

if HAS_TORCH:
    class DQN(nn.Module):
        def __init__(self, obs_dim=3, n_actions=3, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )

        def forward(self, x):
            return self.net(x)

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    class ReplayBuffer:
        def __init__(self, capacity=5000):
            self.buffer = deque(maxlen=capacity)

        def push(self, *args):
            self.buffer.append(Transition(*args))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states = np.array([t.state for t in batch])
            actions = np.array([t.action for t in batch])
            rewards = np.array([t.reward for t in batch])
            next_states = np.array([t.next_state for t in batch])
            dones = np.array([t.done for t in batch], dtype=float)
            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.buffer)

    class SafeDQNAgent:
        def __init__(self, obs_dim=3, n_actions=3, safety_weight=1.0, lr=1e-3, gamma=0.99,
                     eps=1.0, eps_decay=0.999, eps_min=0.05, buffer_size=5000, batch_size=64):
            self.q_net = DQN(obs_dim, n_actions)
            self.target_net = DQN(obs_dim, n_actions)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.buffer = ReplayBuffer(buffer_size)
            self.safety_weight = safety_weight
            self.gamma = gamma
            self.epsilon = eps
            self.eps_decay = eps_decay
            self.eps_min = eps_min
            self.n_actions = n_actions
            self.batch_size = batch_size
            self.train_steps = 0

        def select_action(self, state):
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                return int(self.q_net(s).argmax(dim=1).item())

        def update(self):
            if len(self.buffer) < self.batch_size:
                return
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            s = torch.FloatTensor(states)
            a = torch.LongTensor(actions).unsqueeze(1)
            r = torch.FloatTensor(rewards).unsqueeze(1)
            ns = torch.FloatTensor(next_states)
            d = torch.FloatTensor(dones).unsqueeze(1)

            q_vals = self.q_net(s).gather(1, a)
            with torch.no_grad():
                next_q = self.target_net(ns).max(dim=1, keepdim=True)[0]
                target = r + self.gamma * next_q * (1 - d)

            loss = F.smooth_l1_loss(q_vals, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            self.train_steps += 1
            if self.train_steps % 50 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        def apply_safety_constraint(self, env, state):
            """Modify Q-values to penalize unsafe actions."""
            with torch.no_grad():
                q = self.q_net(torch.FloatTensor(state).unsqueeze(0)).squeeze()
                if env.battery < env.CRITICAL_BATTERY + 10:
                    # Prefer grid when battery is low
                    q[1] -= self.safety_weight * 20  # penalize battery draw
                    q[2] -= self.safety_weight * 10  # penalize defer
            return int(q.argmax().item())


# --- Training ------------------------------------------------------------------

def train_safe_dqn(safety_weight, n_episodes=200):
    env = SmartEnergyEnv()
    agent = SafeDQNAgent(safety_weight=safety_weight)

    episode_costs = []
    episode_violations = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        ep_cost = 0
        ep_violations = 0

        while not done:
            action = agent.apply_safety_constraint(env, state)
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            total_reward += reward
            ep_cost += max(0, -reward)
            if info['violation']:
                ep_violations += 1
            state = next_state

        episode_costs.append(ep_cost)
        episode_violations.append(ep_violations)

        if (ep + 1) % 40 == 0:
            avg_c = np.mean(episode_costs[-40:])
            avg_v = np.mean(episode_violations[-40:])
            print(f"  Lambda={safety_weight:.1f} | Ep {ep+1:4d} | "
                  f"Avg Cost: {avg_c:7.1f} | Avg Violations: {avg_v:3.1f} | "
                  f"Eps: {agent.epsilon:.3f}")

    return episode_costs, episode_violations


# --- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 21: RL-Based Smart Energy Management (Safe RL)")
    print("=" * 70)

    if not HAS_TORCH:
        print("PyTorch is required. Please install: pip install torch")
        return

    lambdas = [0.5, 1.0, 2.0]
    n_episodes = 100
    all_costs = {}
    all_violations = {}

    for lam in lambdas:
        print(f"\n--- Training with safety lambda = {lam} ---")
        costs, violations = train_safe_dqn(lam, n_episodes)
        all_costs[lam] = costs
        all_violations[lam] = violations

    # --- Results ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: Pareto Analysis")
    print("=" * 70)

    pareto_costs = []
    pareto_violations = []
    for lam in lambdas:
        c = np.mean(all_costs[lam][-50:])
        v = np.mean(all_violations[lam][-50:])
        rate = v / 72.0  # 3 days = 72 hours
        pareto_costs.append(c)
        pareto_violations.append(rate)
        print(f"  Lambda={lam:.1f} | Avg Cost: {c:7.1f} | Violation Rate: {rate:.3f}")

    # --- Plots --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 21: Safe RL for Smart Energy Management", fontsize=14, fontweight='bold')

    window = 20
    colors = ['dodgerblue', 'orange', 'tomato']
    for i, lam in enumerate(lambdas):
        smooth_c = np.convolve(all_costs[lam], np.ones(window)/window, mode='valid')
        axes[0].plot(smooth_c, label=f'lambda={lam}', color=colors[i], linewidth=1.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Cost')
    axes[0].set_title('Learning Curve: Cost')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, lam in enumerate(lambdas):
        smooth_v = np.convolve(all_violations[lam], np.ones(window)/window, mode='valid')
        axes[1].plot(smooth_v, label=f'lambda={lam}', color=colors[i], linewidth=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Safety Violations per Episode')
    axes[1].set_title('Learning Curve: Violations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Pareto plot
    axes[2].scatter(pareto_costs, pareto_violations, s=100, c=colors[:len(lambdas)], zorder=5)
    for i, lam in enumerate(lambdas):
        axes[2].annotate(f'lambda={lam}', (pareto_costs[i], pareto_violations[i]),
                         textcoords="offset points", xytext=(10, 5), fontsize=10)
    axes[2].plot(pareto_costs, pareto_violations, '--', color='gray', alpha=0.5)
    axes[2].set_xlabel('Total Cost')
    axes[2].set_ylabel('Constraint Violation Rate')
    axes[2].set_title('Pareto Front: Cost vs Safety')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    plt.savefig(os.path.join(out_dir, "exp21_safe_energy_results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: exp21_safe_energy_results.png")
    plt.show()


if __name__ == "__main__":
    main()

