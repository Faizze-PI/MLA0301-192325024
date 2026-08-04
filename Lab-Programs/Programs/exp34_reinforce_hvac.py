"""
Exp 34: REINFORCE for Smart Home Temperature Optimization
==========================================================
Custom HVAC environment:
  state  = (indoor_temp, outdoor_temp, time_of_day, occupancy)
  action = heating/cooling level (discrete: 5 levels)
  reward = -(energy_cost + comfort_penalty)

gamma = 0.99, lr = 0.001, episodes = 500
Plots: learning curve + example day trajectory showing comfort band.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time


# ---------------------------------------------------------------------------
# HVAC Environment
# ---------------------------------------------------------------------------

class HVACEvn:
    """
    Simplified smart home HVAC environment.

    State: [indoor_temp, outdoor_temp, hour, occupancy]
      - indoor_temp: 15-30 C
      - outdoor_temp: follows daily sinusoidal pattern
      - hour: 0-23
      - occupancy: 0 (away) or 1 (home)

    Action: 0-4 (off, low cool, high cool, low heat, high heat)

    Reward: -(energy_cost + comfort_penalty)
    """

    COMFORT_LOW = 20.0
    COMFORT_HIGH = 24.0

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.indoor_temp = 22.0
        self.hour = 0
        self.day = 0
        self.total_energy = 0.0
        self.total_comfort_penalty = 0.0
        return self._state()

    def _outdoor_temp(self, hour):
        """Sinusoidal outdoor temperature pattern."""
        base = 15.0
        amplitude = 10.0
        return base + amplitude * np.sin(2 * np.pi * (hour - 6) / 24)

    def _occupancy(self, hour):
        """Simple occupancy schedule: home 7-9, away 9-17, home 17-23, away 0-7."""
        if 7 <= hour < 9 or 17 <= hour < 23:
            return 1.0
        return 0.0

    def _state(self):
        outdoor = self._outdoor_temp(self.hour)
        occ = self._occupancy(self.hour)
        return np.array([
            self.indoor_temp / 30.0,
            outdoor / 30.0,
            self.hour / 23.0,
            occ,
        ], dtype=np.float32)

    def step(self, action):
        """Execute one hour of HVAC operation."""
        outdoor = self._outdoor_temp(self.hour)
        occ = self._occupancy(self.hour)

        # HVAC effect
        hvac_power = [0.0, -0.5, -1.5, 0.5, 1.5]  # cooling and heating
        self.indoor_temp += hvac_power[action]

        # Natural drift towards outdoor temp
        drift = 0.02 * (outdoor - self.indoor_temp)
        self.indoor_temp += drift

        # Add small noise
        self.indoor_temp += self.rng.normal(0, 0.05)

        # Clip temperature
        self.indoor_temp = np.clip(self.indoor_temp, 10.0, 35.0)

        # Energy cost
        energy_cost = [0.0, 0.3, 0.8, 0.3, 0.8][action]

        # Comfort penalty
        comfort_penalty = 0.0
        if self.indoor_temp < self.COMFORT_LOW:
            comfort_penalty = (self.COMFORT_LOW - self.indoor_temp) * 0.5
        elif self.indoor_temp > self.COMFORT_HIGH:
            comfort_penalty = (self.indoor_temp - self.COMFORT_HIGH) * 0.5

        # Higher penalty when occupied
        if occ > 0.5:
            comfort_penalty *= 2.0

        reward = -(energy_cost + comfort_penalty)

        self.total_energy += energy_cost
        self.total_comfort_penalty += comfort_penalty

        # Advance time
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day += 1

        done = self.day >= 1  # One day per episode
        return self._state(), reward, done

    @property
    def state_size(self):
        return 4

    @property
    def action_size(self):
        return 5


# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

        # Policy network: 4 -> 64 -> 64 -> 5
        self.W1 = np.random.randn(state_size, 64) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, action_size) * 0.01
        self.b3 = np.zeros(action_size)

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _forward(self, s):
        h1 = np.maximum(0, s @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return self._softmax(logits)

    def select_action(self, state):
        s = np.array(state, dtype=np.float32).reshape(1, -1)
        probs = self._forward(s)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]

    def update(self, states, actions, rewards):
        """REINFORCE policy gradient update."""
        # Compute discounted returns
        returns = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient
        for t in range(len(states)):
            s = np.array(states[t], dtype=np.float32).reshape(1, -1)
            probs = self._forward(s)[0]
            a = actions[t]

            # Gradient: log pi(a|s) * G
            grad_logits = probs.copy()
            grad_logits[a] -= 1.0  # d log pi / d logits
            grad_logits *= -returns[t]  # negative because we want to maximize

            # Backprop
            h1 = np.maximum(0, s @ self.W1 + self.b1)
            h2 = np.maximum(0, h1 @ W2_placeholder if False else h1 @ self.W2 + self.b2)

            # Update output layer
            self.W3 -= self.lr * np.outer(h2[0], grad_logits)
            self.b3 -= self.lr * grad_logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(episodes=500):
    print(f"Training REINFORCE HVAC agent for {episodes} episodes ...")
    print(f"  gamma=0.99, lr=0.001")
    print(f"  Comfort band: {HVACEvn.COMFORT_LOW}-{HVACEvn.COMFORT_HIGH} C")

    env = HVACEvn(seed=42)
    agent = REINFORCEAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=0.001,
        gamma=0.99,
    )

    rewards_per_episode = []
    energy_per_episode = []
    comfort_per_episode = []

    t0 = time.time()

    for ep in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        agent.update(states, actions, rewards)

        total_reward = sum(rewards)
        rewards_per_episode.append(total_reward)
        energy_per_episode.append(env.total_energy)
        comfort_per_episode.append(env.total_comfort_penalty)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_per_episode[-100:])
            avg_e = np.mean(energy_per_episode[-100:])
            avg_c = np.mean(comfort_per_episode[-100:])
            print(f"  Episode {ep+1}: avg_reward={avg_r:.2f}, "
                  f"avg_energy={avg_e:.1f}, avg_comfort_pen={avg_c:.2f}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    return agent, rewards_per_episode, energy_per_episode, comfort_per_episode


# ---------------------------------------------------------------------------
# Example Day Trajectory
# ---------------------------------------------------------------------------

def simulate_day(agent):
    """Simulate one full day and record temperatures."""
    env = HVACEvn(seed=123)
    state = env.reset()
    temps = []
    actions_taken = []
    outdoor_temps = []
    occupancy = []

    done = False
    while not done:
        action, _ = agent.select_action(state)
        next_state, reward, done = env.step(action)

        temps.append(env.indoor_temp)
        actions_taken.append(action)
        outdoor_temps.append(env._outdoor_temp((env.hour - 1) % 24))
        occupancy.append(env._occupancy((env.hour - 1) % 24))

        state = next_state

    return temps, actions_taken, outdoor_temps, occupancy


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(rewards, energy, comfort, agent):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Learning curve
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, color="blue")
    window = min(30, len(rewards) // 5)
    if window > 1:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smooth, color="blue", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("REINFORCE Training: Reward Curve")
    ax.grid(True, alpha=0.3)

    # 2. Energy vs Comfort trade-off
    ax = axes[0, 1]
    ax.scatter(energy, comfort, alpha=0.3, s=10, color="purple")
    ax.set_xlabel("Total Energy Cost")
    ax.set_ylabel("Comfort Penalty")
    ax.set_title("Energy vs Comfort Trade-off")
    ax.grid(True, alpha=0.3)

    # 3. Example day trajectory
    ax = axes[1, 0]
    temps, actions, outdoor, occ = simulate_day(agent)
    hours = list(range(24))

    ax.plot(hours, temps, "b-o", markersize=4, label="Indoor Temp", linewidth=2)
    ax.plot(hours, outdoor, "r--s", markersize=3, label="Outdoor Temp", alpha=0.7)
    ax.axhspan(HVACEvn.COMFORT_LOW, HVACEvn.COMFORT_HIGH, alpha=0.2, color="green", label="Comfort Band")

    # Shade occupancy
    for h in hours:
        if occ[h] > 0.5:
            ax.axvspan(h, h + 1, alpha=0.1, color="yellow")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Example Day: Temperature Trajectory")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.3)

    # 4. Actions taken
    ax = axes[1, 1]
    action_labels = ["Off", "Low Cool", "High Cool", "Low Heat", "High Heat"]
    colors = ["gray", "blue", "darkblue", "red", "darkred"]
    for a in range(5):
        times = [h for h in range(24) if actions[h] == a]
        if times:
            ax.bar(times, [1] * len(times), color=colors[a], label=action_labels[a], alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Action")
    ax.set_title("HVAC Actions During Example Day")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 23)
    ax.set_yticks([])

    fig.suptitle("Exp 34: REINFORCE for Smart Home HVAC Optimization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp34_reinforce_hvac.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir}/exp34_reinforce_hvac.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent, rewards, energy, comfort = train(episodes=500)

    print(f"\nFinal Metrics (last 50 episodes):")
    print(f"  Avg reward:     {np.mean(rewards[-50:]):.2f}")
    print(f"  Avg energy:     {np.mean(energy[-50:]):.1f}")
    print(f"  Avg comfort:    {np.mean(comfort[-50:]):.2f}")

    plot_results(rewards, energy, comfort, agent)
