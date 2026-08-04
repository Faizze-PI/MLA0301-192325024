"""
Experiment 8: Reward Visualization in Reinforcement Learning
----------------------------------------------------------------
Aim: To simulate an RL agent in a simple environment and visualize cumulative
rewards, episode rewards, and learning performance using Matplotlib graphs.

Agent: tabular Q-learning trained on FrozenLake-v1 (non-slippery, so the
agent can reliably learn a deterministic optimal path).
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def q_learning_frozenlake(n_episodes=2000, alpha=0.8, gamma=0.95,
                           epsilon_start=1.0, epsilon_decay=0.999,
                           epsilon_min=0.01, seed=1):
    rng = np.random.default_rng(seed)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        done = False

        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update rule
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    env.close()
    return Q, episode_rewards


if __name__ == "__main__":
    print("Training tabular Q-learning agent on FrozenLake-v1 (2000 episodes)...")
    Q, episode_rewards = q_learning_frozenlake(n_episodes=2000)

    cumulative_rewards = np.cumsum(episode_rewards)
    window = 50
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    success_rate = np.mean(episode_rewards[-200:]) * 100
    print(f"\nSuccess rate over the last 200 episodes: {success_rate:.1f}%")
    print("Learned Q-table (rounded):")
    print(np.round(Q, 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(episode_rewards, alpha=0.3, label="Episode reward")
    axes[0].plot(range(window - 1, len(episode_rewards)), moving_avg,
                 label=f"{window}-episode moving average", linewidth=2)
    axes[0].set_title("Episode Rewards over Training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumulative_rewards)
    axes[1].set_title("Cumulative Reward over Training")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment8_reward_visualization.png", dpi=150)
    print("\nPlot saved as 'experiment8_reward_visualization.png'")
    plt.show()
