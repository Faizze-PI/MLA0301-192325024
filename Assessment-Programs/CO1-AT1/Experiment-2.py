"""
Experiment 2: Familiarization with OpenAI Gym/Gymnasium Environments
--------------------------------------------------------------------
Aim: To create, initialize, and interact with standard Reinforcement Learning
environments such as FrozenLake, CartPole, and MountainCar using Gymnasium to
understand the concepts of states, actions, rewards, and episodes.
"""

import gymnasium as gym


def explore_env(env_name, n_steps=10, **env_kwargs):
    print(f"\n{'=' * 60}")
    print(f"Environment: {env_name}")
    print("=" * 60)

    env = gym.make(env_name, **env_kwargs)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"Initial state: {obs}")

    total_reward = 0
    for step in range(1, n_steps + 1):
        action = env.action_space.sample()  # random action for demonstration
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Step {step:2d}: action={action}  next_state={obs}  "
            f"reward={reward}  terminated={terminated}  truncated={truncated}"
        )
        if terminated or truncated:
            print(">> Episode finished, resetting environment.")
            obs, info = env.reset()

    print(f"Total reward collected over {n_steps} steps: {total_reward}")
    env.close()


if __name__ == "__main__":
    # FrozenLake: discrete states (0-15), discrete actions (0-3: LEFT/DOWN/RIGHT/UP)
    explore_env("FrozenLake-v1", n_steps=10, is_slippery=False)

    # CartPole: continuous state (cart pos, vel, pole angle, angular vel), 2 actions
    explore_env("CartPole-v1", n_steps=10)

    # MountainCar: continuous state (position, velocity), 3 actions
    explore_env("MountainCar-v0", n_steps=10)
