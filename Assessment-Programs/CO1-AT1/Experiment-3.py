"""
Experiment 3: Implementation of the Reinforcement Learning Framework
--------------------------------------------------------------------
Aim: To implement the basic Reinforcement Learning framework by designing an
agent that interacts with an environment through states, actions, rewards, and
policies using Python.
"""

import random


class GridWorldEnv:
    """A simple size x size grid world. Agent starts at top-left, goal is
    bottom-right. Every step costs -1, reaching the goal gives +10."""

    def __init__(self, size=4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.state = (0, 0)
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == "UP":
            x = max(0, x - 1)
        elif action == "DOWN":
            x = min(self.size - 1, x + 1)
        elif action == "LEFT":
            y = max(0, y - 1)
        elif action == "RIGHT":
            y = min(self.size - 1, y + 1)
        self.state = (x, y)

        done = self.state == self.goal
        reward = 10 if done else -1
        return self.state, reward, done


class Agent:
    """A simple agent following a uniform random policy."""

    def __init__(self, actions):
        self.actions = actions

    def policy(self, state):
        return random.choice(self.actions)


def run_episode(env, agent, max_steps=50, verbose=True):
    state = env.reset()
    total_reward = 0
    for t in range(1, max_steps + 1):
        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        if verbose:
            print(f"Step {t:2d}: state={state} action={action:5s} "
                  f"next_state={next_state} reward={reward}")
        state = next_state
        if done:
            print(f"\nGoal reached in {t} steps! Total reward: {total_reward}")
            return total_reward, t
    print(f"\nMax steps reached without finding goal. Total reward: {total_reward}")
    return total_reward, max_steps


if __name__ == "__main__":
    random.seed(7)
    env = GridWorldEnv(size=4)
    agent = Agent(env.actions)

    print("=" * 60)
    print("Single episode (random policy) -- detailed trace")
    print("=" * 60)
    run_episode(env, agent, max_steps=50, verbose=True)

    print("\n" + "=" * 60)
    print("Summary over 5 episodes")
    print("=" * 60)
    for ep in range(1, 6):
        total_reward, steps = run_episode(env, agent, max_steps=50, verbose=False)
        print(f"Episode {ep}: steps={steps:3d}  total_reward={total_reward}")
