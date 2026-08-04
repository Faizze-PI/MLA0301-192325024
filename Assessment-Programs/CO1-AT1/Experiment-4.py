"""
Experiment 4: Simulation of a Markov Decision Process (MDP)
------------------------------------------------------------
Aim: To develop a simple Markov Decision Process model and simulate state
transitions, reward functions, and policy execution for understanding
sequential decision-making problems.

This uses the classic "Student MDP" (Sutton & Barto style teaching example):
a student can Study, browse Facebook, go to the Pub, Quit, or Sleep, moving
between Class1 -> Class2 -> Class3 -> Pass -> Sleep with rewards attached to
each transition.
"""

import random

# Transition model: mdp[state][action] = [(probability, next_state, reward), ...]
mdp = {
    "Class1": {
        "Study": [(1.0, "Class2", -2)],
        "Facebook": [(1.0, "FacebookState", -1)],
    },
    "Class2": {
        "Study": [(1.0, "Class3", -2)],
        "Sleep": [(1.0, "Sleep", 0)],
    },
    "Class3": {
        "Study": [(1.0, "Pass", 10)],
        "Pub": [(0.2, "Class1", 1), (0.4, "Class2", 1), (0.4, "Class3", 1)],
    },
    "FacebookState": {
        "Facebook": [(1.0, "FacebookState", -1)],
        "Quit": [(1.0, "Class1", 0)],
    },
    "Pass": {
        "Sleep": [(1.0, "Sleep", 0)],
    },
    "Sleep": {},  # terminal state, no outgoing actions
}


def sample_next_state(outcomes, rng):
    """Sample (next_state, reward) according to the transition probabilities."""
    r = rng.random()
    cumulative = 0.0
    for prob, next_state, reward in outcomes:
        cumulative += prob
        if r <= cumulative:
            return next_state, reward
    return outcomes[-1][1], outcomes[-1][2]  # fallback for float rounding


def random_policy(state, available_actions, rng):
    return rng.choice(available_actions)


def simulate_mdp(policy, start_state="Class1", max_steps=20, seed=None):
    rng = random.Random(seed)
    state = start_state
    trajectory = [state]
    total_reward = 0

    for t in range(1, max_steps + 1):
        if not mdp.get(state):
            print(f"Reached terminal state '{state}'. Ending simulation.")
            break

        available_actions = list(mdp[state].keys())
        action = policy(state, available_actions, rng)
        outcomes = mdp[state][action]
        next_state, reward = sample_next_state(outcomes, rng)

        total_reward += reward
        print(f"Step {t:2d}: {state:14s} --[{action:8s}]--> {next_state:14s} (reward={reward:+d})")

        trajectory.append(next_state)
        state = next_state

    return trajectory, total_reward


if __name__ == "__main__":
    print("=" * 70)
    print("Simulating Student MDP with a random policy")
    print("=" * 70)
    trajectory, total_reward = simulate_mdp(random_policy, start_state="Class1", seed=7)

    print("\nTrajectory:", " -> ".join(trajectory))
    print(f"Total reward for this episode: {total_reward}")

    print("\n" + "=" * 70)
    print("Running 5 independent episodes to see reward variability")
    print("=" * 70)
    for ep in range(1, 6):
        _, ep_reward = simulate_mdp(random_policy, start_state="Class1",
                                     max_steps=20, seed=ep)
        print(f"\nEpisode {ep} total reward: {ep_reward}")
        print("-" * 70)
