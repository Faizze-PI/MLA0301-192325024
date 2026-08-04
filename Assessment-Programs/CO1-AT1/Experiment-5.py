"""
Experiment 5: Bellman Equation Demonstration
----------------------------------------------
Aim: To implement Bellman Expectation and Bellman Optimality Equations for
calculating state-value and action-value functions and analyze the
convergence of value estimation.

Environment: classic 4x4 GridWorld (Sutton & Barto), two terminal states at
opposite corners, every non-terminal move costs -1.
"""

import numpy as np

GRID_SIZE = 4
GAMMA = 0.9
THETA = 1e-4  # convergence threshold

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_EFFECTS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
TERMINAL_STATES = [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]


def step(state, action):
    if state in TERMINAL_STATES:
        return state, 0
    dx, dy = ACTION_EFFECTS[action]
    x, y = state
    nx = max(0, min(GRID_SIZE - 1, x + dx))
    ny = max(0, min(GRID_SIZE - 1, y + dy))
    reward = -1
    return (nx, ny), reward


def bellman_expectation(gamma=GAMMA, theta=THETA, max_iter=1000):
    """Iterative policy evaluation for a uniform random policy
    (Bellman Expectation Equation)."""
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    action_prob = 1.0 / len(ACTIONS)

    for iteration in range(1, max_iter + 1):
        V_new = np.copy(V)
        delta = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in TERMINAL_STATES:
                    continue
                v = 0.0
                for a in ACTIONS:
                    (nx, ny), reward = step((x, y), a)
                    v += action_prob * (reward + gamma * V[nx, ny])
                V_new[x, y] = v
                delta = max(delta, abs(V_new[x, y] - V[x, y]))
        V = V_new
        if delta < theta:
            print(f"Bellman Expectation Equation converged after {iteration} iterations.")
            break
    return V


def bellman_optimality(gamma=GAMMA, theta=THETA, max_iter=1000):
    """Value iteration using the Bellman Optimality Equation."""
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    for iteration in range(1, max_iter + 1):
        V_new = np.copy(V)
        delta = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in TERMINAL_STATES:
                    continue
                action_values = []
                for a in ACTIONS:
                    (nx, ny), reward = step((x, y), a)
                    action_values.append(reward + gamma * V[nx, ny])
                V_new[x, y] = max(action_values)
                delta = max(delta, abs(V_new[x, y] - V[x, y]))
        V = V_new
        if delta < theta:
            print(f"Bellman Optimality Equation converged after {iteration} iterations.")
            break
    return V


def extract_greedy_policy(V, gamma=GAMMA):
    policy = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) in TERMINAL_STATES:
                policy[x, y] = "GOAL"
                continue
            best_action, best_value = None, -np.inf
            for a in ACTIONS:
                (nx, ny), reward = step((x, y), a)
                value = reward + gamma * V[nx, ny]
                if value > best_value:
                    best_value, best_action = value, a
            policy[x, y] = best_action
    return policy


if __name__ == "__main__":
    print("=" * 60)
    print("Bellman Expectation Equation (random policy evaluation)")
    print("=" * 60)
    V_expectation = bellman_expectation()
    print("State-Value function V(s):")
    print(np.round(V_expectation, 2))

    print("\n" + "=" * 60)
    print("Bellman Optimality Equation (value iteration)")
    print("=" * 60)
    V_optimal = bellman_optimality()
    print("Optimal state-value function V*(s):")
    print(np.round(V_optimal, 2))

    print("\nGreedy policy derived from V*(s):")
    policy = extract_greedy_policy(V_optimal)
    print(policy)
