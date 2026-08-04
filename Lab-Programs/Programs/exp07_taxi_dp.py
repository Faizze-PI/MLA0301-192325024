"""
Experiment 07 - Dynamic Programming (Policy Iteration) on Taxi-v3
==================================================================
Policy Iteration using env.P from gymnasium.make("Taxi-v3").

Steps:
  1. Policy Evaluation   – iterate V(s) until |Δ| < threshold
  2. Policy Improvement  – update policy to greedy w.r.t. Q(s,a)
  3. Repeat until policy converges

gamma=0.99, eval threshold=1e-4.
Prints number of cycles to convergence + average reward over 100 test episodes.
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Hyperparameters ------------------------------------------------------------
GAMMA       = 0.99
THRESHOLD   = 1e-4
MAX_EVAL    = 1000        # max sweeps inside policy evaluation
TEST_EPS    = 100         # episodes for final evaluation


# -- Policy Evaluation ---------------------------------------------------------
def policy_evaluation(P, nS, policy, gamma=GAMMA, threshold=THRESHOLD):
    V = np.zeros(nS)
    for sweep in range(1, MAX_EVAL + 1):
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            a = policy[s]
            # Bellman expectation: V(s) = Σ p(s',r|s,a) [r + gamma V(s')]
            v_new = sum(prob * (r + gamma * V[s_])
                        for prob, s_, r, _ in P[s][a])
            V[s] = v_new
            delta = max(delta, abs(v_new - v_old))
        if delta < threshold:
            return V, sweep
    return V, MAX_EVAL


# -- Policy Improvement --------------------------------------------------------
def policy_improvement(P, nS, nA, V, gamma=GAMMA):
    policy = np.zeros(nS, dtype=int)
    stable = True
    for s in range(nS):
        old_action = policy[s]
        q_values = []
        for a in range(nA):
            q = sum(prob * (r + gamma * V[s_])
                    for prob, s_, r, _ in P[s][a])
            q_values.append(q)
        policy[s] = int(np.argmax(q_values))
        if old_action != policy[s]:
            stable = False
    return policy, stable


# -- Policy Iteration ----------------------------------------------------------
def policy_iteration(P, nS, nA, gamma=GAMMA, threshold=THRESHOLD):
    policy = np.random.randint(0, nA, size=nS)
    V = np.zeros(nS)

    for cycle in range(1, 200):
        # Evaluate
        V, eval_sweeps = policy_evaluation(P, nS, policy, gamma, threshold)
        # Improve
        policy, stable = policy_improvement(P, nS, nA, V, gamma)
        if stable:
            return policy, V, cycle, eval_sweeps
    return policy, V, 200, eval_sweeps


# -- Evaluate learned policy ---------------------------------------------------
def evaluate_policy(env, policy, episodes=TEST_EPS, max_steps=200):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        for _ in range(max_steps):
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)
    return np.array(total_rewards)


# -- Visualise V(s) as heatmap -------------------------------------------------
def plot_value_grid(V, grid_shape=(5, 5), title="V(s) Heatmap"):
    grid = V.reshape(grid_shape)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, label="V(s)")
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            plt.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                     fontsize=9, color="black")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp07_taxi_dp_value.png'), dpi=150)
    print("Plot saved: exp07_taxi_dp_value.png")
    plt.close()


# -- Main -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Experiment 07: Policy Iteration on Taxi-v3 ===\n")

    env = gym.make("Taxi-v4", render_mode=None)
    nS = int(env.observation_space.n)   # 500 states
    nA = int(env.action_space.n)        # 6 actions

    print(f"States : {nS}")
    print(f"Actions: {nA}")
    print(f"gamma = {GAMMA}, threshold = {THRESHOLD}\n")

    policy, V, cycles, eval_sweeps = policy_iteration(
        env.unwrapped.P, nS, nA, GAMMA, THRESHOLD
    )

    print(f"Converged in {cycles} policy-iteration cycle(s)")
    print(f"  (last evaluation took {eval_sweeps} sweep(s))")
    print(f"Unique actions in policy: {np.unique(policy, return_counts=True)}\n")

    # Test
    rewards = evaluate_policy(env, policy, episodes=TEST_EPS)
    print(f"Average reward over {TEST_EPS} test episodes: {rewards.mean():.2f} "
          f"(std: {rewards.std():.2f})")
    print(f"Min: {rewards.min():.0f}  Max: {rewards.max():.0f}")

    # Plot
    try:
        plot_value_grid(V, grid_shape=(5, 5),
                        title="Taxi-v3 V(s) from Policy Iteration")
    except Exception:
        print("(Value heatmap skipped – reshape mismatch is harmless)")

    env.close()
    print("\nDone.")

