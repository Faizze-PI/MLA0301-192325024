"""
Experiment 09 - TD(0), SARSA, and Q-Learning: Warehouse Robot
================================================================
Custom 6x6 grid with obstacles. All three algorithms with identical params:
  alpha=0.1, gamma=0.95, epsilon=0.1, episodes=500.

Overlaid reward-per-episode plot (SARSA vs Q-learning).
TD(0) value-error convergence separate plot.

Note: SARSA tends toward safer paths (avoids edges near obstacles);
      Q-learning takes riskier shortcuts (exploits knowledge).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# -- Grid Setup -----------------------------------------------------------------
ROWS, COLS = 6, 6
START = (0, 0)
GOAL  = (5, 5)

# Obstacles form an L-shape barrier
OBSTACLES = {
    (1, 1), (1, 2), (1, 3),
    (2, 3), (3, 3), (3, 4), (3, 5),
}

# Hazard zones – stepping here gives a negative reward (risk)
HAZARDS = {(2, 0), (4, 1), (5, 3)}

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # U, D, L, R
ACTION_NAMES = {0: "^", 1: "v", 2: "<-", 3: "->"}
N_ACTIONS = 4

# -- Hyperparameters ------------------------------------------------------------
ALPHA    = 0.1
GAMMA    = 0.95
EPSILON  = 0.1
EPISODES = 500
MAX_STEPS = 200


# -- Environment ----------------------------------------------------------------
def step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in OBSTACLES:
        next_state = (nr, nc)
    else:
        next_state = (r, c)  # bounced off wall/obstacle

    if next_state == GOAL:
        reward = 10.0
    elif next_state in HAZARDS:
        reward = -2.0
    elif next_state == state:  # bumped into obstacle
        reward = -1.0
    else:
        reward = -0.1  # small living cost

    done = (next_state == GOAL)
    return next_state, reward, done


# -- Epsilon-greedy policy -----------------------------------------------------
def epsilon_greedy(Q, state, epsilon=EPSILON):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(Q[state]))


# -- Algorithm 1: TD(0) Prediction ---------------------------------------------
def td_zero(value_fn=None, episodes=EPISODES):
    """TD(0) prediction for state-value function V(s)."""
    V = defaultdict(float)
    if value_fn:
        V.update(value_fn)

    errors_per_ep = []

    for ep in range(1, episodes + 1):
        state = START
        total_error = 0.0

        for _ in range(MAX_STEPS):
            action = np.random.randint(N_ACTIONS)  # random walk
            next_state, reward, done = step(state, action)

            # TD(0) update
            td_target = reward + (0 if done else GAMMA * V[next_state])
            td_error  = td_target - V[state]
            V[state] += ALPHA * td_error
            total_error += abs(td_error)

            state = next_state
            if done:
                break

        errors_per_ep.append(total_error)

        if ep % 100 == 0:
            print(f"  TD(0)  ep {ep:4d}  |  avg |d|={np.mean(errors_per_ep[-100:]):.4f}")

    return dict(V), errors_per_ep


# -- Algorithm 2: SARSA --------------------------------------------------------
def sarsa(episodes=EPISODES):
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    rewards_per_ep = []

    for ep in range(1, episodes + 1):
        state = START
        action = epsilon_greedy(Q, state)
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            next_state, reward, done = step(state, action)
            next_action = epsilon_greedy(Q, next_state)

            # SARSA update (on-policy)
            td_target = reward + (0 if done else GAMMA * Q[next_state][next_action])
            Q[state][action] += ALPHA * (td_target - Q[state][action])

            total_reward += reward
            state = next_state
            action = next_action
            if done:
                break

        rewards_per_ep.append(total_reward)

        if ep % 100 == 0:
            avg = np.mean(rewards_per_ep[-100:])
            print(f"  SARSA  ep {ep:4d}  |  avg reward={avg:.3f}")

    return dict(Q), rewards_per_ep


# -- Algorithm 3: Q-Learning ---------------------------------------------------
def q_learning(episodes=EPISODES):
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    rewards_per_ep = []

    for ep in range(1, episodes + 1):
        state = START
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = epsilon_greedy(Q, state)
            next_state, reward, done = step(state, action)

            # Q-Learning update (off-policy)
            td_target = reward + (0 if done else GAMMA * np.max(Q[next_state]))
            Q[state][action] += ALPHA * (td_target - Q[state][action])

            total_reward += reward
            state = next_state
            if done:
                break

        rewards_per_ep.append(total_reward)

        if ep % 100 == 0:
            avg = np.mean(rewards_per_ep[-100:])
            print(f"  Q-Learn ep {ep:4d}  |  avg reward={avg:.3f}")

    return dict(Q), rewards_per_ep


# -- Extract greedy policy from Q ----------------------------------------------
def extract_policy(Q):
    policy = {}
    for s in Q:
        policy[s] = int(np.argmax(Q[s]))
    return policy


# -- Plotting -------------------------------------------------------------------
def plot_reward_comparison(rewards_sarsa, rewards_qlearn):
    window = 20
    ep = range(1, EPISODES + 1)

    sarsa_smooth   = np.convolve(rewards_sarsa,   np.ones(window)/window, mode="valid")
    qlearn_smooth  = np.convolve(rewards_qlearn, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(ep, rewards_sarsa,  alpha=0.2, color="blue")
    plt.plot(ep, rewards_qlearn, alpha=0.2, color="orange")
    plt.plot(range(window, EPISODES + 1), sarsa_smooth,
             color="blue", linewidth=2, label="SARSA (smoothed)")
    plt.plot(range(window, EPISODES + 1), qlearn_smooth,
             color="orange", linewidth=2, label="Q-Learning (smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA vs Q-Learning – Warehouse Robot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp09_td_reward_comparison.png'), dpi=150)
    print("\nPlot saved: exp09_td_reward_comparison.png")
    plt.close()


def plot_td_error_convergence(errors):
    window = 20
    smoothed = np.convolve(errors, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(errors) + 1), errors, alpha=0.25, label="Per-episode |d|")
    plt.plot(range(window, len(errors) + 1), smoothed,
             color="red", linewidth=2, label=f"Smoothed ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Absolute TD Error")
    plt.title("TD(0) Value-Error Convergence – Warehouse Robot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs', r'exp09_td_error_convergence.png'), dpi=150)
    print("Plot saved: exp09_td_error_convergence.png")
    plt.close()


def plot_grid_with_policy(policy, title, filename):
    grid = np.full((ROWS, COLS), " ", dtype=object)
    for (r, c) in OBSTACLES:
        grid[r][c] = "#"
    for (r, c) in HAZARDS:
        grid[r][c] = "H"
    grid[GOAL[0]][GOAL[1]] = "G"
    grid[START[0]][START[1]] = "S"

    for (r, c), a in policy.items():
        if (r, c) not in OBSTACLES and (r, c) != GOAL and (r, c) != START:
            grid[r][c] = ACTION_NAMES.get(a, "?")

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r][c]
            if val == "#":
                color = "black"
            elif val == "H":
                color = "red"
            elif val == "G":
                color = "green"
            elif val == "S":
                color = "blue"
            else:
                color = "black"
            ax.text(c, r, val, ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True, linewidth=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved: {filename}")
    plt.close()


# -- Main -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Experiment 09: TD(0), SARSA, Q-Learning – Warehouse Robot ===\n")
    print(f"Grid: {ROWS}x{COLS}  |  Obstacles: {len(OBSTACLES)}  "
          f"|  Hazards: {len(HAZARDS)}")
    print(f"Start={START}  Goal={GOAL}")
    print(f"alpha={ALPHA}  gamma={GAMMA}  epsilon={EPSILON}  episodes={EPISODES}\n")

    # TD(0)
    print("[1] TD(0) Prediction...")
    V_td, errors = td_zero(episodes=EPISODES)
    print()

    # SARSA
    print("[2] SARSA...")
    Q_sarsa, rewards_sarsa = sarsa(episodes=EPISODES)
    print()

    # Q-Learning
    print("[3] Q-Learning...")
    Q_qlearn, rewards_qlearn = q_learning(episodes=EPISODES)
    print()

    # Policies
    policy_sarsa  = extract_policy(Q_sarsa)
    policy_qlearn = extract_policy(Q_qlearn)

    # Plots
    print("[4] Generating plots...")
    plot_reward_comparison(rewards_sarsa, rewards_qlearn)
    plot_td_error_convergence(errors)
    plot_grid_with_policy(policy_sarsa,
                          "SARSA Policy (safer near obstacles)",
                          os.path.join(r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs", "exp09_sarsa_policy.png"))
    plot_grid_with_policy(policy_qlearn,
                          "Q-Learning Policy (riskier, shorter paths)",
                          os.path.join(r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs", "exp09_qlearn_policy.png"))

    # Summary
    print(f"\n--- Summary ---")
    print(f"SARSA   final avg reward (last 100): {np.mean(rewards_sarsa[-100:]):.3f}")
    print(f"Q-Learn final avg reward (last 100): {np.mean(rewards_qlearn[-100:]):.3f}")
    print(f"TD(0)   final avg |d|    (last 100): {np.mean(errors[-100:]):.4f}")
    print("\nNote: SARSA tends toward safer paths avoiding hazard zones,")
    print("      while Q-Learning exploits optimal shortcuts near edges.")
    print("Done.")

