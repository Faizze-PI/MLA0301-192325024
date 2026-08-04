"""
Exp02 - RL Agent for Smart Home Robot Navigation (Tabular Q-learning)
======================================================================
Custom 5x5 grid with obstacles. Goal = charging dock.
Actions : up / down / left / right
Method  : Tabular Q-learning
Plot    : reward-per-episode learning curve + final policy arrows on grid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- grid definition ----------------------------------------------
# 0 = free, 1 = obstacle, 2 = goal (charging dock)
GRID = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 2],
], dtype=int)

ROWS, COLS = GRID.shape
START = (0, 0)
GOAL  = (4, 4)

# -- actions ------------------------------------------------------
ACTIONS     = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAME = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
N_ACTIONS   = 4

# -- hyper-parameters ---------------------------------------------
ALPHA   = 0.1
GAMMA   = 0.95
EPSILON = 0.1
EPISODES = 500
MAX_STEPS = 200

def inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def step(state, action):
    """Deterministic step. Returns (next_state, reward, done)."""
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if not inside(nr, nc) or GRID[nr, nc] == 1:
        nr, nc = r, c                       # bump into wall / obstacle
        reward = -1.0
    elif (nr, nc) == GOAL:
        reward = 10.0
    else:
        reward = -0.1

    done = (nr, nc) == GOAL
    return (nr, nc), reward, done

# -- Q-learning ---------------------------------------------------
def q_learning():
    Q = np.zeros((ROWS, COLS, N_ACTIONS))
    episode_rewards = []

    for ep in range(EPISODES):
        state = START
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            # epsilon-greedy
            if np.random.rand() < EPSILON:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state[0], state[1]]))

            next_state, reward, done = step(state, action)
            total_reward += reward

            # Q-update
            best_next = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * best_next - Q[state[0], state[1], action]
            )

            state = next_state
            if done:
                break

        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {ep+1:4d}  avg reward(last 100)={avg:+.2f}")

    return Q, episode_rewards

# -- extract greedy policy ----------------------------------------
def extract_policy(Q):
    policy = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r, c] == 1:
                policy[r, c] = -1          # obstacle
            elif (r, c) == GOAL:
                policy[r, c] = -2          # goal
            else:
                policy[r, c] = int(np.argmax(Q[r, c]))
    return policy

# -- plotting -----------------------------------------------------
ARROW = {0: "^", 1: "v", 2: "<", 3: ">"}

def plot_reward_curve(rewards, filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\exp02_reward_curve.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Per episode")
    # moving average
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(rewards)), ma, color="orange",
                linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning: Reward per Episode (Smart Home Navigation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")

def plot_policy_grid(policy, filename=r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs\exp02_policy_grid.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_aspect("equal")

    # colour cells
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r, c] == 1:
                color = "#444444"
            elif (r, c) == GOAL:
                color = "#2ecc71"
            elif (r, c) == START:
                color = "#3498db"
            else:
                color = "#ecf0f1"
            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=color, edgecolor="white", linewidth=2)
            ax.add_patch(rect)

            # arrow
            if GRID[r, c] == 0 and (r, c) != GOAL:
                a = policy[r, c]
                if a in ARROW:
                    ax.text(c, r, ARROW[a], ha="center", va="center",
                            fontsize=18, fontweight="bold", color="#2c3e50")

    # labels
    ax.text(GOAL[1], GOAL[0], "⚡", ha="center", va="center", fontsize=22)
    ax.text(START[1], START[0], "🤖", ha="center", va="center", fontsize=22)

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Optimal Policy (Greedy from Q-table)")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  Saved: {filename}")

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Exp 02 : Smart Home Robot Navigation  (Tabular Q-Learning)")
    print("=" * 60)

    print("\n[1] Grid layout (0=free, 1=wall, 2=goal):")
    print(GRID)

    print("\n[2] Training Q-learning agent ...")
    Q, rewards = q_learning()

    print(f"\n[3] Final average reward (last 100 eps): {np.mean(rewards[-100:]):+.2f}")

    print("\n[4] Extracting greedy policy ...")
    policy = extract_policy(Q)

    print("\n[5] Policy map:")
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if GRID[r, c] == 1:
                row_str += "  # "
            elif (r, c) == GOAL:
                row_str += "  G "
            elif (r, c) == START:
                row_str += "  S "
            else:
                row_str += f"  {ARROW[policy[r, c]]} "
        print(row_str)

    print("\n[6] Saving plots ...")
    outputs_dir = r"C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs"
    plot_reward_curve(rewards, os.path.join(outputs_dir, "exp02_reward_curve.png"))
    plot_policy_grid(policy, os.path.join(outputs_dir, "exp02_policy_grid.png"))

    print("\n[Done] Experiment 02 complete.")

if __name__ == "__main__":
    main()

