"""
Exp 22 – Q-Learning for Grid-Based Pac-Man Game
=================================================
An 8x8 grid with walls, <=5 food pellets (+10 each, removed on collection),
and a ghost that moves randomly each step (collision = −50, episode ends).

State = (agent_row, agent_col, frozen_food_bitmap)
  – food_bitmap is a 5-bit integer encoding which of the 5 pellet positions remain.
  – total states ~ 64 x 2⁵ = 2 048

Agent learns via Q-learning (alpha=0.1, gamma=0.9, epsilon decays 1.0->0.05 over 2 000 episodes).
We track per-episode reward and a rolling win-rate (agent collects all food before ghost).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import os, itertools

# -- Grid configuration ------------------------------------------------
ROWS, COLS = 8, 8
WALLS = {(0, 3), (1, 3), (2, 3), (3, 3), (5, 5), (5, 6), (6, 5)}
AGENT_START = (0, 0)
GHOST_START = (7, 7)
FOOD_POSITIONS = [(1, 1), (2, 6), (4, 2), (6, 7), (7, 1)]  # <=5 pellets
FOOD_REWARD = 10
GHOST_PENALTY = -50
STEP_PENALTY = -1
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
ACTION_NAMES = ["R", "L", "D", "U"]

# -- Q-learning hyperparameters ----------------------------------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPISODES = 2000
MAX_STEPS = 200
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / (EPISODES * 0.8)


# -- Environment helpers -----------------------------------------------
def food_bitmap(food_set):
    """Return a 5-bit integer bitmap; bit i set ⇔ FOOD_POSITIONS[i] still present."""
    bm = 0
    for i, fp in enumerate(FOOD_POSITIONS):
        if fp in food_set:
            bm |= (1 << i)
    return bm


def make_state(pos, food_set):
    return (pos[0], pos[1], food_bitmap(food_set))


def step(agent_pos, food_set, ghost_pos, action):
    """Execute one timestep. Returns (new_agent_pos, new_food_set, reward, done)."""
    dr, dc = ACTIONS[action]
    nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
    if (nr, nc) in WALLS or not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = agent_pos  # stay

    new_agent = (nr, nc)
    reward = STEP_PENALTY
    new_food = set(food_set)
    if new_agent in new_food:
        new_food.remove(new_agent)
        reward += FOOD_REWARD

    # ghost random walk
    g_choices = []
    for gdr, gdc in ACTIONS:
        gr, gc = ghost_pos[0] + gdr, ghost_pos[1] + gdc
        if 0 <= gr < ROWS and 0 <= gc < COLS and (gr, gc) not in WALLS:
            g_choices.append((gr, gc))
    ghost_pos = g_choices[np.random.randint(len(g_choices))] if g_choices else ghost_pos

    done = False
    if new_agent == ghost_pos:
        reward += GHOST_PENALTY
        done = True
    if len(new_food) == 0:
        done = True  # win

    return new_agent, new_food, ghost_pos, reward, done


# -- Q-table -----------------------------------------------------------
Q = defaultdict(lambda: np.zeros(len(ACTIONS)))


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTIONS))
    return int(np.argmax(Q[state]))


def train():
    epsilon = EPSILON_START
    episode_rewards = []
    wins = 0
    win_history = []

    for ep in range(EPISODES):
        agent = AGENT_START
        ghost = GHOST_START
        food = set(FOOD_POSITIONS)
        total_reward = 0
        done = False
        t = 0

        while not done and t < MAX_STEPS:
            s = make_state(agent, food)
            a = choose_action(s, epsilon)
            agent, food, ghost, r, done = step(agent, food, ghost, a)
            s2 = make_state(agent, food)
            best_next = np.max(Q[s2])
            Q[s][a] += ALPHA * (r + GAMMA * best_next * (1 - done) - Q[s][a])
            total_reward += r
            t += 1

        # win = agent collected all food without ghost catch
        if len(food) == 0 and agent != ghost:
            wins += 1

        epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)
        episode_rewards.append(total_reward)
        win_history.append(wins / (ep + 1))

        if (ep + 1) % 200 == 0:
            avg = np.mean(episode_rewards[-200:])
            print(f"  Episode {ep+1:5d} | avg reward {avg:7.1f} | win-rate {win_history[-1]:.2%} | epsilon={epsilon:.3f}")

    return episode_rewards, win_history


def plot_results(rewards, win_rates):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(rewards, alpha=0.3, color="steelblue", label="Per-episode")
    window = 100
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(rewards)), smoothed, color="navy", label=f"{window}-ep MA")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Pac-Man Q-Learning – Reward Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(win_rates, color="green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Win-Rate")
    ax2.set_title("Win-Rate Over Training")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "exp22_pacman_results.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved -> {path}")
    plt.close()


def demo_greedy(n=3):
    print("\n  -- Greedy demo (3 episodes) --")
    for d in range(n):
        agent, ghost = AGENT_START, GHOST_START
        food = set(FOOD_POSITIONS)
        total_r = 0
        for t in range(MAX_STEPS):
            s = make_state(agent, food)
            a = int(np.argmax(Q[s]))
            agent, food, ghost, r, done = step(agent, food, ghost, a)
            total_r += r
            if done:
                break
        status = "WIN" if len(food) == 0 else "CAUGHT"
        print(f"    Demo {d+1}: steps={t+1} reward={total_r:.0f} [{status}]")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("Exp 22 – Pac-Man Q-Learning")
    print("=" * 60)
    rewards, win_rates = train()
    plot_results(rewards, win_rates)
    demo_greedy()
    print("\nDone.")

