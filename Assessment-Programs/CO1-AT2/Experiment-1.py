"""
Experiment 1: Grid World Navigation with Obstacles
------------------------------------------------------------------------
Aim: Develop a Grid World navigation RL agent that reaches the goal while
avoiding obstacles. Define state space, action space, reward function, and
constraints.

State space:  (row, col) in a 5x5 grid  → 25 discrete states
Actions:      Up, Down, Left, Right       → 4 discrete actions
Rewards:      +100 at goal, -100 at obstacle, -1 per step
Constraints:  Obstacles block movement (agent gets penalty and stays put)
Algorithm:    Tabular Q-learning
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

GRID_ROWS, GRID_COLS = 5, 5
GOAL = (4, 4)
OBSTACLES = {(1, 1), (1, 3), (2, 3), (3, 1)}
START = (0, 0)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
N_ACTIONS = len(ACTIONS)

ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
N_EPISODES = 200
MAX_STEPS = 100


def step(state, action_idx):
    dr, dc = ACTIONS[action_idx]
    r, c = state[0] + dr, state[1] + dc
    if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
        next_state = (r, c)
    else:
        next_state = state

    if next_state in OBSTACLES:
        return state, -100, False
    if next_state == GOAL:
        return next_state, 100, True
    return next_state, -1, False


def train_q_learning():
    Q = np.zeros((GRID_ROWS, GRID_COLS, N_ACTIONS))
    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(N_EPISODES):
        state = START
        total_reward = 0
        for _ in range(MAX_STEPS):
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state[0], state[1]]))

            next_state, reward, done = step(state, action)
            best_next = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * best_next - Q[state[0], state[1], action]
            )
            state = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward (last 10): {avg:.1f} | Epsilon: {epsilon:.3f}")

    return Q, episode_rewards


def extract_policy(Q):
    policy = np.zeros((GRID_ROWS, GRID_COLS), dtype=object)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in OBSTACLES:
                policy[r, c] = "X"
            elif (r, c) == GOAL:
                policy[r, c] = "G"
            else:
                policy[r, c] = ACTION_NAMES[int(np.argmax(Q[r, c]))]
    return policy


def plot_grid(Q, episode_rewards):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    grid = np.zeros((GRID_ROWS, GRID_COLS))
    for r, c in OBSTACLES:
        grid[r, c] = -1
    grid[GOAL[0], GOAL[1]] = 1
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
    policy = extract_policy(Q)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ax.text(c, r, policy[r, c], ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Learned Policy (X=obstacle, G=goal)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], label="Obstacle / Free / Goal")

    ax = axes[1]
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), moving_avg, linewidth=2, label=f"{window}-ep moving avg")
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment1_grid_world.png", dpi=150)
    print("\nPlot saved as 'experiment1_grid_world.png'")


if __name__ == "__main__":
    print("Experiment 1: Grid World Navigation with Obstacles\n")
    print("State space: (row, col) in 5x5 grid = 25 states")
    print("Actions: Up, Down, Left, Right = 4 actions")
    print("Rewards: +100 goal, -100 obstacle, -1 per step")
    print("Constraint: Obstacles block movement\n")
    print("Training Q-learning agent...\n")

    Q, rewards = train_q_learning()
    policy = extract_policy(Q)

    print("\nLearned Policy:")
    for r in range(GRID_ROWS):
        row_str = " | ".join(f"{policy[r,c]:^5}" for c in range(GRID_COLS))
        print(f"  {row_str}")

    plot_grid(Q, rewards)
    print("\nExperiment 1 completed successfully!")
