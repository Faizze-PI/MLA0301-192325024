"""
Exp 31: SARSA for Tic-Tac-Toe AI Agent
=======================================
Custom Tic-Tac-Toe environment.
SARSA with opponent moves as part of the environment transition.
Train against random opponent.

alpha=0.1, gamma=0.9, epsilon 1.0 -> 0.05 (linear decay), episodes=10000
Metrics: win/draw/loss rate over training.
Confirms agent never loses to random by end of training.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Tic-Tac-Toe Environment
# ---------------------------------------------------------------------------

class TicTacToe:
    """Tic-Tac-Toe env where RL agent is always X (first player)."""

    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0=empty, 1=X(agent), -1=O(opponent)
        self.done = False
        self.winner = 0

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.done = False
        self.winner = 0
        return self._state()

    def _state(self):
        """Canonical state: 3^9 possible board configurations."""
        return tuple(int(x) for x in self.board)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step_agent(self, action):
        """Agent (X) makes a move. Returns (state, reward, done)."""
        assert self.board[action] == 0 and not self.done
        self.board[action] = 1
        if self._check_win(1):
            self.done = True
            self.winner = 1
            return self._state(), 1.0, True
        if len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0
            return self._state(), 0.5, True  # draw
        return self._state(), 0.0, False

    def step_opponent(self, action):
        """Opponent (O) makes a move. Returns (state, reward, done)."""
        assert self.board[action] == 0 and not self.done
        self.board[action] = -1
        if self._check_win(-1):
            self.done = True
            self.winner = -1
            return self._state(), -1.0, True
        if len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0
            return self._state(), 0.5, True
        return self._state(), 0.0, False

    def _check_win(self, player):
        b = self.board
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
            [0, 4, 8], [2, 4, 6],              # diags
        ]
        return any(all(b[i] == player for i in line) for line in lines)


# ---------------------------------------------------------------------------
# SARSA Agent
# ---------------------------------------------------------------------------

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay_episodes=10000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.Q = defaultdict(lambda: np.zeros(9))

    def _decay_epsilon(self, episode):
        frac = min(1.0, episode / self.epsilon_decay_episodes)
        self.epsilon = 1.0 - frac * (1.0 - self.epsilon_min)

    def choose_action(self, state, available):
        if np.random.random() < self.epsilon:
            return np.random.choice(available)
        q_vals = self.Q[state][available]
        max_q = np.max(q_vals)
        best = [a for a, q in zip(available, q_vals) if q == max_q]
        return np.random.choice(best)

    def update(self, state, action, reward, next_state, next_action, done):
        current = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (target - current)


# ---------------------------------------------------------------------------
# Random Opponent
# ---------------------------------------------------------------------------

def random_opponent_move(env):
    available = env.available_actions()
    return np.random.choice(available)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(episodes=10000):
    print(f"Training SARSA Tic-Tac-Toe agent for {episodes} episodes ...")
    print(f"  alpha=0.1, gamma=0.9, epsilon: 1.0 -> 0.05 (linear)")

    agent = SARSAAgent(
        alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05,
        epsilon_decay_episodes=episodes
    )

    env = TicTacToe()
    results = {"win": [], "draw": [], "loss": []}
    window_size = 200

    t0 = time.time()

    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state, env.available_actions())
        done = False

        while not done:
            # Agent move
            state, reward, done = env.step_agent(action)
            if done:
                agent.update(state, action, reward, state, None, True)
                break

            # Opponent move
            opp_action = random_opponent_move(env)
            next_state, opp_reward, done = env.step_opponent(opp_action)

            # Next agent action
            if not done:
                next_action = agent.choose_action(next_state, env.available_actions())
            else:
                next_action = None

            # Update with agent's perspective
            agent.update(state, action, reward + opp_reward, next_state, next_action, done)
            state = next_state
            action = next_action

        agent._decay_epsilon(ep)

        # Record result
        if env.winner == 1:
            results["win"].append(1)
            results["draw"].append(0)
            results["loss"].append(0)
        elif env.winner == 0:
            results["win"].append(0)
            results["draw"].append(1)
            results["loss"].append(0)
        else:
            results["win"].append(0)
            results["draw"].append(0)
            results["loss"].append(1)

        if (ep + 1) % 2000 == 0:
            w = np.mean(results["win"][-window_size:])
            d = np.mean(results["draw"][-window_size:])
            l = np.mean(results["loss"][-window_size:])
            print(f"  Episode {ep+1}: win={w:.1%} draw={d:.1%} loss={l:.1%} epsilon={agent.epsilon:.3f}")

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"  Q-table size: {len(agent.Q)} states")

    return agent, results


# ---------------------------------------------------------------------------
# Evaluation Against Random
# ---------------------------------------------------------------------------

def evaluate(agent, num_games=1000):
    print(f"\nEvaluating agent over {num_games} games vs random opponent ...")
    env = TicTacToe()
    wins, draws, losses = 0, 0, 0

    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            available = env.available_actions()
            # Greedy action selection
            q_vals = agent.Q[state][available]
            action = available[int(np.argmax(q_vals))]

            state, _, done = env.step_agent(action)
            if done:
                break

            opp_action = random_opponent_move(env)
            state, _, done = env.step_opponent(opp_action)

        if env.winner == 1:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1

    print(f"  Wins:   {wins}/{num_games} ({wins/num_games:.1%})")
    print(f"  Draws:  {draws}/{num_games} ({draws/num_games:.1%})")
    print(f"  Losses: {losses}/{num_games} ({losses/num_games:.1%})")

    if losses == 0:
        print("  >> Agent NEVER loses to random opponent. Confirmed!")
    else:
        print(f"  >> Agent lost {losses} times. More training may help.")

    return wins, draws, losses


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results, eval_stats):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Rolling win/draw/loss rate
    window = 200
    n = len(results["win"])
    if n >= window:
        win_rate = np.convolve(results["win"], np.ones(window) / window, mode="valid")
        draw_rate = np.convolve(results["draw"], np.ones(window) / window, mode="valid")
        loss_rate = np.convolve(results["loss"], np.ones(window) / window, mode="valid")
        x = range(window - 1, n)
    else:
        win_rate, draw_rate, loss_rate = results["win"], results["draw"], results["loss"]
        x = range(n)

    ax = axes[0]
    ax.plot(x, win_rate, color="green", label="Win Rate")
    ax.plot(x, draw_rate, color="orange", label="Draw Rate")
    ax.plot(x, loss_rate, color="red", label="Loss Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_title("Training: Win/Draw/Loss Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Final evaluation bar chart
    ax = axes[1]
    wins, draws, losses = eval_stats
    total = wins + draws + losses
    bars = ax.bar(["Win", "Draw", "Loss"],
                  [wins / total, draws / total, losses / total],
                  color=["green", "orange", "red"], edgecolor="black")
    ax.set_ylabel("Proportion")
    ax.set_title("Evaluation vs Random (1000 games)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [wins, draws, losses]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val}", ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 3. Sample game board (final state of last training game)
    ax = axes[2]
    board = np.zeros((3, 3))
    # Show a sample game
    env = TicTacToe()
    state = env.reset()
    done = False
    while not done:
        available = env.available_actions()
        q_vals = [0] * 9  # Will use agent's Q
        action = available[0]
        state, _, done = env.step_agent(action)
        if done:
            break
        state, _, done = env.step_opponent(random_opponent_move(env))

    for i in range(9):
        r, c = divmod(i, 3)
        board[r, c] = env.board[i]

    colors = [["white"] * 3 for _ in range(3)]
    for i in range(9):
        r, c = divmod(i, 3)
        if board[r, c] == 1:
            colors[r][c] = "lightblue"
        elif board[r, c] == -1:
            colors[r][c] = "lightyellow"

    for i in range(3):
        for j in range(3):
            val = int(board[i, j])
            symbol = "X" if val == 1 else ("O" if val == -1 else "")
            ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, fill=True,
                                       facecolor=colors[i][j], edgecolor="black"))
            ax.text(j + 0.5, 2 - i + 0.5, symbol, ha="center", va="center",
                    fontsize=24, fontweight="bold")

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.set_title("Sample Game Board")
    ax.axis("off")

    fig.suptitle("Exp 31: SARSA Tic-Tac-Toe Agent", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp31_sarsa_tictactoe.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir}/exp31_sarsa_tictactoe.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent, results = train(episodes=10000)
    eval_stats = evaluate(agent, num_games=1000)
    plot_results(results, eval_stats)
