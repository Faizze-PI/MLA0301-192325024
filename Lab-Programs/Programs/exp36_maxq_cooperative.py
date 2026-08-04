"""
Experiment 36: MAXQ Framework for Cooperative Hierarchical Multi-Agent Task
=============================================================================
Two agents cooperatively assemble an item:
  - Agent A fetches Part 1 and delivers it to the assembly point.
  - Agent B fetches Part 2 and delivers it to the assembly point.
  - Both agents must arrive at the assembly point for task completion.
MAXQ value function decomposition per agent with shared completion reward.
2-level hierarchy: primitive actions (move, pick, drop) and composite fetch task.
Gamma = 0.95.

Outputs:
  - Task-completion-time learning curve over episodes
  - Trace of a successful cooperative episode (state grid visualization)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import random

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
GRID_SIZE = 6
ASSEMBLY_POINT = (5, 5)
PART1_LOCATION = (0, 0)
PART2_LOCATION = (5, 0)

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
PRIMITIVE_ACTIONS = ["move", "pick", "drop"]
ALL_ACTIONS = ACTION_NAMES + ["pick", "drop"]

MAX_EPISODES = 800
GAMMA = 0.95
ALPHA_PRIMITIVE = 0.15
ALPHA_COMPOSITE = 0.10
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995


def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


class CooperativeEnv:
    """Multi-agent cooperative assembly environment."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos_a = list(PART1_LOCATION)
        self.pos_b = list(PART2_LOCATION)
        self.has_part_a = False
        self.has_part_b = False
        self.part1_delivered = False
        self.part2_delivered = False
        self.done = False
        self.steps = 0
        self.trace = []
        self._record()
        return self._state()

    def _state(self):
        return (
            tuple(self.pos_a),
            tuple(self.pos_b),
            self.has_part_a,
            self.has_part_b,
            self.part1_delivered,
            self.part2_delivered,
        )

    def _record(self):
        self.trace.append({
            "pos_a": tuple(self.pos_a),
            "pos_b": tuple(self.pos_b),
            "has_a": self.has_part_a,
            "has_b": self.has_part_b,
            "del_a": self.part1_delivered,
            "del_b": self.part2_delivered,
        })

    def step_agent(self, agent, action_idx):
        """Execute one action for one agent. Returns (next_state, reward, done, info)."""
        if self.done:
            return self._state(), 0.0, True, {}

        self.steps += 1
        reward = -0.1  # small step penalty
        info = {}

        if agent == "A":
            pos = self.pos_a
            has = "has_part_a"
        else:
            pos = self.pos_b
            has = "has_part_b"

        if action_idx < 4:
            # Primitive move
            dr, dc = ACTIONS[action_idx]
            nr, nc = pos[0] + dr, pos[1] + dc
            if in_bounds(nr, nc):
                pos[0], pos[1] = nr, nc
            else:
                reward -= 1.0  # wall penalty
        elif action_idx == 4:
            # Pick
            if agent == "A" and tuple(pos) == PART1_LOCATION and not self.has_part_a:
                self.has_part_a = True
                reward += 5.0
                info["event"] = "A_picked_part1"
            elif agent == "B" and tuple(pos) == PART2_LOCATION and not self.has_part_b:
                self.has_part_b = True
                reward += 5.0
                info["event"] = "B_picked_part2"
            else:
                reward -= 0.5
        elif action_idx == 5:
            # Drop at assembly point
            if tuple(pos) == ASSEMBLY_POINT:
                if agent == "A" and self.has_part_a and not self.part1_delivered:
                    self.part1_delivered = True
                    self.has_part_a = False
                    reward += 10.0
                    info["event"] = "A_delivered_part1"
                elif agent == "B" and self.has_part_b and not self.part2_delivered:
                    self.part2_delivered = True
                    self.has_part_b = False
                    reward += 10.0
                    info["event"] = "B_delivered_part2"
                else:
                    reward -= 0.5
            else:
                reward -= 0.5

        # Check completion
        if self.part1_delivered and self.part2_delivered and not self.done:
            self.done = True
            reward += 50.0  # cooperative completion bonus
            info["event"] = "task_complete"

        if self.steps >= 200 and not self.done:
            self.done = True
            reward -= 20.0

        self._record()
        return self._state(), reward, self.done, info


# ---------------------------------------------------------------------------
# MAXQ Agent (per-agent Q-value decomposition)
# ---------------------------------------------------------------------------
class MAXQAgent:
    """MAXQ decomposition for a single agent in the cooperative task."""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.q_primitive = {}   # Q for primitive actions (move/pick/drop)
        self.q_composite = {}   # V for composite fetch task completion
        self.epsilon = EPSILON_START
        self.total_steps = 0

    def _state_key(self, state):
        return state

    def _primitive_q(self, s, a):
        key = (s, a)
        if key not in self.q_primitive:
            self.q_primitive[key] = 0.0
        return self.q_primitive[key]

    def _composite_v(self, s):
        key = s
        if key not in self.q_composite:
            self.q_composite[key] = 0.0
        return self.q_composite[key]

    def _epsilon_greedy_primitive(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_vals = [self._primitive_q(state, a) for a in available_actions]
        max_q = max(q_vals)
        best = [a for a, q in zip(available_actions, q_vals) if q == max_q]
        return random.choice(best)

    def choose_composite_action(self, env_state):
        """Decide whether to pursue composite fetch or use primitive."""
        # Composite: continue fetch subtask; primitive: pick/drop/goal actions
        self.total_steps += 1
        return "composite"

    def act(self, env, env_state):
        """Choose and execute a primitive action within composite fetch."""
        pos_a = env_state[0] if self.agent_id == "A" else env_state[1]
        has_part = env_state[2] if self.agent_id == "A" else env_state[3]
        delivered = env_state[4] if self.agent_id == "A" else env_state[5]

        if delivered:
            # Stay at assembly point
            action_idx = 0  # any move is fine, task done for this agent
            return action_idx

        available = list(range(4))  # moves always available

        if not has_part:
            # Goal: pick part at specific location
            target = PART1_LOCATION if self.agent_id == "A" else PART2_LOCATION
            if tuple(pos_a) == target:
                available.append(4)  # pick
        else:
            # Goal: deliver to assembly point
            if tuple(pos_a) == ASSEMBLY_POINT:
                available.append(5)  # drop

        action_idx = self._epsilon_greedy_primitive(env_state, available)
        return action_idx

    def update(self, state, action, reward, next_state, done):
        """Update primitive Q-values (MAXQ-style: V-composite subtracted)."""
        v_next = 0.0 if done else self._composite_v(next_state)
        v_current = self._composite_v(state)

        target = reward + GAMMA * v_next - v_current
        old_q = self._primitive_q(state, action)
        self.q_primitive[(state, action)] = old_q + ALPHA_PRIMITIVE * (target - old_q)

    def update_composite(self, state, reward, next_state, done):
        """Update composite V-function."""
        v_next = 0.0 if done else self._composite_v(next_state)
        target = reward + GAMMA * v_next
        old_v = self._composite_v(state)
        self.q_composite[state] = old_v + ALPHA_COMPOSITE * (target - old_v)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def train_cooperative():
    """Train two MAXQ agents cooperatively."""
    env = CooperativeEnv()
    agent_a = MAXQAgent("A")
    agent_b = MAXQAgent("B")

    completion_times = []
    successes = 0

    print("Training cooperative MAXQ agents...")
    print(f"{'Episode':>8} | {'Steps':>6} | {'Reward':>8} | {'Epsilon':>7} | {'Success':>7}")
    print("-" * 55)

    for ep in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < 200:
            for agent_label, agent_obj in [("A", agent_a), ("B", agent_b)]:
                if done:
                    break
                action = agent_obj.act(env, state)
                next_state, reward, done, info = env.step_agent(agent_label, action)
                ep_reward += reward

                agent_obj.update(state, action, reward, next_state, done)
                agent_obj.update_composite(state, reward, next_state, done)

                state = next_state
                step_count += 1

        agent_a.decay_epsilon()
        agent_b.decay_epsilon()

        completed = env.part1_delivered and env.part2_delivered
        completion_times.append(step_count if completed else 200)
        if completed:
            successes += 1

        if ep % 100 == 0 or ep == MAX_EPISODES - 1:
            recent = completion_times[-50:]
            avg = np.mean(recent)
            print(f"{ep:8d} | {step_count:6d} | {ep_reward:8.1f} | "
                  f"{agent_a.epsilon:7.3f} | {'YES' if completed else 'NO':>7}")

    print(f"\nTotal successes: {successes}/{MAX_EPISODES} "
          f"({100*successes/MAX_EPISODES:.1f}%)")

    return env, agent_a, agent_b, completion_times


def plot_learning_curve(completion_times):
    """Plot task-completion-time learning curve."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(completion_times, alpha=0.3, color="steelblue", linewidth=0.5, label="Per episode")

    # Smoothed curve
    window = 30
    if len(completion_times) >= window:
        smoothed = np.convolve(completion_times, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(completion_times)), smoothed,
                color="darkblue", linewidth=2, label=f"Smoothed (w={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps to Completion", fontsize=12)
    ax.set_title("MAXQ Cooperative Task: Completion Time Learning Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 210)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "exp36_learning_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def trace_episode(env, agent_a, agent_b):
    """Run one successful episode and visualize the trace."""
    state = env.reset()
    done = False
    step = 0
    trace_data = []

    while not done and step < 200:
        for label, agent in [("A", agent_a), ("B", agent_b)]:
            if done:
                break
            action = agent.act(env, state)
            next_state, reward, done, info = env.step_agent(label, action)
            trace_data.append({
                "step": step,
                "agent": label,
                "action": ACTION_NAMES[action] if action < 4 else ("pick" if action == 4 else "drop"),
                "pos": env.pos_a if label == "A" else env.pos_b,
                "has_part": env.has_part_a if label == "A" else env.has_part_b,
                "delivered": env.part1_delivered if label == "A" else env.part2_delivered,
            })
            state = next_state
            step += 1

    # Visualize trace on grid
    fig, axes = plt.subplots(1, min(len(trace_data), 8), figsize=(4 * min(len(trace_data), 8), 4))
    if min(len(trace_data), 8) == 1:
        axes = [axes]

    sample_indices = np.linspace(0, len(trace_data) - 1, min(len(trace_data), 8), dtype=int)

    for idx, si in enumerate(sample_indices):
        ax = axes[idx]
        td = trace_data[si]

        # Draw grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                ax.add_patch(plt.Rectangle((c, GRID_SIZE - 1 - r), 1, 1,
                                           fill=False, edgecolor="gray", linewidth=0.5))

        # Assembly point
        ax.add_patch(plt.Rectangle((ASSEMBLY_POINT[1], GRID_SIZE - 1 - ASSEMBLY_POINT[0]),
                                   1, 1, facecolor="lightgreen", edgecolor="green", linewidth=1.5))
        ax.text(ASSEMBLY_POINT[1] + 0.5, GRID_SIZE - 1 - ASSEMBLY_POINT[0] + 0.5,
                "ASM", ha="center", va="center", fontsize=7, fontweight="bold")

        # Part locations
        if not td["delivered"]:
            if not td["has_part"]:
                for part_loc, name in [(PART1_LOCATION, "P1"), (PART2_LOCATION, "P2")]:
                    ax.add_patch(plt.Circle((part_loc[1] + 0.5, GRID_SIZE - 1 - part_loc[0] + 0.5),
                                            0.3, facecolor="orange", edgecolor="darkorange"))
                    ax.text(part_loc[1] + 0.5, GRID_SIZE - 1 - part_loc[0] + 0.5,
                            name, ha="center", va="center", fontsize=7, fontweight="bold")

        # Agent A
        pos_a = td["pos"]
        marker_a = "^" if not td["has_part"] else "s"
        color_a = "blue" if td["agent"] != "A" else "red"
        if td["agent"] == "A":
            color_a = "red"
        ax.plot(pos_a[1] + 0.5, GRID_SIZE - 1 - pos_a[0] + 0.5, marker=marker_a,
                color="blue", markersize=12, markeredgecolor="black", markeredgewidth=1)
        ax.text(pos_a[1] + 0.5, GRID_SIZE - 1 - pos_a[0] + 0.85, "A",
                ha="center", va="center", fontsize=8, color="blue", fontweight="bold")

        # Agent B
        pos_b = td["pos"]
        marker_b = "^" if not td["has_part"] else "s"
        if td["agent"] == "B":
            color_b = "red"
        else:
            color_b = "purple"
        ax.plot(pos_b[1] + 0.5, GRID_SIZE - 1 - pos_b[0] + 0.5, marker=marker_b,
                color="purple", markersize=12, markeredgecolor="black", markeredgewidth=1)
        ax.text(pos_b[1] + 0.5, GRID_SIZE - 1 - pos_b[0] + 0.15, "B",
                ha="center", va="center", fontsize=8, color="purple", fontweight="bold")

        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect("equal")
        ax.set_title(f"Step {td['step']}\n{td['agent']}:{td['action']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Trace of Cooperative Episode (MAXQ)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "exp36_episode_trace.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    return trace_data


def main():
    print("=" * 60)
    print("EXPERIMENT 36: MAXQ Cooperative Hierarchical Multi-Agent Task")
    print("=" * 60)
    print(f"\nEnvironment: {GRID_SIZE}x{GRID_SIZE} grid")
    print(f"Agent A starts at {PART1_LOCATION} (fetch Part 1)")
    print(f"Agent B starts at {PART2_LOCATION} (fetch Part 2)")
    print(f"Assembly point: {ASSEMBLY_POINT}")
    print(f"Gamma={GAMMA}, Episodes={MAX_EPISODES}\n")

    env, agent_a, agent_b, completion_times = train_cooperative()

    plot_learning_curve(completion_times)

    print("\n--- Episode Trace ---")
    trace = trace_episode(env, agent_a, agent_b)
    for t in trace[:20]:
        print(f"  Step {t['step']:3d}: Agent {t['agent']} -> {t['action']:5s} "
              f"pos={t['pos']} has_part={t['has_part']} delivered={t['delivered']}")
    if len(trace) > 20:
        print(f"  ... ({len(trace)} total steps)")

    print("\nExperiment 36 complete.")


if __name__ == "__main__":
    main()
