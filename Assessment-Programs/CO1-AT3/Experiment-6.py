"""
Experiment 6: Traffic Signal with Emergency Vehicle Priority
------------------------------------------------------------------------
Aim: Implement an RL agent controlling traffic signals ensuring emergency
vehicles receive immediate priority. Explain exploration, exploitation,
and policy learning modified for safety constraints.

States:      (queue_NS, queue_EW, emergency)  → 3x3x2 = 18 states
Actions:     Green-NS, Green-EW, All-Red (emergency stop)  → 3 actions
Rewards:     +10 per car cleared, +500 emergency served, -50 emergency wait
Constraint:  Emergency must be served within 1 timestep
Algorithm:   Q-learning with safety penalty
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

N_QUEUES = 3       # 0, 1-3, 4+ cars (discretized)
N_EW_QUEUES = 3
N_EMERGENCY = 2    # 0 or 1
N_ACTIONS = 3      # Green-NS, Green-EW, All-Red
ACTION_NAMES = ["Green-NS", "Green-EW", "All-Red"]

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.993
N_EPISODES = 300
MAX_STEPS = 50

EMERGENCY_PENALTY = -50
EMERGENCY_BONUS = 500
CLEAR_REWARD = 10
WAIT_PENALTY = -2


def discretize_queue(q):
    if q == 0: return 0
    if q <= 3: return 1
    return 2


def step(state, action):
    q_ns, q_ew, emergency = state

    reward = 0
    cars_cleared = 0
    emergency_served = False

    if emergency == 1:
        if action == 2:  # All-Red → emergency passes
            reward += EMERGENCY_BONUS
            emergency_served = True
        else:
            reward += EMERGENCY_PENALTY

    # Clear cars based on green
    if action == 0 and not emergency_served:  # Green-NS
        cars_cleared = min(q_ns + 1, 5)
    elif action == 1 and not emergency_served:  # Green-EW
        cars_cleared = min(q_ew + 1, 5)

    reward += cars_cleared * CLEAR_REWARD

    # New arrivals (random)
    new_ns = np.random.randint(0, 3)
    new_ew = np.random.randint(0, 3)
    new_emergency = 1 if np.random.rand() < 0.15 else 0

    next_q_ns = max(q_ns - (cars_cleared if action == 0 else 0) + new_ns, 0)
    next_q_ew = max(q_ew - (cars_cleared if action == 1 else 0) + new_ew, 0)

    next_state = (discretize_queue(next_q_ns), discretize_queue(next_q_ew),
                  0 if emergency_served else new_emergency)

    return next_state, reward, False


def train():
    Q = np.zeros((N_QUEUES, N_EW_QUEUES, N_EMERGENCY, N_ACTIONS))
    epsilon = EPSILON_START
    episode_rewards = []
    emergency_response_times = []

    for ep in range(N_EPISODES):
        state = (0, 0, 0)
        total_reward = 0
        responses = []
        step_count = 0

        for _ in range(MAX_STEPS):
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state[0], state[1], state[2]]))

            next_state, reward, _ = step(state, action)
            best_next = np.max(Q[next_state[0], next_state[1], next_state[2]])
            Q[state[0], state[1], state[2], action] += ALPHA * (
                reward + GAMMA * best_next - Q[state[0], state[1], state[2], action]
            )

            if state[2] == 1 and action == 2:
                responses.append(step_count)

            state = next_state
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        if responses:
            emergency_response_times.append(np.mean(responses))
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"  Episode {ep+1}/{N_EPISODES} | Avg reward: {avg:.1f} | Epsilon: {epsilon:.3f}")

    return Q, episode_rewards, emergency_response_times


def plot_results(episode_rewards, response_times):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    window = 10
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(episode_rewards)), ma, linewidth=2, label=f"{window}-ep avg")
    ax.plot(episode_rewards, alpha=0.3, label="Episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if response_times:
        ax.plot(response_times, linewidth=1.5, color="red")
        ax.set_xlabel("Episode (with emergency)")
        ax.set_ylabel("Avg Emergency Response Time (steps)")
        ax.set_title("Emergency Vehicle Response Time")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No emergencies recorded", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("experiment6_traffic_signal.png", dpi=150)
    print("Plot saved as 'experiment6_traffic_signal.png'")


if __name__ == "__main__":
    print("Experiment 6: Traffic Signal with Emergency Priority\n")
    print(f"States: {N_QUEUES}x{N_EW_QUEUES}x{N_EMERGENCY} = {N_QUEUES*N_EW_QUEUES*N_EMERGENCY} discrete")
    print(f"Actions: {ACTION_NAMES}")
    print(f"Rewards: +{CLEAR_REWARD}/car, +{EMERGENCY_BONUS} emergency, {EMERGENCY_PENALTY} emergency wait\n")
    print("Training Q-learning agent...\n")

    Q, rewards, responses = train()
    print(f"\nEmergency episodes with response data: {len(responses)}")
    if responses:
        print(f"Final avg response time: {responses[-1]:.2f} steps")

    plot_results(rewards, responses)
    print("\nExperiment 6 completed successfully!")
