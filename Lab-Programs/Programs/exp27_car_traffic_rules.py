"""
Exp 27 – Autonomous Car with Traffic-Rule Policies
====================================================
Custom graph representing city roads:
  • 10 nodes (intersections), directed edges (one-way segments)
  • Stop-sign nodes (mandatory full stop -> time penalty)
  • Right-of-way rules (yield at certain edges)

Three rule-based policies + Q-learning agent with rule-violation penalty.
Metrics compared: travel time, rule violations, safety score.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# -- Road network ------------------------------------------------------
# Nodes 0-9, directed edges (from, to, travel_time)
EDGES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 2), (3, 4, 4), (4, 5, 2),
    (5, 6, 3), (6, 7, 2), (7, 8, 3), (8, 9, 2),
    (1, 5, 5), (2, 6, 4), (3, 7, 5),
    (5, 1, 4), (6, 2, 3), (7, 3, 4),
    (0, 5, 6), (5, 9, 4),
    (9, 0, 8),
]
NODES = 10
START, GOAL = 0, 9

# Traffic rules
STOP_NODES = {2, 5, 7}          # mandatory stop -> +2 time penalty
YIELD_EDGES = {(1, 5), (2, 6)}  # must yield (slow down -> +1 time)
ONE_WAY = {(3, 4)}              # one-way: only this direction allowed
VIOLATION_PENALTY = 20           # penalty for breaking rules

# Build adjacency
ADJ = defaultdict(list)
for u, v, t in EDGES:
    ADJ[u].append((v, t))


# -- Rule-based policies -----------------------------------------------
def policy_naive(start, goal):
    """Greedy BFS ignoring all rules."""
    from collections import deque
    parent = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == goal:
            break
        for nb, _ in ADJ[node]:
            if nb not in parent:
                parent[nb] = node
                q.append(nb)
    if goal not in parent:
        return [], 999, 0
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    cost = sum(t for u, v, t in EDGES if (u, v) in set(zip(path, path[1:])) or False)
    # compute cost from path
    total_t = 0
    violations = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        found = [t for u, v, t in EDGES if u == edge[0] and v == edge[1]]
        if found:
            total_t += found[0]
        # check stop node
        if path[i] in STOP_NODES:
            total_t += 2  # stop penalty but naive ignores it
        # check yield
        if edge in YIELD_EDGES:
            total_t += 1
    return path, total_t, violations


def policy_stop_respecting(start, goal):
    """BFS that adds stop penalties."""
    from collections import deque
    parent = {start: (None, 0)}
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb, t in ADJ[node]:
            extra = 2 if nb in STOP_NODES else 0
            alt = dist[node] + t + extra
            if nb not in dist or alt < dist[nb]:
                dist[nb] = alt
                parent[nb] = (node, t)
                q.append(nb)
    if goal not in dist:
        return [], 999, 0
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node][0]
    path.reverse()
    return path, dist[goal], 0


def policy_full_rules(start, goal):
    """Dijkstra with all rule penalties."""
    import heapq
    dist = {start: 0}
    parent = {start: (None, 0)}
    visited = set()
    pq = [(0, start)]
    violations = 0

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for nb, t in ADJ[node]:
            extra = 0
            if nb in STOP_NODES:
                extra += 2
            if (node, nb) in YIELD_EDGES:
                extra += 1
            alt = d + t + extra
            if nb not in dist or alt < dist[nb]:
                dist[nb] = alt
                parent[nb] = (node, t)
                heapq.heappush(pq, (alt, nb))

    if goal not in dist:
        return [], 999, 0

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node][0]
    path.reverse()
    return path, dist[goal], 0


# -- Q-learning agent --------------------------------------------------
Q = defaultdict(lambda: np.zeros(NODES))
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.1
QL_EPISODES = 3000


def travel_time(u, v):
    for a, b, t in EDGES:
        if a == u and b == v:
            return t
    return 999


def rule_penalty(u, v):
    penalty = 0
    if v in STOP_NODES:
        penalty += 2
    if (u, v) in YIELD_EDGES:
        penalty += 1
    # check one-way violation (going against allowed direction)
    if (v, u) in ONE_WAY:
        penalty += VIOLATION_PENALTY
    return penalty


def qlearn_train():
    for ep in range(QL_EPISODES):
        state = START
        for _ in range(50):
            if state == GOAL:
                break
            valid = [v for v, _ in ADJ[state]]
            if not valid:
                break
            if np.random.rand() < EPSILON:
                a = np.random.choice(valid)
            else:
                qvals = Q[state][valid]
                a = valid[int(np.argmax(qvals))]
            t = travel_time(state, a)
            pen = rule_penalty(state, a)
            reward = -(t + pen)
            next_state = a
            best_next = np.max(Q[next_state]) if ADJ[next_state] else 0
            Q[state][a] += ALPHA * (reward + GAMMA * best_next - Q[state][a])
            state = next_state


def qlearn_navigate():
    path = [START]
    state = START
    total_t, violations = 0, 0
    visited = set()
    for _ in range(30):
        if state == GOAL:
            break
        valid = [v for v, _ in ADJ[state]]
        if not valid:
            break
        qvals = Q[state][valid]
        a = valid[int(np.argmax(qvals))]
        t = travel_time(state, a)
        pen = rule_penalty(state, a)
        total_t += t
        if (state, a) in ONE_WAY:
            violations += 1
        if pen > 0 and (state, a) in YIELD_EDGES:
            pass  # yield is not a violation, just slow
        if a in STOP_NODES:
            pass  # stop is legal
        path.append(a)
        state = a
        visited.add((path[-2], a) if len(path) > 1 else None)
    return path, total_t, violations


def safety_score(violations, total_t):
    """Lower violations and time -> higher safety."""
    return max(0, 100 - violations * 30 - total_t)


def plot_results(results):
    out_dir = r'C:\Users\Faizze-PI\Desktop\SIMATS Subjects\MLA03 - ClassStuff\Lab Programs\Outputs'
    names = list(results.keys())
    times = [results[n]["time"] for n in names]
    violations = [results[n]["violations"] for n in names]
    safety = [results[n]["safety"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    axes[0].barh(names, times, color=colors[:len(names)])
    axes[0].set_xlabel("Travel Time (steps + penalties)")
    axes[0].set_title("Travel Time")

    axes[1].barh(names, violations, color=colors[:len(names)])
    axes[1].set_xlabel("Rule Violations")
    axes[1].set_title("Violations")

    axes[2].barh(names, safety, color=colors[:len(names)])
    axes[2].set_xlabel("Safety Score")
    axes[2].set_title("Safety (higher = better)")

    plt.tight_layout()
    path = os.path.join(out_dir, "exp27_traffic_rules_results.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved -> {path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Exp 27 – Autonomous Car with Traffic Rules")
    print("=" * 60)

    # Rule-based policies
    p1, t1, v1 = policy_naive(START, GOAL)
    p2, t2, v2 = policy_stop_respecting(START, GOAL)
    p3, t3, v3 = policy_full_rules(START, GOAL)

    # Q-learning
    qlearn_train()
    p4, t4, v4 = qlearn_navigate()

    results = {
        "Naive BFS":          {"path": p1, "time": t1, "violations": v1, "safety": safety_score(v1, t1)},
        "Stop-Respecting":    {"path": p2, "time": t2, "violations": v2, "safety": safety_score(v2, t2)},
        "Full Rules (Dijk)":  {"path": p3, "time": t3, "violations": v3, "safety": safety_score(v3, t3)},
        "Q-Learning":         {"path": p4, "time": t4, "violations": v4, "safety": safety_score(v4, t4)},
    }

    for name, info in results.items():
        print(f"\n  {name}:")
        print(f"    Path       : {' -> '.join(map(str, info['path']))}")
        print(f"    Time       : {info['time']}")
        print(f"    Violations : {info['violations']}")
        print(f"    Safety     : {info['safety']}")

    plot_results(results)
    print("\nDone.")

