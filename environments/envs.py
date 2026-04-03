"""
envs.py — Environment constructors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core import MDP


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def gridworld(width, height, goal_xy, step_cost=-1.0, goal_reward=0.0,
              slip=0.0, gamma=0.99):
    """4-action gridworld with optional slip. Goal is absorbing terminal."""
    S, A = width * height, 4

    def idx(x, y): return y * width + x
    goal = idx(*goal_xy)

    T = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    for s in range(S):
        x, y = s % width, s // width
        for a in range(A):
            if s == goal:
                T[s, a, goal] = 1.0
                continue

            dx, dy = moves[a]
            nx, ny = x + dx, y + dy
            ns_intended = idx(nx, ny) if (0 <= nx < width and 0 <= ny < height) else s

            for a2 in range(A):
                dx2, dy2 = moves[a2]
                nx2, ny2 = x + dx2, y + dy2
                ns2 = idx(nx2, ny2) if (0 <= nx2 < width and 0 <= ny2 < height) else s
                prob = slip / A
                T[s, a, ns2] += prob
                R[s, a, ns2] += prob * (step_cost + (goal_reward if ns2 == goal else 0.0))

            T[s, a, ns_intended] += (1.0 - slip)
            R[s, a, ns_intended] += (1.0 - slip) * (
                step_cost + (goal_reward if ns_intended == goal else 0.0))

    d0 = np.zeros(S)
    d0[idx(0, 0)] = 1.0
    # Goal is absorbing with R=0 (not marked terminal) so shaping invariance
    # holds: V(goal) naturally converges to 0, and shifts to -Phi(goal) under
    # shaping, preserving V'(s)-V'(s) differences everywhere.
    return MDP(S=S, A=A, T=T, R=R, gamma=gamma, d0=d0)


def make_chain_mdp(n: int = 10, gamma: float = 0.95,
                   reward_type: str = 'terminal') -> MDP:
    """Linear chain: action 0 = stay, action 1 = advance. Last state terminal."""
    S, A = n, 2
    T = np.zeros((S, A, S))
    for s in range(S - 1):
        T[s, 0, s] = 1.0
        T[s, 1, s + 1] = 1.0
    T[S - 1, :, S - 1] = 1.0 # absorbing terminal

    R = np.zeros((S, A, S))
    rng_r = np.random.default_rng(0)

    if reward_type == 'terminal':
        R[:, :, S - 1] = 1.0
    elif reward_type == 'dense':
        for sp in range(S):
            R[:, :, sp] = sp / (S - 1)
    elif reward_type == 'lottery':
        R[:, :, S - 1] = rng_r.uniform(0.5, 1.5)
    elif reward_type == 'progress':
        for s in range(S - 1):
            R[s, 1, s + 1] = 0.1
        R[:, :, S - 1] += 1.0

    return MDP(S=S, A=A, T=T, R=R, gamma=gamma, terminal={S - 1})


def make_grid_mdp(rows: int = 5, cols: int = 5,
                  gamma: float = 0.95,
                  reward_type: str = 'goal') -> MDP:
    """5x5 grid with goal, optional cliff. 4 actions (UDLR)."""
    S = rows * cols
    A = 4
    T = np.zeros((S, A, S))
    R = np.zeros((S, A, S))

    def idx(r, c): return r * cols + c
    def coord(s): return s // cols, s % cols

    goal = idx(rows - 1, cols - 1)
    cliff = {idx(0, c) for c in range(1, cols - 1)}
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for s in range(S):
        r, c = coord(s)
        if s == goal:
            T[s, :, s] = 1.0
            continue
        for a, (dr, dc) in enumerate(deltas):
            nr = max(0, min(rows - 1, r + dr))
            nc = max(0, min(cols - 1, c + dc))
            T[s, a, idx(nr, nc)] += 1.0

    if reward_type == 'goal':
        R[:, :, goal] = 1.0
    elif reward_type == 'local':
        for s in range(S):
            r, c = coord(s)
            dist = abs(r - (rows - 1)) + abs(c - (cols - 1))
            R[s, :, :] = -dist / (rows + cols)
    elif reward_type == 'cliff':
        R[:, :, goal] = 1.0
        for s in cliff:
            R[:, :, s] = -1.0

    return MDP(S=S, A=A, T=T, R=R, gamma=gamma, terminal={goal})
