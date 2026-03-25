"""
core.py — MDP dataclass + all DP / MCE / shaping functions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# MDP Primitive
# ---------------------------------------------------------------------------

@dataclass
class MDP:
    """
    Finite tabular MDP.
      S: number of states
      A: number of actions
      T: (S, A, S) transition probabilities, T[s,a,s'] = P(s'|s,a)
      R: (S, A, S) rewards
      gamma: discount factor in (0,1)
      terminal: set of absorbing state indices
      d0: (S,) start state distribution (defaults to uniform)
    """
    S: int
    A: int
    T: np.ndarray
    R: np.ndarray
    gamma: float = 0.95
    terminal: set = field(default_factory=set)
    d0: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.d0 is None:
            self.d0 = np.ones(self.S) / self.S
        else:
            self.d0 = np.asarray(self.d0, dtype=float)
            self.d0 /= self.d0.sum()

    def expected_reward(self):
        return (self.T * self.R).sum(axis=2)  # (S, A)


# ---------------------------------------------------------------------------
# Core DP
# ---------------------------------------------------------------------------

def value_iteration(mdp: MDP, tol: float = 1e-10, max_iter: int = 20_000):
    """Returns V*, Q*, pi* via synchronous value iteration."""
    V = np.zeros(mdp.S)
    ER = mdp.expected_reward()

    for _ in range(max_iter):
        Q = ER + mdp.gamma * (mdp.T * V[None, None, :]).sum(axis=2)
        for s in mdp.terminal:
            Q[s, :] = 0.0
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    Q = ER + mdp.gamma * (mdp.T * V[None, None, :]).sum(axis=2)
    for s in mdp.terminal:
        Q[s, :] = 0.0
    pi = Q.argmax(axis=1)
    return V, Q, pi


def policy_evaluation(mdp: MDP, pi, tol: float = 1e-10, max_iter: int = 200_000):
    """V^pi for deterministic pi (shape [S]) or stochastic pi (shape [S, A])."""
    pi_arr = np.asarray(pi)
    if pi_arr.ndim == 1:
        idx = np.arange(mdp.S)
        P = mdp.T[idx, pi_arr, :]                                    # (S, S)
        r = (mdp.T[idx, pi_arr, :] * mdp.R[idx, pi_arr, :]).sum(1)  # (S,)
    else:
        pi_arr = pi_arr.astype(float)
        P = (pi_arr[:, :, None] * mdp.T).sum(axis=1)
        r = (pi_arr[:, :, None] * mdp.T * mdp.R).sum(axis=(1, 2))

    V = np.zeros(mdp.S)
    for _ in range(max_iter):
        V_new = r + mdp.gamma * (P @ V)
        for s in mdp.terminal:
            V_new[s] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def value_under_random_policy(mdp: MDP, tol: float = 1e-8, max_iter: int = 20_000):
    """V^rand(s) under uniform random policy (convenience wrapper)."""
    V = np.zeros(mdp.S)
    ER = mdp.expected_reward()
    uniform_ER = ER.mean(axis=1)

    for _ in range(max_iter):
        T_avg = mdp.T.mean(axis=1)
        V_new = uniform_ER + mdp.gamma * T_avg @ V
        for s in mdp.terminal:
            V_new[s] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    return V_new


def discounted_occupancy(mdp: MDP, pi):
    """
    Discounted state visitation: d = (1-gamma) sum_t gamma^t d_t.
    Solves (I - gamma P_pi^T) d = (1-gamma) d0.
    """
    pi_arr = np.asarray(pi)
    if pi_arr.ndim == 1:
        P = mdp.T[np.arange(mdp.S), pi_arr, :]
    else:
        P = (pi_arr[:, :, None].astype(float) * mdp.T).sum(axis=1)

    A = np.eye(mdp.S) - mdp.gamma * P.T
    b = (1.0 - mdp.gamma) * mdp.d0
    d = np.linalg.solve(A, b)
    d = np.clip(d, 0.0, None)
    return d / d.sum()


def finite_horizon_optimal_policy(mdp: MDP, h: int):
    """
    Optimal policy for truncated h-step horizon (backward induction).
    Returns (V0, pi_h) — value at t=0 and greedy policy at t=0.
    """
    if h <= 0:
        return np.zeros(mdp.S), np.zeros(mdp.S, dtype=int)

    V_next = np.zeros(mdp.S)
    pi0 = np.zeros(mdp.S, dtype=int)

    for t in reversed(range(h)):
        Q_t = (mdp.T * (mdp.R + mdp.gamma * V_next[None, None, :])).sum(axis=2)
        for s in mdp.terminal:
            Q_t[s, :] = 0.0
        V_t = Q_t.max(axis=1)
        if t == 0:
            pi0 = Q_t.argmax(axis=1)
        V_next = V_t

    return V_next, pi0


def finite_horizon_lookahead_policy(mdp: MDP, h: int, V_terminal: np.ndarray):
    """
    Depth-h lookahead with terminal value bootstrap:
      J_h(s) = max_pi E[ sum_{t<h} gamma^t r_t + gamma^h V_terminal(S_h) | s ]
    Returns (V0, pi_h).
    """
    V_terminal = np.asarray(V_terminal, dtype=float)
    if h <= 0:
        return V_terminal.copy(), np.zeros(mdp.S, dtype=int)

    V_next = V_terminal.copy()
    pi0 = np.zeros(mdp.S, dtype=int)

    for t in reversed(range(h)):
        Q_t = (mdp.T * (mdp.R + mdp.gamma * V_next[None, None, :])).sum(axis=2)
        for s in mdp.terminal:
            Q_t[s, :] = 0.0
        V_t = Q_t.max(axis=1)
        if t == 0:
            pi0 = Q_t.argmax(axis=1)
        V_next = V_t

    return V_next, pi0


# ---------------------------------------------------------------------------
# MCE (Maximum Causal Entropy)
# ---------------------------------------------------------------------------

def soft_value_iteration(mdp: MDP, alpha: float = 1.0,
                         tol: float = 1e-10, max_iter: int = 20_000):
    """
    Soft Bellman recursion:
        Q^S(s,a) = E[R] + gamma * sum_s' T(s,a,s') V^S(s')
        V^S(s)   = alpha * logsumexp(Q^S(s,:) / alpha)
    Returns (V_soft, Q_soft, pi_mce).
    """
    V = np.zeros(mdp.S)
    ER = mdp.expected_reward()

    for _ in range(max_iter):
        Q = ER + mdp.gamma * (mdp.T * V[None, None, :]).sum(axis=2)
        for s in mdp.terminal:
            Q[s, :] = 0.0
        V_new = alpha * logsumexp(Q / alpha, axis=1)
        for s in mdp.terminal:
            V_new[s] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    Q_soft = ER + mdp.gamma * (mdp.T * V[None, None, :]).sum(axis=2)
    for s in mdp.terminal:
        Q_soft[s, :] = 0.0
    return V, Q_soft, mce_policy(Q_soft, alpha)


def mce_policy(Q_soft: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """MCE policy: pi[s,a] = softmax(Q^S(s,:) / alpha). Returns (S, A)."""
    scaled = Q_soft / alpha
    log_Z = logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(scaled - log_Z)


def mce_objective(mdp: MDP, Q_soft: np.ndarray, alpha: float = 1.0) -> float:
    """J_MCE = mean V^S(s) over non-terminal states."""
    V_soft = alpha * logsumexp(Q_soft / alpha, axis=1)
    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')
    return float(np.mean(V_soft[non_terminal]))


# ---------------------------------------------------------------------------
# Potential Shaping
# ---------------------------------------------------------------------------

def add_potential_shaping(mdp: MDP, Phi):
    """R' = R + gamma*Phi(s') - Phi(s). Returns new MDP."""
    Phi = np.asarray(Phi, dtype=float)
    F = mdp.gamma * Phi[None, None, :] - Phi[:, None, None]
    return MDP(S=mdp.S, A=mdp.A, T=mdp.T, R=mdp.R + F,
               gamma=mdp.gamma, terminal=mdp.terminal, d0=mdp.d0.copy())
