"""
Finite MDP Lab — agenticity proxies & shaping-invariant metrics
================================================================
Exact dynamic programming on tabular MDPs. No learning loops.

Week 2 metrics (use d0, verify shaping invariance):
  - control_advantage: V* vs baseline policy value
  - one_step_recovery: recovery after state perturbation
  - planning_pressure: cost of depth-limited lookahead

Week 3 proxies (shaping-invariant agenticity measures):
  - advantage_gap: mean action-value spread per state
  - vstar_variance_corrected: Var(V* - V^rand) over states
  - early_action_mi: I(early actions; G | s0) - I(late actions; G | s0) (Mutual Information between early actions and return, high signals irreversibility)
  - advantage_sparsity: fraction of near-zero advantages

MCE (Maximum Causal Entropy):
  - soft_value_iteration: entropy-regularised Bellman recursion
  - mce_policy: softmax policy extraction
  - mce_objective: soft value averaged over states
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import entropy as scipy_entropy
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


# ---------------------------------------------------------------------------
# Week 2 Metrics (shaping-invariant, use d0)
# ---------------------------------------------------------------------------

def control_advantage(mdp: MDP, pi_baseline):
    """A_ctrl = E_{s~d0}[ V*(s) - V^{pi0}(s) ]."""
    V_star, _, _ = value_iteration(mdp)
    V0 = policy_evaluation(mdp, pi_baseline)
    return float(mdp.d0 @ (V_star - V0))


def one_step_recovery(mdp: MDP, pi_baseline):
    """
    RecAdv = E_{s~d*} E_{a~Uniform(A)} E_{s'~T(s,a,·)}[ V*(s') - V^{pi0}(s') ].
    Random action taken from s, then recovery advantage measured at landing state s'.
    Invariant: both values shift by -Phi(s'), which cancels.
    """
    _, _, pi_star = value_iteration(mdp)
    d_star = discounted_occupancy(mdp, pi_star)

    V_star = policy_evaluation(mdp, pi_star)
    V0 = policy_evaluation(mdp, pi_baseline)

    # uniform random action kernel: P(s'|s) = (1/A) * sum_a T(s,a,s')
    rand_kernel = mdp.T.mean(axis=1)  # (S, S)
    dist_S0 = d_star @ rand_kernel
    return float(dist_S0 @ (V_star - V0))


def planning_pressure(mdp: MDP, h: int):
    """
    P_h = E_{s~d0}[ V*(s) - V^{pi_h}(s) ] where pi_h is depth-h
    lookahead with V* terminal bootstrap.
    """
    V_star, _, _ = value_iteration(mdp)
    _, pi_h = finite_horizon_lookahead_policy(mdp, h, V_terminal=V_star)
    V_pi_h = policy_evaluation(mdp, pi_h)
    return float(mdp.d0 @ (V_star - V_pi_h))


# ---------------------------------------------------------------------------
# Week 3 Proxies (shaping-invariant agenticity measures)
# ---------------------------------------------------------------------------

def advantage_gap(mdp: MDP, Q: np.ndarray, V: np.ndarray) -> float:
    """Mean(max_a A* - min_a A*) over non-terminal states. Built on A* = Q* - V*."""
    A_star = Q - V[:, None]
    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')
    A_nt = A_star[non_terminal]
    return float(np.mean(A_nt.max(axis=1) - A_nt.min(axis=1)))


def vstar_variance_corrected(mdp: MDP, V_star: np.ndarray,
                              V_rand: np.ndarray) -> float:
    """Var(V* - V^rand) over non-terminal states. Phi cancels in the difference."""
    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')
    return float(np.var(V_star[non_terminal] - V_rand[non_terminal]))


def early_action_mi(mdp: MDP, pi: np.ndarray,
                    horizon: int = 60,
                    n_episodes: int = 2000,
                    early_cutoff: int = None,
                    n_bins: int = 10,
                    min_samples_per_s0: int = 15,
                    rng: np.random.Generator = None) -> dict:
    """
    I(A_{1:k}; G | s_0) - I(A_{k+1:T}; G | s_0).
    Conditioned on s_0 for shaping invariance. Uses epsilon-greedy (eps=0.3).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if early_cutoff is None:
        early_cutoff = max(1, horizon // 3)

    epsilon = 0.3
    start_states, action_seqs, returns = [], [], []

    for _ in range(n_episodes):
        s0 = rng.integers(0, mdp.S)
        s = s0
        G, discount, actions = 0.0, 1.0, []

        for t in range(horizon):
            if s in mdp.terminal:
                actions.extend([-1] * (horizon - t))
                break
            a = rng.integers(0, mdp.A) if rng.random() < epsilon else pi[s]
            actions.append(a)
            s_next = rng.choice(mdp.S, p=mdp.T[s, a, :])
            G += discount * mdp.R[s, a, s_next]
            discount *= mdp.gamma
            s = s_next

        start_states.append(s0)
        action_seqs.append(actions[:horizon])
        returns.append(G)

    start_states = np.array(start_states)
    returns = np.array(returns)
    action_seqs = np.array(action_seqs)

    def hash_seq(seqs):
        shifted = seqs + 1
        A_eff = mdp.A + 1
        hashes = np.zeros(len(seqs), dtype=np.int64)
        for t in range(seqs.shape[1]):
            hashes = hashes * A_eff + shifted[:, t]
        return hashes

    early_h = hash_seq(action_seqs[:, :early_cutoff])
    late_h = hash_seq(action_seqs[:, early_cutoff:])

    def empirical_mi(X, Y):
        x_vals, x_inv = np.unique(X, return_inverse=True)
        y_vals, y_inv = np.unique(Y, return_inverse=True)
        n = len(X)
        if n < 2:
            return 0.0
        joint = np.zeros((len(x_vals), len(y_vals)))
        np.add.at(joint, (x_inv, y_inv), 1.0)
        joint /= n
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        mask = joint > 0
        mi = np.sum(joint[mask] * np.log(joint[mask] / (px * py + 1e-12)[mask]))
        return max(0.0, float(mi))

    unique_s0 = np.unique(start_states)
    mi_early_parts, mi_late_parts, group_weights = [], [], []

    for s0 in unique_s0:
        mask = start_states == s0
        n_s0 = mask.sum()
        if n_s0 < min_samples_per_s0:
            continue

        G_s0 = returns[mask]
        r_min, r_max = G_s0.min(), G_s0.max()
        if abs(r_max - r_min) < 1e-10:
            mi_early_parts.append(0.0)
            mi_late_parts.append(0.0)
            group_weights.append(n_s0)
            continue

        G_bins_s0 = np.digitize(
            G_s0, bins=np.linspace(r_min, r_max + 1e-8, n_bins + 1)[1:-1])
        mi_early_parts.append(empirical_mi(early_h[mask], G_bins_s0))
        mi_late_parts.append(empirical_mi(late_h[mask], G_bins_s0))
        group_weights.append(n_s0)

    if not group_weights:
        return {'mi_early': 0.0, 'mi_late': 0.0, 'mi_diff': 0.0,
                'H_G': 0.0, 'early_cutoff': early_cutoff}

    w = np.array(group_weights, dtype=float)
    w /= w.sum()
    mi_early = float(np.dot(w, mi_early_parts))
    mi_late = float(np.dot(w, mi_late_parts))

    r_min, r_max = returns.min(), returns.max()
    if abs(r_max - r_min) < 1e-10:
        h_g = 0.0
    else:
        G_bins_all = np.digitize(
            returns, bins=np.linspace(r_min, r_max + 1e-8, n_bins + 1)[1:-1])
        g_dist = np.bincount(G_bins_all, minlength=n_bins).astype(float)
        g_dist /= g_dist.sum()
        h_g = float(scipy_entropy(g_dist + 1e-12))

    norm = h_g + 1e-8
    return {
        'mi_early': round(mi_early / norm, 4),
        'mi_late':  round(mi_late  / norm, 4),
        'mi_diff':  round((mi_early - mi_late) / norm, 4),
        'H_G':      round(h_g, 4),
        'early_cutoff': early_cutoff,
    }


def advantage_sparsity(mdp: MDP, Q: np.ndarray, V: np.ndarray,
                       threshold: float = 1e-6) -> float:
    """Fraction of non-terminal (s,a) with |A*(s,a)| < threshold."""
    A_star = Q - V[:, None]
    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')
    return float((np.abs(A_star[non_terminal]) < threshold).mean())


def agenticity_score(mdp: MDP, weights: dict = None,
                     horizon: int = 60, n_episodes: int = 2000,
                     verbose: bool = True,
                     rng: np.random.Generator = None) -> dict:
    """
    Composite agenticity (all shaping-invariant).
    Weights: adv_gap=0.25, vstar_var=0.40, mi_diff=0.35.
    """
    if weights is None:
        weights = {'adv_gap': 0.25, 'vstar_var': 0.40, 'mi_diff': 0.35}
    if rng is None:
        rng = np.random.default_rng(42)

    V_star, Q_star, pi_star = value_iteration(mdp)
    V_rand = value_under_random_policy(mdp)

    ag = advantage_gap(mdp, Q_star, V_star)
    vv = vstar_variance_corrected(mdp, V_star, V_rand)
    mi = early_action_mi(mdp, pi_star, horizon=horizon,
                         n_episodes=n_episodes, rng=rng)
    asp = advantage_sparsity(mdp, Q_star, V_star)

    ag_norm = float(1 / (1 + np.exp(-ag * 5)))
    vv_norm = float(1 / (1 + np.exp(-vv / 2.0)))
    mi_norm = float(np.clip((mi['mi_diff'] + 1) / 2, 0, 1))

    composite = (weights['adv_gap'] * ag_norm +
                 weights['vstar_var'] * vv_norm +
                 weights['mi_diff'] * mi_norm)

    result = {
        'adv_gap': round(ag, 4),
        'adv_gap_norm': round(ag_norm, 4),
        'vstar_var_raw': round(vv, 4),
        'vstar_var_norm': round(vv_norm, 4),
        'mi_early': mi['mi_early'],
        'mi_late':  mi['mi_late'],
        'mi_diff':  mi['mi_diff'],
        'mi_diff_norm': round(mi_norm, 4),
        'adv_sparsity': round(asp, 4),
        'composite': round(composite, 4),
    }

    if verbose:
        print(f"  Advantage gap (raw / norm):             {ag:.4f} / {ag_norm:.4f}")
        print(f"  V*-V^rand variance (raw / norm):        {vv:.4f} / {vv_norm:.4f}")
        print(f"  MI|s0 (early / late / diff):            {mi['mi_early']:.4f} / {mi['mi_late']:.4f} / {mi['mi_diff']:+.4f}")
        print(f"  Advantage sparsity (diagnostic):        {asp:.4f}")
        print(f"  ── COMPOSITE: {composite:.4f}")

    return result


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
    T[S - 1, :, S - 1] = 1.0

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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # --- Week 2: shaping invariance verification ---
    print("=" * 70)
    print("WEEK 2 — SHAPING INVARIANCE VERIFICATION (5x5 gridworld, slip=0.1)")
    print("=" * 70)

    gw = gridworld(5, 5, goal_xy=(4, 4), step_cost=-1.0, slip=0.1, gamma=0.99)
    pi0 = np.ones((gw.S, gw.A)) / gw.A

    Actrl = control_advantage(gw, pi0)
    R1 = one_step_recovery(gw, pi0)
    Ph = planning_pressure(gw, h=3)
    print(f"  A_ctrl = {Actrl:.6f}")
    print(f"  R1     = {R1:.6f}")
    print(f"  P_h(3) = {Ph:.6f}")

    Phi = np.random.default_rng(0).normal(size=gw.S)
    gw2 = add_potential_shaping(gw, Phi)
    Actrl2 = control_advantage(gw2, pi0)
    R1_2 = one_step_recovery(gw2, pi0)
    Ph2 = planning_pressure(gw2, h=3)
    print(f"  Δ A_ctrl = {Actrl2 - Actrl:.2e}  (should be ~0)")
    print(f"  Δ R1     = {R1_2 - R1:.2e}  (should be ~0)")
    print(f"  Δ P_h    = {Ph2 - Ph:.2e}  (should be ~0)")

    # --- Week 3: agenticity proxies ---
    print("\n" + "=" * 70)
    print("WEEK 3 — AGENTICITY PROXIES (shaping-invariant)")
    print("=" * 70)

    cases = [
        ("Chain — Terminal",   make_chain_mdp(10, reward_type='terminal')),
        ("Chain — Dense",      make_chain_mdp(10, reward_type='dense')),
        ("Chain — Lottery",    make_chain_mdp(10, reward_type='lottery')),
        ("Chain — Progress",   make_chain_mdp(10, reward_type='progress')),
        ("Grid  — Goal",      make_grid_mdp(5, 5, reward_type='goal')),
        ("Grid  — Local",     make_grid_mdp(5, 5, reward_type='local')),
        ("Grid  — Cliff",     make_grid_mdp(5, 5, reward_type='cliff')),
    ]

    summary = []
    for name, mdp in cases:
        print(f"\n>>> {name}")
        r = agenticity_score(mdp, verbose=True, rng=rng)
        summary.append((name, r['composite'], r['adv_gap_norm'],
                        r['vstar_var_norm'], r['mi_diff'], r['adv_sparsity']))

    print("\n" + "=" * 70)
    print("SUMMARY (descending composite)")
    print("=" * 70)
    print(f"{'MDP':<25} {'Comp':>6} {'AdvGp':>6} {'V*var':>6} {'MIdif':>6} {'ASpar':>6}")
    print("-" * 70)
    for row in sorted(summary, key=lambda x: -x[1]):
        name, comp, ag, vv, mi, asp = row
        print(f"{name:<25} {comp:>6.4f} {ag:>6.4f} {vv:>6.4f} {mi:>+6.4f} {asp:>6.4f}")

    # --- MCE demo ---
    print("\n" + "=" * 70)
    print("MCE DEMO — Grid-Cliff")
    print("=" * 70)

    cliff_mdp = make_grid_mdp(5, 5, reward_type='cliff')
    _, _, pi_hard = value_iteration(cliff_mdp)

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        V_soft, Q_soft, pi_mce = soft_value_iteration(cliff_mdp, alpha=alpha)
        J = mce_objective(cliff_mdp, Q_soft, alpha=alpha)
        hard_onehot = np.zeros_like(pi_mce)
        hard_onehot[np.arange(cliff_mdp.S), pi_hard] = 1.0
        dist = np.abs(pi_mce - hard_onehot).sum(axis=1).mean()
        print(f"  alpha={alpha:<6g}  J_MCE={J:+.4f}  L1(pi_mce,pi*)={dist:.4f}")

    print("  (alpha->0: hard optimal | alpha->inf: uniform)")

    print("\n" + "=" * 70)
    print("MCE DEMO — Chain-Dense")
    print("=" * 70)

    dense_mdp = make_chain_mdp(10, reward_type='dense')
    _, _, pi_hard_dense = value_iteration(dense_mdp)

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        V_soft, Q_soft, pi_mce = soft_value_iteration(dense_mdp, alpha=alpha)
        J = mce_objective(dense_mdp, Q_soft, alpha=alpha)
        hard_onehot = np.zeros_like(pi_mce)
        hard_onehot[np.arange(dense_mdp.S), pi_hard_dense] = 1.0
        dist = np.abs(pi_mce - hard_onehot).sum(axis=1).mean()
        print(f"  alpha={alpha:<6g}  J_MCE={J:+.4f}  L1(pi_mce,pi*)={dist:.4f}")

    print("  (alpha->0: hard optimal | alpha->inf: uniform)")
