import numpy as np

# ----------------------------
# 1) Finite MDP container
# ----------------------------
class FiniteMDP:
    def __init__(self, T, R, gamma, start_dist=None):
        """
        T: shape [S, A, S] transition probabilities
        R: shape [S, A, S] rewards
        gamma: discount in (0,1)
        start_dist: shape [S] distribution over start states (defaults to uniform)
        """
        self.T = np.asarray(T, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.gamma = float(gamma)
        assert self.T.ndim == 3 and self.R.shape == self.T.shape
        self.S, self.A, self.S2 = self.T.shape
        assert self.S == self.S2
        assert 0 < self.gamma < 1

        row_sums = self.T.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Each T[s,a,:] must sum to 1.")

        if start_dist is None:
            self.d0 = np.ones(self.S) / self.S
        else:
            self.d0 = np.asarray(start_dist, dtype=float)
            self.d0 = self.d0 / self.d0.sum()

# ----------------------------
# 2) Value iteration (optimal infinite-horizon)
# ----------------------------
def value_iteration(mdp: FiniteMDP, tol=1e-10, max_iters=100_000):
    '''
    Optimal value iteration for infinite-horizon discounted MDP.
    Returns (V_star, Q_star, pi_star)
    '''
    V = np.zeros(mdp.S)
    for _ in range(max_iters):
        Q = (mdp.T * (mdp.R + mdp.gamma * V[None, None, :])).sum(axis=2)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = (mdp.T * (mdp.R + mdp.gamma * V[None, None, :])).sum(axis=2)
    pi_star = Q.argmax(axis=1)
    return V, Q, pi_star

# ----------------------------
# 3) Policy evaluation (given pi)
# ----------------------------
def policy_evaluation(mdp: FiniteMDP, pi, tol=1e-10, max_iters=200_000):
    """
    pi: deterministic array [S] with actions 0..A-1, or stochastic policy [S,A]
    returns V^pi
    """
    if np.asarray(pi).ndim == 1:
        P = mdp.T[np.arange(mdp.S), pi, :]  # [S,S]
        r = (mdp.T[np.arange(mdp.S), pi, :] * mdp.R[np.arange(mdp.S), pi, :]).sum(axis=1)  # [S]
    else:
        pi = np.asarray(pi, dtype=float)  # [S,A]
        P = (pi[:, :, None] * mdp.T).sum(axis=1)  # [S,S]
        r = (pi[:, :, None] * mdp.T * mdp.R).sum(axis=(1, 2))  # [S]

    V = np.zeros(mdp.S)
    for _ in range(max_iters):
        V_new = r + mdp.gamma * (P @ V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V

# ----------------------------
# 4) Discounted state occupancy d^pi
# ----------------------------
def discounted_occupancy(mdp: FiniteMDP, pi):
    """
    d = (1-gamma) sum_t gamma^t d_t, where d_0 = mdp.d0, d_{t+1} = d_t P_pi.
    Finite MDP: solve (I - gamma P_pi^T) d = (1-gamma) d0.
    """
    if np.asarray(pi).ndim == 1:
        P = mdp.T[np.arange(mdp.S), pi, :]  # [S,S]
    else:
        pi = np.asarray(pi, dtype=float)
        P = (pi[:, :, None] * mdp.T).sum(axis=1)  # [S,S]

    I = np.eye(mdp.S)
    A = I - mdp.gamma * P.T
    b = (1.0 - mdp.gamma) * mdp.d0
    d = np.linalg.solve(A, b)
    d = np.clip(d, 0.0, None)
    d = d / d.sum()
    return d

# ----------------------------
# 5) Finite-horizon DP (for planning pressure)
# ----------------------------
def finite_horizon_optimal_policy(mdp: FiniteMDP, h: int):
    """
    Computes the optimal deterministic policy for the truncated-horizon objective:
        V_h*(s) = max_pi E[ sum_{t=0}^{h-1} gamma^t r_t | s, pi ].
    Returns:
        V0: the truncated-horizon optimal value at time 0, shape [S]
        pi_h: deterministic policy (argmax at time 0), shape [S]
    """
    if h <= 0:
        # horizon 0 means no rewards collected; any policy is optimal
        return np.zeros(mdp.S), np.zeros(mdp.S, dtype=int)

    V_next = np.zeros(mdp.S)  # V_{t+1}
    pi0 = np.zeros(mdp.S, dtype=int)

    # Backward induction: for t = h-1,...,0
    for t in reversed(range(h)):
        Q_t = (mdp.T * (mdp.R + mdp.gamma * V_next[None, None, :])).sum(axis=2)  # [S,A]
        V_t = Q_t.max(axis=1)  # [S]
        if t == 0:
            pi0 = Q_t.argmax(axis=1)
        V_next = V_t

    return V_next, pi0

def finite_horizon_lookahead_policy(mdp: FiniteMDP, h: int, V_terminal: np.ndarray):
    """
    Compute a depth-h lookahead policy with terminal value V_terminal:

      J_h(s) = max_pi E[ sum_{t=0}^{h-1} gamma^t r_t + gamma^h V_terminal(S_h) | S_0=s ]

    Returns:
      V0: value at time 0 for this objective, shape [S]
      pi_h: greedy policy at time 0 (deterministic), shape [S]
    """
    V_terminal = np.asarray(V_terminal, dtype=float)
    assert V_terminal.shape == (mdp.S,)

    if h <= 0:
        # horizon 0: act to maximize terminal value immediately (no reward collected)
        # but since no action is taken at t=0 in this convention, return arbitrary actions
        return V_terminal.copy(), np.zeros(mdp.S, dtype=int)

    V_next = V_terminal.copy()          # V_{t+1}; start from terminal
    pi0 = np.zeros(mdp.S, dtype=int)

    for t in reversed(range(h)):
        Q_t = (mdp.T * (mdp.R + mdp.gamma * V_next[None, None, :])).sum(axis=2)  # [S,A]
        V_t = Q_t.max(axis=1)
        if t == 0:
            pi0 = Q_t.argmax(axis=1)
        V_next = V_t

    return V_next, pi0


# ----------------------------
# 6) Potential shaping
# ----------------------------
def add_potential_shaping(mdp: FiniteMDP, Phi):
    """
    Returns new mdp with R' = R + gamma*Phi(s') - Phi(s)
    Phi: shape [S]
    """
    Phi = np.asarray(Phi, dtype=float)
    assert Phi.shape == (mdp.S,)
    F = mdp.gamma * Phi[None, None, :] - Phi[:, None, None]
    R2 = mdp.R + F
    return FiniteMDP(mdp.T, R2, mdp.gamma, start_dist=mdp.d0)

# ----------------------------
# 7) Metrics
# ----------------------------
def control_advantage(mdp: FiniteMDP, pi_baseline):
    """
    A_ctrl = E_{s~d0}[ V*(s) - V^{pi0}(s) ]
    """
    V_star, _, _ = value_iteration(mdp)
    V0 = policy_evaluation(mdp, pi_baseline)
    return float(mdp.d0 @ (V_star - V0))

def one_step_recovery(mdp: FiniteMDP, perturb_kernel, pi_baseline):
    """
    RecAdv = E_{s~d*} E_{S0~Ppert(.|s)} [ V^{pi*}(S0) - V^{pi0}(S0) ].

    This is invariant to potential-based shaping because both value functions
    shift by the same -Phi(S0) term.

    Args:
      perturb_kernel: matrix Ppert[s, s0] = P(S0=s0 | s)
      pi_baseline: deterministic [S] or stochastic [S,A] baseline policy pi0
    """
    _, _, pi_star = value_iteration(mdp)
    d_star = discounted_occupancy(mdp, pi_star)

    Ppert = np.asarray(perturb_kernel, dtype=float)
    assert Ppert.shape == (mdp.S, mdp.S)
    Ppert = Ppert / Ppert.sum(axis=1, keepdims=True)

    V_star = policy_evaluation(mdp, pi_star)
    V0 = policy_evaluation(mdp, pi_baseline)

    # s ~ d*, S0 ~ Ppert(.|s) => distribution over S0 is d_star @ Ppert
    dist_S0 = d_star @ Ppert  # shape [S]
    return float(dist_S0 @ (V_star - V0))

def planning_pressure(mdp: FiniteMDP, h: int):
    """
    Invariant planning pressure using depth-h lookahead with terminal V*:

      pi_h = argmax of E[ sum_{t=0}^{h-1} gamma^t r_t + gamma^h V*(S_h) ]

      P_h = E_{s~d0}[ V*(s) - V^{pi_h}(s) ].

    This fixes the non-invariance of truncating with terminal 0.
    """
    V_star, _, _ = value_iteration(mdp)
    _, pi_h = finite_horizon_lookahead_policy(mdp, h, V_terminal=V_star)
    V_pi_h = policy_evaluation(mdp, pi_h)
    return float(mdp.d0 @ (V_star - V_pi_h))

# ----------------------------
# 8) Example environment: simple gridworld
# ----------------------------
def gridworld(width, height, goal_xy, step_cost=-1.0, goal_reward=0.0, slip=0.0, gamma=0.99):
    """
    4-action gridworld with optional slip.
    Actions: 0=up,1=right,2=down,3=left
    Goal is absorbing (stays in goal).
    Reward: step_cost each move; entering goal adds goal_reward (if s' is goal).
    """
    S = width * height
    A = 4

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
                R[s, a, goal] = 0.0
                continue

            dx, dy = moves[a]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                ns_intended = s
            else:
                ns_intended = idx(nx, ny)

            # slip: with prob slip choose random action uniformly
            for a2 in range(A):
                prob = (slip / A)
                dx2, dy2 = moves[a2]
                nx2, ny2 = x + dx2, y + dy2
                if not (0 <= nx2 < width and 0 <= ny2 < height):
                    ns2 = s
                else:
                    ns2 = idx(nx2, ny2)
                T[s, a, ns2] += prob
                R[s, a, ns2] += prob * (step_cost + (goal_reward if ns2 == goal else 0.0))

            # intended part
            T[s, a, ns_intended] += (1.0 - slip)
            R[s, a, ns_intended] += (1.0 - slip) * (step_cost + (goal_reward if ns_intended == goal else 0.0))

    d0 = np.zeros(S)
    d0[idx(0, 0)] = 1.0
    return FiniteMDP(T, R, gamma, start_dist=d0)

# ----------------------------
# 9) Demo run
# ----------------------------
if __name__ == "__main__":
    mdp = gridworld(5, 5, goal_xy=(4, 4), step_cost=-1.0, goal_reward=0.0, slip=0.1, gamma=0.99)

    # Baseline policy: uniform random
    pi0 = np.ones((mdp.S, mdp.A)) / mdp.A

    # Perturbation: with prob p teleport uniformly; else stay
    p = 0.2
    Ppert = (p / mdp.S) * np.ones((mdp.S, mdp.S)) + (1.0 - p) * np.eye(mdp.S)

    Actrl = control_advantage(mdp, pi0)
    R1 = one_step_recovery(mdp, Ppert, pi0)
    Ph = planning_pressure(mdp, h=3)

    print("A_ctrl:", Actrl)
    print("R1:", R1)
    print("P_h (h=3):", Ph)

    # Potential shaping test
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=mdp.S)
    mdp2 = add_potential_shaping(mdp, Phi)

    Actrl2 = control_advantage(mdp2, pi0)
    R1_2 = one_step_recovery(mdp2, Ppert, pi0)
    Ph2 = planning_pressure(mdp2, h=3)

    print("A_ctrl (shaped):", Actrl2)
    print("R1 (shaped):", R1_2)
    print("P_h (shaped, h=3):", Ph2)
    print("Differences:", Actrl2 - Actrl, R1_2 - R1, Ph2 - Ph)