"""
week3.py — Week 3/4 PAMs + agenticity_score.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import entropy as scipy_entropy  # noqa: F401 (scipy stubs absent)
from core import (MDP, value_iteration,
                  value_under_random_policy, soft_value_iteration)
from week2 import control_advantage, one_step_recovery, planning_pressure


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


# ---------------------------------------------------------------------------
# New PAMs (Week 4)
# ---------------------------------------------------------------------------

def effective_time_horizon(
    mdp: MDP,
    pi: np.ndarray,
    horizon: int = 200,
    n_episodes: int = 1000,
    rng: np.random.Generator = None,
) -> float:
    """
    Reward-weighted centre-of-mass across timesteps under pi from d0.

    H_eff = sum_t (t * mass[t]) / sum_t mass[t]

    mass[t] = sum over episodes of gamma^t * |r_t| at step t.
    Discounting is baked in so the measure respects the agent's time preference.

    NOT shaping-invariant: potential shaping F = gamma*Phi(s') - Phi(s)
    adds a nonzero term at every step, spreading reward mass across all
    timesteps and shifting H_eff.

    Returns 0.0 if total reward mass < 1e-10 (zero-reward MDP).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mass = np.zeros(horizon)

    for _ in range(n_episodes):
        s = rng.choice(mdp.S, p=mdp.d0)
        discount = 1.0
        for t in range(horizon):
            if s in mdp.terminal:
                break
            a = int(pi[s])
            s_next = rng.choice(mdp.S, p=mdp.T[s, a, :])
            r = mdp.R[s, a, s_next]
            mass[t] += discount * abs(r)
            discount *= mdp.gamma
            s = s_next

    total = mass.sum()
    if total < 1e-10:
        return 0.0
    t_idx = np.arange(horizon, dtype=float)
    return float(np.dot(t_idx, mass) / total)


def mce_policy_entropy(
    mdp: MDP,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> dict:
    """
    Entropy of the MCE policy weighted by d0 over non-terminal states.

    H(pi_MCE) = -sum_s d0(s) sum_a pi_MCE(s,a) log pi_MCE(s,a)

    d0 is renormalised over non-terminal states before weighting.

    Shaping-invariant under all three reward equivalence transforms:
    1. Positive scaling: scales Q by constant c, cancels inside softmax.
    2. Potential shaping: Phi(s) term is action-independent, cancels inside
       softmax at each state s.
    3. S'-redistribution: delta(s,a,s') with sum_{s'} T(s,a,s')*delta=0
       preserves E[R(s,a,s')], so Q^S unchanged, pi_MCE unchanged.

    Returns:
      entropy_raw:  H(pi_MCE) in nats
      entropy_norm: H / log(A), in [0,1]; 1=uniform, 0=deterministic
      alpha:        temperature used
    """
    _, _, pi_mce = soft_value_iteration(mdp, alpha=alpha)

    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return {'entropy_raw': float('nan'), 'entropy_norm': float('nan'), 'alpha': alpha}

    H_s = -np.sum(pi_mce * np.log(pi_mce + eps), axis=1)

    d0_nt = mdp.d0[non_terminal]
    d0_nt = d0_nt / d0_nt.sum()

    entropy_raw = float(np.dot(d0_nt, H_s[non_terminal]))
    entropy_norm = float(entropy_raw / np.log(mdp.A))

    return {
        'entropy_raw': entropy_raw,
        'entropy_norm': entropy_norm,
        'alpha': alpha,
    }


def agenticity_score(mdp: MDP, weights: dict = None,
                     horizon: int = 60, n_episodes: int = 2000,
                     verbose: bool = True,
                     rng: np.random.Generator = None,
                     compute_mi: bool = True,
                     pp_horizon: int = 3,
                     w2_scales: dict = None) -> dict:
    """
    Composite agenticity (all shaping-invariant).

    Default weights (W3 only): adv_gap=0.25, vstar_var=0.40, mi_diff=0.35.

    w2_scales: if provided, W2 metrics (ctrl_adv, one_step_recovery) are
    normalised via 1-exp(-x/scale) and included in the composite.
    Default weights when w2_scales is given:
      adv_gap=0.20, vstar_var=0.30, mi_diff=0.25, ctrl_adv=0.15, one_step=0.10.
    Scales dict keys: 'ctrl_adv', 'one_step' (95th-pct values from Q1 are
    a good choice; pass the 'w2_empirical_scales' entry from run_pam_experiment).
    planning_pressure is excluded from the composite (always near 0).

    compute_mi:  if False, skip early_action_mi and renormalise composite.
    pp_horizon:  lookahead depth h for planning_pressure (default 3).
    """
    if weights is None:
        if w2_scales is not None:
            weights = {'adv_gap': 0.20, 'vstar_var': 0.30, 'mi_diff': 0.25,
                       'ctrl_adv': 0.15, 'one_step': 0.10}
        else:
            weights = {'adv_gap': 0.25, 'vstar_var': 0.40, 'mi_diff': 0.35}
    if rng is None:
        rng = np.random.default_rng(42)

    V_star, Q_star, pi_star = value_iteration(mdp)
    V_rand = value_under_random_policy(mdp)

    # Uniform random policy as baseline for Week 2 metrics
    pi_rand = np.ones((mdp.S, mdp.A)) / mdp.A

    # Week 2 metrics
    ctrl_adv  = control_advantage(mdp, pi_rand)
    one_step  = one_step_recovery(mdp, pi_rand)
    plan_pres = planning_pressure(mdp, h=pp_horizon)

    # Week 3 proxies
    ag  = advantage_gap(mdp, Q_star, V_star)
    vv  = vstar_variance_corrected(mdp, V_star, V_rand)
    asp = advantage_sparsity(mdp, Q_star, V_star)

    if compute_mi:
        mi = early_action_mi(mdp, pi_star, horizon=horizon,
                             n_episodes=n_episodes, rng=rng)
    else:
        mi = {'mi_early': None, 'mi_late': None, 'mi_diff': None}

    h_eff   = effective_time_horizon(mdp, pi_star, horizon=horizon,
                                     n_episodes=n_episodes, rng=rng)
    mce_ent = mce_policy_entropy(mdp, alpha=1.0)

    ag_norm = float(1 - np.exp(-ag * 1.0))
    vv_norm = float(1 - np.exp(-vv / 2.0))
    mi_norm = float(np.clip((mi['mi_diff'] + 1) / 2, 0, 1)) if compute_mi else None

    # W2 normalisation (only when scales are provided)
    def _nw(x: float, key: str) -> float:
        s = w2_scales.get(key, 1.0) if w2_scales else 1.0
        return float(1 - np.exp(-x / s)) if s > 1e-10 else 0.0

    ctrl_adv_norm  = _nw(ctrl_adv, 'ctrl_adv')  if w2_scales else None
    one_step_norm  = _nw(one_step, 'one_step')   if w2_scales else None

    # Build composite dynamically — auto-renormalises across active components
    components: dict = {'adv_gap': ag_norm, 'vstar_var': vv_norm}
    if compute_mi:
        components['mi_diff'] = mi_norm  # type: ignore[assignment]
    if w2_scales is not None:
        components['ctrl_adv'] = ctrl_adv_norm   # type: ignore[assignment]
        components['one_step']  = one_step_norm  # type: ignore[assignment]

    w_total = sum(weights.get(k, 0.0) for k in components)
    if w_total < 1e-10:
        composite = 0.0
    else:
        composite = sum(weights.get(k, 0.0) * v
                        for k, v in components.items()) / w_total

    result = {
        # Week 3 proxies (in composite)
        'adv_gap': round(ag, 4),
        'adv_gap_norm': round(ag_norm, 4),
        'vstar_var_raw': round(vv, 4),
        'vstar_var_norm': round(vv_norm, 4),
        'mi_early': mi['mi_early'],
        'mi_late':  mi['mi_late'],
        'mi_diff':  mi['mi_diff'],
        'mi_diff_norm': round(mi_norm, 4) if mi_norm is not None else None,
        'adv_sparsity': round(asp, 4),
        'composite': round(composite, 4),
        # Week 4 PAMs (diagnostic, not in composite)
        'h_eff_raw': round(h_eff, 4),
        'h_eff_norm': round(1 - np.exp(-h_eff / 10.0), 4),
        'mce_entropy_raw': round(mce_ent['entropy_raw'], 4),
        'mce_entropy_norm': round(mce_ent['entropy_norm'], 4),
        # Week 2 metrics
        'ctrl_adv': round(ctrl_adv, 4),
        'ctrl_adv_norm': round(ctrl_adv_norm, 4) if ctrl_adv_norm is not None else None,
        'one_step_recovery': round(one_step, 4),
        'one_step_norm': round(one_step_norm, 4) if one_step_norm is not None else None,
        'planning_pressure': round(plan_pres, 4),
    }

    if verbose:
        in_comp = w2_scales is not None
        ca_str = (f" / norm={ctrl_adv_norm:.4f}" if in_comp else " (diag)")
        os_str = (f" / norm={one_step_norm:.4f}"  if in_comp else " (diag)")
        print(f"  [W2] Control advantage:                 {ctrl_adv:.4f}{ca_str}")
        print(f"  [W2] One-step recovery:                 {one_step:.4f}{os_str}")
        print(f"  [W2] Planning pressure (h={pp_horizon}):          {plan_pres:.4f} (diag)")
        print(f"  Advantage gap (raw / norm):             {ag:.4f} / {ag_norm:.4f}")
        print(f"  V*-V^rand variance (raw / norm):        {vv:.4f} / {vv_norm:.4f}")
        if compute_mi:
            print(f"  MI|s0 (early / late / diff):            {mi['mi_early']:.4f} / {mi['mi_late']:.4f} / {mi['mi_diff']:+.4f}")
        else:
            print(f"  MI|s0:                                  (skipped)")
        print(f"  Effective time horizon (raw / norm):    {h_eff:.4f} / {result['h_eff_norm']:.4f}")
        print(f"  MCE entropy (raw / norm):               {mce_ent['entropy_raw']:.4f} / {mce_ent['entropy_norm']:.4f}")
        print(f"  Advantage sparsity (diagnostic):        {asp:.4f}")
        print(f"  ── COMPOSITE: {composite:.4f}")

    return result
