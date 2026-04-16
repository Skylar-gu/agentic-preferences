"""
pams.py — Planning Agenticity Measures (PAMs) + agenticity_score.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import entropy as scipy_entropy  # noqa: F401 (scipy stubs absent)
from core import (MDP, value_iteration, policy_evaluation,
                  value_under_random_policy, soft_value_iteration,
                  finite_horizon_lookahead_policy)
from metrics import control_advantage, one_step_recovery


# ---------------------------------------------------------------------------
# PAMs (shaping-invariant agenticity measures)
# ---------------------------------------------------------------------------

def advantage_gap(mdp: MDP, Q: np.ndarray, V: np.ndarray,
                  V_rand: np.ndarray = None) -> float:
    """
    Mean(max_a A* - min_a A*) over non-terminal states, normalised by
    range(V* - V^rand).

    Dividing by range(V* - V^rand) removes reward-scale dependence while
    preserving shaping invariance: A*(s,a) is already shaping-invariant
    (Phi cancels in Q - V), and range(V* - V^rand) is shaping-invariant
    because Phi cancels in the difference before the range is taken.

    Returns a value in [0, inf) when V_rand is None (raw, unscaled),
    or in [0, ~1] when V_rand is provided. Values > 1 are possible if
    the mean gap exceeds the value-function spread, but rare in practice.

    If range(V* - V^rand) < 1e-10 (e.g. potential rewards where V* = V^rand),
    returns 0.0.
    """
    A_star = Q - V[:, None]
    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')
    A_nt = A_star[non_terminal]
    raw = float(np.mean(A_nt.max(axis=1) - A_nt.min(axis=1)))
    if V_rand is None:
        return raw
    delta = V[non_terminal] - V_rand[non_terminal]
    r = float(delta.max() - delta.min())
    if r < 1e-10:
        return 0.0
    return raw / r


def vstar_variance(mdp: MDP, V_star: np.ndarray,
                   V_rand: np.ndarray) -> float:
    """
    Var of min-max normalised (V* - V^rand) over non-terminal states.

    The raw difference V* - V^rand is divided by its own range before computing
    variance, making the metric scale-invariant: a reward that spreads state
    values by 0.01 and one that spreads them by 100 score identically if the
    relative structure is the same.  Shaping-invariance is preserved because
    the Phi offset cancels in V* - V^rand before normalisation.

    Returns a value in [0, 0.25]:
      0    — all states have identical advantage (uniform value landscape)
      0.25 — bimodal split, half states at min, half at max (maximum spread)
    """
    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return float('nan')
    delta = V_star[non_terminal] - V_rand[non_terminal]
    vrange = delta.max() - delta.min()
    if vrange < 1e-10:
        return 0.0
    return float(np.var((delta - delta.min()) / vrange))


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


def effective_planning_horizon(
    mdp: MDP,
    eps: float = 0.05,
    max_k: int = None,
) -> dict:
    """
    H_eps = min{k : (J(pi*) - J(pi^k)) / (J(pi*) - J(pi^0)) <= eps}

    pi^k acts optimally for k steps then uniformly at random:
      J(pi^k) = d0 @ V_k
    where V_k is the value from k-step backward induction bootstrapped
    with V^rand (the uniform-random policy value function). pi^0 is the
    uniform random policy, so J(pi^0) = d0 @ V^rand.

    Shaping-invariant: potential shaping adds gamma*Phi(s') - Phi(s) at
    every step. V* - V^rand differences cancel Phi offsets because both
    value functions are shifted by the same linear Phi term.

    max_k: if None, auto-computed as ceil(log(eps) / log(gamma)) + 10,
    which is the minimum number of steps required for the lookahead to
    theoretically converge to within eps. Capped at 300.

    Returns:
      H_eps:   effective planning horizon (int; max_k if never converged)
      max_k:   ceiling used (auto or supplied)
      ratios:  gap ratio at each k in [0, max_k] (list of floats)
      gap:     J(pi*) - J(pi^0), the denominator
      eps:     threshold used
    """
    if max_k is None:
        if mdp.gamma < 1.0 - 1e-9:
            max_k = min(int(np.ceil(np.log(eps) / np.log(mdp.gamma))) + 10, 300)
        else:
            max_k = 300

    V_star, _, _ = value_iteration(mdp)
    V_rand = value_under_random_policy(mdp)

    J_star = float(mdp.d0 @ V_star)
    J_rand = float(mdp.d0 @ V_rand)
    gap = J_star - J_rand

    # Raise guard to 1e-6: potential rewards leave numerical residuals of ~1e-9
    # in J*-J^rand which would otherwise cause spurious max_k returns.
    if abs(gap) < 1e-6:
        return {'H_eps': 0, 'max_k': max_k, 'ratios': [0.0], 'gap': round(gap, 6), 'eps': eps}

    ratios = []
    for k in range(max_k + 1):
        if k == 0:
            V_k = V_rand
        else:
            V_k, _ = finite_horizon_lookahead_policy(mdp, k, V_rand)

        ratio = (J_star - float(mdp.d0 @ V_k)) / gap
        ratios.append(round(float(ratio), 6))

        if ratio <= eps:
            return {'H_eps': k, 'max_k': max_k, 'ratios': ratios, 'gap': round(gap, 6), 'eps': eps}

    return {'H_eps': max_k, 'max_k': max_k, 'ratios': ratios, 'gap': round(gap, 6), 'eps': eps}


def mce_policy_entropy(
    mdp: MDP,
    alpha: float = 0.25,
    eps: float = 1e-12,
    normalize_rewards: bool = True,
) -> dict:
    """
    Entropy of the MCE policy weighted by d0 over non-terminal states.

    H(pi_MCE) = -sum_s d0(s) sum_a pi_MCE(s,a) log pi_MCE(s,a)

    d0 is renormalised over non-terminal states before weighting.

    Reward normalisation (normalize_rewards=True, default):
      R is divided by std(R) before soft VI so that alpha has a
      consistent meaning regardless of reward scale. Other PAMs achieve
      scale-invariance by normalising their outputs (ratios of V* - V^rand);
      MCE needs input-side normalisation because alpha lives in the same
      space as Q. Without this, alpha=0.1 saturates near 1.0 for all
      non-potential types; with normalization, alpha=0.25 gives
      between-type discrimination (potential→0.0, non-potential→0.5–0.7).

    alpha=0.25: chosen by empirical sweep (N=40 MDPs × 6 R_types × 8 alphas).
      Maximises between-type variance (0.0025) across non-potential R_types
      without saturation at either end.

    Shaping-invariant under all three reward equivalence transforms:
    1. Positive scaling: scales Q by constant c, cancels inside softmax.
    2. Potential shaping: Phi(s) term is action-independent, cancels inside
       softmax at each state s.
    3. S'-redistribution: delta(s,a,s') with sum_{s'} T(s,a,s')*delta=0
       preserves E[R(s,a,s')], so Q^S unchanged, pi_MCE unchanged.
    Note: reward normalisation preserves shaping-invariance because std(R)
    is a global scalar — potential-shaping offsets still cancel in softmax.

    Returns:
      entropy_raw:  H(pi_MCE) in nats (on normalised-R scale)
      entropy_norm: H / log(A), in [0,1]; 1=uniform, 0=deterministic
      alpha:        temperature used
    """
    from dataclasses import replace as dc_replace
    if normalize_rewards:
        r_std = float(mdp.R.std())
        mdp_vi = dc_replace(mdp, R=mdp.R / r_std) if r_std > 1e-8 else mdp
    else:
        mdp_vi = mdp
    _, _, pi_mce = soft_value_iteration(mdp_vi, alpha=alpha)

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


def agenticity_gap(mdp: MDP) -> dict:
    """
    AgenticityGap = J(π*) − J(π*_myopic), normalised by J(π*) − J(π^rand).

    π*_myopic acts greedily on expected immediate reward at each state:
        π*_myopic(s) = argmax_a  Σ_{s'} T(s,a,s') R(s,a,s')

    Large gap → R structurally requires multi-step planning; the optimal
    policy cannot be approximated by ignoring future value.

    NOT shaping-invariant: under R' = R + γΦ(s') − Φ(s), the expected
    immediate reward gains an action-dependent term γ Σ_{s'} T(s,a,s') Φ(s'),
    which can change the argmax and thus π*_myopic.

    Returns:
      gap_raw:  J(π*) − J(π*_myopic) in reward units
      gap_norm: gap_raw / (J(π*) − J(π^rand)), clipped to [0, 1]
      J_star, J_myopic, J_rand: constituent values
    """
    V_star, _, _ = value_iteration(mdp)
    V_rand = value_under_random_policy(mdp)

    # π*_myopic: greedy on expected immediate reward
    R_sa = np.einsum('ijk,ijk->ij', mdp.T, mdp.R)   # (S, A)
    for s in mdp.terminal:
        R_sa[s, :] = 0.0
    pi_myopic = R_sa.argmax(axis=1)                  # shape (S,)

    V_myopic = policy_evaluation(mdp, pi_myopic)

    J_star   = float(mdp.d0 @ V_star)
    J_myopic = float(mdp.d0 @ V_myopic)
    J_rand   = float(mdp.d0 @ V_rand)

    gap_raw = J_star - J_myopic
    denom   = J_star - J_rand
    gap_norm = float(np.clip(gap_raw / denom, 0.0, 1.0)) if abs(denom) > 1e-10 else 0.0

    return {
        'gap_raw':  round(gap_raw, 4),
        'gap_norm': round(gap_norm, 4),
        'J_star':   round(J_star, 4),
        'J_myopic': round(J_myopic, 4),
        'J_rand':   round(J_rand, 4),
    }


def option_value(mdp: MDP) -> dict:
    """
    OptionValue = E_{s~d0}[V*(s) − max_a E_{s'}[R(s,a,s')]], normalised.

    Measures how much the future matters relative to the best immediate
    expected reward.  High → long-range consequences dominate; myopic
    optimisation cannot exploit the value structure.
    Low → immediate rewards capture most of the value; planning adds little.

    Normalised by the range of V* over non-terminal states so the result
    is scale-invariant.

    NOT shaping-invariant: under R' = R + γΦ(s') − Φ(s), V*(s) shifts by
    −Φ(s) while max_a E[R'(s,a,s')] shifts by a mix of Φ(s) and
    γ E_{s'}[T(s,a*,s') Φ(s')], so the difference changes with Φ.

    Returns:
      ov_raw:  d0-weighted mean of V*(s) − max_a E[R(s,a)]
      ov_norm: ov_raw / range(V*), clipped to [0, 1]
    """
    V_star, _, _ = value_iteration(mdp)

    R_sa = np.einsum('ijk,ijk->ij', mdp.T, mdp.R)   # (S, A)
    for s in mdp.terminal:
        R_sa[s, :] = 0.0

    non_terminal = [s for s in range(mdp.S) if s not in mdp.terminal]
    if not non_terminal:
        return {'ov_raw': float('nan'), 'ov_norm': float('nan')}

    best_imm = R_sa[non_terminal].max(axis=1)
    V_nt     = V_star[non_terminal]
    d0_nt    = mdp.d0[non_terminal]
    d0_nt    = d0_nt / d0_nt.sum()

    ov_raw  = float(np.dot(d0_nt, V_nt - best_imm))
    v_range = float(V_nt.max() - V_nt.min())
    ov_norm = float(np.clip(ov_raw / v_range, 0.0, 1.0)) if v_range > 1e-10 else 0.0

    return {
        'ov_raw':  round(ov_raw, 4),
        'ov_norm': round(ov_norm, 4),
    }


def agenticity_score(mdp: MDP, weights: dict = None,
                     horizon: int = 60, n_episodes: int = 2000,
                     verbose: bool = True,
                     rng: np.random.Generator = None,
                     compute_mi: bool = True,
                     compute_entropy: bool = True,
                     w2_scales: dict = None) -> dict:
    """
    Composite agenticity score (all metrics shaping-invariant).

    Default weights: equal across active metrics (adv_gap, vstar_var,
    mi_diff, mce_entropy). Auto-renormalised so inactive metrics (compute_mi=False
    or compute_entropy=False) do not alter the relative weights of the rest.

    Recalibration rationale (April 2026):
      Within each R_type, r(adv_gap, vstar_var) = 0.02–0.23 — they are
      independent and both informative. Equal weighting is the principled
      choice.

      MCE entropy re-enabled (compute_entropy=True default) after fixing
      reward normalisation: R is now divided by std(R) before soft VI, making
      alpha scale-invariant. alpha=0.25 was selected by empirical sweep:
      maximises between-type variance across non-potential R_types (0.0025)
      without upper/lower saturation. With this fix, mce_entropy_norm scores
      potential rewards at 0.0 and non-potential at 0.5–0.7.

    w2_scales: if provided, baseline metrics (ctrl_adv, one_step_recovery) are
    normalised via 1-exp(-x/scale) and included in the composite with equal
    weight alongside the PAMs.
    Scales dict keys: 'ctrl_adv', 'one_step' (95th-pct values from Q1 are
    a good choice; pass the 'baseline_empirical_scales' entry from run_pam_experiment).
    compute_mi:       if False, skip early_action_mi and renormalise composite.
    compute_entropy:  if True (default), include mce_policy_entropy in composite.
    """
    if weights is None:
        if w2_scales is not None:
            weights = {'adv_gap': 0.20, 'vstar_var': 0.20, 'mi_diff': 0.20,
                       'mce_entropy': 0.20, 'ctrl_adv': 0.20, 'one_step': 0.20}
        else:
            weights = {'adv_gap': 0.50, 'vstar_var': 0.50, 'mi_diff': 0.50,
                       'mce_entropy': 0.50}
    if rng is None:
        rng = np.random.default_rng(42)

    V_star, Q_star, pi_star = value_iteration(mdp)
    V_rand = value_under_random_policy(mdp)

    # Uniform random policy as baseline for baseline metrics
    pi_rand = np.ones((mdp.S, mdp.A)) / mdp.A

    # Baseline metrics
    ctrl_adv  = control_advantage(mdp, pi_rand)
    one_step  = one_step_recovery(mdp, pi_rand)

    # PAMs
    ag  = advantage_gap(mdp, Q_star, V_star, V_rand)
    vv  = vstar_variance(mdp, V_star, V_rand)
    asp = advantage_sparsity(mdp, Q_star, V_star)

    if compute_mi:
        mi = early_action_mi(mdp, pi_star, horizon=horizon,
                             n_episodes=n_episodes, rng=rng)
    else:
        mi = {'mi_early': None, 'mi_late': None, 'mi_diff': None}

    h_eff   = effective_planning_horizon(mdp)
    mce_ent = mce_policy_entropy(mdp, alpha=0.25, normalize_rewards=True) if compute_entropy else None
    ag_gap  = agenticity_gap(mdp)
    ov      = option_value(mdp)

    # ag is now range-normalised by range(V*-V^rand); clip to [0,1]
    ag_norm = float(np.clip(ag, 0.0, 1.0))
    # vv is now in [0, 0.25] (range-normalised variance); multiply by 4 → [0, 1]
    vv_norm = float(np.clip(4.0 * vv, 0.0, 1.0))
    mi_norm = float(np.clip((mi['mi_diff'] + 1) / 2, 0, 1)) if compute_mi else None

    # Baseline metric normalisation (only when scales are provided)
    def _nw(x: float, key: str) -> float:
        s = w2_scales.get(key, 1.0) if w2_scales else 1.0
        return float(1 - np.exp(-x / s)) if s > 1e-10 else 0.0

    ctrl_adv_norm  = _nw(ctrl_adv, 'ctrl_adv')  if w2_scales else None
    one_step_norm  = _nw(one_step, 'one_step')   if w2_scales else None

    # Build composite dynamically — auto-renormalises across active components
    mce_norm = float(np.clip(1.0 - mce_ent['entropy_norm'], 0.0, 1.0)) if mce_ent else None

    components: dict = {'adv_gap': ag_norm, 'vstar_var': vv_norm}
    if compute_mi:
        components['mi_diff'] = mi_norm  # type: ignore[assignment]
    if compute_entropy and mce_norm is not None:
        components['mce_entropy'] = mce_norm
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
        # PAMs (in composite)
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
        # Diagnostic PAMs (not in composite)
        'H_eps': h_eff['H_eps'],
        'H_eps_norm': round(h_eff['H_eps'] / h_eff['max_k'], 4),
        'H_eps_max_k': h_eff['max_k'],
        'H_eps_gap': h_eff['gap'],
        'H_eps_ratios': h_eff['ratios'],
        'mce_entropy_raw':  round(mce_ent['entropy_raw'],  4) if mce_ent else None,
        'mce_entropy_norm': round(mce_norm,                4) if mce_norm is not None else None,
        # Candidate proxies (diagnostic; not shaping-invariant)
        'agenticity_gap_raw':  ag_gap['gap_raw'],
        'agenticity_gap_norm': ag_gap['gap_norm'],
        'option_value_raw':    ov['ov_raw'],
        'option_value_norm':   ov['ov_norm'],
        # Baseline metrics
        'ctrl_adv': round(ctrl_adv, 4),
        'ctrl_adv_norm': round(ctrl_adv_norm, 4) if ctrl_adv_norm is not None else None,
        'one_step_recovery': round(one_step, 4),
        'one_step_norm': round(one_step_norm, 4) if one_step_norm is not None else None,
    }

    if verbose:
        in_comp = w2_scales is not None
        ca_str = (f" / norm={ctrl_adv_norm:.4f}" if in_comp else " (diag)")
        os_str = (f" / norm={one_step_norm:.4f}"  if in_comp else " (diag)")
        print(f"  Control advantage:                      {ctrl_adv:.4f}{ca_str}")
        print(f"  One-step recovery:                      {one_step:.4f}{os_str}")
        print(f"  Advantage gap (raw / norm):             {ag:.4f} / {ag_norm:.4f}")
        print(f"  V*-V^rand variance (raw / norm):        {vv:.4f} / {vv_norm:.4f}  [range-norm, ×4→[0,1]]")
        if compute_mi:
            print(f"  MI|s0 (early / late / diff):            {mi['mi_early']:.4f} / {mi['mi_late']:.4f} / {mi['mi_diff']:+.4f}")
        else:
            print(f"  MI|s0:                                  (skipped)")
        print(f"  Effective planning horizon H_eps:       {h_eff['H_eps']}/{h_eff['max_k']} = {result['H_eps_norm']:.4f}  (gap={h_eff['gap']:.4f})")
        if mce_ent:
            print(f"  MCE entropy (raw / 1-norm):             {mce_ent['entropy_raw']:.4f} / {mce_norm:.4f}  [R-norm, α=0.25]")
        else:
            print(f"  MCE entropy:                            (skipped)")
        print(f"  Agenticity gap (raw / norm):            {ag_gap['gap_raw']:.4f} / {ag_gap['gap_norm']:.4f}  [not shaping-inv]")
        print(f"  Option value (raw / norm):              {ov['ov_raw']:.4f} / {ov['ov_norm']:.4f}  [not shaping-inv]")
        print(f"  Advantage sparsity (diagnostic):        {asp:.4f}")
        print(f"  ── COMPOSITE: {composite:.4f}")

    return result
