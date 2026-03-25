"""
experiments.py — Batch experiment machinery.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core import MDP
from week3 import agenticity_score
from envs import make_chain_mdp, make_grid_mdp


# ---------------------------------------------------------------------------
# Random MDP Generator
# ---------------------------------------------------------------------------

def random_mdp(
    S: int,
    A: int,
    gamma: float = 0.95,
    k: int = 3,
    R_type: str = 'gaussian',
    R_scale: float = 1.0,
    terminal_states: int = 1,
    rng: np.random.Generator = None,
) -> MDP:
    """
    Generate a random tabular MDP.

    Transition: for each (s,a) pair, sample k successor states uniformly
    without replacement from non-terminal states, assign Dirichlet(1,...,1)
    weights. Terminal states are absorbing (T[s,a,s]=1).

    Reward types:
      'gaussian':  R[s,a,s'] ~ N(0, R_scale)
      'uniform':   R[s,a,s'] ~ Uniform(0, 1)
      'bernoulli': R[s,a,s'] = 1 w.p. R_scale, else 0
                   (R_scale is the sparsity; default 0.1 for sparse rewards)
      'potential': R(s,a,s') = gamma*Phi(s') - Phi(s), Phi(s) ~ N(0, R_scale).
                   Non-agentic by construction — does not change policy ordering.
                   Useful as a negative control: PAMs should score these low.
      'goal':      R(s,a,s') = 1[s'=g] for a uniformly random non-terminal goal g.
                   Goal-conditioned sparse reward; structurally realistic for
                   navigation domains. Agenticity varies with goal location.

    Terminal states: last `terminal_states` indices {S-terminal_states, ..., S-1}.
    d0: uniform over non-terminal states.
    """
    if rng is None:
        rng = np.random.default_rng()

    terminal = set(range(S - terminal_states, S)) if terminal_states > 0 else set()
    non_terminal = [s for s in range(S) if s not in terminal]

    T = np.zeros((S, A, S))

    for s in range(S):
        for a in range(A):
            if s in terminal:
                T[s, a, s] = 1.0
                continue
            pool = non_terminal if non_terminal else list(range(S))
            k_eff = min(k, len(pool))
            successors = rng.choice(pool, size=k_eff, replace=False)
            weights = rng.dirichlet(np.ones(k_eff))
            T[s, a, successors] = weights

    if R_type == 'gaussian':
        R = rng.normal(0, R_scale, size=(S, A, S))
    elif R_type == 'uniform':
        R = rng.uniform(0, 1, size=(S, A, S))
    elif R_type == 'bernoulli':
        R = (rng.uniform(0, 1, size=(S, A, S)) < R_scale).astype(float)
    elif R_type == 'potential':
        Phi = rng.normal(0, R_scale, size=S)
        # F[s,a,s'] = gamma*Phi[s'] - Phi[s]; action-independent shaping
        R = gamma * Phi[None, None, :] - Phi[:, None, None]
        R = np.broadcast_to(R, (S, A, S)).copy()
    elif R_type == 'goal':
        pool = non_terminal if non_terminal else list(range(S))
        g = int(rng.choice(pool))
        R = np.zeros((S, A, S))
        R[:, :, g] = 1.0
    else:
        raise ValueError(
            f"Unknown R_type: {R_type!r}. "
            "Choose 'gaussian', 'uniform', 'bernoulli', 'potential', or 'goal'."
        )

    for s in terminal:
        R[s, :, :] = 0.0

    d0 = np.zeros(S)
    if non_terminal:
        d0[non_terminal] = 1.0 / len(non_terminal)
    else:
        d0[:] = 1.0 / S

    return MDP(S=S, A=A, T=T, R=R, gamma=gamma, terminal=terminal, d0=d0)


# ---------------------------------------------------------------------------
# W2 normalization utility
# ---------------------------------------------------------------------------

def norm_w2(ctrl_adv, one_step, plan_press, scales):
    """Normalize W2 metrics using empirical 95th-pct scales: 1-exp(-x/scale)."""
    def _n(x, s):
        return float(1 - np.exp(-x / s)) if s > 1e-10 else 0.0
    return {
        'ctrl_adv_norm':   _n(ctrl_adv,  scales.get('ctrl_adv', 1.0)),
        'one_step_norm':   _n(one_step,  scales.get('one_step', 1.0)),
        'plan_press_norm': _n(plan_press, scales.get('plan_press', 1.0)),
    }


# ---------------------------------------------------------------------------
# Batch PAM Experiment (Q1/Q2/Q3)
# ---------------------------------------------------------------------------

def run_pam_experiment(
    n_random_mdps: int = 200,
    S_values: list = None,
    A: int = 4,
    gamma: float = 0.95,
    k: int = 3,
    R_types: list = None,
    R_scales: dict = None,
    T_structures: list = None,
    n_fixed_T: int = 10,
    rng_seed: int = 42,
    verbose: bool = False,
    include_mi: bool = False,
) -> dict:
    """
    Batch experiment comparing PAMs across random and human-made MDPs.

    Q1 — PAM correlation (T fixed, R varies):
      For each (S, R_type): fix one canonical T, sample n_random_mdps R's,
      compute all PAMs. q1 stores norm PAM matrix; q1_raw stores raw values.

    Q2 — T-dependence (variance decomposition):
      For each (S, R_type, T_structure): sample n_fixed_T transition matrices
      with the given topology, for each T sample n_R=50 R's and compute
      composite. Reports within-T variance (R's share) and between-T variance
      (T's share). T_structures can include 'random', 'chain', 'grid'.

    Q3 — Human-made vs random:
      Run agenticity_score on the 7 built-in MDPs. Results include n_actions
      so the MCE entropy |A| confound is visible in the printout.

    R_scales: per-type R_scale overrides, e.g. {'bernoulli': 0.5}.
              Default: {'bernoulli': 0.5} to avoid all-zero reward matrices.
    T_structures: list of topology names for Q2. Default: ['random'].
                  'chain' = linear chain (A0=advance, A1=retreat, A2=stay, A3=stochastic)
                  'grid'  = 2D grid with 4-directional actions
    include_mi: if True, compute early_action_mi (slow, n_episodes reduced to 300).
    """
    if S_values is None:
        S_values = [5, 10, 20]
    if R_types is None:
        R_types = ['gaussian', 'uniform', 'bernoulli', 'potential', 'goal']
    if R_scales is None:
        R_scales = {'bernoulli': 0.5}
    if T_structures is None:
        T_structures = ['random']

    rng = np.random.default_rng(rng_seed)
    PAM_NAMES = ['adv_gap', 'vstar_var', 'mi_diff', 'h_eff', 'mce_entropy']

    def _score_mdp(mdp, n_ep=500):
        return agenticity_score(
            mdp, verbose=False, rng=rng,
            compute_mi=include_mi,
            n_episodes=(300 if include_mi else n_ep),
        )

    def _pam_vector_norm(r):
        return np.array([
            r['adv_gap_norm'],
            r['vstar_var_norm'],
            r['mi_diff'] if r['mi_diff'] is not None else float('nan'),
            r['h_eff_norm'],
            r['mce_entropy_norm'],
        ])

    def _pam_vector_raw(r):
        return np.array([
            r['adv_gap'],
            r['vstar_var_raw'],
            r['mi_diff'] if r['mi_diff'] is not None else float('nan'),
            r['h_eff_raw'],
            r['mce_entropy_raw'],
            r['ctrl_adv'],
            r['one_step_recovery'],
            r['planning_pressure'],
        ])

    def _r_scale_for(R_type):
        return R_scales.get(R_type, 1.0)

    def _make_structured_T(S, structure, terminal, local_rng):
        """Build a topology-specific transition matrix."""
        non_term = [s for s in range(S) if s not in terminal]
        T = np.zeros((S, A, S))

        if structure == 'random':
            for s in range(S):
                for a in range(A):
                    if s in terminal:
                        T[s, a, s] = 1.0
                        continue
                    pool = non_term if non_term else list(range(S))
                    k_eff = min(k, len(pool))
                    succ = local_rng.choice(pool, size=k_eff, replace=False)
                    T[s, a, succ] = local_rng.dirichlet(np.ones(k_eff))

        elif structure == 'chain':
            # A0=advance, A1=retreat, A2=stay, A3=noisy-advance
            max_nt = max(non_term) if non_term else S - 1
            min_nt = min(non_term) if non_term else 0
            for s in range(S):
                if s in terminal:
                    T[s, :, s] = 1.0
                    continue
                s_fwd  = min(s + 1, max_nt) if (s + 1) not in terminal else s
                s_back = max(s - 1, min_nt) if (s - 1) not in terminal else s
                T[s, 0, s_fwd]  = 1.0   # advance
                T[s, 1, s_back] = 1.0   # retreat
                T[s, 2, s]      = 1.0   # stay
                # noisy advance: if at boundary s_fwd==s, just stay
                if s_fwd != s:
                    T[s, 3, s_fwd] = 0.7
                    T[s, 3, s]     = 0.3
                else:
                    T[s, 3, s] = 1.0

        elif structure == 'grid':
            rows = max(2, int(np.round(np.sqrt(S - len(terminal)))))
            cols = max(2, (S - len(terminal) + rows - 1) // rows)
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up/down/left/right
            for s in non_term:
                r_s, c_s = s // cols, s % cols
                for a, (dr, dc) in enumerate(deltas[:A]):
                    nr = max(0, min(rows - 1, r_s + dr))
                    nc = max(0, min(cols - 1, c_s + dc))
                    s_next = nr * cols + nc
                    if s_next >= S or s_next in terminal:
                        s_next = s  # boundary: stay
                    T[s, a, s_next] += 1.0
            for s in terminal:
                T[s, :, s] = 1.0
        else:
            raise ValueError(f"Unknown T_structure: {structure!r}. "
                             "Choose 'random', 'chain', or 'grid'.")
        return T

    # ------------------------------------------------------------------
    # Q1 — fix T, vary R
    # ------------------------------------------------------------------
    q1 = {}
    q1_raw = {}
    for S in S_values:
        for R_type in R_types:
            if verbose:
                print(f"Q1: S={S}, R_type={R_type}")

            canonical = random_mdp(S, A, gamma=gamma, k=k,
                                   R_type='gaussian', terminal_states=1,
                                   rng=np.random.default_rng(0))
            T_fixed = canonical.T
            terminal_fixed = canonical.terminal
            d0_fixed = canonical.d0

            pam_norm = []
            pam_raw  = []
            for _ in range(n_random_mdps):
                tmp_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                tmp = random_mdp(S, A, gamma=gamma, k=k,
                                 R_type=R_type,
                                 R_scale=_r_scale_for(R_type),
                                 terminal_states=1,
                                 rng=tmp_rng)
                tmp_mdp = MDP(S=S, A=A, T=T_fixed, R=tmp.R,
                              gamma=gamma, terminal=terminal_fixed,
                              d0=d0_fixed.copy())
                r = _score_mdp(tmp_mdp)
                pam_norm.append(_pam_vector_norm(r))
                pam_raw.append(_pam_vector_raw(r))

            q1[(S, R_type)]     = np.array(pam_norm)
            q1_raw[(S, R_type)] = np.array(pam_raw)

    # ------------------------------------------------------------------
    # Compute empirical 95th-percentile W2 scales from Q1 raw data
    # ------------------------------------------------------------------
    def _compute_w2_scales(q1_raw_dict, percentile=95):
        pools = [[], [], []]  # ctrl_adv, one_step, plan_press
        for mat in q1_raw_dict.values():
            for i in range(5, 8):
                col = mat[:, i]
                pools[i-5].extend(col[~np.isnan(col)].tolist())
        keys = ['ctrl_adv', 'one_step', 'plan_press']
        return {k: float(np.percentile(v, percentile)) if v else 1.0
                for k, v in zip(keys, pools)}

    w2_scales = _compute_w2_scales(q1_raw)

    # ------------------------------------------------------------------
    # Q2 — variance decomposition: T vs R, across T_structures
    # ------------------------------------------------------------------
    q2 = {}
    n_R = 50
    for structure in T_structures:
        for S in S_values:
            for R_type in R_types:
                if verbose:
                    print(f"Q2: structure={structure}, S={S}, R_type={R_type}")

                terminal_q2 = set(range(S - 1, S))
                non_term_q2 = [s for s in range(S) if s not in terminal_q2]
                d0_q2 = np.zeros(S)
                d0_q2[non_term_q2] = 1.0 / len(non_term_q2)

                group_means = []
                group_composites_all = []

                for _ in range(n_fixed_T):
                    t_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                    T_t = _make_structured_T(S, structure, terminal_q2, t_rng)

                    composites = []
                    for _ in range(n_R):
                        r_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                        tmp = random_mdp(S, A, gamma=gamma, k=k,
                                         R_type=R_type,
                                         R_scale=_r_scale_for(R_type),
                                         terminal_states=1,
                                         rng=r_rng)
                        mdp_t = MDP(S=S, A=A, T=T_t, R=tmp.R,
                                    gamma=gamma, terminal=terminal_q2,
                                    d0=d0_q2.copy())
                        r = _score_mdp(mdp_t)
                        composites.append(r['composite'])

                    composites = np.array(composites)
                    group_means.append(composites.mean())
                    group_composites_all.append(composites)

                group_means = np.array(group_means)
                within_var  = float(np.mean([np.var(c) for c in group_composites_all]))
                between_var = float(np.var(group_means))
                total_var   = within_var + between_var
                ratio = between_var / total_var if total_var > 1e-12 else float('nan')

                q2[(structure, S, R_type)] = {
                    'within_var':  within_var,
                    'between_var': between_var,
                    'ratio':       ratio,
                }

    # ------------------------------------------------------------------
    # Q3 — human-made MDPs
    # ------------------------------------------------------------------
    human_cases = [
        ("Chain-Terminal",  make_chain_mdp(10, reward_type='terminal')),
        ("Chain-Dense",     make_chain_mdp(10, reward_type='dense')),
        ("Chain-Lottery",   make_chain_mdp(10, reward_type='lottery')),
        ("Chain-Progress",  make_chain_mdp(10, reward_type='progress')),
        ("Grid-Goal",       make_grid_mdp(5, 5, reward_type='goal')),
        ("Grid-Local",      make_grid_mdp(5, 5, reward_type='local')),
        ("Grid-Cliff",      make_grid_mdp(5, 5, reward_type='cliff')),
    ]

    q3 = {}
    for name, mdp in human_cases:
        if verbose:
            print(f"  scoring {name}...")
        r = _score_mdp(mdp)
        q3[name] = {**r, **norm_w2(r['ctrl_adv'], r['one_step_recovery'],
                                    r['planning_pressure'], w2_scales),
                    'n_actions': mdp.A}

    meta = {
        'S_values': S_values,
        'n_random_mdps': n_random_mdps,
        'PAM_names': PAM_NAMES,
        'include_mi': include_mi,
        'rng_seed': rng_seed,
        'noisy_mi_estimates': include_mi,
        'R_scales': R_scales,
        'T_structures': T_structures,
        'w2_empirical_scales': w2_scales,
    }

    results = {'q1': q1, 'q1_raw': q1_raw, 'q2': q2, 'q3': q3, 'meta': meta}

    if verbose:
        _print_pam_results(results)

    return results


def _print_pam_results(results: dict) -> None:
    """Pretty-print Q1/Q2/Q3 results from run_pam_experiment."""
    PAM_COLS = ['adv_gap', 'vstar_var', 'mi_diff', 'h_eff', 'mce_ent']
    PAM_COLS_RAW = ['adv_gap', 'vstar_var', 'mi_diff', 'h_eff', 'mce_ent', 'ctrl_adv', 'one_step', 'plan_press']
    W = 9  # column width

    def _fmt(v, w, decimals=4):
        return f"{'N/A':>{w}s}" if (v is None or np.isnan(v)) else f"{v:>{w}.{decimals}f}"

    def _col_stats(mat):
        """Per-column mean and std, skipping NaN, without triggering warnings."""
        means = np.full(mat.shape[1], np.nan)
        stds  = np.full(mat.shape[1], np.nan)
        for i in range(mat.shape[1]):
            valid = mat[:, i][~np.isnan(mat[:, i])]
            if len(valid) > 0:
                means[i] = valid.mean()
                stds[i]  = valid.std()
        return means, stds

    # ------------------------------------------------------------------
    # Q1 — PAM distributions and correlations (norm + raw)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1 — PAM DISTRIBUTIONS (T fixed, R varies)")
    print("    norm = normalised [0,1]  |  raw = original scale")
    print("=" * 70)
    hdr = f"  {'':18s}" + "".join(f"{n:>{W}s}" for n in PAM_COLS)

    for (S, R_type), mat in sorted(results['q1'].items()):
        label = f"S={S} {R_type}"
        norm_means, norm_stds = _col_stats(mat)
        raw_mat = results['q1_raw'].get((S, R_type))

        print(f"\n  {label}")
        print(hdr)
        print(f"  {'norm mean':18s}" + "".join(_fmt(v, W) for v in norm_means))
        print(f"  {'norm std':18s}" + "".join(_fmt(v, W) for v in norm_stds))
        if raw_mat is not None:
            raw_means, raw_stds = _col_stats(raw_mat)
            # Show all 8 raw columns
            hdr_raw = f"  {'':18s}" + "".join(f"{n:>{W}s}" for n in PAM_COLS_RAW)
            print(hdr_raw)
            print(f"  {'raw mean':18s}" + "".join(_fmt(v, W) for v in raw_means))
            print(f"  {'raw std':18s}" + "".join(_fmt(v, W) for v in raw_stds))

        # Pearson correlation on norm matrix (skip all-NaN / zero-variance cols)
        valid_cols = [i for i in range(mat.shape[1])
                      if not np.isnan(norm_stds[i]) and norm_stds[i] > 1e-10]
        if len(valid_cols) >= 2:
            corr = np.corrcoef(mat[:, valid_cols].T)
            col_names = [PAM_COLS[i] for i in valid_cols]
            print(f"  {'Pearson corr':18s}[{', '.join(col_names)}]")
            for i, row in enumerate(corr):
                print(f"    {col_names[i]:<14s}" + "".join(_fmt(v, W, 3) for v in row))

    # ------------------------------------------------------------------
    # Q2 — variance decomposition (grouped by T_structure)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q2 — T vs R VARIANCE DECOMPOSITION")
    print("  ratio = between_T_var / total_var  (>0.5 → T dominates)")
    print("=" * 70)
    print(f"  {'T_struct':<10s} {'S':<5s} {'R_type':<12s} "
          f"{'within_var':>11s} {'between_var':>12s} {'ratio':>7s}  verdict")
    print("  " + "-" * 70)
    for (structure, S, R_type), v in sorted(results['q2'].items()):
        ratio = v['ratio']
        verdict = "T dominates" if (not np.isnan(ratio) and ratio > 0.5) else "R dominates"
        ratio_str = _fmt(ratio, 7, 3)
        print(f"  {structure:<10s} {S:<5d} {R_type:<12s} "
              f"{v['within_var']:>11.5f} {v['between_var']:>12.5f} {ratio_str}  {verdict}")

    # ------------------------------------------------------------------
    # Q3 — human-made MDPs  (note |A| confound for MCE entropy)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q3 — HUMAN-MADE MDPs")
    print("  Note: chains have A=2, grids have A=4. MCE entropy is H/log(A),")
    print("  so grids mechanically score higher — control for |A| before comparing.")
    print("=" * 70)
    score_cols = ['composite', 'adv_gap_norm', 'vstar_var_norm', 'h_eff_norm', 'mce_entropy_norm',
                  'ctrl_adv_norm', 'one_step_norm', 'plan_press_norm']
    hdrs3 = ['comp', 'adv_gap', 'v*var', 'h_eff', 'mce_ent', 'ctrl_adv', 'one_step', 'plan_press']
    print(f"  {'MDP':<22s} {'|A|':>4s}" + "".join(f"{h:>9s}" for h in hdrs3))
    print("  " + "-" * 95)
    rows = [(name, r) for name, r in results['q3'].items()]
    for name, r in sorted(rows, key=lambda x: -x[1]['composite']):
        a_str = f"{r['n_actions']:>4d}"
        vals = "".join(f"{r[c]:>9.4f}" for c in score_cols)
        print(f"  {name:<22s} {a_str}{vals}")

    print(f"\n  MI column: {'included' if results['meta']['include_mi'] else 'excluded (compute_mi=False)'}.")
